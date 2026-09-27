"""Budgeted fixed-B magnetic-grid refinement using all 144 bands.

B=2*pi/36, mu=.5, T=.04, qprobe=2*pi/24, eta=.015, Coulomb=.8,
axial interaction=8, and the original 161 frequencies are held fixed.
All bare magnetic-block responses are summed before applying RPA.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from lattice_response import Parameters,Transitions,interaction_matrix,loss_matrix,rpa_response,parameter_dict
from magnetic_bloch_response import (
    magnetic_bloch_spectrum,magnetic_bloch_transitions,iter_magnetic_bloch_spectra,
)
from extend_transverse import memory_status,sha,csv_write,scaled_error,complex_pair

ROOT=Path(__file__).resolve().parent
SETTINGS=dict(mu=.5,temperature=.04,eta=.015,coulomb=.8,axial_interaction=8.,
              numerator=1,denominator=36,qprobe=2*np.pi/24,tolerance=1e-3)
PROBE=.45+.015j
FREQUENCY_BATCH=4


class BudgetExceeded(RuntimeError):
    pass


def fast_density_responses(tr,omegas,eta):
    """Algebraically identical resolvent, batched with real BLAS arrays.

    No transition or weight is removed. For w=wr+i*wi and d=a+i*b,
    w*d=(wr*a-wi*b)+i*(wr*b+wi*a); the ac channel uses conjugated w.
    """
    weights=np.stack((tr.cc,tr.aa,tr.ca.real,tr.ca.imag))
    response=np.empty((len(omegas),2,2),complex)
    for first in range(0,len(omegas),FREQUENCY_BATCH):
        last=min(first+FREQUENCY_BATCH,len(omegas))
        real=omegas[None,first:last]-tr.delta[:,None]
        imaginary=np.square(real)
        imaginary+=eta**2
        np.divide(real,imaginary,out=real)
        np.reciprocal(imaginary,out=imaginary)
        imaginary*=-eta
        re=weights@real
        im=weights@imaginary
        response[first:last,0,0]=re[0]+1j*im[0]
        response[first:last,1,1]=re[1]+1j*im[1]
        response[first:last,0,1]=(re[2]-im[3])+1j*(im[2]+re[3])
        response[first:last,1,0]=(re[2]+im[3])+1j*(im[2]-re[3])
    return response


def validate_evaluator(omegas):
    p=Parameters(Lx=3,Ly=4,Nkz=24,theta_x=.29,theta_y=-.43,theta_z=.19)
    spec=magnetic_bloch_spectrum(p,1,3)
    records=[]
    for shift in (-1,1):
        tr=magnetic_bloch_transitions(spec,.5,.04,shift)
        expected=np.array([tr.density_response(complex(w,.015)) for w in omegas])
        actual=fast_density_responses(tr,omegas,.015)
        error=float(np.max(np.abs(actual-expected)))
        budget=128*np.finfo(float).eps*p.volume*max(1.,float(np.max(np.abs(expected))))
        records.append(dict(case=f"magnetic_twisted_qshift{shift:+d}",
                            maximum_absolute_error=error,roundoff_budget=budget,passed=bool(error<=budget)))
    fixture=Transitions(delta=np.array([-1.2,.13,.91]),cc=np.array([-.4,.2,.6]),
        aa=np.array([-.3,.5,.7]),ca=np.array([-.13+.07j,.03-.11j,.17+.19j]),
        conduction=np.array([False,True,True]),Q=.4,q=.4027,vertex_error=0.,density_value=0.)
    expected=np.array([fixture.density_response(complex(w,.015)) for w in omegas])
    error=float(np.max(np.abs(fast_density_responses(fixture,omegas,.015)-expected)))
    budget=128*np.finfo(float).eps*max(1.,float(np.max(np.abs(expected))))
    records.append(dict(case="nonreal_cross_weights_algebra_fixture",maximum_absolute_error=error,
                        roundoff_budget=budget,passed=bool(error<=budget)))
    return dict(records=records,all_checks_pass=all(row["passed"] for row in records),
                transition_cutoff=None,frequency_batch=FREQUENCY_BATCH,
                method="All-transition real/imaginary resolvent sum versus core complex resolvent")


def main(output,budget_seconds):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Magnetic refinement output must be empty")
    sources=[ROOT/name for name in ("lattice_response.py","bloch_response.py",
        "magnetic_bloch_response.py","extend_transverse.py","run_magnetic_refinement.py")]
    inputs=[ROOT/"magnetic_bloch_validation"/"diagnostic.json",
            ROOT/"refinement"/"L48_Nk192_flux+0_fixed_mu.csv"]
    source_hashes={p.name:sha(p) for p in sources}
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    validation=json.loads(inputs[0].read_text(encoding="utf-8"))
    if validation["status"]!="PASS":
        raise RuntimeError("Magnetic provider validation is not PASS")
    if validation["source_sha256"]["magnetic_bloch_response.py"]!=source_hashes["magnetic_bloch_response.py"]:
        raise RuntimeError("Magnetic provider changed since its validation")
    with inputs[1].open(newline="",encoding="utf-8") as stream:
        omegas=np.array([float(row["omega"]) for row in csv.DictReader(stream)])
    if len(omegas)!=161 or not np.allclose(omegas,np.linspace(.025,1.2,161),rtol=0,atol=1e-15):
        raise ValueError("Original frequency grid changed")
    output.mkdir(parents=True,exist_ok=True)
    logpath=output/"execution.log"
    def log(message):
        print(message,flush=True)
        with logpath.open("a",encoding="utf-8") as stream:
            stream.write(message+"\n")
    start=time.perf_counter()
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),status="RUNNING",
        settings=SETTINGS,frequency_points=161,source_sha256=source_hashes,input_sha256=input_hashes,
        budget_seconds=budget_seconds,memory_before=memory_status(),calculations=[],comparisons=[],
        budget_decisions=[],all_selected_directions_converged=False,collective_pole_or_gap_certified=False,
        limitations=["Only B=+2*pi/36 and fixed mu=.5 are explored here.",
          "Finite-eta sampled-response convergence is not a pole or continuum-limit certificate.",
          "Non-divisor Nky grids avoid trivially equivalent magnetic ky samples.",
          "Partial block sums are never treated as completed spectra."])
    def checkpoint():
        (output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    checkpoint()
    def evaluate(nx,ny,nk):
        case_start=time.perf_counter()
        p=Parameters(Lx=36*nx,Ly=ny,Nkz=nk)
        blocks=nx*ny
        pairs_per_block=nk*144**2
        estimated_peak=256*pairs_per_block+128*1024**2
        minimum_available=memory_status()["available_physical_bytes"]
        if 2*estimated_peak+512*1024**2>minimum_available:
            raise MemoryError("Insufficient RAM for one all-band magnetic block")
        log(f"[magnetic] Nkx={nx}, Nky={ny}, Nkz={nk}; Lx={p.Lx}, Ly={p.Ly}, flux={blocks}")
        bare=np.zeros((len(omegas),2,2),complex)
        probe_full=np.zeros((3,3),complex)
        moments=np.zeros((2,2,2),complex)
        density_value,vertex_error,count=0.,0.,0
        for i,spec in enumerate(iter_magnetic_bloch_spectra(p,1,36,blocks_per_chunk=1)):
            free=memory_status()["available_physical_bytes"]
            minimum_available=min(minimum_available,free)
            if 2*estimated_peak+512*1024**2>free:
                raise MemoryError("Available RAM fell below the all-band block budget")
            tr=magnetic_bloch_transitions(spec,.5,.04,nk//24)
            bare+=fast_density_responses(tr,omegas,.015)
            probe_full+=tr.full_response(PROBE)
            moments+=np.array(tr.moments())
            density_value+=tr.density_value
            vertex_error=max(vertex_error,tr.vertex_error)
            count+=tr.delta.size
            report["active_case"]=dict(Nkx=nx,Nky=ny,Nkz=nk,completed_blocks=i+1,total_blocks=blocks,
                retained_transition_count=count,elapsed_seconds=time.perf_counter()-case_start)
            del tr,spec
            if i==0 or (i+1)%4==0 or i+1==blocks:
                log(f"  blocks {i+1}/{blocks}, elapsed {time.perf_counter()-case_start:.1f}s")
                checkpoint()
            # A nearly finished admitted calculation may finish past the soft budget.
            if time.perf_counter()-start>budget_seconds+60 and i+1<.9*blocks:
                np.savez(output/f"partial_Nkx{nx}_Nky{ny}_Nk{nk}.npz",bare=bare,
                         probe_full=probe_full,moments=moments,completed_blocks=i+1,total_blocks=blocks)
                raise BudgetExceeded("Stopped an incomplete grid after exceeding the soft budget plus 60s")
        Q=2*np.sin(SETTINGS["qprobe"]/2)
        coupling=interaction_matrix(Q,.8,8.)
        response=np.array([rpa_response(value,coupling) for value in bare])
        probe_rpa=rpa_response(probe_full,coupling)
        contraction=np.array([PROBE,0.,-Q])
        ward=lambda value:float(max(np.max(np.abs(contraction@value)),np.max(np.abs(value@contraction))))
        power_min=min(float(np.linalg.eigvalsh(loss_matrix(complex(w,.015)*value)).min())
                      for w,value in zip(omegas,response))
        record=dict(Nkx=nx,Nky=ny,Nkz=nk,Lx=p.Lx,Ly=p.Ly,flux=blocks,B=2*np.pi/36,
            parameters=parameter_dict(p),density=density_value,retained_transition_count=count,
            expected_transition_count=blocks*pairs_per_block,bands_per_block=144,
            vertex_error=vertex_error,vertex_tolerance=2e-11,
            charge_ward_bare=ward(probe_full),charge_ward_rpa=ward(probe_rpa),ward_tolerance=2e-10,
            equal_time_density_commutator_max=float(np.max(np.abs(moments[0]))),
            minimum_rpa_weighted_loss_eigenvalue=power_min,weighted_passivity_tolerance=2e-10,
            weighted_passivity_frequency_points=161,probe_frequency=complex_pair(PROBE),
            probe_bare_ca=complex_pair(probe_full[0,1]),probe_rpa_ca=complex_pair(probe_rpa[0,1]),
            estimated_peak_bytes=estimated_peak,minimum_observed_available_bytes=minimum_available,
            elapsed_seconds=time.perf_counter()-case_start)
        record["all_exact_identity_checks_pass"]=bool(count==blocks*pairs_per_block
            and vertex_error<2e-11 and max(record["charge_ward_bare"],record["charge_ward_rpa"],
            record["equal_time_density_commutator_max"])<2e-10 and power_min>-2e-10)
        rows=[]
        for w,b,r in zip(omegas,bare,response):
            row=dict(Nkx=nx,Nky=ny,Nkz=nk,Lx=p.Lx,Ly=p.Ly,flux=blocks,B=2*np.pi/36,
                     ensemble="fixed_mu",mu=.5,density=density_value,q=SETTINGS["qprobe"],omega=float(w),eta=.015)
            for prefix,matrix in (("rpa",r),("bare",b)):
                for a,c in ((0,0),(0,1),(1,0),(1,1)):
                    row[f"{prefix}_{a}{c}_re"]=float(matrix[a,c].real)
                    row[f"{prefix}_{a}{c}_im"]=float(matrix[a,c].imag)
            rows.append(row)
        filename=f"Nkx{nx}_Nky{ny}_Nk{nk}_B1of36_fixed_mu.csv"
        csv_write(output/filename,rows)
        record["csv"]=filename
        report["calculations"].append(record)
        report.pop("active_case",None)
        checkpoint()
        if not record["all_exact_identity_checks_pass"]:
            raise RuntimeError("Completed grid failed an identity or weighted-passivity check")
        return response,record
    def affordable(nx,ny,nk,kind):
        elapsed=time.perf_counter()-start
        done=report["calculations"]
        rate=sum(r["elapsed_seconds"] for r in done)/sum(r["Nkx"]*r["Nky"]*r["Nkz"] for r in done)
        predicted=1.25*rate*nx*ny*nk
        decision=dict(kind=kind,Nkx=nx,Nky=ny,Nkz=nk,elapsed_seconds=elapsed,
                      predicted_seconds_with_25_percent_margin=predicted,
                      admitted=elapsed+predicted<=budget_seconds)
        report["budget_decisions"].append(decision)
        log("Budget admission: "+json.dumps(decision))
        checkpoint()
        return decision["admitted"]
    def compare(coarse,fine,c0,c1,kind):
        error=scaled_error(coarse,fine)
        comparison=dict(kind=kind,coarse_Nkx=c0["Nkx"],coarse_Nky=c0["Nky"],coarse_Nkz=c0["Nkz"],
            fine_Nkx=c1["Nkx"],fine_Nky=c1["Nky"],fine_Nkz=c1["Nkz"],B=2*np.pi/36,
            maximum_scaled_response_error=error,tolerance=1e-3,converged=error<=1e-3)
        report["comparisons"].append(comparison)
        csv_write(output/"comparisons.csv",report["comparisons"])
        log("Comparison: "+json.dumps(comparison))
        checkpoint()
        return comparison
    try:
        log("Preflight: "+json.dumps(report["memory_before"]))
        validation=validate_evaluator(omegas)
        (output/"evaluator_validation.json").write_text(json.dumps(validation,indent=2)+"\n",encoding="utf-8")
        report["evaluator_validation_pass"]=validation["all_checks_pass"]
        if not validation["all_checks_pass"]:
            raise RuntimeError("Fast resolvent differs from the complete core response")
        previous,previous_case=evaluate(1,7,96)
        transverse=None
        for nx,ny in ((2,13),(4,25)):
            if not affordable(nx,ny,96,"magnetic_transverse"):
                raise BudgetExceeded("Next transverse grid exceeds the measured remaining budget")
            current,current_case=evaluate(nx,ny,96)
            transverse=compare(previous,current,previous_case,current_case,"magnetic_transverse")
            previous,previous_case=current,current_case
            if transverse["converged"]:
                break
        report["final_transverse_comparison"]=transverse
        nx,ny=previous_case["Nkx"],previous_case["Nky"]
        if not affordable(nx,ny,192,"fixed_B_kz"):
            raise BudgetExceeded("Final Nkz192 check exceeds the measured remaining budget")
        current,current_case=evaluate(nx,ny,192)
        longitudinal=compare(previous,current,previous_case,current_case,"fixed_B_kz")
        report["final_kz_comparison"]=longitudinal
        report["all_selected_directions_converged"]=bool(transverse["converged"] and longitudinal["converged"])
        report["status"]=("SELECTED_MAGNETIC_MESHES_CONVERGED" if report["all_selected_directions_converged"]
                          else "MAGNETIC_REFINEMENT_COMPLETED_UNCONVERGED")
    except BudgetExceeded as exc:
        report["status"]="BUDGET_LIMIT_REACHED"
        report["incomplete_reason"]=str(exc)
        log(str(exc))
    except Exception as exc:
        report["status"]="MAGNETIC_REFINEMENT_FAILED"
        report["failure"]={"type":type(exc).__name__,"message":str(exc)}
        log(traceback.format_exc())
    report["sources_unchanged"]=source_hashes=={p.name:sha(p) for p in sources}
    report["inputs_unchanged"]=input_hashes=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    if not report["sources_unchanged"] or not report["inputs_unchanged"]:
        report["status"]="SOURCE_OR_INPUT_CHANGED"
    report["all_completed_identity_checks_pass"]=bool(report.get("evaluator_validation_pass",False)
        and report["calculations"] and all(r["all_exact_identity_checks_pass"] for r in report["calculations"]))
    report["elapsed_seconds"]=time.perf_counter()-start
    report["completed_utc"]=datetime.now(timezone.utc).isoformat()
    log("Final status: "+report["status"])
    report["output_sha256"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="diagnostic.json"}
    checkpoint()
    if report["status"] in ("MAGNETIC_REFINEMENT_FAILED","SOURCE_OR_INPUT_CHANGED"):
        raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--budget-seconds",type=float,default=600.)
    args=parser.parse_args()
    if not 0<args.budget_seconds<=600:
        parser.error("Use a positive computational budget no greater than 600 seconds")
    main(args.output.resolve(),args.budget_seconds)