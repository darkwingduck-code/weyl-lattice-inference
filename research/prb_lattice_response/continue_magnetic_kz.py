"""Complete the final fixed-B kz check while preserving the 600s-run snapshot.

The follow-up computational budget is 1200s INCLUDING the prior completed run.
All physics, frequency samples, denominator36 bands, and tolerance stay fixed.
"""
from __future__ import annotations

import argparse,csv,json,time,traceback
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from lattice_response import Parameters,interaction_matrix,loss_matrix,rpa_response,parameter_dict
from magnetic_bloch_response import iter_magnetic_bloch_spectra,magnetic_bloch_transitions
from run_magnetic_refinement import SETTINGS,PROBE,fast_density_responses,BudgetExceeded
from extend_transverse import memory_status,sha,csv_write,scaled_error,complex_pair

ROOT=Path(__file__).resolve().parent


def main(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Continuation evidence must be empty")
    prior_dir=ROOT/"magnetic_refinement"
    prior_path=prior_dir/"diagnostic.json"
    prior=json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["status"]!="BUDGET_LIMIT_REACHED" or prior["settings"]!=SETTINGS:
        raise RuntimeError("Expected the unchanged budget-limited prior calculation")
    transverse=prior["final_transverse_comparison"]
    if not transverse["converged"] or not prior["all_completed_identity_checks_pass"]:
        raise RuntimeError("Prior transverse response or exact identities have not passed")
    for name,value in prior["source_sha256"].items():
        if sha(ROOT/name)!=value:
            raise RuntimeError("Prior source changed: "+name)
    for name,value in prior["output_sha256"].items():
        if sha(prior_dir/name)!=value:
            raise RuntimeError("Prior output changed: "+name)
    coarse_case=prior["calculations"][-1]
    if (coarse_case["Nkx"],coarse_case["Nky"],coarse_case["Nkz"])!=(2,13,96):
        raise RuntimeError("Expected final 2x13 magnetic momentum grid at Nkz96")
    coarse_path=prior_dir/coarse_case["csv"]
    with coarse_path.open(newline="",encoding="utf-8") as stream:
        coarse_rows=list(csv.DictReader(stream))
    omegas=np.array([float(row["omega"]) for row in coarse_rows])
    if len(omegas)!=161 or not np.allclose(omegas,np.linspace(.025,1.2,161),rtol=0,atol=1e-15):
        raise RuntimeError("Prior frequency samples changed")
    coarse=np.array([[[complex(float(row[f"rpa_{a}{b}_re"]),float(row[f"rpa_{a}{b}_im"]))
                       for b in range(2)] for a in range(2)] for row in coarse_rows])
    p=replace(Parameters(**coarse_case["parameters"]),Nkz=192)
    sources=[ROOT/name for name in ("lattice_response.py","bloch_response.py",
        "magnetic_bloch_response.py","extend_transverse.py","run_magnetic_refinement.py","continue_magnetic_kz.py")]
    source_hashes={path.name:sha(path) for path in sources}
    input_hashes={str(path.relative_to(ROOT)):sha(path) for path in (prior_path,coarse_path)}
    prior_elapsed=prior["elapsed_seconds"]
    predicted=coarse_case["elapsed_seconds"]*2*1.25
    if prior_elapsed+predicted>1200:
        raise BudgetExceeded("Measured prediction exceeds authorized total1200s budget")
    output.mkdir(parents=True,exist_ok=True)
    logpath=output/"execution.log"
    def log(message):
        print(message,flush=True)
        with logpath.open("a",encoding="utf-8") as stream:
            stream.write(message+"\n")
    start=time.perf_counter()
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),status="RUNNING",
        settings=SETTINGS,parameters=parameter_dict(p),Nkx=2,Nky=13,Nkz=192,
        flux=26,B=2*np.pi/36,frequency_points=161,source_sha256=source_hashes,input_sha256=input_hashes,
        prior_status=prior["status"],prior_evidence_verified=True,prior_elapsed_seconds=prior_elapsed,
        total_computational_budget_seconds=1200,predicted_seconds_with_25_percent_margin=predicted,
        final_transverse_comparison=transverse,collective_pole_or_gap_certified=False,
        all_selected_directions_converged=False,
        limitations=["The earlier600s budget-limited result is preserved without modification.",
          "This confirms sampled responses only at B=+2*pi/36, fixed mu, and eta=.015.",
          "Finite-eta grid convergence does not establish a collective pole, gap, or infinite-system theorem."])
    def checkpoint():
        (output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    checkpoint()
    try:
        available=memory_status()["available_physical_bytes"]
        expected_per_block=192*144**2
        estimate=256*expected_per_block+128*1024**2
        report["memory_before"]=memory_status()
        report["estimated_peak_bytes"]=estimate
        if 2*estimate+512*1024**2>available:
            raise MemoryError("Insufficient available RAM for one full144-band block")
        log(f"[continuation] Nkx=2,Nky=13,Nkz=192; prior runtime={prior_elapsed:.1f}s, total budget=1200s")
        bare=np.zeros((161,2,2),complex)
        probe_full=np.zeros((3,3),complex)
        moments=np.zeros((2,2,2),complex)
        density_value,vertex_error,count=0.,0.,0
        minimum_available=available
        for i,spec in enumerate(iter_magnetic_bloch_spectra(p,1,36,blocks_per_chunk=1)):
            available=memory_status()["available_physical_bytes"]
            minimum_available=min(minimum_available,available)
            if 2*estimate+512*1024**2>available:
                raise MemoryError("Available RAM fell below the full-band block budget")
            tr=magnetic_bloch_transitions(spec,.5,.04,8)
            bare+=fast_density_responses(tr,omegas,.015)
            probe_full+=tr.full_response(PROBE)
            moments+=np.array(tr.moments())
            density_value+=tr.density_value
            vertex_error=max(vertex_error,tr.vertex_error)
            count+=tr.delta.size
            del tr,spec
            report["completed_blocks"]=i+1
            report["retained_transition_count"]=count
            if i==0 or (i+1)%4==0 or i+1==26:
                log(f"  blocks {i+1}/26, continuation elapsed={time.perf_counter()-start:.1f}s")
                checkpoint()
            if prior_elapsed+time.perf_counter()-start>1260 and i+1<24:
                np.savez(output/"incomplete_block_sum.npz",bare=bare,probe_full=probe_full,moments=moments,
                         completed_blocks=i+1,total_blocks=26)
                raise BudgetExceeded("Continuation exceeded total budget plus60s before nearing completion")
        Q=2*np.sin(SETTINGS["qprobe"]/2)
        interaction=interaction_matrix(Q,.8,8.)
        response=np.array([rpa_response(value,interaction) for value in bare])
        probe_rpa=rpa_response(probe_full,interaction)
        contraction=np.array([PROBE,0.,-Q])
        ward=lambda value:float(max(np.max(np.abs(contraction@value)),np.max(np.abs(value@contraction))))
        power_min=min(float(np.linalg.eigvalsh(loss_matrix(complex(w,.015)*value)).min())
                      for w,value in zip(omegas,response))
        checks=dict(density=density_value,retained_transition_count=count,
            expected_transition_count=26*expected_per_block,bands_per_block=144,
            vertex_error=vertex_error,vertex_tolerance=2e-11,
            charge_ward_bare=ward(probe_full),charge_ward_rpa=ward(probe_rpa),ward_tolerance=2e-10,
            equal_time_density_commutator_max=float(np.max(np.abs(moments[0]))),
            minimum_rpa_weighted_loss_eigenvalue=power_min,weighted_passivity_tolerance=2e-10,
            weighted_passivity_frequency_points=161,probe_frequency=complex_pair(PROBE),
            probe_bare_ca=complex_pair(probe_full[0,1]),probe_rpa_ca=complex_pair(probe_rpa[0,1]),
            minimum_observed_available_bytes=minimum_available)
        checks["all_exact_identity_checks_pass"]=bool(count==26*expected_per_block and vertex_error<2e-11
            and max(checks["charge_ward_bare"],checks["charge_ward_rpa"],checks["equal_time_density_commutator_max"])<2e-10
            and power_min>-2e-10)
        report["checks"]=checks
        rows=[]
        for w,b,r in zip(omegas,bare,response):
            row=dict(Nkx=2,Nky=13,Nkz=192,Lx=p.Lx,Ly=p.Ly,flux=26,B=2*np.pi/36,
                     ensemble="fixed_mu",mu=.5,density=density_value,q=SETTINGS["qprobe"],omega=float(w),eta=.015)
            for prefix,matrix in (("rpa",r),("bare",b)):
                for a,c in ((0,0),(0,1),(1,0),(1,1)):
                    row[f"{prefix}_{a}{c}_re"]=float(matrix[a,c].real)
                    row[f"{prefix}_{a}{c}_im"]=float(matrix[a,c].imag)
            rows.append(row)
        filename="Nkx2_Nky13_Nk192_B1of36_fixed_mu.csv"
        csv_write(output/filename,rows)
        report["csv"]=filename
        error=scaled_error(coarse,response)
        comparison=dict(kind="fixed_B_kz",coarse_Nkx=2,coarse_Nky=13,coarse_Nkz=96,
            fine_Nkx=2,fine_Nky=13,fine_Nkz=192,B=2*np.pi/36,
            maximum_scaled_response_error=error,tolerance=1e-3,converged=error<=1e-3)
        report["final_kz_comparison"]=comparison
        csv_write(output/"comparisons.csv",[transverse,comparison])
        log("Final kz comparison: "+json.dumps(comparison))
        if not checks["all_exact_identity_checks_pass"]:
            raise RuntimeError("Continuation failed an exact-identity or weighted-passivity check")
        report["all_selected_directions_converged"]=bool(transverse["converged"] and comparison["converged"])
        report["status"]=("SELECTED_MAGNETIC_MESHES_CONVERGED" if report["all_selected_directions_converged"]
                          else "MAGNETIC_CONTINUATION_COMPLETED_UNCONVERGED")
    except BudgetExceeded as exc:
        report["status"]="CONTINUATION_BUDGET_LIMIT_REACHED"
        report["incomplete_reason"]=str(exc)
        log(str(exc))
    except Exception as exc:
        report["status"]="MAGNETIC_CONTINUATION_FAILED"
        report["failure"]={"type":type(exc).__name__,"message":str(exc)}
        log(traceback.format_exc())
    report["sources_unchanged"]=source_hashes=={path.name:sha(path) for path in sources}
    report["inputs_unchanged"]=input_hashes=={str(path.relative_to(ROOT)):sha(path) for path in (prior_path,coarse_path)}
    if not report["sources_unchanged"] or not report["inputs_unchanged"]:
        report["status"]="SOURCE_OR_INPUT_CHANGED"
    report["elapsed_seconds"]=time.perf_counter()-start
    report["total_computational_seconds"]=prior_elapsed+report["elapsed_seconds"]
    report["completed_utc"]=datetime.now(timezone.utc).isoformat()
    log("Final status: "+report["status"])
    report["output_sha256"]={path.name:sha(path) for path in output.iterdir()
                             if path.is_file() and path.name!="diagnostic.json"}
    checkpoint()
    if report["status"] in ("MAGNETIC_CONTINUATION_FAILED","SOURCE_OR_INPUT_CHANGED"):
        raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    main(args.output.resolve())