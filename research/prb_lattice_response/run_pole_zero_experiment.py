"""Actual finite RPA mode catalog and zero-field exterior-pole refinements."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from lattice_response import Parameters,Transitions,interaction_matrix
from bloch_response import bloch_spectrum,bloch_transitions
from pole_zero_analysis import (RationalResponse,transition_modes,charge_zero_intervals,
                               regular_pole_residue,exterior_real_poles)

ROOT=Path(__file__).resolve().parent
POLE_TOL=1e-5
RESIDUE_TOL=.01

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,complex): return dict(real=value.real,imag=value.imag)
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)
def write_json(path,value):
    path.write_text(json.dumps(value,default=serial,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def entries(prefix,matrix):
    return {f"{prefix}_{i}{j}_{part}":float(getattr(matrix[i,j],part))
            for i in range(2) for j in range(2) for part in ("real","imag")}

def main(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("A new empty output directory is required")
    output.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/name for name in ("lattice_response.py","bloch_response.py",
             "pole_zero_analysis.py","run_pole_zero_experiment.py")]
    hashes={p.name:sha(p) for p in sources}
    report=dict(status="RUNNING",started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,settings=dict(mu=.5,T=.04,g=.8,Ua=8.,q=2*np.pi/24,
        pole_relative_tolerance=POLE_TOL,residue_relative_tolerance=RESIDUE_TOL),
        submission_ready=False,low_energy_hybrid_gap_certified=False,
        continuum_landau_sheet_constructed=False)
    def log(message):
        print(message,flush=True)
        with (output/"execution.log").open("a",encoding="utf-8") as f:f.write(message+"\n")
    def checkpoint():write_json(output/"diagnostic.json",report)
    checkpoint()
    start=time.perf_counter()

    # A complete finite spectrum, without rounding transition energies or weights.
    p=Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    tr=bloch_transitions(bloch_spectrum(p),.5,.04,1)
    response=RationalResponse(tr);u=interaction_matrix(tr.Q,.8,8)
    log(f"Dense finite realization: {len(response.delta)} nonzero transitions")
    modes=transition_modes(tr,u,max_transitions=3500)
    assert modes["reconstruction_error"]<2e-9
    poles,residues=modes["poles"],modes["residues"]
    first=np.einsum("n,nab->ab",poles,residues)
    m0,m1=tr.moments()
    moment_error=float(np.max(np.abs(first-m1)))
    assert moment_error<2e-8
    records=[]
    for i,(pole,residue,weight) in enumerate(zip(poles,residues,modes["weighted_charge_weights"])):
        records.append(dict(index=i,pole=float(pole),**entries("residue",residue),
            weighted_charge=float(weight),nearest_bare_distance=float(np.min(abs(pole-response.delta))),
            exact_zero_charge_weight=bool(weight==0)))
    write_csv(output/"finite_poles.csv",records)
    zeros=charge_zero_intervals(modes)
    # Check resolved zeros against the original rational response, where regular.
    for row in zeros:
        if row["status"]=="resolved_charge_zero":
            z=row["zero"]
            row["nearest_bare_distance"]=float(np.min(np.abs(z-response.delta)))
            if row["nearest_bare_distance"]>1e-8:
                row["direct_charge_response_absolute"]=float(abs(response.dressed(z,u)[0,0]))
    write_csv(output/"finite_charge_zeros.csv",zeros)
    checks=[]
    chosen=np.flatnonzero((poles>0)&(poles<1.2))
    chosen=chosen[np.argsort(modes["weighted_charge_weights"][chosen])[-8:]]
    for index in chosen:
        row=dict(pole=float(poles[index]))
        try:
            result=regular_pole_residue(response,u,poles[index])
            row.update(status="regular_matrix_residue_checked",nullity=result["nullity"],
                residue_error=float(np.max(abs(result["residue_matrix"]-residues[index]))),
                eigenvalue_residual=result["eigenvalue_residual"])
        except ValueError as error:
            row.update(status="unresolved_regular_formula",reason=str(error))
        checks.append(row)
    report["finite_realization"]=dict(parameters=p.__dict__,
        retained_transitions=modes["retained_transitions"],
        exactly_zero_weights_removed=modes["exactly_zero_weights_removed"],
        reconstruction_error=modes["reconstruction_error"],first_moment_error=moment_error,
        moment0_error=modes["moment0_error"],regular_residue_checks=checks,
        zeros_resolved=sum(x["status"]=="resolved_charge_zero" for x in zeros),
        multiplication_zero_count=sum(x["status"]=="multiplication_zero" for x in zeros),
        zero_intervals_unresolved=sum(x["status"].startswith("unresolved") for x in zeros),
        scope="Complete finite modal realization; mode collectivity and continuum not inferred")
    finite_spectra=[]
    for eta in (.03,.015,.0075):
        for omega in np.linspace(.02,1.2,481):
            value=response.dressed(complex(omega,eta),u)
            finite_spectra.append(dict(omega=float(omega),eta=eta,loss_cc=float(-value[0,0].imag)))
    write_csv(output/"finite_regulator_scan.csv",finite_spectra)
    log("Finite modal reconstruction and first moment PASS")
    del modes

    # Analytically known degeneracy: D=N=D'=0 yet chi_cc retains a pole.
    energy,strength,coupling=.6,.04,4.
    delta=np.array([energy,-energy,energy,-energy])
    synthetic=Transitions(delta,np.array([strength,-strength,0,0]),
        np.array([0,0,strength,-strength]),np.zeros(4,complex),np.zeros(4,bool),
        Q=1.,q=1.,vertex_error=0.,density_value=0.)
    rs=RationalResponse(synthetic);us=np.eye(2)*coupling
    lam=float(np.sqrt(energy**2+2*energy*strength*coupling))
    matrix=regular_pole_residue(rs,us,lam,expected_nullity=2)
    expected=energy*strength/lam
    assert np.max(abs(matrix["residue_matrix"]-np.eye(2)*expected))<1e-12
    counter=[]
    for eta in np.geomspace(1e-8,.03,81):
        c=rs.cofactor(lam+1j*eta,us)
        counter.append(dict(eta=float(eta),determinant_abs=abs(c["determinant"]),
            numerator_abs=abs(c["numerator"]),charge_abs=abs(rs.dressed(lam+1j*eta,us)[0,0])))
    write_csv(output/"degenerate_counterexample.csv",counter)
    report["degenerate_counterexample"]=dict(pole=lam,charge_residue=expected,
        cofactor_at_pole=rs.cofactor(lam,us),nullity=matrix["nullity"],
        full_residue=matrix["residue_matrix"],classification="actual charge pole despite D=N=0")
    checkpoint()

    exterior=[]
    def evaluate(size,nk,qfactor=1.,axial=8.,stage="refinement"):
        spec=bloch_spectrum(Parameters(Lx=size,Ly=size,Nkz=nk))
        tr=bloch_transitions(spec,.5,.04,int(round(nk/24*qfactor)))
        r=RationalResponse(tr);u=interaction_matrix(tr.Q,.8,axial)
        scan=exterior_real_poles(r,u,samples=64)
        result=dict(L=size,Nkz=nk,q=tr.q,Ua=axial,stage=stage,density=tr.density_value,
            active_transitions=len(r.delta),raw_band_pair_slots=len(tr.delta),**scan)
        exterior.append(result)
        write_json(output/"exterior_searches.json",exterior)
        log(f"Exterior L={size},Nk={nk},qfactor={qfactor},Ua={axial}: "
            +str([(x["pole"],x.get("residue_matrix",np.zeros((2,2)))[0,0].real)
                  for x in scan["roots"]]))
        return result
    refined=[]
    comparisons=[]
    for size,nk in ((8,48),(16,96),(24,192),(48,192),(48,384)):
        current=evaluate(size,nk);refined.append(current)
        if len(refined)>1:
            old=refined[-2]
            rec=dict(coarse_L=old["L"],fine_L=size,coarse_Nkz=old["Nkz"],fine_Nkz=nk)
            a=[r for r in old["roots"] if r["residue_status"]=="isolated_nullspace_checked"]
            b=[r for r in current["roots"] if r["residue_status"]=="isolated_nullspace_checked"]
            if len(a)==len(b)==1:
                ra,rb=a[0]["residue_matrix"],b[0]["residue_matrix"]
                rec.update(pole_relative_error=abs(a[0]["pole"]-b[0]["pole"])/abs(b[0]["pole"]),
                    charge_residue_relative_error=float(abs(ra[0,0]-rb[0,0])/abs(rb[0,0])),
                    axial_residue_relative_error=float(abs(ra[1,1]-rb[1,1])/abs(rb[1,1])))
                rec["passes_declared_tolerances"]=bool(rec["pole_relative_error"]<=POLE_TOL
                    and max(rec["charge_residue_relative_error"],rec["axial_residue_relative_error"])<=RESIDUE_TOL)
            else:
                rec.update(passes_declared_tolerances=False,reason="Not one matched isolated exterior root")
            comparisons.append(rec);write_csv(output/"exterior_refinement.csv",comparisons)
    report["exterior_refinement"]=comparisons
    # Coupling/momentum scan at a stated finite grid; search remains exterior-only.
    for factor in (.5,1.,2.):
        for axial in (4.,8.,12.):
            evaluate(24,192,factor,axial,stage="q_U_scan")
    report["exterior_searches"]=exterior
    report["exterior_scope"]=("Roots lie above all active bare transitions and are finite undamped poles. "
        "No conclusion about unresolved edge slivers, internal modes, low-energy hybrid gap, or materials.")
    checkpoint()

    import matplotlib
    matplotlib.rcParams["pdf.fonttype"]=42
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
    for eta,color in zip((.03,.015,.0075),("#173f5f","#d14932","#26866d")):
        rows=[x for x in finite_spectra if x["eta"]==eta]
        axes[0].plot([x["omega"] for x in rows],[x["loss_cc"] for x in rows],
            color=color,lw=1.1,label=f"eta={eta:g}")
    axes[0].set(xlabel="Frequency",ylabel="-Im chi_cc",title="Finite lattice: eta is a regulator")
    axes[0].legend(fontsize=8)
    for key,label,style in (("determinant_abs","|D|","-"),("numerator_abs","|Nc|","--"),
                            ("charge_abs","|chi_cc|",":")):
        axes[1].loglog([x["eta"] for x in counter],[x[key] for x in counter],
            style,lw=1.3,label=label)
    axes[1].set(xlabel="eta at z=lambda+i eta",title="D=Nc=0 can retain a charge pole")
    axes[1].legend(fontsize=8)
    fig.savefig(output/"finite_modes.pdf");fig.savefig(output/"finite_modes.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.7),constrained_layout=True)
    for ua,color in zip((4.,8.,12.),("#173f5f","#d14932","#26866d")):
        cases=[x for x in exterior if x["stage"]=="q_U_scan" and x["Ua"]==ua and len(x["roots"])==1
               and x["roots"][0]["residue_status"]=="isolated_nullspace_checked"]
        if not cases:continue
        q=[x["q"] for x in cases]
        axes[0].plot(q,[x["roots"][0]["pole"] for x in cases],"o-",color=color,label=f"Ua={ua:g}")
        for ax,index in zip(axes[1:],(0,1)):
            ax.semilogy(q,[x["roots"][0]["residue_matrix"][index,index].real for x in cases],
                "o-",color=color,label=f"Ua={ua:g}")
    for ax,label in zip(axes,("Exterior pole frequency","Charge residue","Local axial residue")):
        ax.set(xlabel="q",ylabel=label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("B=0, L=24, Nk=192; all bands, positive exterior search only",fontsize=11)
    fig.savefig(output/"exterior_modes.pdf");fig.savefig(output/"exterior_modes.png",dpi=180);plt.close(fig)
    assert hashes=={p.name:sha(p) for p in sources},"Source changed during computation"
    report.update(status="FINITE_POLES_ZEROS_AND_EXTERIOR_SCAN_COMPLETED",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-start)
    log("Complete: finite pole/zero method and bounded exterior experiments; no low-energy gap certification.")
    report["output_sha256"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="diagnostic.json"}
    checkpoint()

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    destination=ap.parse_args().output.resolve()
    try:
        main(destination)
    except Exception as error:
        diagnostic=destination/"diagnostic.json"
        if diagnostic.exists():
            record=json.loads(diagnostic.read_text(encoding="utf-8"))
            record.update(status="FAILED",failure_type=type(error).__name__,failure=str(error))
            write_json(diagnostic,record)
        raise

