"""Calibrated detector sensitivity on fixed, full-band RPA input data.

This is a deterministic distinguishability calculation, not an experiment,
confidence interval, or proof of global inverse uniqueness. Both candidate
models share the Hamiltonian and vary positive diagonal interactions only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from low_energy_mode_audit import load_response

ROOT=Path(__file__).resolve().parent
MODES=("charge_complex","charge_loss","charge_and_axial_complex")
DATASETS=("B0","Bplus","joint")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prediction(pi,q,g,ua):
    u=np.diag([g/(2*np.sin(q/2))**2,ua])
    return np.linalg.solve(np.eye(2)-pi@u,pi)

def derivatives(chi,q,g,ua):
    ec=np.diag([1.,0.]);ea=np.diag([0.,1.])
    return g/(2*np.sin(q/2))**2*(chi@ec@chi),ua*(chi@ea@chi)

def residual_vector(model,truth,mode):
    channels=(0,1) if mode=="charge_and_axial_complex" else (0,)
    result=[]
    for c in channels:
        a,b=model[:,c,c],truth[:,c,c]
        if mode=="charge_loss":a,b=-a.imag,-b.imag
        scale=np.linalg.norm(b)
        if scale==0:raise ValueError("Calibrated channel has zero reference norm")
        d=(a-b)/scale
        result.extend([d.real,d.imag] if np.iscomplexobj(d) else [d])
    return np.concatenate(result)/np.sqrt(len(channels))

def selected_vectors(models,truths,mode,dataset):
    names=("B0","Bplus") if dataset=="joint" else (dataset,)
    return np.concatenate([residual_vector(models[n],truths[n],mode) for n in names])/np.sqrt(len(names))

def numerical_profile(cost,grid):
    values=np.array([cost(x) for x in grid])
    ids=[i for i in range(1,len(grid)-1) if values[i]<=values[i-1] and values[i]<=values[i+1]]
    candidates=[dict(g=float(grid[0]),cost=float(values[0]),kind="grid_endpoint",success=True),
                dict(g=float(grid[-1]),cost=float(values[-1]),kind="grid_endpoint",success=True)]
    for i in ids:
        result=minimize_scalar(cost,bounds=(grid[i-1],grid[i+1]),method="bounded",options={"xatol":1e-12})
        candidates.append(dict(g=float(result.x),cost=float(result.fun),kind="bounded_local_refinement",success=bool(result.success)))
    # Always retain the sampled grid minimum even if refinement fails.
    j=int(values.argmin())
    candidates.append(dict(g=float(grid[j]),cost=float(values[j]),kind="sampled_grid_minimum",success=True))
    accepted=[x for x in candidates if x["success"] and np.isfinite(x["cost"])]
    best=min(accepted,key=lambda x:x["cost"])
    return dict(**best,refinement_candidates=len(ids),all_refinements_successful=all(x["success"] for x in candidates),
                global_minimum_certified=False)

def csv_write(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(output):
    if output.exists() and any(output.iterdir()):raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    inputs={"B0":ROOT/"transverse_extension/L72_Nk384_flux+0_fixed_mu.csv",
            "Bplus":ROOT/"magnetic_kz_continuation/Nkx2_Nky13_Nk192_B1of36_fixed_mu.csv"}
    source_paths=[ROOT/x for x in ("detector_identifiability_audit.py","test_detector_identifiability_audit.py",
                  "low_energy_mode_audit.py","lattice_response.py")]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [*inputs.values(),*source_paths]}
    bare,truths,meta,frequencies={},{},{},{}
    for name,path in inputs.items():
        w,m,info=load_response(path);mask=(w>=.05)&(w<=.35)
        bare[name]=m["bare"][mask];truths[name]=prediction(bare[name],info["q"],.8,8.)
        frequencies[name]=w[mask];meta[name]=dict(**info,samples=int(mask.sum()),frequency_min=float(w[mask].min()),frequency_max=float(w[mask].max()))
        assert np.max(abs(truths[name]-m["rpa"][mask]))<2e-13
    def models(g,ua):return {n:prediction(p,meta[n]["q"],g,ua) for n,p in bare.items()}
    pairs=[]
    for ua in (0.,4.,8.,12.,16.,24.):
        model=models(.8,ua)
        for name in ("B0","Bplus"):
            for mode in MODES:
                error=np.linalg.norm(residual_vector(model[name],truths[name],mode))
                cc=np.linalg.norm((model[name]-truths[name])[:,0,0])/np.linalg.norm(truths[name][:,0,0])
                aa=np.linalg.norm((model[name]-truths[name])[:,1,1])/np.linalg.norm(truths[name][:,1,1])
                pairs.append(dict(dataset=name,mode=mode,g=.8,ua=ua,normalized_mismatch=float(error),
                                  complex_charge_relative_l2=float(cc),complex_axial_relative_l2=float(aa)))
    csv_write(output/"same_coulomb_hypotheses.csv",pairs)
    profiles=[];uas=np.linspace(0,24,49);ggrid=np.linspace(.2,1.6,65)
    for dataset in DATASETS:
        for mode in MODES:
            for ua in uas:
                def cost(g):
                    v=selected_vectors(models(g,ua),truths,mode,dataset)
                    return float(v@v)
                fit=numerical_profile(cost,ggrid)
                profiles.append(dict(dataset=dataset,mode=mode,ua=float(ua),best_g=fit["g"],
                    normalized_mismatch=float(np.sqrt(fit["cost"])),cost=fit["cost"],
                    all_refinements_successful=fit["all_refinements_successful"],
                    candidates=fit["refinement_candidates"],global_minimum_certified=False))
            print(f"Profile completed: {dataset} / {mode}",flush=True)
    csv_write(output/"profiled_coulomb.csv",profiles)
    svd=[]
    for dataset in DATASETS:
        for mode in MODES:
            columns=[]
            for parameter in (0,1):
                deriv={n:derivatives(truths[n],meta[n]["q"],.8,8.)[parameter] for n in bare}
                # residual_vector(derivative+truth, truth) applies the same scale.
                column=selected_vectors({n:truths[n]+deriv[n] for n in bare},truths,mode,dataset)
                columns.append(column)
            jac=np.column_stack(columns);s=np.linalg.svd(jac,compute_uv=False)
            svd.append(dict(dataset=dataset,mode=mode,parameterization="ln(g),ln(Ua)",
                largest_singular=float(s[0]),smallest_singular=float(s[1]),
                condition_number=float(s[0]/s[1]) if s[1]>0 else None,
                norm_charge_interaction_column=float(np.linalg.norm(jac[:,0])),
                norm_axial_interaction_column=float(np.linalg.norm(jac[:,1])),
                interpretation="Local calibrated sensitivity, not global identifiability or statistical information bound"))
    csv_write(output/"local_sensitivity.csv",svd)
    admitted=[]
    for dataset in DATASETS:
        for mode in MODES:
            rr=[r for r in profiles if r["dataset"]==dataset and r["mode"]==mode]
            for threshold in (.001,.01):
                selected=[r for r in rr if r["normalized_mismatch"]<=threshold]
                admitted.append(dict(dataset=dataset,mode=mode,declared_relative_mismatch_threshold=threshold,
                    accepted_sampled_ua=[r["ua"] for r in selected],
                    interpretation="Sampled deterministic tolerance set; not a confidence interval, not a certified continuous interval"))
    dcheck=[]
    for name,p in bare.items():
        q=meta[name]["q"];chi=truths[name];a,b=derivatives(chi,q,.8,8.);step=1e-5
        for idx,(g,ua,expected) in enumerate(((.8,8.,a),(.8,8.,b))):
            plus=prediction(p,q,g*np.exp(step) if idx==0 else g,ua*np.exp(step) if idx==1 else ua)
            minus=prediction(p,q,g*np.exp(-step) if idx==0 else g,ua*np.exp(-step) if idx==1 else ua)
            difference=(plus-minus)/(2*step)
            dcheck.append(dict(dataset=name,parameter=idx,relative_error=float(np.linalg.norm(difference-expected)/np.linalg.norm(expected))))
    assert max(x["relative_error"] for x in dcheck)<1e-7
    assert all(r["normalized_mismatch"]<1e-8 for r in profiles if r["ua"]==8.)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    for dataset,color,mark,ls in [("B0","#0072B2","o","-"),("Bplus","#D55E00","s","--"),("joint","#009E73","^",":")]:
        for ax,mode in zip(axes,("charge_complex","charge_and_axial_complex")):
            rr=[r for r in profiles if r["dataset"]==dataset and r["mode"]==mode]
            ax.semilogy([r["ua"] for r in rr],[max(r["normalized_mismatch"],1e-10) for r in rr],
                color=color,ls=ls,marker=mark,markevery=6,mfc="none",ms=6,lw=1.1,label=dataset)
    for ax,title in zip(axes,("Calibrated charge response","Charge plus local axial response")):
        ax.axhline(.01,color=".5",ls="--",lw=.7,label="1% mismatch scenario")
        ax.axvline(8,color=".7",ls=":",lw=.8)
        ax.set(xlabel="Candidate axial interaction Ua",ylabel="Profiled relative mismatch",title=title,ylim=(1e-6,2))
        ax.legend(fontsize=8)
    fig.suptitle("Full-band hypotheses: g is profiled over [0.2, 1.6]; synthetic calibrated data")
    fig.savefig(output/"detector_profiles.pdf");fig.savefig(output/"detector_profiles.png",dpi=170);plt.close(fig)
    report=dict(status="CALIBRATED_DETECTOR_PROFILE_AUDIT_COMPLETE",completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,settings=dict(true_g=.8,true_ua=8.,mu=.5,T=.04,
            ua_grid=uas.tolist(),g_grid=ggrid.tolist(),calibration_known=True,material_data_used=False),
        datasets=meta,profile_rows=len(profiles),same_g_pairs=len(pairs),sensitivity=svd,
        tolerance_sets=admitted,derivative_checks=dcheck,
        source_input_sha256=hashes,inputs_sources_unchanged=all(sha(ROOT/k)==v for k,v in hashes.items()),
        limitations=["Finite sampled frequency window, fixed mu and imposed eta=.015.",
            "B0 and Bplus have different equilibrium density; their joint fit assumes the same microscopic g and Ua.",
            "Local axial response is a specified theoretical source/detector, not an implemented experimental probe.",
            "Known phase/amplitude calibration is assumed for complex responses; loss-only profiles are separate.",
            "Tolerance sets are deterministic mismatch scenarios, not measured noise, confidence intervals or uniqueness proofs.",
            "Bounds restrict the hypotheses; the numerical g profile is not certified globally.",
            "General response derivative algebra and inverse conditioning are known; novelty requires the model-specific comparison."])
    assert report["inputs_sources_unchanged"]
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","profile_rows","elapsed_seconds","derivative_checks")},indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);main(ap.parse_args().output)
