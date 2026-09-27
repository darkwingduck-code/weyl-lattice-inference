"""Calibration nuisance control, kept separate from the frozen calibrated audit."""
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from detector_identifiability_audit import ROOT,sha,prediction,numerical_profile
from low_energy_mode_audit import load_response

def gain_profile(models,truths,names,mode):
    out={name:models[name].copy() for name in names};gains={}
    channels=(1,) if mode=="axial_gain_only" else (0,1)
    for c in channels:
        # Shared positive real gain across both fields, channel-normalized.
        numer=denom=0.
        for name in names:
            t=truths[name][:,c,c];m=models[name][:,c,c];scale=np.linalg.norm(t)
            numer+=float(np.vdot(m,t).real)/(scale*scale)
            denom+=float(np.vdot(m,m).real)/(scale*scale)
        if denom<=0:raise ValueError("No nonzero predicted signal to calibrate")
        gain=max(0.,numer/denom);gains[str(c)]=gain
        for name in names:out[name][:,c,c]*=gain
    residual=[]
    for name in names:
        for c in (0,1):
            d=(out[name][:,c,c]-truths[name][:,c,c])/np.linalg.norm(truths[name][:,c,c])
            residual.extend([d.real,d.imag])
    return np.concatenate(residual)/np.sqrt(2*len(names)),gains

def self_test():
    t={"A":np.array([np.diag([1+2j,3+4j]),np.diag([2+1j,4+3j])])}
    m={"A":t["A"].copy()};m["A"][:,1,1]*=7
    v,g=gain_profile(m,t,("A",),"axial_gain_only")
    assert np.linalg.norm(v)<1e-14 and abs(g["1"]-1/7)<1e-14
    m["A"][:,1,1]*=-1
    v,g=gain_profile(m,t,("A",),"axial_gain_only")
    assert g["1"]==0 and np.linalg.norm(v)>.5
    # A shared gain cannot hide a field-dependent amplitude distortion.
    t["B"]=t["A"].copy();m={n:t[n].copy() for n in t}
    m["A"][:,1,1]*=2;m["B"][:,1,1]*=3
    v,g=gain_profile(m,t,("A","B"),"axial_gain_only")
    assert np.linalg.norm(v)>.05
    return {"positive_gain_recovered":True,"negative_gain_rejected":True,"field_dependent_distortion_not_hidden_by_shared_gain":True}

def main(output):
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    inputs={n:ROOT/p for n,p in {"B0":"transverse_extension/L72_Nk384_flux+0_fixed_mu.csv",
             "Bplus":"magnetic_kz_continuation/Nkx2_Nky13_Nk192_B1of36_fixed_mu.csv"}.items()}
    paths=[*inputs.values(),Path(__file__),ROOT/"detector_identifiability_audit.py",ROOT/"low_energy_mode_audit.py",ROOT/"lattice_response.py"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    pi,truth,q={},{},{}
    for n,p in inputs.items():
        w,data,meta=load_response(p);mask=(w>=.05)&(w<=.35);pi[n]=data["bare"][mask];q[n]=meta["q"]
        truth[n]=prediction(pi[n],q[n],.8,8.)
    rows=[]
    for dataset in ("B0","Bplus","joint"):
        names=("B0","Bplus") if dataset=="joint" else (dataset,)
        for mode in ("axial_gain_only","both_gains"):
            for ua in np.linspace(0,24,49):
                def cost(g):
                    model={n:prediction(pi[n],q[n],g,ua) for n in names}
                    v,_=gain_profile(model,truth,names,mode);return float(v@v)
                fit=numerical_profile(cost,np.linspace(.2,1.6,65))
                model={n:prediction(pi[n],q[n],fit["g"],ua) for n in names}
                v,gains=gain_profile(model,truth,names,mode)
                rows.append(dict(dataset=dataset,mode=mode,ua=float(ua),best_g=fit["g"],
                    charge_gain=gains.get("0",1.),axial_gain=gains["1"],
                    normalized_mismatch=float(np.linalg.norm(v)),refinement_success=fit["all_refinements_successful"]))
            print(dataset,mode,"completed",flush=True)
    with (output/"calibration_profiles.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sets=[]
    for dataset in ("B0","Bplus","joint"):
        for mode in ("axial_gain_only","both_gains"):
            for threshold in (.001,.01):
                sets.append(dict(dataset=dataset,mode=mode,threshold=threshold,
                    accepted_sampled_ua=[r["ua"] for r in rows if r["dataset"]==dataset and r["mode"]==mode and r["normalized_mismatch"]<=threshold]))
    report=dict(status="CALIBRATION_NUISANCE_CONTROL_COMPLETE",checked_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,self_tests=self_test(),rows=len(rows),tolerance_sets=sets,
        source_input_sha256=hashes,all_sources_inputs_unchanged=all(sha(ROOT/p)==v for p,v in hashes.items()),
        limitations=["Positive real amplitude gains only; phase/frequency dependence is not profiled.",
            "Gains are shared across fields in joint fits, and each channel is separately normalized by its reference norm.",
            "Deterministic sampled mismatch sets, not confidence intervals or global inverse uniqueness.",
            "No experimental axial detector or calibration procedure is supplied."])
    assert report["all_sources_inputs_unchanged"]
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"rows":len(rows),"elapsed_seconds":report["elapsed_seconds"]}))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);main(ap.parse_args().output)
