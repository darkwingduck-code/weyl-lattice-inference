"""Detector-design controls across declared finite-q response datasets.

All costs are deterministic channel/dataset-normalized relative L2 RMS values.
Numerical eta variants are never combined as independent experimental data.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from detector_identifiability_audit import prediction,derivatives,numerical_profile
ROOT=Path(__file__).resolve().parent
MODES=("charge_only","calibrated","shared_axial_gain","separate_axial_gains")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csv_write(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def calibrated_vector(model,truth,names,mode="calibrated"):
    channels=(0,) if mode=="charge_only" else (0,1)
    pieces=[]
    for name in names:
        for c in channels:
            scale=np.linalg.norm(truth[name][:,c,c])
            if scale==0:raise ValueError("Zero reference channel norm")
            d=(model[name][:,c,c]-truth[name][:,c,c])/scale
            pieces.extend((d.real,d.imag))
    return np.concatenate(pieces)/np.sqrt(len(names)*len(channels))
def profiled_residual(models,truths,names,mode):
    if mode not in MODES:raise ValueError("Unknown calibration policy")
    if mode in ("calibrated","charge_only"):
        return calibrated_vector(models,truths,names,mode),{}
    fitted={n:models[n].copy() for n in names};gains={}
    groups=[names] if mode=="shared_axial_gain" else [(n,) for n in names]
    for group in groups:
        numerator=denominator=0.
        for n in group:
            t,m=truths[n][:,1,1],models[n][:,1,1]
            scale2=float(np.vdot(t,t).real)
            if scale2<=0:raise ValueError("Zero reference axial channel")
            numerator+=float(np.vdot(m,t).real)/scale2
            denominator+=float(np.vdot(m,m).real)/scale2
        if denominator<=0:raise ValueError("Zero model axial signal")
        a=max(0.,numerator/denominator)
        for n in group:fitted[n][:,1,1]*=a;gains[n]=a
    return calibrated_vector(fitted,truths,names),gains
def nuisance_columns(truths,names,mode):
    if mode not in ("shared_axial_gain","separate_axial_gains"):
        return np.zeros((len(calibrated_vector(truths,truths,names,mode)),0))
    groups=[names] if mode=="shared_axial_gain" else [(n,) for n in names]
    columns=[]
    for group in groups:
        shifted={n:truths[n].copy() for n in names}
        for n in group:shifted[n][:,1,1]+=truths[n][:,1,1]
        columns.append(calibrated_vector(shifted,truths,names))
    return np.column_stack(columns)
def sensitivity(bare,truths,metadata,names,mode):
    columns=[]
    for j in (0,1):
        shifted={n:truths[n]+derivatives(truths[n],metadata[n]["q"],.8,8.)[j] for n in names}
        columns.append(calibrated_vector(shifted,truths,names,mode))
    jac=np.column_stack(columns);nuisance=nuisance_columns(truths,names,mode)
    if nuisance.shape[1]:
        # Orthogonal projection follows eliminating a local real amplitude.
        q,r=np.linalg.qr(nuisance,mode="reduced")
        if np.min(abs(np.diag(r)))<1e-13:raise ValueError("Dependent or zero nuisance columns")
        reduced=jac-q@(q.T@jac)
    else:reduced=jac
    singular=np.linalg.svd(reduced,compute_uv=False)
    threshold=64*np.finfo(float).eps*max(1.,np.linalg.norm(jac))
    return dict(largest_singular=float(singular[0]),smallest_singular=float(singular[1]),
        condition_number=float(singular[0]/singular[1]) if singular[1]>threshold else None,
        numerically_rank_deficient=bool(singular[1]<=threshold),numerical_rank_threshold=float(threshold),
        nuisance_columns=nuisance.shape[1],norm_Ua_column_before=float(np.linalg.norm(jac[:,1])),
        norm_Ua_column_after=float(np.linalg.norm(reduced[:,1])),
        interpretation="Local log(g),log(Ua) sensitivity after real amplitude projection; not Fisher coverage or global inverse uniqueness")


CASE_DESIGN={
    "baseline":(1.,.5,.015),"qhalf":(.5,.5,.015),"qdouble":(2.,.5,.015),
    "mu_low":(1.,.25,.015),"mu_high":(1.,.75,.015),
    "eta_low":(1.,.5,.0075),"eta_high":(1.,.5,.03)}
BASE_FREQUENCIES_SOURCE=ROOT/"low_energy_mode_audit/transition_sectors.csv"

def read_checked(path,hashes,expected=None):
    path=Path(path).resolve();data=path.read_bytes()
    digest=hashlib.sha256(data).hexdigest();key=str(path)
    if expected is not None and digest!=expected:raise ValueError("Input hash mismatch: "+key)
    if key in hashes and hashes[key]!=digest:raise ValueError("Input changed during run: "+key)
    hashes[key]=digest
    return data.decode("utf-8")

def normalized_manifest_keys(mapping):
    # The Windows sampler serializes Path keys with backslashes.
    normalized={}
    for name,digest in mapping.items():
        name=name.replace("\\","/")
        if name in normalized and normalized[name]!=digest:
            raise ValueError("Conflicting normalized manifest hashes")
        normalized[name]=digest
    return normalized


def require_numbers(actual,expected,label,tolerance=2e-14):
    a=np.asarray(actual,float);e=np.asarray(expected,float)
    if a.shape!=e.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)) or np.max(abs(a-e),initial=0.)>tolerance:
        raise ValueError("Invalid physical design or metadata: "+label)

def matrix_rows(rows,prefix):
    value=np.array([[[complex(float(r[f"{prefix}_{a}{b}_re"]),float(r[f"{prefix}_{a}{b}_im"])) for b in range(2)] for a in range(2)] for r in rows])
    if not np.all(np.isfinite(value)):raise ValueError("Nonfinite response: "+prefix)
    return value

def load_datasets(index_path,window,hashes=None):
    import io
    if window not in ("fixed_lab","scaled_q"):raise ValueError("Unknown frequency window")
    hashes={} if hashes is None else hashes
    initial=set(hashes)
    index=json.loads(read_checked(index_path,hashes))
    if index.get("complete_14_final_datasets") is not True:raise ValueError("Incomplete final-grid index")
    frequency_rows=list(csv.DictReader(io.StringIO(read_checked(BASE_FREQUENCIES_SOURCE,hashes))))
    base=np.array([float(r["omega"]) for r in frequency_rows if r["sector"]=="total"])
    if len(base)!=45 or not np.all(np.isfinite(base)):raise ValueError("Invalid frozen frequency input")
    reports={}
    index_hashes=normalized_manifest_keys(index["input_sha256"])
    for field in ("B0","Bplus"):
        rel=field+"_final/diagnostic.json";path=index_path.parent/rel
        expected=index_hashes.get(rel)
        if expected is None:raise ValueError("Missing final diagnostic hash")
        report=json.loads(read_checked(path,hashes,expected))
        if (report.get("status")!="PARAMETER_SECTOR_SAMPLING_COMPLETE" or report.get("complete_torus") is not True
            or report.get("sources_unchanged") is not True or report.get("inputs_unchanged") is not True
            or report.get("field")!=field or report.get("mesh")!="final"):
            raise ValueError("Final diagnostic is incomplete or inconsistent")
        for mapping in ("source_sha256","input_sha256"):
            if not report.get(mapping):raise ValueError("Missing sampler provenance")
            for name,digest in report[mapping].items():read_checked(ROOT/name,hashes,digest)
        reports[field]=report
    bare,truth,metadata={},{},{}
    found=set()
    for item in index["datasets"]:
        n=item["id"];field,case=item["field"],item["case"]
        if field not in reports or case not in CASE_DESIGN or n!=field+"_"+case or n in bare:
            raise ValueError("Unexpected or duplicate dataset id/field/case")
        factor,mu,eta=CASE_DESIGN[case];q=factor*np.pi/12
        expected_meta=[q,mu,.04,eta]
        require_numbers([item[k] for k in ("q","mu","T","eta")],expected_meta,n)
        mesh=dict(Lx=72,Ly=72 if field=="B0" else 13,Nkz=384 if field=="B0" else 192,
                  mass=0.,wilson=1.,hopping=1.,exchange=1.,theta_x=0.,theta_y=0.,theta_z=0.)
        report=reports[field];record=report["cases"][case]
        for owner,label in ((item["mesh"],"index"),(report["parameters"],"diagnostic"),(record["parameters"],"case")):
            require_numbers([owner[k] for k in mesh],list(mesh.values()),n+" "+label+" mesh")
        require_numbers([record[k] for k in ("q","mu","T","eta")],expected_meta,n+" diagnostic")
        if record["checks"].get("identity_checks_pass") is not True:raise ValueError("Full-band identity checks failed")
        density=float(item["density"])
        require_numbers([density],[record["density"]],n+" density")
        if not 0<density<4:raise ValueError("Invalid full-volume density")
        for key,expected in (("fixed_lab_frequencies",base),("scaled_q_frequencies",base*factor)):
            require_numbers(item[key],expected,n+" "+key)
            require_numbers(record[key],expected,n+" diagnostic "+key)
        path=(index_path.parent/item["csv"]).resolve()
        if path!=(index_path.parent/(field+"_final")/(case+".csv")).resolve():
            raise ValueError("CSV outside declared final field/case")
        digest=item.get("csv_sha256")
        if digest is None or report["output_sha256"].get(path.name)!=digest:raise ValueError("Missing or conflicting CSV hash")
        rows=list(csv.DictReader(io.StringIO(read_checked(path,hashes,digest))))
        omega=np.array([float(r["omega"]) for r in rows])
        if not len(omega) or not np.all(np.isfinite(omega)):raise ValueError("Nonfinite or empty frequency input")
        bfield=0. if field=="B0" else 2*np.pi/36
        for key,value in dict(q=q,mu=mu,T=.04,eta=eta,B=bfield,density=density,Lx=mesh["Lx"],Ly=mesh["Ly"],Nkz=mesh["Nkz"]).items():
            require_numbers([float(r[key]) for r in rows],np.full(len(rows),value),n+" CSV "+key)
        wanted=base if window=="fixed_lab" else base*factor
        ids=[int(np.argmin(abs(omega-w))) for w in wanted]
        if len(set(ids))!=45 or max(abs(omega[i]-w) for i,w in zip(ids,wanted))>2e-14:
            raise ValueError("Declared frequency window missing or not distinct")
        selected=[rows[i] for i in ids];p=matrix_rows(selected,"bare")
        chi=prediction(p,q,.8,8.)
        if not np.all(np.isfinite(chi)) or np.max(abs(matrix_rows(selected,"rpa")-chi))>2e-12:
            raise ValueError("Cached truth response mismatch")
        status=item.get("convergence_status")
        if status not in ("pass","fail","not_certified"):raise ValueError("Missing convergence status")
        if status!="not_certified":
            conv=item["convergence"];cp=(index_path.parent/conv["coarse_csv"]).resolve()
            if cp!=(index_path.parent/(field+"_coarse")/(case+".csv")).resolve():raise ValueError("Unexpected coarse CSV")
            cr=json.loads(read_checked(cp.parent/"diagnostic.json",hashes))
            if (cr.get("status")!="PARAMETER_SECTOR_SAMPLING_COMPLETE" or cr.get("complete_torus") is not True
                or cr.get("sources_unchanged") is not True or cr.get("inputs_unchanged") is not True
                or cr.get("field")!=field or cr.get("mesh")!="coarse"):
                raise ValueError("Incomplete or inconsistent coarse diagnostic")
            for mapping in ("source_sha256","input_sha256"):
                if not cr.get(mapping):raise ValueError("Missing coarse sampler provenance")
                if normalized_manifest_keys(cr[mapping])!=normalized_manifest_keys(report[mapping]):
                    raise ValueError("Coarse and final sampler provenance disagree")
                for name,source_digest in cr[mapping].items():read_checked(ROOT/name,hashes,source_digest)
            coarse_mesh=dict(mesh)
            coarse_mesh.update(Lx=48 if field=="B0" else 72,Ly=48 if field=="B0" else 13,
                               Nkz=192 if field=="B0" else 96)
            coarse_record=cr["cases"][case]
            for owner,label in ((cr["parameters"],"diagnostic"),(coarse_record["parameters"],"case")):
                require_numbers([owner[k] for k in coarse_mesh],list(coarse_mesh.values()),n+" coarse "+label+" mesh")
            require_numbers([coarse_record[k] for k in ("q","mu","T","eta")],expected_meta,n+" coarse diagnostic")
            if coarse_record["checks"].get("identity_checks_pass") is not True:
                raise ValueError("Coarse full-band identity checks failed")
            coarse_density=float(coarse_record["density"])
            require_numbers([coarse_density],[coarse_density],n+" coarse density")
            if not 0<coarse_density<4:raise ValueError("Invalid coarse full-volume density")
            for key,expected in (("fixed_lab_frequencies",base),("scaled_q_frequencies",base*factor)):
                require_numbers(coarse_record[key],expected,n+" coarse diagnostic "+key)
            if cr["output_sha256"].get(cp.name)!=conv["coarse_sha256"] or conv["fine_sha256"]!=digest:
                raise ValueError("Convergence file hashes disagree")
            coarse=list(csv.DictReader(io.StringIO(read_checked(cp,hashes,conv["coarse_sha256"]))))
            require_numbers([float(r["omega"]) for r in coarse],omega,n+" convergence frequencies",0.)
            for key,value in dict(q=q,mu=mu,T=.04,eta=eta,B=bfield,density=coarse_density,
                                  Lx=coarse_mesh["Lx"],Ly=coarse_mesh["Ly"],Nkz=coarse_mesh["Nkz"]).items():
                require_numbers([float(r[key]) for r in coarse],np.full(len(coarse),value),n+" coarse CSV "+key)
            errors={key:float(np.max(abs(matrix_rows(rows,key)-matrix_rows(coarse,key))/(1+abs(matrix_rows(rows,key)))))
                    for key in ("bare","low","rpa","naive","matched")}
            require_numbers(list(conv["scaled_errors"][k] for k in errors),list(errors.values()),n+" convergence errors",2e-14)
            if conv["tolerance"]!=.001 or (max(errors.values())<=.001)!=(status=="pass") or conv["pass_all"]!=(status=="pass"):
                raise ValueError("Convergence status or tolerance disagrees with data")
        bare[n]=p;truth[n]=chi
        metadata[n]={**item,"window":window,"selected_frequencies":wanted.tolist(),"samples":45}
        found.add((field,case))
    if found!={(f,c) for f in reports for c in CASE_DESIGN} or len(bare)!=14:
        raise ValueError("Require the complete 14-dataset final-grid design")
    return bare,truth,metadata,[Path(p) for p in hashes if p not in initial]


def dataset_groups(metadata):
    lookup={(m["field"],m["case"]):n for n,m in metadata.items()}
    groups=[dict(id="single_"+n,names=[n],kind="single") for n in sorted(metadata)]
    for case in ("baseline","qhalf","qdouble","mu_low","mu_high","eta_low","eta_high"):
        groups.append(dict(id="field_pair_"+case,names=[lookup[f,case] for f in ("B0","Bplus")],kind="field_pair"))
    for axis,cases in (("q",("qhalf","baseline","qdouble")),("mu",("mu_low","baseline","mu_high"))):
        for field in ("B0","Bplus","both"):
            fields=("B0","Bplus") if field=="both" else (field,)
            groups.append(dict(id=axis+"_scan_"+field,names=[lookup[f,c] for f in fields for c in cases],kind=axis+"_scan"))
    for group in groups:
        # eta is a numerical broadening, never a second experimental dataset.
        if len({metadata[n]["eta"] for n in group["names"]})!=1:
            raise ValueError("Combining different numerical eta values is forbidden")
    return groups

def main(index_path,output):
    if output.exists() and any(output.iterdir()):raise RuntimeError("New empty output required")
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    source_paths=[Path(__file__).resolve(),ROOT/"test_parameter_calibration_audit.py",ROOT/"test_parameter_calibration_reader.py",ROOT/"detector_identifiability_audit.py"]
    profiles=[];sensitivities=[];all_groups=[];hashes={str(p):sha(p) for p in source_paths}
    for window in ("fixed_lab","scaled_q"):
        bare,truth,meta,inputs=load_datasets(index_path,window,hashes)
        groups=dataset_groups(meta)
        if window=="scaled_q":
            groups=[g for g in groups if any(meta[n]["case"] in ("qhalf","qdouble") for n in g["names"])]
        for group in groups:
            names=tuple(group["names"])
            policies=MODES if len(names)>1 else MODES[:3]
            group_record=dict(**group,window=window,datasets={n:meta[n] for n in names},
                physical_assumptions=["Same g and Ua across this group","Each channel has known phase; only declared real nonnegative amplitude nuisances are varied","Fixed chemical potentials do not imply fixed density","Multiple numerical eta values are never combined"])
            all_groups.append(group_record)
            for mode in policies:
                sr=sensitivity(bare,truth,meta,names,mode)
                sensitivities.append(dict(group=group["id"],window=window,mode=mode,**sr))
                for ua in np.linspace(0,24,49):
                    def models(g):
                        return {n:prediction(bare[n],meta[n]["q"],g,ua) for n in names}
                    def cost(g):
                        v,_=profiled_residual(models(g),truth,names,mode)
                        return float(v@v)
                    fit=numerical_profile(cost,np.linspace(.2,1.6,65))
                    v,gains=profiled_residual(models(fit["g"]),truth,names,mode)
                    profiles.append(dict(group=group["id"],kind=group["kind"],window=window,mode=mode,datasets=len(names),ua=float(ua),
                        best_g=fit["g"],normalized_mismatch=float(np.linalg.norm(v)),
                        axial_gains=json.dumps(gains,sort_keys=True),
                        all_refinements_successful=fit["all_refinements_successful"],global_minimum_certified=False,
                        all_declared_grid_comparisons_pass=all(meta[n].get("convergence_status")=="pass" for n in names)))
            csv_write(output/"profiles.csv",profiles)
            print(f'{window} {group["id"]}: {len(profiles)} profiles accumulated',flush=True)
    assert all(sha(Path(p))==v for p,v in hashes.items())
    # Every policy contains the synthetic truth; it must reproduce itself.
    self_errors=[r["normalized_mismatch"] for r in profiles if r["ua"]==8.]
    if max(self_errors)>1e-7:raise RuntimeError("Truth parameter failed numerical profile reproduction")
    csv_write(output/"local_sensitivity.csv",sensitivities)
    dump(output/"groups.json",all_groups)
    sets=[]
    for group in all_groups:
        rows=[r for r in profiles if r["group"]==group["id"] and r["window"]==group["window"]]
        for mode in sorted({r["mode"] for r in rows}):
            for threshold in (.001,.01):
                accepted=[r["ua"] for r in rows if r["mode"]==mode and r["normalized_mismatch"]<=threshold]
                sets.append(dict(group=group["id"],window=group["window"],mode=mode,threshold=threshold,accepted_sampled_ua=accepted))
    dump(output/"tolerance_sets.json",sets)
    result=dict(status="PARAMETER_CALIBRATION_COMPLETE",completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,profile_count=len(profiles),group_window_count=len(all_groups),
        source_input_sha256=hashes,inputs_sources_unchanged=True,maximum_truth_profile_error=max(self_errors),
        all_local_refinements_successful=all(r["all_refinements_successful"] for r in profiles),
        global_inverse_unique_certified=False,statistical_confidence_intervals=False,
        scope="Only the explicit final-grid datasets/windows and shared or separate calibration assumptions; consult case convergence records")
    dump(output/"diagnostic.json",result)
    dump(output/"manifest.json",{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-index",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();main(args.input_index.resolve(),args.output.resolve())
