"""Post-result scalar screening explanation; no parameters are fitted."""
from pathlib import Path
import csv,hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/"scalar_screening_explanation"
    if out.exists() and any(out.iterdir()):raise RuntimeError("New empty output required")
    out.mkdir(parents=True,exist_ok=True)
    paths={n:ROOT/f"uv_matching_audit/{n}_zero_real_reference/diagnostic.json" for n in ("B0","Bplus")}
    gainpath=ROOT/"detector_calibration_followup/calibration_profiles.csv"
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(),gainpath,Path(__file__).resolve()]}
    A={n:json.loads(p.read_text())["reference_excluded"][1][1]["real"] for n,p in paths.items()}
    with gainpath.open(encoding="utf-8",newline="") as f:fits=list(csv.DictReader(f))
    rows=[]
    for r in fits:
        if r["mode"]!="axial_gain_only":continue
        ua=float(r["ua"]);gain=float(r["axial_gain"])
        pred={n:(1-ua*a)/(1-8*a) for n,a in A.items()}
        if r["dataset"] in A:
            a=pred[r["dataset"]]
            rows.append(dict(dataset=r["dataset"],ua=ua,fitted_response_gain=gain,approximate_scalar_gain=a,relative_gain_difference=abs(a-gain)/abs(gain),definition="One gain multiplying chi_aa, not one factor per external vertex"))
    with (out/"gain_comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for n in A:
        rr=[r for r in rows if r["dataset"]==n]
        summary[n]=dict(A_excluded_at_imaginary_reference=A[n],maximum_relative_gain_difference=max(r["relative_gain_difference"] for r in rr),ua4=next(r for r in rr if r["ua"]==4),ua12=next(r for r in rr if r["ua"]==12))
    record=dict(status="POST_RESULT_SCALAR_EXPLANATION_COMPLETE",checked_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),source_input_sha256=hashes,summary=summary,formula="chi_a(U)=A/(1-U*A); response_gain(U_trial -> U_truth=8)=(1-U_trial*A)/(1-8*A)",assumptions=["Constant real scalar axial polarization approximated by excluded-sector A at z=i*0.015","Neglect low-sector frequency dependence and cross-channel coupling","Interpretation developed after seeing fitted calibration results; no preregistration or independent held-out prediction is claimed","U_trial values and actual gains come from existing 49-point control; no new fits"],gap_certified=False,novel_identity_claimed=False)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(),gainpath,Path(__file__).resolve()]}
    (out/"diagnostic.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
