"""Independent direct cofactor sums, without importing the local-root evaluator."""
import hashlib,json
from pathlib import Path
import mpmath as mp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=Path(__file__).with_suffix(".json")
    if out.exists():raise FileExistsError("Frozen independent output already exists")
    mp.mp.dps=200
    data_path=ROOT/"band_precision_audit/local_poles_110/all_3456_transitions_110.json"
    paths=[ROOT/"zero_full_input_followup"/d/"cases.json" for d in ("representatives_01","remaining_000_363","remaining_364_727")]
    inputs=[Path(__file__).resolve(),data_path,*paths]
    hashes={str(p):sha(p) for p in inputs}
    rows=json.loads(data_path.read_text(encoding="utf-8"))
    transitions=[(mp.mpf(r["delta"]),mp.mpf(r["cc"]),mp.mpf(r["aa"]),mp.mpc(r["ca"]["real"],r["ca"]["imag"])) for r in rows]
    cases=sum([json.loads(p.read_text(encoding="utf-8")) for p in paths],[])
    by_scale=sorted(cases,key=lambda c:mp.mpf(c["pole_zero_separation"]))
    selected={by_scale[i]["raw_index"]:by_scale[i] for i in np.linspace(0,750,9,dtype=int)}
    for c in (max(cases,key=lambda c:abs(mp.mpf(c["direct"]["unscaled_charge_response"]))),
              min(cases,key=lambda c:mp.mpf(c["direct"]["normalized_dielectric_denominator"]))):
        selected[c["raw_index"]]=c
    a,b=float(.8).as_integer_ratio();g=mp.mpf(a)/b
    a,b=float(2*np.sin((2*np.pi/24)/2)).as_integer_ratio();Q=mp.mpf(a)/b
    vc=g/Q**2;ua=mp.mpf(8)
    checked=[]
    for c in selected.values():
        z=mp.mpf(c["zero"]["center"])+mp.mpf(c["zero"]["offset"])
        cc=mp.fsum(wc/(z-d) for d,wc,wa,wx in transitions)
        aa=mp.fsum(wa/(z-d) for d,wc,wa,wx in transitions)
        ca=mp.fsum(wx/(z-d) for d,wc,wa,wx in transitions)
        ac=mp.fsum(mp.conj(wx)/(z-d) for d,wc,wa,wx in transitions)
        determinant=cc*aa-ca*ac
        numerator=cc-ua*determinant
        denominator=1-vc*cc-ua*aa+vc*ua*determinant
        chi=numerator/denominator
        scale=1+mp.fsum(abs(wc/(z-d)) for d,wc,wa,wx in transitions)
        dielectric_ratio=abs(denominator)/(1+abs(vc*cc)+abs(ua*aa)+abs(vc*ua*determinant))
        assert abs(chi)<mp.mpf("1e-50") and dielectric_ratio>mp.mpf("1e-50")
        checked.append(dict(raw_index=c["raw_index"],old_interval_index=c["old_interval_index"],
            absolute_charge_response=mp.nstr(abs(chi),40),charge_only_normalized_residual=mp.nstr(abs(chi)/scale,40),
            normalized_dielectric_denominator=mp.nstr(dielectric_ratio,40),raw_slots=len(transitions)))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    result=dict(status="INDEPENDENT_DIRECT_COFACTOR_REVIEW_PASS",arithmetic_digits=200,input_digits=110,
        root_module_imported=False,checks=checked,source_input_sha256=hashes,inputs_unchanged=True,
        selected_cases=len(checked),maximum_absolute_charge_response=mp.nstr(max(mp.mpf(r["absolute_charge_response"]) for r in checked),40),
        limits=["Selected scale-spanning and worst-residual cases, not another full 751 recomputation",
                "No outward rounding or physical input error bound; old interval mapping still uncertified"])
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
