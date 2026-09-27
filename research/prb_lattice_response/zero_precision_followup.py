"""Bounded precision follow-up for frozen finite-lattice charge-zero evidence.

No band, transition weight, or frozen output is modified. Decimal calculations
treat the stored binary64 inputs literally; they are not a high-precision
Hamiltonian diagonalization. Directed-rounding brackets certify only the stated
rounded positive weighted modal transfer, not automatically the original RPA.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal,Context,localcontext,ROUND_FLOOR,ROUND_CEILING
from pathlib import Path
import numpy as np
from lattice_response import Parameters,interaction_matrix
from bloch_response import bloch_spectrum,bloch_transitions
from pole_zero_analysis import RationalResponse

ROOT=Path(__file__).resolve().parent
D=lambda x:Decimal.from_float(float(x))
PREC=110
DIRECT_TOL=Decimal("1e-7")
OBSERVED_EIGEN_DIFFERENCE=Decimal("3.8191672047105385e-14")

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as stream:return list(csv.DictReader(stream))

class DecimalRational:
    """Full raw or exactly factored G S G* transition sum."""
    def __init__(self,tr,u,kind="raw",precision=PREC):
        self.precision=precision
        active=(tr.cc!=0)|(tr.aa!=0)|(tr.ca!=0)
        self.delta=[D(x) for x in tr.delta[active]]
        self.u=(D(u[0,0].real),D(u[1,1].real))
        if kind=="raw":
            self.columns=[[D(x) for x in a[active]] for a in (tr.cc,tr.aa,tr.ca.real,tr.ca.imag)]
        elif kind=="factored":
            response=RationalResponse(tr)
            self.columns=[[] for _ in range(4)]
            with localcontext(Context(prec=precision)):
                for i in range(len(response.delta)):
                    a,b=response.G[:,i]
                    ar,ai,br,bi=map(D,(a.real,a.imag,b.real,b.imag))
                    s=D(response.sign[i])
                    for column,value in zip(self.columns,(s*(ar*ar+ai*ai),s*(br*br+bi*bi),
                                                         s*(ar*br+ai*bi),s*(ai*br-ar*bi))):
                        column.append(value)
        else:raise ValueError("Unknown weight convention")

    def entries(self,x):
        with localcontext(Context(prec=self.precision)):
            inverse=[1/(x-d) for d in self.delta]
            return [sum((w*v for w,v in zip(column,inverse)),Decimal(0)) for column in self.columns]

    def evaluate(self,x,center=None):
        """Return scaled N,D and their derivatives; N/D is always chi_cc."""
        with localcontext(Context(prec=self.precision)):
            vc,ua=self.u
            if center is None:
                if x in self.delta:raise ValueError("Exact bare transition")
                inverses=[1/(x-d) for d in self.delta]
                p=[sum((w*t for w,t in zip(c,inverses)),Decimal(0)) for c in self.columns]
                dp=[-sum((w*t*t for w,t in zip(c,inverses)),Decimal(0)) for c in self.columns]
                det=p[0]*p[1]-p[2]*p[2]-p[3]*p[3]
                dd=dp[0]*p[1]+p[0]*dp[1]-2*p[2]*dp[2]-2*p[3]*dp[3]
                return p[0]-ua*det,1-vc*p[0]-ua*p[1]+vc*ua*det,dp[0]-ua*dd,-vc*dp[0]-ua*dp[1]+vc*ua*dd
            t=x-center
            if t==0:raise ValueError("Laurent multiplication zero at the bare transition is excluded")
            selected=[i for i,d in enumerate(self.delta) if d==center]
            other=[i for i,d in enumerate(self.delta) if d!=center]
            if not selected:raise ValueError("Laurent center must be an actual bare transition")
            if any(x==self.delta[i] for i in other):raise ValueError("Another bare transition")
            inverse=[1/(x-self.delta[i]) for i in other]
            w=[sum((c[i] for i in selected),Decimal(0)) for c in self.columns]
            r=[sum((c[i]*v for i,v in zip(other,inverse)),Decimal(0)) for c in self.columns]
            dr=[-sum((c[i]*v*v for i,v in zip(other,inverse)),Decimal(0)) for c in self.columns]
            q=[a+t*b for a,b in zip(w,r)]
            dq=[a+t*b for a,b in zip(r,dr)]
            det=q[0]*q[1]-q[2]*q[2]-q[3]*q[3]
            dd=dq[0]*q[1]+q[0]*dq[1]-2*q[2]*dq[2]-2*q[3]*dq[3]
            return (t*q[0]-ua*det,t*t-t*(vc*q[0]+ua*q[1])+vc*ua*det,
                    q[0]+t*dq[0]-ua*dd,
                    2*t-vc*q[0]-ua*q[1]-t*(vc*dq[0]+ua*dq[1])+vc*ua*dd)

    def root(self,start,which="numerator",bounds=None,laurent=False):
        with localcontext(Context(prec=self.precision)):
            x=start if isinstance(start,Decimal) else D(start)
            center=min(self.delta,key=lambda d:abs(x-d)) if laurent else None
            left=max((d for d in self.delta if d<x),default=x-100)
            right=min((d for d in self.delta if d>x),default=x+100)
            if x in self.delta:raise ValueError("Starting point is a bare pole")
            if bounds:
                left=max(left,bounds[0]);right=min(right,bounds[1])
            if not left<x<right:raise ValueError("Starting point outside its regular component")
            converged=False
            for iteration in range(30):
                n,den,dn,dd=self.evaluate(x,center)
                value,derivative=(n,dn) if which=="numerator" else (den,dd)
                if derivative==0:raise ValueError("Zero derivative")
                step=value/derivative
                target=x-step
                for _ in range(80):
                    if left<target<right:break
                    step/=2;target=x-step
                if not left<target<right:raise ValueError("Could not stay in regular component")
                x=target
                if abs(step)<Decimal("1e-85")*max(1,abs(x)):
                    converged=True;break
            n,den,dn,dd=self.evaluate(x,center)
            if not converged:raise ValueError("Precision Newton did not converge")
            if x in self.delta:raise ValueError("Spurious bare multiplication root")
            return x,dict(iterations=iteration+1,laurent_center=str(center) if center is not None else None,
                          numerator=str(n),denominator=str(den),numerator_derivative=str(dn),
                          denominator_derivative=str(dd),nearest_bare_distance=str(min(abs(x-d) for d in self.delta)),
                          regular_component=[str(left),str(right)])

def deflated_sign_interval(poles,weights,index,offset,precision=PREC):
    """Outward-rounded enclosure of w_j+t sum_{i!=j} w_i/(lambda_j+t-lambda_i)."""
    if weights[index]<=0:raise ValueError("A nonzero positive modal weight is required")
    floor,ceil=Context(prec=precision,rounding=ROUND_FLOOR),Context(prec=precision,rounding=ROUND_CEILING)
    lower=upper=Decimal(0)
    for i,(pole,weight) in enumerate(zip(poles,weights)):
        if i==index:continue
        dlo=floor.subtract(floor.add(poles[index],offset),pole)
        dhi=ceil.subtract(ceil.add(poles[index],offset),pole)
        if dlo<=0<=dhi:raise ValueError("Deflated interval reaches another modal pole")
        lower=floor.add(lower,floor.divide(weight,dhi))
        upper=ceil.add(upper,ceil.divide(weight,dlo))
    if offset>=0:
        lo=floor.add(weights[index],floor.multiply(offset,lower))
        hi=ceil.add(weights[index],ceil.multiply(offset,upper))
    else:
        lo=floor.add(weights[index],floor.multiply(offset,upper))
        hi=ceil.add(weights[index],ceil.multiply(offset,lower))
    return lo,hi

def modal_bracket(poles,weights,index,estimate,left,right):
    with localcontext(Context(prec=PREC)):
        offsets=sorted((estimate/2,estimate*2))
        if not left<poles[index]+offsets[0]<poles[index]+offsets[1]<right:
            raise ValueError("Initial deflated bracket leaves its original pole interval")
        a=deflated_sign_interval(poles,weights,index,offsets[0])
        b=deflated_sign_interval(poles,weights,index,offsets[1])
        certified=(a[1]<0<b[0]) or (b[1]<0<a[0])
        return dict(status="CERTIFIED_ROUNDED_WEIGHTED_MODAL_BRACKET" if certified else "UNRESOLVED_MODAL_BRACKET",
                    offset_bracket=[str(x) for x in offsets],left_sign_enclosure=[str(x) for x in a],
                    right_sign_enclosure=[str(x) for x in b],
                    scope="Directed-rounding certificate for literal CSV positive weighted transfer only.")

def decimal_modal(x,poles,residues,replacement=None):
    with localcontext(Context(prec=PREC)):
        total=Decimal(0)
        for i,(pole,residue) in enumerate(zip(poles,residues)):
            if replacement is not None and i==replacement[0]:pole,residue=replacement[1:]
            total+=residue/(x-pole)
        return total

def verify_direct(model,start,left,right,poles,residues):
    x,detail=model.root(start,bounds=(left,right),laurent=True)
    with localcontext(Context(prec=PREC)):
        n,den,_,_=model.evaluate(x, D(detail["laurent_center"]))
        if den==0:raise ValueError("Numerator root also has a dielectric pole")
        scale=sum((abs(r/(x-p)) for p,r in zip(poles,residues)),Decimal(0))
        normalized=abs(n/den)/scale
        # Independent unscaled evaluation cross-checks the Laurent algebra.
        n0,d0,_,_=model.evaluate(x)
        normalized_unscaled=abs(n0/d0)/scale
        detail.update(root_decimal=str(x),root_float=float(x),normalized_laurent_charge=str(normalized),
                      normalized_unscaled_charge=str(normalized_unscaled),
                      original_modal_interval_contains_root=bool(left<x<right),
                      status="VERIFIED_FIXED_INPUT_LAURENT_ZERO" if max(normalized,normalized_unscaled)<=DIRECT_TOL
                      else "UNRESOLVED_FIXED_INPUT_ZERO",
                      input_scope="Exact supplied float64 energies and raw cc/aa/ca; not a high-precision Hamiltonian.")
        return detail

def main(output):
    if output.exists() and any(output.iterdir()):raise RuntimeError("New empty output required")
    started=time.perf_counter()
    source_paths=[Path(__file__).resolve(),ROOT/"test_zero_precision_followup.py"]
    frozen={str(p):sha(p) for p in source_paths}
    for folder in ("mode_analysis","zero_audit","zero_refinement/guarded_run"):
        for p in (ROOT/folder).rglob("*"):
            if p.is_file():frozen[str(p)]=sha(p)
    prod=json.loads((ROOT/"mode_analysis/diagnostic.json").read_text())
    for name,expected in prod["source_sha256"].items():
        if sha(ROOT/name)!=expected:raise RuntimeError("Changed scientific source "+name)
        frozen[str(ROOT/name)]=expected
    for name in ("zero_audit.py","refine_charge_zeros.py"):frozen[str(ROOT/name)]=sha(ROOT/name)
    for name,expected in prod["output_sha256"].items():
        if sha(ROOT/"mode_analysis"/name)!=expected:raise RuntimeError("Changed production artifact "+name)
    for folder in ("zero_audit","zero_refinement/guarded_run"):
        manifest=json.loads((ROOT/folder/"manifest.json").read_text())
        if any(sha(ROOT/folder/name)!=value for name,value in manifest.items()):raise RuntimeError("Changed audit manifest payload")
    audit=json.loads((ROOT/"zero_audit/zero_checks.json").read_text())
    rows=read_csv(ROOT/"mode_analysis/finite_poles.csv")
    pf=np.array([float(r["pole"]) for r in rows]);wf=np.array([float(r["weighted_charge"]) for r in rows])
    poles=[D(x) for x in pf];weights=[D(x) for x in wf]
    residues=[D(r["residue_00_real"]) for r in rows]
    params=Parameters(**prod["finite_realization"]["parameters"]);settings=prod["settings"]
    tr=bloch_transitions(bloch_spectrum(params),settings["mu"],settings["T"],1)
    if abs(tr.q-settings["q"])>1e-14:raise RuntimeError("Unexpected q shift")
    u=interaction_matrix(tr.Q,settings["g"],settings["Ua"])
    raw=DecimalRational(tr,u,"raw");factored=DecimalRational(tr,u,"factored")
    output.mkdir(parents=True,exist_ok=True)
    classified=[]
    selected=[]
    for row in audit:
        if not row["status"].startswith("PRESERVED"):continue
        left,right=row["left_pole"],row["right_pole"]
        margin=1e-10*max(1.,abs(left),abs(right))
        side=right if np.sum(wf/(left+margin-pf))>0 else left
        index=int(np.flatnonzero(pf==side)[0])
        keep=np.arange(len(pf))!=index
        regular=math.fsum((wf[keep]/(pf[index]-pf[keep])).tolist())
        with localcontext(Context(prec=PREC)):
            estimate=-weights[index]/D(regular)
            separation=abs(estimate)
            ulp=D(abs(np.spacing(pf[index])))
            record=dict(index=row["index"],original_status=row["status"],modal_endpoint_index=index,
                        modal_endpoint=str(poles[index]),modal_weight=str(weights[index]),
                        estimated_offset=str(estimate),offset_to_float_ulp=float(separation/ulp),
                        offset_to_observed_eigen_difference=float(separation/OBSERVED_EIGEN_DIFFERENCE),
                        observed_eigen_difference_is_rigorous_bound=False)
            try:record.update(modal_bracket(poles,weights,index,estimate,D(left),D(right)))
            except (ValueError,ArithmeticError) as error:record.update(status="UNRESOLVED_MODAL_ENCLOSURE",reason=str(error))
            if separation>100*OBSERVED_EIGEN_DIFFERENCE:selected.append((row,poles[index]+estimate))
            record["direct_rational_followup_selected"]=bool(separation>100*OBSERVED_EIGEN_DIFFERENCE)
            classified.append(record)
        if len(classified)%100==0:
            print(f"[modal] {len(classified)}/782",flush=True)
            dump(output/"modal_interval_classification.json",classified)
    dump(output/"modal_interval_classification.json",classified)
    direct=[]
    bare=[r for r in audit if r["status"]=="UNRESOLVED_NEAR_BARE_TRANSITION"]
    jobs=[(r,D(r["zero"]),"previous_bare_proximity") for r in bare]+[(r,x,"selected_previous_pole_proximity") for r,x in selected]
    for row,start,group in jobs:
        record=dict(index=row["index"],group=group,previous_status=row["status"])
        try:record.update(verify_direct(raw,start,D(row["left_pole"]),D(row["right_pole"]),poles,residues))
        except (ValueError,ArithmeticError) as error:record.update(status="UNRESOLVED_DIRECT_FOLLOWUP",reason=str(error))
        direct.append(record)
        if len(direct)%10==0:print(f"[direct] {len(direct)}/{len(jobs)}",flush=True)
        dump(output/"direct_laurent_cases.json",direct)
    prior=json.loads((ROOT/"zero_refinement/guarded_run/refined_candidates.json").read_text())
    four=[]
    for record in prior:
        if record["original_index"] not in (457,1420,1551,2195):continue
        with localcontext(Context(prec=PREC)):
            x,xd=raw.root(Decimal(record["decimal_diagnostic"]["root_decimal"]),laurent=True)
            j=int(np.argmin(abs(pf-float(x))))
            lam_raw,rd=raw.root(poles[j],which="denominator")
            lam_g,gd=factored.root(poles[j],which="denominator")
            n_g,d_g,_,dd_g=factored.evaluate(lam_g)
            residue_g=n_g/dd_g
            ng,dg,_,_=factored.evaluate(x);chi_g=ng/dg
            chi_modal=decimal_modal(x,poles,residues)
            corrected_lambda=decimal_modal(x,poles,residues,(j,lam_g,residues[j]))
            corrected_both=decimal_modal(x,poles,residues,(j,lam_g,residue_g))
            exact_lambda_change=residues[j]*(lam_g-poles[j])/((x-lam_g)*(x-poles[j]))
            measured_error=abs(chi_modal-chi_g)
            nr,dr,_,_=raw.evaluate(x)
            pr,pg=raw.entries(x),factored.entries(x)
            vc,ua=raw.u
            bound_entries=[sum((abs(a-b)/abs(x-d) for a,b,d in zip(cr,cg,raw.delta)),Decimal(0))
                           for cr,cg in zip(raw.columns,factored.columns)]
            ca_bound=(bound_entries[2]**2+bound_entries[3]**2).sqrt()
            pi_infinity_bound=max(bound_entries[0]+ca_bound,bound_entries[1]+ca_bound)
            cg_abs=(pg[2]**2+pg[3]**2).sqrt();cr_abs=(pr[2]**2+pr[3]**2).sqrt()
            left_inverse_bound=max(abs(1-ua*pg[1])+ua*cg_abs,vc*cg_abs+abs(1-vc*pg[0]))/abs(dg)
            right_inverse_bound=max(abs(1-ua*pr[1])+vc*cr_abs,ua*cr_abs+abs(1-vc*pr[0]))/abs(dr)
            rpa_factorization_bound=left_inverse_bound*pi_infinity_bound*right_inverse_bound
            explanation=dict(index=record["original_index"],nearest_modal_index=j,zero_raw=str(x),
                             pole_csv=str(poles[j]),pole_raw_exact_input=str(lam_raw),pole_exact_GSG=str(lam_g),
                             raw_to_factored_pole_shift=str(lam_g-lam_raw),
                             factored_to_csv_realization_shift=str(poles[j]-lam_g),
                             zero_to_csv_pole_distance=str(abs(x-poles[j])),
                             raw_charge_at_raw_zero=str(nr/dr),factored_charge_at_raw_zero=str(chi_g),
                             polarization_factorization_infinity_bound=str(pi_infinity_bound),
                             rpa_factorization_absolute_bound=str(rpa_factorization_bound),
                             factorization_bound_contains_observed_difference=bool(abs(chi_g-nr/dr)<=rpa_factorization_bound),
                             factorization_bound_scope="Analytic resolvent perturbation norm bound evaluated at 110 digits, not directed-rounding certification.",
                             rounded_modal_charge_at_raw_zero=str(chi_modal),
                             modal_after_one_pole_position_correction=str(corrected_lambda),
                             modal_after_one_pole_position_and_residue_correction=str(corrected_both),
                             exact_one_pole_position_perturbation=str(exact_lambda_change),
                             one_pole_position_correction_identity_error=str(abs((corrected_lambda-chi_modal)-exact_lambda_change)),
                             absolute_modal_factored_discrepancy=str(measured_error),
                             residual_fraction_after_position_correction=float(abs(corrected_lambda-chi_g)/measured_error),
                             residual_fraction_after_position_and_residue_correction=float(abs(corrected_both-chi_g)/measured_error),
                             residue_csv=str(residues[j]),residue_exact_GSG=str(residue_g),
                             scope="Raw rounded W vs exact arithmetic GSG* vs rounded modal sum; realization shift includes matrix construction, Cholesky, eigensolve and CSV rounding.")
            four.append(explanation)
            print(f"[conditioning] index={record['original_index']} residual fraction={explanation['residual_fraction_after_position_and_residue_correction']:.4g}",flush=True)
            dump(output/"four_mode_precision_cases.json",four)
    if any(sha(Path(p))!=v for p,v in frozen.items()):raise RuntimeError("Frozen input/source changed during computation")
    summary=dict(status="BOUNDED_ZERO_PRECISION_FOLLOWUP_COMPLETED",completed_utc=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.perf_counter()-started,source_input_sha256=frozen,
                 modal_cases=len(classified),modal_status_counts=dict(Counter(x["status"] for x in classified)),
                 offset_above_observed_eigen_difference=sum(x["offset_to_observed_eigen_difference"]>1 for x in classified),
                 offset_below_float_ulp=sum(x["offset_to_float_ulp"]<1 for x in classified),
                 direct_cases=len(direct),direct_status_counts=dict(Counter(x["status"] for x in direct)),
                 direct_groups={g:dict(Counter(x["status"] for x in direct if x["group"]==g)) for g in ("previous_bare_proximity","selected_previous_pole_proximity")},
                 four_conditioning_cases=len(four),frozen_inputs_unchanged=True,
                 complete_physical_zero_catalog_certified=False,submission_ready=False,
                 retained_all_nonzero_modal_and_transition_weights=True,
                 limitations=["Directed rounding certifies literal rounded weighted modal transfer only.",
                              "110-digit direct/Laurent arithmetic uses original binary64 band energies and matrix weights.",
                              "Empirical eigenvalue differences select bounded follow-up work; they are not rigorous Hamiltonian uncertainty bounds.",
                              "No guard was globally removed: bare-proximity cases use a separate regularized Laurent calculation, reject t=0 and check the unscaled rational response.",
                              "Unselected direct-RPA intervals remain physically unresolved even if their rounded-modal bracket is certified."])
    dump(output/"diagnostic.json",summary)
    dump(output/"manifest.json",{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"})
    print(json.dumps({k:summary[k] for k in ("status","modal_status_counts","direct_groups","four_conditioning_cases")},indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output.resolve())
