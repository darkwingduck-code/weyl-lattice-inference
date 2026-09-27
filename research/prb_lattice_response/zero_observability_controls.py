"""Fixed-bare-Pi diagonal-RPA controls; no new theorem or continuum claim."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from lattice_response import Parameters,interaction_matrix
from bloch_response import bloch_spectrum,bloch_transitions
from zero_audit import raw_response
ROOT=Path(__file__).resolve().parent

def components(pi,dp,ua):
    det=pi[0,0]*pi[1,1]-pi[0,1]*pi[1,0]
    dd=dp[0,0]*pi[1,1]+pi[0,0]*dp[1,1]-dp[0,1]*pi[1,0]-pi[0,1]*dp[1,0]
    return pi[0,0]-ua*det,1-ua*pi[1,1],dp[0,0]-ua*dd,-ua*dp[1,1]

def scalar_response(pi,dp,vc,ua):
    n,a,dn,da=components(pi,dp,ua);d=a-vc*n
    return n/d,(dn*a-n*da)/(d*d),n,d

def regular_zero(pi,dp,vc,ua,tolerance=1e-10):
    n,a,_,_=components(pi,dp,ua);d=a-vc*n
    return bool(abs(d)>tolerance*max(1.,abs(a),abs(vc*n)) and abs(n/d)<=tolerance)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(output):
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/x for x in ["mode_analysis/finite_charge_zeros.csv","zero_audit/zero_checks.json",
            "zero_observability_controls.py","lattice_response.py","bloch_response.py","zero_audit.py"]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    audit=json.loads(paths[1].read_text(encoding="utf-8"))
    choices=[r for r in audit if r["status"]=="PASS_NUMERICAL_ZERO" and 0<r["zero"]<1.2]
    selected=sorted(choices,key=lambda r:r["nearest_modal_pole_distance"],reverse=True)[:8]
    p=Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    tr=bloch_transitions(bloch_spectrum(p),.5,.04,1);gs=[0.,.2,.4,.8,1.6]
    exact,finite,identity=[],[],[]
    for row in selected:
        x=row["zero"];radius=.45*min(row["nearest_modal_pole_distance"],row["nearest_bare_distance"])
        for _ in range(6):
            pi,dp=raw_response(tr,x);n,a,dn,da=components(pi,dp,8.)
            candidate=x-float((n/dn).real)
            if not row["zero"]-radius<candidate<row["zero"]+radius:
                raise ArithmeticError("Newton correction left protected component")
            if candidate==x: break
            x=candidate
        for g in gs:
            u=interaction_matrix(tr.Q,g,8.);pi,dp=raw_response(tr,x)
            cc,slope,n,d=scalar_response(pi,dp,u[0,0],8.)
            direct=np.linalg.solve(np.eye(2)-pi@u,pi)[0,0]
            exact.append(dict(index=row["index"],zero=x,g=g,vc=float(u[0,0]),
                abs_chi=float(abs(cc)),direct_difference=float(abs(direct-cc)),
                denominator_absolute=float(abs(d)),slope_real=float(slope.real),
                slope_imag=float(slope.imag),zero_shift=float(x-row["zero"])))
            for eta in [.03,.015,.0075,.001,1e-5]:
                value=scalar_response(*raw_response(tr,x+1j*eta),u[0,0],8.)[0]
                def objective(omega):
                    return float(abs(scalar_response(*raw_response(tr,omega+1j*eta),u[0,0],8.)[0])**2)
                fit=minimize_scalar(objective,bounds=(x-radius,x+radius),method="bounded",options={"xatol":1e-14})
                best=min([(x-radius,objective(x-radius)),(x+radius,objective(x+radius)),
                          (float(fit.x),float(fit.fun))],key=lambda a:a[1])
                finite.append(dict(index=row["index"],zero=x,g=g,eta=eta,
                    abs_chi_at_real_zero=float(abs(value)),leading_eta_prediction=float(eta*abs(slope)),
                    leading_relative_error=float(abs(value-1j*eta*slope)/max(abs(value),1e-300)),
                    search_lower=x-radius,search_upper=x+radius,minimum_omega=best[0],
                    minimum_abs_chi=float(np.sqrt(best[1])),minimum_shift=best[0]-x,
                    classification="INTERIOR_LOCAL_WINDOW_MINIMUM" if abs(best[0]-x)<.99*radius else "WINDOW_EDGE_NO_INTERIOR_MINIMUM"))
    for z in [.17+.02j,.77+.1j,1.1+.15j,.4-.2j]:
        pi,dp=raw_response(tr,z)
        for g in gs:
            u=interaction_matrix(tr.Q,g,8.)
            actual=scalar_response(pi,dp,u[0,0],8.)[0]
            expected=np.linalg.solve(np.eye(2)-pi@u,pi)[0,0]
            identity.append(dict(z_real=z.real,z_imag=z.imag,g=g,absolute_error=float(abs(actual-expected))))
    write_csv(output/"regular_zero_coulomb_scan.csv",exact)
    write_csv(output/"finite_eta_local_minima.csv",finite)
    write_csv(output/"complex_identity_checks.csv",identity)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":10})
    chosen=selected[0]["index"];fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for g in [0.,.8,1.6]:
        rr=sorted([r for r in finite if r["index"]==chosen and r["g"]==g],key=lambda r:r["eta"])
        axes[0].loglog([r["eta"] for r in rr],[r["abs_chi_at_real_zero"] for r in rr],"o-",label=f"g={g:g}")
        axes[1].semilogx([r["eta"] for r in rr],[r["minimum_shift"] for r in rr],"o-",label=f"g={g:g}")
    rr=sorted([r for r in finite if r["index"]==chosen and r["g"]==0],key=lambda r:r["eta"])
    axes[0].loglog([r["eta"] for r in rr],[r["leading_eta_prediction"] for r in rr],"k--",lw=1,label="eta |chi prime|")
    axes[0].set(xlabel="Regulator eta",ylabel="|chi_cc(zero + i eta)|",title="A real zero is lifted by finite eta")
    axes[1].set(xlabel="Regulator eta",ylabel="Local minimum minus real zero",title="Local-window minima vary with g")
    axes[1].axhline(0,color="gray",ls=":",lw=.8)
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(output/"zero_observability_controls.pdf");fig.savefig(output/"zero_observability_controls.png",dpi=170);plt.close(fig)
    unchanged=all(sha(ROOT/k)==v for k,v in hashes.items());assert unchanged
    maxres=max(r["abs_chi"] for r in exact);maxidentity=max(r["absolute_error"] for r in identity)
    assert maxres<1e-10 and maxidentity<1e-12
    report=dict(status="REGULAR_ZERO_COULOMB_AND_REGULATOR_CONTROLS_PASS",checked_utc=datetime.now(timezone.utc).isoformat(),
        parameters=p.__dict__,settings=dict(mu=.5,T=.04,q=tr.q,Ua=8.,g=gs),
        selected_zero_indices=[r["index"] for r in selected],
        selection="Eight largest modal separations among independently passed low-energy positive zeros",
        zero_count=len(selected),coulomb_cases=len(exact),finite_eta_cases=len(finite),
        maximum_abs_charge_at_regular_zero=maxres,maximum_complex_matrix_identity_error=maxidentity,
        maximum_slope_spread=float(max(np.ptp([r["slope_real"] for r in exact if r["index"]==i]) for i in [r["index"] for r in selected])),
        finite_eta_nonzero_cases=sum(r["abs_chi_at_real_zero"]>1e-14 for r in finite),
        window_edge_cases=sum(r["classification"].startswith("WINDOW_EDGE") for r in finite),
        input_and_source_sha256=hashes,inputs_and_sources_unchanged=unchanged,
        limitations=["Fixed bare Pi and diagonal frequency-independent interactions only.",
            "Regular zeros only; shared numerator/denominator roots require cancellation analysis.",
            "Standard Schur identity and finite-size experiment, not a new theorem.",
            "No resolution of prior827 intervals or continuum low-energy gap is inferred.",
            "Local |chi| minima are not loss minima or a global antiresonance catalogue.",
            "Eta is imposed numerical broadening, not a collision lifetime."])
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_and_source_sha256","limitations","parameters"]},indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);run(ap.parse_args().output)
