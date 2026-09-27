"""Independent matrix/source controls for the declared zero-orbital-field model.

This tests model algebra and a concrete symmetry-allowed calibration ambiguity.
It does not supply a material phonon, electron-phonon matrix element or detector.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from lattice_response import (I2,SX,SY,SZ,AXIAL,MASS,SPIN_Z,GAMMA,
                              Parameters,bloch_hamiltonian)
ROOT=Path(__file__).resolve().parent
P=MASS;T=np.kron(I2,1j*SY);C2X=np.kron(I2,-1j*SX)
A=C2X@T
C4Z=np.kron(I2,(I2-1j*SZ)/np.sqrt(2))
M5=np.kron(SY,I2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def anti(u,m):return u@m.conj()@u.conj().T
def unit(u,m):return u@m@u.conj().T
def err(m):return float(np.max(abs(m)))
def sign_label(m,transformed):
    if err(m-transformed)<1e-14:return 1
    if err(m+transformed)<1e-14:return -1
    return "mixes"
def node(s,h5,hz):
    # Spin+ lower and spin- upper energies cross where R+ + R- = 2b.
    # b=1, t=r=1, kx=ky=m=0. Both spin blocks remain exact 2x2.
    def radius(k,sigma):
        return np.hypot(sigma*(np.sin(k)+hz)+h5,1-np.cos(k))
    k=brentq(lambda x:radius(x,1)+radius(x,-1)-2,
             s*np.pi/3-.1,s*np.pi/3+.1,xtol=5e-15)
    energy=(radius(k,-1)-radius(k,1))/2
    # Verify the analytic spin-block construction against the imported 4x4 H.
    energies=np.linalg.eigvalsh(bloch_hamiltonian([0,0,k])+h5*AXIAL+hz*GAMMA[2])
    return k,float(energy),float(max(abs(energies[1:3]-energy)))
def csv_write(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError("New output required")
    out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__).resolve(),ROOT/"lattice_response.py"]
    hashes={str(p):sha(p) for p in paths}
    checks=[];basis=[]
    for a,pa in zip(("I","x","y","z"),(I2,SX,SY,SZ)):
        for b,pb in zip(("I","x","y","z"),(I2,SX,SY,SZ)):
            m=np.kron(pa,pb)
            row=dict(operator="tau_"+a+"_sigma_"+b,P=sign_label(m,unit(P,m)),
                     T=sign_label(m,anti(T,m)),A_C2xT=sign_label(m,anti(A,m)),
                     C4z=sign_label(m,unit(C4Z,m)))
            row["allowed_P_odd_A_even_C4_even"]=(row["P"]==-1 and row["A_C2xT"]==1 and row["C4z"]==1)
            basis.append(row)
    allowed=[r["operator"] for r in basis if r["allowed_P_odd_A_even_C4_even"]]
    assert allowed==["tau_x_sigma_I","tau_x_sigma_z"]
    points=[(.21,-.32,.43),(-.67,.81,-1.14),(2.3,-2.7,.58),(-2.1,-.11,2.6)]
    for k0 in points:
        k=np.array(k0);h=bloch_hamiltonian(k);ka=k*np.array([-1,1,1])
        m=float(np.sum(1-np.cos(k)))
        values=dict(P_H=err(unit(P,h)-bloch_hamiltonian(-k)),
            T_H_exchange_reversed=err(anti(T,h)-bloch_hamiltonian(-k,Parameters(exchange=-1))),
            A_H_fixed_exchange=err(anti(A,h)-bloch_hamiltonian(ka)),
            torque=err(1j*(h@AXIAL-AXIAL@h)+2*m*M5),
            allowed_pseudoscalar_source=err(anti(A,h+.037*AXIAL)-(bloch_hamiltonian(ka)+.037*AXIAL)),
            allowed_mixed_source=err(anti(A,h+.037*(AXIAL+.7*GAMMA[2]))-(bloch_hamiltonian(ka)+.037*(AXIAL+.7*GAMMA[2]))),
            forbidden_polar_Qz_source=err(anti(A,h+.037*AXIAL)-(bloch_hamiltonian(ka)-.037*AXIAL)),
            false_axial_conservation=err(h@AXIAL-AXIAL@h))
        assert max(v for n,v in values.items() if n not in ("forbidden_polar_Qz_source","false_axial_conservation"))<1e-13
        assert abs(values["forbidden_polar_Qz_source"]-2*.037)<1e-13
        assert values["false_axial_conservation"]>1e-2
        checks.append(dict(k=k.tolist(),**values))
    algebra=dict(M5_kinetic_anticommutator=max(err(M5@g+g@M5) for g in GAMMA),
        M5_mass_anticommutator=err(M5@MASS+MASS@M5),
        M5_exchange_commutator=err(M5@SPIN_Z-SPIN_Z@M5),
        Gamma5_kinetic_commutator=max(err(AXIAL@g-g@AXIAL) for g in GAMMA),
        Gamma5_false_mass_anticommutator=max(err(AXIAL@g+g@AXIAL) for g in GAMMA),
        mixed_source_T_spurion=err(anti(T,AXIAL+.7*GAMMA[2])-(AXIAL-.7*GAMMA[2])),
        Gamma5_Gamma3_trace_inner_product=float(np.trace(AXIAL@GAMMA[2]).real/4))
    assert max(v for n,v in algebra.items() if n!="Gamma5_false_mass_anticommutator")<1e-14
    assert algebra["Gamma5_false_mass_anticommutator"]==2.
    projectors=[];shifts=[]
    for s in (-1,1):
        k=[0,0,s*np.pi/3];e,u=np.linalg.eigh(bloch_hamiltonian(k));v=u[:,np.argsort(abs(e))[:2]]
        pg5=v.conj().T@AXIAL@v;pg3=v.conj().T@GAMMA[2]@v
        expected=-s*np.sqrt(3)/2
        projectors.append(dict(node_sign=s,charge_projection_error=err(v.conj().T@v-np.eye(2)),
            axial_scalar_error=err(pg5-expected*np.eye(2)),
            axial_projected_eigenvalues=np.linalg.eigvalsh(pg5).tolist(),
            Gamma3_projected_eigenvalues=np.linalg.eigvalsh(pg3).tolist(),
            Gamma3_projected_trace=float(np.trace(pg3).real)))
        assert err(pg5-expected*np.eye(2))<1e-14
        assert abs(np.trace(pg3))<1e-14
        for g5,gz,label in ((1.,0.,"Gamma5"),(0.,1.,"Gamma3"),(1.,.7,"Gamma5_plus_0p7_Gamma3")):
            for step in (1e-3,1e-4,1e-5):
                kp,ep,rp=node(s,g5*step,gz*step);km,em,rm=node(s,-g5*step,-gz*step)
                dk=(kp-km)/(2*step);de=(ep-em)/(2*step)
                row=dict(node_sign=s,source=label,step=step,energy_slope=de,
                    expected_energy_slope=expected*g5,momentum_slope=dk,expected_momentum_slope=-gz,
                    energy_slope_error=abs(de-expected*g5),momentum_slope_error=abs(dk+gz),
                    maximum_4x4_crossing_residual=max(rp,rm))
                assert max(row["energy_slope_error"],row["momentum_slope_error"])<2e-6
                assert row["maximum_4x4_crossing_residual"]<1e-13
                shifts.append(row)
    csv_write(out/"pauli_symmetries.csv",basis);csv_write(out/"node_source_shifts.csv",shifts)
    assert all(sha(Path(p))==v for p,v in hashes.items())
    result=dict(status="MODEL_SOURCE_SYMMETRY_CONTROLS_PASS",source_sha256=hashes,
        input_sources_unchanged=True,allowed_fixed_b_local_basis=allowed,
        point_checks=checks,operator_algebra=algebra,node_projectors=projectors,
        max_smallest_step_energy_slope_error=max(r["energy_slope_error"] for r in shifts if r["step"]==1e-5),
        max_smallest_step_momentum_slope_error=max(r["momentum_slope_error"] for r in shifts if r["step"]==1e-5),
        physical_conclusion="Gamma5 and b.Gamma both permit a pseudoscalar source in this ideal representation; node energy splitting alone does not calibrate its full-band vertex",
        limits=["No real crystal/mode/optical coupling supplied","Bloch Hamiltonian at zero orbital field, fixed exchange b_z=1","Numerical checks support the Pauli algebra; not proof of arbitrary material symmetry","Finite source derivative convergence only at the stated steps"])
    (out/"diagnostic.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    main(ap.parse_args().output.resolve())
