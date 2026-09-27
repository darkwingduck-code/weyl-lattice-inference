"""Selected local RPA poles from 110-digit band inputs, with all thermal slots.

A rank-one Schur complement avoids adding a sub-ulp pole shift to its bare
center while solving. This is a numerical local continuation, not an interval
certificate for all poles/zeros or an assertion of continuum collectivity.
"""
from __future__ import annotations
import argparse,json,sys,time,unittest
from pathlib import Path
from datetime import datetime,timezone
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from band_precision_audit import (ROOT,exact,sha,dump,plain_row,diagonalize,
    transition_array,float_matrices)
from bloch_response import bloch_spectrum
from lattice_response import Parameters


def trace(a):return sum(a[j,j] for j in range(a.rows))
def weight(r):return mp.matrix([[r['cc'],r['ca']],[mp.conj(r['ca']),r['aa']]])

def polarization_offset(rows,center,t,omit=None,derivative=False):
    picked=[(j,r) for j,r in enumerate(rows) if j!=omit]
    d=[center-r['delta']+t for _,r in picked]
    if any(x==0 for x in d):raise ValueError('Exact bare transition in local evaluation')
    factors=[-1/x**2 if derivative else 1/x for x in d]
    def entry(name,conjugate=False):
        return mp.fsum((mp.conj(r[name]) if conjugate else r[name])*f for (_,r),f in zip(picked,factors))
    return mp.matrix([[entry('cc'),entry('ca')],[entry('ca',True),entry('aa')]])

def local_values(rows,index,t,u):
    center=rows[index]['delta']; w=weight(rows[index])
    p=polarization_offset(rows,center,t,omit=index)
    dp=polarization_offset(rows,center,t,omit=index,derivative=True)
    a=mp.eye(2)-p*u; left=a**-1
    shift=trace(u*left*w)
    derivative=1-trace(u*left*dp*u*left*w)
    return t-shift,derivative,left*w*left.H,mp.norm(a,p=2)*mp.norm(left,p=2)

def refine(rows,index,u):
    center=rows[index]['delta']
    separation=min(abs(center-r['delta']) for j,r in enumerate(rows) if j!=index)
    if separation==0:raise ValueError('Exact degenerate bare cluster needs summed-rank treatment')
    w=weight(rows[index]); scale=mp.norm(w,p=2)
    if scale==0:raise ValueError('Zero weight cannot define this local branch')
    defect=abs(mp.det(w))/scale**2
    if defect>mp.mpf('1e-100'):raise ValueError('Rank-one coherent-vertex prerequisite failed')
    value,_,_,_=local_values(rows,index,mp.mpf(0),u)
    t=-value
    if abs(t)>=separation/4 or t==0:raise ValueError('First shift not inside the bounded local component')
    converged=False
    for iteration in range(12):
        f,df,_,_=local_values(rows,index,t,u)
        if df==0:raise ValueError('Non-simple local root')
        dt=f/df; candidate=t-dt
        if abs(candidate)>=separation/4 or candidate==0:raise ValueError('Newton leaves protected component')
        t=candidate
        if abs(dt)<=abs(t)*mp.mpf('1e-90'):
            converged=True;break
    if not converged:raise ValueError('Local refinement did not converge')
    f,df,numerator,condition=local_values(rows,index,t,u)
    residue=numerator/df
    bracket=[t-abs(t)*mp.mpf('1e-25'),t+abs(t)*mp.mpf('1e-25')]
    fa=local_values(rows,index,bracket[0],u)[0]
    fb=local_values(rows,index,bracket[1],u)[0]
    sign_change=bool(fa.real<0<fb.real or fb.real<0<fa.real)
    checks=[]
    for factor in ('0.001','0.00001','0.0000001'):
        epsilon=abs(t)*mp.mpf(factor)
        p=polarization_offset(rows,center,t+1j*epsilon)
        dressed=(mp.eye(2)-p*u)**-1*p
        scaled=1j*epsilon*dressed
        checks.append(dict(epsilon_to_shift=factor,absolute_epsilon=mp.nstr(epsilon,40),
            scaled_matrix_relative_error=mp.nstr(mp.norm(scaled-residue,p=2)/mp.norm(residue,p=2),25)))
    return dict(status='HIGH_PRECISION_LOCAL_NUMERICAL_POLE',raw_index=index,
        center=mp.nstr(center,110),offset=mp.nstr(t.real,110),pole=mp.nstr((center+t).real,110),
        nearest_other_bare_distance=mp.nstr(separation,40),offset_to_separation=mp.nstr(abs(t)/separation,25),
        rank_one_relative_determinant=mp.nstr(defect,10),iterations=iteration+1,
        normalized_schur_residual=mp.nstr(abs(f)/abs(t),25),
        regular_background_condition=mp.nstr(condition,25),denominator_derivative=mp.nstr(df.real,40),
        residue_cc=mp.nstr(residue[0,0].real,70),residue_aa=mp.nstr(residue[1,1].real,70),
        weighted_charge=mp.nstr(((center+t)*residue[0,0]).real,70),
        local_sign_change=sign_change,offset_sign_bracket=[mp.nstr(x,110) for x in bracket],
        independent_full_transition_residue_checks=checks,
        rigorous_interval_certificate=False,continuum_collective_mode_certified=False)

class LocalPoleControls(unittest.TestCase):
    def test_exact_lorentz_pole_and_residue(self):
        with mp.workdps(110):
            e=mp.mpf('.6');a=mp.mpf('.00000001');u=mp.diag([mp.mpf('.8'),mp.mpf(8)])
            rows=[dict(delta=e,cc=a,aa=mp.mpf(0),ca=mp.mpc(0)),
                  dict(delta=-e,cc=-a,aa=mp.mpf(0),ca=mp.mpc(0))]
            result=refine(rows,0,u); expected=mp.sqrt(e*e+2*e*a*u[0,0])
            self.assertLess(abs(mp.mpf(result['pole'])-expected),mp.mpf('1e-100'))
            self.assertLess(abs(mp.mpf(result['residue_cc'])-e*a/expected),mp.mpf('1e-70'))
            self.assertTrue(result['local_sign_change'])
    def test_degenerate_transition_not_silently_treated_rank_one(self):
        with mp.workdps(110):
            rows=[dict(delta=mp.mpf(1),cc=mp.mpf('.01'),aa=mp.mpf(0),ca=mp.mpc(0))]*2
            with self.assertRaisesRegex(ValueError,'degenerate'):refine(rows,0,mp.eye(2))
    def test_fullrank_target_is_rejected(self):
        with mp.workdps(110):
            rows=[dict(delta=mp.mpf(1),cc=mp.mpf('.01'),aa=mp.mpf('.01'),ca=mp.mpc(0)),
                  dict(delta=mp.mpf(-1),cc=mp.mpf('-.01'),aa=mp.mpf('-.01'),ca=mp.mpc(0))]
            with self.assertRaisesRegex(ValueError,'Rank-one'):refine(rows,0,mp.eye(2))

def main(output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('New empty directory required')
    output.mkdir(parents=True,exist_ok=True); start=time.perf_counter();mp.mp.dps=110
    sources=[Path(__file__).resolve(),ROOT/'band_precision_audit.py',ROOT/'test_band_precision_audit.py',
        ROOT/'lattice_response.py',ROOT/'bloch_response.py',ROOT/'band_precision_audit/run_01/manifest.json',
        ROOT/'band_precision_audit/run_01/representative_cases.json']
    frozen={str(p):sha(p) for p in sources}
    report=dict(status='RUNNING',source_input_sha256=frozen,precision=110,
        started_utc=datetime.now(timezone.utc).isoformat(),all_751_intervals_resolved=False)
    dump(output/'diagnostic.json',report)
    p=Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    spec=bloch_spectrum(p);states,blocks=diagonalize(spec,float_matrices(spec),'literal_k')
    rows=transition_array(spec,states)
    dump(output/'all_216_blocks_110.json',blocks)
    dump(output/'all_3456_transitions_110.json',[plain_row(r) for r in rows])
    # Coulomb Q is the same literal binary64 finite-q value used in the original experiment.
    import numpy as np
    Q=2*np.sin((2*np.pi/24)/2);u=mp.diag([exact(.8)/exact(Q)**2,exact(8)])
    representatives=json.loads((ROOT/'band_precision_audit/run_01/representative_cases.json').read_text())
    results=[]
    for representative in representatives:
        idx=representative['association']['nearest_raw_index']
        result=refine(rows,idx,u)
        old=mp.mpf(representative['association']['modal_weight'])
        result.update(interval_index=representative['index'],original_modal_weight=mp.nstr(old,80),
            original_modal_to_refined_weight_ratio=mp.nstr(old/mp.mpf(result['weighted_charge']),30),
            old_to_new_pole_distance=mp.nstr(abs(exact(representative['association']['modal_endpoint'])-mp.mpf(result['pole'])),30),
            certified_one_to_one_old_catalog_mapping=False)
        results.append(result)
        dump(output/'selected_local_poles.json',results)
        print(f"interval {representative['index']}: local pole and residue calculated",flush=True)
    # Evaluate the isolated effect of restoring the 802 slots in the same 110-digit response.
    initial=json.loads((ROOT/'band_precision_audit/run_01/diagnostic.json').read_text())
    restored=set(initial['thermal_resurrection']['indices']); reduced=[r for i,r in enumerate(rows) if i not in restored]
    impact=[]
    for omega in (.02,.1,.26,.5,.8,1.2):
        for eta in (.015,.0075):
            z=mp.mpc(exact(omega),exact(eta))
            ph=polarization_offset(rows,mp.mpf(0),z)
            pl=polarization_offset(reduced,mp.mpf(0),z)
            ch=(mp.eye(2)-ph*u)**-1*ph;cl=(mp.eye(2)-pl*u)**-1*pl
            impact.append(dict(omega=omega,eta=eta,dressed_relative=mp.nstr(mp.norm(ch-cl,p=2)/mp.norm(ch,p=2),30),
                polarization_relative=mp.nstr(mp.norm(ph-pl,p=2)/mp.norm(ph,p=2),30)))
    dump(output/'isolated_802_restoration_response.json',impact)
    if frozen!={str(p):sha(p) for p in sources}:raise RuntimeError('Changed source/input during run')
    report.update(status='SELECTED_LOCAL_POLE_NUMERICAL_REFINEMENT_COMPLETED',elapsed_seconds=time.perf_counter()-start,
        selected_count=len(results),all_751_intervals_resolved=False,source_inputs_unchanged=True,
        max_normalized_schur_residual=max(float(x['normalized_schur_residual']) for x in results),
        max_last_residue_check_relative=max(float(x['independent_full_transition_residue_checks'][-1]['scaled_matrix_relative_error']) for x in results),
        max_isolated_802_dressed_relative=max(float(x['dressed_relative']) for x in impact),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='All 110-digit band/vertex/thermal inputs; 23 local rank-one Schur poles with full remaining background. Numerical evidence, not outward-rounded theorem.')
    dump(output/'diagnostic.json',report)
    dump(output/'manifest.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalPoleControls))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.output is None:parser.error('--output or --self-test is required')
    target=args.output.resolve()
    try:main(target)
    except Exception as error:
        if target.exists():dump(target/'failure.json',dict(error_type=type(error).__name__,message=str(error)))
        raise
