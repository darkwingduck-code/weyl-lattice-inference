"""Final reproducible delivery checks and journal for the bounded band audit."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import mpmath as mp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from band_precision_audit import ROOT,sha,dump,exact,old_rows
from local_pole_refinement import local_values,polarization_offset
from bloch_response import bloch_spectrum,bloch_transitions
from lattice_response import Parameters,occupations

BASE=Path(__file__).resolve().parent

def load_rows(path):
    rows=[]
    for raw in json.loads(path.read_text()):
        row={}
        for key,value in raw.items():
            if isinstance(value,int):row[key]=value
            elif isinstance(value,dict):row[key]=mp.mpc(value['real'],value['imag'])
            else:row[key]=mp.mpf(value)
        rows.append(row)
    return rows

def moment(rows):
    a=mp.fsum(r['cc'] for r in rows);b=mp.fsum(r['aa'] for r in rows);c=mp.fsum(r['ca'] for r in rows)
    return mp.norm(mp.matrix([[a,c],[mp.conj(c),b]]),p=2)

def main():
    output=BASE/'delivery_review'
    if output.exists() and any(output.iterdir()):raise RuntimeError('New empty delivery_review required')
    output.mkdir(parents=True,exist_ok=True);mp.mp.dps=110
    frozen={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    for directory in (BASE/'run_01',BASE/'local_poles_110'):
        for name,value in json.loads((directory/'manifest.json').read_text()).items():
            if sha(directory/name)!=value:raise RuntimeError('Changed manifest file '+name)
            frozen[str(directory/name)]=value
        diagnostic=json.loads((directory/'diagnostic.json').read_text())
        for name,value in diagnostic['source_input_sha256'].items():
            if sha(Path(name))!=value:raise RuntimeError('Changed input '+name)
            frozen[name]=value
    rows=load_rows(BASE/'local_poles_110/all_3456_transitions_110.json')
    results=json.loads((BASE/'local_poles_110/selected_local_poles.json').read_text())
    p=Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    spec=bloch_spectrum(p);tr=bloch_transitions(spec,.5,.04,1);raw=old_rows(tr)
    f=occupations(spec,.5,.04)
    saturated=0;bothone=0
    for i in range(p.Nkz):
        df=f[:,i,:,None]-f[:,(i+1)%p.Nkz,None,:]
        saturated+=int(np.sum(df==0))
        bothone+=int(np.sum((f[:,i,:,None]==1)&(f[:,(i+1)%p.Nkz,None,:]==1)))
    Q=2*np.sin((2*np.pi/24)/2);u=mp.diag([exact(.8)/exact(Q)**2,exact(8)])
    checks=[]
    for result in results:
        index=result['raw_index'];center=mp.mpf(result['center']);t=mp.mpf(result['offset'])
        fv,derivative,numerator,condition=local_values(rows,index,t,u);residue=numerator/derivative
        if center!=rows[index]['delta']:raise RuntimeError('Center does not map to exact stored raw index')
        if mp.mpf(result['weighted_charge'])<=0:raise RuntimeError('Wrong weighted residue sign')
        item=dict(interval_index=result['interval_index'],real_offset_schur_residual_relative=mp.nstr(abs(fv)/abs(t),25),
            real_offset_schur_imag_relative=mp.nstr(abs(fv.imag)/abs(t),25),
            imaginary_shift_at_real_offset=mp.nstr(abs(fv.imag),25),
            imaginary_newton_correction_relative=mp.nstr(abs((fv/derivative).imag)/abs(t),25),
            serialized_residue_diagonal_relative=mp.nstr(max(abs(residue[k,k].real-mp.mpf(result['residue_cc' if k==0 else 'residue_aa']))/max(abs(residue[k,k]),mp.mpf('1e-300')) for k in range(2)),25),
            checks=[])
        for factor in ('0.000000001','0.00000000001'):
            epsilon=abs(t)*mp.mpf(factor)
            full=polarization_offset(rows,center,t+1j*epsilon)
            dressed=(mp.eye(2)-full*u)**-1*full
            error=mp.norm(1j*epsilon*dressed-residue,p=2)/mp.norm(residue,p=2)
            item['checks'].append(dict(epsilon_to_shift=factor,relative_error=mp.nstr(error,25)))
        checks.append(item)
    dump(output/'closer_residue_checks.json',checks)
    delta=sorted(r['delta'] for r in rows)
    magnitudes=[abs(r['cc']) for r in rows]
    stats=dict(raw_slot_count=len(rows),recorded_delta_zero_count=sum(x==0 for x in delta),
        distinct_recorded_delta_count=len(set(delta)),minimum_recorded_delta_separation=mp.nstr(min(b-a for a,b in zip(delta,delta[1:])),30),
        all_charge_weights_nonzero=all(x!=0 for x in magnitudes),min_abs_charge_weight=mp.nstr(min(magnitudes),30),
        minimum_squared_charge_vertex=mp.nstr(min(abs(r['charge_vertex'])**2 for r in rows),30),
        original_float_df_zero_count=saturated,original_both_occupations_exactly_one=bothone,
        observable_pole_count_certified=False,
        moment0_frobenius=dict(original_float=mp.nstr(moment(raw),30),high_precision_110=mp.nstr(moment(rows),30)),
        all_751_intervals_resolved=False)
    nonzero=[i for i,r in enumerate(raw) if r['cc']!=0]
    errors=[abs(raw[i]['cc']/rows[i]['cc']-1) for i in nonzero]
    stats['old_nonzero_raw_charge_relative_errors']=dict(count=len(errors),max=mp.nstr(max(errors),25),
        above_1percent=sum(x>mp.mpf('.01') for x in errors),above_1e_6=sum(x>mp.mpf('1e-6') for x in errors),
        note='Reporting bins only; no weight has been deleted or replaced.')
    dump(output/'input_slot_statistics.json',stats)
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    env['TEMP']=env['TMP']='H:/Research/.paper-runtime'
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):env[key]='1'
    commands=[('band_negative_controls',[sys.executable,'-m','unittest','discover','-s','research/prb_lattice_response','-p','test_band_precision_audit.py','-v']),
              ('local_pole_negative_controls',[sys.executable,str(BASE/'local_pole_refinement.py'),'--self-test'])]
    test_reports=[]
    for name,command in commands:
        run=subprocess.run(command,cwd=ROOT.parents[1],env=env,capture_output=True,text=True)
        path=output/(name+'.txt');path.write_text(run.stdout+run.stderr,encoding='utf-8')
        test_reports.append(dict(name=name,command=command,returncode=run.returncode,log_sha256=sha(path)))
        if run.returncode:raise RuntimeError('Negative control failed: '+name)
    report=dict(status='DELIVERY_CHECKS_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),source_input_sha256=frozen,
        tests=test_reports,test_count=9,max_closer_residue_relative=max(float(x['checks'][-1]['relative_error']) for x in checks),
        max_real_offset_schur_residual_relative=max(float(x['real_offset_schur_residual_relative']) for x in checks),
        max_real_offset_schur_imag_relative=max(float(x['real_offset_schur_imag_relative']) for x in checks),
        max_imaginary_newton_correction_relative=max(float(x['imaginary_newton_correction_relative']) for x in checks),
        source_inputs_unchanged=False)
    if frozen!={name:sha(Path(name)) for name in frozen}:raise RuntimeError('Inputs changed during delivery review')
    report['source_inputs_unchanged']=True
    dump(output/'diagnostic.json',report)
    dump(output/'manifest.json',{path.name:sha(path) for path in output.iterdir() if path.is_file() and path.name!='manifest.json'})
    # Human-readable report appended only after successful scientific/evidence checks.
    initial=json.loads((BASE/'run_01/diagnostic.json').read_text())
    local=json.loads((BASE/'local_poles_110/diagnostic.json').read_text())
    reps=json.loads((BASE/'run_01/representative_cases.json').read_text())
    lines=['','## Completed run 01: actual Hamiltonian and thermal-input audit',
        '',f"Started {initial['started_utc']}; completed {initial['completed_utc']}; wall time {initial['elapsed_seconds']:.3f} seconds.",
        'The 216-block, three-input-convention, 80-digit calculation completed without changing any of its 11 source/input files. The source-start/end hashes are in run_01/diagnostic.json; its manifest binds each output byte.',
        '',f"The predeclared stratification selected {len(reps)} representatives. Their 39 distinct endpoint blocks were repeated at 110 digits. All 23 raw transitions remained nonzero. The largest relative 80-to-110 difference among cc, aa, ca and df/V was {initial['representatives']['max_relative_80_110']:.6e}. No symmetry-exact raw zero was established or substituted.",
        '',f"Every one of the 802 originally zero input slots is nonzero with stable high-precision occupations. All 802 already revive with the SAME literal float eigenpairs; therefore increasing only the eigensolver precision cannot repair discarded occupation differences. Direct inspection confirms {bothone} cases in which both original occupations were stored as exactly 1. The stable formula retains their unequal vacancy tails. Restored |Wcc| ranges from {initial['thermal_resurrection']['min_restored_abs_charge_weight']} to {initial['thermal_resurrection']['max_restored_abs_charge_weight']}.",
        '',f"All 3,456 recorded high-precision slots have nonzero delta and charge weight. The smallest recorded delta separation is {stats['minimum_recorded_delta_separation']}; the smallest squared charge vertex is {stats['minimum_squared_charge_vertex']}. These are numerical properties of this particular table. They are NOT a count of distinct visible dressed poles. Exact spectral degeneracy, coupling rank, residue cancellation and continuum interpretation remain separate questions.",
        '',f"Of the 751 unresolved-interval associations, {initial['associations']['within_roundoff_window']} nearest bare frequencies lie inside the declared numerical proximity window; none has a second raw transition within that window. No association is a certified one-to-one old/new RPA mode match. The original 751 unresolved status remains unchanged.",
        '',f"At the 12 predeclared finite-eta points, the largest full dressed matrix relative difference from original float to literal-k/80-digit input was {initial['response_comparison']['max_dressed_relative']:.6e}. The largest restored-slot polarization relative norm was {initial['response_comparison']['max_restored_polarization_relative']:.6e}. The separate 110-digit calculation with ONLY the restored 802 slots toggled gave a maximum dressed matrix relative change of {local['max_isolated_802_dressed_relative']:.6e}. Tiny response changes on these finite-eta points do not preserve a complete literal real-axis pole/zero catalogue by themselves.",
        '',f"The equal-time M0 matrix Frobenius norm changes from {stats['moment0_frobenius']['original_float']} to {stats['moment0_frobenius']['high_precision_110']}. Among the 2,654 originally nonzero charge weights, {stats['old_nonzero_raw_charge_relative_errors']['above_1percent']} differ by more than 1 percent and {stats['old_nonzero_raw_charge_relative_errors']['above_1e_6']} by more than 1e-6 after the full high-precision input calculation. These bins diagnose loss of relative accuracy near occupation saturation; they are not deletion thresholds.",
        '', '## Exact local rank-one identity and selected pole continuation',
        '', 'A second immutable run recalculated ALL 216 literal-k blocks and ALL 3,456 transitions at 110 digits. For the selected target beta with coherent rank-one W, write Pi(z)=P_reg(z)+W/t, t=z-beta; U=diag(Vc,Ua); L(z)=(I-P_reg(z)U)^(-1). The exact local matrix identity is',
        '', '    chi(z) = chi_reg(z) + L(z) W L(z)^dagger / [t - k(z)]  (real z)',
        '    k(z) = tr[U L(z) W]',
        '    k_prime(z) = tr[U L(z) P_reg_prime(z) U L(z) W]',
        '    R = L(lambda) W L(lambda)^dagger / [1 - k_prime(lambda)].',
        '', 'At complex z the right factor is (I-U P_reg(z))^(-1), not the Hermitian adjoint; the residue formula above is evaluated at the real pole. The independent complex-frequency check uses the full original matrix inversion and therefore does not use that real-axis shortcut.',
        '', 'The local code excludes exact bare degeneracy and a full-rank target. It preserves the full regular background, uses the analytic derivative, and restricts |t| below one quarter of the nearest other bare separation. The source and tests preserve explicit rejection controls. Offsets are solved as separate arbitrary-precision numbers, avoiding premature addition to beta. Numerical two-sided signs are recorded, but no interval-arithmetic root theorem is claimed.',
        '',f"The 23 selected local numerical poles completed in {local['elapsed_seconds']:.3f} seconds. Their direct full-transition i*epsilon*chi comparison converges as epsilon/|t| decreases from 1e-3 to 1e-5 to 1e-7. The final independent delivery calculation extends this to 1e-9 and 1e-11, with maximum matrix-relative residue difference {report['max_closer_residue_relative']:.6e} at 1e-11. The source/input hash checks passed after both runs.",
        '', 'The following table compares an old rounded-modal endpoint with the selected nearby high-precision local pole. A ratio far from one diagnoses that the old tiny residue is not quantitatively reliable for this nearby continuation. Frequency proximity and residue comparison are evidence, not a globally certified mode matching. Positive and negative poles use the nonnegative weighted charge value lambda*Rcc.',
        '', '| Old interval | Raw index | Original abs(Wcc) | 110-digit lambda*Rcc | Old modal / refined weight | Shift / next bare gap |',
        '|---:|---:|---:|---:|---:|---:|']
    rep_by={x['index']:x for x in reps}
    for result in results:
        rep=rep_by[result['interval_index']]
        lines.append(f"| {result['interval_index']} | {result['raw_index']} | {float(rep['input_stage_weights']['original_float']):.6e} | {float(result['weighted_charge']):.6e} | {float(result['original_modal_to_refined_weight_ratio']):.6e} | {float(result['offset_to_separation']):.3e} |")
    lines += ['', 'The largest mismatch is approximately 1.13e13 for interval 1483: its charge vertex magnitude is approximately 0.999857, so its small raw weight comes from thermal occupation rather than a symmetry-forbidden charge matrix element. At interval 1293, the old nonzero raw occupation subtraction already changes the amplitude by a factor approximately 1.916; this is distinct from modal realization leakage. These are quantified local examples, not a declaration that every weak mode is artificial.',
        '', '## Attempts, verification and exact reproduction',
        '', '- Preflight attempt 1 failed only on the mpmath norm API spelling; the original error and correction are recorded above. No production evidence was overwritten.',
        '- The corrected band negative controls passed 6/6. The separate local-pole controls passed 3/3: exact Lorentz pole/residue, rejection of a degenerate bare target, and rejection of a full-rank target.',
        '- run_01 completed. local_poles_110 completed. delivery_review re-read both manifests and their recorded source/input hashes, performed closer full-response residue comparisons, recorded input-slot statistics, and captured both test command logs. All source/input comparisons passed.',
        '', 'Additional exact commands from the repository root:',
        '', '    python research/prb_lattice_response/band_precision_audit/local_pole_refinement.py --self-test',
        '    python research/prb_lattice_response/band_precision_audit/local_pole_refinement.py --output research/prb_lattice_response/band_precision_audit/local_poles_110',
        '    python research/prb_lattice_response/band_precision_audit/finalize_review.py',
        '', 'All output directories are append-only evidence: for a new reproduction use a fresh copy of the audit folder or rename the new output destinations in a preserved new runner; do not overwrite these outputs. finalize_review.py intentionally refuses a nonempty delivery_review and appends this journal only after all checks finish.',
        '', '## What this continuation establishes and what remains',
        '', 'It establishes actual Hamiltonian/vertex/occupation precision stability for 23 explicitly selected raw transitions, complete finite-input thermal restoration on this grid, separate finite-eta response impact, and local full-background pole/residue continuation for those 23 representatives. It does not establish exact symmetry zeros, a new count of observable poles, a complete rematching of 751 intervals, continuum poles, damping, a hybrid gap, novelty or submission readiness. High-precision arithmetic residuals are not error bars on a material model or an interval proof.',
        '', 'The two distinct remaining tasks are (a) stable full-input real-axis catalogue construction, with exact/near-degenerate spectral groups and rank examined before any pole count; and (b) physical low-energy/continuum and UV-matching research. A finite-eta curve alone cannot replace either task.',
        '', 'This journal records reproducible hypotheses, selections, computations, failures and evidence; it does not record private internal deliberation.']
    journal=ROOT/'detailed_band_precision_journal.md'
    with journal.open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_input_sha256'},indent=2))

if __name__=='__main__':main()
