"""Independent finite-Hamiltonian/vertex precision audit; frozen inputs stay untouched.

All 216 four-band blocks and all 3456 transitions are retained. Numerical
closeness to a bare transition is association evidence, never a modal identity.
No high-precision 3456-dimensional RPA eigensystem is claimed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import mpmath as mp
import numpy as np
from lattice_response import Parameters, MASS, GAMMA, SPIN_Z, AXIAL
from bloch_response import bloch_spectrum, bloch_transitions

ROOT = Path(__file__).resolve().parent
MODES = ('literal_matrix', 'literal_trig', 'literal_k')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def exact(x):
    a, b = float(x).as_integer_ratio()
    return mp.mpf(a)/b

def mpmatrix(a):
    return mp.matrix([[mp.mpc(exact(z.real), exact(z.imag)) for z in row] for row in np.asarray(a, complex)])

def stable_df(ei, ej, mu, temperature):
    """f(ei)-f(ej), preserving even sub-precision vacancies near f=1."""
    if temperature <= 0: raise ValueError('Positive temperature required')
    x, y = (ei-mu)/temperature, (ej-mu)/temperature
    return mp.exp(x)*mp.expm1(y-x)/((1+mp.exp(x))*(1+mp.exp(y)))

def float_matrices(spec):
    p=spec.parameters; kx,ky,kz=spec.kx,spec.ky,spec.kz
    transverse=((p.mass+p.wilson*(3-np.cos(kx)-np.cos(ky)))[:,None,None]*MASS
        +(p.hopping*np.sin(kx))[:,None,None]*GAMMA[0]
        +(p.hopping*np.sin(ky))[:,None,None]*GAMMA[1]+p.exchange*SPIN_Z)
    longitudinal=(-p.wilson*np.cos(kz)[:,None,None]*MASS
        +p.hopping*np.sin(kz)[:,None,None]*GAMMA[2])
    return transverse[:,None,:,:]+longitudinal[None,:,:,:]

def block_hamiltonian(spec, arrays, block, iz, mode):
    if mode=='literal_matrix': return mpmatrix(arrays[block,iz])
    if mode not in ('literal_trig','literal_k'): raise ValueError('Unknown input meaning')
    p=spec.parameters; k=(spec.kx[block],spec.ky[block],spec.kz[iz])
    sine=[exact(np.sin(x)) if mode=='literal_trig' else mp.sin(exact(x)) for x in k]
    cosine=[exact(np.cos(x)) if mode=='literal_trig' else mp.cos(exact(x)) for x in k]
    h=(exact(p.mass)+exact(p.wilson)*(3-sum(cosine)))*mpmatrix(MASS)
    h+=exact(p.exchange)*mpmatrix(SPIN_Z)
    for i in range(3): h+=exact(p.hopping)*sine[i]*mpmatrix(GAMMA[i])
    return h

def diagonalize(spec, arrays, mode, pairs=None):
    pairs=pairs if pairs is not None else [(b,i) for b in range(len(spec.kx)) for i in range(len(spec.kz))]
    result={}; evidence=[]
    for b,i in pairs:
        h=block_hamiltonian(spec,arrays,b,i,mode)
        e,v=mp.eighe(h)
        residual=mp.norm(h*v-v*mp.diag(e),p=2)
        orthogonal=mp.norm(v.H*v-mp.eye(4),p=2)
        gap=min(e[j+1]-e[j] for j in range(3))
        result[b,i]=(e,v)
        evidence.append(dict(block=b,kz_index=i,energies=[mp.nstr(x,mp.mp.dps) for x in e],
             residual=mp.nstr(residual,10),orthogonality=mp.nstr(orthogonal,10),
             minimum_band_gap=mp.nstr(gap,20),particle_hole_energy_defect=mp.nstr(max(abs(e[j]+e[3-j]) for j in range(4)),10)))
    return result,evidence

def transition_array(spec, states, only=None):
    p=spec.parameters; axial=mpmatrix(AXIAL); rows=[]
    indices=range(p.Nkz*p.Lx*p.Ly*16) if only is None else only
    for index in indices:
        iz,rem=divmod(int(index),p.Lx*p.Ly*16); b,rem=divmod(rem,16); n,m=divmod(rem,4)
        jz=(iz+1)%p.Nkz
        ei,vi=states[b,iz]; ej,vj=states[b,jz]
        c=(vi[:,n].H*vj[:,m])[0]; a=(vi[:,n].H*axial*vj[:,m])[0]
        df=stable_df(ei[n],ej[m],exact(.5),exact(.04))/p.volume
        rows.append(dict(index=int(index),block=b,kz_index=iz,initial_band=n,final_band=m,
            initial_energy=ei[n],final_energy=ej[m],delta=ej[m]-ei[n],df_over_volume=df,
            charge_vertex=c,axial_vertex=a,cc=df*abs(c)**2,aa=df*abs(a)**2,ca=df*c*mp.conj(a)))
    return rows

def plain_row(row):
    out={}
    for key,value in row.items():
        if isinstance(value,int): out[key]=value
        elif isinstance(value,mp.mpc): out[key]=dict(real=mp.nstr(value.real,mp.mp.dps),imag=mp.nstr(value.imag,mp.mp.dps))
        else: out[key]=mp.nstr(value,mp.mp.dps)
    return out

def old_rows(tr):
    return [dict(index=i,delta=exact(tr.delta[i]),cc=exact(tr.cc[i]),aa=exact(tr.aa[i]),
        ca=mp.mpc(exact(tr.ca[i].real),exact(tr.ca[i].imag))) for i in range(len(tr.delta))]

def response(rows,z,vc,ua,select=None):
    picked=rows if select is None else [rows[i] for i in select]
    cc=mp.fsum(r['cc']/(z-r['delta']) for r in picked)
    aa=mp.fsum(r['aa']/(z-r['delta']) for r in picked)
    ca=mp.fsum(r['ca']/(z-r['delta']) for r in picked)
    ac=mp.fsum(mp.conj(r['ca'])/(z-r['delta']) for r in picked)
    p=mp.matrix([[cc,ca],[ac,aa]])
    return p, (mp.eye(2)-p*mp.diag([vc,ua]))**-1*p

def choose_cases(cases):
    """Deterministic sign/log-weight strata; no transition is removed."""
    strata={}
    for row in cases:
        w=float(row['modal_weight']); exponent=math.log10(w)
        bucket=0 if exponent < -45 else 1 if exponent < -32 else 2 if exponent < -20 else 3
        key=('negative' if float(row['modal_endpoint'])<0 else 'positive')+f'_scale{bucket}'
        strata.setdefault(key,[]).append(row)
    chosen={}
    for key,items in sorted(strata.items()):
        items=sorted(items,key=lambda r:(float(r['modal_weight']),r['index']))
        for j in sorted({0,len(items)//2,len(items)-1}):
            chosen[items[j]['index']]=(items[j],key)
    return [dict(**r,selection_stratum=key) for _,(r,key) in sorted(chosen.items())]

def projector_strength(vi,vj,operator,initial,final):
    """Basis-invariant sum over declared degenerate band subspaces."""
    pi=sum((vi[:,n]*vi[:,n].H for n in initial),mp.zeros(vi.rows))
    pj=sum((vj[:,n]*vj[:,n].H for n in final),mp.zeros(vj.rows))
    return mp.re(mp.trace(pi*operator*pj*operator.H)) if hasattr(mp,'trace') else mp.re(sum((pi*operator*pj*operator.H)[i,i] for i in range(vi.rows)))

def weak_residue_estimate(rows,index,vc,ua):
    """Leading isolated-weak-transition dressing, not a certified RPA pole."""
    r=rows[index]; beta=r['delta']; others=[j for j in range(len(rows)) if j!=index]
    separation=min(abs(beta-rows[j]['delta']) for j in others)
    if separation == 0: return dict(status='DEGENERATE_TRANSITION_REQUIRES_GROUPING')
    preg,_=response(rows,beta,vc,ua,others); u=mp.diag([vc,ua]); left=(mp.eye(2)-preg*u)**-1
    w=mp.matrix([[r['cc'],r['ca']],[mp.conj(r['ca']),r['aa']]])
    rw=left*w*left.H
    shift=sum((u*left*w)[j,j] for j in range(2))
    return dict(status='LEADING_WEAK_TRANSITION_ESTIMATE_ONLY',bare=mp.nstr(beta,80),
        nearest_other_bare_distance=mp.nstr(separation,20),pole_shift_estimate=mp.nstr(shift.real,40),
        shift_to_bare_separation=mp.nstr(abs(shift)/separation,15),
        charge_residue_estimate=mp.nstr(rw[0,0].real,40),
        weighted_charge_estimate=mp.nstr(beta*rw[0,0].real,40),
        scope='lambda*R_cc; first order in this raw W, no eigensystem or certified mode matching')

def main(output):
    if output.exists() and any(output.iterdir()): raise RuntimeError('Require new empty output')
    output.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    frozen_paths=[Path(__file__).resolve(),ROOT/'test_band_precision_audit.py',ROOT/'lattice_response.py',
        ROOT/'bloch_response.py',ROOT/'pole_zero_analysis.py',ROOT/'run_pole_zero_experiment.py',
        ROOT/'mode_analysis/diagnostic.json',ROOT/'mode_analysis/finite_poles.csv',
        ROOT/'zero_precision_followup/diagnostic.json',ROOT/'zero_precision_followup/modal_interval_classification.json',
        ROOT/'zero_precision_followup/direct_laurent_cases.json']
    frozen={str(x):sha(x) for x in frozen_paths}
    def log(message):
        line=datetime.now(timezone.utc).isoformat()+' '+message
        print(line,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),source_input_sha256=frozen,
        versions=dict(python=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__),
        arithmetic_precision=80,repeat_precision=110,complete_physical_zero_catalog_certified=False,
        high_precision_full_RPA_eigensystem_computed=False,submission_ready=False)
    dump(output/'diagnostic.json',report)
    for name,expected in json.loads((ROOT/'mode_analysis/diagnostic.json').read_text())['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Changed frozen production source '+name)
    p=Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    spec=bloch_spectrum(p); tr=bloch_transitions(spec,.5,.04,1); arrays=float_matrices(spec)
    check=np.linalg.eigh(arrays)
    if not np.array_equal(check[0],spec.energies) or not np.array_equal(check[1],spec.vectors):
        raise RuntimeError('Reconstructed float matrices do not reproduce original eigensystem byte-exactly')
    cases=[r for r in json.loads((ROOT/'zero_precision_followup/modal_interval_classification.json').read_text())
           if not r['direct_rational_followup_selected']]
    assert len(cases)==751
    selected=choose_cases(cases)
    mp.mp.dps=80
    stages={'original_float':old_rows(tr)}
    float_states={(b,i):(mp.matrix([exact(x) for x in spec.energies[b,i]]),mpmatrix(spec.vectors[b,i]))
        for b in range(len(spec.kx)) for i in range(len(spec.kz))}
    stages['stable_occupation_float_eigenpairs']=transition_array(spec,float_states)
    dump(output/'transitions_stable_occupation_float_eigenpairs.json',[plain_row(r) for r in stages['stable_occupation_float_eigenpairs']])
    report.update(parameters=p.__dict__,block_count=len(spec.kx)*len(spec.kz),raw_transition_count=len(tr.delta),
        selected_cases=len(selected),original_zero_weight_count=int(sum((tr.cc==0)&(tr.aa==0)&(tr.ca==0))),
        float_matrix_eigensystem_reproduced_byte_exact=True,
        input_meanings=dict(original_float='Frozen NumPy eigenpairs/float occupations/weights',
            stable_occupation_float_eigenpairs='Same literal float eigenpairs, stable arbitrary precision occupations and vertex products',
            literal_matrix='Exact binary64 entries assembled by original code; high precision diagonalization',
            literal_trig='Exact binary64 sin/cos values, high precision matrix assembly and diagonalization',
            literal_k='Exact binary64 momentum grid, arbitrary precision sin/cos, assembly and diagonalization'))
    dump(output/'grid.json',dict(parameters=p.__dict__,kx_hex=[x.hex() for x in spec.kx],
        ky_hex=[x.hex() for x in spec.ky],kz_hex=[x.hex() for x in spec.kz],
        scope='Fixed literal float momenta, not a regenerated exact-pi twist grid'))
    log(f'Frozen input gate PASS; {len(cases)} unresolved intervals; {len(selected)} deterministic representatives; 216 blocks')
    evidence={}; states=None
    for mode in MODES:
        log('Start 80-digit full block diagonalization: '+mode)
        states,ev=diagonalize(spec,arrays,mode)
        stages[mode]=transition_array(spec,states); evidence[mode]=ev
        dump(output/f'blocks_{mode}.json',ev)
        dump(output/f'transitions_{mode}.json',[plain_row(r) for r in stages[mode]])
        log('Finished '+mode)
    baseline=stages['original_float']; high=stages['literal_k']
    lost=[i for i,r in enumerate(baseline) if r['cc']==0 and r['aa']==0 and r['ca']==0]
    resurrected=[i for i in lost if high[i]['cc']!=0 or high[i]['aa']!=0 or high[i]['ca']!=0]
    report['thermal_resurrection']=dict(original_zero_count=len(lost),high_precision_nonzero_count=len(resurrected),
        nonzero_stable_occ_same_float_eigenpairs=sum(any(stages['stable_occupation_float_eigenpairs'][i][k]!=0 for k in ('cc','aa','ca')) for i in lost),
        indices=resurrected,max_restored_abs_charge_weight=mp.nstr(max(abs(high[i]['cc']) for i in resurrected),30),
        min_restored_abs_charge_weight=mp.nstr(min(abs(high[i]['cc']) for i in resurrected),30),
        exact_finite_T_zero_inferred=False)
    mappings=[]
    for row in cases:
        z=float(row['modal_endpoint']); order=np.argsort(abs(tr.delta-z)); j=int(order[0]); gap=float(abs(tr.delta[order[1]]-tr.delta[j]))
        assoc_distance=abs(float(tr.delta[j])-z); budget=128*np.finfo(float).eps*max(1.,abs(z))
        other=[int(k) for k in order if abs(tr.delta[k]-tr.delta[j])<=budget]
        mappings.append(dict(index=row['index'],endpoint_index=row['modal_endpoint_index'],modal_endpoint=z,
            modal_weight=row['modal_weight'],nearest_raw_index=j,nearest_bare_distance=assoc_distance,
            near_bare_window=budget,within_roundoff_scale_window=bool(assoc_distance<=budget),
            nearby_raw_indices=other,second_transition_distance=gap,
            occupation_restored=bool(j in resurrected),
            raw_weight_original=mp.nstr(abs(baseline[j]['cc']),30),
            raw_weight_high_precision=mp.nstr(abs(high[j]['cc']),30),
            modal_identity_certified=False))
    dump(output/'all_751_associations.json',mappings)
    byindex={r['index']:r for r in mappings}
    raw_indices=sorted({byindex[r['index']]['nearest_raw_index'] for r in selected})
    pairs=sorted({(high[j]['block'],k) for j in raw_indices for k in (high[j]['kz_index'],(high[j]['kz_index']+1)%p.Nkz)})
    log(f'Repeat at 110 digits: {len(pairs)} blocks, {len(raw_indices)} representative raw transitions')
    with mp.workdps(110):
        repeat_states,repeat_ev=diagonalize(spec,arrays,'literal_k',pairs)
        repeats=transition_array(spec,repeat_states,raw_indices)
        repeat={r['index']:r for r in repeats}
        dump(output/'repeat_110_blocks.json',repeat_ev)
        dump(output/'repeat_110_transitions.json',[plain_row(r) for r in repeats])
        reps=[]
        vc=exact(.8)/exact(tr.Q)**2; ua=exact(8.)
        for case in selected:
            association=byindex[case['index']]; j=association['nearest_raw_index']; fine=repeat[j]; coarse=high[j]
            metric={}
            for key in ('cc','aa','ca','df_over_volume'):
                metric[key+'_relative_80_110']=mp.nstr(abs(coarse[key]-fine[key])/max(abs(fine[key]),mp.mpf('1e-300')),20)
            metric['charge_vertex_abs_80']=mp.nstr(abs(coarse['charge_vertex']),30)
            metric['charge_vertex_abs_110']=mp.nstr(abs(fine['charge_vertex']),30)
            metric['delta_absolute_80_110']=mp.nstr(abs(coarse['delta']-fine['delta']),20)
            relative=max(mp.mpf(metric[k+'_relative_80_110']) for k in ('cc','aa','ca','df_over_volume'))
            classification='ROBUST_NONZERO_RAW_TRANSITION' if fine['cc']!=0 and relative<mp.mpf('1e-50') else 'UNRESOLVED_RAW_VERTEX_PRECISION'
            group=association['nearby_raw_indices']
            groupcc=mp.fsum(abs(high[k]['cc']) for k in group)
            groupaa=mp.fsum(abs(high[k]['aa']) for k in group)
            groupca=mp.fsum(mp.sign(high[k]['delta'])*high[k]['ca'] for k in group)
            determinant=groupcc*groupaa-abs(groupca)**2
            estimate=weak_residue_estimate(high,j,vc,ua)
            reps.append(dict(index=case['index'],selection_stratum=case['selection_stratum'],association=association,
                classification=classification,precision_comparison=metric,
                input_stage_weights={name:mp.nstr(abs(values[j]['cc']),40) for name,values in stages.items()},
                raw_indices_in_near_bare_group=group,positive_group_charge=mp.nstr(groupcc,40),
                positive_group_axial=mp.nstr(groupaa,40),positive_group_determinant=mp.nstr(determinant,40),
                weak_transition_dressing_estimate=estimate,exact_symmetry_zero_proved=False,
                note='Raw positive weight stability is not proof of this CSV modal residue or zero interval.'))
        dump(output/'representative_cases.json',reps)
    # Full finite-eta direct RPA comparisons: no modal basis, no dropped transition.
    log('Compute finite-eta responses and isolated restored-transition contribution')
    vc=exact(.8)/exact(tr.Q)**2; ua=exact(8.); comparisons=[]
    for omega in (.02,.1,.26,.5,.8,1.2):
        for eta in (.015,.0075):
            z=mp.mpc(exact(omega),exact(eta)); calculated={name:response(values,z,vc,ua) for name,values in stages.items()}
            p0,c0=calculated['original_float']; ph,ch=calculated['literal_k']
            restored,_=response(high,z,vc,ua,resurrected)
            changes={}
            chain=list(stages)
            for before,after in zip(chain,chain[1:]):
                pa,ca=calculated[after]; pb,cb=calculated[before]
                changes[before+' -> '+after]=dict(polarization_absolute=mp.nstr(mp.norm(pa-pb,p=2),25),
                    dressed_absolute=mp.nstr(mp.norm(ca-cb,p=2),25),
                    dressed_relative=mp.nstr(mp.norm(ca-cb,p=2)/mp.norm(cb,p=2),25))
            comparisons.append(dict(omega=omega,eta=eta,total_dressed_relative=mp.nstr(mp.norm(ch-c0,p=2)/mp.norm(c0,p=2),25),
                restored_polarization_absolute=mp.nstr(mp.norm(restored,p=2),25),
                restored_polarization_relative=mp.nstr(mp.norm(restored,p=2)/mp.norm(ph,p=2),25),
                original_loss_cc=mp.nstr(-c0[0,0].imag,25),high_precision_loss_cc=mp.nstr(-ch[0,0].imag,25),
                stage_differences=changes))
    dump(output/'finite_eta_response_comparison.json',comparisons)
    report['block_validation']={name:dict(max_eigen_residual=max(float(r['residual']) for r in ev),
        max_orthogonality=max(float(r['orthogonality']) for r in ev),
        minimum_band_gap=min(float(r['minimum_band_gap']) for r in ev)) for name,ev in evidence.items()}
    report['associations']=dict(count=len(mappings),within_roundoff_window=sum(x['within_roundoff_scale_window'] for x in mappings),
        multiple_raw_nearby=sum(len(x['nearby_raw_indices'])>1 for x in mappings),
        modal_identity_certified_count=0,scope='Nearest bare association only; no 751-root rematching')
    report['representatives']=dict(count=len(reps),status_counts=dict(Counter(x['classification'] for x in reps)),
        repeat_blocks=len(pairs),max_relative_80_110=max(float(x['precision_comparison'][k+'_relative_80_110']) for x in reps for k in ('cc','aa','ca','df_over_volume')),
        exact_symmetry_zero_proved_count=0)
    report['response_comparison']=dict(points=len(comparisons),max_dressed_relative=max(float(x['total_dressed_relative']) for x in comparisons),
        max_restored_polarization_relative=max(float(x['restored_polarization_relative']) for x in comparisons),
        eta_is_physical_damping=False)
    report['limitations']=['No high precision full transition-space RPA diagonalization or complete zero catalogue.',
        '80 and 110 digit agreement is numerical evidence, not interval-certified Hamiltonian error bounds.',
        'Only the 751 interval associations are exhaustive; selected raw stability does not resolve those intervals.',
        'Input literal k retains original binary64 grid; exact pi/twist regeneration is a different untested input convention.',
        'Weak-transition dressing is leading-order local evidence and must not be identified with a certified modal residue.',
        'Finite-temperature resurrected weights alter literal rational degree; small finite-eta differences do not certify all real-axis roots.']
    if frozen!={str(x):sha(x) for x in frozen_paths}:raise RuntimeError('Source/input changed during execution')
    report.update(status='BOUNDED_HAMILTONIAN_PRECISION_AUDIT_COMPLETED',completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,frozen_source_inputs_unchanged=True)
    log('Completed; original frozen inputs unchanged; no all-zero-interval certification')
    dump(output/'diagnostic.json',report)
    dump(output/'manifest.json',{x.name:sha(x) for x in output.iterdir() if x.is_file() and x.name!='manifest.json'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); destination=args.output.resolve()
    try:main(destination)
    except Exception as error:
        if destination.exists():
            dump(destination/'failure.json',dict(error_type=type(error).__name__,message=str(error),utc=datetime.now(timezone.utc).isoformat()))
        raise
