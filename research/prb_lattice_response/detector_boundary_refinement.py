"""Target the three detector-threshold flips without altering frozen evidence.

Sample higher meshes through immutable all-band helpers; independently reprofile
coarse/final/higher g, with own-mesh truth and (separately) shared higher truth.
Deterministic RMS thresholds are not confidence intervals or convergence bounds.
"""
from __future__ import annotations
import argparse, csv, gc, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import parameter_sector_sampling as sampler
from lattice_response import Parameters
from magnetic_bloch_response import iter_magnetic_bloch_spectra
from detector_identifiability_audit import numerical_profile
from parameter_calibration_independent_review import (Snapshot, load_windows,
    cofactor_response, residual, models_for, truths_for, extract_matrix, read_csv)
from extend_transverse import memory_status, sha, csv_write
from run_pole_zero_experiment import write_json

ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'detector_boundary_refinement'
INDEX=ROOT/'parameter_sector_sampling/calibration_inputs.json'
TARGETS=(
 dict(id='Bplus_eta_low_Ua1',window='fixed_lab',names=('Bplus_eta_low',),ua=1.,threshold=.01),
 dict(id='B0_qscan_Ua20',window='fixed_lab',names=('B0_qhalf','B0_baseline','B0_qdouble'),ua=20.,threshold=.001),
 dict(id='B0_qdouble_scaled_Ua17',window='scaled_q',names=('B0_qdouble',),ua=17.,threshold=.001))
SOURCE_NAMES=tuple(dict.fromkeys((*sampler.SOURCES,'detector_boundary_refinement.py',
 'test_detector_boundary_refinement.py','detector_identifiability_audit.py',
 'parameter_calibration_independent_review.py')))


def snapshot_sources():
    return {str(ROOT/n):sha(ROOT/n) for n in SOURCE_NAMES}


def preserve_sources(hashes):
    bad=[p for p,h in hashes.items() if sha(Path(p))!=h]
    if bad:raise RuntimeError('Frozen sources changed: '+repr(bad))


def new_output(path):
    path=Path(path).resolve()
    if path.exists() and any(path.iterdir()):raise FileExistsError('New empty output required: '+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path


def mesh_spec(field):
    if field=='B0':return Parameters(Lx=96,Ly=96,Nkz=768),0,1,32,('qhalf','baseline','qdouble')
    if field=='Bplus':return Parameters(Lx=72,Ly=13,Nkz=384),1,36,1,('eta_low',)
    raise ValueError('Unknown field')


def accumulate(p,numerator,denominator,chunk,specs,*,progress=None,pilot_chunks=0,admission=None):
    """Only bare responses are added across complete magnetic momentum blocks."""
    results={name:sampler._empty_result(s) for name,s in specs.items()}
    completed=0;total_blocks=p.Lx//denominator*p.Ly
    for ib,spectrum in enumerate(iter_magnetic_bloch_spectra(p,numerator,denominator,blocks_per_chunk=chunk)):
        if admission is not None and memory_status()['available_physical_bytes']<admission:
            raise MemoryError('Runtime block memory admission failed')
        for shift in sorted({s['qshift'] for s in specs.values()}):
            vertices=sampler.transition_vertices(spectrum,shift)
            selected={n:s for n,s in specs.items() if s['qshift']==shift}
            for mu in sorted({s['mu'] for s in selected.values()}):
                tr,stats=sampler.weighted_transitions(spectrum,vertices,mu,.04)
                for name,s in selected.items():
                    if s['mu']!=mu:continue
                    results[name]['bare']+=sampler.two_sector_response(tr,s['omega'],s['eta'])
                    sampler._add_stats(results[name],tr,stats,s)
                del tr
            del vertices
        completed+=len(spectrum.block_indices)
        del spectrum;gc.collect()
        if progress:progress(ib+1,completed,total_blocks)
        if pilot_chunks and ib+1>=pilot_chunks:break
    return results,completed


def sample(output,field,pilot_chunks=0):
    output=new_output(output);start=time.perf_counter()
    p,num,den,chunk,names=mesh_spec(field)
    base,_,_=sampler.read_sectors(ROOT/sampler.INPUTS[0])
    specs=sampler.case_specs(base,p.Nkz,names)
    hashes=snapshot_sources()
    input_hashes={str(INDEX):sha(INDEX),str(ROOT/sampler.INPUTS[0]):sha(ROOT/sampler.INPUTS[0])}
    npairs=p.Nkz*chunk*(4*den)**2
    estimate=192*npairs+384*1024**2;admission=estimate+1024**3
    report=dict(status='RUNNING',invocation=sys.argv,started_utc=datetime.now(timezone.utc).isoformat(),
        parameters=p.__dict__,field=field,numerator=num,denominator=den,blocks_per_chunk=chunk,
        source_sha256=hashes,input_sha256=input_hashes,expected_total_blocks=p.Lx//den*p.Ly,
        estimated_peak_array_and_allocator_bytes=estimate,required_available_bytes=admission,
        memory_before=memory_status(),cases={},blocks_completed=0,complete_torus=False,
        physical_B=2*np.pi*num/den,transition_policy='All bands and pairs; no occupation or spectral cutoff',
        response_policy='Sum bare full and PP blocks first, then one RPA; stable df from frozen sampler',
        collective_poles_or_gap_certified=False,transverse_convergence_certified=False)
    def checkpoint():write_json(output/'diagnostic.json',report)
    def log(message):
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as f:f.write(message+'\n')
    def progress(chunks,done,total):
        elapsed=time.perf_counter()-start
        report.update(blocks_completed=done,elapsed_seconds=elapsed,
            estimated_total_seconds_from_completed_blocks=elapsed*total/done)
        checkpoint()
        log(f'{field} chunks={chunks}, blocks={done}/{total}, elapsed={elapsed:.1f}s, projected={elapsed*total/done:.1f}s')
    checkpoint()
    try:
        if report['memory_before']['available_physical_bytes']<admission:raise MemoryError('Initial memory admission failed')
        result,done=accumulate(p,num,den,chunk,specs,progress=progress,pilot_chunks=pilot_chunks,admission=admission)
        complete=done==report['expected_total_blocks'];report['complete_torus']=complete
        expected=done*p.Nkz*(4*den)**2
        for name,s in specs.items():
            report['cases'][name]=sampler.finish_case(output,name,field,p,s,result[name],complete,expected)
        report['status']='PILOT_PARTIAL_TORUS_COMPLETE' if not complete else (
            'DETECTOR_BOUNDARY_SAMPLING_COMPLETE' if all(m['checks']['identity_checks_pass'] for m in report['cases'].values())
            else 'FULL_BAND_IDENTITY_FAILURE')
    except Exception as exc:
        report.update(status='DETECTOR_BOUNDARY_SAMPLING_FAILED',failure=dict(type=type(exc).__name__,message=str(exc)))
        log(traceback.format_exc())
    report['sources_unchanged']=all(sha(Path(p))==h for p,h in hashes.items())
    report['inputs_unchanged']=all(sha(Path(p))==h for p,h in input_hashes.items())
    if not report['sources_unchanged'] or not report['inputs_unchanged']:report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-start
    log('Final status: '+report['status'])
    report['output_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='diagnostic.json'}
    checkpoint()
    if report['status'] not in ('DETECTOR_BOUNDARY_SAMPLING_COMPLETE','PILOT_PARTIAL_TORUS_COMPLETE'):raise SystemExit(1)
    return report


def profile(data,truth,names,ua):
    """Same 65-point g profile; independent denser-grid comparison is evidence only."""
    def cost(g):
        vector,_=residual(models_for(data,names,g,ua),truth,names,'charge_only')
        return float(vector@vector)
    fit=numerical_profile(cost,np.linspace(.2,1.6,65))
    check=numerical_profile(cost,np.linspace(.2,1.6,129))
    # A declared physical reference is a valid common candidate for every Ua.
    # Near an exact zero cost, scipy's bounded location error can exceed the
    # RMS consistency budget even though it is harmless for finite profiles.
    anchor_cost=cost(.8)
    for candidate in (fit,check):
        candidate['unanchored_g']=candidate['g']
        candidate['unanchored_cost']=candidate['cost']
        candidate['reference_g_candidate_cost']=anchor_cost
        candidate['reference_g_candidate_selected']=anchor_cost<candidate['cost']
        if candidate['reference_g_candidate_selected']:
            candidate.update(g=.8,cost=anchor_cost,kind='declared_reference_g_candidate',success=True)
    fit['normalized_mismatch']=float(np.sqrt(fit['cost']))
    fit['denser_grid_mismatch']=float(np.sqrt(check['cost']))
    fit['denser_grid_g']=check['g']
    fit['denser_grid_mismatch_difference']=abs(fit['normalized_mismatch']-fit['denser_grid_mismatch'])
    fit['denser_grid_all_refinements_successful']=check['all_refinements_successful']
    fit['optimizer_comparison_pass']=bool(fit['all_refinements_successful'] and check['all_refinements_successful']
        and fit['denser_grid_mismatch_difference']<1e-10)
    return fit


def add_higher(windows,higher_root,snapshot):
    for field in ('B0','Bplus'):
        folder=higher_root/(field+'_higher')
        report=json.loads(snapshot.read(folder/'diagnostic.json'))
        if report['status']!='DETECTOR_BOUNDARY_SAMPLING_COMPLETE' or not report['complete_torus']:
            raise ValueError('Incomplete higher mesh: '+field)
        expected_p,_,_,_,names=mesh_spec(field)
        if report['parameters']!=expected_p.__dict__ or set(report['cases'])!=set(names):
            raise ValueError('Higher mesh or case inventory mismatch')
        for p,h in {**report['source_sha256'],**report['input_sha256']}.items():snapshot.read(Path(p),h)
        for case,metadata in report['cases'].items():
            if not metadata['checks']['identity_checks_pass'] or not metadata['complete_torus']:
                raise ValueError('Higher full-band checks failed')
            path=folder/metadata['csv']
            rows=read_csv(snapshot.read(path,report['output_sha256'][path.name]))
            omega=np.array([float(r['omega']) for r in rows])
            for key,value in dict(q=metadata['q'],mu=.5,T=.04,eta=metadata['eta'],
                                  Lx=expected_p.Lx,Ly=expected_p.Ly,Nkz=expected_p.Nkz,density=metadata['density']).items():
                if not np.allclose([float(r[key]) for r in rows],value,rtol=0,atol=2e-14):
                    raise ValueError('Higher CSV metadata mismatch: '+key)
            for window,levels in windows.items():
                wanted=np.array(metadata[window+'_frequencies'])
                ids=[int(np.argmin(abs(omega-w))) for w in wanted]
                if len(ids)!=45 or len(set(ids))!=45 or max(abs(omega[i]-w) for i,w in zip(ids,wanted))>2e-14:
                    raise ValueError('Higher frequency window missing')
                selected=[rows[i] for i in ids];pi=extract_matrix(selected,'bare');saved=extract_matrix(selected,'rpa')
                if np.max(abs(cofactor_response(pi,metadata['q'],.8,8.)-saved))>2e-12:
                    raise ValueError('Higher RPA/cofactor disagreement')
                levels.setdefault('higher',{})[field+'_'+case]=dict(pi=pi,q=metadata['q'],omega=wanted,
                    item=dict(field=field,case=case,parameters=expected_p.__dict__,convergence_status='not_certified',
                              q=metadata['q'],mu=.5,eta=metadata['eta']),source_csv=str(path),saved_rpa=saved)


def compare_detector_truth(coarse,fine,names,window,coarse_label,fine_label):
    a,b=truths_for(coarse),truths_for(fine);rows=[]
    for name in names:
        row=dict(dataset=name,window=window,coarse_level=coarse_label,fine_level=fine_label,
                 q=fine[name]['q'],normalization='Relative L2 and max-absolute/sup(fine), separately from absolute-scaled kernel norm')
        for channel,label in ((0,'cc'),(1,'aa')):
            av,bv=a[name][:,channel,channel],b[name][:,channel,channel]
            row[label+'_relative_l2']=float(np.linalg.norm(bv-av)/np.linalg.norm(bv))
            row[label+'_relative_sup']=float(np.max(abs(bv-av))/np.max(abs(bv)))
            row[label+'_max_absolute_difference']=float(np.max(abs(bv-av)))
            row[label+'_fine_l2_normalizer']=float(np.linalg.norm(bv))
        row['rpa_all_cells_max_abs_over_one_plus_fine']=float(np.max(abs(b[name]-a[name])/(1+abs(b[name]))))
        rows.append(row)
    return rows


def plot_results(output,own,shared):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    labels=['coarse','fine','higher']
    for ax,t in zip(axes,TARGETS):
        for rows,label,color,marker in ((own,'Own mesh truth','#0072B2','o'),(shared,'Shared higher truth','#D55E00','s')):
            rr=[r for r in rows if r['target']==t['id'] and r['ua']==t['ua']]
            rr.sort(key=lambda r:labels.index(r['model_level']))
            if rr:ax.plot([labels.index(r['model_level']) for r in rr],[100*(r['normalized_mismatch']-t['threshold']) for r in rr],
                label=label,color=color,marker=marker,mfc='none',lw=1.,ls='--' if label.startswith('Shared') else '-')
        ax.axhline(0,color='black',ls=':',lw=.8,label='Original threshold')
        ax.set(xticks=range(3),xticklabels=labels,ylabel='RMS minus threshold (percentage points)',title=t['id'])
        ax.legend(fontsize=7)
    fig.savefig(output/'threshold_margins.png',dpi=170);fig.savefig(output/'threshold_margins.pdf');plt.close(fig)


def profiles(output,higher_root=None):
    output=new_output(output);start=time.perf_counter();snapshot=Snapshot()
    for name in SOURCE_NAMES:snapshot.read(ROOT/name)
    index,windows,cache_error=load_windows(INDEX,snapshot)
    old_path=ROOT/'parameter_calibration_independent_review/run_01/coarse_at_final_fit.csv'
    old=read_csv(snapshot.read(old_path))
    if higher_root is not None:add_higher(windows,Path(higher_root).resolve(),snapshot)
    own=[];shared=[];truth_comparison=[];original=[]
    levels=('coarse','fine','higher') if higher_root is not None else ('coarse','fine')
    for t in TARGETS:
        window=t['window'];names=t['names'];data=windows[window]
        old_group={'Bplus_eta_low_Ua1':'single_Bplus_eta_low','B0_qscan_Ua20':'q_scan_B0',
                   'B0_qdouble_scaled_Ua17':'single_B0_qdouble'}[t['id']]
        previous=[r for r in old if r['group']==old_group and r['window']==window and
                  r['mode']=='charge_only' and float(r['ua'])==t['ua']]
        if len(previous)!=1:raise ValueError('Original flip row missing')
        original.extend(previous)
        candidates=sorted({t['ua']-.5,t['ua'],t['ua']+.5,8.})
        for level in levels:
            own_truth=truths_for(data[level])
            for ua in candidates:
                for truth_kind,truth,out in [('own_mesh',own_truth,own)]+(
                    [('shared_higher',truths_for(data['higher']),shared)] if higher_root is not None else []):
                    fit=profile(data[level],truth,names,ua)
                    row=dict(target=t['id'],window=window,mode='charge_only',ua=ua,model_level=level,
                        truth_reference=truth_kind,threshold=t['threshold'],**fit,
                        threshold_minus_rms=t['threshold']-fit['normalized_mismatch'],
                        within_declared_threshold=fit['normalized_mismatch']<=t['threshold'],
                        g_bias_from_true=fit['g']-.8,confidence_interval=False)
                    out.append(row)
        for lower,upper in zip(levels,levels[1:]):
            truth_comparison.extend(compare_detector_truth(data[lower],data[upper],names,window,lower,upper))
    # Deduplicate qdouble records that legitimately appear under two windows.
    truth_comparison=list({(r['dataset'],r['window'],r['coarse_level'],r['fine_level']):r for r in truth_comparison}.values())
    comparisons=[]
    for kind,rows in [('own_mesh',own),('shared_higher',shared)]:
        for t in TARGETS:
            for ua in sorted({r['ua'] for r in rows if r['target']==t['id']}):
                for lower,upper in zip(levels,levels[1:]):
                    a=next(r for r in rows if r['target']==t['id'] and r['ua']==ua and r['model_level']==lower)
                    b=next(r for r in rows if r['target']==t['id'] and r['ua']==ua and r['model_level']==upper)
                    comparisons.append(dict(target=t['id'],ua=ua,truth_reference=kind,coarse_level=lower,fine_level=upper,
                        threshold=t['threshold'],coarse_rms=a['normalized_mismatch'],fine_rms=b['normalized_mismatch'],
                        rms_change=b['normalized_mismatch']-a['normalized_mismatch'],
                        coarse_margin=a['threshold_minus_rms'],fine_margin=b['threshold_minus_rms'],
                        classification_flip=a['within_declared_threshold']!=b['within_declared_threshold'],
                        g_change=b['g']-a['g'],rigorous_error_bound=False))
    csv_write(output/'own_mesh_truth_profiles.csv',own)
    if shared:csv_write(output/'shared_higher_truth_profiles.csv',shared)
    csv_write(output/'profile_grid_changes.csv',comparisons)
    csv_write(output/'detector_truth_grid_differences.csv',truth_comparison)
    csv_write(output/'original_fixed_g_flip_rows.csv',original)
    plot_results(output,own,shared)
    snapshot.check()
    success=all(r['optimizer_comparison_pass'] for r in own+shared)
    report=dict(status='TARGETED_PROFILE_REFINEMENT_COMPLETE' if success else 'TARGETED_PROFILE_OPTIMIZER_UNRESOLVED',
        invocation=sys.argv,elapsed_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat(),
        input_source_sha256=snapshot.hashes,inputs_sources_unchanged=True,cached_cofactor_max_error=cache_error,
        target_count=3,own_mesh_profile_count=len(own),shared_higher_profile_count=len(shared),
        higher_mesh_included=higher_root is not None,all_optimizer_comparisons_pass=success,
        g_bounds=[.2,1.6],g_grid_points=65,independent_g_grid_points=129,
        thresholds=[.001,.01],scientific_or_statistical_pass_inferred=False,
        collective_gap_certified=False,magnetic_transverse_convergence_certified=False,
        common_truth_reason='Own-mesh truth can cancel common discretization error; shared higher truth is a separate numerical experiment.',
        limitations=['Higher mesh is a numerical reference, not exact continuum truth.',
            'B0 refinement changes transverse and longitudinal meshes together; Bplus changes only longitudinal mesh.',
            'Same fixed-mu Hamiltonian and real interaction family; calibrated complex charge response only.',
            'Numerical candidate search is not a certified global g minimum; sampled Ua points are not confidence intervals.',
            'No arbitrary multiple of mesh difference is treated as an error bound.'])
    report['output_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='diagnostic.json'}
    write_json(output/'diagnostic.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_source_sha256','output_sha256')},indent=2))
    for t in TARGETS:
        print(t['id'],[(r['model_level'],r['g'],r['normalized_mismatch'],r['within_declared_threshold'])
                       for r in own if r['target']==t['id'] and r['ua']==t['ua']],flush=True)
    if not success:raise SystemExit(1)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    sp=sub.add_parser('sample');sp.add_argument('--field',choices=('B0','Bplus'),required=True)
    sp.add_argument('--output',type=Path,required=True);sp.add_argument('--pilot-chunks',type=int,default=0)
    pp=sub.add_parser('profiles');pp.add_argument('--output',type=Path,required=True);pp.add_argument('--higher-root',type=Path)
    args=ap.parse_args()
    if args.action=='sample':sample(args.output,args.field,args.pilot_chunks)
    else:profiles(args.output,args.higher_root)