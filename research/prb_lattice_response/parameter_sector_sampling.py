"""OFAT q/mu/eta extension with exact all-band magnetic blocks and stable df.

Existing sources/evidence are read-only. Positive-positive means Ei>0 and Ej>0,
not an energy-transfer cutoff. Sum bare blocks before the single RPA solve.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.special import expit
from lattice_response import (AXIAL,GAMMA,MASS,Parameters,Transitions,
    interaction_matrix,rpa_response,loss_matrix)
from magnetic_bloch_response import iter_magnetic_bloch_spectra
from extend_transverse import memory_status,sha,csv_write
from run_pole_zero_experiment import write_json
from uv_matching_audit import read_sectors,method_responses,detector_metrics,match_response

ROOT=Path(__file__).resolve().parent
BASE_Q=2*np.pi/24
CASE_SETTINGS={
 'baseline':(1.,.5,.015),'qhalf':(.5,.5,.015),'qdouble':(2.,.5,.015),
 'mu_low':(1.,.25,.015),'mu_high':(1.,.75,.015),
 'eta_low':(1.,.5,.0075),'eta_high':(1.,.5,.03)}
SOURCES=('parameter_sector_sampling.py','test_parameter_sector_sampling.py',
 'lattice_response.py','magnetic_bloch_response.py','extend_transverse.py',
 'run_pole_zero_experiment.py','uv_matching_audit.py','low_energy_mode_audit.py')
INPUTS=('low_energy_mode_audit/transition_sectors.csv',
 'uv_matching_audit/sector_sampling/B0_L72_Nk384.csv')


def stable_fermi_difference(x,y):
    """f(x)-f(y) without subtracting two rounded occupations; x=(Ei-mu)/T.

    No threshold or clipping. Exact x==y gives zero. Finite binary64 input
    energies remain binary64; this does not recover lost eigenvalue precision.
    """
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Finite dimensionless energies required')
    lo=np.minimum(x,y);hi=np.maximum(x,y)
    return np.sign(y-x)*expit(-lo)*expit(hi)*(-np.expm1(lo-hi))


@dataclass
class Vertices:
    delta:np.ndarray
    charge:np.ndarray
    axial:np.ndarray
    conduction:np.ndarray
    q:float
    Q:float
    vertex_error:float
    shift:int
    plane_pairs:int


def transition_vertices(spec,qshift):
    p=spec.parameters
    if isinstance(qshift,bool) or int(qshift)!=qshift or qshift==0 or abs(qshift)>=p.Nkz:
        raise ValueError('Nonzero integer |qshift|<Nkz required')
    shift=int(qshift);q=2*np.pi*shift/p.Nkz;Q=2*np.sin(q/2)
    dim=4*spec.denominator;plane=len(spec.block_indices)*dim**2;n=p.Nkz*plane
    delta=np.empty(n);charge=np.empty(n,complex);axial=np.empty(n,complex);mask=np.empty(n,bool)
    eye=np.eye(spec.denominator);ao=np.kron(eye,AXIAL);mo=np.kron(eye,MASS);gz=np.kron(eye,GAMMA[2])
    error=0.
    for i,k in enumerate(spec.kz):
        j=(i+shift)%p.Nkz;udag=spec.vectors[:,i].conj().swapaxes(-1,-2);v=spec.vectors[:,j]
        r=udag@v;a=(udag@ao)@v;d=spec.energies[:,j,None,:]-spec.energies[:,i,:,None]
        jo=p.wilson*np.sin(k+q/2)*mo+p.hopping*np.cos(k+q/2)*gz
        current=(udag@jo)@v
        error=max(error,float(np.max(abs(Q*current-d*r))))
        sl=slice(i*plane,(i+1)*plane)
        delta[sl]=d.ravel();charge[sl]=r.ravel();axial[sl]=a.ravel()
        mask[sl]=((spec.energies[:,i,:,None]>0)&(spec.energies[:,j,None,:]>0)).ravel()
    return Vertices(delta,charge,axial,mask,q,Q,error,shift,plane)


def weighted_transitions(spec,v,mu,T,legacy=False):
    if T<=0:raise ValueError('Positive temperature required')
    p=spec.parameters;n=v.delta.size;cc=np.empty(n);aa=np.empty(n);ca=np.empty(n,complex)
    restored=0;df_nonzero=0
    for i in range(p.Nkz):
        j=(i+v.shift)%p.Nkz;x=(spec.energies[:,i,:,None]-mu)/T;y=(spec.energies[:,j,None,:]-mu)/T
        old=expit(-x)-expit(-y)
        df=old if legacy else stable_fermi_difference(x,y)
        restored+=int(np.count_nonzero((old==0)&(df!=0)));df_nonzero+=int(np.count_nonzero(df))
        df=df.ravel()/p.volume;sl=slice(i*v.plane_pairs,(i+1)*v.plane_pairs)
        cc[sl]=df*np.abs(v.charge[sl])**2;aa[sl]=df*np.abs(v.axial[sl])**2
        ca[sl]=df*v.charge[sl]*v.axial[sl].conj()
    tr=Transitions(v.delta,cc,aa,ca,v.conduction,v.Q,v.q,v.vertex_error,
                   float(expit((mu-spec.energies)/T).sum()/p.volume))
    return tr,dict(raw_pairs=n,nonzero_occupation_differences=df_nonzero,
                   restored_occupation_differences=restored)


def two_sector_response(tr,omega,eta,batch=2):
    """All pairs, real BLAS, total and positive-positive; no spectral cutoff."""
    omega=np.asarray(omega,float)
    if eta<=0 or batch<1:raise ValueError('Positive eta and frequency batch required')
    n=tr.delta.size;weights=np.empty((8,n),float)
    for i,value in enumerate((tr.cc,tr.aa,tr.ca.real,tr.ca.imag)):
        weights[i]=value;np.multiply(value,tr.conduction,out=weights[i+4])
    result=np.empty((2,len(omega),2,2),complex)
    for first in range(0,len(omega),batch):
        last=min(first+batch,len(omega));real=omega[None,first:last]-tr.delta[:,None]
        den=np.square(real);den+=eta*eta;np.divide(real,den,out=real)
        np.reciprocal(den,out=den);den*=-eta
        re=weights@real;im=weights@den
        for sector,offset in ((0,0),(1,4)):
            k=offset;block=result[sector,first:last]
            block[:,0,0]=re[k]+1j*im[k];block[:,1,1]=re[k+1]+1j*im[k+1]
            block[:,0,1]=(re[k+2]-im[k+3])+1j*(im[k+2]+re[k+3])
            block[:,1,0]=(re[k+2]+im[k+3])+1j*(im[k+2]-re[k+3])
    return result


def case_specs(base_omega,nkz,names=None):
    specs={}
    for name in names or CASE_SETTINGS:
        factor,mu,eta=CASE_SETTINGS[name];shift=int(round(nkz*factor/24))
        q=2*np.pi*shift/nkz
        if abs(q-factor*BASE_Q)>1e-14:raise ValueError('Nkz cannot represent requested physical q')
        fixed=np.asarray(base_omega,float);scaled=fixed*factor
        omega=np.unique(np.r_[0.,fixed,scaled])
        specs[name]=dict(case=name,q=q,qshift=shift,mu=mu,T=.04,eta=eta,omega=omega,
                         fixed_lab_frequencies=fixed,scaled_q_frequencies=scaled)
    return specs


def _moment(tr,mask=None):
    cc=tr.cc if mask is None else tr.cc[mask];aa=tr.aa if mask is None else tr.aa[mask]
    ca=tr.ca if mask is None else tr.ca[mask]
    a=np.sum(ca);return np.array([[np.sum(cc),a],[a.conjugate(),np.sum(aa)]])


def _empty_result(s):
    return dict(bare=np.zeros((2,len(s['omega']),2,2),complex),moment0=np.zeros((2,2,2),complex),
        current_probe=np.zeros((3,3),complex),density=0.,vertex_error=0.,raw_pairs=0,
        nonzero_occupation_differences=0,restored_occupation_differences=0)


def _add_stats(result,tr,stats,s):
    result['moment0'][0]+=_moment(tr);result['moment0'][1]+=_moment(tr,tr.conduction)
    result['current_probe']+=tr.full_response(.275*s['q']/BASE_Q+1j*s['eta'])
    result['density']+=tr.density_value;result['vertex_error']=max(result['vertex_error'],tr.vertex_error)
    for key in stats:result[key]+=stats[key]


def _matrix_columns(row,prefix,value):
    for a in range(2):
        for b in range(2):
            row[f'{prefix}_{a}{b}_re']=float(value[a,b].real)
            row[f'{prefix}_{a}{b}_im']=float(value[a,b].imag)


def finish_case(output,name,field,p,s,result,complete,expected):
    w=s['omega'];total,low=result['bare'];u=interaction_matrix(2*np.sin(s['q']/2),.8,8.)
    methods,dynamic,constant=method_responses(total,low,u,total[0]-low[0])
    metrics={}
    for window in ('fixed_lab','scaled_q'):
        wanted=s[window+'_frequencies'];ix=np.array([int(np.argmin(abs(w-x))) for x in wanted])
        metrics[window]={method:detector_metrics(w[ix],methods['full'][ix],value[ix]) for method,value in methods.items()}
    z=.275*s['q']/BASE_Q+1j*s['eta'];contract=np.array([z,0.,-2*np.sin(s['q']/2)])
    ward=lambda mat:float(max(np.max(abs(contract@mat)),np.max(abs(mat@contract))))
    current=result['current_probe'];moment=result['moment0'];bareward=ward(current);rpaward=ward(rpa_response(current,u))
    mineig=min(float(np.linalg.eigvalsh(loss_matrix(complex(x,s['eta'])*mat)).min()) for x,mat in zip(w,methods['full']))
    exact=float(np.max(abs(methods['full']-methods['matched_dynamic'])))
    checks=dict(raw_pairs=result['raw_pairs'],expected_raw_pairs=expected,retains_all_raw_pairs=result['raw_pairs']==expected,
        nonzero_occupation_differences=result['nonzero_occupation_differences'],
        restored_occupation_differences=result['restored_occupation_differences'],vertex_error=result['vertex_error'],
        moment0_total=moment[0],moment0_positive_positive=moment[1],moment0_excluded=moment[0]-moment[1],
        full_moment0_max=float(np.max(abs(moment[0]))),positive_positive_moment0_max=float(np.max(abs(moment[1]))),
        total_charge_ward_bare=bareward,total_charge_ward_rpa=rpaward,charge_ward_probe=z,
        minimum_total_rpa_weighted_loss_eigenvalue=mineig,dynamic_matching_identity_max_error=exact)
    checks['identity_checks_pass']=bool(complete and checks['retains_all_raw_pairs'] and
        result['vertex_error']<2e-11 and max(checks['full_moment0_max'],bareward,rpaward)<2e-10 and mineig>=-2e-10 and exact<2e-12)
    rows=[]
    for i,x in enumerate(w):
        row=dict(omega=float(x),q=s['q'],mu=s['mu'],T=s['T'],eta=s['eta'],B=0. if field=='B0' else 2*np.pi/36,
            density=result['density'],Lx=p.Lx,Ly=p.Ly,Nkz=p.Nkz)
        for prefix,value in (('bare',total),('low',low),('rpa',methods['full']),
                             ('naive',methods['naive_low_bare_U']),('matched',methods['matched_constant'])):
            _matrix_columns(row,prefix,value[i])
        rows.append(row)
    path=output/f'{name}.csv';csv_write(path,rows)
    metadata={key:value for key,value in s.items() if key!='omega'}
    metadata.update(field=field,parameters=p.__dict__,density=result['density'],csv=path.name,
        all_frequencies=w,reference_z=complex(0.,s['eta']),checks=checks,metrics=metrics,
        reference_excluded=total[0]-low[0],reference_matching=constant[0],
        all_frequencies_excluded_axial_real_range=[float((total-low)[:,1,1].real.min()),float((total-low)[:,1,1].real.max())],
        convergence_status='not_certified',complete_torus=complete)
    write_json(output/f'{name}_metadata.json',metadata)
    return metadata


def read_response_csv(path,prefix):
    import csv
    with Path(path).open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    w=np.array([float(r['omega']) for r in rows])
    values=np.array([[[complex(float(r[f'{prefix}_{a}{b}_re']),float(r[f'{prefix}_{a}{b}_im'])) for b in range(2)] for a in range(2)] for r in rows])
    return w,values


def compare_outputs(coarse,fine):
    metrics={}
    for name in ('bare','low','rpa','naive','matched'):
        cw,c=read_response_csv(coarse,name);fw,f=read_response_csv(fine,name)
        if not np.array_equal(cw,fw):raise ValueError('Convergence frequency grids differ')
        metrics[name]=float(np.max(abs(f-c)/(1+abs(f))))
    return dict(scaled_errors=metrics,tolerance=.001,pass_all=max(metrics.values())<=.001,
        definition='Maximum over declared omega union and all complex 2x2 cells of |fine-coarse|/(1+|fine|)')


def assemble_index(output_root):
    datasets=[];inputhash={}
    for field in ('B0','Bplus'):
        final=output_root/f'{field}_final';coarse=output_root/f'{field}_coarse'
        reportpath=final/'diagnostic.json'
        if not reportpath.exists():continue
        report=json.loads(reportpath.read_text(encoding='utf-8'))
        if report['status']!='PARAMETER_SECTOR_SAMPLING_COMPLETE':continue
        inputhash[str(reportpath.relative_to(output_root))]=sha(reportpath)
        for name,m in report['cases'].items():
            item={key:m[key] for key in ('field','case','q','mu','T','eta','density','fixed_lab_frequencies','scaled_q_frequencies')}
            item.update(id=field+'_'+name,csv=str((final/m['csv']).relative_to(output_root)).replace('\\','/'),
                        mesh=m['parameters'],convergence_status='not_certified')
            cp=coarse/m['csv'];fp=final/m['csv']
            if cp.exists() and (coarse/'diagnostic.json').exists():
                cr=json.loads((coarse/'diagnostic.json').read_text(encoding='utf-8'))
                if cr['status']=='PARAMETER_SECTOR_SAMPLING_COMPLETE':
                    comparison=compare_outputs(cp,fp)
                    comparison.update(coarse_csv=str(cp.relative_to(output_root)).replace('\\','/'),coarse_sha256=sha(cp),fine_sha256=sha(fp),
                        refinement='B0 combined L48Nk192 to L72Nk384; not separate-direction certification' if field=='B0' else 'Bplus fixed transverse grid Nkz96 to192; no new transverse certification')
                    item.update(convergence=comparison,convergence_status='pass' if comparison['pass_all'] else 'fail')
            item['csv_sha256']=sha(fp);datasets.append(item)
    index=dict(schema_version=1,datasets=datasets,input_sha256=inputhash,
        complete_14_final_datasets=len(datasets)==14,collective_poles_or_gap_certified=False,
        df_policy='Stable binary64 occupation difference; no band or weight cutoff',
        note='Pass refers only to the explicitly recorded comparison, not all directions or eta->0.')
    write_json(output_root/'calibration_inputs.json',index)
    return index


def run(output_root,field,mesh,pilot_blocks=0,names=None):
    output=output_root/((f'pilot_{pilot_blocks}_' if pilot_blocks else '')+field+'_'+mesh)
    if output.exists() and any(output.iterdir()):raise RuntimeError('New empty evidence directory required')
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    def log(text):
        print(text,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as stream:stream.write(text+'\n')
    p=Parameters(Lx=72 if mesh=='final' or field=='Bplus' else 48,
        Ly=(13 if field=='Bplus' else (72 if mesh=='final' else 48)),
        Nkz=(192 if mesh=='final' else 96) if field=='Bplus' else (384 if mesh=='final' else 192))
    magnetic=field=='Bplus';chunk=1 if magnetic else 64;denominator=36 if magnetic else 1
    blocks=p.Lx//denominator*p.Ly;chunks=(blocks+chunk-1)//chunk
    base_omega,_,_=read_sectors(ROOT/INPUTS[0]);specs=case_specs(base_omega,p.Nkz,names)
    sources={x:sha(ROOT/x) for x in SOURCES};inputs={x:sha(ROOT/x) for x in INPUTS}
    report=dict(status='RUNNING',field=field,mesh=mesh,parameters=p.__dict__,source_sha256=sources,input_sha256=inputs,
        invocation=sys.argv,df_policy='Stable expit/expm1 difference; baseline legacy control uses the same eigenpairs.',
        transition_policy='All bands and all pairs retained. Low sector has both eigenenergies strictly positive.',
        complete_torus=False,case_settings={n:{k:v for k,v in s.items() if k!='omega'} for n,s in specs.items()},
        cases={},collective_poles_or_gap_certified=False,blocks_completed=0)
    checkpoint=lambda:write_json(output/'diagnostic.json',report)
    results={n:_empty_result(s) for n,s in specs.items()}
    legacy=np.zeros_like(results['baseline']['bare']) if 'baseline' in specs else None
    n=p.Nkz*chunk*(4*denominator)**2
    estimate=int(192*n+384*1024**2);admission=estimate+1024**3
    report.update(estimated_peak_array_and_allocator_bytes=estimate,required_available_bytes=admission,memory_before=memory_status())
    checkpoint();log(f'{field} {mesh}: chunks={chunks}, pair estimate/chunk={n}, required free={admission/1e9:.3f} GB')
    try:
        if memory_status()['available_physical_bytes']<admission:raise MemoryError('Initial block memory admission failed')
        for ib,spec in enumerate(iter_magnetic_bloch_spectra(p,int(magnetic),denominator,blocks_per_chunk=chunk)):
            free=memory_status()['available_physical_bytes']
            if free<admission:raise MemoryError('Runtime block memory admission failed')
            for shift in sorted(set(s['qshift'] for s in specs.values())):
                vertices=transition_vertices(spec,shift)
                selected={n:s for n,s in specs.items() if s['qshift']==shift}
                for mu in sorted(set(s['mu'] for s in selected.values())):
                    tr,stats=weighted_transitions(spec,vertices,mu,.04)
                    for name,s in selected.items():
                        if s['mu']!=mu:continue
                        results[name]['bare']+=two_sector_response(tr,s['omega'],s['eta'])
                        _add_stats(results[name],tr,stats,s)
                    del tr
                    if 'baseline' in selected and mu==.5:
                        old,_=weighted_transitions(spec,vertices,.5,.04,legacy=True)
                        s=specs['baseline'];legacy+=two_sector_response(old,s['omega'],s['eta']);del old
                del vertices
            report['blocks_completed']+=len(spec.block_indices);del spec;gc.collect()
            elapsed=time.perf_counter()-started
            log(f'{field} {mesh}: chunks {ib+1}/{chunks}, elapsed={elapsed:.1f}s, free={free/1e9:.2f} GB')
            report['elapsed_seconds']=elapsed;checkpoint()
            if pilot_blocks and ib+1>=pilot_blocks:break
        complete=report['blocks_completed']==blocks;report['complete_torus']=complete
        expected=(report['blocks_completed'] if not complete else blocks)*p.Nkz*(4*denominator)**2
        for name,s in specs.items():
            report['cases'][name]=finish_case(output,name,field,p,s,results[name],complete,expected)
        if legacy is not None:
            s=specs['baseline'];stable=results['baseline']['bare'];u=interaction_matrix(2*np.sin(s['q']/2),.8,8.)
            oldrpa=np.array([rpa_response(x,u) for x in legacy[0]])
            newrpa=np.array([rpa_response(x,u) for x in stable[0]])
            control=dict(same_eigenpairs_stable_vs_legacy_bare_max_absolute_error=float(np.max(abs(stable-legacy))),
                same_eigenpairs_stable_vs_legacy_rpa_max_absolute_error=float(np.max(abs(newrpa-oldrpa))),
                restored_occupation_differences=results['baseline']['restored_occupation_differences'])
            rows=[]
            for i,w in enumerate(s['omega']):
                row=dict(omega=float(w),q=s['q'],mu=.5,T=.04,eta=.015)
                _matrix_columns(row,'bare',legacy[0,i]);_matrix_columns(row,'low',legacy[1,i]);_matrix_columns(row,'rpa',oldrpa[i]);rows.append(row)
            csv_write(output/'baseline_legacy_df_control.csv',rows)
            if complete and mesh=='final':
                ow,ot,ol=read_sectors(ROOT/(INPUTS[0] if magnetic else INPUTS[1]));ix=[int(np.argmin(abs(s['omega']-x))) for x in ow]
                old=np.array([ot,ol]);control['legacy_vs_frozen_bare_max_absolute_error']=float(np.max(abs(legacy[:,ix]-old)))
                control['stable_vs_frozen_bare_max_absolute_error']=float(np.max(abs(stable[:,ix]-old)))
                control['frozen_input']=INPUTS[0] if magnetic else INPUTS[1]
            report['baseline_df_control']=control
        report['status']='PILOT_PARTIAL_TORUS_COMPLETE' if not complete else (
            'PARAMETER_SECTOR_SAMPLING_COMPLETE' if all(x['checks']['identity_checks_pass'] for x in report['cases'].values()) else 'FULL_BAND_IDENTITY_FAILURE')
    except Exception as exc:
        report['status']='PARAMETER_SECTOR_SAMPLING_FAILED';report['failure']=dict(type=type(exc).__name__,message=str(exc));log(traceback.format_exc())
    report['sources_unchanged']=sources=={x:sha(ROOT/x) for x in SOURCES}
    report['inputs_unchanged']=inputs=={x:sha(ROOT/x) for x in INPUTS}
    if not report['sources_unchanged'] or not report['inputs_unchanged']:report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-started;log('Final status: '+report['status'])
    report['output_sha256']={x.name:sha(x) for x in output.iterdir() if x.is_file() and x.name!='diagnostic.json'}
    checkpoint();assemble_index(output_root)
    if report['status'] not in ('PARAMETER_SECTOR_SAMPLING_COMPLETE','PILOT_PARTIAL_TORUS_COMPLETE'):raise SystemExit(1)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--field',choices=('B0','Bplus'));parser.add_argument('--mesh',choices=('coarse','final'),default='final')
    parser.add_argument('--pilot-blocks',type=int,default=0);parser.add_argument('--cases',nargs='+',choices=tuple(CASE_SETTINGS))
    parser.add_argument('--assemble',action='store_true')
    a=parser.parse_args()
    if a.assemble:assemble_index(a.output_root.resolve())
    else:
        if not a.field:parser.error('--field required unless --assemble')
        run(a.output_root.resolve(),a.field,a.mesh,a.pilot_blocks,a.cases)
