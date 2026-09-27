"""Bounded all-band samples for UV matching: Bplus reference and B0 grids.

Only total and positive-positive sector responses are retained as additive
outputs. All raw transitions enter the total. Separate-sector M0 and Ward
residuals are reported, not assumed zero or repaired.
"""
import argparse,json,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from lattice_response import Parameters,interaction_matrix,rpa_response,loss_matrix
from magnetic_bloch_response import iter_magnetic_bloch_spectra,magnetic_bloch_transitions
from low_energy_mode_audit import sector_response
from extend_transverse import memory_status,tile_transitions,sha,csv_write
from run_pole_zero_experiment import write_json
from uv_matching_audit import read_sectors,method_responses
ROOT=Path(__file__).resolve().parent


def matrix_moment(tr,mask):
    cc=float(np.sum(tr.cc[mask]));aa=float(np.sum(tr.aa[mask]));ca=complex(np.sum(tr.ca[mask]))
    return np.array([[cc,ca],[ca.conjugate(),aa]])


def sample_case(output,label,p,omega,magnetic,log):
    started=time.perf_counter();pairs=p.Nkz*(144**2 if magnetic else min(128,p.Lx*p.Ly)*16)
    estimate=384*pairs+256*1024**2;admission=2*estimate+512*1024**2
    before=memory_status()
    if before['available_physical_bytes']<admission:raise MemoryError('Complete transition block not admitted')
    if magnetic:
        iterator=iter_magnetic_bloch_spectra(p,1,36,blocks_per_chunk=1);blocks=p.Lx//36*p.Ly
    else:
        iterator=range(0,p.Lx*p.Ly,128);blocks=(p.Lx*p.Ly+127)//128
    result=np.zeros((2,len(omega),2,2),complex);moments=np.zeros((2,2,2),complex)
    fullprobe=np.zeros((3,3),complex);lowprobe=np.zeros((3,3),complex)
    count=0;vertex=0.;density=0.;q=2*np.pi/24;z=.275+.015j;Q=2*np.sin(q/2)
    for i,item in enumerate(iterator):
        if memory_status()['available_physical_bytes']<admission:raise MemoryError('Available memory fell below fixed block admission')
        if magnetic:tr=magnetic_bloch_transitions(item,.5,.04,p.Nkz//24)
        else:tr=tile_transitions(p,item,min(item+128,p.Lx*p.Ly),.5,.04,p.Nkz//24)
        result+=sector_response(tr,omega,.015,np.zeros(tr.delta.size,dtype=bool))[:2]
        moments[0]+=tr.moments()[0];moments[1]+=matrix_moment(tr,tr.conduction)
        fullprobe+=tr.full_response(z)
        low=replace(tr,cc=tr.cc*tr.conduction,aa=tr.aa*tr.conduction,ca=tr.ca*tr.conduction)
        lowprobe+=low.full_response(z)
        count+=tr.delta.size;vertex=max(vertex,tr.vertex_error);density+=tr.density_value
        del tr,low,item
        if i==0 or (i+1)%8==0 or i+1==blocks:log(f'{label}: blocks {i+1}/{blocks}, elapsed {time.perf_counter()-started:.1f}s')
    contraction=np.array([z,0.,-Q]);u=interaction_matrix(Q,.8,8.)
    ward=lambda x:float(max(np.max(abs(contraction@x)),np.max(abs(x@contraction))))
    expected=p.Nkz*((p.Lx//36*p.Ly*144**2) if magnetic else p.Lx*p.Ly*16)
    fullrpa=np.array([rpa_response(x,u) for x in result[0]])
    checks=dict(raw_count=count,expected_raw_count=expected,vertex_error=vertex,density=density,
        total_moment0=moments[0],positive_positive_moment0=moments[1],excluded_moment0=moments[0]-moments[1],
        total_moment0_max=float(np.max(abs(moments[0]))),positive_positive_moment0_max=float(np.max(abs(moments[1]))),
        total_charge_ward_bare=ward(fullprobe),total_charge_ward_rpa=ward(rpa_response(fullprobe,u)),
        positive_positive_charge_ward_bare=ward(lowprobe),
        positive_positive_charge_ward_rpa_naive=ward(rpa_response(lowprobe,u)),
        minimum_total_weighted_loss_eigenvalue=min(float(np.linalg.eigvalsh(loss_matrix(complex(w,.015)*x)).min()) for w,x in zip(omega,fullrpa)),
        memory_before=before,estimated_peak_bytes=estimate,admission_required_bytes=admission,
        elapsed_seconds=time.perf_counter()-started,parameters=p.__dict__,magnetic=magnetic,q=q,eta=.015)
    checks['all_total_identity_checks_pass']=bool(count==expected and vertex<2e-11 and max(checks['total_moment0_max'],checks['total_charge_ward_bare'],checks['total_charge_ward_rpa'])<2e-10 and checks['minimum_total_weighted_loss_eigenvalue']>-2e-10)
    rows=[]
    for sector,values in zip(('total','positive_positive'),result):
        for w,mat in zip(omega,values):
            row=dict(sector=sector,omega=float(w),eta=.015)
            for a in range(2):
                for b in range(2):row[f'bare_{a}{b}_re']=float(mat[a,b].real);row[f'bare_{a}{b}_im']=float(mat[a,b].imag)
            rows.append(row)
    csv_write(output/f'{label}.csv',rows);write_json(output/f'{label}_checks.json',checks)
    if not checks['all_total_identity_checks_pass']:raise RuntimeError('Total full-band identities failed: '+label)
    return result,checks


def compare_case(coarse,fine,q):
    u=interaction_matrix(2*np.sin(q/2),.8,8.)
    cres,_,_=method_responses(coarse[0],coarse[1],u,coarse[0,0]-coarse[1,0])
    fres,_,_=method_responses(fine[0],fine[1],u,fine[0,0]-fine[1,0])
    metrics={}
    for name,c,f in [('bare_total',coarse[0],fine[0]),('bare_positive_positive',coarse[1],fine[1])]+[(key,cres[key],fres[key]) for key in ('full','naive_low_bare_U','matched_constant')]:
        metrics[name]=float(np.max(abs(f-c)/(1+abs(f))))
    return dict(scaled_errors=metrics,tolerance=.001,all_selected_responses_pass=max(metrics.values())<=.001,
                definition='max over all sampled complex 2x2 cells of |fine-coarse|/(1+|fine|)')


def main(output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('New empty sample directory required')
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def log(message):
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as stream:stream.write(message+'\n')
    source_names=('lattice_response.py','magnetic_bloch_response.py','bloch_response.py','low_energy_mode_audit.py','extend_transverse.py','run_pole_zero_experiment.py','uv_matching_audit.py','sample_uv_sectors.py')
    sources=[ROOT/n for n in source_names]
    prior=ROOT/'low_energy_mode_audit/transition_sectors.csv'
    report=dict(status='RUNNING',source_sha256={x.name:sha(x) for x in sources},input_sha256={str(prior.relative_to(ROOT)):sha(prior)},
        settings=dict(mu=.5,temperature=.04,q=2*np.pi/24,eta=.015,Bplus=2*np.pi/36),
        sector_definition='Both initial and final energies strictly >0; original occupation differences and vertices. All other transitions retained in the total.',
        cases={},comparisons={},collective_gap_certified=False)
    checkpoint=lambda:write_json(output/'diagnostic.json',report)
    checkpoint()
    try:
        magnetic,checks=sample_case(output,'Bplus_final_reference',Parameters(Lx=72,Ly=13,Nkz=192),np.array([0.,.05,.2746875,.35]),True,log)
        report['cases']['Bplus_final_reference']=checks
        ow,ot,ol=read_sectors(prior);old=np.array([ot,ol]);fresh=np.array([magnetic[:,i] for i in (1,2,3)]).swapaxes(0,1)
        indices=[int(np.argmin(abs(ow-w))) for w in (.05,.2746875,.35)]
        reproduction=float(np.max(abs(fresh-old[:,indices])))
        report['Bplus_existing_frequency_reproduction_error']=reproduction
        if reproduction>2e-12:raise RuntimeError('Bplus static-reference run fails old frequency reproduction')
        checkpoint()
        omega=np.unique(np.r_[0.,ow]);arrays={}
        for size,nk in ((48,192),(72,192),(72,384)):
            label=f'B0_L{size}_Nk{nk}'
            arrays[label],report['cases'][label]=sample_case(output,label,Parameters(Lx=size,Ly=size,Nkz=nk),omega,False,log)
            checkpoint()
        report['comparisons']['B0_transverse_L48_to72_Nk192']=compare_case(arrays['B0_L48_Nk192'],arrays['B0_L72_Nk192'],2*np.pi/24)
        report['comparisons']['B0_longitudinal_L72_Nk192_to384']=compare_case(arrays['B0_L72_Nk192'],arrays['B0_L72_Nk384'],2*np.pi/24)
        report['all_selected_B0_responses_converged']=all(c['all_selected_responses_pass'] for c in report['comparisons'].values())
        report['status']='UV_SECTOR_SAMPLING_COMPLETE' if report['all_selected_B0_responses_converged'] else 'UV_SECTOR_SAMPLING_COMPLETE_B0_UNCONVERGED'
    except Exception as exc:
        report['status']='UV_SECTOR_SAMPLING_FAILED';report['failure']=dict(type=type(exc).__name__,message=str(exc));log(traceback.format_exc())
    report['sources_unchanged']=report['source_sha256']=={x.name:sha(x) for x in sources}
    report['input_unchanged']=report['input_sha256'][str(prior.relative_to(ROOT))]==sha(prior)
    if not report['sources_unchanged'] or not report['input_unchanged']:report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-start;log('Final status: '+report['status'])
    report['output_sha256']={x.name:sha(x) for x in output.iterdir() if x.is_file() and x.name!='diagnostic.json'}
    checkpoint()
    if report['status'] in ('UV_SECTOR_SAMPLING_FAILED','SOURCE_OR_INPUT_CHANGED'):raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
