"""Low-energy finite-eta response audit with exact all-band control sums.

A loss maximum or a zero of Re(epsilon) is not an analytic collective pole.
No analytic continuation, pole residue, intrinsic damping, or hybrid-gap
certificate is inferred from the finite-eta samples in this first-stage audit.
"""
from __future__ import annotations
import argparse,csv,json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks,peak_prominences
from lattice_response import Parameters,Transitions,interaction_matrix,rpa_response,loss_matrix
from magnetic_bloch_response import iter_magnetic_bloch_spectra,magnetic_bloch_transitions,magnetic_bloch_spectrum
from run_magnetic_refinement import fast_density_responses
from extend_transverse import memory_status,sha,csv_write
from run_pole_zero_experiment import write_json
ROOT=Path(__file__).resolve().parent
CONTROL_SETTINGS=[(.8,0.,False),(.8,4.,False),(.8,8.,False),(.8,12.,False),(.4,8.,False),(0.,8.,False),(0.,0.,False),(.8,8.,True)]


def load_response(path):
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    omega=np.array([float(r['omega']) for r in rows])
    matrices={prefix:np.array([[[complex(float(r[f'{prefix}_{a}{b}_re']),float(r[f'{prefix}_{a}{b}_im']))
               for b in range(2)] for a in range(2)] for r in rows]) for prefix in ('bare','rpa')}
    if len(rows)!=161 or not np.allclose(omega,np.linspace(.025,1.2,161),rtol=0,atol=1e-15):
        raise ValueError('Unchanged complete original frequency grid required')
    return omega,matrices,dict(q=float(rows[0]['q']),eta=float(rows[0]['eta']),B=float(rows[0]['B']),density=float(rows[0]['density']))


def track_channels(matrices):
    """Continuous nearest-eigenvalue matching; unresolved near degeneracy exposed."""
    values=[]
    for matrix in matrices:
        val=np.linalg.eigvals(matrix)
        if not values:val=val[np.lexsort((val.imag,val.real))]
        elif np.sum(np.abs(val-values[-1]))>np.sum(np.abs(val[::-1]-values[-1])):val=val[::-1]
        values.append(val)
    result=np.array(values)
    return result,float(np.min(np.abs(result[:,0]-result[:,1])))


def loss_peaks(omega,loss):
    ids,_=find_peaks(loss)
    prominences=peak_prominences(loss,ids)[0] if len(ids) else []
    records=[]
    for i,prominence in zip(ids,prominences):
        level=loss[i]/2
        left=next((j for j in range(i-1,-1,-1) if loss[j]<=level),None)
        right=next((j for j in range(i+1,len(loss)) if loss[j]<=level),None)
        record=dict(index=int(i),omega=float(omega[i]),height=float(loss[i]),prominence=float(prominence),
                    finite_eta_peak_only=True,half_height_width=None)
        if left is not None and right is not None:
            wl=float(np.interp(level,[loss[left],loss[left+1]],[omega[left],omega[left+1]]))
            wr=float(np.interp(level,[loss[right],loss[right-1]],[omega[right],omega[right-1]]))
            record.update(half_height_width=wr-wl,left_half_height=wl,right_half_height=wr)
        records.append(record)
    return records


def analyze(omega,bare,q,eta,coulomb=.8,axial=8.,remove_mixing=False):
    pi=np.array(bare,copy=True)
    if remove_mixing:pi[:,0,1]=0.;pi[:,1,0]=0.
    u=interaction_matrix(2*np.sin(q/2),coulomb,axial)
    sq=np.sqrt(u)
    response=np.array([rpa_response(p,u) for p in pi])
    kernel=np.eye(2)-sq[None]@pi@sq[None]
    eigenvalues,branch_separation=track_channels(kernel)
    singular=np.linalg.svd(kernel,compute_uv=False)[:,-1]
    det=np.linalg.det(kernel)
    mixing=np.abs(kernel[:,0,1]*kernel[:,1,0])
    diag=np.abs(kernel[:,0,0]*kernel[:,1,1])
    fraction=np.divide(mixing,diag,out=np.full_like(mixing,np.inf),where=diag!=0)
    power=np.array([np.linalg.eigvalsh(loss_matrix(complex(w,eta)*x)).min() for w,x in zip(omega,response)])
    peaks={name:loss_peaks(omega,-response[:,c,c].imag) for name,c in [('charge',0),('axial',1)]}
    summary=dict(coulomb=coulomb,axial=axial,remove_mixing=remove_mixing,eta=eta,q=q,
        minimum_dielectric_singular_value=float(singular.min()),minimum_at_omega=float(omega[singular.argmin()]),
        minimum_tracked_branch_separation=branch_separation,
        maximum_offdiagonal_determinant_fraction=float(fraction.max()),
        maximum_bare_axial_real=float(pi[:,1,1].real.max()),minimum_weighted_loss_eigenvalue=float(power.min()),
        peaks=peaks,analytic_poles_residues_and_intrinsic_widths=None,hybrid_gap_certified=False)
    return dict(response=response,kernel=kernel,eigenvalues=eigenvalues,singular=singular,determinant=det,
                mixing_fraction=fraction,summary=summary)


def response_rows(omega,bare,result,case,dataset):
    rows=[]
    for i,w in enumerate(omega):
        row=dict(dataset=dataset,case=case,omega=float(w),eta=result['summary']['eta'],
            coulomb=result['summary']['coulomb'],axial=result['summary']['axial'],remove_mixing=result['summary']['remove_mixing'],
            minimum_singular_value=float(result['singular'][i]),mixing_determinant_fraction=float(result['mixing_fraction'][i]))
        for prefix,matrix in [('bare',bare[i]),('rpa',result['response'][i]),('dielectric',result['kernel'][i])]:
            for a in range(2):
                for b in range(2):row[f'{prefix}_{a}{b}_re']=float(matrix[a,b].real);row[f'{prefix}_{a}{b}_im']=float(matrix[a,b].imag)
        for branch in range(2):
            row[f'eigenchannel{branch}_re']=float(result['eigenvalues'][i,branch].real)
            row[f'eigenchannel{branch}_im']=float(result['eigenvalues'][i,branch].imag)
        rows.append(row)
    return rows


def sector_response(tr,omega,eta,diagonal_mask):
    """Independent all-pair total, positive-positive, equal-band-index sums.

    The latter is a sorted-band diagnostic, not invariant at band degeneracies.
    Complements retain ALL off-diagonal/inter-sign transitions in the total.
    """
    base=np.stack((tr.cc,tr.aa,tr.ca.real,tr.ca.imag))
    weights=np.concatenate((base,base*tr.conduction[None,:],base*diagonal_mask[None,:]),axis=0)
    output=np.empty((3,len(omega),2,2),complex)
    for start in range(0,len(omega),4):
        end=min(start+4,len(omega))
        d=1/(omega[None,start:end]+1j*eta-tr.delta[:,None])
        re,im=weights@d.real,weights@d.imag
        for k in range(3):
            a,b,c,e=re[4*k:4*k+4];ai,bi,ci,ei=im[4*k:4*k+4]
            output[k,start:end,0,0]=a+1j*ai;output[k,start:end,1,1]=b+1j*bi
            output[k,start:end,0,1]=(c-ei)+1j*(ci+e)
            output[k,start:end,1,0]=(c+ei)+1j*(ci-e)
    return output


def validate_sector_path():
    p=Parameters(Lx=3,Ly=4,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    spec=magnetic_bloch_spectrum(p,1,3)
    records=[]
    for shift in (-1,1):
        tr=magnetic_bloch_transitions(spec,.5,.04,shift)
        diag=np.broadcast_to(np.eye(12,dtype=bool),(24,4,12,12)).ravel()
        w=np.array([.091,.275,.91]);got=sector_response(tr,w,.0075,diag)
        expected=[]
        for mask in (np.ones(tr.delta.size,dtype=bool),tr.conduction,diag):
            subset=Transitions(delta=tr.delta,cc=tr.cc*mask,aa=tr.aa*mask,ca=tr.ca*mask,
                conduction=tr.conduction,Q=tr.Q,q=tr.q,vertex_error=tr.vertex_error,density_value=tr.density_value)
            expected.append([subset.density_response(complex(x,.0075)) for x in w])
        error=float(np.max(np.abs(got-np.array(expected))))
        records.append(dict(qshift=shift,maximum_absolute_error=error,passed=error<2e-12,raw_pairs=tr.delta.size))
    return dict(all_checks_pass=all(x['passed'] for x in records),records=records)


def figure_cached(output,datasets,analyses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for name,(w,mat,meta) in datasets.items():
        a=analyses[name][2]
        axes[0,0].plot(w,-a['response'][:,0,0].imag,label=name+' RPA')
        axes[0,0].plot(w,-mat['bare'][:,0,0].imag,'--',alpha=.6,label=name+' bare')
        axes[1,0].plot(w,a['singular'],label=name)
    w=datasets['Bplus'][0]
    for a in analyses['Bplus'][:4]:axes[0,1].plot(w,-a['response'][:,1,1].imag,label='Ua='+str(a['summary']['axial']))
    axes[1,1].plot(w,-analyses['Bplus'][2]['response'][:,0,0].imag,label='Full two-channel')
    axes[1,1].plot(w,-analyses['Bplus'][-1]['response'][:,0,0].imag,'--',label='Cross kernel set to zero')
    for ax,title in zip(axes.ravel(),['Charge loss, Ua=8','Axial loss, B=2pi/36','Dielectric minimum singular value','Mixing control, B=2pi/36']):
        ax.set_title(title);ax.set_xlabel('omega');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set_xlim(.025,.4);axes[0,1].set_xlim(.025,.4);axes[1,1].set_xlim(.025,.4)
    axes[1,0].set_ylim(bottom=0)
    fig.suptitle('Finite-eta response audit: peaks alone do not certify a hybrid gap')
    for ext in ('png','pdf'):fig.savefig(output/f'cached_response_audit.{ext}',dpi=170)
    plt.close(fig)


def figure_controls(output,w,totals,sectors,q):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for eta,pi in totals.items():
        result=analyze(w,pi,q,eta)
        axes[0].plot(w,-result['response'][:,0,0].imag,label='eta='+str(eta))
        axes[1].plot(w,result['singular'],label='eta='+str(eta))
    for label,pi in [('All transitions',sectors[0]),('Positive-positive',sectors[1]),('Equal sorted band index',sectors[2]),('Other band indices',sectors[0]-sectors[2])]:
        axes[2].plot(w,-pi[:,0,0].imag,label=label)
    for ax,title in zip(axes,['Charge loss, Ua=8','Dielectric minimum singular value','Bare charge loss sectors, eta=.015']):
        ax.set_title(title);ax.set_xlabel('omega');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[1].set_ylim(bottom=0)
    fig.suptitle('Same final magnetic mesh; eta controls are not intrinsic linewidths')
    for ext in ('png','pdf'):fig.savefig(output/f'eta_sector_controls.{ext}',dpi=170)
    plt.close(fig)


def main(output,controls):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh evidence output required')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def log(message):
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as stream:stream.write(message+'\n')
    paths=dict(B0=ROOT/'transverse_extension/L72_Nk384_flux+0_fixed_mu.csv',
               Bplus=ROOT/'magnetic_kz_continuation/Nkx2_Nky13_Nk192_B1of36_fixed_mu.csv')
    refs=[ROOT/'transverse_extension/diagnostic.json',ROOT/'magnetic_kz_continuation/diagnostic.json']
    sources=[ROOT/x for x in ('lattice_response.py','magnetic_bloch_response.py','run_magnetic_refinement.py',
        'extend_transverse.py','run_pole_zero_experiment.py','low_energy_mode_audit.py','test_low_energy_mode_audit.py')]
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={x.name:sha(x) for x in sources},input_sha256={str(x.relative_to(ROOT)):sha(x) for x in [*paths.values(),*refs]},
        settings=dict(mu=.5,temperature=.04,q=2*np.pi/24,coulomb_controls=[0.,.4,.8],axial_controls=[0.,4.,8.,12.]),
        low_energy_pair_certified=False,hybrid_gap_certified=False,analytic_continuum_sheet_constructed=False,
        unresolved_requirements=['Complex pole and matrix residue tracking on a justified continuum sheet',
          'Intrinsic widths separated from artificial eta and finite mesh spacing',
          'Two branches tracked through a detuning control with a stable minimum separation',
          'q, B and density/ensemble controls; fixed-mu B0 and Bplus have different densities',
          'Mesh refinement repeated at smaller eta; eta=.015 convergence is not transferred automatically'],
        limits=['This is a finite-response first-stage audit, not proof that no low-energy collective pole exists.',
          'Equal sorted-band-index sectors are basis/label dependent near degeneracies; energy-positive sectors include inter-subband transitions.',
          'Sector polarizations are added before RPA; sector losses alone are not dressed modes.'])
    checkpoint=lambda:write_json(output/'diagnostic.json',report)
    checkpoint()
    try:
        for ref in refs:
            prior=json.loads(ref.read_text(encoding='utf-8'))
            for name,value in prior.get('source_sha256',{}).items():
                if sha(ROOT/name)!=value:raise RuntimeError('Prior source changed: '+name)
            for name,value in prior.get('output_sha256',{}).items():
                if sha(ref.parent/name)!=value:raise RuntimeError('Prior output changed: '+name)
        datasets={name:load_response(path) for name,path in paths.items()}
        analyses={};all_rows=[];cached=[]
        for name,(w,mat,meta) in datasets.items():
            analyses[name]=[]
            for index,(g,ua,remove) in enumerate(CONTROL_SETTINGS):
                result=analyze(w,mat['bare'],meta['q'],meta['eta'],g,ua,remove)
                analyses[name].append(result);all_rows+=response_rows(w,mat['bare'],result,index,name)
                cached.append(dict(dataset=name,case=index,**{**meta,**result['summary']}))
            error=float(np.max(np.abs(analyses[name][2]['response']-mat['rpa'])))
            if error>2e-13:raise RuntimeError('Stored RPA does not match the retained bare matrix')
            log(f'{name}: stored RPA error={error:.3e}; minimum dielectric singular value={analyses[name][2]["singular"].min():.8g}')
        csv_write(output/'cached_controls.csv',all_rows)
        report['cached_cases']=cached
        figure_cached(output,datasets,analyses)
        report['status']='CACHED_RESPONSE_AUDIT_COMPLETE';checkpoint()
        if controls:
            validation=validate_sector_path();write_json(output/'sector_path_validation.json',validation)
            if not validation['all_checks_pass']:raise RuntimeError('Sector path validation failed')
            p=Parameters(Lx=72,Ly=13,Nkz=192)
            w=np.unique(np.r_[np.linspace(.05,.35,41),.09109375,.10578125,.2746875,.28203125])
            etas=(.0075,.015,.03);totals={eta:np.zeros((len(w),2,2),complex) for eta in etas}
            sectors=np.zeros((3,len(w),2,2),complex)
            pairs=192*144**2;estimate=384*pairs+256*1024**2;admission=2*estimate+512*1024**2
            memory=memory_status()
            if memory['available_physical_bytes']<admission:raise MemoryError('All-band one-block memory budget not admitted')
            report['control_memory']=dict(before=memory,estimated_peak_bytes=estimate,admission_required_bytes=admission)
            moments=np.zeros((2,2,2),complex);probe=np.zeros((3,3),complex);count=0;vertex=0.;density=0.
            diag=np.broadcast_to(np.eye(144,dtype=bool),(192,1,144,144)).ravel()
            log(f'Final magnetic grid controls: 26 blocks, {len(w)} frequencies, all 144 bands, eta={etas}')
            report['status']='MAGNETIC_CONTROLS_RUNNING';checkpoint()
            for i,spec in enumerate(iter_magnetic_bloch_spectra(p,1,36,blocks_per_chunk=1)):
                if memory_status()['available_physical_bytes']<admission:raise MemoryError('Available memory fell below fixed all-band admission')
                tr=magnetic_bloch_transitions(spec,.5,.04,8)
                section=sector_response(tr,w,.015,diag);sectors+=section;totals[.015]+=section[0]
                for eta in (.0075,.03):totals[eta]+=fast_density_responses(tr,w,eta)
                moments+=np.array(tr.moments());probe+=tr.full_response(.275+.015j)
                vertex=max(vertex,tr.vertex_error);density+=tr.density_value;count+=tr.delta.size
                del tr,spec,section
                report['completed_blocks']=i+1;report['control_raw_transition_count']=count
                if i==0 or (i+1)%4==0 or i==25:log(f'Control blocks {i+1}/26; elapsed {time.perf_counter()-started:.1f}s');checkpoint()
            q=2*np.pi/24;Q=2*np.sin(q/2);contraction=np.array([.275+.015j,0.,-Q])
            ward=lambda x:float(max(np.max(np.abs(contraction@x)),np.max(np.abs(x@contraction))))
            checks=dict(raw_transition_count=count,expected_raw_transition_count=26*pairs,vertex_error=vertex,
                moment0_max=float(np.max(np.abs(moments[0]))),charge_ward_bare=ward(probe),
                charge_ward_rpa=ward(rpa_response(probe,interaction_matrix(Q,.8,8.))),density=density)
            checks['all_identity_checks_pass']=bool(count==26*pairs and vertex<2e-11 and max(checks['moment0_max'],checks['charge_ward_bare'],checks['charge_ward_rpa'])<2e-10)
            rows=[];summary=[]
            for eta,pi in totals.items():
                for index,(g,ua,remove) in enumerate(CONTROL_SETTINGS):
                    result=analyze(w,pi,q,eta,g,ua,remove)
                    rows+=response_rows(w,pi,result,index,'Bplus_final_mesh')
                    summary.append(dict(case=index,**result['summary']))
            csv_write(output/'eta_controls.csv',rows)
            sector_rows=[]
            for s,label in enumerate(('total','positive_positive','equal_sorted_band_index')):
                for i,freq in enumerate(w):
                    row=dict(sector=label,omega=float(freq),eta=.015)
                    for a in range(2):
                        for b in range(2):row[f'bare_{a}{b}_re']=float(sectors[s,i,a,b].real);row[f'bare_{a}{b}_im']=float(sectors[s,i,a,b].imag)
                    sector_rows.append(row)
            csv_write(output/'transition_sectors.csv',sector_rows)
            report['control_checks']=checks;report['eta_cases']=summary
            report['control_grid']=dict(Nkx=2,Nky=13,Nkz=192,B=2*np.pi/36,q=q,frequency_points=len(w),etas=etas)
            report['sector_definition']=dict(positive_positive='Both eigenenergies >0; includes different positive-energy subbands',
                equal_sorted_band_index='Initial sorted index equals final sorted index; not a degeneracy-invariant band decomposition',
                complements='All other transitions obtained as exact total minus displayed diagnostic sector')
            figure_controls(output,w,totals,sectors,q)
            if not checks['all_identity_checks_pass']:raise RuntimeError('Exact control identity check failed')
            if min(x['minimum_weighted_loss_eigenvalue'] for x in summary)<-2e-10:raise RuntimeError('Weighted passivity failed')
        report['status']='LOW_ENERGY_FINITE_RESPONSE_AUDIT_COMPLETE'
        report['conclusion']='Existing finite-response evidence does not establish a pair of low-energy collective poles or a stable hybrid gap. Loss peaks remain candidates, not certified modes.'
    except Exception as exc:
        report['status']='LOW_ENERGY_AUDIT_FAILED';report['failure']=dict(type=type(exc).__name__,message=str(exc));log(traceback.format_exc())
    report['sources_unchanged']=report['source_sha256']=={x.name:sha(x) for x in sources}
    report['inputs_unchanged']=report['input_sha256']=={str(x.relative_to(ROOT)):sha(x) for x in [*paths.values(),*refs]}
    if not report['sources_unchanged'] or not report['inputs_unchanged']:report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-started
    log('Final status: '+report['status'])
    report['output_sha256']={x.name:sha(x) for x in output.iterdir() if x.is_file() and x.name!='diagnostic.json'}
    checkpoint()
    if report['status']!='LOW_ENERGY_FINITE_RESPONSE_AUDIT_COMPLETE':raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controls',action='store_true')
    args=parser.parse_args();main(args.output.resolve(),args.controls)
