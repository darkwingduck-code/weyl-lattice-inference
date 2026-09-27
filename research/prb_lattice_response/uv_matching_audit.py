"""Detector-resolved all-band versus positive-positive response matching.

The matrix elimination identity is standard algebra, not a new result. The
excluded sector is defined by energy-sign labels, not a high-frequency cutoff.
A background frozen at z_ref keeps that declared complex response; it is not
silently replaced by its real part or called an eta->0 static limit.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from low_energy_mode_audit import loss_peaks
from extend_transverse import sha,csv_write
from run_pole_zero_experiment import write_json
ROOT=Path(__file__).resolve().parent
METHODS=('full','naive_low_bare_U','matched_dynamic','matched_constant','Ueff_only',
         'constant_background_no_vertices','constant_vertices_no_background')


def solve_response(pi,u):
    # Ueff is in general complex, frequency-dependent, and non-Hermitian.
    # The bare Hermitian-interaction validator must not be applied to Ueff.
    return np.linalg.solve(np.eye(2)-pi@u,pi)


def match_response(low,excluded,u):
    left=np.linalg.inv(np.eye(2)-excluded@u)
    right=np.linalg.inv(np.eye(2)-u@excluded)
    background=left@excluded
    effective=u+u@background@u
    dressed_low=solve_response(low,effective)
    return dict(response=background+left@dressed_low@right,background=background,
                left_vertex=left,right_vertex=right,effective_interaction=effective,
                dressed_low=dressed_low)


def read_sectors(path):
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    result={}
    for name in ('total','positive_positive'):
        selected=sorted([r for r in rows if r['sector']==name],key=lambda r:float(r['omega']))
        omega=np.array([float(r['omega']) for r in selected])
        matrix=np.array([[[complex(float(r[f'bare_{a}{b}_re']),float(r[f'bare_{a}{b}_im'])) for b in range(2)] for a in range(2)] for r in selected])
        result[name]=(omega,matrix)
    if not np.array_equal(result['total'][0],result['positive_positive'][0]):raise ValueError('Sector frequency mismatch')
    return result['total'][0],result['total'][1],result['positive_positive'][1]


def method_responses(total,low,u,reference_excluded):
    records={name:[] for name in METHODS}
    dynamic=[];constant=[]
    for t,l in zip(total,low):
        d=match_response(l,t-l,u);c=match_response(l,reference_excluded,u)
        dynamic.append(d);constant.append(c)
        values=dict(full=solve_response(t,u),naive_low_bare_U=solve_response(l,u),matched_dynamic=d['response'],
            matched_constant=c['response'],Ueff_only=c['dressed_low'],
            constant_background_no_vertices=c['background']+c['dressed_low'],
            constant_vertices_no_background=c['left_vertex']@c['dressed_low']@c['right_vertex'])
        for name,value in values.items():records[name].append(value)
    return {name:np.array(value) for name,value in records.items()},dynamic,constant


def detector_metrics(omega,truth,prediction):
    result={}
    for name,a,b in (('cc',0,0),('aa',1,1),('ca',0,1),('ac',1,0)):
        ref,got=truth[:,a,b],prediction[:,a,b]
        amplitude=float(np.max(np.abs(ref)));imaginary=float(np.max(np.abs(ref.imag)))
        absolute=float(np.max(np.abs(got-ref)));imagerr=float(np.max(np.abs(got.imag-ref.imag)))
        item=dict(maximum_complex_absolute_error=absolute,
            normalized_complex_sup_error=absolute/amplitude if amplitude else None,
            maximum_imaginary_absolute_error=imagerr,
            normalized_imaginary_sup_error=imagerr/imaginary if imaginary else None,
            normalization='One detector-specific supremum over the entire declared frequency window, not pointwise relative error')
        if a==b:
            peaks=loss_peaks(omega,-got.imag)
            item['all_sampled_loss_peaks']=peaks
            item['principal_sampled_loss_peak']=max(peaks,key=lambda p:p['height']) if peaks else None
        result[name]=item
    return result


def analyze_dataset(omega,total,low,q,eta,reference_excluded,reference):
    u=np.diag([.8/(2*np.sin(q/2))**2,8.])
    responses,dynamic,constant=method_responses(total,low,u,reference_excluded)
    high=total-low
    metrics={name:detector_metrics(omega,responses['full'],value) for name,value in responses.items()}
    exact=float(np.max(np.abs(responses['matched_dynamic']-responses['full'])))
    budget=512*np.finfo(float).eps*max(1.,float(np.max(np.abs(responses['full']))))
    report=dict(q=q,eta=eta,coulomb=.8,axial=8.,frequency_min=float(omega.min()),frequency_max=float(omega.max()),frequency_points=len(omega),
        reference=reference,reference_excluded=reference_excluded,reference_matching=match_response(low[0],reference_excluded,u),
        excluded_axial_real_range=[float(high[:,1,1].real.min()),float(high[:,1,1].real.max())],
        excluded_frequency_variation_max=float(np.max(np.abs(high-reference_excluded))),
        metrics=metrics,dynamic_identity_max_error=exact,dynamic_identity_roundoff_budget=budget,dynamic_identity_pass=bool(exact<=budget),
        collective_poles_or_gap_certified=False,
        method_scope='Exact matrix factorization is an algebra check. Constant-background accuracy and detector distortion are empirical results in this finite-eta window.')
    rows=[]
    for method,matrix in responses.items():
        for i,w in enumerate(omega):
            row=dict(method=method,omega=float(w),eta=eta)
            for prefix,value in [('response',matrix[i]),('full_bare',total[i]),('low_bare',low[i]),('excluded_bare',high[i]),
                                 ('dynamic_Ueff',dynamic[i]['effective_interaction'])]:
                for a in range(2):
                    for b in range(2):row[f'{prefix}_{a}{b}_re']=float(value[a,b].real);row[f'{prefix}_{a}{b}_im']=float(value[a,b].imag)
            rows.append(row)
    return report,rows,responses


def plot_case(output,name,omega,responses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    styles={'full':dict(color='black',lw=2.3,zorder=2),
        'naive_low_bare_U':dict(color='#c32b30',ls='--',lw=1.2),
        'Ueff_only':dict(color='#8157a5',ls=':',lw=1.2),
        'matched_constant':dict(color='#087d62',ls='--',lw=.9,marker='o',mfc='none',ms=4,markevery=(1,5),zorder=5)}
    for method,style in styles.items():
        chi=responses[method]
        axs[0,0].plot(omega,-chi[:,0,0].imag,label=method,**style)
        axs[0,1].plot(omega,-chi[:,1,1].imag,label=method,**style)
        axs[1,0].plot(omega,np.abs(chi[:,0,1]),label=method,**style)
    for detector,a,b,color in [('cc',0,0,'#1765ad'),('aa',1,1,'#c32b30'),('ca',0,1,'#087d62')]:
        scale=np.max(np.abs(responses['full'][:,a,b]))
        err=np.abs(responses['matched_constant'][:,a,b]-responses['full'][:,a,b])/scale
        axs[1,1].semilogy(omega,err,label=detector,color=color)
    for ax,title in zip(axs.ravel(),['Charge loss','Axial loss','Magnitude of cross response','Frozen-background complex error / full sup']):
        ax.set_title(title);ax.set_xlabel('omega');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(name+': all-band response and declared background matching')
    for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=170)
    plt.close(fig)


def main(input_csv,output,static_csv=None,label='Bplus_cached'):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh evidence output required')
    output.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/x for x in ('lattice_response.py','low_energy_mode_audit.py','extend_transverse.py','run_pole_zero_experiment.py','uv_matching_audit.py','test_uv_matching_audit.py')]
    inputs=[input_csv]+([static_csv] if static_csv else [])
    source_hash={p.name:sha(p) for p in sources};input_hash={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    omega,total,low=read_sectors(input_csv)
    eta=.015;q=2*np.pi/24
    positive=omega>0
    if static_csv:
        sw,st,sl=read_sectors(static_csv);index=np.where(sw==0.)[0]
        if len(index)!=1:raise ValueError('One exact zero-real-frequency reference required')
        reference=st[index[0]]-sl[index[0]]
        ref=dict(real_frequency=0.,eta=eta,definition='z_ref=i*eta at finite q and finite temperature; not the eta->0 static limit')
    else:
        reference=total[0]-low[0]
        ref=dict(real_frequency=float(omega[0]),eta=eta,definition='First sampled complex-frequency background, not a zero-frequency or static limit')
    omega,total,low=omega[positive],total[positive],low[positive]
    report,rows,responses=analyze_dataset(omega,total,low,q,eta,reference,ref)
    alternatives=[]
    for wanted in (.05,.2,.35):
        i=int(np.argmin(abs(omega-wanted)))
        alt,_,_=analyze_dataset(omega,total,low,q,eta,total[i]-low[i],dict(real_frequency=float(omega[i]),eta=eta,definition='Frozen finite-frequency sensitivity control'))
        alternatives.append(dict(reference=alt['reference'],metrics=alt['metrics']['matched_constant']))
    report.update(status='UV_MATCHING_AUDIT_COMPLETE' if report['dynamic_identity_pass'] else 'UV_MATCHING_ALGEBRA_FAILED',label=label,
        reference_sensitivity=alternatives,source_sha256=source_hash,input_sha256=input_hash,
        source_unchanged=source_hash=={p.name:sha(p) for p in sources},inputs_unchanged=input_hash=={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        retained_sector_definition='Both initial and final eigenenergies >0, with original finite-T occupations and vertices. The excluded complement may contain low-energy transitions.',
        limitations=['No new matrix-identity theorem is claimed.','Separate-sector equal-time moments need not vanish.',
            'Effective interaction is a full complex matrix. External source vertices and the additive excluded response are retained.',
            'Finite-eta detector agreement does not certify poles, gaps, or experimental accessibility of the local axial source.'])
    csv_write(output/'responses.csv',rows);plot_case(output,label,omega,responses)
    report['output_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'diagnostic.json',report)
    print(report['status']);print('Dynamic identity error:',report['dynamic_identity_max_error'])
    for method in ('naive_low_bare_U','Ueff_only','matched_constant'):
        print(method,{name:report['metrics'][method][name]['normalized_complex_sup_error'] for name in ('cc','aa','ca')})
    if not report['dynamic_identity_pass'] or not report['source_unchanged'] or not report['inputs_unchanged']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-reference',type=Path);parser.add_argument('--label',default='Bplus_cached')
    a=parser.parse_args();main(a.input.resolve(),a.output.resolve(),a.static_reference.resolve() if a.static_reference else None,a.label)
