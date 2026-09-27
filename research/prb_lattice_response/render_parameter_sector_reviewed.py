"""Reviewed figure layout: shared outside legend, no annotation across peaks.

The first renderer and figures are preserved. All numeric evidence is unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_parameter_sector_outputs import read
from extend_transverse import sha
from run_pole_zero_experiment import write_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES=('baseline','qhalf','qdouble','mu_low','mu_high','eta_low','eta_high')
LABELS=('base','q/2','2q','mu=.25','mu=.75','eta/2','2eta')


def render(directory,output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh figure directory required')
    output.mkdir(parents=True,exist_ok=True)
    index=json.loads((directory/'calibration_inputs.json').read_text(encoding='utf-8'))
    lookup={x['id']:x for x in index['datasets']}
    inputs={str(directory/'calibration_inputs.json'):sha(directory/'calibration_inputs.json')}
    fig,axs=plt.subplots(2,3,figsize=(12.5,7.5))
    fig.subplots_adjust(left=.07,right=.985,top=.83,bottom=.13,hspace=.42,wspace=.28)
    for ir,field in enumerate(('B0','Bplus')):
        for ic,case in enumerate(('qhalf','baseline','qdouble')):
            item=lookup[field+'_'+case];path=directory/item['csv'];rows,values=read(path)
            inputs[str(path)]=sha(path);omega=np.array([float(r['omega']) for r in rows]);keep=omega>0
            ax=axs[ir,ic]
            for prefix,label,style in (
                ('rpa','full',dict(color='#191919',lw=2.0,zorder=2)),
                ('naive','positive-only, bare U',dict(color='#c92a30',lw=1.2,ls='--',zorder=3)),
                ('matched','frozen background',dict(color='#00816e',lw=.75,ls=(0,(2,3)),marker='o',mfc='none',ms=4,
                                                    markevery=(2,7),zorder=5))):
                ax.plot(omega[keep],-values[prefix][keep,0,0].imag,label=label,**style)
            fixed=item['fixed_lab_frequencies'];ax.axvspan(min(fixed),max(fixed),color='#dce5ec',alpha=.35,zorder=0)
            ax.set_title(f"{field}, {case}; mesh check: {item['convergence_status']}",fontsize=10)
            ax.set_xlabel('omega');ax.set_ylabel('-Im chi_cc');ax.grid(alpha=.18)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.008),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Finite-eta charge response: fixed lab and q-scaled windows sampled together\nShading: fixed lab window. Hollow markers expose overlapping matched curves.\nSampled peaks are not pole or gap certificates.',fontsize=12,y=.99)
    for ext in ('png','pdf'):fig.savefig(output/f'q_window_responses.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7.7),constrained_layout=True)
    for ir,field in enumerate(('B0','Bplus')):
        for ic,detector in enumerate(('cc','aa','ca')):
            ax=axs[ir,ic];x=np.arange(7)
            for method,window,label,offset,color,marker in (
                ('naive_low_bare_U','fixed_lab','naive: fixed',-.24,'#c92a30','x'),
                ('matched_constant','fixed_lab','matched: fixed',-.08,'#087d62','o'),
                ('naive_low_bare_U','scaled_q','naive: scaled',.08,'#a85198','+'),
                ('matched_constant','scaled_q','matched: scaled',.24,'#2468b4','s')):
                vals=[]
                for case in CASES:
                    item=lookup[field+'_'+case];mp=directory/item['csv'];mp=mp.with_name(mp.stem+'_metadata.json')
                    meta=json.loads(mp.read_text(encoding='utf-8'));inputs[str(mp)]=sha(mp)
                    vals.append(meta['metrics'][window][method][detector]['normalized_complex_sup_error'])
                ax.semilogy(x+offset,vals,ls='none',marker=marker,color=color,mfc='none',ms=5,label=label)
            labels=[label+('*' if lookup[field+'_'+case]['convergence_status']!='pass' else '') for label,case in zip(LABELS,CASES)]
            ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_title(field+' / '+detector)
            ax.set_ylabel('max |approx-full| / max |full|');ax.grid(alpha=.2,axis='y')
            ax.legend(fontsize=7)
    fig.suptitle('Detector-specific complex sup errors; displaced markers keep identical windows visible\n* = recorded mesh comparison fails or is not certified; pass is limited to that comparison',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(output/f'matching_scope_errors.{ext}',dpi=170)
    plt.close(fig)
    write_json(output/'diagnostic.json',dict(status='PARAMETER_FIGURES_RENDERED',source_sha256={
        Path(__file__).name:sha(Path(__file__)), 'audit_parameter_sector_outputs.py':sha(Path(__file__).with_name('audit_parameter_sector_outputs.py'))},
        input_sha256=inputs,output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        notes=['Data unchanged. Hollow foreground markers distinguish overlap.',
               'No analytic pole/gap claim. Convergence labels refer only to the recorded comparison.']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.directory.resolve(),a.output.resolve())
