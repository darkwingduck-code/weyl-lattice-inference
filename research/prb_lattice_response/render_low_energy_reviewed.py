"""Render visibility-reviewed plots exclusively from frozen low-energy CSVs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extend_transverse import sha
ROOT=Path(__file__).resolve().parent

def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))

def selected(data,dataset,case,eta=None):
    return sorted([r for r in data if r.get('dataset')==dataset and int(r['case'])==case
        and (eta is None or float(r['eta'])==eta)],key=lambda r:float(r['omega']))

def values(data,key):return np.array([float(r[key]) for r in data])

def save(fig,output,name):
    for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=180)
    plt.close(fig)

def main(output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('New figure directory required')
    output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'low_energy_mode_audit'
    inputs=[base/name for name in ('cached_controls.csv','eta_controls.csv','transition_sectors.csv','diagnostic.json','independent_summary.json')]
    hashes={str(path.relative_to(ROOT)):sha(path) for path in inputs}
    cached,eta_data,sector_data=[rows(path) for path in inputs[:3]]
    fig,axs=plt.subplots(2,2,figsize=(11.5,8.5),constrained_layout=True)
    colors=['#1765ad','#df6d15','#b11f34']
    for color,eta in zip(colors,[.0075,.015,.03]):
        data=selected(eta_data,'Bplus_final_mesh',2,eta);w=values(data,'omega')
        axs[0,0].plot(w,-values(data,'rpa_00_im'),color=color,label=f'eta={eta}')
        axs[0,1].plot(w,values(data,'minimum_singular_value'),color=color,label=f'eta={eta}')
    sectors={name:sorted([r for r in sector_data if r['sector']==name],key=lambda r:float(r['omega']))
        for name in ('total','positive_positive','equal_sorted_band_index')}
    w=values(sectors['total'],'omega');total=-values(sectors['total'],'bare_00_im')
    positive=-values(sectors['positive_positive'],'bare_00_im');equal=-values(sectors['equal_sorted_band_index'],'bare_00_im')
    ax=axs[1,0]
    ax.plot(w,total,color='#202020',lw=2.2,label='All transitions',zorder=2)
    ax.plot(w,positive,color='#d8750d',ls='--',lw=1.2,marker='o',ms=5,mfc='none',mew=1.1,markevery=(0,5),label='Positive-positive',zorder=4)
    ax.plot(w,equal,color='#128746',ls=':',lw=.9,marker='^',ms=5,mfc='none',mew=1.1,markevery=(2,5),label='Equal sorted band index',zorder=5)
    ax.text(.03,.73,'Curves overlap at this scale.\nOpen markers distinguish their samples.',transform=ax.transAxes,fontsize=9,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    remp=np.abs((total-positive)/total);reme=np.abs((total-equal)/total)
    axs[1,1].semilogy(w,remp,color='#d8750d',ls='--',lw=1.2,marker='o',mfc='none',ms=4,markevery=(0,5),label='Outside positive-positive')
    axs[1,1].semilogy(w,reme,color='#128746',ls=':',lw=1.,marker='^',mfc='none',ms=4,markevery=(2,5),label='Other sorted band indices')
    axs[1,1].text(.03,.08,'Equal-index labels depend on the basis\nnear band degeneracies.',transform=axs[1,1].transAxes,fontsize=9,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    titles=['(a) Charge loss: same final magnetic mesh','(b) Dielectric minimum singular value',
            '(c) Bare charge-loss sectors, eta=.015','(d) Fractional remainder of bare charge loss']
    for ax,title in zip(axs.ravel(),titles):
        ax.set_title(title,fontsize=11);ax.set_xlabel('omega');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    axs[0,1].set_ylim(bottom=0);axs[1,1].set_ylabel('|total - sector| / |total|')
    axs[0,0].set_ylabel('-Im chi_cc');axs[1,0].set_ylabel('-Im Pi_cc')
    fig.suptitle('Low-energy response controls: finite-eta widths do not certify an intrinsic gap',fontsize=13)
    save(fig,output,'low_energy_controls')
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for name,color in [('B0','#1765ad'),('Bplus','#128746')]:
        data=selected(cached,name,2);w=values(data,'omega')
        axs[0,0].plot(w,-values(data,'rpa_00_im'),color=color,label=name+' RPA')
        axs[0,0].plot(w,-values(data,'bare_00_im'),color=color,ls='--',lw=1.,label=name+' bare')
        axs[1,0].plot(w,values(data,'minimum_singular_value'),color=color,label=name)
    for case,color in zip(range(4),['#1765ad','#d8750d','#128746','#b11f34']):
        data=selected(cached,'Bplus',case);w=values(data,'omega')
        axs[0,1].semilogy(w,-values(data,'rpa_11_im'),color=color,label='Ua='+data[0]['axial'])
    full=selected(cached,'Bplus',2);removed=selected(cached,'Bplus',7);w=values(full,'omega')
    yf=-values(full,'rpa_00_im');yr=-values(removed,'rpa_00_im')
    axs[1,1].plot(w,yf,color='#1765ad',lw=2,label='Full two-channel',zorder=2)
    axs[1,1].plot(w,yr,color='#bd252b',ls='--',lw=.9,marker='o',mfc='none',ms=4,markevery=(1,5),
        label='Cross kernel set to zero',zorder=5)
    inset=axs[1,1].inset_axes([.1,.5,.44,.32])
    inset.plot(w,100*(yf-yr)/yf,color='#bd252b',lw=1.)
    inset.set_xlim(.22,.32);inset.set_title('Charge-loss difference (%)',fontsize=8);inset.tick_params(labelsize=7);inset.grid(alpha=.2)
    for ax,title in zip(axs.ravel(),['(a) Charge loss, Ua=8','(b) Axial loss on a log scale, Bplus',
            '(c) Dielectric minimum singular value','(d) Mixing control with visible difference']):
        ax.set_title(title,fontsize=11);ax.set_xlabel('omega');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in (axs[0,0],axs[0,1],axs[1,1]):ax.set_xlim(.025,.4)
    axs[1,0].set_ylim(bottom=0)
    fig.suptitle('Interaction controls from frozen response matrices',fontsize=13)
    save(fig,output,'low_energy_interaction_controls')
    report=dict(status='FIGURES_GENERATED_FROM_FROZEN_CSV',source_sha256={Path(__file__).name:sha(Path(__file__))},
        input_sha256=hashes,inputs_unchanged=hashes=={str(path.relative_to(ROOT)):sha(path) for path in inputs},
        numeric_data_modified=False,visibility_changes=['Hollow staggered markers and thin foreground dashed lines for coincident sector curves',
            'Separate fractional-remainder panel; explicit basis/degeneracy caveat',
            'Log-scale axial interaction control; percent-difference inset for mixing control'],
        output_sha256={path.name:sha(path) for path in output.iterdir() if path.is_file()})
    (output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(report['status']);print(output)
    if not report['inputs_unchanged']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
