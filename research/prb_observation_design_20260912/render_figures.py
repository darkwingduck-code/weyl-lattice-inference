"""Publication-size figures and transparent summaries of frozen CSV evidence."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'

def read(rel):return list(csv.DictReader((ROOT/rel).open(encoding='utf-8')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    inputs=['validation/full_profiles.csv','validation/noise_replicas.csv','validation/nonfactorizable_controls.csv','validation/common_higher_joint_bias.csv','validation/raw_matrix_hypotheses.csv','nuisance_robustness/frequency_gain_raw_profiles.csv','nuisance_robustness/weak_signal_noise.csv','nuisance_robustness/baseline_same_reference_grid.csv','pilot/signal_amplitudes.csv']
    h={p:sha(ROOT/p) for p in inputs};p,n,v,b,raw,free,weak,base,amp=[read(f) for f in inputs]
    groups=collections.defaultdict(list)
    for r in n:groups[(r['dataset'],r['noise_kind'],r['sigma_multiplier'],r['method'])].append(r)
    for r in weak:
        q={**r,'dataset':'B0_mu_low','noise_kind':'absolute','sigma_multiplier':'0.01'}
        groups[('B0_mu_low','absolute','0.01',r['method'])].append(q)
    summary=[]
    for k,rows in groups.items():
        u=np.array([float(r['ua']) for r in rows]);g=np.array([float(r['g']) for r in rows])
        summary.append(dict(dataset=k[0],noise_kind=k[1],sigma_multiplier=float(k[2]),method=k[3],replicas=len(rows),ua_mean=float(u.mean()),ua_bias=float(u.mean()-8),ua_rmse=float(np.sqrt(np.mean((u-8)**2))),ua_min=float(u.min()),ua_max=float(u.max()),ua_p05=float(np.quantile(u,.05)),ua_p95=float(np.quantile(u,.95)),ua_boundary_hits=int(np.sum((u<1e-4)|(u>23.9999))),g_rmse=float(np.sqrt(np.mean((g-.8)**2))),optimizer_failed=sum(r['success']!='True' for r in rows)))
    with (OUT/'noise_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    plt.rcParams.update({'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    C={'diag_separate_complex':'#0072B2','diag_separate_tilt':'#333333','closure':'#D55E00','cross_product':'#CC3311'}
    fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
    styles=[('diag_separate_complex','Complex gain','--','o',1.2),('diag_separate_tilt','Complex affine gain',':','s',.8),('closure',r'$\chi_{cc}$ and $W$','-','D',1.3),('cross_product','Cross-multiplied residual','--','+',.7)]
    for j,(policy,label,ls,mark,lw) in enumerate(styles):
        rr=[r for r in p if r['dataset']=='Bplus_baseline' and r['window']=='fixed_lab' and r['policy']==policy]
        axs[0].plot([float(r['ua']) for r in rr],[100*float(r['rms']) for r in rr],label=label,color=C[policy],ls=ls,marker=mark,mfc='none',ms=4,markevery=(j,7),lw=lw,zorder=2+j)
    axs[0].set(xlabel=r'$U_a/t$',ylabel='Block-normalized RMS (%)',title=r'(a) $B=2\pi/36,\ \mu/t=0.5$')
    axs[0].set_yscale('symlog',linthresh=.01);axs[0].set_ylim(0,3000);axs[0].legend(loc='upper right',framealpha=.9)
    for j,(dataset,label,color) in enumerate([('Bplus_baseline',r'$B=2\pi/36,\ \mu/t=0.5$','#D55E00'),('B0_mu_low',r'$B=0,\ \mu/t=0.25$','#0072B2')]):
        for k,(rows,kind,marker) in enumerate([(raw,'Affine transfer','o'),(free,'Transfer at every frequency','s')]):
            rr=[r for r in rows if r['dataset']==dataset and float(r['ua_trial']) in (4.,16.) and (r.get('degree','1')=='1')]
            axs[1].plot([0 if float(r['ua_trial'])==4 else 1 for r in rr],[float(r['rms']) for r in rr],color=color,ls='-' if k==0 else '--',marker=marker,mfc='none',ms=5,lw=.8,zorder=2+k,label=(label if k==0 else None))
    axs[1].axhline(1,color='.4',ls=':',lw=.8);axs[1].text(.5,1.15,'One-noise-unit guide',ha='center',fontsize=8)
    axs[1].set(xticks=[0,1],xticklabels=[r'$U_a/t=4$',r'$U_a/t=16$'],ylabel='Raw matrix RMS / noise scale',title='(b) Same absolute noise floor')
    axs[1].set_yscale('log');axs[1].set_ylim(.25,30);axs[1].legend(loc='upper right',fontsize=7.8)
    axs[1].text(.5,.28,'Circles: affine; squares: independent frequency',ha='center',fontsize=7.5)
    for ext in ('pdf','png'):fig.savefig(OUT/f'observation_profiles.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
    for dataset,label,col in [('Bplus_baseline',r'$B=2\pi/36,\ \mu/t=0.5$','#D55E00'),('B0_mu_low',r'$B=0,\ \mu/t=0.25$','#0072B2')]:
        for method,ls,marker in [('raw_matrix_affine','-','o'),('cross_product','--','s')]:
            rr=sorted([r for r in summary if r['dataset']==dataset and r['noise_kind']=='absolute' and r['method']==method],key=lambda r:r['sigma_multiplier'])
            axs[0].loglog([r['sigma_multiplier'] for r in rr],[r['ua_rmse'] for r in rr],color=col,ls=ls,marker=marker,mfc='none',lw=1,ms=4,label=label if method=='raw_matrix_affine' else None)
    axs[0].set(xlabel=r'$\sigma/\mathrm{RMS}(\chi_{aa})$',ylabel=r'Empirical RMSE of $U_a/t$',title='(a) 100 noise replicas per condition')
    axs[0].legend(loc='upper left');axs[0].text(.00013,.0006,'Solid: raw matrix; dashed: cross product',fontsize=7.7)
    vals=[]
    keys=[('Bplus_baseline','absolute','0.001','raw_matrix_affine'),('Bplus_baseline','absolute','0.01','raw_matrix_affine'),('B0_mu_low','absolute','0.001','raw_matrix_affine'),('B0_mu_low','absolute','0.01','raw_matrix_affine')]
    for k in keys:vals.append([float(r['ua']) for r in groups[k]])
    axs[1].boxplot(vals,tick_labels=[r'$B_+,10^{-3}$',r'$B_+,10^{-2}$',r'$B_0,10^{-3}$',r'$B_0,10^{-2}$'],whis=(0,100),widths=.5,medianprops={'color':'#CC3311'},boxprops={'linewidth':.9})
    axs[1].axhline(8,color='.3',ls=':',lw=.8);axs[1].set(ylabel=r'Recovered $U_a/t$',title='(b) Full range, affine matrix estimator',ylim=(0,25))
    axs[1].text(4,21,'17/100 at\nupper bound',ha='center',fontsize=8)
    for ext in ('pdf','png'):fig.savefig(OUT/f'observation_noise.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
    sel=[base[0]]+[r for r in b if r['window']=='fixed_lab' and r['model_level']=='fine' and r['dataset'] in ('B0_baseline','B0_qdouble','Bplus_eta_low')]
    axs[0].bar(np.arange(len(sel)),[abs(float(r['ua_bias'])) for r in sel],color=['#D55E00','#0072B2','#0072B2','#D55E00'],width=.6)
    axs[0].set_yscale('log');axs[0].set(xticks=range(len(sel)),xticklabels=[r'$B_+,q_0$',r'$B_0,q_0$',r'$B_0,2q_0$',r'$B_+,\eta/2$'],ylabel=r'$|\widehat U_a-U_a|/t$',title='(a) Empirical mesh bias, common data')
    axs[0].text(.02,.98,'First: coarse model / fine data\nOthers: fine model / higher data',transform=axs[0].transAxes,va='top',fontsize=7.8)
    for j,(dataset,label,col) in enumerate([('Bplus_baseline',r'$B_+,\mu/t=0.5$','#D55E00'),('B0_mu_low',r'$B_0,\mu/t=0.25$','#0072B2')]):
        for kind,ls,marker in [('ca_entry_gain','--','s'),('aa_to_ca_leakage','-','o')]:
            rr=[r for r in v if r['dataset']==dataset and r['perturbation']==kind and float(r['amplitude'])>0]
            axs[1].plot([100*float(r['amplitude']) for r in rr],[float(r['ua_bias']) for r in rr],color=col,ls=ls,marker=marker,mfc='none',lw=1,ms=4,label=label if kind=='aa_to_ca_leakage' else None)
    axs[1].set(xlabel='Deliberate transfer error (%)',ylabel=r'Bias of $U_a/t$',title='(b) Factorization is an assumption')
    axs[1].legend(loc='lower left');axs[1].text(.04,.12,'Solid: axial leakage into cross entry\nDashed: one-entry gain error',transform=axs[1].transAxes,fontsize=7.7)
    for ext in ('pdf','png'):fig.savefig(OUT/f'observation_limits.{ext}',dpi=220)
    plt.close(fig)
    meta=dict(status='FIGURES_AND_SUMMARIES_COMPLETE',source_sha256=sha(Path(__file__)),input_sha256=h,outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},figsize_inches=[6.7,3.0],font_base_points=8.5,noise_distribution_not_confidence_interval=True)
    (OUT/'diagnostic.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()