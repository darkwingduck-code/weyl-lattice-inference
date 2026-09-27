"""Visible separate scales for own-mesh and shared-reference profile margins."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def run():
 inputs={name:ROOT/'higher_profile_comparison'/name for name in ('own_mesh_truth_profiles.csv','shared_higher_truth_profiles.csv')}
 hashes={str(p):sha(p) for p in inputs.values()}
 own=rows(inputs['own_mesh_truth_profiles.csv']);shared=rows(inputs['shared_higher_truth_profiles.csv'])
 out=ROOT/'reviewed_figures'
 if out.exists():raise FileExistsError('Preserve any previous renderer output')
 out.mkdir()
 targets=[('Bplus_eta_low_Ua1',1.,'B=2pi/36; eta=0.0075',1.,['72x13x96','72x13x192','72x13x384']),
 ('B0_qscan_Ua20',20.,'B=0; three q, fixed lab window',.1,['48x48x192','72x72x384','96x96x768']),
 ('B0_qdouble_scaled_Ua17',17.,'B=0; double q, scaled window',.1,['48x48x192','72x72x384','96x96x768'])]
 fig,axes=plt.subplots(2,3,figsize=(12.9,7.4),layout='constrained')
 plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
 for col,(target,ua,title,threshold,labels) in enumerate(targets):
  for row,data in enumerate((own,shared)):
   ax=axes[row,col]
   ax.axhline(0,color='.3',lw=.8,ls=':',zorder=0,label='Declared threshold')
   for candidate,color,marker,style,label in [(ua,'#0072B2' if row==0 else '#D55E00','o' if row==0 else 's','-',f'Hypothesis Ua={ua:g}')]+(
       [(8.,'#009E73','^','--','True Ua=8 control')] if row else []):
    selected=[next(r for r in data if r['target']==target and float(r['ua'])==candidate and r['model_level']==level) for level in ('coarse','fine','higher')]
    values=[100*(float(r['normalized_mismatch'])-float(r['threshold'])) for r in selected]
    ax.plot(range(3),values,color=color,marker=marker,mfc='none',mew=1.3,ms=7,lw=1.1,ls=style,zorder=3,label=label)
   ax.set_xticks(range(3),labels,fontsize=8)
   ax.set_xlabel('Lx x Ly x Nkz',fontsize=9)
   ax.set_ylabel('RMS - threshold (percentage points)',fontsize=9)
   kind='Own-mesh truth' if row==0 else 'Shared higher-mesh truth'
   ax.set_title(f'{kind}\n{title}; limit={threshold:g}%',fontsize=10)
   ax.legend(fontsize=7,framealpha=.9)
   ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
   ax.grid(axis='y',lw=.4,alpha=.25)
 fig.suptitle('Separate reference choices and vertical scales; below zero meets only the declared RMS threshold',fontsize=11)
 fig.savefig(out/'threshold_reference_controls.png',dpi=180)
 fig.savefig(out/'threshold_reference_controls.pdf')
 plt.close(fig)
 if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('Numerical input changed')
 report={'status':'RENDERED_FROM_FROZEN_PROFILES','source_sha256':sha(Path(__file__)),
  'input_sha256':hashes,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
  'transform':'100*(RMS-threshold); each panel own vertical scale; no numerical evidence changed',
  'visual_inspection_performed_by_script':False}
 (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':run()