"""Reflow the existing Matplotlib plots without changing a plotted data value."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'paper/prb_submission/render_figures.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def traces(fig):
 return [[{'x':np.asarray(l.get_xdata(),dtype=float).tolist(),'y':np.asarray(l.get_ydata(),dtype=float).tolist()} for l in ax.lines] for ax in fig.axes]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'paper/prb_strengthened/editorial_figures');a=ap.parse_args()
 out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('frozen_prb_figures',ORIGINAL);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 original_sha=sha(ORIGINAL);old.HERE=out;records={}
 plt.rcParams.update({'font.size':9,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
 def save(fig,name):
  before=traces(fig);axes=fig.axes
  for leg in list(fig.legends):leg.remove()
  for text in list(fig.texts):text.remove()
  if name in ('figure1_uv','figure5_eta'):
   fig.set_size_inches(6.8,4.1)
   gs=GridSpec(2,2,figure=fig,left=.11,right=.975,bottom=.12,top=.82 if name=='figure1_uv' else .90,wspace=.38,hspace=.70)
   for i,ax in enumerate(axes):ax.set_position(gs[i//2,i%2].get_position(fig))
   if name=='figure1_uv':
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.52,1.005),ncol=2,frameon=False)
   else:
    for annotation in list(axes[2].texts):annotation.remove()
    fig.text(.52,.96,r'Finite-$\eta$ widths do not certify a gap',ha='center',fontsize=9)
  elif name=='figure2_qwindows':
   fig.set_size_inches(6.8,4.45)
   gs=GridSpec(2,3,figure=fig,left=.105,right=.985,bottom=.17,top=.88,wspace=.48,hspace=.88)
   for i,ax in enumerate(axes):
    case,field=divmod(i,2);ax.set_position(gs[field,case].get_position(fig))
    q=[r'$q_0/2$',r'$q_0$',r'$2q_0$'][case];b=r'$B=0$' if field==0 else r'$B_+$'
    ax.set_title(f'({chr(97+3*field+case)}) {b}, $q$={q}\nMesh: '+('fail' if field==1 and case==2 else 'pass'),fontsize=8.5,loc='left',pad=6)
    ax.set_ylabel(r'$-\mathrm{Im}\,\chi_{cc}$ ($t^{-1}$)' if case==0 else '')
    ax.set_xlabel(r'$\omega$ ($t$)')
   h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.51,-.005),ncol=3,frameon=False)
  elif name=='figure3_calibration':
   fig.set_size_inches(6.8,3.1)
   gs=GridSpec(1,3,figure=fig,left=.09,right=.985,bottom=.19,top=.72,wspace=.43)
   names=['Two fields; baseline',r'Three $q$ values',r'Three $\mu$ values']
   for i,ax in enumerate(axes):
    ax.set_position(gs[0,i].get_position(fig));ax.set_title(f'({chr(97+i)}) '+names[i]+('\nSome input meshes fail' if i else ''),fontsize=8.5,loc='left',pad=7)
    ax.set_xlabel(r'Trial $U_a$ ($t$)');ax.set_ylabel('Relative L2 RMS (%)' if i==0 else '')
    ax.set_xticks([0,4,8,12,16,20,24])
   h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.52,1.0),ncol=2,frameon=False)
  elif name=='figure4_reference':
   fig.set_size_inches(6.8,4.55)
   gs=GridSpec(2,3,figure=fig,left=.09,right=.985,bottom=.13,top=.78,wspace=.51,hspace=.85)
   labels=[r'A: $B_+$, low $\eta$',r'B: $B=0$, three $q$',r'C: $B=0$, scaled $2q_0$']
   for i,ax in enumerate(axes):
    case,ref=divmod(i,2);ax.set_position(gs[ref,case].get_position(fig))
    ax.set_title(f'({chr(97+3*ref+case)}) '+labels[case]+'\n'+('Own-mesh reference' if ref==0 else 'Common higher reference'),loc='left',fontsize=8.1,pad=7)
    ax.set_ylabel('RMS - limit (pp)' if case==0 else '');ax.set_xlabel('Model mesh');ax.set_xticks([0,1,2],['Coarse','Fine','Higher'])
   from matplotlib.lines import Line2D
   h=[Line2D([],[],color='#0072b2',marker='o',mfc='none',lw=1,label='Own: trial interaction'),Line2D([],[],color='#d55e00',marker='s',mfc='none',lw=1,label='Common: trial interaction'),Line2D([],[],color='#009e73',marker='^',mfc='none',ls='--',lw=1,label=r'Common: true $U_a=8$')]
   fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,fontsize=8)
  else:raise ValueError(name)
  after=traces(fig);assert before==after,'Plot data changed'
  fig.canvas.draw()
  for ext in ('pdf','png'):fig.savefig(out/(name+'.'+ext),dpi=180)
  clips=[]
  with fitz.open(out/(name+'.pdf')) as pdf:
   page=pdf[0]
   for block in page.get_text('dict')['blocks']:
    for line in block.get('lines',[]):
     for span in line['spans']:
      bb=fitz.Rect(span['bbox'])
      if bb.x0<-.5 or bb.y0<-.5 or bb.x1>page.rect.width+.5 or bb.y1>page.rect.height+.5:clips.append({'text':span['text'],'bounds':list(bb)})
  records[name]={'size_inches':fig.get_size_inches().tolist(),'trace_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'trace_count':sum(len(x) for x in before),'all_trace_values_unchanged':True,'text_outside_canvas':clips}
  plt.close(fig)
 old.save=save
 for fun in (old.uv,old.qwindows,old.calibration,old.reference,old.eta):fun()
 assert sha(ORIGINAL)==original_sha
 assert all(sha(ROOT/n)==h for n,h in old.USED.items())
 report={'status':'PLOTS_REFLOWED_FROM_FROZEN_DATA','source_sha256':sha(__file__),'original_plot_source_sha256':original_sha,'input_sha256':old.USED,'all_inputs_unchanged':True,'new_fits_or_resampling':False,'figures':records,'output_sha256':{p.name:sha(p) for p in out.glob('figure*.*')}}
 (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'output':str(out),'figures':records},indent=2))
if __name__=='__main__':main()
