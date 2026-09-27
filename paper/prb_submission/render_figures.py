"""Manuscript-sized figures from frozen CSVs; no scientific fitting or resampling."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent/"journal_figures_final"
HERE.mkdir(exist_ok=True)
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"research/prb_lattice_response"
USED={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(relative):
 p=DATA/relative;USED[str(p.relative_to(ROOT))]=sha(p)
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def vals(r,key):return np.array([float(x[key]) for x in r])
def ordered(r):return sorted(r,key=lambda x:float(x["omega"]))
def common(ax,title,y):
 ax.set_title(title,loc="left",fontsize=9.2,pad=7)
 ax.set_xlabel(r"Frequency $\omega$ ($t$)");ax.set_ylabel(y);ax.grid(alpha=.16,lw=.5)
def save(fig,name):
 for ext in ("pdf","png"):fig.savefig(HERE/f"{name}.{ext}",dpi=200)
 plt.close(fig)
def uv():
 raw=rows("uv_matching_audit/Bplus_zero_real_reference/responses.csv")
 data={m:ordered([x for x in raw if x["method"]==m]) for m in {x["method"] for x in raw}}
 fig,axes=plt.subplots(2,2,figsize=(6.6,5.8))
 fig.subplots_adjust(left=.12,right=.98,bottom=.09,top=.84,wspace=.36,hspace=.52)
 styles=[("full","Full","#222222","-",None,0,1.7,2),("naive_low_bare_U",r"Positive-positive; bare $U$","#c62828","--",None,0,1.1,3),
 ("Ueff_only",r"Effective $U$ only","#8456a6",":","s",3,1.0,4),("matched_constant","Frozen full matching","#087d62","--","o",1,.8,6)]
 for name,label,color,ls,marker,offset,lw,z in styles:
  r=data[name];w=vals(r,"omega")
  for ax,key,transform in [(axes[0,0],"00",lambda a:-a.imag),(axes[0,1],"11",lambda a:-a.imag),(axes[1,0],"01",np.abs)]:
   a=vals(r,f"response_{key}_re")+1j*vals(r,f"response_{key}_im")
   ax.plot(w,transform(a),label=label,color=color,ls=ls,lw=lw,marker=marker,mfc="none",ms=4,markevery=(offset,6),zorder=z)
 for key,label,col,marker in [("00","cc","#1765ad","o"),("11","aa","#c62828","s"),("01","ca","#087d62","^")]:
  f=data["full"];a=data["matched_constant"];w=vals(f,"omega")
  yf=vals(f,f"response_{key}_re")+1j*vals(f,f"response_{key}_im")
  ya=vals(a,f"response_{key}_re")+1j*vals(a,f"response_{key}_im")
  axes[1,1].semilogy(w,abs(ya-yf)/max(abs(yf)),label=label,color=col,lw=1,marker=marker,mfc="none",ms=3.5,markevery=7)
 for ax,title,y in zip(axes.flat,["(a) Charge loss","(b) Axial loss","(c) Mixed response magnitude","(d) Frozen relative complex error"],[r"$-\mathrm{Im}\,\chi_{cc}$ ($t^{-1}$)",r"$-\mathrm{Im}\,\chi_{aa}$ ($t^{-1}$)",r"$|\chi_{ca}|$ ($t^{-1}$)",r"$|\chi^{\mathrm{fr}}-\chi|/\max|\chi|$"]):common(ax,title,y)
 axes[1,1].legend(loc="lower right",framealpha=.95)
 h,l=axes[0,0].get_legend_handles_labels()
 fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
 save(fig,"figure1_uv")
def qwindows():
 fig,axes=plt.subplots(3,2,figsize=(6.6,7.6))
 fig.subplots_adjust(left=.12,right=.98,bottom=.15,top=.91,wspace=.36,hspace=.73)
 for i,(case,qtext) in enumerate([("qhalf",r"$q_0/2$"),("baseline",r"$q_0$"),("qdouble",r"$2q_0$")]):
  for j,field in enumerate(("B0","Bplus")):
   ax=axes[i,j];r=ordered([x for x in rows(f"parameter_sector_sampling/{field}_final/{case}.csv") if float(x["omega"])>0])
   w=vals(r,"omega");ax.axvspan(.05,.35,color="#dce5ed",alpha=.45,zorder=0)
   for prefix,label,color,ls,marker,offset,lw,z in [("rpa","Full","#222222","-",None,0,1.6,2),("naive",r"Positive-positive; bare $U$","#c62828","--",None,0,1.0,3),("matched","Frozen matching","#087d62","--","o",2,.8,5)]:
    ax.plot(w,-vals(r,f"{prefix}_00_im"),label=label,color=color,ls=ls,lw=lw,marker=marker,mfc="none",ms=3.5,markevery=(offset,7),zorder=z)
   status="fail" if field=="Bplus" and case=="qdouble" else "pass"
   common(ax,f"({chr(97+2*i+j)}) {'$B=0$' if j==0 else '$B=2π/36$'}, $q$={qtext}\nMesh comparison: {status}",r"$-\mathrm{Im}\,\chi_{cc}$ ($t^{-1}$)")
   ax.set_xlim(float(min(w)),float(max(w)));ax.tick_params(axis="x",pad=2)
 h,l=axes[0,0].get_legend_handles_labels()
 fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,.003),ncol=2,frameon=False)
 save(fig,"figure2_qwindows")
def calibration():
 data=rows("parameter_calibration_audit/run_01/profiles.csv")
 fig,axes=plt.subplots(3,1,figsize=(6.6,7.6))
 fig.subplots_adjust(left=.13,right=.97,bottom=.08,top=.86,hspace=.51)
 policies=[("charge_only","Charge only","#a77900",":","^",0,1.2,2),("calibrated","Both calibrated","#0072b2","-","o",2,1.2,3),
 ("separate_axial_gains","Separate axial gains","#555555","--","s",4,1.2,4),("shared_axial_gain","Shared axial gain","#c62828","--","D",6,.8,6)]
 for i,(ax,group,title) in enumerate(zip(axes,["field_pair_baseline","q_scan_both","mu_scan_both"],["Two fields; baseline",r"Two fields; three $q$ values","Two fields; three chemical potentials"])):
  highest=0
  for mode,label,color,ls,marker,offset,lw,z in policies:
   r=sorted([x for x in data if x["group"]==group and x["window"]=="fixed_lab" and x["mode"]==mode],key=lambda x:float(x["ua"]))
   assert len(r)==49
   x=vals(r,"ua");y=100*vals(r,"normalized_mismatch");highest=max(highest,max(y))
   ax.plot(x,y,label=label,color=color,ls=ls,lw=lw,marker=marker,mfc="white",ms=4,markevery=(offset,8),zorder=z)
  ax.axhline(1,color=".5",ls=":",lw=.8,zorder=0)
  ax.set_yscale("symlog",linthresh=.01);ax.set_ylim(0,highest*1.25)
  ax.set_xticks([0,4,8,12,16,20,24]);ax.set_xlabel(r"Trial axial interaction $U_a$ ($t$)" if i==2 else "")
  ax.set_ylabel("Relative L2 RMS (%)");ax.grid(alpha=.15,lw=.5)
  ax.set_title(f"({chr(97+i)}) {title}"+("\nSome input mesh comparisons fail" if i else ""),loc="left",fontsize=9.2,pad=6)
 h,l=axes[0].get_legend_handles_labels()
 fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
 save(fig,"figure3_calibration")

def reference():
 own=rows("detector_boundary_refinement/higher_profile_comparison/own_mesh_truth_profiles.csv")
 shared=rows("detector_boundary_refinement/higher_profile_comparison/shared_higher_truth_profiles.csv")
 targets=[("Bplus_eta_low_Ua1",1,r"A: $B=2\pi/36$; $\eta=0.0075$",1,["72x13x96","72x13x192","72x13x384"]),
 ("B0_qscan_Ua20",20,r"B: $B=0$; three $q$; fixed window",.1,["48x48x192","72x72x384","96x96x768"]),
 ("B0_qdouble_scaled_Ua17",17,r"C: $B=0$; $2q_0$; scaled window",.1,["48x48x192","72x72x384","96x96x768"])]
 fig,axes=plt.subplots(3,2,figsize=(6.6,8.0))
 fig.subplots_adjust(left=.12,right=.98,bottom=.075,top=.81,wspace=.39,hspace=.76)
 for i,(target,ua,title,limit,labels) in enumerate(targets):
  for j,data in enumerate((own,shared)):
   ax=axes[i,j];ax.axhline(0,color=".4",ls=":",lw=.8,zorder=0)
   candidates=[(ua,"#0072b2" if j==0 else "#d55e00","o" if j==0 else "s","-",f"Trial $U_a$={ua:g}")]
   if j:candidates.append((8,"#009e73","^","--",r"True $U_a=8$"))
   for u,col,marker,ls,label in candidates:
    rr=[next(x for x in data if x["target"]==target and float(x["ua"])==u and x["model_level"]==level) for level in ("coarse","fine","higher")]
    ax.plot(range(3),[100*(float(x["normalized_mismatch"])-float(x["threshold"])) for x in rr],color=col,lw=1.1,marker=marker,mfc="none",mew=1,ms=5,ls=ls,zorder=4,label=label)
   ax.set_xticks(range(3),["Coarse","Final","Higher"]);ax.set_xlim(-.14,2.14)
   ax.set_xlabel("Model mesh");ax.set_ylabel("RMS - limit (pp)")
   ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3),useMathText=True)
   ax.grid(axis="y",alpha=.16,lw=.5)
   ax.set_title(f"({chr(97+2*i+j)}) {title}\nLimit {limit:g}%; $U_a$={ua:g}",loc="left",fontsize=8.6,pad=8)
 # Figure-level legends never obscure points. Left/right headings encode reference.
 fig.text(.29,.984,"Own-mesh reference",ha="center",fontsize=9.5,weight="bold")
 fig.text(.77,.984,"Common higher reference",ha="center",fontsize=9.5,weight="bold")
 from matplotlib.lines import Line2D
 handles=[Line2D([],[],color="#0072b2",marker="o",mfc="none",lw=1,label=r"Own: trial $U_a$"),
 Line2D([],[],color="#d55e00",marker="s",mfc="none",lw=1,label=r"Common: trial $U_a$"),
 Line2D([],[],color="#009e73",marker="^",mfc="none",ls="--",lw=1,label=r"Common: true $U_a=8$")]
 fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.949),ncol=3,frameon=False,fontsize=8.2,handlelength=1.8,columnspacing=1)
 save(fig,"figure4_reference")
def eta():
 raw=rows("low_energy_mode_audit/eta_controls.csv")
 sector=rows("low_energy_mode_audit/transition_sectors.csv")
 fig,axes=plt.subplots(2,2,figsize=(6.6,5.8))
 fig.subplots_adjust(left=.12,right=.98,bottom=.09,top=.89,wspace=.37,hspace=.57)
 for e,col,ls in [(.0075,"#1765ad","-"),(.015,"#df6d15","--"),(.03,"#b11f34",":")]:
  rr=ordered([x for x in raw if x["dataset"]=="Bplus_final_mesh" and int(x["case"])==2 and float(x["eta"])==e]);w=vals(rr,"omega")
  axes[0,0].plot(w,-vals(rr,"rpa_00_im"),color=col,ls=ls,lw=1.2,label=fr"$\eta={e}$")
  axes[0,1].plot(w,vals(rr,"minimum_singular_value"),color=col,ls=ls,lw=1.2)
 ss={name:ordered([x for x in sector if x["sector"]==name]) for name in ("total","positive_positive","equal_sorted_band_index")}
 w=vals(ss["total"],"omega");total=-vals(ss["total"],"bare_00_im")
 for name,label,col,ls,marker,offset,lw,z in [("total","All","#222222","-",None,0,1.6,2),("positive_positive","Positive-positive","#d8750d","--","o",0,.85,4),("equal_sorted_band_index","Equal index","#128746",":","^",2,.8,5)]:
  y=-vals(ss[name],"bare_00_im")
  axes[1,0].plot(w,y,color=col,ls=ls,lw=lw,marker=marker,mfc="none",ms=4,markevery=(offset,5),label=label,zorder=z)
  if name!="total":axes[1,1].semilogy(w,abs((total-y)/total),color=col,ls=ls,lw=lw,marker=marker,mfc="none",ms=3.5,markevery=(offset,5),label=label)
 axes[0,0].legend(loc="upper left",framealpha=.95)
 axes[1,0].legend(loc="upper left",framealpha=.95)
 axes[1,0].text(.025,.57,"Curves nearly overlap",transform=axes[1,0].transAxes,fontsize=8.5)
 axes[1,1].legend(loc="upper left",framealpha=.95)
 for ax,title,y in zip(axes.flat,["(a) Charge loss","(b) Dielectric singular value","(c) Bare charge-loss sectors","(d) Sector remainder"],[r"$-\mathrm{Im}\,\chi_{cc}$ ($t^{-1}$)",r"$\sigma_{\min}(I-\Pi U)$",r"$-\mathrm{Im}\,\Pi_{cc}$ ($t^{-1}$)",r"$|L_{\mathrm{tot}}-L_s|/|L_{\mathrm{tot}}|$"]):common(ax,title,y)
 axes[0,1].set_ylim(bottom=0)
 fig.text(.5,.965,r"Finite-$\eta$ widths do not certify a gap",ha="center",fontsize=10)
 save(fig,"figure5_eta")
def main():
 if (HERE/"diagnostic.json").exists():raise FileExistsError("Preserve rendered manuscript figures; use a new version folder.")
 plt.rcParams.update({"font.size":9,"axes.labelsize":9,"axes.titlesize":9.2,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,"pdf.fonttype":42,"ps.fonttype":42,"axes.formatter.use_mathtext":False})
 for fun in (uv,qwindows,calibration,reference,eta):fun()
 assert all(sha(ROOT/p)==h for p,h in USED.items())
 import fitz
 sizes={}
 for p in HERE.glob("figure*.pdf"):
  doc=fitz.open(p);normal=[]
  for page in doc:
   for block in page.get_text("dict")["blocks"]:
    for line in block.get("lines",[]):
     for span in line["spans"]:
      if span["text"].strip():normal.append({"size_pt":span["size"],"text":span["text"]})
  sizes[p.name]={"page_width_in":doc[0].rect.width/72,"min_text_pt":min(x["size_pt"] for x in normal),"smallest_spans":sorted(normal,key=lambda x:x["size_pt"])[:8]}
 report={"status":"MANUSCRIPT_FIGURES_FROM_FROZEN_CSV","source_sha256":sha(Path(__file__)),"input_sha256":USED,"inputs_unchanged":True,"data_modified":False,"nominal_width_in":6.6,"font_audit":sizes,"output_sha256":{p.name:sha(p) for p in HERE.glob("figure*.*")},"visual_review":"pending"}
 (HERE/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({k:v for k,v in report.items() if k not in ("input_sha256","output_sha256")},indent=2))
if __name__=="__main__":main()
