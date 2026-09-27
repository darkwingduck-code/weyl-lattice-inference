"""Readable comparisons of the frozen parameter profiles; no fitting here."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(data,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError("New figure directory required")
    out.mkdir(parents=True,exist_ok=True)
    files=[data/n for n in ("profiles.csv","local_sensitivity.csv","tolerance_sets.json","groups.json","diagnostic.json")]
    hashes={str(p.resolve()):sha(p) for p in [Path(__file__).resolve(),*files]}
    with files[0].open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    def selected(group,window,mode):
        a=[r for r in rows if r["group"]==group and r["window"]==window and r["mode"]==mode]
        a.sort(key=lambda r:float(r["ua"]))
        if len(a)!=49:raise ValueError("Incomplete profile")
        return np.array([float(r["ua"]) for r in a]),100*np.array([float(r["normalized_mismatch"]) for r in a]),a
    policies=[
        ("charge_only","Charge only","#A77900",":","^",(0,8),1.25,2),
        ("calibrated","Both calibrated","#0072B2","-","o",(2,8),1.35,3),
        ("separate_axial_gains","Separate axial gains","#555555","--","s",(4,8),1.4,4),
        ("shared_axial_gain","Shared axial gain","#C62828",(0,(2,1.5)),"D",(6,8),.95,6)]
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":10})
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.4),layout="constrained")
    summaries=[]
    for ax,(group,title) in zip(axes,[("field_pair_baseline","Two fields, baseline"),
            ("q_scan_both","Two fields, three q values"),("mu_scan_both","Two fields, three chemical potentials")]):
        certified=True
        for mode,label,color,ls,marker,every,lw,zorder in policies:
            x,y,a=selected(group,"fixed_lab",mode)
            ax.plot(x,y,color=color,ls=ls,marker=marker,markevery=every,mfc="white",
                    ms=4.7,lw=lw,zorder=zorder,label=label)
            certified &= all(r["all_declared_grid_comparisons_pass"]=="True" for r in a)
            r=next(r for r in a if float(r["ua"])==4.)
            summaries.append(dict(group=group,window="fixed_lab",mode=mode,ua=4.,
                mismatch_percent=100*float(r["normalized_mismatch"]),best_g=float(r["best_g"]),
                gains=json.loads(r["axial_gains"]),all_declared_grid_comparisons_pass=certified))
        ax.axhline(1.,color="#888888",lw=.7,zorder=0)
        ax.set_yscale("symlog",linthresh=.01);ax.set_ylim(bottom=0)
        ax.set_title(title+("\n" if certified else "\nGrid-comparison limits apply"),fontsize=10)
        ax.set_xlabel("Trial axial interaction Ua");ax.set_xticks([0,4,8,12,16,20,24])
        ax.grid(alpha=.12);ax.set_ylabel("Channel/dataset relative L2 RMS (%)")
    axes[0].legend(fontsize=8,loc="upper left")
    fig.suptitle("Calibration assumptions govern combined-data discrimination",fontsize=13)
    for ext in ("png","pdf"):fig.savefig(out/("combined_calibration."+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.2),layout="constrained")
    for ax,mode,title in zip(axes,("shared_axial_gain","separate_axial_gains"),("Shared gain across conditions","Independent gain per condition")):
        for window,label,color,ls,marker,offset,zorder in [
                ("fixed_lab","Fixed laboratory window","#0072B2","-","o",1,3),
                ("scaled_q","Window scaled with q","#C62828",(0,(2,1.5)),"D",5,6)]:
            x,y,a=selected("q_scan_both",window,mode)
            ax.plot(x,y,color=color,ls=ls,marker=marker,markevery=(offset,8),mfc="white",ms=4.8,
                    lw=.95 if window=="scaled_q" else 1.4,zorder=zorder,label=label)
        ax.axhline(1.,color="#888888",lw=.7,zorder=0);ax.set_yscale("symlog",linthresh=.01)
        ax.set_ylim(bottom=0);ax.set_xlabel("Trial axial interaction Ua")
        ax.set_ylabel("Channel/dataset relative L2 RMS (%)");ax.set_title(title,fontsize=10)
        ax.set_xticks([0,4,8,12,16,20,24]);ax.grid(alpha=.12);ax.legend(fontsize=8)
    fig.suptitle("Two fields and three q values: observation-window control",fontsize=12)
    for ext in ("png","pdf"):fig.savefig(out/("frequency_window_control."+ext),dpi=180)
    plt.close(fig)
    for p,d in hashes.items():
        if sha(Path(p))!=d:raise RuntimeError("Source/data changed")
    result=dict(status="FIGURES_RENDERED_REQUIRES_VISUAL_REVIEW",source_input_sha256=hashes,
        selected_ua4_comparisons=summaries,figure_scope="Explicit finite-grid calculations, not confidence regions",
        visibility={"red_series":"Thin dashed line on foreground; markers staggered along Ua samples",
            "zero_values":"Preserved on symmetric-log axes; 0.01 percent linear region, no artificial positive floor"})
    (out/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();main(a.data.resolve(),a.output.resolve())
