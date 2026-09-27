"""Render the calibration counter-control from frozen profile CSVs; no fitting."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def main():
    output=ROOT/"detector_calibration_figure_reviewed"
    if output.exists() and any(output.iterdir()): raise RuntimeError("Fresh output required")
    output.mkdir(parents=True,exist_ok=True)
    p1=ROOT/"detector_identifiability_audit/profiled_coulomb.csv"
    p2=ROOT/"detector_calibration_followup/calibration_profiles.csv"
    before={str(p.relative_to(ROOT)):sha(p) for p in (p1,p2,Path(__file__).resolve())}
    known,nuisance=read(p1),read(p2)
    fig,axes=plt.subplots(1,3,figsize=(13.2,5.5),sharey=True)
    for ax,dataset,title in zip(axes,("B0","Bplus","joint"),("Zero field","B = 2 pi / 36","Both fields; shared parameters")):
        for rows,mode,label,style in [
          (known,"charge_complex","Charge, calibrated",dict(color="#2369a8",lw=1.4,ls="-",marker="s",mfc="white",ms=4,markevery=(1,6),zorder=3)),
          (known,"charge_and_axial_complex","Charge + axial, calibrated",dict(color="#252525",lw=1.8,ls="-",marker="o",mfc="white",ms=4,markevery=(3,6),zorder=2)),
          (nuisance,"axial_gain_only","Charge + axial, fitted axial gain",dict(color="#c92f39",lw=1.0,ls="--",marker="^",mfc="white",ms=4,markevery=(5,6),zorder=5))
        ]:
            selected=sorted([r for r in rows if r["dataset"]==dataset and r["mode"]==mode],key=lambda r:float(r["ua"]))
            ua=np.array([float(r["ua"]) for r in selected])
            err=np.array([float(r["normalized_mismatch"])*100 for r in selected])
            ax.plot(ua,err,label=label,**style)
        ax.axhline(1,color="#977025",ls=":",lw=1.0,zorder=1)
        ax.axvline(8,color="#a3a3a3",ls=":",lw=.8,zorder=1)
        ax.set_yscale("symlog",linthresh=.05,linscale=.5)
        ax.set_ylim(0,300);ax.set_xlim(0,24)
        ax.set_xticks([0,4,8,12,16,20,24])
        ax.set_yticks([0,.05,.1,1,10,100]);ax.set_yticklabels(["0","0.05","0.1","1","10","100"])
        ax.grid(alpha=.18);ax.set_title(title,fontsize=11)
        ax.set_xlabel("Trial axial interaction Ua")
    axes[0].set_ylabel("Profiled normalized response mismatch (%)")
    axes[2].text(23,1.18,"1% comparison threshold",ha="right",fontsize=8,color="#78551b")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.09),ncol=1,fontsize=9,frameon=False)
    fig.suptitle("Axial amplitude calibration changes interaction distinguishability",fontsize=14,y=.97)
    fig.text(.5,.02,"Truth: g = 0.8, Ua = 8; g profiled over [0.2, 1.6]. 49 trial Ua values; no statistical confidence claim.\n41 complex-frequency samples; eta = 0.015. Gain is positive and shared between fields. Linear scale below 0.05%.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,top=.87,bottom=.32,wspace=.08)
    for ext in ("png","pdf"): fig.savefig(output/("calibration_counter_control."+ext),dpi=180)
    plt.close(fig)
    after={str(p.relative_to(ROOT)):sha(p) for p in (p1,p2,Path(__file__).resolve())}
    assert before==after
    record={"status":"RENDERED_AWAITING_VISUAL_REVIEW","input_source_sha256":before,"raw_profiles_changed":False,"fitting_performed":False,"display":{"yscale":"symlog","linear_threshold_percent":.05,"zero_mismatch_not_floored":True,"legend_overlap_control":"Distinct offset hollow markers; fitted-gain red dashed curve in foreground"}}
    (output/"diagnostic.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Calibration counter-control rendered from frozen profiles")
if __name__=="__main__":main()
