"""Render a clarified control figure from immutable observability CSV data."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def main():
    source=ROOT/"zero_observability_controls/finite_eta_local_minima.csv"
    rows=list(csv.DictReader(source.open(encoding="utf-8")))
    out=ROOT/"zero_observability_figure_reviewed"
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout="constrained")
    styles=[(0.,"#0072B2","o","-",1.8,2),(.8,"#009E73","s","--",1.25,3),(1.6,"#D55E00","^",":",.95,4)]
    for g,col,mark,ls,lw,zorder in styles:
        rr=sorted([r for r in rows if r["index"]=="1565" and float(r["g"])==g],key=lambda r:float(r["eta"]))
        xx=[float(r["eta"]) for r in rr];yy=[float(r["abs_chi_at_real_zero"]) for r in rr]
        axes[0].loglog(xx,yy,color=col,marker=mark,mfc="none",ms=8,lw=lw,ls=ls,zorder=zorder,label=f"g = {g:g}")
        yy=[float(r["minimum_shift"]) for r in rr]
        axes[1].semilogx(xx,yy,color=col,lw=lw,ls=ls,zorder=zorder)
        for r,x,y in zip(rr,xx,yy):
            edge=r["classification"].startswith("WINDOW_EDGE")
            axes[1].plot(x,y,color=col,marker="x" if edge else mark,mfc="none",ms=8,ls="none",zorder=zorder+3)
    rr=sorted([r for r in rows if r["index"]=="1565" and float(r["g"])==0],key=lambda r:float(r["eta"]))
    axes[0].loglog([float(r["eta"]) for r in rr],[float(r["leading_eta_prediction"]) for r in rr],"k--",lw=.8,label=r"$\eta |\chi'_{cc}(z_0)|$",zorder=1)
    axes[0].set(xlabel=r"Regulator $\eta$",ylabel=r"$|\chi_{cc}(z_0+i\eta)|$",title="Regular zero at z0 = 0.777148")
    bound=float(rr[0]["search_lower"])-float(rr[0]["zero"])
    axes[1].axhline(bound,color=".5",lw=.8,ls="--")
    axes[1].annotate("x: best sampled candidate lies at boundary",xy=(.0017,bound),xytext=(.000017,-.00175),fontsize=9,
                     arrowprops={"arrowstyle":"->","color":".4"},bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    axes[1].set(xlabel=r"Regulator $\eta$",ylabel=r"Selected local-window candidate $-z_0$",title="Broadening obscures the real zero")
    axes[1].axhline(0,color=".7",lw=.7,ls=":")
    axes[0].legend(fontsize=9)
    fig.savefig(out/"zero_observability.pdf");fig.savefig(out/"zero_observability.png",dpi=170);plt.close(fig)
    corrected=[]
    for row in rows:
        item=dict(row);item["source_classification"]=item["classification"]
        item["classification"]="BOUNDARY_BEST_EVALUATED_CANDIDATE" if item["classification"].startswith("WINDOW_EDGE") else "INTERIOR_OPTIMIZER_CANDIDATE"
        item["optimizer_success_recorded"]=False;item["interior_extrema_existence_certified"]=False
        corrected.append(item)
    with (out/"clarified_finite_eta_candidates.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(corrected[0]));writer.writeheader();writer.writerows(corrected)
    (out/"interpretation_correction.txt").write_text(
        "The original bounded optimizer compared two endpoints and one fitted candidate.\\n"
        "Its status WINDOW_EDGE_NO_INTERIOR_MINIMUM is too strong: it does not prove absence of other interior extrema.\\n"
        "The clarified table records boundary-selected or interior-optimizer candidates only.\\n"
        "Optimizer success was not recorded in the original run; no certified local/global minimum is claimed.\\n"
        "All original CSV data and source are preserved. No physical zero or pole finding is altered.\\n",encoding="utf-8")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={"status":"RENDERED_CLASSIFICATION_AWARE_FIGURE","input_sha256":{str(source.relative_to(ROOT)):sha(source)},
            "source_sha256":{Path(__file__).name:sha(Path(__file__))},
            "output_sha256":{p.name:sha(p) for p in out.iterdir()},"finite_eta_is_not_physical_damping":True,
            "scope":"Index1565 only; x points are search-window endpoints, not certified local minima or detected antiresonances."}
    (out/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
