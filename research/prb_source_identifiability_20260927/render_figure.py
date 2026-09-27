"""Publication figure from completed independent-source results."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_study import ROOT, inputs, dress_response, contract_mixed, dump, sha


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    profiles_path = ROOT / "deterministic_v2/profiles.csv"
    noise_path = ROOT / "noise_v1/summary.json"
    profiles = list(csv.DictReader(profiles_path.open(encoding="utf-8")))
    noise = json.loads(noise_path.read_text())
    paths, data = inputs()
    item = data["Bplus"]
    response = dress_response(item["bare"], item["Q"], .8, 8.)
    closure = lambda m: m[:, 0, 1]*m[:, 1, 0]/m[:, 1, 1]
    truth = closure(contract_mixed(response, .7, -.3))
    grid = np.linspace(-1, 1, 81)
    error = np.array([[np.linalg.norm(closure(contract_mixed(response, r, s))-truth)/
                       np.linalg.norm(truth) for r in grid] for s in grid])
    np.savez_compressed(output / "closure_landscape.npz", lambda_grid=grid,
                        relative_closure_error=error)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 6.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 6.15), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=(1.1, 1.))
    ax = fig.add_subplot(layout[0, 0])
    color = ax.contourf(grid, grid, np.log10(np.maximum(error, 1e-4)),
                       levels=np.linspace(-4, 2, 25), cmap="viridis", extend="max")
    ax.plot([.7, -.3], [-.3, .7], "o", ms=6, mfc="none", mec="white", mew=1.3)
    ax.plot([-1, 1], [-1, 1], "--", color="white", alpha=.7, lw=.8)
    ax.set(xlabel=r"Readout direction $\lambda_r$", ylabel=r"Source direction $\lambda_s$",
           title="(a) An unordered source class")
    ax.set_aspect("equal")
    fig.colorbar(color, ax=ax, shrink=.9, label=r"$\log_{10}(\|W-W_*\|_2/\|W_*\|_2)$")
    ax = fig.add_subplot(layout[0, 1])
    policies = [("symmetric", "Bplus", "diagonal", "0.45", ":", "Equal: diagonal $B_+$"),
                ("symmetric", "Bplus", "full", "#2378b5", "-", "Equal: full $B_+$"),
                ("symmetric", "joint", "full", "#07517e", "-", "Equal: full $B_0+B_+$"),
                ("asymmetric", "Bplus", "full", "#d97924", "--", "Unequal: full $B_+$"),
                ("asymmetric", "joint", "full", "#9f4816", "--", "Unequal: full $B_0+B_+$")]
    for truth_name, group, observation, color, style, label in policies:
        rows = [r for r in profiles if (r["truth"], r["group"], r["observation"]) ==
                (truth_name, group, observation)]
        ax.plot([float(r["Ua_fixed"]) for r in rows],
                [float(r["rms_noise_units"]) for r in rows],
                marker="o", ms=2.5, lw=1.2, color=color, linestyle=style, label=label)
    ax.set(xlabel=r"Trial axial interaction $U_a/t$", ylabel="Profile RMS / complex noise RMS",
           title="(b) Independent directions profiled")
    ax.set_yscale("symlog", linthresh=.1)
    ax.axvline(8, color="0.7", lw=.7, zorder=0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="lower right", frameon=False)
    ax = fig.add_subplot(layout[1, :])
    cases = [("symmetric", "Bplus", "full", "known", "$B_+$\nknown"),
             ("symmetric", "Bplus", "full", "shared", "$B_+$\nshared"),
             ("symmetric", "Bplus", "full", "independent", "$B_+$\nindependent"),
             ("symmetric", "joint", "full", "independent", "$B_0+B_+$\nindependent"),
             ("asymmetric", "Bplus", "diagonal", "independent", "$B_+$\ndiagonal"),
             ("asymmetric", "B0", "full", "independent", "$B_0$\nfull"),
             ("asymmetric", "Bplus", "full", "independent", "$B_+$\nfull"),
             ("asymmetric", "joint", "full", "independent", "$B_0+B_+$\nfull")]
    values = []
    for name, group, observation, model, _ in cases:
        record = next(r for r in noise["results"] if (r["truth"], r["group"],
            r["observation"], r["source_model"]) == (name, group, observation, model))
        values.append(record["Ua"]["rmse"])
    ax.bar(np.arange(len(values)), values,
           color=["#2378b5"]*4+["#d97924"]*4, width=.7)
    for index, value in enumerate(values):
        ax.text(index, value*1.09, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(cases)), [case[-1] for case in cases])
    ax.set_yscale("log")
    ax.set_ylim(.012, 1.8)
    ax.set(ylabel=r"Empirical $U_a/t$ RMSE",
           title="(c) Equal directions (blue) and unequal directions (orange); 30 paired realizations")
    ax.axvline(3.5, color="0.75", lw=.8)
    fig.savefig(output / "independent_sources.pdf")
    fig.savefig(output / "independent_sources.png", dpi=200)
    plt.close(fig)
    dump(output / "manifest.json", {str(path): sha(path) for path in
        [Path(__file__), profiles_path, noise_path, *paths.values(),
         output/"independent_sources.pdf", output/"closure_landscape.npz"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
