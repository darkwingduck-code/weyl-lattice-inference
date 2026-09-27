"""Render fixed-budget results and sampling-uncertainty diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fixed_budget import PLAN, dump, sha, write_csv


def run(input_dir, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new figure/report directory")
    summary = json.loads((input_dir / "summary.json").read_text())
    samples = [row for i in range(4) for row in json.loads(
        (input_dir / f"fits_{i}.json").read_text())["results"]]
    output.mkdir(parents=True, exist_ok=True)
    order = ["B0_only", "Bplus_25pct", "Bplus_50pct", "Bplus_75pct", "Bplus_only"]
    fractions = [0., .25, .5, .75, 1.]
    diagnostics, losses = [], {}
    for row in summary["summaries"]:
        selected = sorted((r for r in samples if r["truth"] == row["truth"] and
            r["policy"] == row["policy"]), key=lambda r:r["repetition"])
        sq = np.array([(r["Ua"]-8.)**2 for r in selected])
        losses[(row["truth"], row["policy"])] = sq
        se = np.std(sq, ddof=1)/np.sqrt(len(sq))/(2*row["Ua_rmse"])
        diagnostics.append(dict(**row, rmse_monte_carlo_se_delta_method=float(se)))
    paired = []
    for truth in PLAN["truths"]:
        for policy in order:
            for baseline in ("B0_only", "Bplus_only"):
                delta = losses[(truth, policy)]-losses[(truth, baseline)]
                paired.append(dict(truth=truth, policy=policy, baseline=baseline,
                    mean_squared_error_difference=float(delta.mean()),
                    paired_monte_carlo_se=float(np.std(delta, ddof=1)/np.sqrt(len(delta)))))
    write_csv(output / "sampling_diagnostics.csv", diagnostics)
    write_csv(output / "paired_loss_differences.csv", paired)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
    for ax, truth, color in zip(axes, PLAN["truths"], ("#176b9e", "#ba5714")):
        rows = [next(r for r in diagnostics if r["truth"] == truth and r["policy"] == p) for p in order]
        ax.errorbar(fractions, [r["Ua_rmse"] for r in rows],
            yerr=[r["rmse_monte_carlo_se_delta_method"] for r in rows],
            color=color, marker="o", capsize=3, label="Empirical RMSE (30 repeats)")
        ax.plot(fractions, [r["Ua_local_sd"] for r in rows], "--", color="0.25", label="Local Fisher SD")
        ax.set_title("Equal directions" if truth == "symmetric" else "Unequal directions")
        ax.set_xlabel("Fraction of total time at finite field")
        ax.set_ylabel(r"Interaction error / $t$")
        ax.set_xticks(fractions)
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "fixed_budget.pdf")
    fig.savefig(output / "fixed_budget.png", dpi=200)
    plt.close(fig)
    lines = ["# Fixed total acquisition budget", "",
        "The experiment fixes one total acquisition time across fields. Within a field, "
        "time is distributed equally over the same four entries and 45 frequencies. "
        "A field receiving fraction f has per-entry complex noise RMS sigma0/sqrt(f), "
        "where sigma0=1.2931669440932318e-4. Equal independent white detector noise and "
        "zero field-switching overhead are declared assumptions. Operator directions are "
        "shared across fields; affine complex gains are fitted separately in each field.", "",
        "| Truth | Policy | Ua RMSE | Local Fisher SD | Approx. MC SE of RMSE |",
        "|---|---|---:|---:|---:|"]
    for r in diagnostics:
        lines.append(f"| {r['truth']} | {r['policy']} | {r['Ua_rmse']:.6f} | {r['Ua_local_sd']:.6f} | {r['rmse_monte_carlo_se_delta_method']:.6f} |")
    lines += ["", f"The empirical best policies on this five-point grid are {summary['empirical_best_by_truth']}. "
        f"The smallest worst-truth RMSE is obtained by {summary['empirical_minimax_policy']}. "
        "These are selections among the declared policies and two truth pairs, not an "
        "optimization over all time allocations or a uniform robustness theorem.", "",
        "The local covariance is the inverse interaction Fisher matrix after projecting "
        "both affine gains and the two source directions, with both g and Ua fitted. "
        "Physical coordinates are (g/.8, Ua/8); the exact circular-complex Gaussian "
        "factor 2 is included. Local Fisher SD is not a finite-sample coverage guarantee.", "",
        f"Retained optimizer successes: {sum(r['success_count'] for r in diagnostics)}/300; "
        f"bound hits: {sum(r['bound_count'] for r in diagnostics)}; "
        f"feasible-truth objective failures: {sum(r['truth_upper_bound_failures'] for r in diagnostics)}; "
        f"unsuccessful alternative starts: {sum(r['unsuccessful_starts'] for r in diagnostics)}/1500. "
        "All estimates and every start are retained, including failures.", "",
        "Error bars are one approximate Monte Carlo standard error of empirical RMSE "
        "(delta method from 30 squared errors), not parameter confidence intervals. "
        "Paired squared-error differences against both single-field baselines are "
        "retained separately. Small differences should not be assigned a robust ranking "
        "from this limited repeated sample.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [Path(__file__), input_dir / "summary.json", input_dir / "information.json"]
    paths += [input_dir / f"fits_{i}.json" for i in range(4)]
    dump(output / "manifest.json", {str(p): sha(p) for p in paths})
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve())
