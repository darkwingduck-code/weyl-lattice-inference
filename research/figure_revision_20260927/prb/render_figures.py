"""Presentation-only revisions of PRB Figures 4 and 8 from frozen plot data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCREENING = REPO / "research/prb_screening_information_20260927"
SOURCE = REPO / "research/prb_source_identifiability_20260927"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.canvas.draw()
    clipped = []
    hidden_ticks = set()
    for axis in fig.axes:
        for ticks, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in ticks.get_major_ticks()+ticks.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    hidden_ticks.update((tick.label1, tick.label2))
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text() or artist in hidden_ticks:
            continue
        bounds = artist.get_window_extent(fig.canvas.get_renderer())
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
            clipped.append(artist.get_text())
    assert not clipped, clipped
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=220)
    return {"figure_inches": fig.get_size_inches().tolist(), "visible_text_clipping": clipped}


def screening(output):
    # Same rows, normalization, colors, styles, scales and limits as the frozen
    # make_normalized_figure.py. Only marker presentation and legend change.
    rows = read_csv(SCREENING / "run_02/information.csv")
    methods = ("full_calibrated", "full_profiled", "axial_calibrated", "axial_profiled")
    labels = ("All entries, calibrated", "All entries, gains profiled",
              "Axial auto, calibrated", "Axial auto, gain profiled")
    styles = (("#1f77b4", "-"), ("#1f77b4", "--"), ("#d95f02", "-."), ("#d95f02", ":"))
    markers = ("^", "s", "o", "D")
    fig, axes = plt.subplots(1, 2, figsize=(8., 3.3), sharey=True, layout="constrained")
    checks = []
    for axis, field in zip(axes, ("B0", "Bplus")):
        subset = [r for r in rows if r["field"] == field]
        u = np.array([float(r["Ua"]) for r in subset])
        assert len(u) == 81 and u[0] == 8.
        for index, (name, label, (color, style), marker) in enumerate(zip(methods, labels, styles, markers)):
            values = np.array([float(r[name]) for r in subset])
            normalized = values/values[0]
            line, = axis.loglog(u, normalized, label=label, color=color, linestyle=style,
                               linewidth=1.1, marker=marker, markersize=4.7,
                               markerfacecolor="white", markeredgewidth=1.05,
                               markevery=(3*index, 12), zorder=2+index)
            np.testing.assert_array_equal(line.get_xdata(), u)
            np.testing.assert_array_equal(line.get_ydata(), normalized)
            checks.append(dict(field=field, method=name, marker=marker,
                               markevery=[3*index, 12], unchanged_plotted_samples=len(u)))
        axis.set(title=field, xlabel=r"$U_a$ (fixed-$\Pi$ continuation)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel(r"$I_{U_aU_a}(U_a)/I_{U_aU_a}(8)$")
    axes[1].legend(fontsize=7, loc="lower left", handlelength=3.2, numpoints=1)
    fig.suptitle("Fixed absolute noise; local Gaussian benchmark", fontsize=10)
    result = save(fig, output, "screening_information_normalized")
    result["data_checks"] = checks
    plt.close(fig)
    return result


def independent_sources(output):
    # Plotting statements follow the frozen render_figure.py. The closure grid
    # is loaded from its retained output, with no susceptibility recalculation.
    profiles = read_csv(SOURCE / "deterministic_v2/profiles.csv")
    noise = json.loads((SOURCE / "noise_v1/summary.json").read_text(encoding="utf-8"))
    with np.load(SOURCE / "figures_v2/closure_landscape.npz") as saved:
        grid = saved["lambda_grid"].copy()
        error = saved["relative_closure_error"].copy()
    assert grid.shape == (81,) and error.shape == (81, 81)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        # Fixed panel slots avoid a constrained-layout/axes-divider interaction
        # that can move labels when PDF and PNG backends draw sequentially.
        fig = plt.figure(figsize=(7.2, 6.15))
        source_axis = fig.add_axes([.085, .565, .335, .375])
        color = source_axis.contourf(grid, grid, np.log10(np.maximum(error, 1e-4)),
                                    levels=np.linspace(-4, 2, 25), cmap="viridis", extend="max")
        source_axis.plot([.7, -.3], [-.3, .7], "o", ms=6, mfc="none", mec="white", mew=1.3)
        source_axis.plot([-1, 1], [-1, 1], "--", color="white", alpha=.7, lw=.8)
        source_axis.set(xlabel=r"Readout direction $\lambda_r$", ylabel=r"Source direction $\lambda_s$",
                        title="(a) An unordered source class")
        source_axis.set_aspect("equal")
        # A dedicated axis tracks the square panel's actual right edge.
        divider = make_axes_locatable(source_axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=.06)
        fig.colorbar(color, cax=colorbar_axis,
                     label=r"$\log_{10}(\|W-W_*\|_2/\|W_*\|_2)$")
        axis = fig.add_axes([.65, .55, .335, .395])
        profile_axis = axis
        policies = [("symmetric", "Bplus", "diagonal", "0.45", ":", "Equal: diagonal $B_+$"),
                    ("symmetric", "Bplus", "full", "#2378b5", "-", "Equal: full $B_+$"),
                    ("symmetric", "joint", "full", "#07517e", "-", "Equal: full $B_0+B_+$"),
                    ("asymmetric", "Bplus", "full", "#d97924", "--", "Unequal: full $B_+$"),
                    ("asymmetric", "joint", "full", "#9f4816", "--", "Unequal: full $B_0+B_+$")]
        profile_points = []
        for truth, group, observation, color, style, label in policies:
            selected = [r for r in profiles if (r["truth"], r["group"], r["observation"]) == (truth, group, observation)]
            axis.plot([float(r["Ua_fixed"]) for r in selected], [float(r["rms_noise_units"]) for r in selected],
                      marker="o", ms=2.5, lw=1.2, color=color, linestyle=style, label=label)
            profile_points.append(len(selected))
        axis.set(xlabel=r"Trial axial interaction $U_a/t$", ylabel="Profile RMS / complex noise RMS",
                 title="(b) Independent directions profiled")
        axis.set_yscale("symlog", linthresh=.1)
        axis.axvline(8, color="0.7", lw=.7, zorder=0)
        axis.set_xticks([0, 4, 8, 12, 16, 20, 24])
        axis.legend(loc="lower right", frameon=False)
        axis = fig.add_axes([.085, .095, .9, .34])
        cases = [("symmetric", "Bplus", "full", "known", "$B_+$\nknown"),
                 ("symmetric", "Bplus", "full", "shared", "$B_+$\nshared"),
                 ("symmetric", "Bplus", "full", "independent", "$B_+$\nindependent"),
                 ("symmetric", "joint", "full", "independent", "$B_0+B_+$\nindependent"),
                 ("asymmetric", "Bplus", "diagonal", "independent", "$B_+$\ndiagonal"),
                 ("asymmetric", "B0", "full", "independent", "$B_0$\nfull"),
                 ("asymmetric", "Bplus", "full", "independent", "$B_+$\nfull"),
                 ("asymmetric", "joint", "full", "independent", "$B_0+B_+$\nfull")]
        values = []
        for truth, group, observation, model, _ in cases:
            row = next(r for r in noise["results"] if (r["truth"], r["group"], r["observation"], r["source_model"]) ==
                       (truth, group, observation, model))
            values.append(row["Ua"]["rmse"])
        axis.bar(np.arange(len(values)), values, color=["#2378b5"]*4+["#d97924"]*4, width=.7)
        for index, value in enumerate(values):
            axis.text(index, value*1.09, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        axis.set_xticks(np.arange(len(cases)), [case[-1] for case in cases])
        axis.set_yscale("log")
        axis.set_ylim(.012, 1.8)
        axis.set(ylabel=r"Empirical $U_a/t$ RMSE",
                 title="(c) Equal directions (blue) and unequal directions (orange); 30 paired realizations")
        axis.axvline(3.5, color="0.75", lw=.8)
        result = save(fig, output, "independent_sources")
        renderer = fig.canvas.get_renderer()
        colorbar_labels = [colorbar_axis.yaxis.label, *colorbar_axis.get_yticklabels()]
        profile_labels = [profile_axis.yaxis.label, *profile_axis.get_yticklabels()]
        for left in colorbar_labels:
            for right in profile_labels:
                assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)), (left.get_text(), right.get_text())
        gap_inches = (colorbar_axis.get_position().x0-source_axis.get_position().x1)*fig.get_figwidth()
        assert abs(gap_inches-.06) < 1e-8, gap_inches
        result.update(colorbar_panel_gap_points=72*gap_inches,
                      retained_grid_shape=list(error.shape), retained_profile_points=profile_points,
                      retained_rmse_values=values)
        plt.close(fig)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory beneath this renderer's directory")
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("Output must be a new child directory of research/figure_revision_20260927/prb")
    inputs = [SCREENING / "run_02/information.csv", SCREENING / "make_normalized_figure.py",
              SOURCE / "deterministic_v2/profiles.csv", SOURCE / "noise_v1/summary.json",
              SOURCE / "figures_v2/closure_landscape.npz", SOURCE / "render_figure.py"]
    before = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figures = {"Figure 4": screening(output), "Figure 8": independent_sources(output)}
    assert all(sha(REPO / name) == digest for name, digest in before.items())
    manifest = dict(status="PASS", scope="Presentation only; frozen source data unchanged; no fitting or susceptibility computation",
                    renderer_sha256=sha(__file__), input_sha256=before,
                    output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                    original_figure_mapping={"Figure 4": "screening_information.tex:54 / screening_information_normalized.pdf",
                                             "Figure 8": "source_directions.tex:59 / independent_sources.pdf"},
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, figures=figures)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "PASS", "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
