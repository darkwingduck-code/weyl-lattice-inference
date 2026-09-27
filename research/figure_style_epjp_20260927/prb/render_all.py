"""Restyle all eleven PRB figures using their frozen renderers and plot data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
import fitz
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from figure_style import snapshot_figure, style_figure, assert_science_unchanged

MAPPING = {
    "figure1_uv": ("main", 1, "editorial_figures_v2/figure1_uv.pdf"),
    "figure2_qwindows": ("main", 2, "editorial_figures_v2/figure2_qwindows.pdf"),
    "figure3_calibration": ("main", 3, "editorial_figures_v2/figure3_calibration.pdf"),
    "screening_information_normalized": ("main", 4, "editorial_figures_v2/screening_information_normalized.pdf"),
    "observation_profiles": ("main", 5, "observation_figures/observation_profiles.pdf"),
    "observation_noise": ("main", 6, "observation_figures/observation_noise.pdf"),
    "observation_limits": ("main", 7, "observation_figures/observation_limits.pdf"),
    "independent_sources": ("main", 8, "editorial_figures_v2/independent_sources.pdf"),
    "fixed_budget": ("main", 9, "editorial_figures_v2/fixed_budget.pdf"),
    "figure4_reference": ("supplement", 1, "editorial_figures_v2/figure4_reference.pdf"),
    "figure5_eta": ("supplement", 2, "editorial_figures_v2/figure5_eta.pdf"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def loaded(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def known_inputs():
    hashes = {}
    def add(mapping, base=REPO):
        for name, digest in mapping.items():
            normalized = name.replace("\\", "/")
            if "/research/" in normalized:
                normalized = "research/"+normalized.split("/research/", 1)[1]
                path = REPO / normalized
            else:
                path = base / normalized
            assert sha(path) == digest, path
            hashes[path.relative_to(REPO).as_posix()] = digest
    add(read(REPO / "paper/prb_strengthened/editorial_figures_v2/provenance.json")["input_sha256"])
    add(read(REPO / "research/prb_refinement_20260927/observation_figures_corrected/diagnostic.json")["input_sha256"],
        REPO / "research/prb_observation_design_20260912")
    add(read(REPO / "research/figure_revision_20260927/prb/run_05/manifest.json")["input_sha256"])
    add(read(REPO / "research/prb_fixed_budget_20260927/figures_v1/manifest.json"))
    paths = ["scripts/render_prb_editorial_figures.py", "paper/prb_submission/render_figures.py",
             "research/prb_observation_design_20260912/render_figures_final.py",
             "research/figure_revision_20260927/prb/render_figures.py",
             "research/prb_fixed_budget_20260927/render_results.py",
             "research/figure_style_epjp_20260927/figure_style.py"]
    for name in paths:
        hashes[name] = sha(REPO / name)
    return hashes


def reflow(fig, name):
    axes = fig.axes
    if name in ("figure1_uv", "figure5_eta"):
        fig.set_size_inches(7.2, 5.2)
        grid = GridSpec(2, 2, figure=fig, left=.115, right=.98, bottom=.12,
                        top=.78 if name == "figure1_uv" else .84, wspace=.43, hspace=.98)
        for index, axis in enumerate(axes):
            axis.set_position(grid[index//2, index%2].get_position(fig))
    elif name in ("figure2_qwindows", "figure4_reference"):
        reference = name == "figure4_reference"
        fig.set_size_inches(7.2, 7.2 if reference else 5.8)
        grid = GridSpec(2, 3, figure=fig, left=.11, right=.955 if reference else .985,
                        bottom=.10 if reference else .17, top=.74 if reference else .82,
                        wspace=.55 if reference else .65, hspace=1.50 if reference else 1.15)
        for index, axis in enumerate(axes):
            case, row = divmod(index, 2)
            axis.set_position(grid[row, case].get_position(fig))
            if reference:
                for location in ("left", "center", "right"):
                    title = axis.get_title(loc=location)
                    axis.set_title(title.replace("Own-mesh reference", "Own-mesh\nreference")
                                   .replace("Common higher reference", "Common higher\nreference"), loc=location)
                axis.yaxis.get_offset_text().set_x(1.)
                axis.yaxis.get_offset_text().set_ha("right")
        return [axes[i] for i in (0, 2, 4, 1, 3, 5)]
    elif name == "figure3_calibration":
        fig.set_size_inches(7.2, 3.7)
        grid = GridSpec(1, 3, figure=fig, left=.09, right=.985, bottom=.22, top=.64, wspace=.48)
        for index, axis in enumerate(axes):
            axis.set_position(grid[0, index].get_position(fig))
    elif name.startswith("observation_"):
        fig.set_size_inches(7.2, 4.6)
    elif name == "screening_information_normalized":
        fig.set_size_inches(7.2, 3.8)
    elif name == "fixed_budget":
        fig.set_size_inches(7.2, 3.8)
    elif name == "independent_sources":
        fig.set_size_inches(7.2, 7.1)
        axes[0].set_position([.10, .58, .335, .34])
        axes[2].set_position([.65, .57, .335, .35])
        axes[3].set_position([.10, .13, .885, .26])
        for label in axes[3].get_xticklabels():
            label.set_rotation(35)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        return [axes[i] for i in (0, 2, 3)]
    return axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("Choose a new output directory beneath this script")
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshots").mkdir()
    inputs = known_inputs()
    records = {}
    original_savefig = Figure.savefig

    def savefig(fig, filename, *args, **kwargs):
        path = Path(filename)
        name = path.stem
        if name in MAPPING and not getattr(fig, "_prb_publication_style", False):
            before = snapshot_figure(fig)
            selected = reflow(fig, name)
            titles = style_figure(fig, panel_axes=selected, keep_titles=True)
            digest = assert_science_unchanged(before, fig)
            fig._prb_publication_style = True
            dump(output / "snapshots" / f"{name}.json", before)
            records[name] = dict(document=MAPPING[name][0], figure_number=MAPPING[name][1],
                                 paper_target=MAPPING[name][2], original_pdf_sha256=sha(REPO / "paper/prb_strengthened" / MAPPING[name][2]),
                                 source_snapshot_sha256=digest, numerical_artists_limits_colors_unchanged=True,
                                 panel_report=titles, size_inches=fig.get_size_inches().tolist(), output_pdf=str(path.with_suffix(".pdf").relative_to(output)))
        return original_savefig(fig, filename, *args, **kwargs)

    with patch.object(Figure, "savefig", savefig):
        with patch.object(sys, "argv", ["render_prb_editorial_figures.py", "--output", str(output / "editorial")]):
            runpy.run_path(str(REPO / "scripts/render_prb_editorial_figures.py"), run_name="__main__")
        observation = loaded(REPO / "research/prb_observation_design_20260912/render_figures_final.py", "restyle_observation")
        observation.OUT = output / "observation"
        observation.main()
        revision = loaded(REPO / "research/figure_revision_20260927/prb/render_figures.py", "restyle_revision")
        (output / "revision").mkdir()
        revision.screening(output / "revision")
        revision.independent_sources(output / "revision")
        sys.path.insert(0, str(REPO / "research/prb_fixed_budget_20260927"))
        budget = loaded(REPO / "research/prb_fixed_budget_20260927/render_results.py", "restyle_budget")
        budget.run(REPO / "research/prb_fixed_budget_20260927/run_01", output / "budget")
    assert set(records) == set(MAPPING), set(MAPPING)-set(records)
    assert all(sha(REPO / name) == digest for name, digest in inputs.items())
    for name, record in records.items():
        with fitz.open(output / record["output_pdf"]) as document:
            page = document[0]
            clipped = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        box = fitz.Rect(span["bbox"])
                        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > page.rect.width+.5 or box.y1 > page.rect.height+.5:
                            clipped.append(span["text"])
            record["text_clipping"] = clipped
            record["pdf_sha256"] = sha(output / record["output_pdf"])
    manifest = dict(status="RENDERED_REQUIRES_VISUAL_REVIEW", renderer_sha256=sha(__file__), input_sha256=inputs,
                    all_frozen_inputs_unchanged=True, no_scientific_model_recalculation=True,
                    typography=dict(family="DejaVu Sans", panel_bold_points=11, axis_points=10, tick_legend_points=9),
                    figures=records)
    dump(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "figures": len(records), "clipping": {n: r["text_clipping"] for n, r in records.items() if r["text_clipping"]}}, indent=2))


if __name__ == "__main__":
    main()
