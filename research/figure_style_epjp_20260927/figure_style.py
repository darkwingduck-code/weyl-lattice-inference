"""Shared typography and read-only scientific-artist checks for paper figures."""
from __future__ import annotations

import hashlib
import json
import re

import matplotlib as mpl
from matplotlib.colors import to_rgba
from matplotlib.text import Text
import numpy as np


def _array(value):
    array = np.asarray(value)
    if array.dtype.hasobject:
        payload = repr(array.tolist()).encode()
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _paint(artist):
    result = {"alpha": artist.get_alpha(), "visible": artist.get_visible()}
    for name in ("facecolor", "edgecolor"):
        method = getattr(artist, "get_" + name, None)
        if method is not None:
            result[name] = _array(method())
    return result


def snapshot_figure(fig):
    """Fingerprint data, limits, scales, colors and marker identities, not layout."""
    fig.canvas.draw()
    axes = []
    for ax in fig.axes:
        entry = {"limits": [list(ax.get_xlim()), list(ax.get_ylim())],
                 "scales": [ax.get_xscale(), ax.get_yscale()],
                 "lines": [], "collections": [], "patches": [], "images": []}
        for line in ax.lines:
            entry["lines"].append({
                "x": _array(line.get_xdata()), "y": _array(line.get_ydata()),
                "color": list(to_rgba(line.get_color())),
                "alpha": line.get_alpha(), "marker": str(line.get_marker()),
                "markersize": line.get_markersize(),
                "markerface": list(to_rgba(line.get_markerfacecolor())),
                "markeredge": list(to_rgba(line.get_markeredgecolor())),
                "linestyle": line.get_linestyle(), "linewidth": line.get_linewidth(),
                "label": line.get_label(),
            })
        for coll in ax.collections:
            item = {"offsets": _array(coll.get_offsets()),
                    "paths": [_array(p.vertices) for p in coll.get_paths()],
                    "paint": _paint(coll), "label": coll.get_label(),
                    "array": None if coll.get_array() is None else _array(coll.get_array())}
            if hasattr(coll, "get_sizes"):
                item["sizes"] = _array(coll.get_sizes())
            if hasattr(coll, "get_cmap"):
                item["cmap"] = coll.get_cmap().name
                item["clim"] = list(coll.get_clim())
            entry["collections"].append(item)
        for patch in ax.patches:
            entry["patches"].append({"path": _array(patch.get_path().vertices),
                                      "paint": _paint(patch),
                                      "geometry": _array(patch.get_patch_transform().get_matrix())})
        for im in ax.images:
            entry["images"].append({"array": _array(im.get_array()),
                                     "extent": list(im.get_extent()),
                                     "cmap": im.get_cmap().name,
                                     "clim": list(im.get_clim()), "alpha": im.get_alpha()})
        axes.append(entry)
    return {"axes": axes}


def assert_science_unchanged(before, fig):
    after = snapshot_figure(fig)
    if before != after:
        changed = [i for i, (a, b) in enumerate(zip(before["axes"], after["axes"])) if a != b]
        raise AssertionError(f"Scientific artists changed in axes {changed}")
    return hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()


def _strip_panel(text):
    for pattern in (r"^\s*\$\\(?:bf|mathbf)\{\(([a-zA-Z])\)\}\$\s*",
                    r"^\s*\(([a-zA-Z])\)\s*"):
        match = re.match(pattern, text)
        if match:
            return match.group(1).lower(), text[match.end():].strip()
    return None, text.strip()


def style_figure(fig, panel_axes=None, panel_labels=None, keep_titles=False):
    """Apply 11/10/9 pt typography; return removed titles for caption review.

    Callers own layout/physical figure dimensions. Default labels use axes order;
    explicitly pass panel_axes when colorbars or a different panel order occur.
    keep_titles=True retains the prefix-stripped title as regular 10 pt text.
    """
    mpl.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for text in fig.findobj(match=Text):
        text.set_fontfamily("DejaVu Sans")
        text.set_fontsize(9)
        text.set_fontweight("normal")
        text.set_math_fontfamily("dejavusans")
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis="both", which="both", labelsize=9)
        for legend in ([ax.get_legend()] if ax.get_legend() is not None else []):
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_fontsize(9)
            legend.get_title().set_fontsize(9)
    for legend in fig.legends:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(9)
        legend.get_title().set_fontsize(9)
    selected = list(fig.axes if panel_axes is None else panel_axes)
    labels = list(panel_labels) if panel_labels is not None else [chr(97+i) for i in range(len(selected))]
    if len(labels) != len(selected):
        raise ValueError("One panel label is required for each panel axis")
    report = []
    for ax, label in zip(selected, labels):
        titles = []
        for loc in ("left", "center", "right"):
            text = ax.get_title(loc=loc)
            if text:
                _, descriptor = _strip_panel(text)
                titles.append({"location": loc, "original": text, "descriptor": descriptor})
                ax.set_title(descriptor if keep_titles else "", loc=loc,
                             fontsize=10, fontweight="normal", pad=22 if keep_titles else 6)
        for text in list(ax.texts):
            if text.get_gid() == "publication-panel-label":
                text.remove()
            else:
                letter, remainder = _strip_panel(text.get_text())
                if letter is not None and not remainder:
                    text.remove()
        ax.text(0.0, 1.025, f"({str(label).strip('()').lower()})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="bold",
                fontfamily="DejaVu Sans", color="black", clip_on=False,
                gid="publication-panel-label")
        report.append({"axis_index": fig.axes.index(ax), "panel": label, "titles": titles})
    return report
