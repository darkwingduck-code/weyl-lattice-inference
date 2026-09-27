"""Regenerate the presentation figure without repeating the response scan."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("run_01"))
args = parser.parse_args()
RUN = (ROOT / args.run).resolve()
if RUN.parent != ROOT or not RUN.is_dir():
    raise ValueError("Run must be an existing direct child of this research directory.")
with (RUN / "information.csv").open() as handle:
    rows = list(csv.DictReader(handle))
methods = ("full_calibrated", "full_profiled", "axial_calibrated", "axial_profiled")
labels = ("All entries, calibrated", "All entries, gains profiled",
          "Axial auto, calibrated", "Axial auto, gain profiled")
styles = (("#1f77b4", "-"), ("#1f77b4", "--"), ("#d95f02", "-."), ("#d95f02", ":"))
fig, axes = plt.subplots(1, 2, figsize=(8., 3.3), sharey=True, constrained_layout=True)
for axis, field in zip(axes, ("B0", "Bplus")):
    subset = [r for r in rows if r["field"] == field]
    u = [float(r["Ua"]) for r in subset]
    for name, label, (color, style) in zip(methods, labels, styles):
        values = np.array([float(r[name]) for r in subset])
        axis.loglog(u, values / values[0], label=label, color=color, linestyle=style)
    axis.set(title=field, xlabel=r"$U_a$ (fixed-$\Pi$ continuation)")
    axis.grid(alpha=.2)
axes[0].set_ylabel(r"$I_{U_aU_a}(U_a)/I_{U_aU_a}(8)$")
axes[1].legend(fontsize=7, loc="lower left")
fig.suptitle("Fixed absolute noise; local Gaussian benchmark", fontsize=10)
pdf = RUN / "screening_information_normalized.pdf"
fig.savefig(pdf)
fig.savefig(RUN / "screening_information_normalized.png", dpi=160)
plt.close(fig)
files = [Path(__file__), RUN / "information.csv", pdf]
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(RUN / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
