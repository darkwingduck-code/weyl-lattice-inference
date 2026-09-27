# PRB figure typography revision

Approved delivery: `run_02/assets_mapping.json` maps all nine main-text and
two supplementary figures to their existing paper paths. PDF and PNG files
are alongside their renderer-specific provenance. `run_02/visual_review.json`
records the final visual approval; the render manifest deliberately records
the earlier, pre-review render stage.

Reproduce from the repository root into a **new** directory:

```powershell
python -B research/figure_style_epjp_20260927/prb/render_all.py --output research/figure_style_epjp_20260927/prb/reproduction/styled_figures
python -m ruff check --no-cache --select F,E9 research/figure_style_epjp_20260927/prb/render_all.py
```

The wrapper invokes the frozen plot renderers, then applies the shared
Matplotlib style before PDF/PNG export. It neither edits the original
renderers/results/paper nor recomputes response matrices or inference fits.
The budget renderer summarizes already stored fits using its existing code.
All original input checksums are checked before and after rendering.

The common helper compares numerical plot arrays, paths, axis limits/scales,
colors, line and marker properties before and after the typography/layout
change. All 11 comparisons passed. The original five editorial figures also
retain their renderer's independent trace checks. The final PDFs contain
36 standalone lowercase panel labels, all DejaVu Sans Bold at 11 points;
axis labels are 10 points and tick/legend text 9 points. All PDF text spans
lie within their page bounds. Ruff F/E9 passes.

All 11 PNGs were visually inspected: nine final PNGs are byte-identical to
the inspected first pass; the two revised layouts were inspected again.
The first pass remains in the original source project; this release includes
only the final styled edition.

- Main Fig. 4 retains hollow triangle, square, circle and diamond markers
  with the existing staggered sample locations and legend.
- Main Fig. 8 retains the 0.06-inch (4.32-point) source-panel/colorbar gap.
  The full lower-panel category labels are rotated 35 degrees to avoid
  overlap; their words and associated data are unchanged.
- Supplement Fig. 1 moves scientific-notation offsets to the upper right
  and wraps existing titles so they do not collide with panel labels or
  extend beyond the canvas.

All canvases are 7.2 inches wide for the PRB two-column figure width.
No manuscript text is included. The release root README identifies the
versioned final assets and the standalone reproduction command.
