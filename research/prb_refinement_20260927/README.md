# Directional full-band refinement

The design is recorded in `notes/revision_20260927_plan.md`. Five streamed
three-operator calculations retain every band pair at fixed physical inputs.
No new numerical dependency is required.

| Field | Mesh name | Lx, Ly, Nkz |
|---|---|---|
| Bplus | x_refined | 144, 13, 192 |
| Bplus | y_refined | 72, 25, 192 |
| Bplus | z_refined | 72, 13, 384 |
| Bplus | combined | 144, 25, 384 |
| B0 | higher | 96, 96, 768 |

Each directory includes the bare and dressed complex response, Ward/moment/
passivity checks, physical parameters, input/source hashes, elapsed time, and
observed memory use. The complete stored response is small because transition
pairs are summed in streaming blocks, not discarded by a weight cutoff.

From the repository root, a fresh independent reproduction of one grid is:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B research/prb_mixed_source_response_20260914/run_large_grid.py --field Bplus --mesh combined --output reproduction/Bplus_combined
```

Use the other field/mesh pairs in the table for the remaining grids. Output
directories must be new; completed scientific runs are never overwritten.
Compare the arrays and metadata in a reproduced `response_3channel.npz` with
the archived grid, rather than relying on byte identity of a ZIP container.

The following commands audit the five archived grids and regenerate the
comparison/table outputs into new directories:

```powershell
python -B research/prb_refinement_20260927/compare_refinement.py --self-check
python -B research/prb_refinement_20260927/compare_refinement.py --require-all --output reproduction/grid_comparison
python -B research/prb_refinement_20260927/render_refinement.py --output reproduction/refinement_tables
```

`final_comparison/` is the manuscript response comparison. It retains two
explicit units: the earlier symmetric-truth noise RMS `1.817608636e-4` and
the new two-direction source-study RMS `1.293166944e-4`. The response arrays
are identical under this unit conversion. The manuscript tables use the
second convention, shared with the common-reference fits.

The fitting driver and its frozen starting points are in
`research/prb_source_identifiability_20260927/`. The final
`common_Bplus_{final,x,y,z}_v1` runs all fit the same combined observation;
`common_B0_v1` uses the higher zero-field observation, and `common_joint_v1`
uses both. Correctly specified fits must improve on the feasible unfitted
synthetic-truth objective. The failed earlier three-start zero-field result
is retained and is not interpreted as a quadrature bias.

`observation_figures_corrected/` only repairs the older figure's statement
about condition-dependent, nearly equal noise scales. Its nine scientific
input hashes match the previous figure exactly. Neither this correction nor
finite-grid agreement establishes a continuum error bound.
