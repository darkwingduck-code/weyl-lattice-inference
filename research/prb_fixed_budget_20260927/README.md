# Fixed total acquisition budget

`frozen_plan.json` fixes the response inputs, two true operator pairs, five
field-time allocations, noise, parameter bounds, five starts and 30 paired
realizations before calculation. Only the higher B0 and combined Bplus meshes
are used. Existing scientific source files are imported without modification.

This is a full-matrix to full-matrix comparison. A total time T is distributed
over fields, and equally over the same four complex entries and 45 frequencies
within each field. If field b receives fraction f_b, its per-entry complex
noise RMS is sigma_b = sigma0/sqrt(f_b). A single field receives f_b=1.
The detector noise prefactors are assumed equal; switching overhead and drift
are absent from this declared model. The charge channel remains calibrated.

The least-squares norm is `sqrt(sum_b,j |residual_b,j/sigma_b|^2 / M)`, where
M counts the actually acquired complex entries over the active fields. No
factor sqrt(2) is introduced by realification. For local information the
Jacobian is instead multiplied by sqrt(2M), giving the true circular-complex
Gaussian Fisher factor 2. Gain columns and both direction columns are
projected out; the remaining 2x2 interaction Fisher matrix is inverted with
both g and Ua fitted. Coordinates are `(g/.8, Ua/8)`.

The optimizer adapts the previous analytic full-matrix fitter only to apply
field-dependent noise weights. Forward RPA, external contraction, analytic
physical derivatives, transfer functions and projection helpers are reused.
Five fixed starts are used for every fit. All attempts and failures are saved.
The unfitted synthetic truth gives an objective upper-bound diagnostic, not
an optimizer start or an estimate. A failed upper-bound check must not be
interpreted as evidence that a measurement allocation is poor.

Run from the repository root in PowerShell, using a new result directory:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B -m unittest discover -s research/prb_fixed_budget_20260927 -p test_fixed_budget.py -v
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase information --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase fit --shard 0 --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase fit --shard 1 --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase fit --shard 2 --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase fit --shard 3 --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/run_fixed_budget.py --phase summary --output research/prb_fixed_budget_20260927/reproduce
python -B research/prb_fixed_budget_20260927/render_results.py --input research/prb_fixed_budget_20260927/reproduce --output research/prb_fixed_budget_20260927/reproduce_figures
```

The four fit commands may run concurrently in four separate processes with
one BLAS thread each. They regenerate identical paired noise from the same
seed but execute disjoint jobs. Reusing a populated shard is rejected.
`run_01` is the archived experiment; `figures_v1` contains the figure, complete
table, approximate Monte Carlo sampling errors and paired loss differences.
The latter are sampling diagnostics, not parameter confidence intervals.

The meaningful regression checks independently finite-difference the weighted
Jacobian, check unit complex RMS for unequal field weights, and verify that
splitting a fixed budget between identical responses does not manufacture
additional Fisher information. Source and NPZ hashes are checked before and
after execution. No eigenproblem is recomputed.
