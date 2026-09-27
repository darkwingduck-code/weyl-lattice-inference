# Independent confirmation of two selected contrasts

`confirmation_plan.json` was frozen before this calculation. The 30-repeat
pilot selected two comparisons: equal source directions, 75% magnetic time
versus magnetic-only; unequal directions, 25% magnetic time versus zero-field
only. The confirmation uses a different seed, 2026092721, and exactly 200
paired repetitions for each truth. It neither searches new allocations nor
adds a 50:50 comparison. Independence refers to the generated detector noise;
the physical model, response arrays and acquisition assumptions are unchanged.

All 800 fits import the unchanged pilot `Problem` and its five initial points,
bounds, noise scale, response helpers and field-specific affine gains. No
module configuration is overwritten. Each repetition uses one standard complex
noise array per field, rescaled by each policy's sigma0/sqrt(time fraction).
Candidate and baseline share these random arrays. The two truth pairs have
different arrays. Four processes execute disjoint repetition indices modulo
four while regenerating the same complete RNG sequence, so pair membership
does not depend on execution order.

The primary observation for repetition i is

```
D_i = (Ua_candidate_i - 8)^2 - (Ua_baseline_i - 8)^2.
```

Report its mean and sample-standard-deviation/sqrt(200). Negative mean favors
the candidate. The interval is mean +/- z*SE, with
`z=NormalDist().inv_cdf(.9875)` = approximately 2.2414. These two-sided 97.5%
intervals use a Bonferroni correction for the two fixed contrasts, targeting
95% simultaneous coverage under the stated normal approximation. They concern
Monte Carlo expected squared-error differences, not confidence intervals for
an interaction parameter, exact finite-sample guarantees, or apparatus data.

Every estimate and all five starts are retained. A retained optimizer failure,
parameter-bound hit or failure to improve the feasible unfitted-truth objective
blocks a primary precision claim for that contrast. The code does not silently
discard failed cases or introduce replacement simulations.

From the repository root, use a new directory. The four fit commands may run
concurrently, with at most four processes and one BLAS thread each:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B research/prb_fixed_budget_20260927/run_confirmation.py --phase fit --shard 0 --output research/prb_fixed_budget_20260927/reproduce_confirmation
python -B research/prb_fixed_budget_20260927/run_confirmation.py --phase fit --shard 1 --output research/prb_fixed_budget_20260927/reproduce_confirmation
python -B research/prb_fixed_budget_20260927/run_confirmation.py --phase fit --shard 2 --output research/prb_fixed_budget_20260927/reproduce_confirmation
python -B research/prb_fixed_budget_20260927/run_confirmation.py --phase fit --shard 3 --output research/prb_fixed_budget_20260927/reproduce_confirmation
python -B research/prb_fixed_budget_20260927/run_confirmation.py --phase summary --output research/prb_fixed_budget_20260927/reproduce_confirmation
```

`confirmation_01/fits_0.json` through `fits_3.json` contain all attempts.
The final directory also contains `summary.json`, `policy_summary.csv`,
`paired_differences.csv` and `contrasts.csv`. Original pilot inputs, code,
plan and result files remain untouched. Each confirmation shard checks the
same scientific source/input hashes before and after execution, and the
summary verifies agreement between shards and current files.
