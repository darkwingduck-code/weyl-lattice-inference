# PRB mixed Gamma5/Gamma3 source-response validation

This isolated follow-up computes the full longitudinal response in the basis
`(charge, Gamma5, Gamma3)` and asks how a mixed source/readout direction affects
inference of `g` and `Ua`. Existing code, data, and manuscripts are read-only.
No result from this directory should enter a manuscript until production mesh
and convergence gates pass.

## Files

- `frozen_design.json`: machine-readable choices fixed before the pilot.
- `design_plan.md`: equations, validation gates, inference plan, and limits.
- `mixed_vertex_response.py`: streaming 3x3 bare response, embedded RPA,
  mixed-vertex contraction, Ward/passivity/closure diagnostics.
- `run_large_grid.py`: B+/B0 pilot, coarse, and final runners.
- `analyze_inference.py`: invariant and raw-gain inference controls.
- `test_mixed_vertex_response.py`: algebra, legacy agreement, direct-operator,
  Ward, passivity, and magnetic-cell phase tests.
- `noise_design.json`: noise seed, RMS definition, fit variants, boundaries,
  sample counts, metrics, and acceptance gates frozen before Monte Carlo.
- `noise_sensitivity.py`: paired B+/B0 final/coarse synthetic-noise study.

## Environment and checks

Run from `H:\Research\weyl-collective-modes` in PowerShell:

```powershell
& .\scripts\paper_environment.ps1
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s research\prb_mixed_source_response_20260914 -p test_mixed_vertex_response.py -v
```

End-to-end pilot:

```powershell
python research\prb_mixed_source_response_20260914\run_large_grid.py --field Bplus --mesh pilot --frequency-batch 4 --output research\prb_mixed_source_response_20260914\pilot\Bplus_pilot
python research\prb_mixed_source_response_20260914\analyze_inference.py --input research\prb_mixed_source_response_20260914\pilot\Bplus_pilot\response_3channel.npz --output research\prb_mixed_source_response_20260914\pilot_analysis\Bplus_pilot
```

## Production commands

Use new empty output directories. Do not overwrite a completed run.

```powershell
python research\prb_mixed_source_response_20260914\run_large_grid.py --field Bplus --mesh coarse --frequency-batch 4 --output research\prb_mixed_source_response_20260914\production\Bplus_coarse
python research\prb_mixed_source_response_20260914\run_large_grid.py --field Bplus --mesh final --frequency-batch 4 --output research\prb_mixed_source_response_20260914\production\Bplus_final
python research\prb_mixed_source_response_20260914\run_large_grid.py --field B0 --mesh final --frequency-batch 4 --output research\prb_mixed_source_response_20260914\production\B0_final
python research\prb_mixed_source_response_20260914\analyze_inference.py --input research\prb_mixed_source_response_20260914\production\Bplus_final\response_3channel.npz --reference research\prb_mixed_source_response_20260914\production\Bplus_coarse\response_3channel.npz --output research\prb_mixed_source_response_20260914\production_analysis\Bplus_final
```

Run coarse and final sequentially because the B+ eigensystems dominate RAM.
The output diagnostic includes wall time, resident memory, raw-pair count,
source hashes, all numeric gates, and `scientific_claim_ready=false`.

## Completed pilot (2026-09-15)

The B+ pilot (`36 x 3 x 48`, flux `1/36`, 45 frequencies) completed in
7.522 s and retained all 2,985,984 raw transition pairs. Peak observed process
RSS was 108,519,424 bytes (103.5 MiB). The charge-current vertex error was
`1.69e-14`, the full zeroth-moment error `1.91e-17`, left/right Ward errors
`2.89e-17` and `2.68e-17`, and the weighted-passivity minimum was positive
(`6.48e-4`). Embedded 3x3 dressing reproduced the established 2x2 RPA block
exactly at recorded precision.

The inference analysis completed in 9.337 s. With the declared `lambda=0.7`,
both invariant and raw-gain fits recovered `g=0.8, Ua=8`. Modeling the same
data as pure `Gamma5` instead moved the invariant fit to approximately
`g=0.903, Ua=0` with nonzero residual `0.0225`; reversing the sign drove `Ua`
to the upper fit bound with residual `0.317`. These values are pilot findings,
not converged physical claims. The local three-parameter Jacobian condition
number was about 196, so noise sensitivity must be quantified on production
data even though noiseless joint recovery succeeded.

Eight unit tests passed in 0.040 s. They cover the mixed contraction formula,
legacy RPA embedding, transfer closure, Windows memory measurement, zero-field
legacy agreement, direct mixed-operator agreement, Ward/moment/passivity
identities, and complete-torus magnetic-cell phase invariance. Pilot inputs,
source hashes, outputs, and per-directory manifests are stored under `pilot/`
and `pilot_analysis/`.

## Completed B+ production-grid calculation (2026-09-15)

The coarse run (`72 x 13 x 96`) retained 51,757,056 pairs in 136.038 s
with 125,001,728 bytes (119.2 MiB) peak observed RSS. The final run
(`72 x 13 x 192`) retained 103,514,112 pairs in 269.832 s with 157,159,424
bytes (149.9 MiB) peak observed RSS. Both passed all declared numerical
identities. Their charge/Gamma5 blocks agree with the prior independent
baseline at scaled errors below `1.6e-15` bare and `5.6e-16` dressed.

The final-to-coarse continuous convergence diagnostics are `5.73e-4` for the
full bare 3x3 response, `3.35e-4` for the dressed 3x3 and mixed `lambda=0.7`
responses, and `2.56e-6` for transfer closure. Component-normalized dressed
errors range from `3.17e-5` (`Gamma3,Gamma3`) to `5.99e-3`
(`charge,Gamma5`). No pass threshold was selected after seeing these values.

The final-grid noiseless inference retains the pilot conclusion: the correct
mixed direction recovers `(g,Ua)=(0.8,8)`, while a pure-Gamma5 model gives
approximately `(0.9031,0)` with invariant residual `0.02214`, and the
sign-reversed model reaches the `Ua=24` fit boundary with residual `0.31746`.
The local `(g,Ua,lambda)` condition number is `200.5`. This establishes a
stable field-specific numerical result for the declared synthetic test. The
later sections add B0 and detector-noise controls; they do not turn the assumed
operator into a material vertex or certify a global inverse solution.

## Completed B0 production-grid calculation (2026-09-15)

B0 coarse (`48 x 48 x 192`) retained 7,077,888 pairs in 17.491 s;
B0 final (`72 x 72 x 384`) retained 31,850,496 pairs in 79.380 s. Peak
observed RSS stayed below 100 MiB. Both passed all response identities and
reproduced the existing charge/Gamma5 baselines below `2.7e-16` bare and
`1.3e-16` dressed. Final-to-coarse scaled errors were `1.46e-4` bare,
`1.16e-4` dressed/mixed, and `2.66e-6` for closure. These numbers combine
transverse and longitudinal refinement and therefore do not isolate separate
directional convergence rates.

The noiseless B0 direction-misspecification control is field dependent. A
pure-Gamma5 raw model fitted to lambda=0.7 truth reached `Ua=24` with
`g=0.7404`; the sign-reversed model also reached `Ua=24` with `g=0.9896`.
The correct and jointly estimated directions recovered `(0.8,8,0.7)`.

## Frozen absolute-noise study

`noise_design.json` fixes a common absolute complex RMS across all response
entries and both fields. The Monte Carlo uses 50 realizations at `epsilon=1e-3`
and 25 at `epsilon=1e-2`; the reduced count was selected before execution
because the paired design requires 600 three-start nonlinear fits. Run:

```powershell
python research\prb_mixed_source_response_20260914\noise_sensitivity.py --workers 4 --output research\prb_mixed_source_response_20260914\noise_sensitivity\run_v1
```

The same noisy realization is fitted with final and coarse response templates,
first with `lambda=0.7` known and then with lambda jointly estimated. Boundary
solutions are retained and counted. Bias, RMSE, standard deviation, five
quantiles, optimizer status, residuals, empirical noise RMS, and paired mesh
effects are written without a post hoc bias threshold.

The frozen study completed 600 fits in 306.30 s. All optimizers succeeded,
all estimates were finite, and empirical/declared complex-noise RMS ratios
ranged from 0.9908 to 1.0064. Final-template RMSE values were:

| Field | epsilon | lambda | g RMSE | Ua RMSE | lambda RMSE | boundary result |
|---|---:|---|---:|---:|---:|---|
| B+ | 0.001 | known | 0.000492 | 0.0412 | -- | no hits |
| B+ | 0.001 | joint | 0.000504 | 0.0478 | 0.0118 | no hits |
| B+ | 0.01 | known | 0.00392 | 0.324 | -- | no hits |
| B+ | 0.01 | joint | 0.00423 | 0.543 | 0.178 | lambda upper hit 5/25 |
| B0 | 0.001 | known | 0.00231 | 0.101 | -- | no hits |
| B0 | 0.001 | joint | 0.00223 | 0.143 | 0.00698 | no hits |
| B0 | 0.01 | known | 0.0229 | 0.785 | -- | no hits |
| B0 | 0.01 | joint | 0.0235 | 2.05 | 0.0976 | no hits |

At `epsilon=0.01`, freeing lambda materially enlarges the `Ua` uncertainty;
the B+ lambda fit also becomes boundary limited. Paired coarse-minus-final
effects are smaller than the corresponding noise RMSE in this design (for
example, joint-lambda RMSE `0.00701` versus `0.178` for B+, and `0.000305`
versus `0.0976` for B0). This supports numerical stability under the tested
mesh change, while demonstrating that operator-direction uncertainty remains
a detector-limited inverse problem. The synthetic Gaussian model, assumed
real lambda, 50/25 sample counts, and lack of coverage/global-optimum proof
keep the result marked `manuscript_ready=false`.

## Verified delivery bundle

Build the deterministic evidence archive after all validations:

```powershell
python research\prb_mixed_source_response_20260914\build_delivery.py --name prb_mixed_source_validation_local_01
```

The archive includes code, frozen designs, tests, pilot evidence, all four
B+/B0 production response sets, inference analyses, and noise results. Its
external JSON manifest records every member's SHA-256, CRC32, byte size, the
ZIP SHA-256, member order, and CRC/readback verification. ZIP/cache files are
excluded from the archive inventory. Choose a new unused edition name for each
later build so that an earlier archive is never overwritten.

## Interpretation boundary

The mixed coefficient is a controlled local-operator deformation. It is not a
material-specific light-scattering or transport vertex. B+ and B0 production
convergence and the frozen detector-noise calculation now establish the
numerical behavior of this synthetic design. A claim about a real probe still
needs a microscopic material vertex, an experimentally motivated covariance,
larger samples with coverage or tail validation, and a new transverse B+
detector-norm check. Joint `(g,Ua,lambda)` fits are local multistart diagnostics
and carry no global uniqueness claim.
