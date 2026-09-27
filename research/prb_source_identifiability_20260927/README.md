# Independent source and readout directions

The completed evidence is described in [results.md](results.md). The integration
fragments are `manuscript_extension.tex` and `supplement_methods.tex`; they do
not edit the paper. The publication figure is
`figures_v2/independent_sources.pdf`.

Primary result editions:

- `deterministic_v2`: 48 noiseless fits and 40 sampled interaction profiles.
- `noise_v1`: 240 paired Gaussian-noise fits, 30 realizations per policy.
- `information_v4`: calibration/source nuisance projection and exact source
  exchange controls. The JSON key `interaction_whitened_singular_values` means
  complex-RMS scaling `J_R=(Re J,Im J)/sigma`; true Gaussian Fisher is
  `2 J_R.T J_R`. Interaction coordinates are `(g/.8, Ua/8)`.
- `unordered_v2`: six noiseless linear source-class recoveries and 3000
  conditional noisy estimates. Negative discriminants and estimator bias
  are retained. The noisy experiment holds interactions fixed.
- `norm_corrected_noise`: fresh 600-fit repetition of the prior mixed-source
  study after fixing its complex-RMS residual normalization.
- `mesh_*`: fits to identical, higher-resolution observations using lower
  templates. These use this study's fixed absolute sigma, not the older noise
  study's different normalization.
- `common_*`: final common-reference comparisons using all five frozen starts.
  The earlier three-start `mesh_B0_higher` run missed the near-truth basin for
  the symmetric full-matrix independent-source case; its failure is preserved.
  Correctly specified cases must improve on the unfitted synthetic truth's
  objective before their parameter shifts are interpreted as mesh effects.
- `common_summary_v1`: current input/source hash audit for all six final mesh
  datasets, 48 known/independent-source fit shifts in `fit_bias.csv`, and a
  compact full-matrix table. All 72 retained fits succeed and all 60 applicable
  truth-objective upper bounds pass. The mesh methods fragment is
  `common_reference_extension.tex`.

Earlier output editions are retained as implementation history; their source
hashes need not match the final scripts. `deterministic_v1` was interrupted
before a completed dataset because finite-difference derivatives were slow.
The same design was completed using analytic derivatives checked against
independent centered differences. No existing scientific result was overwritten.

Run from the repository root in PowerShell. Set BLAS threads to one:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B -m unittest discover -s research/prb_source_identifiability_20260927 -p test_source_protocol.py -v
python -B -m unittest discover -s research/prb_source_identifiability_20260927 -p test_common_reference.py -v
python -B -m unittest discover -s research/prb_mixed_source_response_20260914 -p test_mixed_vertex_response.py -v
python -B research/prb_source_identifiability_20260927/run_study.py --phase deterministic --output research/prb_source_identifiability_20260927/reproduce_deterministic
python -B research/prb_source_identifiability_20260927/run_study.py --phase noise --output research/prb_source_identifiability_20260927/reproduce_noise
python -B research/prb_source_identifiability_20260927/analyze_information.py --output research/prb_source_identifiability_20260927/reproduce_information
python -B research/prb_source_identifiability_20260927/recover_source_class.py --output research/prb_source_identifiability_20260927/reproduce_unordered
python -B research/prb_source_identifiability_20260927/render_figure.py --output research/prb_source_identifiability_20260927/reproduce_figure
python -B research/prb_mixed_source_response_20260914/noise_sensitivity.py --workers 4 --output research/prb_source_identifiability_20260927/reproduce_norm_corrected_noise
```

The renderer reads the archived primary fit/noise editions and regenerates
their published figure. Every command requires a new empty output directory;
choose another suffix when repeating it.

The final common-reference comparison accepts one or both field pairs and
uses all five original frozen starting points uniformly:

```powershell
python -B research/prb_source_identifiability_20260927/run_common_reference.py --input-bplus research/prb_mixed_source_response_20260914/production/Bplus_final/response_3channel.npz --reference-bplus research/prb_refinement_20260927/Bplus_combined/response_3channel.npz --output research/prb_source_identifiability_20260927/reproduce_common_reference
python -B research/prb_source_identifiability_20260927/run_common_reference.py --input-b0 research/prb_mixed_source_response_20260914/production/B0_final/response_3channel.npz --reference-b0 research/prb_refinement_20260927/B0_higher/response_3channel.npz --input-bplus research/prb_mixed_source_response_20260914/production/Bplus_final/response_3channel.npz --reference-bplus research/prb_refinement_20260927/Bplus_combined/response_3channel.npz --output research/prb_source_identifiability_20260927/reproduce_joint_common_reference
python -B research/prb_source_identifiability_20260927/summarize_common_reference.py --output research/prb_source_identifiability_20260927/reproduce_common_summary
```

To reproduce the individual magnetic refinement comparisons, substitute
`research/prb_refinement_20260927/Bplus_x_refined/response_3channel.npz`,
`Bplus_y_refined/response_3channel.npz`, or `Bplus_z_refined/response_3channel.npz`
for the magnetic `--input-bplus` path and use a different output directory.
The summary command audits the six archived final datasets named in
`common_summary_v1/audit.json` and regenerates their table without refitting.

No eigensystem is recomputed by these scripts. They use complete three-operator
NPZ response inputs. The raw objective retains both cross entries independently.
Gain fitting and source-class recovery do not establish a material probe,
global inverse uniqueness, confidence coverage, or continuum convergence.
