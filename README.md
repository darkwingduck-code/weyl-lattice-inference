# Weyl lattice inference: data and code

Research artifacts for **Screening and cross-response constraints on interaction
inference in a Weyl lattice**, Gyeongtae Im. Version 1.1.0, 27 September 2026.
Journal target: *Physical Review B*. This release does not imply acceptance.

This separate repository contains the lattice and observation calculations,
retained numerical outcomes, fixed designs, tests and figure inputs for this
paper. It does not copy Git history, the oscillator paper or private editorial
notes. All failures belonging to the retained numerical studies remain visible.

## Verify and reproduce

Use Python 3.11 or newer with the recorded package versions:

```sh
python -m pip install -r requirements.txt
python verify_release.py --tests
```

This checks every payload checksum and runs the targeted reader, response,
source-direction and fixed-budget tests. It does not repeat the large magnetic
band summations or the full Monte Carlo study.

| Directory under `research/` | Role |
|---|---|
| `prb_lattice_response` | Four-band response, Ward/moment controls, screening and calibration reference data |
| `prb_observation_design_20260912` | Matrix/diagonal observation policies and detector-noise studies |
| `prb_mixed_source_response_20260914` | Complete three-operator response and mixed-probe generators |
| `prb_screening_information_20260927/run_02` | Fixed-bare-response large-interaction continuation |
| `prb_source_identifiability_20260927` | Independent source/readout fits and the source-exchange equivalence |
| `prb_refinement_20260927` | Directional/higher-grid responses and common-observation comparisons |
| `prb_fixed_budget_20260927` | 300-fit pilot and independent 800-fit confirmation at fixed total acquisition time |
| `figure_revision_20260927/prb/run_05` | Readability revision of Figures 4 and 8; numerical arrays unchanged |

The version 1.0.0 nine main-text and two supplemental figure PDFs are preserved
under `paper/prb_strengthened/editorial_figures_v2` and `observation_figures`.
These directories contain figure assets only, without manuscript text.

Follow the study READMEs and frozen plans for exact reproduction commands and
primary output editions. The older observation experiment has a portable
wrapper for historical absolute input paths:

```sh
python -B research/prb_observation_design_20260912/standalone_reproduce.py --stage checks --output reproduction/observation_check
python -B research/figure_revision_20260927/prb/render_figures.py --output research/figure_revision_20260927/prb/reproduction/figures
python -B scripts/render_prb_editorial_figures.py --output reproduction/editorial_figures
python -B research/prb_observation_design_20260912/render_figures_final.py --output reproduction/observation_figures
```

The last two commands recreate the earlier five lattice/calibration controls
and three observation figures. The editorial renderer imports the released
`paper/prb_submission/render_figures.py` helper; no manuscript source is needed.

Choose a new output directory on each run. The large-grid generators stream all
band pairs and may take substantial time; the compact stored response arrays
permit subsequent inference without repeating that step. Summarizing the
independent confirmation requires all four completed shards; its README records
the commands and the fixed 200-pair stopping rule.

The independent confirmation supports one of the two preselected fixed-budget
comparisons; the other remains unresolved. RPA, known bare response, specified
noise and probe families remain assumptions, not experimental validation.

## Figure styling in version 1.1.0

Version 1.1.0 preserves the scientific calculations and version 1.0.0 history.
It adds the final publication figure typography: standalone bold lowercase
panel labels, consistent axis/tick/legend sizes, and verified text layout.
The shared Matplotlib style checks numerical arrays, axis limits, colors,
line styles and markers before and after each display-only revision.
No model fitting, eigensolver calculation or Monte Carlo study is repeated.

The eleven final styled figures and their checks are under
`research/figure_style_epjp_20260927/prb/run_02`; `assets_mapping.json` maps each
asset to its paper figure number. Earlier figure directories below retain the
version 1.0.0 assets as provenance. The hollow triangle/square/circle/diamond
markers in Figure 4 and the reduced Figure 8 colorbar gap are preserved.

```sh
python -B research/figure_style_epjp_20260927/prb/render_all.py --output research/figure_style_epjp_20260927/prb/reproduction/styled_figures
```

Use a new output directory for each render. The release verification also
checks every final asset and the preserved original source/data hashes.

## Provenance, citation and licensing

`release_manifest.json` records original paths and exact SHA256 checksums.
Historical absolute paths in diagnostics identify the original calculation
machine; they do not change the physical data. The portable reader validates
the same inputs after relocation.

Use `CITATION.cff` together with an immutable commit/release version. A GitHub
release is not a DOI, and no DOI has been invented for this archive. A
DOI-bearing archival copy can be added later. Existing file-specific licenses
remain applicable; this export adds no blanket license to previously unlicensed
materials.
