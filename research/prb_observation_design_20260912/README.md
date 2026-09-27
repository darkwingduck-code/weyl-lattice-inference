# PRB observation-design follow-up

The new result is a quantitative, conditional observation protocol: both mixed responses, combined with calibrated charge response, constrain the axial interaction even when factorized axial source/readout transfers have unknown complex phase and amplitude. At the selected magnetic baseline, fitting completely independent complex gains at every frequency leaves representative wrong interactions11.12/9.08 common-noise RMS units apart. The same noise model yields sharply poorer interaction recovery in a weak-mixing low-density condition.

This does not establish an experimental local-axial probe, global inverse uniqueness, or continuum accuracy. The source/readout factorization and known bare operator response are essential; explicit leakage controls show bias even when residuals appear small.

- [Detailed Korean approach and results](detailed_research_journal.md)
- [English manuscript section](new_observation_section.tex)
- [Observation profiles and raw signal scale](figures_final/observation_profiles.pdf)
- [Noise recovery and failure regime](figures_final/observation_noise.pdf)
- [Mesh bias and transfer violations](figures_final/observation_limits.pdf)
- [Reproduction instructions](reproducibility.md)
- [Primary machine-readable evidence](validation/diagnostic.json)
- [Unconstrained-frequency and covariance controls](nuisance_robustness/diagnostic.json)
- [Frozen prospective design](frozen_design.json)
- [Delivery manifest](delivery_manifest.json)

Original four-band data, old source files, manuscripts and research companions were not edited. The primary1000 noise realizations generated1500 estimator outputs; the additional100 weak-signal realizations generated200 outputs. These are deterministic computational checks under an explicit synthetic noise model, not independent physical measurements.