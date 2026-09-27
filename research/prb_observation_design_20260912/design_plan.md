# Observation design: prospective plan, 2026-09-12

This new study reuses frozen four-band equilibrium response data. No Hamiltonian, occupations, quadrature, manuscript, or previous results are altered.

## Questions and controls

1. Does adding q, field, or chemical-potential conditions recover Ua when the complex axial diagonal response has an unknown nonnegative real gain, complex gain, or a weak frequency-dependent transfer?
2. Does measuring the complete charge/local-axial response matrix add a calibration-invariant constraint? The candidate is W=Y_ca Y_ac/Y_aa, for Y=diag(1,d) chi diag(1,s). Here axial source s and readout d may be nonzero complex functions; charge calibration is known. The same s,d must act consistently on all matrix entries at each condition/frequency. This is a synthetic operator-level protocol, not a demonstrated material probe.
3. Is the putative improvement larger than empirical mesh differences and noise amplification in weak mixed channels? No relative-channel normalization by itself establishes accessibility.

## Pilot and freeze

Pilot uses only the final meshes, eta=.015, truth(g,Ua)=(.8,8), alternative Ua=4 and16, and conditions B0/Bplus baseline/qhalf/qdouble/mu_low/mu_high. Compare diagonal common/per-condition real/complex gains and complex affine frequency transfer against calibrated charge plus W. Each dataset/channel is normalized by its truth L2 norm; same number of complex frequencies, equal sample weights (not quadrature). We report the raw cross-channel amplitudes and W amplitude separately. Candidate windows fixed_lab and scaled_q remain distinct, and eta datasets never enter one joint fit. All pilot outcomes are retained.

Before held-out production, freeze a selected design and criteria in frozen_design.json using pilot evidence only. The final design must retain failure/negative controls, and use the simplest single or multi-control condition showing meaningful improvement. Do not call a minimum global: profile all sampled g minima plus endpoints and a common g=.8 candidate, g in [.2,1.6].

Held-out validation: Ua truths4/12/16 and changed g(.6,1.0), eta=.0075/.03 separately, and the available higher meshes (B0 qhalf/base/qdouble; Bplus eta_low). Compare models with one common higher reference, including Ua8, rather than canceling error through own-mesh truth. Add finite synthetic noise replicas with explicit independent circular-complex noise and both per-channel relative noise and a common absolute floor; no experimental confidence interval is claimed. Independently validate cofactor/inverse identities, transfer cancellation, and failure when independent entry gains violate the source/readout factorization.

## Inputs and limits

- ../prb_lattice_response/parameter_sector_sampling/calibration_inputs.json: final14, checked source/output hashes, coarse references.
- ../prb_lattice_response/detector_boundary_refinement/B0_higher and Bplus_higher:4 higher datasets, preserved incomplete transverse certification.
- Frozen calibration/readers may be imported read-only; new logic and results remain in this folder.

No new large diagonalization is planned. Initial cost estimate:10-15min to pilot;30-60min including validation and reporting, revised after measured timings. Cached response fitting has small memory use. H runtime/caches and one BLAS thread only.

The numerical kernel comparison tolerance .001 is not detector-relative precision. Higher grids are comparison references, not exact continuum data. No new gap, experimental Gamma5 source, global inverse uniqueness, or statistical coverage theorem.