# Conditional large-interaction information continuation

Design fixed before this calculation. Read the existing B0_final and
Bplus_final three-channel NPZ files, retain their charge/Gamma5 block, and
reuse `dress_response`. No new Kubo calculation or data fitting is involved.
The Coulomb parameter is fixed at g=0.8. The scan is 81 logarithmically spaced
Ua values from 8 through 10000. This is a mathematical continuation of the
stated RPA response, not a claim of physical validity at large interaction.

The true external row and column gains are unity and independent of Ua and
frequency. One absolute circular-complex noise standard deviation is fixed
as 0.001 times the RMS of all four complex response entries, both fields,
at Ua=8. It stays fixed over the scan. No noise realizations are needed for
the Gaussian Fisher information calculation.

Compare (1) all four entries with calibrated gains, (2) all four entries
after profiling independent constant complex gains on both rows and both
columns, (3) calibrated axial auto-response, and (4) axial auto-response
after profiling one unknown constant complex gain. The Coulomb parameter,
bare response, frequency grid, and source operator are held known. Report
total information, not an average over observations; the channel budgets
therefore differ. Complex noise obeys E|epsilon|^2=sigma^2, giving the factor
2/sigma^2 on realified derivative norms.

Use the analytic derivative dchi/dUa=chi[:,a]chi[a,:]. Project nuisance
columns with a real SVD after normalizing nonzero columns, using a relative
rank threshold 1e-11. Repeat at thresholds 1e-10 and 1e-12 as a numerical
check. Verify the derivative against centered differences at Ua=8,80 and
the response against the direct 2x2 inverse. Fit reported tail log slopes
only to Ua>=1000; do not force theoretical exponents or interpret a zero
leading coefficient as evidence for a generic scaling. Save every scan
row, inputs/source hashes, numerical checks, one PDF, and an English TeX
derivation/results fragment. Use one process and one BLAS thread.
