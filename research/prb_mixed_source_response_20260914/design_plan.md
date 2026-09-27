# Mixed source/readout response: frozen calculation plan

## Purpose

The existing PRB calculation treats charge and the local axial bilinear
`Gamma5` as the measured response basis. Symmetry also permits `Gamma3` in a
probe vertex. This calculation tests a concrete, limited question: whether
using `O_s = Gamma5 + lambda_s Gamma3` as the source and
`O_r = Gamma5 + lambda_r Gamma3` as the readout changes the inferred Coulomb
and axial couplings. It does not add a `Gamma3` interaction.

The design in `frozen_design.json` was fixed before inspecting the pilot. The
primary same-source/readout coefficient is `lambda=0.7`; `0` and `-0.7` are
predeclared controls. Every control is reported, including adverse results.

## Response construction

At every magnetic Bloch block and longitudinal momentum transfer, calculate
the full-band bare retarded matrix

```text
Pi_ab(z,q) = (1/V) sum_knm
  [f(E_kn)-f(E_k+q,m)] <kn|Gamma_a|k+q,m>
  <k+q,m|Gamma_b|kn> / [z-(E_k+q,m-E_kn)]
```

for `a,b in (charge,Gamma5,Gamma3)`. The implementation streams one magnetic
block and a small frequency batch at a time. It retains every particle-hole
pair and uses the existing stable Fermi-difference routine.

Only the charge/Gamma5 block is dressed:

```text
U = diag(g/Q^2, Ua),  Q=2 sin(q/2)
chi = Pi + Pi[:,I] (1-U Pi[I,I])^-1 U Pi[I,:], I=(charge,Gamma5).
```

The measurable two-channel kernel is `R^dagger chi S`, where the second
columns of `R` and `S` are `(0,1,lambda_r)` and `(0,1,lambda_s)`. This order is
essential: the interaction is applied in the microscopic charge/Gamma5 basis,
then external vertices are contracted.

## Validation gates

1. Compare the upper-left 2x2 bare kernel with the established implementation.
2. Compare contraction of the 3x3 kernel with an independent direct
   calculation using `Gamma5+lambda Gamma3`.
3. Check finite-q charge-current vertex identity and both Ward contractions.
4. Check the zeroth spectral moment and weighted retarded passivity.
5. Check magnetic-cell phase invariance on a complete finite torus.
6. For existing production meshes, compare the charge/Gamma5 block against the
   saved result before interpreting the new Gamma3 entries.
7. Verify transfer closure under independent complex source/readout gains.

The numeric tolerances are frozen in `frozen_design.json`. Passing a mesh
convergence test and passing detector precision are separate requirements.

## Inference experiment

Synthetic truth uses `(g,Ua,lambda)=(0.8,8,0.7)`. The invariant fit uses
`chi_cc` and `W=chi_cO chi_Oc/chi_OO`, quantities unchanged by independent
complex gains on the mixed-source row and column. Raw-response fits include
independent affine complex gains in frequency. The analysis compares the
correct direction, pure `Gamma5`, the sign-reversed direction, fixed-`Ua`
hypotheses, a 21-point lambda scan, and a three-parameter local Jacobian.

Exact recovery with the correct fixed lambda is a software self-consistency
gate. Joint recovery of lambda is reported as an identifiability result, not
required to pass: a flat or multimodal direction is itself relevant evidence.

## Evidence threshold

The pilot is an end-to-end plumbing and invariant check only. It remains marked
`scientific_claim_ready=false`. Manuscript interpretation requires at least
B+ coarse/final agreement, B0 final comparison, all numerical gates, and an
explicit separation of mesh stability from parameter-detection precision.

## Resource envelope

Historical two-channel runs used about 17 min for B+ coarse, 34 min for B+
final, 3 min for B0 coarse, and 12 min for B0 final. Adding all nine operator
products and Ward rows/columns increases contraction work; eigensystem cost is
unchanged. Plan about 1.5--3 times those wall times with BLAS threads pinned to
one, and retain at least 3 GiB free RAM for B+ production. The runner records
wall time, resident memory, hashes, raw pair counts, and all gates.
