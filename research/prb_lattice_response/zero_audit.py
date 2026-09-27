"""Independent numerical audit of a frozen finite RPA mode/zero CSV catalog.

Recomputes Pi and Pi' from raw Bloch transitions, not the production pole/zero
helpers. Numerical consistency is separate from a complete scalar pole/zero
certificate or any continuum/collectivity claim.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
LIMITS = dict(normalized_zero_residual=1e-7, newton_to_pole_distance=1e-4,
              bare_distance_relative=1e-8, dielectric_condition=1e12,
              matrix_residue_relative=1e-6, kernel_residual=1e-10,
              kernel_separation=1e-8, moment0_relative=1e-10)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows_from_bytes(value):
    return list(csv.DictReader(value.decode("utf-8-sig").splitlines()))


def matrix_from_row(row):
    return np.array([[complex(float(row[f"residue_{i}{j}_real"]),
                              float(row[f"residue_{i}{j}_imag"]))
                      for j in range(2)] for i in range(2)])


def raw_response(tr, z):
    """Direct full transition sum and analytic derivative; no weight cutoff."""
    distance = complex(z) - np.asarray(tr.delta)
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    if np.any(distance[active] == 0):
        raise ValueError("Evaluation at an active bare transition")
    d = distance[active]
    weights = np.empty((len(d), 2, 2), dtype=complex)
    weights[:, 0, 0], weights[:, 1, 1] = tr.cc[active], tr.aa[active]
    weights[:, 0, 1], weights[:, 1, 0] = tr.ca[active], tr.ca[active].conj()
    return (np.einsum("nab,n->ab", weights, 1/d),
            -np.einsum("nab,n->ab", weights, 1/d**2))


def direct_rpa(tr, u, z):
    pi, dpi = raw_response(tr, z)
    left, right = np.eye(2) - pi @ u, np.eye(2) - u @ pi
    chi = np.linalg.solve(left, pi)
    derivative = np.linalg.solve(left, dpi) @ np.linalg.solve(right, np.eye(2))
    return pi, dpi, chi, derivative, float(np.linalg.cond(left))


def zero_check(tr, u, zero, poles, residues, weights):
    zero = float(zero)
    visible = weights > 0
    pp, ww = poles[visible], weights[visible]
    d = zero - pp
    nearest = float(np.min(abs(d)))
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    bare_distance = float(np.min(abs(zero - tr.delta[active])))
    result = dict(zero=zero, nearest_modal_pole_distance=nearest,
                  nearest_bare_distance=bare_distance)
    if nearest == 0:
        return dict(result, status="FAIL_CANDIDATE_IS_MODAL_POLE")
    phi = float(np.sum(ww/d))
    phi_scale = float(np.sum(abs(ww/d)))
    phi_prime = -float(np.sum(ww/d**2))
    modal_newton = abs(phi/phi_prime) if phi_prime else None
    result.update(weighted_transfer_absolute=abs(phi),
                  weighted_transfer_normalized=abs(phi)/max(phi_scale, np.finfo(float).tiny),
                  weighted_newton_location_estimate=modal_newton,
                  weighted_newton_to_pole_distance=modal_newton/nearest if modal_newton is not None else None)
    if bare_distance <= LIMITS["bare_distance_relative"]*max(1., abs(zero)):
        return dict(result, status="UNRESOLVED_NEAR_BARE_TRANSITION")
    try:
        pi, dpi, chi, derivative, condition = direct_rpa(tr, u, zero)
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(result, status="UNRESOLVED_DIRECT_SOLVE", reason=str(error))
    result["dielectric_condition"] = condition
    if not np.isfinite(condition) or condition > LIMITS["dielectric_condition"]:
        return dict(result, status="UNRESOLVED_DIELECTRIC_CONDITION")
    cc_scale = float(np.sum(abs(residues[:, 0, 0]/(zero-poles))))
    normalized = float(abs(chi[0, 0])/max(cc_scale, np.finfo(float).tiny))
    correction = float(abs(chi[0, 0]/derivative[0, 0])) if derivative[0, 0] != 0 else None
    detpi = np.linalg.det(pi)
    numerator = pi[0, 0] - u[1, 1]*detpi
    numerator_scale = abs(pi[0, 0]) + abs(u[1, 1]*detpi)
    result.update(direct_charge_absolute=float(abs(chi[0, 0])),
                  direct_charge_normalized=normalized,
                  direct_newton_location_estimate=correction,
                  direct_newton_to_pole_distance=correction/nearest if correction is not None else None,
                  direct_charge_derivative_real=float(derivative[0, 0].real),
                  cofactor_numerator_normalized=float(abs(numerator)/max(numerator_scale, np.finfo(float).tiny)))
    passed = (normalized <= LIMITS["normalized_zero_residual"]
              and result["weighted_transfer_normalized"] <= LIMITS["normalized_zero_residual"]
              and correction is not None and correction/nearest <= LIMITS["newton_to_pole_distance"]
              and modal_newton is not None and modal_newton/nearest <= LIMITS["newton_to_pole_distance"])
    result["status"] = "PASS_NUMERICAL_ZERO" if passed else "FAIL_NUMERICAL_ZERO"
    return result


def residue_check(tr, u, pole, expected):
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    distance = float(np.min(abs(pole-tr.delta[active])))
    result = dict(pole=float(pole), nearest_bare_distance=distance)
    if distance <= 1e-11*max(1., abs(pole)):
        return dict(result, status="UNRESOLVED_BARE_COLLISION")
    pi, dpi = raw_response(tr, pole)
    values, rotation = np.linalg.eigh(u)
    sq = (rotation*np.sqrt(values)) @ rotation.conj().T
    k, dk = np.eye(2)-sq@pi@sq, -sq@dpi@sq
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    order = np.argsort(abs(eigenvalues))
    kernel_error, other = float(abs(eigenvalues[order[0]])), float(abs(eigenvalues[order[1]]))
    result.update(kernel_residual=kernel_error, other_kernel_eigenvalue_absolute=other)
    if kernel_error > LIMITS["kernel_residual"]:
        return dict(result, status="FAIL_NOT_A_REGULAR_ROOT")
    if other <= LIMITS["kernel_separation"]:
        return dict(result, status="UNRESOLVED_MULTIPLICITY")
    v = eigenvectors[:, order[0]]
    slope = float(np.real(v.conj() @ dk @ v))
    if pole*slope <= 0:
        return dict(result, status="FAIL_RESIDUE_METRIC", derivative_metric=slope)
    amplitude = pi @ sq @ v
    actual = np.outer(amplitude, amplitude.conj())/slope
    difference = actual-expected
    scale = float(np.linalg.norm(expected))
    relative = float(np.linalg.norm(difference)/max(scale, np.finfo(float).tiny))
    charge_scale, axial_scale = float(abs(expected[0, 0])), float(abs(expected[1, 1]))
    result.update(matrix_absolute_error=float(np.max(abs(difference))),
                  matrix_relative_error=relative, derivative_metric=slope,
                  charge_absolute_error=float(abs(difference[0, 0])),
                  charge_relative_error=float(abs(difference[0, 0])/charge_scale) if charge_scale else None,
                  axial_relative_error=float(abs(difference[1, 1])/axial_scale) if axial_scale else None)
    result["status"] = ("PASS_NUMERICAL_RESIDUE" if relative <= LIMITS["matrix_residue_relative"]
                        else "FAIL_NUMERICAL_RESIDUE")
    return result


def eigenpair_audit(tr, u, csv_poles, csv_residues):
    """Independent numpy eigensolve, with raw weights and no mode cutoff."""
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    delta = np.asarray(tr.delta)[active]
    sign = np.sign(delta)
    if np.any(sign == 0) or len(delta) > 3500:
        raise ValueError("Dense audit requires nonzero energies and at most 3500 active transitions")
    cc, aa, ca = tr.cc[active], tr.aa[active], tr.ca[active]
    g = np.empty((2, len(delta)), complex)
    for i in range(len(delta)):
        a, b, c = sign[i]*cc[i], sign[i]*aa[i], sign[i]*ca[i]
        if a >= b:
            g[0, i] = np.sqrt(a)
            g[1, i] = c.conjugate()/g[0, i]
        else:
            g[1, i] = np.sqrt(b)
            g[0, i] = c/g[1, i]
    factor_error = max(float(np.max(abs(sign*abs(g[0])**2-cc))),
                       float(np.max(abs(sign*abs(g[1])**2-aa))),
                       float(np.max(abs(sign*g[0]*g[1].conj()-ca))))
    h = np.diag(abs(delta)).astype(complex)+g.conj().T @ u @ g
    lower = np.linalg.cholesky(h)
    cholesky_relative = float(np.linalg.norm(h-lower@lower.conj().T)/np.linalg.norm(h))
    hermitian = (lower.conj().T*sign[None, :]) @ lower
    hermitian_defect = float(np.max(abs(hermitian-hermitian.conj().T)))
    values, vectors = np.linalg.eigh(hermitian)
    norm_a = float(np.max(abs(values)))
    residual_matrix = hermitian@vectors - vectors*values[None, :]
    residual_norms = np.linalg.norm(residual_matrix, axis=0)
    relative_residuals = residual_norms/(norm_a+abs(values))
    orthogonal = vectors.conj().T@vectors-np.eye(len(values))
    orthogonal_max = float(np.max(abs(orthogonal)))
    orthogonal_fro = float(np.linalg.norm(orthogonal))
    f = np.linalg.solve(lower.conj(), g.T).T
    amplitudes = f@vectors
    recomputed = np.einsum("an,bn,n->nab", amplitudes, amplitudes.conj(), values)
    if len(values) != len(csv_poles) or np.any(np.diff(csv_poles) < 0):
        raise ValueError("CSV spectrum length/order differs from complete realization")
    eig_difference = abs(values-csv_poles)
    individual = []
    for i in range(len(values)):
        absolute = float(np.max(abs(recomputed[i]-csv_residues[i])))
        scale = float(np.linalg.norm(csv_residues[i]))
        individual.append(dict(index=i, eigenvalue=float(values[i]),
                               absolute_eigenpair_residual=float(residual_norms[i]),
                               relative_eigenpair_residual=float(relative_residuals[i]),
                               csv_eigenvalue_absolute_difference=float(eig_difference[i]),
                               raw_basis_residue_absolute_difference=absolute,
                               raw_basis_residue_relative_difference=float(np.linalg.norm(recomputed[i]-csv_residues[i])/scale) if scale else None))
    # Close groups serve numerical comparison only; no declaration of degeneracy.
    groups = []
    edges = [0]+list(np.flatnonzero(np.diff(csv_poles)>1e-8)+1)+[len(values)]
    for left, right in zip(edges[:-1], edges[1:]):
        expected, actual = csv_residues[left:right].sum(axis=0), recomputed[left:right].sum(axis=0)
        scale = float(np.linalg.norm(expected))
        groups.append(dict(first_index=int(left), end_index_exclusive=int(right),
                           size=int(right-left), frequency_min=float(csv_poles[left]),
                           frequency_max=float(csv_poles[right-1]),
                           residue_absolute_difference=float(np.max(abs(actual-expected))),
                           residue_relative_difference=float(np.linalg.norm(actual-expected)/scale) if scale else None))
    eig_relative = float(np.max(eig_difference)/max(1., norm_a))
    result = dict(status="INDEPENDENT_EIGENPAIR_AUDIT_COMPLETED", retained_transitions=len(delta),
                  factorization_absolute_error=factor_error, cholesky_relative_frobenius_error=cholesky_relative,
                  hermitian_max_defect=hermitian_defect,
                  maximum_eigenpair_relative_residual=float(np.max(relative_residuals)),
                  orthogonality_max_entry_error=orthogonal_max, orthogonality_frobenius_error=orthogonal_fro,
                  csv_eigenvalue_max_absolute_difference=float(np.max(eig_difference)),
                  csv_eigenvalue_max_relative_difference=eig_relative,
                  eigensystem_numerical_checks_pass=bool(np.max(relative_residuals)<=1e-12 and
                     orthogonal_max<=1e-12 and eig_relative<=1e-11),
                  comparison_group_separation=1e-8, comparison_groups=len(groups),
                  maximum_group_residue_absolute_difference=max(row["residue_absolute_difference"] for row in groups),
                  scope="Every eigenpair retained. Close comparison groups are not asserted to be exactly degenerate; raw individual-basis residues may rotate.")
    return result, individual, groups


def self_test():
    """Closed-form two-oscillator zero, displaced-zero and wrong-residue controls."""
    energies = np.array([.7, 1.3, 1.8])
    strengths = np.array([.08, .04, .05])
    tr = SimpleNamespace(delta=np.ravel(np.column_stack((energies, -energies))),
                         cc=np.array([.08, -.08, .04, -.04, 0., 0.]),
                         aa=np.array([0., 0., 0., 0., .05, -.05]),
                         ca=np.zeros(6, complex))
    u = np.diag([.4, .3])
    a, b = 2*energies[:2]*strengths[:2]
    zero_squared = (a*energies[1]**2+b*energies[0]**2)/(a+b)
    total = energies[0]**2+energies[1]**2+u[0, 0]*(a+b)
    product = energies[0]**2*energies[1]**2+u[0, 0]*(a+b)*zero_squared
    positive = np.sqrt(np.sort(np.roots([1., -total, product])))
    poles = np.array([-positive[1], -positive[0], positive[0], positive[1]])
    residues = np.zeros((4, 2, 2), complex)
    for i, pole in enumerate(poles):
        residues[i, 0, 0] = (a+b)*(pole*pole-zero_squared)/(2*pole*(2*pole*pole-total))
    weights = (poles*residues[:, 0, 0]).real
    good = zero_check(tr, u, np.sqrt(zero_squared), poles, residues, weights)
    bad = zero_check(tr, u, np.sqrt(zero_squared)+1e-3, poles, residues, weights)
    residue_good = residue_check(tr, u, poles[-1], residues[-1])
    residue_bad = residue_check(tr, u, poles[-1], residues[-1]*1.1)
    assert good["status"] == "PASS_NUMERICAL_ZERO", good
    assert bad["status"] == "FAIL_NUMERICAL_ZERO", bad
    assert residue_good["status"] == "PASS_NUMERICAL_RESIDUE", residue_good
    assert residue_bad["status"] == "FAIL_NUMERICAL_RESIDUE", residue_bad
    z = .41+.17j
    _, _, _, analytic, _ = direct_rpa(tr, u, z)
    step = 1e-6
    finite = (direct_rpa(tr, u, z+step)[2]-direct_rpa(tr, u, z-step)[2])/(2*step)
    derivative_error = float(np.max(abs(analytic-finite)))
    assert derivative_error < 1e-8, derivative_error
    result = dict(status="SELF_TEST_PASS", true_zero=good, displaced_zero=bad,
                  true_residue=residue_good, perturbed_residue=residue_bad,
                  finite_difference_derivative_error=derivative_error)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(source, output):
    from lattice_response import Parameters, interaction_matrix
    from bloch_response import bloch_spectrum, bloch_transitions
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Audit output must be a new empty directory")
    names = ["finite_poles.csv", "finite_charge_zeros.csv", "diagnostic.json"]
    initial = {name: (source/name).read_bytes() for name in names}
    diagnostic = json.loads(initial["diagnostic.json"])
    finite = diagnostic["finite_realization"]
    declared = diagnostic.get("output_sha256", {})
    for name in names[:2]:
        if name in declared and sha_bytes(initial[name]) != declared[name]:
            raise RuntimeError("Production CSV hash mismatch: "+name)
    source_hashes = {name: sha_bytes((ROOT/name).read_bytes())
                     for name in diagnostic["source_sha256"]}
    if source_hashes != diagnostic["source_sha256"]:
        raise RuntimeError("Production source hashes no longer match the frozen inputs")
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"self_test.json", self_test())
    evidence = output/"inputs"
    evidence.mkdir()
    for name, value in initial.items():
        (evidence/name).write_bytes(value)
    print("[stage] frozen CSV bytes and source hashes verified", flush=True)
    parameters = Parameters(**finite["parameters"])
    settings = diagnostic["settings"]
    qshift_float = settings["q"]*parameters.Nkz/(2*np.pi)
    qshift = int(round(qshift_float))
    if abs(qshift-qshift_float) > 1e-12:
        raise ValueError("Requested q is not an integer grid shift")
    tr = bloch_transitions(bloch_spectrum(parameters), settings["mu"], settings["T"], qshift)
    u = interaction_matrix(tr.Q, settings["g"], settings["Ua"])
    modal_rows = rows_from_bytes(initial["finite_poles.csv"])
    poles = np.array([float(row["pole"]) for row in modal_rows])
    residues = np.array([matrix_from_row(row) for row in modal_rows])
    weights = np.array([float(row["weighted_charge"]) for row in modal_rows])
    zero_rows = rows_from_bytes(initial["finite_charge_zeros.csv"])
    audit = []
    for index, row in enumerate(zero_rows):
        status = row["status"]
        if status == "resolved_charge_zero":
            checked = zero_check(tr, u, float(row["zero"]), poles, residues, weights)
        elif status == "multiplication_zero":
            origin = float(row["zero"])
            _, _, chi0, _, condition = direct_rpa(tr, u, 0.)
            passed = abs(origin) <= 1e-10 and abs(chi0[0, 0]) > 1e-12
            checked = dict(status="PASS_MULTIPLICATION_ZERO" if passed else "FAIL_MULTIPLICATION_ZERO",
                           zero=origin, chi_cc_at_origin_real=float(chi0[0, 0].real),
                           chi_cc_at_origin_imag=float(chi0[0, 0].imag),
                           dielectric_condition_at_origin=condition)
        else:
            checked = dict(status="PRESERVED_"+status.upper())
        audit.append(dict(index=index, original_status=status,
                          left_pole=float(row["left_pole"]), right_pole=float(row["right_pole"]), **checked))
    print(f"[stage] independently evaluated {len(audit)} zero intervals", flush=True)
    residue_audit = []
    for check in finite["regular_residue_checks"]:
        pole = float(check["pole"])
        candidates = np.flatnonzero(poles == pole)
        if len(candidates) != 1:
            result = dict(pole=pole, status="UNRESOLVED_EXACT_MULTIPLICITY", multiplicity=len(candidates))
        else:
            i = int(candidates[0])
            nearby = np.flatnonzero((abs(poles-pole) <= 1e-8) & (np.arange(len(poles)) != i))
            if len(nearby):
                result = dict(pole=pole, status="UNRESOLVED_NEARBY_MODAL_EIGENVALUES",
                              nearby_count=len(nearby))
            else:
                result = residue_check(tr, u, pole, residues[i])
        result["production_check_status"] = check["status"]
        residue_audit.append(result)
    reconstructed = []
    for z in [.17+.13j, .43+.19j, 1.07+.07j, 2.31+.11j, -.31+.23j]:
        actual = direct_rpa(tr, u, z)[2]
        from_modes = np.einsum("nab,n->ab", residues, 1/(z-poles))
        reconstructed.append(dict(z_real=z.real, z_imag=z.imag,
                                  absolute_error=float(np.max(abs(actual-from_modes))),
                                  relative_error=float(np.linalg.norm(actual-from_modes)/np.linalg.norm(actual))))
    m0, m1 = tr.moments()
    modal_m0, modal_m1 = residues.sum(axis=0), np.einsum("n,nab->ab", poles, residues)
    identity = dict(raw_moment0_error=float(np.max(abs(m0))),
                    modal_moment0_error=float(np.max(abs(modal_m0))),
                    first_moment_absolute_error=float(np.max(abs(modal_m1-m1))),
                    charge_weight_relation_max_error=float(np.max(abs(weights-poles*residues[:, 0, 0].real))),
                    factorization_assessment="No re-factorization or modal diagonalization in this audit")
    counts = dict(Counter(row["status"] for row in audit))
    resolved = [row for row in audit if row["original_status"] == "resolved_charge_zero"]
    zero_pass = bool(resolved) and all(row["status"] == "PASS_NUMERICAL_ZERO" for row in resolved)
    residue_pass = bool(residue_audit) and all(row["status"] == "PASS_NUMERICAL_RESIDUE" for row in residue_audit)
    frozen_unchanged = all((source/name).read_bytes() == initial[name] for name in names[:2])
    current_hashes = {name: sha_bytes((ROOT/name).read_bytes()) for name in source_hashes}
    if not frozen_unchanged or current_hashes != source_hashes:
        raise RuntimeError("Frozen CSV/source bytes changed during the audit")
    print("[stage] starting independent dense eigensystem audit", flush=True)
    eigensystem, eigenpairs, groups = eigenpair_audit(tr, u, poles, residues)
    dump(output/"eigensystem.json", eigensystem)
    dump(output/"eigenpairs.json", eigenpairs)
    dump(output/"residue_comparison_groups.json", groups)
    print("[stage] independent eigenpairs and raw modal CSV compared", flush=True)
    if any((source/name).read_bytes() != initial[name] for name in names[:2]):
        raise RuntimeError("Frozen CSV changed during eigensystem audit")
    if {name: sha_bytes((ROOT/name).read_bytes()) for name in source_hashes} != source_hashes:
        raise RuntimeError("Frozen source changed during eigensystem audit")
    dump(output/"zero_checks.json", audit)
    dump(output/"residue_checks.json", residue_audit)
    result = dict(status="NUMERICAL_AUDIT_COMPLETED", completed_utc=datetime.now(timezone.utc).isoformat(),
                  production_status_at_start=diagnostic["status"],
                  source_directory=str(source), input_sha256={k: sha_bytes(v) for k, v in initial.items()},
                  source_sha256=source_hashes, audit_script_sha256=sha_bytes(Path(__file__).read_bytes()),
                  limits=LIMITS, zero_status_counts=counts, resolved_candidates_count=len(resolved),
                  all_previously_resolved_zeros_pass=zero_pass,
                  selected_residue_count=len(residue_audit), all_selected_residues_pass=residue_pass,
                  reconstruction_checks=reconstructed, moment_checks=identity,
                  eigensystem=eigensystem,
                  input_csv_and_source_bytes_unchanged=True,
                  complete_scalar_zero_catalog_certified=False,
                  submission_ready=False, continuum_or_collectivity_certified=False,
                  scope=["Existing installed Python/numpy/scipy environment; no environment reinstallation.",
                         "All raw transition slots are regenerated; only identically zero matrix weights are skipped during sums.",
                         "No nonzero modal charge weight is removed or classified as structurally dark.",
                         "Newton correction is a local numerical error estimate, not an interval-arithmetic proof.",
                         "Cofactor term-cancellation ratio is supplementary; acceptance uses the dressed spectral scale.",
                         "Near bare transitions, unresolved modal intervals and multiplicities remain explicit.",
                         "The running production diagnostic is snapshotted; its later updates are not treated as an input mutation.",
                         "Numerical audit completion is not a blanket scientific PASS."])
    dump(output/"diagnostic.json", result)
    manifest = {str(p.relative_to(output)).replace("\\", "/"): sha_bytes(p.read_bytes())
                for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
    dump(output/"manifest.json", manifest)
    print(json.dumps({k: result[k] for k in ("status", "zero_status_counts",
          "all_previously_resolved_zeros_pass", "all_selected_residues_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"mode_analysis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.output is None:
            parser.error("--output is required for a production audit")
        main(args.source.resolve(), args.output.resolve())
