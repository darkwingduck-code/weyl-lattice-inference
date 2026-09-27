"""Small conditional RPA continuation; no new lattice response calculation."""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / "research/prb_mixed_source_response_20260914"
sys.path.insert(0, str(OLD))
from mixed_vertex_response import dress_response

UA = np.geomspace(8., 1e4, 81)
G = .8
METHODS = ("full_calibrated", "full_profiled", "axial_calibrated", "axial_profiled")


def realify(value):
    flat = np.asarray(value).ravel()
    return np.r_[flat.real, flat.imag]


def project(vector, columns, tolerance=1e-11):
    matrix = np.column_stack(columns)
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 0] / norms[norms > 0]
    if not matrix.shape[1]:
        return vector, 0
    basis, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    basis = basis[:, :rank]
    return vector - basis @ (basis.T @ vector), rank


def information(chi, derivative, sigma, tolerance=1e-11):
    full = realify(derivative)
    axial = realify(derivative[:, 1, 1])
    nuisance = []
    for side in ("row", "column"):
        for index in range(2):
            change = np.zeros_like(chi)
            if side == "row":
                change[:, index, :] = chi[:, index, :]
            else:
                change[:, :, index] = chi[:, :, index]
            nuisance += [realify(change), realify(1j * change)]
    full_reduced, full_rank = project(full, nuisance, tolerance)
    axial_reduced, axial_rank = project(axial, [realify(chi[:, 1, 1]),
                                               realify(1j * chi[:, 1, 1])], tolerance)
    vectors = (full, full_reduced, axial, axial_reduced)
    output = {name: float(2 * (value @ value) / sigma**2)
              for name, value in zip(METHODS, vectors)}
    output.update(full_nuisance_rank=full_rank, axial_nuisance_rank=axial_rank)
    return output


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("run_01"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.parent != ROOT:
        raise ValueError("Output must be a direct child of this research directory.")
    if output.exists():
        raise FileExistsError("Preserve the completed run; select a new --output directory.")
    paths = {field: OLD / "production" / (field + "_final") / "response_3channel.npz"
             for field in ("B0", "Bplus")}
    data = {}
    for field, path in paths.items():
        with np.load(path) as saved:
            data[field] = dict(omega=saved["omega"].copy(),
                               bare=saved["bare"][:, :2, :2].copy(), Q=float(saved["Q"]))
    calibration = [dress_response(d["bare"], d["Q"], G, 8.) for d in data.values()]
    sigma = .001 * np.sqrt(np.mean(np.concatenate([abs(c).ravel()**2 for c in calibration])))
    rows, checks, tail = [], {}, {}
    for field, item in data.items():
        bare, q = item["bare"], item["Q"]
        inverse = np.linalg.inv(bare)
        inverse[:, 0, 0] -= G / q**2
        alpha, beta = inverse[:, 0, 0], inverse[:, 0, 1]
        gamma, delta = inverse[:, 1, 0], inverse[:, 1, 1]
        z = delta - gamma * beta / alpha
        field_checks = dict(frequency_count=len(item["omega"]),
                            minimum_abs_alpha=float(min(abs(alpha))),
                            maximum_abs_z=float(max(abs(z))),
                            cross_product_norm=float(np.linalg.norm(beta * gamma)),
                            cross_to_diagonal_bare_norm=float(np.linalg.norm(bare[:, 0, 1]) /
                                                             np.linalg.norm(bare[:, 1, 1])),
                            derivative_relative_errors={}, inverse_response_max_relative_error=0.,
                            projection_tolerance_max_relative_change=0.,
                            full_nuisance_ranks=[], axial_nuisance_ranks=[])
        for u in UA:
            chi = dress_response(bare, q, G, float(u))
            derivative = chi[:, :, 1, None] * chi[:, None, 1, :]
            direct_matrix = inverse.copy()
            direct_matrix[:, 1, 1] -= u
            direct = np.linalg.inv(direct_matrix)
            discrepancy = np.linalg.norm(chi - direct) / np.linalg.norm(direct)
            field_checks["inverse_response_max_relative_error"] = max(
                field_checks["inverse_response_max_relative_error"], float(discrepancy))
            values = information(chi, derivative, sigma)
            for tol in (1e-10, 1e-12):
                alternate = information(chi, derivative, sigma, tol)
                change = max(abs(alternate[k] / values[k] - 1) for k in METHODS if values[k] > 0)
                field_checks["projection_tolerance_max_relative_change"] = max(
                    field_checks["projection_tolerance_max_relative_change"], float(change))
            assert values["full_profiled"] <= values["full_calibrated"] * (1 + 1e-10)
            assert values["axial_profiled"] <= values["axial_calibrated"] * (1 + 1e-10)
            assert all(np.isfinite(values[k]) and values[k] >= 0 for k in METHODS)
            rows.append(dict(field=field, Ua=float(u), sigma=float(sigma), **values))
            field_checks["full_nuisance_ranks"].append(values["full_nuisance_rank"])
            field_checks["axial_nuisance_ranks"].append(values["axial_nuisance_rank"])
        for u in (8., 80.):
            chi = dress_response(bare, q, G, u)
            analytic = chi[:, :, 1, None] * chi[:, None, 1, :]
            step = 1e-4 * u
            finite = (dress_response(bare, q, G, u + step) -
                      dress_response(bare, q, G, u - step)) / (2 * step)
            field_checks["derivative_relative_errors"][str(u)] = float(
                np.linalg.norm(analytic - finite) / np.linalg.norm(analytic))
        field_checks["full_nuisance_ranks"] = sorted(set(field_checks["full_nuisance_ranks"]))
        field_checks["axial_nuisance_ranks"] = sorted(set(field_checks["axial_nuisance_ranks"]))
        assert max(field_checks["derivative_relative_errors"].values()) < 1e-5
        assert field_checks["inverse_response_max_relative_error"] < 1e-9
        assert field_checks["projection_tolerance_max_relative_change"] < 1e-6
        checks[field] = field_checks
        subset = [r for r in rows if r["field"] == field and r["Ua"] >= 1e3]
        tail[field] = {}
        for name in METHODS:
            positive = [r for r in subset if r[name] > 0]
            if len(positive) >= 3:
                x = np.log([r["Ua"] for r in positive])
                y = np.log([r[name] for r in positive])
                slope, intercept = np.polyfit(x, y, 1)
                residual = y - (slope * x + intercept)
                tail[field][name] = dict(log_slope=float(slope), samples=len(x),
                                         log_fit_rms=float(np.sqrt(np.mean(residual**2))))
    output.mkdir()
    with (output / "information.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [Path(__file__), ROOT / "plan.md", OLD / "mixed_vertex_response.py",
               REPO / "research/prb_lattice_response/lattice_response.py",
               REPO / "research/prb_lattice_response/parameter_sector_sampling.py", *paths.values()]
    report = dict(scope="mathematical fixed-Pi RPA continuation, not physical validity at large Ua",
                  Coulomb_g=G, scan_count=len(UA), Ua_min=float(UA[0]), Ua_max=float(UA[-1]),
                  absolute_complex_sigma=float(sigma), noise_convention="E|epsilon|^2=sigma^2",
                  true_external_gains="all unity, independent of Ua and frequency",
                  profiled_full_gains="independent constant complex gain for both rows and both columns",
                  fixed_parameters="bare Pi, Coulomb g, operator, grid and noise covariance",
                  information="total local Gaussian Fisher for Ua; factor 2/sigma^2; no confidence-coverage claim",
                  tail_min_Ua=1000, checks=checks, tail=tail,
                  sha256={str(p.relative_to(REPO)): sha(p) for p in sources}, status="PASS")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True, constrained_layout=True)
    styles = (("#1f77b4", "-"), ("#1f77b4", "--"), ("#d95f02", "-"), ("#d95f02", ":"))
    labels = ("All entries, calibrated", "All entries, gains profiled",
              "Axial auto, calibrated", "Axial auto, gain profiled")
    for axis, field in zip(axes, data):
        subset = [r for r in rows if r["field"] == field]
        for name, (color, style), label in zip(METHODS, styles, labels):
            axis.loglog(UA, [r[name] for r in subset], color=color, linestyle=style, label=label)
        axis.set(title=field, xlabel=r"$U_a$ (fixed-$\Pi$ continuation)")
        axis.grid(alpha=.2, which="major")
    axes[0].set_ylabel(r"Total local Fisher information $I_{U_aU_a}$")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.savefig(output / "screening_information.pdf")
    plt.close(fig)
    print(json.dumps(dict(sigma=float(sigma), checks=checks, tail=tail), indent=2))


if __name__ == "__main__":
    main()
