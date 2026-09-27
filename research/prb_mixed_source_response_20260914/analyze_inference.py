"""Analyze operator-direction error using one generated three-channel response."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from mixed_vertex_response import (
    apply_transfer, closure, contract_mixed, dress_response, scaled_max_error)

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def physical(bare, Q, g, ua, lr, ls=None):
    return contract_mixed(dress_response(bare, Q, g, ua), lr, ls)

def invariant_vector(model, truth):
    scales = (np.linalg.norm(truth[:, 0, 0]),
              np.linalg.norm(closure(truth)))
    values = np.r_[(model[:, 0, 0] - truth[:, 0, 0]) / scales[0],
                   (closure(model) - closure(truth)) / scales[1]]
    return np.r_[values.real, values.imag] / np.sqrt(2)

def fixed_lambda_fit(bare, Q, truth, lam, ua=None):
    def residual(v):
        g, u = (v[0], v[1]) if ua is None else (v[0], ua)
        return invariant_vector(physical(bare, Q, g, u, lam), truth)
    starts = ((.6, 4.), (.8, 8.), (1., 16.))
    fits = []
    for g, u in starts:
        start = [g, u] if ua is None else [g]
        bounds = ([.2, 0.], [1.6, 24.]) if ua is None else ([.2], [1.6])
        fits.append(least_squares(residual, start, bounds=bounds,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400))
    fit = min(fits, key=lambda value: np.linalg.norm(value.fun))
    return {"g": float(fit.x[0]),
            "ua": float(fit.x[1] if ua is None else ua),
            "rms": float(np.linalg.norm(fit.fun)),
            "success": bool(fit.success),
            "all_starts_success": all(value.success for value in fits),
            "global_minimum_certified": False}

def unknown_lambda_fit(bare, Q, truth):
    def residual(v):
        return invariant_vector(physical(bare, Q, v[0], v[1], v[2]), truth)
    starts = ((.6, 4., -.7), (.8, 8., 0.), (.8, 8., .7),
              (1., 16., .7))
    fits = [least_squares(residual, start,
              bounds=([.2, 0., -1.], [1.6, 24., 1.]),
              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=600)
            for start in starts]
    fit = min(fits, key=lambda value: np.linalg.norm(value.fun))
    return {"g": float(fit.x[0]), "ua": float(fit.x[1]),
            "lambda": float(fit.x[2]), "rms": float(np.linalg.norm(fit.fun)),
            "success": bool(fit.success),
            "all_starts_success": all(value.success for value in fits),
            "global_minimum_certified": False}

def raw_affine_fit(bare, omega, Q, observed, sigma, lam, fit_lambda=False):
    x = (omega - omega.mean()) / np.ptp(omega)
    basis = np.column_stack((np.ones_like(x), x))
    def unpack(v):
        offset = 3 if fit_lambda else 2
        g, ua = v[:2]
        model_lam = v[2] if fit_lambda else lam
        z = v[offset:].reshape(2, 2, 2)
        d = basis @ (z[0, 0] + 1j * z[0, 1])
        s = basis @ (z[1, 0] + 1j * z[1, 1])
        return g, ua, model_lam, d, s
    def residual(v):
        g, ua, model_lam, d, s = unpack(v)
        model = apply_transfer(physical(bare, Q, g, ua, model_lam), d, s)
        delta = (model - observed) / sigma
        # sigma is complex RMS: E|noise|^2 = sigma^2, not per-quadrature RMS.
        return np.r_[delta.real.ravel(), delta.imag.ravel()] / np.sqrt(
            observed.size)
    fits = []
    for g, ua, l0 in ((.6, 4., 0.), (.8, 8., .7), (1., 16., 0.)):
        model = physical(bare, Q, g, ua, l0 if fit_lambda else lam)
        coeff = np.zeros((2, 2, 2))
        for k, (a, b) in enumerate(((1, 0), (0, 1))):
            gain = np.linalg.lstsq(model[:, a, b, None] * basis,
                                   observed[:, a, b], rcond=None)[0]
            coeff[k, 0] = gain.real; coeff[k, 1] = gain.imag
        start = np.r_[g, ua, l0, coeff.ravel()] if fit_lambda else np.r_[
            g, ua, coeff.ravel()]
        low = np.r_[.2, 0., -1., np.full(8, -np.inf)] if fit_lambda else np.r_[
            .2, 0., np.full(8, -np.inf)]
        high = np.r_[1.6, 24., 1., np.full(8, np.inf)] if fit_lambda else np.r_[
            1.6, 24., np.full(8, np.inf)]
        fits.append(least_squares(residual, start, bounds=(low, high),
            xtol=2e-11, ftol=2e-11, gtol=2e-11, max_nfev=500))
    fit = min(fits, key=lambda value: np.linalg.norm(value.fun))
    g, ua, model_lam, _, _ = unpack(fit.x)
    return {"g": float(g), "ua": float(ua), "lambda": float(model_lam),
            "rms_noise_units": float(np.linalg.norm(fit.fun)),
            "success": bool(fit.success),
            "all_starts_success": all(value.success for value in fits),
            "global_minimum_certified": False}

def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def component_convergence(actual, reference):
    names = ("charge", "Gamma5", "Gamma3")
    result = {}
    for a, left in enumerate(names):
        for b, right in enumerate(names):
            delta = np.abs(actual[:, a, b] - reference[:, a, b])
            scale = float(np.max(np.abs(reference[:, a, b])))
            result[f"{left}_{right}"] = {
                "maximum_absolute_error": float(delta.max()),
                "normalized_sup_error": (float(delta.max()/scale)
                                          if scale > 0 else None)}
    return result

def convergence_metrics(bare, omega, Q, reference_path):
    reference_data = np.load(reference_path)
    reference_omega = reference_data["omega"]
    reference_Q = float(reference_data["Q"])
    if not np.array_equal(omega, reference_omega):
        raise ValueError("Convergence reference uses a different frequency grid")
    if abs(Q-reference_Q) > 2e-15:
        raise ValueError("Convergence reference uses a different momentum")
    reference_bare = reference_data["bare"]
    dressed = dress_response(bare, Q, .8, 8.)
    reference_dressed = dress_response(reference_bare, Q, .8, 8.)
    mixed = contract_mixed(dressed, .7)
    reference_mixed = contract_mixed(reference_dressed, .7)
    return {
        "reference": str(reference_path),
        "bare_3channel_scaled_max_error": scaled_max_error(
            bare, reference_bare),
        "dressed_3channel_scaled_max_error": scaled_max_error(
            dressed, reference_dressed),
        "mixed_lambda_0p7_scaled_max_error": scaled_max_error(
            mixed, reference_mixed),
        "closure_scaled_max_error": scaled_max_error(
            closure(mixed), closure(reference_mixed)),
        "bare_component_errors": component_convergence(bare, reference_bare),
        "dressed_component_errors": component_convergence(
            dressed, reference_dressed),
        "acceptance_threshold_predeclared": False,
        "interpretation": "Continuous coarse-to-final diagnostics only; no post hoc pass threshold."
    }

def main(input_path, output, reference_path=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty analysis directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_paths = [Path(__file__), ROOT / "mixed_vertex_response.py",
                    ROOT / "frozen_design.json", input_path]
    if reference_path is not None:
        source_paths.append(reference_path)
    hashes = {str(path): sha(path) for path in source_paths}
    data = np.load(input_path)
    omega, bare = data["omega"], data["bare"]
    Q = float(data["Q"])
    truth_lambda = .7
    truth = physical(bare, Q, .8, 8., truth_lambda)
    x = (omega - omega.mean()) / np.ptp(omega)
    d = .7 * np.exp(.4j) * (1 + .1 * x)
    s = 1.3 * np.exp(-.25j) * (1 + .1j * x)
    observed = apply_transfer(truth, d, s)
    closure_error = scaled_max_error(closure(observed), closure(truth))
    sigma = .001 * np.linalg.norm(truth[:, 1, 1]) / np.sqrt(len(omega))
    fits = []
    for method, lam in (("correct_lambda", .7), ("pure_Gamma5_model", 0.),
                        ("sign_reversed_model", -.7)):
        inv = fixed_lambda_fit(bare, Q, truth, lam)
        raw = raw_affine_fit(bare, omega, Q, observed, sigma, lam)
        fits.append({"method": method, "lambda_model": lam,
                     **{f"invariant_{k}": v for k, v in inv.items()},
                     **{f"raw_{k}": v for k, v in raw.items()}})
    unknown_inv = unknown_lambda_fit(bare, Q, truth)
    unknown_raw = raw_affine_fit(bare, omega, Q, observed, sigma, 0.,
                                 fit_lambda=True)
    hypotheses = []
    for lam in (-.7, 0., .7):
        for ua in (4., 8., 16.):
            result = fixed_lambda_fit(bare, Q, truth, lam, ua=ua)
            hypotheses.append({"lambda_model": lam, "ua_fixed": ua, **result})
    scan = []
    for lam in np.linspace(-1, 1, 21):
        result = fixed_lambda_fit(bare, Q, truth, float(lam))
        scan.append({"lambda_model": float(lam), **result})
    def base_vector(g, ua, lam):
        return invariant_vector(physical(bare, Q, g, ua, lam), truth)
    columns = []
    for plus, minus, step in (((.8*np.exp(1e-5), 8., .7),
                                (.8*np.exp(-1e-5), 8., .7), 2e-5),
                               ((.8, 8.*np.exp(1e-5), .7),
                                (.8, 8.*np.exp(-1e-5), .7), 2e-5),
                               ((.8, 8., .7+1e-5),
                                (.8, 8., .7-1e-5), 2e-5)):
        columns.append((base_vector(*plus) - base_vector(*minus)) / step)
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    convergence = (convergence_metrics(bare, omega, Q, reference_path)
                   if reference_path is not None else None)
    write_csv(output / "fixed_lambda_fits.csv", fits)
    write_csv(output / "fixed_ua_hypotheses.csv", hypotheses)
    write_csv(output / "lambda_scan.csv", scan)
    # Joint lambda recovery is an inference result, not a software gate: a
    # flat/multimodal lambda direction is precisely what this study must expose.
    numerical_pass = (closure_error < 5e-12
        and abs(fits[0]["invariant_g"] - .8) < 1e-7
        and abs(fits[0]["invariant_ua"] - 8.) < 1e-7
        and fits[0]["invariant_rms"] < 1e-9)
    report = {
        "status": "MIXED_SOURCE_INFERENCE_PASS" if numerical_pass else
                  "MIXED_SOURCE_INFERENCE_VALIDATION_FAILED",
        "truth": {"g": .8, "ua": 8., "lambda_source": .7,
                  "lambda_readout": .7},
        "transfer_closure_scaled_error": closure_error,
        "fixed_lambda_fits": fits, "unknown_lambda_invariant_fit": unknown_inv,
        "unknown_lambda_raw_affine_fit": unknown_raw,
        "mesh_convergence": convergence,
        "local_invariant_jacobian_singular_values": singular.tolist(),
        "local_condition_number": float(singular[0] / singular[-1]),
        "elapsed_seconds": time.perf_counter() - started,
        "input_source_sha256": hashes,
        "inputs_sources_unchanged": all(sha(path) == value
                                         for path, value in zip(source_paths,
                                                                hashes.values())),
        "scientific_claim_ready": False,
        "global_optimum_certified": False,
        "limitations": [
            "Pilot data are a finite mesh, not production evidence.",
            "Lambda is a real local-operator coefficient, not a material vertex.",
            "No statistical confidence level or global inverse theorem is claimed.",
            "The fitted interaction still acts only in charge/Gamma5 space."]}
    (output / "diagnostic.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        path.name: sha(path) for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"}, indent=2) + "\n",
        encoding="utf-8")
    if report["status"] != "MIXED_SOURCE_INFERENCE_PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    main(args.input.resolve(), args.output.resolve(),
         args.reference.resolve() if args.reference is not None else None)
