"""Predeclared absolute-complex-noise sensitivity for mixed source/readout fits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analyze_inference import physical, raw_affine_fit
from mixed_vertex_response import apply_transfer

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "noise_design.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfers(omega):
    x = (np.asarray(omega) - np.mean(omega)) / np.ptp(omega)
    readout = .7 * np.exp(.4j) * (1 + .1*x)
    source = 1.3 * np.exp(-.25j) * (1 + .1j*x)
    return readout, source


def complex_noise(rng, shape, sigma):
    return (sigma/np.sqrt(2)) * (
        rng.standard_normal(shape) + 1j*rng.standard_normal(shape))


def distribution_summary(values, truth):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    delta = finite - truth
    quantiles = np.quantile(finite, [.025, .16, .5, .84, .975])
    return {
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(values)-len(finite)),
        "bias": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "sample_standard_deviation": (float(np.std(finite, ddof=1))
                                      if len(finite) > 1 else 0.0),
        "quantiles": {key: float(value) for key, value in zip(
            ("q025", "q16", "q50", "q84", "q975"), quantiles)}}


def parameter_summary(values, truth, bounds, boundary_fraction):
    result = distribution_summary(values, truth)
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    low, high = bounds
    tolerance = boundary_fraction * (high-low)
    lower = int(np.count_nonzero(finite <= low+tolerance))
    upper = int(np.count_nonzero(finite >= high-tolerance))
    result.update({
        "bounds": [low, high], "boundary_tolerance": tolerance,
        "lower_bound_hits": lower, "upper_bound_hits": upper,
        "lower_bound_hit_rate": float(lower/len(finite)),
        "upper_bound_hit_rate": float(upper/len(finite))})
    return result


def load_response(path):
    with np.load(path) as data:
        return (np.asarray(data["omega"]), np.asarray(data["bare"]),
                float(data["Q"]))


def one_realization(task):
    (field, epsilon, realization, seed_state, sigma, omega, Q,
     clean, final_bare, coarse_bare) = task
    rng = np.random.default_rng(seed_state)
    noise = complex_noise(rng, clean.shape, sigma)
    observed = clean + noise
    rows = []
    for model_mesh, bare in (("final", final_bare), ("coarse", coarse_bare)):
        for lambda_mode, fit_lambda in (("known", False), ("joint", True)):
            fit = raw_affine_fit(bare, omega, Q, observed, sigma, .7,
                                 fit_lambda=fit_lambda)
            rows.append({
                "field": field, "epsilon": epsilon,
                "realization": realization, "model_mesh": model_mesh,
                "lambda_mode": lambda_mode, "g": fit["g"],
                "Ua": fit["ua"], "lambda": fit["lambda"],
                "rms_noise_units": fit["rms_noise_units"],
                "success": fit["success"],
                "all_starts_success": fit["all_starts_success"],
                "global_minimum_certified": False})
    return {"field": field, "epsilon": epsilon,
            "realization": realization,
            "empirical_noise_rms": float(np.sqrt(np.mean(np.abs(noise)**2))),
            "rows": rows}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output, workers):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= workers <= 8:
        raise ValueError("Use 1 through 8 workers")
    started = time.perf_counter()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    response = {}
    paths = {}
    for field in design["fields"]:
        spec = design["inputs"][field]
        final_path = ROOT / spec["truth_mesh"]
        coarse_path = ROOT / spec["coarse_model"]
        omega, final_bare, Q = load_response(final_path)
        coarse_omega, coarse_bare, coarse_Q = load_response(coarse_path)
        if not np.array_equal(omega, coarse_omega) or abs(Q-coarse_Q) > 2e-15:
            raise ValueError(f"{field} final/coarse grids are incompatible")
        readout, source = transfers(omega)
        clean = apply_transfer(physical(final_bare, Q, .8, 8., .7),
                               readout, source)
        response[field] = (omega, Q, clean, final_bare, coarse_bare)
        paths[f"{field}_final"] = final_path
        paths[f"{field}_coarse"] = coarse_path
    all_clean = np.concatenate([response[field][2].ravel()
                                for field in design["fields"]])
    common_rms = float(np.sqrt(np.mean(np.abs(all_clean)**2)))
    source_paths = [Path(__file__), ROOT / "analyze_inference.py",
                    ROOT / "mixed_vertex_response.py", DESIGN_PATH,
                    *paths.values()]
    initial_hashes = {str(path): sha(path) for path in source_paths}
    task_specs = []
    for field in design["fields"]:
        for epsilon_text, count in design["epsilon_realizations"].items():
            for realization in range(int(count)):
                task_specs.append((field, float(epsilon_text), realization))
    child_seeds = np.random.SeedSequence(design["seed"]).spawn(len(task_specs))
    tasks = []
    for spec, child in zip(task_specs, child_seeds):
        field, epsilon, realization = spec
        omega, Q, clean, final_bare, coarse_bare = response[field]
        seed_state = child.generate_state(4).tolist()
        tasks.append((field, epsilon, realization, seed_state,
                      epsilon*common_rms, omega, Q, clean,
                      final_bare, coarse_bare))
    if workers == 1:
        results = [one_realization(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(one_realization, tasks, chunksize=1))
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (row["field"], row["epsilon"],
                               row["realization"], row["model_mesh"],
                               row["lambda_mode"]))
    write_csv(output / "estimates.csv", rows)
    truth = design["truth"]
    bounds = design["parameter_bounds"]
    boundary_fraction = design["boundary_handling"][
        "hit_tolerance_fraction_of_parameter_range"]
    summaries = []
    for field in design["fields"]:
        for epsilon_text in design["epsilon_realizations"]:
            epsilon = float(epsilon_text)
            for model_mesh in ("final", "coarse"):
                for lambda_mode in ("known", "joint"):
                    group = [row for row in rows if row["field"] == field
                             and row["epsilon"] == epsilon
                             and row["model_mesh"] == model_mesh
                             and row["lambda_mode"] == lambda_mode]
                    parameters = {
                        name: parameter_summary(
                            [row[name] for row in group], truth[name],
                            bounds[name], boundary_fraction)
                        for name in (("g", "Ua", "lambda")
                                     if lambda_mode == "joint" else ("g", "Ua"))}
                    residual = distribution_summary(
                        [row["rms_noise_units"] for row in group], 0.)
                    summaries.append({
                        "field": field, "epsilon": epsilon,
                        "model_mesh": model_mesh,
                        "lambda_mode": lambda_mode, "count": len(group),
                        "success_fraction": float(np.mean(
                            [row["success"] for row in group])),
                        "all_starts_success_fraction": float(np.mean(
                            [row["all_starts_success"] for row in group])),
                        "parameters": parameters,
                        "residual_noise_units": residual})
    paired = []
    lookup = {(row["field"], row["epsilon"], row["realization"],
               row["model_mesh"], row["lambda_mode"]): row for row in rows}
    for field in design["fields"]:
        for epsilon_text, count in design["epsilon_realizations"].items():
            epsilon = float(epsilon_text)
            for lambda_mode in ("known", "joint"):
                entry = {"field": field, "epsilon": epsilon,
                         "lambda_mode": lambda_mode, "count": int(count),
                         "coarse_minus_final": {}}
                names = ("g", "Ua", "lambda") if lambda_mode == "joint" else ("g", "Ua")
                for name in names:
                    differences = []
                    for realization in range(int(count)):
                        base = (field, epsilon, realization)
                        coarse = lookup[base+("coarse", lambda_mode)][name]
                        final = lookup[base+("final", lambda_mode)][name]
                        differences.append(coarse-final)
                    entry["coarse_minus_final"][name] = distribution_summary(
                        differences, 0.)
                paired.append(entry)
    noise_checks = []
    tolerance = design["numerical_acceptance"][
        "maximum_empirical_noise_rms_relative_error"]
    for field in design["fields"]:
        for epsilon_text in design["epsilon_realizations"]:
            epsilon = float(epsilon_text)
            values = [result["empirical_noise_rms"] for result in results
                      if result["field"] == field
                      and result["epsilon"] == epsilon]
            empirical = float(np.sqrt(np.mean(np.asarray(values)**2)))
            declared = epsilon*common_rms
            noise_checks.append({
                "field": field, "epsilon": epsilon,
                "declared_absolute_complex_rms": declared,
                "aggregate_empirical_complex_rms": empirical,
                "empirical_to_declared_ratio": empirical/declared,
                "within_predeclared_tolerance":
                    abs(empirical/declared-1) <= tolerance})
    finite = all(np.isfinite(row[name]) for row in rows
                 for name in ("g", "Ua", "lambda", "rms_noise_units"))
    success_fraction = float(np.mean([row["success"] for row in rows]))
    expected = design["numerical_acceptance"]["expected_fit_rows"]
    sources_unchanged = all(sha(path) == value
                            for path, value in zip(source_paths,
                                                   initial_hashes.values()))
    passed = (len(rows) == expected and finite
              and success_fraction >= design["numerical_acceptance"][
                  "minimum_optimizer_success_fraction"]
              and all(item["within_predeclared_tolerance"]
                      for item in noise_checks)
              and sources_unchanged)
    report = {
        "status": ("ABSOLUTE_NOISE_SENSITIVITY_PASS" if passed else
                   "ABSOLUTE_NOISE_SENSITIVITY_VALIDATION_FAILED"),
        "design_id": design["design_id"], "seed": design["seed"],
        "workers": workers, "common_clean_response_rms": common_rms,
        "fit_row_count": len(rows), "expected_fit_rows": expected,
        "overall_optimizer_success_fraction": success_fraction,
        "all_estimates_finite": finite, "noise_checks": noise_checks,
        "group_summaries": summaries, "paired_mesh_effects": paired,
        "elapsed_seconds": time.perf_counter()-started,
        "input_source_sha256": initial_hashes,
        "inputs_sources_unchanged": sources_unchanged,
        "global_minimum_certified": False,
        "manuscript_ready": False,
        "manuscript_readiness_reason": design["claim_rule"],
        "limitations": [
            "The noise is synthetic independent complex Gaussian noise.",
            "Lambda is a real assumed local-operator coefficient, not a measured material vertex.",
            "The reduced 50/25 realization design estimates sensitivity but not publication-grade tail probabilities.",
            "Multistart local least squares does not certify a global optimum."]}
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        path.name: sha(path) for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"}, indent=2)+"\n",
        encoding="utf-8")
    if not passed:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run(args.output.resolve(), args.workers)
    print(json.dumps({key: report[key] for key in (
        "status", "fit_row_count", "overall_optimizer_success_fraction",
        "elapsed_seconds")}, indent=2), flush=True)
