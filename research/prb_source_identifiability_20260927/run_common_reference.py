"""Fit lower meshes to fixed higher observations using all five frozen starts."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

from run_study import (DESIGN, OLD, ROOT, clean_observations, dump, fit, inputs,
                       load, real_residual, scalar_row, sha, write_csv)


def run(output, model_paths, reference_paths):
    if not model_paths or model_paths.keys() != reference_paths.keys():
        raise ValueError("Supply a model and reference for the same field(s)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline_paths, baseline = inputs()
    sources = [Path(__file__), ROOT / "run_study.py", ROOT / "frozen_design.json",
               ROOT / "response_derivatives.py", OLD / "mixed_vertex_response.py",
               OLD / "analyze_inference.py", OLD / "noise_sensitivity.py"]
    hashes = {str(path): sha(path) for path in [*sources, *baseline_paths.values(),
              *model_paths.values(), *reference_paths.values()]}
    # The noise scale remains fixed by the original four clean truth datasets.
    baseline_truths = [clean_observations(baseline, directions)
                       for directions in DESIGN["truths"].values()]
    common_rms = float(np.sqrt(np.mean(np.concatenate([
        np.abs(value).ravel()**2 for truth in baseline_truths
        for value in truth.values()]))))
    sigma = DESIGN["epsilon"] * common_rms
    model_data = {field: load(path) for field, path in model_paths.items()}
    reference_data = {field: load(path) for field, path in reference_paths.items()}
    for field in model_data:
        if not np.array_equal(model_data[field]["omega"], reference_data[field]["omega"]):
            raise ValueError("Model and reference frequencies differ")
        if model_data[field]["Q"] != reference_data[field]["Q"]:
            raise ValueError("Model and reference momenta differ")
    fields = list(model_data)
    rows, results = [], []
    for name, directions in DESIGN["truths"].items():
        observed = clean_observations(reference_data, directions)
        unfitted_truth = clean_observations(model_data, directions)
        for observation in DESIGN["observations"]:
            for mode in ("known", "shared", "independent"):
                context = dict(truth=name, group="joint" if len(fields) == 2 else fields[0],
                               observation=observation, source_model=mode)
                result = fit(model_data, observed, fields, observation, mode,
                             directions, sigma, nstarts=len(DESIGN["multistarts"]))
                truth_feasible = mode != "shared" or directions[0] == directions[1]
                truth_rms = float(np.linalg.norm(real_residual(unfitted_truth,
                    observed, fields, observation, sigma))) if truth_feasible else None
                bound_pass = (result["rms_noise_units"] <= truth_rms +
                              1e-7 * max(1., truth_rms)) if truth_feasible else None
                result.update(unfitted_truth_rms=truth_rms,
                    optimizer_upper_bound_pass=bound_pass,
                    valid_for_mesh_bias=truth_feasible and result["success"] and bound_pass)
                results.append({**context, **result})
                rows.append(scalar_row(context, result))
                print(context, result["Ua"], result["rms_noise_units"], flush=True)
    write_csv(output / "mesh_bias.csv", rows)
    report = dict(absolute_sigma=sigma, common_clean_complex_rms=common_rms,
        model_paths={field: str(path) for field, path in model_paths.items()},
        reference_paths={field: str(path) for field, path in reference_paths.items()},
        number_of_starts=len(DESIGN["multistarts"]),
        protocol="All five original frozen starts are used uniformly. The earlier three-start mesh run missed the near-truth basin for symmetric full B0 with independently fitted directions; that result is preserved. This extension is a numerical sensitivity check, not a new truth or noise design.",
        truth_upper_bound="For a correctly specified direction family, the unfitted lower-mesh model at the known synthetic truth and exact transfer functions is a feasible objective upper bound. A larger optimized residual marks numerical failure, not mesh bias. The truth is used only as a diagnostic, not a fitted estimate. Unequal truth is not feasible in the shared-direction family.",
        design=DESIGN, results=results, elapsed_seconds=time.perf_counter()-started,
        input_source_sha256=hashes,
        inputs_sources_unchanged=all(sha(path) == digest for path, digest in hashes.items()),
        global_identifiability_certified=False, statistical_coverage_claimed=False)
    dump(output / "summary.json", report)
    dump(output / "manifest.json", {path.name: sha(path) for path in output.iterdir()
         if path.is_file() and path.name != "manifest.json"})
    print("Completed", report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    for field in ("b0", "bplus"):
        parser.add_argument("--input-" + field, type=Path)
        parser.add_argument("--reference-" + field, type=Path)
    args = parser.parse_args()
    model_paths, reference_paths = {}, {}
    for field, suffix in (("B0", "b0"), ("Bplus", "bplus")):
        model_path = getattr(args, "input_" + suffix)
        reference_path = getattr(args, "reference_" + suffix)
        if model_path is not None:
            model_paths[field] = model_path.resolve()
        if reference_path is not None:
            reference_paths[field] = reference_path.resolve()
    run(args.output.resolve(), model_paths, reference_paths)
