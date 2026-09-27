"""Independent source/readout directions using preserved full-band responses."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "prb_mixed_source_response_20260914"
sys.path.insert(0, str(OLD))
from mixed_vertex_response import apply_transfer, contract_mixed, dress_response
from noise_sensitivity import complex_noise, transfers
from response_derivatives import contracted_derivatives

DESIGN = json.loads((ROOT / "frozen_design.json").read_text(encoding="utf-8"))
PAIRS = {"full": ((0, 0), (0, 1), (1, 0), (1, 1)),
         "diagonal": ((0, 0), (1, 1))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]).copy() for key in ("omega", "bare")}
        result["Q"] = float(data["Q"])
    if result["bare"].shape != (len(result["omega"]), 3, 3):
        raise ValueError("Expected full three-operator response")
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("Nonfinite response input")
    return result


def physical(data, g, ua, lr, ls):
    return contract_mixed(dress_response(data["bare"], data["Q"], g, ua), lr, ls)


def clean_observations(data, directions):
    return {field: apply_transfer(physical(item, .8, 8., *directions),
                                  *transfers(item["omega"]))
            for field, item in data.items()}


def real_residual(predictions, observations, fields, observation, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Positive finite absolute complex noise RMS required")
    differences = np.concatenate([
        np.stack([(predictions[field][:, a, b] - observations[field][:, a, b])
                  for a, b in PAIRS[observation]], axis=-1).ravel()
        for field in fields]) / sigma
    return np.r_[differences.real, differences.imag] / np.sqrt(differences.size)


def fit(data, observations, fields, observation, source_model, directions,
        sigma, *, fixed_ua=None, nstarts=5, jacobian=True):
    """Profile affine complex gains independently in each field.

    Physical optimizer coordinates are g/0.8, Ua/8, lambda_r, lambda_s.
    These same scales define the reported, nuisance-projected Jacobian.
    """
    if source_model not in ("pure", "known", "shared", "independent"):
        raise ValueError("Unknown source model")
    ninteraction = 2 if fixed_ua is None else 1
    ndirection = {"pure": 0, "known": 0, "shared": 1, "independent": 2}[source_model]
    nphysical = ninteraction + ndirection
    basis = {field: np.column_stack((np.ones_like(data[field]["omega"]),
        (data[field]["omega"] - data[field]["omega"].mean()) /
        np.ptp(data[field]["omega"]))) for field in fields}

    def unpack(v):
        g = .8 * v[0]
        ua = 8. * v[1] if fixed_ua is None else fixed_ua
        if source_model == "pure":
            lr, ls = 0., 0.
        elif source_model == "known":
            lr, ls = directions
        elif source_model == "shared":
            lr = ls = v[ninteraction]
        else:
            lr, ls = v[ninteraction:nphysical]
        return g, ua, lr, ls

    cache = {}

    def evaluate(v):
        if "v" in cache and np.array_equal(cache["v"], v):
            return cache["residual"], cache["jacobian"]
        parameters = unpack(v)
        deltas, derivative_blocks = [], []
        for index, field in enumerate(fields):
            coefficients = v[nphysical + 8*index:nphysical + 8*(index+1)].reshape(2, 2, 2)
            d = basis[field] @ (coefficients[0, 0] + 1j*coefficients[0, 1])
            s = basis[field] @ (coefficients[1, 0] + 1j*coefficients[1, 1])
            item = data[field]
            g, ua, lr, ls = parameters
            dressed = dress_response(item["bare"], item["Q"], g, ua)
            matrix = contract_mixed(dressed, lr, ls)
            predicted = apply_transfer(matrix, d, s)
            dg, du, dr, ds = contracted_derivatives(dressed, item["Q"], lr, ls)
            derivatives = [.8*dg] + ([8.*du] if fixed_ua is None else [])
            if source_model == "shared":
                derivatives += [dr+ds]
            elif source_model == "independent":
                derivatives += [dr, ds]
            jac = np.zeros(matrix.shape + (len(v),), complex)
            for column, derivative in enumerate(derivatives):
                jac[..., column] = apply_transfer(derivative, d, s)
            for side in (0, 1):
                for imaginary, phase in enumerate((1., 1j)):
                    for power in (0, 1):
                        column = nphysical + 8*index + 4*side + 2*imaginary + power
                        factor = phase*basis[field][:, power]
                        if side == 0:
                            jac[:, 1, 0, column] = matrix[:, 1, 0]*factor
                            jac[:, 1, 1, column] = matrix[:, 1, 1]*s*factor
                        else:
                            jac[:, 0, 1, column] = matrix[:, 0, 1]*factor
                            jac[:, 1, 1, column] = matrix[:, 1, 1]*d*factor
            deltas.append(np.stack([predicted[:, a, b]-observations[field][:, a, b]
                                  for a, b in PAIRS[observation]], axis=1).ravel())
            derivative_blocks.append(np.stack([jac[:, a, b] for a, b in PAIRS[observation]],
                                               axis=1).reshape(-1, len(v)))
        delta = np.concatenate(deltas)
        jac = np.concatenate(derivative_blocks)
        scale = sigma*np.sqrt(delta.size)
        cache.update(v=np.asarray(v).copy(),
                     residual=np.r_[delta.real, delta.imag]/scale,
                     jacobian=np.vstack((jac.real, jac.imag))/scale)
        return cache["residual"], cache["jacobian"]

    def residual(v):
        return evaluate(v)[0]

    lower = [.25] + ([0.] if fixed_ua is None else []) + [-1.]*ndirection
    upper = [2.] + ([3.] if fixed_ua is None else []) + [1.]*ndirection
    lower += [-np.inf]*(8*len(fields))
    upper += [np.inf]*(8*len(fields))
    attempts = []
    for g0, ua0, lr0, ls0 in DESIGN["multistarts"][:nstarts]:
        start = [g0/.8] + ([ua0/8.] if fixed_ua is None else [])
        if source_model == "shared":
            start += [lr0]
        elif source_model == "independent":
            start += [lr0, ls0]
        parameters = unpack(np.asarray(start))
        for field in fields:
            model = physical(data[field], *parameters)
            coefficients = np.zeros((2, 2, 2))
            if observation == "full":
                for index, (a, b) in enumerate(((1, 0), (0, 1))):
                    gain = np.linalg.lstsq(model[:, a, b, None]*basis[field],
                                           observations[field][:, a, b], rcond=None)[0]
                    coefficients[index, 0], coefficients[index, 1] = gain.real, gain.imag
            else:
                # Diagonal fits may use only the observed diagonal entries.
                coefficients[0, 0, 0] = 1.
                gain = np.linalg.lstsq(model[:, 1, 1, None]*basis[field],
                                       observations[field][:, 1, 1], rcond=None)[0]
                coefficients[1, 0], coefficients[1, 1] = gain.real, gain.imag
            start += coefficients.ravel().tolist()
        attempts.append(least_squares(residual, start, jac=lambda v: evaluate(v)[1], bounds=(lower, upper),
            xtol=2e-10, ftol=2e-10, gtol=2e-10, max_nfev=500))
    best_index = min(range(len(attempts)), key=lambda i: np.linalg.norm(attempts[i].fun))
    best = attempts[best_index]
    g, ua, lr, ls = unpack(best.x)
    bound_hits = [name for name, value, low, high in
                  zip(("g", "Ua", "lambda_r", "lambda_s"),
                      (g, ua, lr, ls), (.2, 0., -1., -1.), (1.6, 24., 1., 1.))
                  if min(abs(value-low), abs(value-high)) <= 1e-5*(high-low)]
    result = dict(g=float(g), Ua=float(ua), lambda_r=float(lr), lambda_s=float(ls),
        rms_noise_units=float(np.linalg.norm(best.fun)), success=bool(best.success),
        all_starts_success=all(item.success for item in attempts),
        evaluations=sum(item.nfev for item in attempts), bound_hits=";".join(bound_hits),
        best_start=best_index, global_minimum_certified=False)
    if jacobian:
        nuisance = best.jac[:, nphysical:]
        u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
        rank = int(np.count_nonzero(s > s[0]*1e-10)) if len(s) and s[0] else 0
        projected = best.jac[:, :nphysical] - u[:, :rank] @ (
            u[:, :rank].T @ best.jac[:, :nphysical])
        singular = np.linalg.svd(projected, compute_uv=False)
        result.update(jacobian_singular_values=singular.tolist(), nuisance_rank=rank,
                      physical_parameter_count=nphysical,
                      local_condition_number=float(singular[0]/singular[-1])
                      if singular[-1] > 0 else None)
    result["start_solutions"] = [dict(zip(("g", "Ua", "lambda_r", "lambda_s"),
        map(float, unpack(item.x))), rms=float(np.linalg.norm(item.fun)),
        success=bool(item.success)) for item in attempts]
    return result


def scalar_row(context, result):
    return {**context, **{key: value for key, value in result.items()
                         if key not in ("start_solutions", "jacobian_singular_values")}}


def inputs():
    paths = {field: OLD / "production" / (field + "_final") / "response_3channel.npz"
             for field in ("B0", "Bplus")}
    return paths, {field: load(path) for field, path in paths.items()}


def run(output, phase, mesh_input=None, reference_input=None, field="Bplus"):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    paths, data = inputs()
    sources = [Path(__file__), ROOT / "frozen_design.json", ROOT / "response_derivatives.py",
               OLD / "mixed_vertex_response.py", OLD / "analyze_inference.py"]
    hashes = {str(path): sha(path) for path in [*sources, *paths.values()]}
    truths = {name: clean_observations(data, directions)
              for name, directions in DESIGN["truths"].items()}
    common_rms = float(np.sqrt(np.mean(np.concatenate([
        np.abs(value).ravel()**2 for truth in truths.values() for value in truth.values()]))))
    sigma = DESIGN["epsilon"] * common_rms
    results, rows = [], []
    if phase == "deterministic":
        for truth_name, directions in DESIGN["truths"].items():
            for group, fields in DESIGN["field_groups"].items():
                for observation in DESIGN["observations"]:
                    for model in DESIGN["source_models"]:
                        context = dict(truth=truth_name, group=group,
                                       observation=observation, source_model=model)
                        result = fit(data, truths[truth_name], fields, observation,
                                     model, directions, sigma)
                        results.append({**context, **result})
                        rows.append(scalar_row(context, result))
                        print(context, result["rms_noise_units"], flush=True)
        write_csv(output / "fits.csv", rows)
        profiles = []
        for truth_name, directions in DESIGN["truths"].items():
            cases = [("Bplus", "full"), ("joint", "full")]
            if truth_name == "symmetric":
                cases += [("Bplus", "diagonal")]
            for group, observation in cases:
                for ua in DESIGN["Ua_profile"]:
                    context = dict(truth=truth_name, group=group,
                                   observation=observation, Ua_fixed=ua)
                    result = fit(data, truths[truth_name], DESIGN["field_groups"][group],
                        observation, "independent", directions, sigma,
                        fixed_ua=ua, nstarts=3, jacobian=False)
                    profiles.append(scalar_row(context, result))
                    print("profile", context, result["rms_noise_units"], flush=True)
        write_csv(output / "profiles.csv", profiles)
    elif phase == "noise":
        # Identical field noise is reused by all policies in each realization.
        rng = np.random.default_rng(DESIGN["seed"])
        for repetition in range(DESIGN["noise_repetitions"]):
            observations = {name: {f: value + complex_noise(rng, value.shape, sigma)
                for f, value in truth.items()} for name, truth in truths.items()}
            for truth_name, group, observation, model in DESIGN["noise_cases"]:
                context = dict(truth=truth_name, group=group, observation=observation,
                               source_model=model, realization=repetition)
                result = fit(data, observations[truth_name], DESIGN["field_groups"][group],
                    observation, model, DESIGN["truths"][truth_name], sigma,
                    nstarts=DESIGN["noise_multistarts"], jacobian=False)
                rows.append(scalar_row(context, result))
            print("noise realization", repetition+1, flush=True)
        write_csv(output / "estimates.csv", rows)
        for case in DESIGN["noise_cases"]:
            name, group, observation, model = case
            selected = [row for row in rows if (row["truth"], row["group"],
                row["observation"], row["source_model"]) == tuple(case)]
            summary = dict(truth=name, group=group, observation=observation,
                           source_model=model, count=len(selected),
                           successes=sum(row["success"] for row in selected),
                           boundary_hits=sum(bool(row["bound_hits"]) for row in selected))
            for parameter, target in zip(("g", "Ua", "lambda_r", "lambda_s"),
                                          (.8, 8., *DESIGN["truths"][name])):
                values = np.array([row[parameter] for row in selected])
                summary[parameter] = dict(bias=float(np.mean(values)-target),
                    rmse=float(np.sqrt(np.mean((values-target)**2))),
                    quantiles=np.quantile(values, [.025, .5, .975]).tolist())
            results.append(summary)
    else:
        if mesh_input is None or reference_input is None:
            raise ValueError("Mesh phase requires --input and --reference")
        model_data, reference_data = load(mesh_input), load(reference_input)
        if not np.array_equal(model_data["omega"], reference_data["omega"]):
            raise ValueError("Mesh comparisons require identical frequencies")
        if model_data["Q"] != reference_data["Q"]:
            raise ValueError("Mesh comparison momentum differs")
        hashes.update({str(path): sha(path) for path in (mesh_input, reference_input)})
        for name, directions in DESIGN["truths"].items():
            observed = clean_observations({field: reference_data}, directions)
            for observation in ("diagonal", "full"):
                for mode in ("known", "shared", "independent"):
                    context = dict(truth=name, field=field, observation=observation,
                                   source_model=mode, model=str(mesh_input),
                                   reference=str(reference_input))
                    result = fit({field: model_data}, observed, [field], observation,
                                 mode, directions, sigma, nstarts=3)
                    results.append({**context, **result})
                    rows.append(scalar_row(context, result))
        write_csv(output / "mesh_bias.csv", rows)
    report = dict(phase=phase, design=DESIGN, absolute_sigma=sigma,
        common_clean_complex_rms=common_rms, results=results,
        elapsed_seconds=time.perf_counter()-started, input_source_sha256=hashes,
        inputs_sources_unchanged=all(sha(path) == digest for path, digest in hashes.items()),
        global_identifiability_certified=False, statistical_coverage_claimed=False)
    dump(output / "summary.json", report)
    dump(output / "manifest.json", {path.name: sha(path) for path in output.iterdir()
                                    if path.is_file() and path.name != "manifest.json"})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("deterministic", "noise", "mesh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--field", choices=("B0", "Bplus"), default="Bplus")
    args = parser.parse_args()
    report = run(args.output.resolve(), args.phase, args.input, args.reference, args.field)
    print("Completed", report["phase"], report["elapsed_seconds"], flush=True)
