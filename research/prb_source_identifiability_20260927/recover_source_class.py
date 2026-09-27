"""Constructively recover an unordered source class from arbitrary-gain data."""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from run_study import (ROOT, OLD, apply_transfer, contract_mixed, dress_response,
                       dump, inputs, sha)
from noise_sensitivity import complex_noise


def recover(dressed, observations, fields):
    columns, targets = [], []
    for field in fields:
        chi, y = dressed[field], observations[field]
        a, b = chi[:, 0, 1], chi[:, 0, 2]
        c, d, e = chi[:, 1, 1], chi[:, 1, 2], chi[:, 2, 2]
        z, v = y[:, 0, 1]*y[:, 1, 0], y[:, 1, 1]
        columns.append(np.column_stack((z*d-v*a*b, z*e-v*b*b)))
        targets.append(v*a*a-z*c)
    matrix, target = np.concatenate(columns), np.concatenate(targets)
    matrix, target = np.vstack((matrix.real, matrix.imag)), np.r_[target.real, target.imag]
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    p, q = map(float, coefficients)
    discriminant = p*p-4*q
    roots = sorted(((p-np.sqrt(discriminant))/2, (p+np.sqrt(discriminant))/2)) if discriminant >= 0 else None
    return dict(p=p, q=q, discriminant=discriminant, roots=roots, rank=int(rank),
                singular_values=singular.tolist(),
                residual=float(np.linalg.norm(matrix@coefficients-target)))


def charge_fit(data, observations, fields):
    def residual(v):
        delta = np.concatenate([(dress_response(data[f]["bare"], data[f]["Q"],
                                 v[0], v[1])[:, 0, 0]-observations[f][:, 0, 0])
                                for f in fields])
        return np.r_[delta.real, delta.imag]
    fits = [least_squares(residual, start, bounds=([.2, 0.], [1.6, 24.]),
                         xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=500)
            for start in ((.55, 3.), (.8, 10.), (1.1, 16.))]
    best = min(fits, key=lambda fit: np.linalg.norm(fit.fun))
    return dict(g=float(best.x[0]), Ua=float(best.x[1]),
                charge_residual=float(np.linalg.norm(best.fun)),
                best_success=bool(best.success), global_certificate=False)


def run(output):
    design_path = ROOT / "unordered_design.json"
    design = json.loads(design_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    paths, data = inputs()
    sources = [Path(__file__), design_path, ROOT/"run_study.py",
               ROOT/"response_derivatives.py", OLD/"mixed_vertex_response.py",
               OLD/"noise_sensitivity.py", OLD/"analyze_inference.py", *paths.values()]
    hashes = {str(path): sha(path) for path in sources}
    dressed = {f: dress_response(item["bare"], item["Q"], .8, 8.) for f, item in data.items()}
    gains_rng, noise_rng = np.random.default_rng(design["gain_seed"]), np.random.default_rng(design["noise_seed"])
    gains = {f: (.7*gains_rng.uniform(.5, 1.5, len(item["omega"]))*np.exp(
                     1j*gains_rng.uniform(-np.pi, np.pi, len(item["omega"]))),
                 1.3*gains_rng.uniform(.5, 1.5, len(item["omega"]))*np.exp(
                     1j*gains_rng.uniform(-np.pi, np.pi, len(item["omega"]))))
             for f, item in data.items()}
    clean = {name: {f: apply_transfer(contract_mixed(chi, *directions), *gains[f])
                    for f, chi in dressed.items()} for name, directions in design["truths"].items()}
    noiseless, estimates = [], []
    for name, observations in clean.items():
        for group, fields in design["groups"].items():
            result = recover(dressed, observations, fields)
            interaction = charge_fit(data, observations, fields)
            fitted = {f: dress_response(data[f]["bare"], data[f]["Q"],
                                        interaction["g"], interaction["Ua"]) for f in fields}
            noiseless.append(dict(truth=name, group=group, **result,
                                  charge_fit=interaction,
                                  sequential_source_recovery=recover(fitted, observations, fields)))
    for realization in range(design["noise_repetitions"]):
        for name, observations in clean.items():
            noisy = {f: y+complex_noise(noise_rng, y.shape, design["sigma"])
                     for f, y in observations.items()}
            for group, fields in design["groups"].items():
                estimates.append(dict(realization=realization, truth=name, group=group,
                                      **recover(dressed, noisy, fields)))
    summaries = []
    for name, directions in design["truths"].items():
        for group in design["groups"]:
            rows = [r for r in estimates if r["truth"] == name and r["group"] == group]
            summary = dict(truth=name, group=group, count=len(rows),
                           negative_discriminants=sum(r["discriminant"] < 0 for r in rows))
            for parameter, target in (("p", sum(directions)), ("q", np.prod(directions))):
                values = np.array([r[parameter] for r in rows])
                summary[parameter] = dict(bias=float(values.mean()-target),
                    rmse=float(np.sqrt(np.mean((values-target)**2))))
            real_roots = np.array([r["roots"] for r in rows if r["roots"] is not None])
            summary["real_root_count"] = len(real_roots)
            summary["conditional_unordered_root_rmse"] = float(np.sqrt(np.mean(
                (real_roots-np.sort(directions))**2))) if len(real_roots) else None
            summaries.append(summary)
    report = dict(design=design, noiseless=noiseless, noise_summaries=summaries,
        all_noiseless_designs_rank_two=all(row["rank"] == 2 for row in noiseless),
        input_source_sha256=hashes,
        inputs_sources_unchanged=all(sha(path) == digest for path, digest in hashes.items()))
    dump(output/"summary.json", report)
    dump(output/"noise_estimates.json", estimates)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
