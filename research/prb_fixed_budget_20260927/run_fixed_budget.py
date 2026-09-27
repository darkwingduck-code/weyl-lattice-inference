"""Fixed acquisition budget, reusing the archived three-channel forward model."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PREVIOUS = ROOT.parent / "prb_source_identifiability_20260927"
sys.path.insert(0, str(PREVIOUS))
from run_study import (apply_transfer, clean_observations, contract_mixed, dress_response,
                       dump, load, physical, sha, transfers, write_csv)
from response_derivatives import contracted_derivatives
from analyze_information import project_out

PLAN = json.loads((ROOT / "frozen_plan.json").read_text())
NAMES = ("g", "Ua", "lambda_r", "lambda_s")
SCALES = np.array([.8, 8., 1., 1.])


def sources():
    old = ROOT.parent / "prb_mixed_source_response_20260914"
    paths = [Path(__file__), ROOT / "frozen_plan.json", PREVIOUS / "run_study.py",
        PREVIOUS / "response_derivatives.py", PREVIOUS / "analyze_information.py",
        PREVIOUS / "frozen_design.json", old / "mixed_vertex_response.py",
        old / "noise_sensitivity.py", old / "analyze_inference.py"]
    paths += [REPO / path for path in PLAN["inputs"].values()]
    return {str(path): sha(path) for path in paths}


def dataset():
    return {field: load(REPO / path) for field, path in PLAN["inputs"].items()}


class Problem:
    def __init__(self, data, observed, fractions):
        self.fields = list(fractions)
        self.data, self.observed = data, observed
        self.sigma = {f: PLAN["sigma0"] / np.sqrt(fractions[f]) for f in self.fields}
        self.basis = {f: np.column_stack((np.ones_like(data[f]["omega"]),
            (data[f]["omega"]-data[f]["omega"].mean())/np.ptp(data[f]["omega"])))
            for f in self.fields}
        self.count = sum(len(data[f]["omega"])*4 for f in self.fields)
        self.cache = None

    def evaluate(self, v):
        if self.cache is not None and np.array_equal(self.cache[0], v):
            return self.cache[1:]
        g, ua, lr, ls = SCALES*v[:4]
        differences, derivatives = [], []
        for index, field in enumerate(self.fields):
            coefficients = v[4+8*index:12+8*index].reshape(2, 2, 2)
            d = self.basis[field] @ (coefficients[0, 0]+1j*coefficients[0, 1])
            s = self.basis[field] @ (coefficients[1, 0]+1j*coefficients[1, 1])
            item = self.data[field]
            dressed = dress_response(item["bare"], item["Q"], g, ua)
            matrix = contract_mixed(dressed, lr, ls)
            prediction = apply_transfer(matrix, d, s)
            jac = np.zeros(matrix.shape+(len(v),), complex)
            for column, derivative in enumerate(contracted_derivatives(dressed, item["Q"], lr, ls)):
                jac[..., column] = SCALES[column]*apply_transfer(derivative, d, s)
            for side in (0, 1):
                for imaginary, phase in enumerate((1., 1j)):
                    for power in (0, 1):
                        column = 4+8*index+4*side+2*imaginary+power
                        factor = phase*self.basis[field][:, power]
                        if side == 0:
                            jac[:, 1, 0, column] = matrix[:, 1, 0]*factor
                            jac[:, 1, 1, column] = matrix[:, 1, 1]*s*factor
                        else:
                            jac[:, 0, 1, column] = matrix[:, 0, 1]*factor
                            jac[:, 1, 1, column] = matrix[:, 1, 1]*d*factor
            differences.append((prediction-self.observed[field]).ravel()/self.sigma[field])
            derivatives.append(jac.reshape(-1, len(v))/self.sigma[field])
        delta, jac = np.concatenate(differences), np.concatenate(derivatives)
        residual = np.r_[delta.real, delta.imag]/np.sqrt(self.count)
        real_jac = np.vstack((jac.real, jac.imag))/np.sqrt(self.count)
        self.cache = (np.array(v).copy(), residual, real_jac)
        return residual, real_jac

    def start(self, parameters, true_gains=False):
        v = list(np.array(parameters)/SCALES)
        for field in self.fields:
            coefficients = np.zeros((2, 2, 2))
            matrix = physical(self.data[field], *parameters)
            for side, (a, b) in enumerate(((1, 0), (0, 1))):
                if true_gains:
                    gain = np.linalg.lstsq(self.basis[field],
                        transfers(self.data[field]["omega"])[side], rcond=None)[0]
                else:
                    gain = np.linalg.lstsq(matrix[:, a, b, None]*self.basis[field],
                        self.observed[field][:, a, b], rcond=None)[0]
                coefficients[side, 0], coefficients[side, 1] = gain.real, gain.imag
            v += coefficients.ravel().tolist()
        return np.array(v)

    def fit(self, truth):
        lower = [.25, 0., -1., -1.] + [-np.inf]*(8*len(self.fields))
        upper = [2., 3., 1., 1.] + [np.inf]*(8*len(self.fields))
        attempts = []
        for start in PLAN["multistarts"]:
            result = least_squares(lambda v: self.evaluate(v)[0], self.start(start),
                jac=lambda v: self.evaluate(v)[1], bounds=(lower, upper),
                xtol=2e-10, ftol=2e-10, gtol=2e-10, max_nfev=500)
            attempts.append(dict(x=result.x.tolist(),
                **dict(zip(NAMES, map(float, SCALES*result.x[:4]))),
                rms=float(np.linalg.norm(result.fun)), success=bool(result.success),
                status=int(result.status), nfev=int(result.nfev)))
        best = min(range(len(attempts)), key=lambda i: attempts[i]["rms"])
        selected = attempts[best]
        upper_rms = float(np.linalg.norm(self.evaluate(self.start(truth, True))[0]))
        hits = [name for name, value, low, high in zip(NAMES,
            (selected[n] for n in NAMES), (.2, 0., -1., -1.), (1.6, 24., 1., 1.))
            if min(abs(value-low), abs(value-high)) <= 1e-5*(high-low)]
        return dict(**{key: selected[key] for key in (*NAMES, "rms", "success")},
            best_start=best, bound_hits=hits, attempts=attempts,
            unfitted_truth_rms=upper_rms,
            truth_upper_bound_pass=selected["rms"] <= upper_rms+1e-7*max(1., upper_rms))

    def information(self, truth):
        _, jac = self.evaluate(self.start(truth, True))
        # The residual used in optimization is RMS normalized; Fisher is not.
        jf = jac*np.sqrt(2*self.count)
        reduced, gain_rank = project_out(jf[:, :4], jf[:, 4:])
        interaction, source_rank = project_out(reduced[:, :2], reduced[:, 2:],
            1e-8*np.linalg.norm(jf[:, :4], 2))
        fisher = interaction.T@interaction
        covariance = np.linalg.inv(fisher)
        return dict(interaction_fisher=fisher.tolist(), gain_rank=gain_rank,
            source_rank=source_rank, Ua_local_sd=8.*float(np.sqrt(covariance[1, 1])),
            g_local_sd=.8*float(np.sqrt(covariance[0, 0])),
            minimum_interaction_fisher_eigenvalue=float(np.linalg.eigvalsh(fisher)[0]))


def calculate(output, phase, shard):
    started = time.perf_counter()
    hashes = sources()
    data = dataset()
    if phase == "information":
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use a new result directory")
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, pair in PLAN["truths"].items():
            clean = clean_observations(data, pair)
            for policy, fractions in PLAN["policies"].items():
                result = Problem(data, clean, fractions).information([.8, 8., *pair])
                rows.append(dict(truth=name, policy=policy, **result))
        target = output / "information.json"
    else:
        if not (output / "information.json").exists():
            raise FileNotFoundError("Run information phase first")
        target = output / f"fits_{shard}.json"
        if target.exists():
            raise FileExistsError("Fit shard already exists")
        rng = np.random.default_rng(PLAN["seed"])
        rows = []
        job = 0
        for repetition in range(PLAN["repetitions"]):
            for name, pair in PLAN["truths"].items():
                clean = clean_observations(data, pair)
                noise = {f: (rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))/np.sqrt(2.)
                         for f, x in clean.items()}
                for policy, fractions in PLAN["policies"].items():
                    current = job
                    job += 1
                    if current % 4 != shard:
                        continue
                    observed = {f: clean[f]+PLAN["sigma0"]*noise[f]/np.sqrt(fractions[f])
                                for f in fractions}
                    result = Problem(data, observed, fractions).fit([.8, 8., *pair])
                    rows.append(dict(truth=name, policy=policy, repetition=repetition, **result))
            if repetition % 5 == 4:
                print("shard", shard, "repetition", repetition+1, flush=True)
    dump(target, dict(plan=PLAN, results=rows, source_sha256=hashes,
        sources_unchanged=all(sha(p)==h for p, h in hashes.items()),
        elapsed_seconds=time.perf_counter()-started))
    print("Completed", target.name, len(rows), time.perf_counter()-started, flush=True)


def summarize(output):
    info = json.loads((output / "information.json").read_text())
    shards = [json.loads((output / f"fits_{i}.json").read_text()) for i in range(4)]
    assert all(d["sources_unchanged"] for d in [info, *shards])
    assert all(sha(p)==h for p, h in info["source_sha256"].items())
    estimates = [r for d in shards for r in d["results"]]
    assert len(estimates) == 300
    rows = []
    for item in info["results"]:
        selected = [r for r in estimates if r["truth"] == item["truth"] and r["policy"] == item["policy"]]
        errors = np.array([r["Ua"]-8. for r in selected])
        row = dict(truth=item["truth"], policy=item["policy"], count=len(selected),
            Ua_rmse=float(np.sqrt(np.mean(errors**2))), Ua_bias=float(np.mean(errors)),
            Ua_local_sd=item["Ua_local_sd"], success_count=sum(r["success"] for r in selected),
            bound_count=sum(bool(r["bound_hits"]) for r in selected),
            truth_upper_bound_failures=sum(not r["truth_upper_bound_pass"] for r in selected),
            unsuccessful_starts=sum(not a["success"] for r in selected for a in r["attempts"]))
        rows.append(row)
    write_csv(output / "summary.csv", rows)
    policies = list(PLAN["policies"])
    worst = {p: max(r["Ua_rmse"] for r in rows if r["policy"] == p) for p in policies}
    result = dict(plan=PLAN, summaries=rows,
        empirical_best_by_truth={name: min((r for r in rows if r["truth"] == name), key=lambda r:r["Ua_rmse"])["policy"] for name in PLAN["truths"]},
        empirical_worst_truth_rmse_by_policy=worst,
        empirical_minimax_policy=min(worst, key=worst.get),
        source_sha256=info["source_sha256"], sources_unchanged=True,
        limits="Local Fisher and 30 paired Monte Carlo repeats; no coverage, global uniqueness, uniform source robustness, or apparatus switching model.")
    dump(output / "summary.json", result)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("information", "fit", "summary"), required=True)
    parser.add_argument("--shard", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.phase == "summary":
        summarize(args.output.resolve())
    else:
        calculate(args.output.resolve(), args.phase, args.shard)
