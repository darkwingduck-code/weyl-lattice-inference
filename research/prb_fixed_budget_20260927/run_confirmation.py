"""Independent paired confirmation of two pilot-selected fixed-budget contrasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

from run_fixed_budget import (PLAN as PILOT, Problem, ROOT, clean_observations,
                              dataset, dump, sha, sources, write_csv)

PLAN = json.loads((ROOT / "confirmation_plan.json").read_text())


def source_hashes():
    result = sources()
    for path in (Path(__file__), ROOT / "confirmation_plan.json", ROOT / "run_01" / "summary.json"):
        result[str(path)] = sha(path)
    return result


def calculate(output, shard):
    started = time.perf_counter()
    target = output / f"fits_{shard}.json"
    if target.exists():
        raise FileExistsError("Confirmation shard already exists")
    hashes = source_hashes()
    data = dataset()
    rng = np.random.default_rng(PLAN["seed"])
    rows = []
    for repetition in range(PLAN["repetitions_per_truth"]):
        for name, contrast in PLAN["contrasts"].items():
            directions = contrast["directions"]
            clean = clean_observations(data, directions)
            noise = {f: (rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))/np.sqrt(2.)
                     for f, x in clean.items()}
            if repetition % 4 != shard:
                continue
            for role in ("candidate", "baseline"):
                policy = contrast[role]
                fractions = PLAN["policies"][policy]
                observed = {f: clean[f]+PILOT["sigma0"]*noise[f]/np.sqrt(fractions[f])
                            for f in fractions}
                fitted = Problem(data, observed, fractions).fit([.8, 8., *directions])
                rows.append(dict(truth=name, role=role, policy=policy,
                                 repetition=repetition, **fitted))
        if repetition % 20 == shard:
            print("shard", shard, "completed through repetition", repetition+1, flush=True)
    assert len(rows) == PLAN["total_fit_count"]//4
    output.mkdir(parents=True, exist_ok=True)
    dump(target, dict(plan=PLAN, inherited_pilot_plan=PILOT, results=rows,
        source_sha256=hashes, sources_unchanged=all(sha(p)==h for p,h in hashes.items()),
        elapsed_seconds=time.perf_counter()-started))
    print("Completed", target.name, len(rows), time.perf_counter()-started, flush=True)


def summarize(output):
    target = output / "summary.json"
    if target.exists():
        raise FileExistsError("Use a new confirmation output directory")
    shards = [json.loads((output / f"fits_{i}.json").read_text()) for i in range(4)]
    assert all(d["sources_unchanged"] and d["source_sha256"] == shards[0]["source_sha256"] for d in shards)
    assert all(sha(p)==h for p,h in shards[0]["source_sha256"].items())
    rows = [r for d in shards for r in d["results"]]
    assert len(rows) == PLAN["total_fit_count"]
    assert len({(r["truth"], r["role"], r["repetition"]) for r in rows}) == len(rows)
    z = NormalDist().inv_cdf(.9875)
    comparisons, summaries, paired_rows = [], [], []
    for name, contrast in PLAN["contrasts"].items():
        by_role = {}
        for role in ("candidate", "baseline"):
            selected = sorted((r for r in rows if r["truth"] == name and r["role"] == role),
                              key=lambda r:r["repetition"])
            assert [r["repetition"] for r in selected] == list(range(PLAN["repetitions_per_truth"]))
            by_role[role] = selected
            error = np.array([r["Ua"]-8. for r in selected])
            summaries.append(dict(truth=name, role=role, policy=contrast[role], count=len(selected),
                Ua_rmse=float(np.sqrt(np.mean(error**2))), Ua_bias=float(error.mean()),
                success_count=sum(r["success"] for r in selected),
                bound_count=sum(bool(r["bound_hits"]) for r in selected),
                truth_upper_bound_failures=sum(not r["truth_upper_bound_pass"] for r in selected),
                unsuccessful_starts=sum(not a["success"] for r in selected for a in r["attempts"])))
        delta = []
        for candidate, baseline in zip(by_role["candidate"], by_role["baseline"]):
            difference = (candidate["Ua"]-8.)**2-(baseline["Ua"]-8.)**2
            delta.append(difference)
            paired_rows.append(dict(truth=name, repetition=candidate["repetition"],
                candidate_Ua=candidate["Ua"], baseline_Ua=baseline["Ua"], squared_error_difference=difference))
        mean, se = float(np.mean(delta)), float(np.std(delta, ddof=1)/np.sqrt(len(delta)))
        relevant = [r for r in summaries if r["truth"] == name]
        valid = all(r["success_count"] == r["count"] and not r["bound_count"] and
                    not r["truth_upper_bound_failures"] for r in relevant)
        comparisons.append(dict(truth=name, candidate=contrast["candidate"], baseline=contrast["baseline"],
            paired_count=len(delta), mean_squared_error_difference=mean, monte_carlo_se=se,
            normal_interval_lower=mean-z*se, normal_interval_upper=mean+z*se,
            optimization_checks_pass=valid,
            lower_candidate_mse_supported=valid and mean+z*se < 0.))
    write_csv(output / "policy_summary.csv", summaries)
    write_csv(output / "paired_differences.csv", paired_rows)
    write_csv(output / "contrasts.csv", comparisons)
    result = dict(plan=PLAN, inherited_pilot_plan=PILOT, z=z, summaries=summaries, contrasts=comparisons,
        source_sha256=shards[0]["source_sha256"], sources_unchanged=True,
        input_sha256={str(output / f"fits_{i}.json"): sha(output / f"fits_{i}.json") for i in range(4)})
    dump(target, result)
    print(json.dumps(dict(z=z, summaries=summaries, contrasts=comparisons), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("fit", "summary"), required=True)
    parser.add_argument("--shard", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.phase == "fit":
        calculate(args.output.resolve(), args.shard)
    else:
        summarize(args.output.resolve())
