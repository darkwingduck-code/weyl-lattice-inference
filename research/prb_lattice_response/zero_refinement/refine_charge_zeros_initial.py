"""Refine failed finite-charge-zero candidates without changing prior evidence.

All limits are imported from the frozen independent audit. Float64 Newton steps
stay within the original modal interval and within the same bare-pole component.
Decimal arithmetic diagnoses evaluation/location resolution using the SAME
float64 transition inputs, not a higher-precision Hamiltonian calculation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np

from zero_audit import LIMITS, direct_rpa, zero_check, matrix_from_row
from lattice_response import Parameters, interaction_matrix
from bloch_response import bloch_spectrum, bloch_transitions

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def safe_component(tr, row):
    x = row["zero"]
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    delta = tr.delta[active]
    if np.min(abs(delta-x)) <= LIMITS["bare_distance_relative"]*max(1., abs(x)):
        raise ValueError("Candidate is within the frozen bare-proximity exclusion")
    lower = float(np.nextafter(row["left_pole"], np.inf))
    upper = float(np.nextafter(row["right_pole"], -np.inf))
    left_bare = delta[delta < x]
    right_bare = delta[delta > x]
    if len(left_bare):
        edge = float(left_bare.max())
        lower = max(lower, edge+LIMITS["bare_distance_relative"]*max(1., abs(edge)))
    if len(right_bare):
        edge = float(right_bare.min())
        upper = min(upper, edge-LIMITS["bare_distance_relative"]*max(1., abs(edge)))
    if not lower < x < upper:
        raise ValueError("Original candidate is outside the protected regular component")
    return lower, upper


def direct_criteria(check):
    return (check.get("direct_charge_normalized", float("inf")) <= LIMITS["normalized_zero_residual"]
            and check.get("direct_newton_to_pole_distance", float("inf")) <= LIMITS["newton_to_pole_distance"]
            and check.get("dielectric_condition", float("inf")) <= LIMITS["dielectric_condition"])


def float_refine(tr, u, row, poles, residues, weights):
    lower, upper = safe_component(tr, row)
    start = float(row["zero"])
    current = start
    history = []
    evaluated = {}
    def evaluate(x):
        x = float(x)
        if not lower < x < upper:
            return None
        if x not in evaluated:
            evaluated[x] = zero_check(tr, u, x, poles, residues, weights)
        return evaluated[x]
    for iteration in range(16):
        check = evaluate(current)
        _, _, chi, derivative, _ = direct_rpa(tr, u, current)
        if derivative[0, 0] == 0:
            break
        step = float(-(chi[0, 0]/derivative[0, 0]).real)
        history.append(dict(iteration=iteration, zero=current, proposed_step=step,
                            direct_normalized=check.get("direct_charge_normalized"),
                            weighted_normalized=check["weighted_transfer_normalized"]))
        if check["status"] == "PASS_NUMERICAL_ZERO":
            break
        target = current+step
        for _ in range(40):
            if lower < target < upper:
                break
            step *= .5
            target = current+step
        candidates = {current, target}
        for center in (current, target):
            down, up = center, center
            for _ in range(4):
                down, up = np.nextafter(down, -np.inf), np.nextafter(up, np.inf)
                candidates.update((float(down), float(up)))
        checks = [evaluate(x) for x in candidates]
        checks = [x for x in checks if x is not None and "direct_charge_normalized" in x]
        best = min(checks, key=lambda x:x["direct_charge_normalized"])
        if best["zero"] == current:
            break
        current = best["zero"]
    best = min(evaluated.values(), key=lambda x:x.get("direct_charge_normalized", float("inf")))
    ulp = float(max(abs(np.nextafter(best["zero"], np.inf)-best["zero"]),
                    abs(best["zero"]-np.nextafter(best["zero"], -np.inf))))
    _, _, _, derivative, _ = direct_rpa(tr, u, best["zero"])
    scale = float(np.sum(abs(residues[:, 0, 0]/(best["zero"]-poles))))
    return dict(original_index=row["index"], original_zero=start,
                protected_interval=[lower, upper], history=history, best=best,
                shift=best["zero"]-start, ulp_spacing=ulp,
                shift_in_ulps=(best["zero"]-start)/ulp,
                local_half_ulp_normalized_response_scale=float(abs(derivative[0, 0])*ulp/(2*scale)),
                float_candidates_evaluated=len(evaluated),
                original_audit_status=row["status"])


def decimal_diagnostic(tr, u, record, poles, residues, weights):
    """High-precision arithmetic for fixed, exactly converted binary64 inputs."""
    active = (tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0)
    convert = lambda x: Decimal.from_float(float(x))
    with localcontext() as context:
        context.prec = 80
        delta = [convert(x) for x in tr.delta[active]]
        columns = [[convert(x) for x in values[active]]
                   for values in (tr.cc, tr.aa, tr.ca.real, tr.ca.imag)]
        vc, ua = convert(u[0, 0].real), convert(u[1, 1].real)
        def response(x):
            inverse = [1/(x-d) for d in delta]
            p = [sum((w*r for w, r in zip(column, inverse)), Decimal(0)) for column in columns]
            dp = [-sum((w*r*r for w, r in zip(column, inverse)), Decimal(0)) for column in columns]
            det = p[0]*p[1]-p[2]*p[2]-p[3]*p[3]
            ddet = dp[0]*p[1]+p[0]*dp[1]-2*p[2]*dp[2]-2*p[3]*dp[3]
            denominator = 1-vc*p[0]-ua*p[1]+vc*ua*det
            numerator = p[0]-ua*det
            ddenominator = -vc*dp[0]-ua*dp[1]+vc*ua*ddet
            dnumerator = dp[0]-ua*ddet
            chi = numerator/denominator
            derivative = (dnumerator*denominator-numerator*ddenominator)/(denominator*denominator)
            return chi, derivative
        x = convert(record["best"]["zero"])
        lower, upper = map(convert, record["protected_interval"])
        count = 0
        for count in range(20):
            chi, derivative = response(x)
            correction = chi/derivative
            candidate = x-correction
            if not lower < candidate < upper:
                raise ValueError("Decimal Newton would cross a protected boundary")
            x = candidate
            if abs(correction) < Decimal("1e-65")*max(Decimal(1), abs(x)):
                break
        chi, derivative = response(x)
        nearest = float(x)
        float_candidates = {nearest}
        down, up = nearest, nearest
        for _ in range(3):
            down, up = np.nextafter(down, -np.inf), np.nextafter(up, np.inf)
            float_candidates.update((float(down), float(up)))
        local = []
        for candidate in sorted(float_candidates):
            exact, _ = response(convert(candidate))
            scale = float(np.sum(abs(residues[:, 0, 0]/(candidate-poles))))
            measured = zero_check(tr, u, candidate, poles, residues, weights)
            local.append(dict(zero=candidate, decimal_charge_at_float=str(exact),
                              decimal_charge_normalized_at_float=float(abs(exact)/convert(scale)),
                              float64_charge_normalized=measured.get("direct_charge_normalized"),
                              full_check=measured))
        best_decimal = min(local, key=lambda row:row["decimal_charge_normalized_at_float"])
        best_float = min(local, key=lambda row:row["float64_charge_normalized"])
        return dict(decimal_precision=80, iterations=count+1, root_decimal=str(x),
                    charge_at_decimal_root=str(chi), derivative_at_decimal_root=str(derivative),
                    nearby_float_candidates=local,
                    smallest_sampled_decimal_normalized_residual=best_decimal["decimal_charge_normalized_at_float"],
                    smallest_sampled_float64_normalized_residual=best_float["float64_charge_normalized"],
                    best_float_check=best_float["full_check"],
                    scope="Exact Decimal conversion of the same float64 transition energies/weights and U; no higher-precision band calculation.")


def self_test():
    from types import SimpleNamespace
    e = np.array([.7, 1.3, 1.8])
    tr = SimpleNamespace(delta=np.ravel(np.column_stack((e, -e))),
                         cc=np.array([.08, -.08, .04, -.04, 0., 0.]),
                         aa=np.array([0., 0., 0., 0., .05, -.05]),
                         ca=np.zeros(6, complex))
    u = np.diag([.4, .3])
    a, b = 2*e[:2]*np.array([.08, .04])
    x0 = (a*e[1]**2+b*e[0]**2)/(a+b)
    total = e[0]**2+e[1]**2+u[0, 0]*(a+b)
    product = e[0]**2*e[1]**2+u[0, 0]*(a+b)*x0
    positive = np.sqrt(np.sort(np.roots([1., -total, product])))
    poles = np.array([-positive[1], -positive[0], positive[0], positive[1]])
    residues = np.zeros((4, 2, 2), complex)
    for i, pole in enumerate(poles):
        residues[i, 0, 0] = (a+b)*(pole*pole-x0)/(2*pole*(2*pole*pole-total))
    weights = (poles*residues[:, 0, 0]).real
    row = dict(index=0, zero=float(np.sqrt(x0)+1e-3), left_pole=float(poles[2]),
               right_pole=float(poles[3]), status="FAIL_NUMERICAL_ZERO")
    record = float_refine(tr, u, row, poles, residues, weights)
    assert record["best"]["status"] == "PASS_NUMERICAL_ZERO", record
    assert abs(record["best"]["zero"]-np.sqrt(x0)) < 1e-12
    rejected = False
    try:
        safe_component(tr, dict(row, zero=1.3))
    except ValueError:
        rejected = True
    assert rejected, "Bare-pole negative control was accepted"
    decimal = decimal_diagnostic(tr, u, record, poles, residues, weights)
    assert abs(float(decimal["root_decimal"])-np.sqrt(x0)) < 1e-12
    result = dict(status="SELF_TEST_PASS", repaired_displaced_zero=record,
                  bare_pole_negative_control_rejected=rejected,
                  decimal_reference=decimal)
    return result


def main(production, baseline, output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Refinement output must be a new empty directory")
    baseline_report = json.loads((baseline/"diagnostic.json").read_text())
    baseline_manifest_bytes = (baseline/"manifest.json").read_bytes()
    baseline_manifest = json.loads(baseline_manifest_bytes)
    for name, expected in baseline_manifest.items():
        if sha(baseline/name) != expected:
            raise RuntimeError("Frozen audit evidence changed: "+name)
    if sha(ROOT/"zero_audit.py") != baseline_report["audit_script_sha256"]:
        raise RuntimeError("Frozen audit source changed")
    frozen = {str(baseline/name): expected for name, expected in baseline_manifest.items()}
    frozen[str(baseline/"manifest.json")] = hashlib.sha256(baseline_manifest_bytes).hexdigest()
    frozen[str(ROOT/"zero_audit.py")] = sha(ROOT/"zero_audit.py")
    production_report = json.loads((production/"diagnostic.json").read_text())
    for name, expected in production_report["source_sha256"].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError("Production scientific source changed: "+name)
        frozen[str(ROOT/name)] = expected
    for name, expected in production_report["output_sha256"].items():
        if sha(production/name) != expected:
            raise RuntimeError("Frozen production output changed: "+name)
        frozen[str(production/name)] = expected
    frozen[str(production/"diagnostic.json")] = sha(production/"diagnostic.json")
    checks = json.loads((baseline/"zero_checks.json").read_text())
    failed = [row for row in checks if row["status"] == "FAIL_NUMERICAL_ZERO"]
    if not failed:
        raise ValueError("No failed candidates to refine")
    modal_rows = read_rows(production/"finite_poles.csv")
    poles = np.array([float(row["pole"]) for row in modal_rows])
    residues = np.array([matrix_from_row(row) for row in modal_rows])
    weights = np.array([float(row["weighted_charge"]) for row in modal_rows])
    parameters = Parameters(**production_report["finite_realization"]["parameters"])
    settings = production_report["settings"]
    shift_float = settings["q"]*parameters.Nkz/(2*np.pi)
    shift = int(round(shift_float))
    if abs(shift-shift_float)>1e-12:
        raise ValueError("Noninteger q shift")
    tr = bloch_transitions(bloch_spectrum(parameters), settings["mu"], settings["T"], shift)
    u = interaction_matrix(tr.Q, settings["g"], settings["Ua"])
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"self_test.json", self_test())
    results = []
    for row in failed:
        record = float_refine(tr, u, row, poles, residues, weights)
        best = record["best"]
        if best["status"] != "PASS_NUMERICAL_ZERO":
            diagnostic = decimal_diagnostic(tr, u, record, poles, residues, weights)
            record["decimal_diagnostic"] = diagnostic
            alternative = diagnostic["best_float_check"]
            if alternative.get("direct_charge_normalized", float("inf")) < best.get("direct_charge_normalized", float("inf")):
                record["best"] = best = alternative
                record["shift"] = best["zero"]-record["original_zero"]
                record["shift_in_ulps"] = record["shift"]/record["ulp_spacing"]
        record["ulp_spacing"] = float(max(abs(np.nextafter(best["zero"], np.inf)-best["zero"]),
                                        abs(best["zero"]-np.nextafter(best["zero"], -np.inf))))
        record["shift"] = best["zero"]-record["original_zero"]
        record["shift_in_ulps"] = record["shift"]/record["ulp_spacing"]
        slope = abs(direct_rpa(tr, u, best["zero"])[3][0, 0])
        scale = float(np.sum(abs(residues[:, 0, 0]/(best["zero"]-poles))))
        record["local_half_ulp_normalized_response_scale"] = float(slope*record["ulp_spacing"]/(2*scale))
        record["direct_response_criteria_pass"] = bool(direct_criteria(best))
        record["all_original_criteria_pass"] = best["status"] == "PASS_NUMERICAL_ZERO"
        if record["all_original_criteria_pass"]:
            record["status"] = "PASS_REFINED_DIRECT_AND_MODAL_CRITERIA"
        elif record["direct_response_criteria_pass"]:
            record["status"] = "DIRECT_CRITERIA_PASS_MODAL_CONSISTENCY_UNRESOLVED"
        else:
            record["status"] = "UNRESOLVED_FLOAT64_RESIDUAL"
        results.append(record)
        dump(output/"refined_candidates.json", results)
        print(f"[refined] index={row['index']} status={record['status']} "
              f"direct={best.get('direct_charge_normalized'):.4g} "
              f"modal={best['weighted_transfer_normalized']:.4g}", flush=True)
    if any(sha(Path(name)) != expected for name, expected in frozen.items()):
        raise RuntimeError("Frozen source/audit/production bytes changed during refinement")
    counts = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0)+1
    result = dict(status="BOUNDED_ZERO_REFINEMENT_COMPLETED",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_input_sha256=frozen, script_sha256=sha(Path(__file__)),
                  unchanged_limits=LIMITS, originally_failed_candidates=len(failed),
                  refinement_status_counts=counts,
                  original_audit_counts=baseline_report["zero_status_counts"],
                  prior_passed_zeros_preserved=baseline_report["zero_status_counts"].get("PASS_NUMERICAL_ZERO", 0),
                  prior_bare_proximity_unresolved_preserved=baseline_report["zero_status_counts"].get("UNRESOLVED_NEAR_BARE_TRANSITION", 0),
                  prior_edge_intervals_unresolved_preserved=baseline_report["zero_status_counts"].get("PRESERVED_UNRESOLVED_ZERO_TOO_CLOSE_TO_POLE", 0),
                  frozen_inputs_unchanged=True, complete_real_zero_catalog_certified=False,
                  submission_ready=False, continuum_or_collectivity_certified=False,
                  audit_metadata_clarification="The frozen baseline moment_checks.factorization_assessment string predates its added dense eigensystem audit. The separate eigensystem section correctly records the performed Cholesky/eigenpair recomputation; no numerical result is changed.",
                  limitations=["Same finite lattice and fixed float64 transition inputs.",
                               "All 16 previous failures and all prior unresolved records remain preserved.",
                               "No tolerance is enlarged. Direct and modal criteria are distinguished.",
                               "Local Newton and sampled neighboring ulps are numerical diagnostics, not interval-arithmetic certification.",
                               "Decimal arithmetic, where used, improves rational evaluation only; it does not improve the underlying band energies or input weights.",
                               "The attained smallest sampled residual is an upper bound on the best attainable residual in that sampled set, not a theorem of a global precision floor."])
    dump(output/"diagnostic.json", result)
    dump(output/"manifest.json", {p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"})
    print(json.dumps(dict(status=result["status"], counts=counts), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=ROOT/"mode_analysis")
    parser.add_argument("--baseline", type=Path, default=ROOT/"zero_audit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2), flush=True)
    else:
        if args.output is None:
            parser.error("--output is required")
        main(args.production.resolve(), args.baseline.resolve(), args.output.resolve())
