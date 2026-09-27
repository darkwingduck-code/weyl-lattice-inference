"""Independent fixed-profile detector audit.

Rebuild diagonal responses by the 2x2 cofactor formula from frozen bare CSVs.
No production module is imported. This checks stored objective values and the
analytic gain optimum at stored g; it does not certify or rerun g optimization.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TOL = 1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run(output):
    if output.exists():
        raise FileExistsError("Preserve the existing independent review; use a new path")
    paths = {"B0": ROOT / "transverse_extension/L72_Nk384_flux+0_fixed_mu.csv",
             "Bplus": ROOT / "magnetic_kz_continuation/Nkx2_Nky13_Nk192_B1of36_fixed_mu.csv"}
    datasets = {}
    for name, path in paths.items():
        rr = [r for r in rows(path) if .05 <= float(r["omega"]) <= .35]
        pi = np.array([[[complex(float(r[f"bare_{a}{b}_re"]),
                                float(r[f"bare_{a}{b}_im"]))
                        for b in range(2)] for a in range(2)] for r in rr])
        datasets[name] = (pi, float(rr[0]["q"]))
    def response(name, g, ua):
        pi, q = datasets[name]
        a, d, b, c = pi[:, 0, 0], pi[:, 1, 1], pi[:, 0, 1], pi[:, 1, 0]
        vc = g / (2 * np.sin(q / 2))**2
        det = a * d - b * c
        denominator = 1 - vc * a - ua * d + vc * ua * det
        return np.column_stack(((a - ua * det) / denominator,
                                (d - vc * det) / denominator))
    truth = {n: response(n, .8, 8.) for n in datasets}
    checks, hash_checks, threshold_checks = [], [], []
    input_paths = set(paths.values())
    specifications = [("detector_identifiability_audit", "profiled_coulomb.csv"),
                      ("detector_calibration_followup", "calibration_profiles.csv")]
    for folder, filename in specifications:
        diag_path, csv_path = ROOT / folder / "diagnostic.json", ROOT / folder / filename
        diagnostic = json.loads(diag_path.read_text(encoding="utf-8"))
        input_paths.update((diag_path, csv_path))
        for rel, expected in diagnostic["source_input_sha256"].items():
            path = ROOT / rel
            actual = sha(path)
            input_paths.add(path)
            hash_checks.append(dict(owner=folder, path=rel, expected=expected,
                                    actual=actual, passed=(actual == expected)))
        current = []
        for row in rows(csv_path):
            names = ("B0", "Bplus") if row["dataset"] == "joint" else (row["dataset"],)
            mode, g, ua = row["mode"], float(row["best_g"]), float(row["ua"])
            model = {n: response(n, g, ua) for n in names}
            gains, gain_errors = {0: 1., 1: 1.}, []
            if folder == "detector_calibration_followup":
                for channel in ((1,) if mode == "axial_gain_only" else (0, 1)):
                    numerator = denominator = 0.
                    for name in names:
                        m, t = model[name][:, channel], truth[name][:, channel]
                        scale2 = np.vdot(t, t).real
                        numerator += np.vdot(m, t).real / scale2
                        denominator += np.vdot(m, m).real / scale2
                    gains[channel] = max(0., numerator / denominator)
                    key = "charge_gain" if channel == 0 else "axial_gain"
                    gain_errors.append(abs(gains[channel] - float(row[key])))
            channels = (0,) if mode in ("charge_complex", "charge_loss") else (0, 1)
            error2 = 0.
            for name in names:
                for channel in channels:
                    a = model[name][:, channel] * gains[channel]
                    t = truth[name][:, channel]
                    if mode == "charge_loss":
                        a, t = -a.imag, -t.imag
                    error2 += np.sum(abs(a - t)**2) / np.sum(abs(t)**2)
            value = float(np.sqrt(error2 / (len(names) * len(channels))))
            error = abs(value - float(row["normalized_mismatch"]))
            item = dict(owner=folder, dataset=row["dataset"], mode=mode, ua=ua, g=g,
                        mismatch=value, absolute_mismatch_error=error,
                        max_gain_error=max(gain_errors, default=0.),
                        passed=bool(error <= TOL and max(gain_errors, default=0.) <= TOL))
            checks.append(item)
            current.append(item)
        for group in diagnostic["tolerance_sets"]:
            threshold = group.get("threshold", group.get("declared_relative_mismatch_threshold"))
            selected = [x["ua"] for x in current if x["dataset"] == group["dataset"]
                        and x["mode"] == group["mode"] and x["mismatch"] <= threshold]
            threshold_checks.append(dict(owner=folder, dataset=group["dataset"],
                mode=group["mode"], threshold=threshold, recalculated=selected,
                passed=(selected == group["accepted_sampled_ua"])))
    passed = (len(checks) == 735 and all(x["passed"] for x in checks)
              and all(x["passed"] for x in hash_checks)
              and all(x["passed"] for x in threshold_checks))
    report = dict(status="PASS" if passed else "FAIL",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        method="Independent 2x2 cofactors and analytic real nonnegative gain least squares",
        scope="Stored g only; optimizer/global minimum, continuous Ua sets, experiments and statistical coverage not certified",
        tolerance=TOL, profile_count=len(checks),
        maximum_absolute_mismatch_error=max(x["absolute_mismatch_error"] for x in checks),
        maximum_absolute_gain_error=max(x["max_gain_error"] for x in checks),
        source_sha256=sha(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(input_paths)},
        inherited_hash_checks=hash_checks, threshold_checks=threshold_checks, profiles=checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "profile_count",
          "maximum_absolute_mismatch_error", "maximum_absolute_gain_error")}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
