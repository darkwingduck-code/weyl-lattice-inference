"""Audit and summarize the six completed common-reference fit datasets."""
import argparse
import json
from pathlib import Path

from run_study import ROOT, dump, sha, write_csv


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    names = ("common_B0_v1", "common_Bplus_final_v1", "common_Bplus_x_v1",
             "common_Bplus_y_v1", "common_Bplus_z_v1", "common_joint_v1")
    reports = {name: json.loads((ROOT / name / "summary.json").read_text()) for name in names}
    rows, checks = [], []
    sigma = 0.00012931669440932318
    for name, report in reports.items():
        assert report["absolute_sigma"] == sigma
        assert report["number_of_starts"] == 5
        assert report["inputs_sources_unchanged"]
        assert all(sha(path) == digest for path, digest in report["input_source_sha256"].items())
        applicable = [row for row in report["results"] if row["optimizer_upper_bound_pass"] is not None]
        assert all(row["success"] for row in report["results"])
        assert all(row["optimizer_upper_bound_pass"] for row in applicable)
        checks.append(dict(dataset=name, fits=len(report["results"]),
            retained_successes=sum(row["success"] for row in report["results"]),
            all_start_successes=sum(row["all_starts_success"] for row in report["results"]),
            applicable_truth_bounds=len(applicable), passed_truth_bounds=len(applicable)))
        for row in report["results"]:
            if row["source_model"] in ("known", "independent"):
                assert row["valid_for_mesh_bias"]
                rows.append(dict(dataset=name, truth=row["truth"], observation=row["observation"],
                    source_model=row["source_model"], delta_g=row["g"]-.8,
                    delta_Ua=row["Ua"]-8., residual_rms=row["rms_noise_units"],
                    unfitted_truth_rms=row["unfitted_truth_rms"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fit_bias.csv", rows)
    lines = ["# Common-reference inference checks", "",
        "All six comparisons use the same fixed source-study complex RMS sigma "
        "`0.00012931669440932318` and all five original starting points. "
        "All 72 retained optima succeeded; all 60 feasible truth-objective upper bounds passed. "
        "Alternative-start failures remain in the original reports. "
        "The 12 unequal-truth/shared-model fits are misspecified controls, not mesh-bias estimates.", "",
        "The magnetic reference is the combined `(144,25,384)` grid in every magnetic row. "
        "The zero-field reference is `(96,96,768)`; the joint comparison uses both.", "",
        "| Lower template | Truth | Delta Ua, known directions | Delta Ua, independent directions | Independent fit R |",
        "|---|---|---:|---:|---:|"]
    for name in names:
        for truth in ("symmetric", "asymmetric"):
            selected = {row["source_model"]: row for row in rows if row["dataset"] == name
                        and row["truth"] == truth and row["observation"] == "full"}
            lines.append(f"| {name} | {truth} | {selected['known']['delta_Ua']:+.6g} | "
                f"{selected['independent']['delta_Ua']:+.6g} | {selected['independent']['residual_rms']:.6g} |")
    lines += ["", "The finite-grid shifts are conditional on the declared source family, "
        "frequencies, broadening, and absolute detector noise. The residual is a per-entry "
        "complex RMS, not a significance threshold. Success of the truth-objective check "
        "does not prove global optimality or continuum convergence.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {str(ROOT / name / "summary.json"): sha(ROOT / name / "summary.json") for name in names}
    hashes[str(Path(__file__))] = sha(Path(__file__))
    dump(output / "audit.json", dict(absolute_sigma=sigma, checks=checks,
         current_source_hashes_match=True, input_summary_sha256=hashes))
    print("Verified", sum(row["fits"] for row in checks), "fits and",
          sum(row["passed_truth_bounds"] for row in checks), "truth upper bounds")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
