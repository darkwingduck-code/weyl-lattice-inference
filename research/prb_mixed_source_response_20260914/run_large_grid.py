"""Generate a three-channel response on a declared B0 or B+ mesh."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / "research" / "prb_lattice_response"
sys.path.insert(0, str(OLD))
from lattice_response import Parameters
from magnetic_bloch_response import iter_magnetic_bloch_spectra
from mixed_vertex_response import (
    accumulate_spectrum, add_accumulators, available_memory_bytes,
    dress_response, empty_accumulator, process_rss_bytes, scaled_max_error,
    verify_legacy_rpa, ward_errors, weighted_passivity_minimum)

MESHES = {
    ("Bplus", "pilot"): (36, 3, 48, 1, 36, 1),
    ("Bplus", "coarse"): (72, 13, 96, 1, 36, 1),
    ("Bplus", "final"): (72, 13, 192, 1, 36, 1),
    ("B0", "pilot"): (12, 12, 48, 0, 1, 32),
    ("B0", "coarse"): (48, 48, 192, 0, 1, 64),
    ("B0", "final"): (72, 72, 384, 0, 1, 64),
    ("Bplus", "x_refined"): (144, 13, 192, 1, 36, 1),
    ("Bplus", "y_refined"): (72, 25, 192, 1, 36, 1),
    ("Bplus", "z_refined"): (72, 13, 384, 1, 36, 1),
    ("Bplus", "combined"): (144, 25, 384, 1, 36, 1),
    ("B0", "higher"): (96, 96, 768, 0, 1, 64),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

def load_frequency(field):
    path = OLD / "parameter_sector_sampling" / f"{field}_final" / "baseline_metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, np.asarray(data["fixed_lab_frequencies"], float)

def read_prior(path, omega, prefix):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if .05 <= float(r["omega"]) <= .35]
    old_omega = np.asarray([float(r["omega"]) for r in rows])
    if not np.array_equal(old_omega, omega):
        raise ValueError("Prior frequency grid differs")
    return np.asarray([[[complex(float(r[f"{prefix}_{a}{b}_re"]),
                                  float(r[f"{prefix}_{a}{b}_im"]))
                         for b in range(2)] for a in range(2)] for r in rows])

def matrix_rows(omega, bare, dressed):
    rows = []
    for iw, w in enumerate(omega):
        row = {"omega": float(w)}
        for prefix, matrix in (("bare", bare), ("rpa", dressed)):
            for a in range(3):
                for b in range(3):
                    row[f"{prefix}_{a}{b}_re"] = float(matrix[iw, a, b].real)
                    row[f"{prefix}_{a}{b}_im"] = float(matrix[iw, a, b].imag)
        rows.append(row)
    return rows

def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def run(output, field, mesh, frequency_batch):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    Lx, Ly, Nkz, numerator, denominator, chunk = MESHES[(field, mesh)]
    p = Parameters(Lx=Lx, Ly=Ly, Nkz=Nkz)
    qshift = Nkz // 24
    q = 2 * np.pi * qshift / Nkz
    Q = 2 * np.sin(q / 2)
    frequency_path, omega = load_frequency(field)
    design = ROOT / "frozen_design.json"
    sources = [ROOT / name for name in (
        "mixed_vertex_response.py", "run_large_grid.py", "frozen_design.json")]
    sources += [OLD / name for name in (
        "lattice_response.py", "magnetic_bloch_response.py",
        "parameter_sector_sampling.py")]
    initial_hashes = {str(path.relative_to(REPO)): sha(path) for path in sources}
    inputs = {str(frequency_path.relative_to(REPO)): sha(frequency_path)}
    total_blocks = (Lx // denominator) * Ly
    total = empty_accumulator(len(omega))
    peak_rss = process_rss_bytes()
    chunks = 0
    for spec in iter_magnetic_bloch_spectra(
            p, numerator, denominator, blocks_per_chunk=chunk):
        part = accumulate_spectrum(spec, qshift, .5, .04, omega, .015,
                                   frequency_batch=frequency_batch)
        add_accumulators(total, part)
        chunks += 1
        rss = process_rss_bytes()
        observed_rss = [x for x in (peak_rss, rss) if x is not None]
        peak_rss = max(observed_rss) if observed_rss else None
        print(f"{field} {mesh}: chunk {chunks}, blocks through "
              f"{int(spec.block_indices[-1]) + 1}/{total_blocks}", flush=True)
    dressed = dress_response(total.bare, Q, .8, 8.)
    wards = ward_errors(total, omega, .015, Q)
    prior = OLD / "parameter_sector_sampling" / f"{field}_{mesh}" / "baseline.csv"
    prior_errors = None
    if prior.exists():
        inputs[str(prior.relative_to(REPO))] = sha(prior)
        prior_errors = {
            "bare": scaled_max_error(total.bare[:, :2, :2],
                                      read_prior(prior, omega, "bare")),
            "rpa": scaled_max_error(dressed[:, :2, :2],
                                     read_prior(prior, omega, "rpa"))}
    checks = {
        "raw_pairs": total.raw_pairs,
        "expected_raw_pairs": total_blocks * Nkz * (4 * denominator) ** 2,
        "charge_vertex_error": total.charge_vertex_error,
        "moment0_max": float(np.max(np.abs(total.moment0))),
        "charge_ward_left": wards["left"],
        "charge_ward_right": wards["right"],
        "weighted_passivity_minimum": weighted_passivity_minimum(
            dressed, omega, .015),
        "generic_vs_legacy_rpa_error": verify_legacy_rpa(
            total.bare, Q, .8, 8.),
        "prior_two_channel_errors": prior_errors}
    numerical_pass = (
        checks["raw_pairs"] == checks["expected_raw_pairs"]
        and checks["charge_vertex_error"] < 5e-11
        and checks["moment0_max"] < 2e-10
        and max(wards.values()) < 2e-10
        and checks["weighted_passivity_minimum"] >= -2e-10
        and checks["generic_vs_legacy_rpa_error"] < 5e-12
        and (prior_errors is None or max(prior_errors.values()) < 5e-10))
    np.savez_compressed(output / "response_3channel.npz", omega=omega,
        bare=total.bare, dressed=dressed, moment0=total.moment0,
        current_row=total.current_row, current_column=total.current_column,
        q=q, Q=Q, eta=.015, mu=.5, temperature=.04,
        parameters_json=json.dumps(p.__dict__), field=field, mesh=mesh)
    write_csv(output / "response_3channel.csv",
              matrix_rows(omega, total.bare, dressed))
    final_hashes = {str(path.relative_to(REPO)): sha(path) for path in sources}
    report = {
        "status": "MIXED_VERTEX_RESPONSE_PASS" if numerical_pass else
                  "MIXED_VERTEX_RESPONSE_VALIDATION_FAILED",
        "field": field, "mesh": mesh, "parameters": p.__dict__,
        "flux": f"{numerator}/{denominator}", "q": q, "qshift": qshift,
        "mu": .5, "temperature": .04, "eta": .015,
        "frequency_count": len(omega), "frequency_range": [float(omega.min()),
                                                            float(omega.max())],
        "channels": ["charge", "Gamma5", "Gamma3"],
        "interacting_channels": ["charge", "Gamma5"],
        "checks": checks, "chunks": chunks,
        "elapsed_seconds": time.perf_counter() - started,
        "available_memory_before_or_end": available_memory_bytes(),
        "peak_observed_process_rss_bytes": peak_rss,
        "source_sha256": initial_hashes, "input_sha256": inputs,
        "sources_unchanged": initial_hashes == final_hashes,
        "scientific_claim_ready": False,
        "limitations": [
            "Gamma3 is an external vertex, not an interacting channel.",
            "Pilot and coarse meshes do not certify the production mesh.",
            "Bplus kz refinement does not certify transverse convergence.",
            "Finite eta is a resolvent regulator, not intrinsic damping."]}
    if not report["sources_unchanged"]:
        report["status"] = "SOURCE_CHANGED_DURING_RUN"
    dump(output / "diagnostic.json", report)
    dump(output / "manifest.json", {
        path.name: sha(path) for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"})
    if report["status"] != "MIXED_VERTEX_RESPONSE_PASS":
        raise SystemExit(1)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", choices=("B0", "Bplus"), required=True)
    parser.add_argument("--mesh", choices=sorted({key[1] for key in MESHES}),
                        required=True)
    parser.add_argument("--frequency-batch", type=int, default=4)
    args = parser.parse_args()
    if (args.field, args.mesh) not in MESHES:
        parser.error("The selected mesh is not defined for this field")
    result = run(args.output.resolve(), args.field, args.mesh,
                 args.frequency_batch)
    print(json.dumps({"status": result["status"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "checks": result["checks"]}, indent=2), flush=True)
