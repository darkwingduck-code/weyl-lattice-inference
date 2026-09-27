"""Bounded, tiled zero-field mesh extension with the original response settings.

Every four-band transition is retained. Transverse tiles contribute to the
BARE polarization; RPA is applied only after their sum, never tile by tile.
The largest permitted production grid is L=96, Nkz=384.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import expit

from lattice_response import (
    AXIAL, GAMMA, MASS, SPIN_Z, Parameters, Transitions,
    interaction_matrix, loss_matrix, parameter_dict, rpa_response,
)
from bloch_response import bloch_spectrum, bloch_transitions

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-3
VERTEX_TOLERANCE = 2e-11
WARD_TOLERANCE = 2e-10
POWER_TOLERANCE = 2e-10
TILE_BLOCKS = 128
PROBE = .45 + .015j


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def memory_status():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_pagefile",
                "available_pagefile", "total_virtual", "available_virtual",
                "available_extended_virtual")]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_physical_bytes": int(value.total_physical),
            "available_physical_bytes": int(value.available_physical)}


def estimated_peak(blocks, nk):
    # Conservative array budget: eigensystem, all-pair weights, several complex
    # resolvent/matrix temporaries, and 128 MiB for libraries/allocator overhead.
    return int(4096 * blocks * nk + 128 * 1024**2)


def scaled_error(coarse, fine):
    return float(np.max(np.abs(coarse - fine) / (1 + np.abs(fine))))


def complex_pair(value):
    return {"real": float(value.real), "imag": float(value.imag)}


def tile_transitions(p, first, last, mu, temperature, qshift):
    """One transverse tile, normalized by the COMPLETE lattice volume."""
    indices = np.arange(first, last)
    kx = (2 * np.pi * (indices // p.Ly) - p.theta_x) / p.Lx
    ky = (2 * np.pi * (indices % p.Ly) - p.theta_y) / p.Ly
    kz = p.kz_grid()
    transverse = (
        (p.mass + p.wilson * (3 - np.cos(kx) - np.cos(ky)))[:, None, None] * MASS
        + (p.hopping * np.sin(kx))[:, None, None] * GAMMA[0]
        + (p.hopping * np.sin(ky))[:, None, None] * GAMMA[1]
        + p.exchange * SPIN_Z)
    longitudinal = (-p.wilson * np.cos(kz)[:, None, None] * MASS
                    + p.hopping * np.sin(kz)[:, None, None] * GAMMA[2])
    energies, vectors = np.linalg.eigh(
        transverse[:, None] + longitudinal[None, :])
    forward = (np.arange(p.Nkz) + qshift) % p.Nkz
    q = 2 * np.pi * qshift / p.Nkz
    Q = 2 * np.sin(q / 2)
    udag = vectors.conj().swapaxes(-1, -2)
    v = vectors[:, forward]
    charge = udag @ v
    axial = (udag @ AXIAL) @ v
    delta = energies[:, forward, None, :] - energies[:, :, :, None]
    midpoint = kz + q / 2
    current_operator = (p.wilson * np.sin(midpoint)[:, None, None] * MASS
                        + p.hopping * np.cos(midpoint)[:, None, None] * GAMMA[2])
    current = (udag @ current_operator[None, :]) @ v
    vertex_error = float(np.max(np.abs(Q * current - delta * charge)))
    occupation = expit((mu - energies) / temperature)
    density_value = float(occupation.sum() / p.volume)
    df = (occupation[:, :, :, None] - occupation[:, forward, None, :]) / p.volume
    return Transitions(
        delta=delta.ravel(), cc=(df * np.abs(charge)**2).ravel(),
        aa=(df * np.abs(axial)**2).ravel(),
        ca=(df * charge * axial.conj()).ravel(),
        conduction=((energies[:, :, :, None] > 0)
                    & (energies[:, forward, None, :] > 0)).ravel(),
        Q=Q, q=q, vertex_error=vertex_error, density_value=density_value)


def evaluate_tiled(p, omegas, settings, log, tile_blocks=TILE_BLOCKS,
                   qshift=None, negative_control=False):
    started = time.perf_counter()
    shift = p.Nkz // 24 if qshift is None else qshift
    if not isinstance(shift, int) or shift == 0 or abs(shift) >= p.Nkz:
        raise ValueError("A nonzero integer qshift inside the kz grid is required")
    q = 2 * np.pi * shift / p.Nkz
    Q = 2 * np.sin(q / 2)
    coupling = interaction_matrix(Q, settings["coulomb"], settings["axial_interaction"])
    before = memory_status()
    estimate = estimated_peak(min(tile_blocks, p.Lx * p.Ly), p.Nkz)
    if estimate * 2 + 512 * 1024**2 > before["available_physical_bytes"]:
        raise MemoryError("Insufficient free RAM for the conservative tile budget")
    bare = np.zeros((len(omegas), 2, 2), complex)
    probe_full = np.zeros((3, 3), complex)
    moments = np.zeros((2, 2, 2), complex)
    incorrect_dressed_sum = np.zeros_like(bare) if negative_control else None
    vertex_error, density_value, count = 0., 0., 0
    minimum_available = before["available_physical_bytes"]
    nblocks = p.Lx * p.Ly
    for first in range(0, nblocks, tile_blocks):
        available = memory_status()["available_physical_bytes"]
        minimum_available = min(minimum_available, available)
        if estimate * 2 + 512 * 1024**2 > available:
            raise MemoryError("Available RAM fell below the conservative tile budget")
        last = min(first + tile_blocks, nblocks)
        tr = tile_transitions(p, first, last, settings["mu0"],
                              settings["temperature"], shift)
        tile_bare = np.array([tr.density_response(complex(w, settings["eta"]))
                              for w in omegas])
        bare += tile_bare
        probe_full += tr.full_response(PROBE)
        moments += np.array(tr.moments())
        if negative_control:
            incorrect_dressed_sum += np.array([
                rpa_response(value, coupling) for value in tile_bare])
        density_value += tr.density_value
        vertex_error = max(vertex_error, tr.vertex_error)
        count += tr.delta.size
        del tr
        if first == 0 or last == nblocks or last % (8 * tile_blocks) == 0:
            log(f"  L={p.Lx}, Nkz={p.Nkz}: blocks {last}/{nblocks}, "
                f"elapsed {time.perf_counter()-started:.1f}s")
    response = np.array([rpa_response(value, coupling) for value in bare])
    probe_rpa = rpa_response(probe_full, coupling)
    contraction = np.array([PROBE, 0., -Q])
    ward_bare = float(max(np.max(np.abs(contraction @ probe_full)),
                          np.max(np.abs(probe_full @ contraction))))
    ward_rpa = float(max(np.max(np.abs(contraction @ probe_rpa)),
                         np.max(np.abs(probe_rpa @ contraction))))
    power_min = min(float(np.linalg.eigvalsh(loss_matrix(
        complex(w, settings["eta"]) * value)).min())
        for w, value in zip(omegas, response))
    moment_zero = float(np.max(np.abs(moments[0])))
    checks = dict(
        L=p.Lx, Ly=p.Ly, Nkz=p.Nkz, flux=0, ensemble="fixed_mu",
        parameters=parameter_dict(p), density=density_value, q=q, Q=Q,
        retained_transition_count=count, expected_transition_count=16*p.volume,
        vertex_error=vertex_error, vertex_tolerance=VERTEX_TOLERANCE,
        charge_ward_bare=ward_bare, charge_ward_rpa=ward_rpa,
        ward_tolerance=WARD_TOLERANCE,
        equal_time_density_commutator_max=moment_zero,
        minimum_rpa_weighted_loss_eigenvalue=power_min,
        weighted_passivity_tolerance=POWER_TOLERANCE,
        weighted_passivity_frequency_points=len(omegas),
        probe_frequency=complex_pair(PROBE),
        probe_bare_ca=complex_pair(probe_full[0, 1]),
        probe_rpa_ca=complex_pair(probe_rpa[0, 1]),
        tile_blocks=tile_blocks, tile_count=(nblocks+tile_blocks-1)//tile_blocks,
        estimated_tile_peak_bytes=estimate,
        estimated_untiled_peak_bytes=estimated_peak(nblocks, p.Nkz),
        minimum_observed_available_bytes=minimum_available,
        memory_before=before, elapsed_seconds=time.perf_counter()-started)
    checks["all_exact_identity_checks_pass"] = bool(
        count == 16*p.volume and vertex_error < VERTEX_TOLERANCE
        and max(ward_bare, ward_rpa, moment_zero) < WARD_TOLERANCE
        and power_min > -POWER_TOLERANCE)
    return dict(bare=bare, response=response, probe_full=probe_full,
                moments=moments, checks=checks,
                incorrect_dressed_sum=incorrect_dressed_sum)


def validate_path(omegas, settings, log):
    p = Parameters(Lx=3, Ly=4, Nkz=24, theta_x=.29, theta_y=-.43,
                   theta_z=.19, mass=.14, hopping=.93, wilson=1.13, exchange=.79)
    direct_spec = bloch_spectrum(p)
    records = []
    for shift in (-1, 1):
        tiled = evaluate_tiled(p, omegas, settings, log, tile_blocks=5,
                               qshift=shift, negative_control=True)
        direct = bloch_transitions(direct_spec, settings["mu0"],
                                    settings["temperature"], shift)
        coupling = interaction_matrix(direct.Q, settings["coulomb"],
                                       settings["axial_interaction"])
        direct_bare = np.array([direct.density_response(complex(w, settings["eta"]))
                               for w in omegas])
        direct_rpa = np.array([rpa_response(value, coupling) for value in direct_bare])
        errors = dict(
            bare_max_absolute=float(np.max(np.abs(tiled["bare"]-direct_bare))),
            rpa_max_absolute=float(np.max(np.abs(tiled["response"]-direct_rpa))),
            current_contact_probe_max_absolute=float(np.max(np.abs(
                tiled["probe_full"]-direct.full_response(PROBE)))),
            moments_max_absolute=float(np.max(np.abs(
                tiled["moments"]-np.array(direct.moments())))),
            density_absolute=abs(tiled["checks"]["density"]-direct.density_value))
        scale = max(1., float(np.max(np.abs(direct_rpa))),
                    float(np.max(np.abs(np.array(direct.moments())))))
        budget = 128*np.finfo(float).eps*p.volume*scale
        wrong_error = scaled_error(tiled["incorrect_dressed_sum"], direct_rpa)
        passed = (max(errors.values()) <= budget
                  and tiled["checks"]["all_exact_identity_checks_pass"]
                  and wrong_error > 1e-6)
        records.append(dict(qshift=shift, errors=errors,
                            roundoff_budget=budget, checks=tiled["checks"],
                            incorrect_tilewise_rpa_scaled_error=wrong_error,
                            negative_control_detection_threshold=1e-6,
                            passed=bool(passed)))
    return dict(method="Tiled bare sum versus complete direct Bloch Transitions",
                rpa_order="Sum all bare tile polarizations before applying RPA",
                parameters=parameter_dict(p), frequency_points=len(omegas),
                records=records, all_checks_pass=all(r["passed"] for r in records))


def make_plot(output, historical, comparisons):
    import matplotlib.pyplot as plt
    combined = historical + comparisons
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    for ax, transverse in zip(axes, (False, True)):
        chosen = [row for row in combined if
                  (row["kind"] == "zero_B_transverse") == transverse]
        values = [float(row["maximum_scaled_response_error"]) for row in chosen]
        labels = []
        for row in chosen:
            if transverse:
                labels.append(f'L {row["coarse_L"]} -> {row["fine_L"]}')
            else:
                labels.append(f'flux {int(row["flux"]):+d}, L {row["fine_L"]}\n'
                              f'Nk {row["coarse_Nkz"]} -> {row["fine_Nkz"]}')
        x = np.arange(len(chosen))
        ax.scatter(x, values, c=["#a33c29" if v > TOLERANCE else "#176b55"
                               for v in values], s=42, zorder=3)
        ax.axhline(TOLERANCE, color="#333333", linestyle="--", linewidth=1,
                   label="Unchanged tolerance: 1e-3")
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("max |fine - coarse| / (1 + |fine|)")
        ax.set_title("Transverse refinement, B=0, Nk=192" if transverse
                     else "Longitudinal refinement at fixed L and flux")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Fixed frequency grid and eta=0.015; no pole or gap certification")
    fig.savefig(output/"convergence.pdf")
    fig.savefig(output/"convergence.png", dpi=180)
    plt.close(fig)


def main(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Extension output must be empty; existing evidence is preserved")
    sources = [ROOT/name for name in (
        "lattice_response.py", "bloch_response.py", "refine_response.py", "extend_transverse.py")]
    inputs = [ROOT/"output_initial"/"diagnostic.json",
              ROOT/"refinement"/"diagnostic.json", ROOT/"refinement"/"refinement.csv",
              ROOT/"refinement"/"L48_Nk192_flux+0_fixed_mu.csv"]
    source_hashes = {path.name: sha(path) for path in sources}
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    initial = json.loads(inputs[0].read_text(encoding="utf-8"))
    settings = initial["settings"]
    for key, value in {"mu0":.5,"temperature":.04,"eta":.015,
                       "coulomb":.8,"axial_interaction":8.}.items():
        if settings[key] != value:
            raise ValueError(f"Unexpected original setting: {key}")
    with inputs[-1].open(newline="", encoding="utf-8") as stream:
        baseline_rows = list(csv.DictReader(stream))
    omegas = np.array([float(row["omega"]) for row in baseline_rows])
    if len(omegas) != 161 or not np.allclose(omegas, np.linspace(.025, 1.2, 161),
                                           rtol=0, atol=1e-15):
        raise ValueError("Baseline frequency grid changed")
    for row in baseline_rows:
        if (int(row["L"]), int(row["Nkz"]), int(row["flux"]), row["ensemble"]) != (
                48, 192, 0, "fixed_mu"):
            raise ValueError("Wrong reference table")
        if (float(row["mu"]) != .5 or float(row["eta"]) != .015
                or abs(float(row["q"])-2*np.pi/24) > 1e-15):
            raise ValueError("Reference response settings changed")
    previous = np.array([[[complex(float(row[f"rpa_{a}{b}_re"]),
                                   float(row[f"rpa_{a}{b}_im"]))
                           for b in range(2)] for a in range(2)] for row in baseline_rows])
    output.mkdir(parents=True, exist_ok=True)
    log_path = output/"execution.log"
    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message+"\n")
    started = time.perf_counter()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  status="RUNNING", settings={k:settings[k] for k in
                    ("mu0","temperature","eta","coulomb","axial_interaction")},
                  q=2*np.pi/24, frequency_points=len(omegas), tolerance=TOLERANCE,
                  source_sha256=source_hashes, input_sha256=input_hashes,
                  preflight_memory=memory_status(), calculations=[], comparisons=[],
                  all_selected_zero_B_directions_converged=False,
                  collective_pole_or_gap_certified=False,
                  limitations=["Convergence is assessed only at 161 sampled frequencies and eta=0.015.",
                    "Zero-field transverse convergence does not establish the magnetic thermodynamic limit.",
                    "A finite-grid nonzero charge-axial response is not proof of a nonzero infinite-grid limit.",
                    "No transition cutoff, broadening change, or tolerance relaxation is used."])
    def checkpoint():
        (output/"diagnostic.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    checkpoint()
    try:
        log("Preflight RAM: "+json.dumps(report["preflight_memory"]))
        validation = validate_path(omegas, settings, log)
        (output/"path_validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
        report["tiled_path_validation_pass"] = validation["all_checks_pass"]
        if not validation["all_checks_pass"]:
            raise RuntimeError("Tiled/direct Bloch path validation failed")
        log("Small twisted-grid direct Bloch comparison and wrong-RPA negative control PASS")
        def evaluate(size, nk):
            log(f"[extension] L={size}, Nkz={nk}, flux=0, fixed_mu")
            result = evaluate_tiled(Parameters(Lx=size,Ly=size,Nkz=nk), omegas, settings, log)
            rows = []
            for w, bare, response in zip(omegas, result["bare"], result["response"]):
                row = dict(L=size,Nkz=nk,flux=0,B=0.,ensemble="fixed_mu",mu=.5,
                           density=result["checks"]["density"],q=2*np.pi/24,
                           omega=float(w),eta=.015)
                for prefix, matrix in (("rpa",response),("bare",bare)):
                    for a,b in ((0,0),(0,1),(1,0),(1,1)):
                        row[f"{prefix}_{a}{b}_re"] = float(matrix[a,b].real)
                        row[f"{prefix}_{a}{b}_im"] = float(matrix[a,b].imag)
                rows.append(row)
            filename = f"L{size}_Nk{nk}_flux+0_fixed_mu.csv"
            csv_write(output/filename, rows)
            report["calculations"].append(dict(csv=filename, **result["checks"]))
            checkpoint()
            if not result["checks"]["all_exact_identity_checks_pass"]:
                raise RuntimeError(f"Identity/passivity check failed at L={size},Nk={nk}")
            return result["response"]
        def compare(coarse, fine, size0, size1, nk0, nk1, kind):
            error = scaled_error(coarse, fine)
            record = dict(kind=kind,flux=0,coarse_L=size0,fine_L=size1,
                          coarse_Nkz=nk0,fine_Nkz=nk1,
                          maximum_scaled_response_error=error,tolerance=TOLERANCE,
                          converged=error<=TOLERANCE)
            report["comparisons"].append(record)
            csv_write(output/"comparisons.csv", report["comparisons"])
            log("Comparison: "+json.dumps(record))
            checkpoint()
            return record
        previous_size = 48
        for size in (72,96):
            current = evaluate(size,192)
            transverse = compare(previous,current,previous_size,size,192,192,"zero_B_transverse")
            previous, previous_size = current, size
            if transverse["converged"]:
                break
        fine_kz = evaluate(previous_size,384)
        longitudinal = compare(previous,fine_kz,previous_size,previous_size,192,384,"zero_B_kz")
        report["final_comparisons"] = dict(zero_B_transverse=transverse,zero_B_kz=longitudinal)
        report["all_selected_zero_B_directions_converged"] = bool(
            transverse["converged"] and longitudinal["converged"])
        historical = json.loads(inputs[1].read_text(encoding="utf-8"))["comparisons"]
        make_plot(output, historical, report["comparisons"])
        report["status"] = ("ZERO_B_SELECTED_MESHES_CONVERGED"
                            if report["all_selected_zero_B_directions_converged"]
                            else "ZERO_B_EXTENSION_COMPLETED_UNCONVERGED")
    except Exception as exc:
        report["status"] = "EXTENSION_FAILED"
        report["failure"] = {"type":type(exc).__name__,"message":str(exc)}
        log(traceback.format_exc())
    report["sources_unchanged"] = source_hashes == {p.name:sha(p) for p in sources}
    report["inputs_unchanged"] = input_hashes == {
        str(p.relative_to(ROOT)):sha(p) for p in inputs}
    if not report["sources_unchanged"] or not report["inputs_unchanged"]:
        report["status"] = "SOURCE_OR_INPUT_CHANGED"
    report["all_exact_identity_checks_pass"] = bool(
        report.get("tiled_path_validation_pass",False) and report["calculations"]
        and all(row["all_exact_identity_checks_pass"] for row in report["calculations"]))
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.perf_counter()-started
    log("Final status: "+report["status"])
    report["output_sha256"] = {p.name:sha(p) for p in output.iterdir()
                               if p.is_file() and p.name != "diagnostic.json"}
    checkpoint()
    if report["status"] in ("EXTENSION_FAILED", "SOURCE_OR_INPUT_CHANGED"):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    main(args.output.resolve())