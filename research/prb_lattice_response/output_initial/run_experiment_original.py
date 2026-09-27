"""Execute the first PRB lattice-response benchmark, retaining unconverged results."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lattice_response import (Parameters, diagonalize, density, fixed_density_mu,
    transitions, interaction_matrix, rpa_response, loss_matrix, axial_identity,
    parameter_dict)

ROOT = Path(__file__).resolve().parent
OMEGAS = np.linspace(.025, 1.2, 161)
MU0, TEMPERATURE, ETA = .5, .04, .015
COULOMB, AXIAL_U = .8, 8.
REFINEMENT_TOLERANCE = 1e-3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Use a new empty output directory to preserve previous runs")
    source_paths = [ROOT/"lattice_response.py", ROOT/"run_experiment.py"]
    sources = {p.name: sha(p) for p in source_paths}
    spectra, diagnostics, meshes, stored = [], [], [], {}
    grids = [(4, 24), (4, 48), (6, 24), (6, 48), (8, 48)]
    reference_parameters = Parameters(Lx=8, Ly=8, Nkz=48)
    reference = diagonalize(reference_parameters, 0)
    target = density(reference, MU0, TEMPERATURE)
    for size, nk in grids:
        p = Parameters(Lx=size, Ly=size, Nkz=nk)
        qshift = nk // 24
        zero = reference if p == reference_parameters else diagonalize(p, 0)
        fluxes = (-1, 0, 1) if size < 8 else (0,)
        for flux in fluxes:
            spec = zero if flux == 0 else diagonalize(p, flux)
            mu_n = fixed_density_mu(spec, target, TEMPERATURE)
            choices = (("fixed_mu", MU0), ("fixed_n", mu_n))
            for ensemble, mu in choices:
                print(f"[run] L={size}, Nkz={nk}, flux={flux}, {ensemble}", flush=True)
                tr = transitions(spec, mu, TEMPERATURE, qshift)
                u = interaction_matrix(tr.Q, COULOMB, AXIAL_U)
                base = dict(L=size, Nkz=nk, flux=flux, B=p.field(flux),
                            ensemble=ensemble, mu=mu, target_density=target,
                            actual_density=tr.density_value, q=tr.q, Q=tr.Q)
                measured, powers, cofactor_errors = [], [], []
                for omega in OMEGAS:
                    z = complex(omega, ETA)
                    bare = tr.density_response(z)
                    dressed = rpa_response(bare, u)
                    denominator = np.linalg.det(np.eye(2) - bare @ u)
                    numerator = bare[0, 0] - AXIAL_U * np.linalg.det(bare)
                    cofactor_errors.append(abs(dressed[0, 0] * denominator - numerator))
                    power_min = float(np.linalg.eigvalsh(loss_matrix(z * dressed)).min())
                    powers.append(power_min)
                    measured.append(dressed)
                    row = dict(base, omega=float(omega), eta=ETA,
                               loss_cc=float(-dressed[0, 0].imag),
                               loss_aa=float(-dressed[1, 1].imag),
                               power_min_eigenvalue=power_min,
                               denominator_re=float(denominator.real),
                               denominator_im=float(denominator.imag),
                               charge_numerator_re=float(numerator.real),
                               charge_numerator_im=float(numerator.imag))
                    for label, matrix in (("bare", bare), ("rpa", dressed)):
                        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
                            row[f"{label}_{a}{b}_re"] = float(matrix[a, b].real)
                            row[f"{label}_{a}{b}_im"] = float(matrix[a, b].imag)
                    spectra.append(row)
                stored[(size, nk, flux, ensemble)] = np.array(measured)
                probe = .45 + .015j
                bare_probe = tr.density_response(probe)
                ward = tr.ward_errors(probe)
                dressed_ward = tr.ward_errors(probe, u)
                torque = axial_identity(spec, mu, TEMPERATURE, qshift, probe, u)
                m0, m1 = tr.moments()
                diagnostic = dict(base, vertex_error=tr.vertex_error,
                    charge_ward_bare=max(ward), charge_ward_rpa=max(dressed_ward),
                    axial_vertex_error=torque["vertex_error"],
                    axial_ward_with_torque=torque["ward_error_with_torque"],
                    axial_ward_without_torque=torque["ward_error_without_torque"],
                    zeroth_moment=float(np.max(np.abs(m0))),
                    first_moment_min_eigenvalue=float(np.linalg.eigvalsh(m1).min()),
                    rpa_power_min_eigenvalue=min(powers),
                    cofactor_identity_error=float(max(cofactor_errors)),
                    bare_mixed_re=float(bare_probe[0, 1].real),
                    bare_mixed_im=float(bare_probe[0, 1].imag))
                diagnostics.append(diagnostic)
                # These validate exact finite-system identities, not mesh convergence.
                assert tr.vertex_error < 2e-11
                assert max(ward + dressed_ward) < 2e-10
                assert torque["vertex_error"] < 2e-11
                assert torque["ward_error_with_torque"] < 2e-10
                assert diagnostic["zeroth_moment"] < 2e-11
                assert diagnostic["first_moment_min_eigenvalue"] > -2e-11
                assert min(powers) > -2e-10
                assert max(cofactor_errors) < 2e-10
                if ensemble == "fixed_n":
                    assert abs(tr.density_value - target) < 2e-11
            meshes.append(dict(parameters=parameter_dict(p), flux=flux,
                               field=p.field(flux), mu_fixed_density=mu_n,
                               target_density=target))
            if flux != 0:
                del spec
    refinement = []
    for size in (4, 6):
        for flux in (-1, 0, 1):
            for ensemble in ("fixed_mu", "fixed_n"):
                a = stored[(size, 24, flux, ensemble)]
                b = stored[(size, 48, flux, ensemble)]
                error = float(np.max(np.abs(a - b) / (1 + np.abs(b))))
                refinement.append(dict(L=size, flux=flux, ensemble=ensemble,
                    q=float(2*np.pi/24), coarse_Nkz=24, fine_Nkz=48,
                    maximum_scaled_response_error=error,
                    tolerance=REFINEMENT_TOLERANCE,
                    converged=error <= REFINEMENT_TOLERANCE,
                    ensemble_note="fixed_n uses one common L=8,Nkz=48,B=0 target for every grid"))
    # Exact fixed-mu, B=0 transverse comparisons; field never changes here.
    transverse = []
    for coarse, fine in ((4, 6), (6, 8)):
        a, b = stored[(coarse, 48, 0, "fixed_mu")], stored[(fine, 48, 0, "fixed_mu")]
        error = float(np.max(np.abs(a-b)/(1+np.abs(b))))
        transverse.append(dict(coarse_L=coarse, fine_L=fine, Nkz=48, B=0,
            mu=MU0, q=float(2*np.pi/24), maximum_scaled_response_error=error,
            tolerance=REFINEMENT_TOLERANCE, converged=error<=REFINEMENT_TOLERANCE))
    write_csv(output/"response.csv", spectra)
    write_csv(output/"diagnostics.csv", diagnostics)
    write_csv(output/"kz_refinement.csv", refinement)
    write_csv(output/"transverse_refinement.csv", transverse)
    make_figure(output, spectra, diagnostics, refinement, transverse)
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FINITE_SYSTEM_IDENTITIES_PASS",
        "identity_cases": len(diagnostics), "response_rows": len(spectra),
        "all_exact_identity_checks_pass": True,
        "response_mesh_converged": all(x["converged"] for x in refinement+transverse),
        "collective_pole_or_gap_certified": False,
        "submission_ready": False,
        "settings": dict(mu0=MU0, temperature=TEMPERATURE, eta=ETA,
                         coulomb=COULOMB, axial_interaction=AXIAL_U,
                         common_target_density=target,
                         target_reference=parameter_dict(reference_parameters),
                         frequency_points=len(OMEGAS), refinement_tolerance=REFINEMENT_TOLERANCE),
        "maxima": {key:max(x[key] for x in diagnostics) for key in
                   ("vertex_error","charge_ward_bare","charge_ward_rpa",
                    "axial_vertex_error","axial_ward_with_torque","cofactor_identity_error")},
        "minimum_rpa_power_eigenvalue": min(x["rpa_power_min_eigenvalue"] for x in diagnostics),
        "meshes": meshes,
        "limitations": [
            "Repulsive direct RPA for an explicit local axial bilinear; no exchange/self-energy or material fit.",
            "tau_x is not a conserved axial charge; Wilson torque is included in its continuity check.",
            "Fixed internal exchange breaks time reversal; no external-B parity is assumed.",
            "Fixed-density runs share a finite-grid L=8,Nkz=48,B=0 reference density, not a continuum density.",
            "Field is quantized; changing L at fixed flux changes B. Only B=0 is used for transverse refinement.",
            "eta is a resolvent regulator, not a microscopic or Mermin collision rate.",
            "Finite-system passivity and Ward checks do not establish continuum convergence, collective poles, or an avoided-crossing gap."
        ],
        "source_sha256": sources, "python": sys.version, "platform": platform.platform(),
        "output_sha256": {p.name:sha(p) for p in output.iterdir() if p.is_file()},
    }
    assert sources == {p.name:sha(p) for p in source_paths}, "Sources changed during run"
    (output/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in
                     ("status","identity_cases","response_rows","response_mesh_converged","maxima")},indent=2))


def make_figure(output, spectra, diagnostics, refinement, transverse):
    plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(10,7), constrained_layout=True)
    colors = {-1:"#0072b2",0:"#333333",1:"#d55e00"}
    for flux in (-1,0,1):
        ensemble = "fixed_n"
        rows=[r for r in spectra if r["L"]==6 and r["Nkz"]==48
              and r["flux"]==flux and r["ensemble"]==ensemble]
        for axis,key in ((axes[0,0],"loss_cc"),(axes[0,1],"loss_aa")):
            axis.plot([r["omega"] for r in rows],[r[key] for r in rows],
                      color=colors[flux],label=f"flux {flux:+d}")
            axis.set(xlabel="Frequency (hopping units)",ylabel=key.replace("loss_","-Im chi_"))
    axes[0,0].set_title("(a) Charge: fixed density on L=6")
    axes[0,1].set_title("(b) Local axial observable: same model")
    axes[0,0].legend(frameon=False)
    for size,marker in ((4,"o"),(6,"s")):
        rows=[d for d in diagnostics if d["L"]==size and d["Nkz"]==48
              and d["ensemble"]=="fixed_n"]
        axes[1,0].plot([d["B"] for d in rows],[d["mu"] for d in rows],
                       marker=marker,label=f"L={size}")
    axes[1,0].set(xlabel="External B (flux per plaquette)",ylabel="Chemical potential",
                  title="(c) Fixed density: one common reference")
    axes[1,0].legend(frameon=False)
    r=[d for d in diagnostics if d["L"]==6 and d["Nkz"]==48
       and d["ensemble"]=="fixed_n"]
    for key,style,label in (("axial_ward_without_torque","o-","Torque omitted"),
                            ("axial_ward_with_torque","s--","Torque included")):
        axes[1,1].semilogy([d["B"] for d in r],
                           [max(d[key],1e-18) for d in r],style,label=label)
    axes[1,1].set(xlabel="External B",ylabel="Axial continuity defect",
                  title="(d) Wilson torque is required")
    axes[1,1].legend(frameon=False)
    fig.suptitle("PRB exploratory lattice benchmark ? finite-system spectra, no certified collective gap",fontsize=11)
    for suffix in ("pdf","png"):
        fig.savefig(output/("lattice_response."+suffix),dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output.resolve())
