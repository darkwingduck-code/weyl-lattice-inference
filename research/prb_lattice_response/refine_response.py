"""Refine the initial PRB spectra without changing frequency, eta, or tolerance."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from lattice_response import (Parameters, diagonalize, fixed_density_mu,
    transitions, interaction_matrix, rpa_response, loss_matrix)
from bloch_response import bloch_spectrum, bloch_transitions

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Refinement output must be empty")
    source_paths = [ROOT/name for name in
        ("lattice_response.py", "bloch_response.py", "refine_response.py")]
    source_hashes = {p.name:sha(p) for p in source_paths}
    baseline = ROOT/"output_initial"
    metadata = json.loads((baseline/"diagnostic.json").read_text())
    initial = list(csv.DictReader((baseline/"response.csv").open()))
    settings = metadata["settings"]
    eta, temperature = settings["eta"], settings["temperature"]
    target = settings["common_target_density"]
    omegas = np.array(sorted({float(row["omega"]) for row in initial}))
    rows, comparisons = [], []

    def read_reference(size, nk, flux, ensemble):
        selected=[r for r in initial if int(r["L"])==size and int(r["Nkz"])==nk
                  and int(r["flux"])==flux and r["ensemble"]==ensemble]
        assert len(selected)==len(omegas)
        return np.array([[[complex(float(r[f"rpa_{a}{b}_re"]),float(r[f"rpa_{a}{b}_im"]))
                           for b in range(2)] for a in range(2)] for r in selected])

    def evaluate(size, nk, flux, ensemble, use_bloch):
        print(f"[refine] L={size}, Nkz={nk}, flux={flux}, {ensemble}",flush=True)
        p=Parameters(Lx=size,Ly=size,Nkz=nk)
        spec=bloch_spectrum(p) if use_bloch else diagonalize(p,flux)
        mu=settings["mu0"] if ensemble=="fixed_mu" else fixed_density_mu(spec,target,temperature)
        tr=(bloch_transitions(spec,mu,temperature,nk//24) if use_bloch
            else transitions(spec,mu,temperature,nk//24))
        coupling=interaction_matrix(tr.Q,settings["coulomb"],settings["axial_interaction"])
        measured=[]
        power_min=float("inf")
        for omega in omegas:
            z=complex(omega,eta)
            bare=tr.density_response(z)
            response=rpa_response(bare,coupling)
            measured.append(response)
            power_min=min(power_min,float(np.linalg.eigvalsh(loss_matrix(z*response)).min()))
            row=dict(L=size,Nkz=nk,flux=flux,B=p.field(flux),ensemble=ensemble,
                     mu=mu,density=tr.density_value,q=tr.q,omega=float(omega),eta=eta)
            for a,b in ((0,0),(0,1),(1,0),(1,1)):
                row[f"rpa_{a}{b}_re"]=float(response[a,b].real)
                row[f"rpa_{a}{b}_im"]=float(response[a,b].imag)
            rows.append(row)
        assert tr.vertex_error<2e-11 and power_min>-2e-10
        assert max(tr.ward_errors(.45+.015j,coupling))<2e-10
        filename=f"L{size}_Nk{nk}_flux{flux:+d}_{ensemble}.csv"
        csv_write(output/filename,rows[-len(omegas):])
        return np.array(measured),dict(vertex_error=tr.vertex_error,power_min=power_min)

    def compare(coarse,fine,description,checks):
        error=float(np.max(np.abs(coarse-fine)/(1+np.abs(fine))))
        record=dict(description,maximum_scaled_response_error=error,
                    tolerance=TOLERANCE,converged=error<=TOLERANCE,**checks)
        comparisons.append(record)
        csv_write(output/"refinement.csv",comparisons)
        print(f"  maximum scaled error={error:.6g}; converged={error<=TOLERANCE}",flush=True)
        return error<=TOLERANCE

    for flux in (-1,1):
        previous=read_reference(6,48,flux,"fixed_n")
        previous_nk=48
        for nk in (96,192,384):
            current,checks=evaluate(6,nk,flux,"fixed_n",False)
            passed=compare(previous,current,dict(kind="fixed_B_kz",flux=flux,
                coarse_L=6,fine_L=6,coarse_Nkz=previous_nk,fine_Nkz=nk),checks)
            previous,previous_nk=current,nk
            if passed:
                break
    previous=read_reference(8,48,0,"fixed_mu")
    previous_size,previous_nk=8,48
    for size,nk in ((8,96),(8,192),(12,192),(24,192),(48,192)):
        current,checks=evaluate(size,nk,0,"fixed_mu",True)
        compare(previous,current,dict(kind="zero_B_kz" if size==previous_size else "zero_B_transverse",
            flux=0,coarse_L=previous_size,fine_L=size,coarse_Nkz=previous_nk,fine_Nkz=nk),checks)
        previous,previous_size,previous_nk=current,size,nk
    assert source_hashes=={p.name:sha(p) for p in source_paths}
    final={key:next(r for r in reversed(comparisons) if r["kind"]==kind and r["flux"]==flux)
           for key,kind,flux in (("negative_B_kz","fixed_B_kz",-1),
                ("positive_B_kz","fixed_B_kz",1),("zero_B_kz","zero_B_kz",0),
                ("zero_B_transverse","zero_B_transverse",0))}
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        status="REFINEMENT_COMPLETED",frequency_points=len(omegas),
        eta=eta,tolerance=TOLERANCE,baseline_sha256=sha(baseline/"diagnostic.json"),
        source_sha256=source_hashes,comparisons=comparisons,
        final_comparisons=final,all_selected_directions_converged=all(r["converged"] for r in final.values()),
        collective_pole_or_gap_certified=False,
        limitations=["Only two nonzero-flux L=6 fixed-density cases are refined here.",
          "Longitudinal convergence at fixed finite L is not a transverse thermodynamic limit.",
          "The B=0 Bloch path is an exact decomposition and is not used at nonzero flux.",
          "No broadening, frequency samples or tolerance were changed to obtain a pass.",
          "This is spectral mesh refinement, not a residue or complex-pole certification."],
        output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"final_comparisons":final,
                     "all_selected_directions_converged":report["all_selected_directions_converged"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output.resolve())
