"""Compare lattice grids in a fixed absolute detector-noise convention."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "prb_mixed_source_response_20260914"
sys.path.insert(0, str(OLD))
from mixed_vertex_response import apply_transfer, contract_mixed, dress_response


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def observed(data, lr=.7, ls=.7):
    omega = data["omega"]
    x = (omega - omega.mean()) / np.ptp(omega)
    physical = contract_mixed(dress_response(data["bare"], float(data["Q"]),
                                             .8, 8.), lr, ls)
    return apply_transfer(physical, .7*np.exp(.4j)*(1+.1*x),
                          1.3*np.exp(-.25j)*(1+.1j*x))


def metrics(model, reference, sigma):
    if model.shape != reference.shape or sigma <= 0:
        raise ValueError("Matching response arrays and positive noise required")
    delta = model - reference
    rms = float(np.sqrt(np.mean(abs(delta)**2)))
    row = {"absolute_complex_rms": rms, "raw_noise_units": rms/sigma}
    for a, b in np.ndindex(2, 2):
        scale = np.linalg.norm(reference[:, a, b])
        row[f"relative_l2_{a}{b}"] = (
            float(np.linalg.norm(delta[:, a, b])/scale) if scale else None)
        row[f"noise_units_{a}{b}"] = float(
            np.sqrt(np.mean(abs(delta[:, a, b])**2))/sigma)
    return row


def self_check():
    # A circular-complex RMS sigma counts each complex entry once, not twice.
    ref = np.ones((7, 2, 2), complex)
    sigma = .2
    result = metrics(ref + sigma*(1+1j)/np.sqrt(2), ref, sigma)
    assert abs(result["raw_noise_units"]-1) < 1e-14
    model = ref.copy()
    model[:, 0, 1] += 2*sigma
    result = metrics(model, ref, sigma)
    assert abs(result["raw_noise_units"]-1) < 1e-14
    assert result["noise_units_10"] == 0
    assert abs(result["noise_units_01"]-2) < 1e-14


def main(output, require_all):
    if output.exists():
        raise FileExistsError("Choose a new comparison directory")
    paths = {field: {
        "coarse": OLD/"production"/f"{field}_coarse"/"response_3channel.npz",
        "final": OLD/"production"/f"{field}_final"/"response_3channel.npz",
    } for field in ("Bplus", "B0")}
    for label in ("x_refined", "y_refined", "z_refined", "combined"):
        paths["Bplus"][label] = ROOT/f"Bplus_{label}"/"response_3channel.npz"
    paths["B0"]["higher"] = ROOT/"B0_higher"/"response_3channel.npz"
    missing = [str(path.relative_to(ROOT.parent.parent))
               for group in paths.values() for path in group.values()
               if not path.is_file()]
    if require_all and missing:
        raise FileNotFoundError("Incomplete refinement: " + ", ".join(missing))
    data = {field: {name: load(path) for name, path in group.items()
                    if path.is_file()} for field, group in paths.items()}
    scale = float(np.sqrt(np.mean([np.mean(abs(observed(data[f]["final"]))**2)
                                   for f in ("Bplus", "B0")])))
    sigma = .001*scale
    source_scale = float(np.sqrt(np.mean([
        np.mean(abs(observed(data[f]["final"], lr, ls))**2)
        for f in ("Bplus", "B0") for lr, ls in ((.7, .7), (.7, -.3))])))
    source_sigma = .001*source_scale
    rows = []
    for field, group in data.items():
        targets = ["final"] + (["combined"] if "combined" in group else [])
        targets += ["higher"] if "higher" in group else []
        for target in targets:
            reference = group[target]
            for label, model in group.items():
                if label == target:
                    continue
                for key in ("omega", "q", "Q", "eta", "mu", "temperature", "field"):
                    if not np.array_equal(model[key], reference[key]):
                        raise ValueError(f"Different physical input {key}: {field}/{label}")
                for lr, ls in ((.7, .7), (.7, -.3)):
                    row = {"field": field, "model_mesh": label,
                           "reference_mesh": target, "lambda_readout": lr,
                           "lambda_source": ls, "sigma": sigma}
                    row.update(metrics(observed(model, lr, ls),
                                       observed(reference, lr, ls), sigma))
                    row["source_study_noise_units"] = (
                        row["absolute_complex_rms"]/source_sigma)
                    row["bare_3channel_relative_l2"] = float(
                        np.linalg.norm(model["bare"]-reference["bare"])
                        /np.linalg.norm(reference["bare"]))
                    rows.append(row)
    sources = [Path(__file__), OLD/"mixed_vertex_response.py"]
    inputs = [path for group in paths.values() for path in group.values()
              if path.is_file()]
    report = {
        "status": "ALL_DECLARED_GRIDS_COMPARED" if not missing else "PARTIAL_COMPARISON",
        "missing": missing, "fixed_common_amplitude": scale, "sigma": sigma,
        "noise_definition": "E|e|^2=sigma^2; one common absolute complex RMS in all entries",
        "amplitude_reference": "Existing Bplus/B0 final, lambda_readout=lambda_source=.7",
        "source_study_sigma": source_sigma,
        "source_study_amplitude_reference": "Existing Bplus/B0 final, both declared source directions; conversion only, original refinement norm retained",
        "continuum_error_certified": False,
        "scope": "Finite-grid differences, with no fitting or post hoc pass threshold",
        "sha256": {str(path.relative_to(ROOT.parent.parent)):
                   hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sources+inputs},
        "comparisons": rows,
    }
    output.mkdir(parents=True)
    with (output/"comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                       encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "missing", "sigma")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("PASS: complex RMS and cross-entry checks")
    elif args.output is None:
        parser.error("--output is required unless --self-check is selected")
    else:
        main(args.output.resolve(), args.require_all)
