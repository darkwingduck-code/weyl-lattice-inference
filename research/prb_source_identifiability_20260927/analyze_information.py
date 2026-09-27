"""Separate source exchange ambiguity from interaction information."""
from pathlib import Path
import argparse
import numpy as np
from run_study import (DESIGN, ROOT, OLD, PAIRS, apply_transfer, clean_observations,
                       dump, inputs, physical, real_residual, sha, transfers)


def project_out(columns, nuisance, absolute_tolerance=0.):
    if nuisance.size == 0:
        return columns, 0
    u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
    rank = int(np.count_nonzero(s > max(s[0]*1e-10, absolute_tolerance))) if len(s) and s[0] else 0
    return columns - u[:, :rank] @ (u[:, :rank].T @ columns), rank


def information(data, fields, observation, directions, sigma, gain_policy):
    truth = clean_observations(data, directions)
    zeros = {f: np.zeros_like(truth[f]) for f in fields}
    parameters = np.array([.8, 8., *directions])
    physical_columns = []
    for index, scale in enumerate((.8, 8., 1., 1.)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += 1e-5*scale
        minus[index] -= 1e-5*scale
        responses = {}
        for f in fields:
            dp, sp = transfers(data[f]["omega"])
            responses[f] = apply_transfer((physical(data[f], *plus) -
                                           physical(data[f], *minus))/(2e-5), dp, sp)
        physical_columns.append(real_residual(responses, zeros, fields, observation, sigma))
    physical_jac = np.column_stack(physical_columns)
    nuisance_columns = []
    for f in fields:
        item = data[f]
        response = physical(item, *parameters)
        d, s = transfers(item["omega"])
        basis = (np.column_stack((np.ones_like(d),
                  (item["omega"]-item["omega"].mean())/np.ptp(item["omega"])))
                 if gain_policy == "affine" else np.eye(len(d)))
        for side in ("readout", "source"):
            for power in range(basis.shape[1]):
                for phase in (1., 1j):
                    change = np.zeros_like(response)
                    if side == "readout":
                        change[:, 1, 0] = response[:, 1, 0]*basis[:, power]*phase
                        change[:, 1, 1] = response[:, 1, 1]*s*basis[:, power]*phase
                    else:
                        change[:, 0, 1] = response[:, 0, 1]*basis[:, power]*phase
                        change[:, 1, 1] = response[:, 1, 1]*d*basis[:, power]*phase
                    changes = {field: np.zeros_like(truth[field]) for field in fields}
                    changes[f] = change
                    nuisance_columns.append(real_residual(changes, zeros, fields, observation, sigma))
    nuisance = np.column_stack(nuisance_columns)
    reduced, nuisance_rank = project_out(physical_jac, nuisance)
    # Finite differences leave numerical residue in exactly unobservable
    # directions. Do not promote that residue to a source nuisance direction.
    source_tolerance = 1e-7*np.linalg.norm(physical_jac, 2)
    interactions, direction_rank = project_out(reduced[:, :2], reduced[:, 2:], source_tolerance)
    singular = np.linalg.svd(reduced, compute_uv=False)
    interaction_singular = np.linalg.svd(interactions, compute_uv=False)
    ncomplex = sum(len(data[f]["omega"])*len(PAIRS[observation]) for f in fields)
    return dict(gain_policy=gain_policy, nuisance_rank=nuisance_rank,
                remaining_source_rank=direction_rank,
                source_numerical_rank_tolerance=source_tolerance,
                physical_singular_values=singular.tolist(),
                interaction_only_singular_values=interaction_singular.tolist(),
                interaction_whitened_singular_values=(interaction_singular*np.sqrt(ncomplex)).tolist(),
                complex_observation_count=ncomplex,
                antisymmetric_direction_norm=float(np.linalg.norm(
                    (reduced[:, 2]-reduced[:, 3])/np.sqrt(2))),
                interaction_condition_number=float(interaction_singular[0]/interaction_singular[-1]))


def exchange(data, directions):
    output = []
    for field, item in data.items():
        model = physical(item, .8, 8., *directions)
        swapped = physical(item, .8, 8., *directions[::-1])
        d, s = transfers(item["omega"])
        if np.any(swapped[:, 1, 0] == 0) or np.any(swapped[:, 0, 1] == 0):
            raise ValueError("Explicit exchange transfer needs nonzero cross responses")
        d2 = d*model[:, 1, 0]/swapped[:, 1, 0]
        s2 = s*model[:, 0, 1]/swapped[:, 0, 1]
        truth = apply_transfer(model, d, s)
        predicted = apply_transfer(swapped, d2, s2)
        basis = np.column_stack((np.ones_like(d),
            (item["omega"]-item["omega"].mean())/np.ptp(item["omega"])))
        d_affine = basis @ np.linalg.lstsq(basis, d2, rcond=None)[0]
        s_affine = basis @ np.linalg.lstsq(basis, s2, rcond=None)[0]
        affine = apply_transfer(swapped, d_affine, s_affine)
        closure = lambda x: x[:, 0, 1]*x[:, 1, 0]/x[:, 1, 1]
        output.append(dict(field=field,
            bare_reciprocity_max=float(np.max(abs(
                item["bare"]-item["bare"].swapaxes(-1, -2)))),
            closure_swap_relative=float(np.linalg.norm(closure(model)-closure(swapped))/
                                        np.linalg.norm(closure(model))),
            arbitrary_gain_exchange_relative=float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth)),
            affine_projection_exchange_relative=float(np.linalg.norm(affine-truth)/np.linalg.norm(truth)),
            required_readout_nonaffinity=float(np.linalg.norm(d2-d_affine)/np.linalg.norm(d2)),
            required_source_nonaffinity=float(np.linalg.norm(s2-s_affine)/np.linalg.norm(s2))))
    return output


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    paths, data = inputs()
    sources = [*paths.values(), Path(__file__), ROOT/"run_study.py",
               ROOT/"frozen_design.json", ROOT/"response_derivatives.py",
               OLD/"mixed_vertex_response.py", OLD/"noise_sensitivity.py",
               OLD/"analyze_inference.py"]
    hashes = {str(path): sha(path) for path in sources}
    truths = {name: clean_observations(data, directions)
              for name, directions in DESIGN["truths"].items()}
    common_rms = float(np.sqrt(np.mean(np.concatenate([
        abs(value).ravel()**2 for truth in truths.values() for value in truth.values()]))))
    sigma = DESIGN["epsilon"]*common_rms
    information_rows, exchange_rows = [], []
    for name, directions in DESIGN["truths"].items():
        exchange_rows += [{"truth": name, **row} for row in exchange(data, directions)]
        for group, fields in DESIGN["field_groups"].items():
            for observation in DESIGN["observations"]:
                for gain_policy in ("affine", "arbitrary_frequency"):
                    information_rows.append(dict(truth=name, group=group,
                        observation=observation, **information(data, fields, observation,
                                                            directions, sigma, gain_policy)))
    result = dict(absolute_complex_sigma=sigma, information=information_rows,
        exchange_controls=exchange_rows,
        interpretation="Local information after nuisance projection, not global uniqueness or confidence. Exchange ambiguity in source labels is separate from interaction information.",
        jacobian_convention="J_R=(Re J,Im J)/sigma in coordinates (g/.8,Ua/8,lambda_r,lambda_s); Gaussian Fisher is 2 J_R.T J_R. The interaction_whitened_singular_values key has no sqrt(2) Fisher factor.",
        input_source_sha256=hashes,
        inputs_sources_unchanged=all(sha(path) == digest for path, digest in hashes.items()))
    dump(output/"summary.json", result)
    print(result["exchange_controls"])
    for row in information_rows:
        print(row["truth"], row["group"], row["observation"], row["gain_policy"],
              row["interaction_only_singular_values"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
