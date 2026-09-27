"""Class-aware direct/streamed exterior comparison; no production source edits.

Unresolved roots remain unresolved and are checked for matching classifications,
boundaries, and diagnostics. Residue matrices are compared only for roots whose
nullspace checks produced matrices in both paths. This does not certify a large
streamed production run.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from stream_exterior_pole import (
    Parameters, bloch_spectrum, bloch_transitions, RationalResponse,
    StreamedRationalResponse, tile_transitions, interaction_matrix,
    exterior_real_poles, sha, write_json,
)

ROOT = Path(__file__).resolve().parent


def compare_scans(direct, streamed, budget):
    failures, errors, root_records = [], {}, []
    for key in ('status', 'scope', 'samples', 'near_edge_unresolved'):
        if direct.get(key) != streamed.get(key):
            failures.append('scan classification: ' + key)
    for key in ('positive_bare_edge', 'eigenvalue_bound', 'search_margin'):
        errors[key] = float(abs(direct[key] - streamed[key]))
    errors['excluded_edge_interval'] = float(np.max(np.abs(
        np.asarray(direct['excluded_edge_interval']) - np.asarray(streamed['excluded_edge_interval']))))
    if len(direct['roots']) != len(streamed['roots']):
        failures.append('root count')
    for i, (a, b) in enumerate(zip(direct['roots'], streamed['roots'])):
        record = dict(index=i, direct_status=a.get('residue_status'),
                      streamed_status=b.get('residue_status'), pole_direct=a['pole'],
                      pole_streamed=b['pole'], direct_has_matrix='residue_matrix' in a,
                      streamed_has_matrix='residue_matrix' in b)
        for key in ('dielectric_channel', 'residue_status', 'residue_failure', 'nearby_root_cluster'):
            if a.get(key) != b.get(key):
                failures.append(f'root{i} classification: {key}')
        for key in ('pole', 'determinant_residual', 'charge_numerator',
                    'nearest_bare_distance', 'cofactor_charge_residue',
                    'nullspace_eigenvalue_residual'):
            if (key in a) != (key in b):
                failures.append(f'root{i} field presence: {key}')
            elif key in a:
                if (a[key] is None) != (b[key] is None):
                    failures.append(f'root{i} unavailable diagnostic: {key}')
                elif a[key] is not None:
                    errors[f'root{i}_{key}'] = float(abs(a[key] - b[key]))
        ah, bh = 'residue_matrix' in a, 'residue_matrix' in b
        if ah != bh:
            failures.append(f'root{i} matrix presence')
        elif a.get('residue_status') == b.get('residue_status') == 'isolated_nullspace_checked':
            if not ah:
                failures.append(f'root{i} isolated root missing matrix')
            else:
                ra, rb = np.asarray(a['residue_matrix']), np.asarray(b['residue_matrix'])
                errors[f'root{i}_residue_matrix'] = float(np.max(np.abs(ra-rb)))
                charge_delta = float(abs(ra[0, 0]-rb[0, 0]))
                charge_relative = charge_delta / float(abs(ra[0, 0])) if ra[0, 0] != 0 else None
                record.update(matrix_compared=True,
                    charge_residue_direct=complex(ra[0, 0]), charge_residue_streamed=complex(rb[0, 0]),
                    charge_residue_absolute_difference=charge_delta,
                    charge_residue_relative_difference=charge_relative,
                    axial_residue_absolute_difference=float(abs(ra[1, 1]-rb[1, 1])))
        elif ah:
            failures.append(f'root{i} unexpected matrix for unresolved classification')
        else:
            record.update(matrix_compared=False, unresolved_preserved=True,
                direct_failure=a.get('residue_failure'), streamed_failure=b.get('residue_failure'))
        root_records.append(record)
    failures.extend(key + ': numeric difference exceeds budget'
                    for key, value in errors.items() if not np.isfinite(value) or value > budget)
    return dict(passed=not failures, failures=failures, absolute_errors=errors, roots=root_records)


def main(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Fresh evidence directory required')
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT/name for name in (
        'lattice_response.py', 'bloch_response.py', 'pole_zero_analysis.py',
        'run_pole_zero_experiment.py', 'extend_transverse.py', 'extend_exterior_pole.py',
        'stream_exterior_pole.py', 'validate_stream_exterior.py')]
    hashes = {p.name: sha(p) for p in sources}
    failed = ROOT/'exterior_kz_streamed'/'diagnostic.json'
    p = Parameters(Lx=3, Ly=4, Nkz=24, theta_x=.21, theta_y=.37, theta_z=.19)
    tr = bloch_transitions(bloch_spectrum(p), .5, .04, 1)
    direct = RationalResponse(tr)
    streamed = StreamedRationalResponse([
        RationalResponse(tile_transitions(p, first, min(first+5, 12), .5, .04, 1))
        for first in range(0, 12, 5)])
    u = interaction_matrix(tr.Q, .8, 8.)
    a, b = exterior_real_poles(direct, u, samples=64), exterior_real_poles(streamed, u, samples=64)
    budget = float(128*np.finfo(float).eps*p.volume*max(1., a['eigenvalue_bound']))
    comparison = compare_scans(a, b, budget)
    response_errors = {}
    for z in (.47+.03j, 15.5, 18.1+.2j):
        for derivative in (0, 1):
            response_errors[f'z={z};derivative={derivative}'] = float(np.max(np.abs(
                direct.polarization(z, derivative)-streamed.polarization(z, derivative))))
    negatives = {}
    altered = copy.deepcopy(b)
    altered['roots'][0]['residue_status'] = 'isolated_nullspace_checked'
    negatives['one_sided_classification_change'] = compare_scans(a, altered, budget)
    altered = copy.deepcopy(b)
    altered['roots'].pop()
    negatives['missing_root'] = compare_scans(a, altered, budget)
    altered = copy.deepcopy(b)
    matrix_index = next(i for i, r in enumerate(altered['roots']) if 'residue_matrix' in r)
    del altered['roots'][matrix_index]['residue_matrix']
    negatives['one_sided_missing_matrix'] = compare_scans(a, altered, budget)
    altered = copy.deepcopy(b)
    altered['roots'][matrix_index]['pole'] += 10*budget
    negatives['pole_beyond_budget'] = compare_scans(a, altered, budget)
    altered = copy.deepcopy(b)
    altered['roots'][0]['residue_failure'] = 'Different unresolved condition'
    negatives['different_unresolved_reason'] = compare_scans(a, altered, budget)
    negatives_pass = all(not item['passed'] for item in negatives.values())
    counts_pass = (direct.original_count == streamed.original_count == p.volume*16
                   and len(direct.delta) == len(streamed.delta)
                   and direct.zero_weight_count == streamed.zero_weight_count)
    passed = bool(comparison['passed'] and negatives_pass and counts_pass
                  and max(response_errors.values()) <= budget)
    write_json(output/'direct_search.json', a)
    write_json(output/'streamed_search.json', b)
    report = dict(status='CLASS_AWARE_SMALL_STREAM_VALIDATION_PASS' if passed else 'CLASS_AWARE_SMALL_STREAM_VALIDATION_FAIL',
        all_tests_pass=passed, completed_utc=datetime.now(timezone.utc).isoformat(), parameters=p.__dict__,
        roundoff_budget=budget, budget_definition='128 eps times volume times max(1, direct exterior spectral bound); unchanged from original validator',
        source_sha256=hashes, input_sha256={str(failed.relative_to(ROOT)): sha(failed)},
        raw_count=direct.original_count, active_count=len(direct.delta),
        exactly_zero_weights_removed=direct.zero_weight_count, counts_pass=counts_pass,
        response_absolute_errors=response_errors, comparison=comparison,
        negative_controls=negatives, all_negative_controls_rejected=negatives_pass,
        gram_factorization_error=streamed.gram_factorization_error,
        scope='Small direct-versus-streamed path validation only. Both unresolved near-edge roots remain unresolved. No large streamed production or low-energy pole/gap certification.')
    report['sources_unchanged'] = hashes == {p.name: sha(p) for p in sources}
    report['all_tests_pass'] = passed and report['sources_unchanged']
    if not report['all_tests_pass']:
        report['status'] = 'CLASS_AWARE_SMALL_STREAM_VALIDATION_FAIL'
    report['output_sha256'] = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    write_json(output/'diagnostic.json', report)
    print(report['status'])
    print('Response absolute errors:', response_errors)
    print('Root records:', comparison['roots'])
    print('All negative controls rejected:', negatives_pass)
    if not report['all_tests_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
