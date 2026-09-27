"""Extend the zero-field HIGH-FREQUENCY exterior pole at unchanged tolerances.

Uses the frozen all-band Bloch/RationalResponse/exterior search APIs. This is
not a low-energy anomaly-mode claim or a continuum-sheet construction.
"""
from __future__ import annotations

import argparse,json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from lattice_response import Parameters,interaction_matrix
from bloch_response import bloch_spectrum,bloch_transitions
from pole_zero_analysis import RationalResponse,exterior_real_poles
from run_pole_zero_experiment import sha,write_json,write_csv
from extend_transverse import memory_status

ROOT=Path(__file__).resolve().parent
SETTINGS=dict(mu=.5,T=.04,g=.8,Ua=8.,q=2*np.pi/24,
              pole_relative_tolerance=1e-5,residue_relative_tolerance=.01)


def decode_matrix(value):
    return np.array([[complex(entry['real'],entry['imag']) if isinstance(entry,dict)
                      else complex(entry) for entry in row] for row in value])


def main(output,nk,reference):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Use a new empty evidence directory')
    reference_data=json.loads(reference.read_text(encoding='utf-8'))
    if reference_data['settings']!=SETTINGS:
        raise ValueError('Reference settings or tolerances differ')
    if nk==192:
        matches=[case for case in reference_data['exterior_searches']
                 if case['L']==48 and case['Nkz']==192 and case['stage']=='refinement']
        if len(matches)!=1:
            raise ValueError('Expected one original L48/Nk192 reference')
        coarse=matches[0]
    else:
        coarse=reference_data['calculation']
        if (coarse['L'],coarse['Nkz'])!=(72,192):
            raise ValueError('Nk384 continuation requires the L72/Nk192 result')
    for name,value in reference_data['source_sha256'].items():
        if sha(ROOT/name)!=value:
            raise RuntimeError('Reference source changed: '+name)
    for name,value in reference_data['output_sha256'].items():
        if sha(reference.parent/name)!=value:
            raise RuntimeError('Reference output changed: '+name)
    original_reference=ROOT/'mode_analysis'/'diagnostic.json'
    original_search=ROOT/'mode_analysis'/'exterior_searches.json'
    original_comparisons=ROOT/'mode_analysis'/'exterior_refinement.csv'
    inputs=list(dict.fromkeys((reference,original_reference,original_search,original_comparisons)))
    sources=[ROOT/name for name in ('lattice_response.py','bloch_response.py','pole_zero_analysis.py',
                                   'run_pole_zero_experiment.py','extend_transverse.py','extend_exterior_pole.py')]
    source_hashes={path.name:sha(path) for path in sources}
    input_hashes={str(path.relative_to(ROOT)):sha(path) for path in inputs}
    output.mkdir(parents=True,exist_ok=True)
    def log(message):
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
    started=time.perf_counter()
    p=Parameters(Lx=72,Ly=72,Nkz=nk)
    raw=p.volume*16
    # Includes raw weights, eigensystems, active-array copies, G factorization,
    # and temporary scaled rank-one checks; conservative, not measured RSS.
    estimated_peak=288*p.volume+256*raw+256*1024**2
    available=memory_status()['available_physical_bytes']
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        settings=SETTINGS,parameters=p.__dict__,source_sha256=source_hashes,input_sha256=input_hashes,
        reference_record=dict(L=coarse['L'],Nkz=coarse['Nkz']),reference_evidence_verified=True,
        memory_before=memory_status(),estimated_peak_bytes=estimated_peak,
        memory_admission_required_bytes=int(1.5*estimated_peak+512*1024**2),
        passes_declared_tolerances=False,longitudinal_at_L72_checked=(nk==384),
        low_energy_hybrid_gap_certified=False,continuum_landau_sheet_constructed=False,
        limitations=['Only a finite high-frequency positive exterior root is followed.',
          'Near-edge excluded slivers, internal poles, and low-energy modes remain outside this search.',
          'Convergence in one grid direction does not establish convergence in another.'])
    def checkpoint():
        write_json(output/'diagnostic.json',report)
    write_json(output/'reference_record.json',coarse)
    checkpoint()
    try:
        if available<report['memory_admission_required_bytes']:
            raise MemoryError('Insufficient RAM for the conservative complete-array budget; use an exact streamed continuation')
        log(f'[exterior] L=72,Nkz={nk}, raw slots={raw}; available={available}, estimated peak={estimated_peak}')
        spec=bloch_spectrum(p)
        log(f'Bloch eigensystem ready in {time.perf_counter()-started:.1f}s')
        tr=bloch_transitions(spec,.5,.04,nk//24)
        vertex_error=tr.vertex_error
        density_value=tr.density_value
        del spec
        report['memory_after_transitions']=memory_status()
        log(f'All transitions ready in {time.perf_counter()-started:.1f}s; vertex={vertex_error:.6g}')
        response=RationalResponse(tr)
        interaction=interaction_matrix(tr.Q,.8,8.)
        del tr
        report['memory_after_rational_response']=memory_status()
        log(f'Rational response: active={len(response.delta)}, exact-zero removed={response.zero_weight_count}')
        scan=exterior_real_poles(response,interaction,samples=64)
        current=dict(L=72,Nkz=nk,q=SETTINGS['q'],Ua=8.,stage='exterior_extension',
            density=density_value,active_transitions=len(response.delta),raw_band_pair_slots=raw,
            exactly_zero_weights_removed=response.zero_weight_count,vertex_error=vertex_error,
            factorization_max_error=response.factorization_max_error,
            moment0_max=float(np.max(np.abs(response.moment0))),**scan)
        report['calculation']=current
        write_json(output/'exterior_search.json',current)
        isolated=lambda case:[item for item in case['roots']
            if item.get('residue_status')=='isolated_nullspace_checked'
            and not item.get('nearby_root_cluster',True)]
        old,new=isolated(coarse),isolated(current)
        comparison=dict(coarse_L=coarse['L'],fine_L=72,coarse_Nkz=coarse['Nkz'],fine_Nkz=nk,
            pole_relative_tolerance=1e-5,residue_relative_tolerance=.01,
            passes_declared_tolerances=False)
        if len(old)==len(new)==1 and len(coarse['roots'])==len(current['roots'])==1:
            a,b=old[0],new[0]
            ra,rb=decode_matrix(a['residue_matrix']),np.asarray(b['residue_matrix'])
            comparison.update(pole_relative_error=abs(a['pole']-b['pole'])/abs(b['pole']),
                charge_residue_relative_error=float(abs(ra[0,0]-rb[0,0])/abs(rb[0,0])),
                axial_residue_relative_error=float(abs(ra[1,1]-rb[1,1])/abs(rb[1,1])))
            comparison['passes_declared_tolerances']=bool(comparison['pole_relative_error']<=1e-5
                and max(comparison['charge_residue_relative_error'],comparison['axial_residue_relative_error'])<=.01)
            report['matched_root']=dict(pole=b['pole'],charge_residue=rb[0,0],axial_residue=rb[1,1],
                residue_matrix=rb,nearest_bare_distance=b['nearest_bare_distance'],
                nullspace_eigenvalue_residual=b['nullspace_eigenvalue_residual'])
        else:
            comparison['unresolved_reason']='Not exactly one isolated matched exterior root in each grid'
        report['comparison']=comparison
        write_csv(output/'comparison.csv',[comparison])
        report['passes_declared_tolerances']=comparison['passes_declared_tolerances']
        report['exact_checks_pass']=bool(vertex_error<2e-11
            and all(item.get('residue_status')=='isolated_nullspace_checked' for item in current['roots']))
        if not report['exact_checks_pass']:
            raise RuntimeError('Vertex or exterior nullspace verification failed')
        report['status']=('EXTERIOR_EXTENSION_CONVERGED' if report['passes_declared_tolerances']
                          else 'EXTERIOR_EXTENSION_COMPLETED_UNCONVERGED')
        if nk==192:
            report['next_required_check']='At the same L72, compare Nkz192 against Nkz384; L48 longitudinal stability is insufficient.'
        log('Comparison: '+json.dumps(comparison))
        if 'matched_root' in report:
            log(f"Pole={report['matched_root']['pole']:.15g}; charge residue={report['matched_root']['charge_residue']}; axial residue={report['matched_root']['axial_residue']}")
    except Exception as exc:
        report['status']='EXTERIOR_EXTENSION_FAILED'
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
        log(traceback.format_exc())
    report['sources_unchanged']=source_hashes=={path.name:sha(path) for path in sources}
    report['inputs_unchanged']=input_hashes=={str(path.relative_to(ROOT)):sha(path) for path in inputs}
    if not report['sources_unchanged'] or not report['inputs_unchanged']:
        report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-started
    report['completed_utc']=datetime.now(timezone.utc).isoformat()
    log('Final status: '+report['status'])
    report['output_sha256']={path.name:sha(path) for path in output.iterdir()
                             if path.is_file() and path.name!='diagnostic.json'}
    checkpoint()
    if report['status'] in ('EXTERIOR_EXTENSION_FAILED','SOURCE_OR_INPUT_CHANGED'):
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nkz',type=int,choices=(192,384),default=192)
    parser.add_argument('--reference-report',type=Path,default=ROOT/'mode_analysis'/'diagnostic.json')
    args=parser.parse_args()
    main(args.output.resolve(),args.nkz,args.reference_report.resolve())