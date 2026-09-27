"""Exact all-transition streamed continuation of the high-frequency exterior pole.

Every nonzero Lehmann weight remains in a bounded set of transverse chunks.
Polarization and its derivative are summed before any RPA operation. The
2x2 factor G below preserves sum(G_chunk G_chunk^dagger) for the exterior
spectral bound only; it is not a reduced modal realization or a band cutoff.
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
from extend_transverse import memory_status,tile_transitions
from extend_exterior_pole import SETTINGS,decode_matrix

ROOT=Path(__file__).resolve().parent


class StreamedRationalResponse:
    """Exact sum of retained chunk responses, with global bare support arrays."""
    cofactor=RationalResponse.cofactor
    dressed=RationalResponse.dressed

    def __init__(self,parts):
        self.parts=tuple(parts)
        self.delta=np.concatenate([part.delta for part in parts])
        self.cc=np.concatenate([part.cc for part in parts])
        self.aa=np.concatenate([part.aa for part in parts])
        self.original_count=sum(part.original_count for part in parts)
        self.zero_weight_count=sum(part.zero_weight_count for part in parts)
        self.moment0=sum((part.moment0 for part in parts),np.zeros((2,2),complex))
        gram=sum((part.G@part.G.conj().T for part in parts),np.zeros((2,2),complex))
        # Cholesky must succeed without eigenvalue clipping or fitted thresholds.
        self.G=np.linalg.cholesky(gram)
        self.gram_factorization_error=float(np.max(np.abs(self.G@self.G.conj().T-gram)))
        self.transition_factorization_error=max(part.factorization_max_error for part in parts)
        if self.gram_factorization_error>128*np.finfo(float).eps*max(1.,float(np.linalg.norm(gram))):
            raise RuntimeError('Exterior bound Gram factorization is not roundoff accurate')

    def polarization(self,z,derivative=0):
        return sum((part.polarization(z,derivative) for part in self.parts),np.zeros((2,2),complex))


def validate_stream():
    p=Parameters(Lx=3,Ly=4,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19)
    tr=bloch_transitions(bloch_spectrum(p),.5,.04,1)
    direct=RationalResponse(tr)
    parts=[RationalResponse(tile_transitions(p,i,min(i+5,12),.5,.04,1)) for i in range(0,12,5)]
    streamed=StreamedRationalResponse(parts)
    errors={}
    for z in (.47+.03j,15.5,18.1+.2j):
        for derivative in (0,1):
            errors[f'polarization_z{z}_derivative{derivative}']=float(np.max(np.abs(
                direct.polarization(z,derivative)-streamed.polarization(z,derivative))))
    u=interaction_matrix(tr.Q,.8,8.)
    a,b=exterior_real_poles(direct,u,samples=64),exterior_real_poles(streamed,u,samples=64)
    errors['bare_edge']=abs(a['positive_bare_edge']-b['positive_bare_edge'])
    errors['spectral_bound']=abs(a['eigenvalue_bound']-b['eigenvalue_bound'])
    same_roots=len(a['roots'])==len(b['roots'])
    if same_roots:
        for i,(old,new) in enumerate(zip(a['roots'],b['roots'])):
            errors[f'pole{i}']=abs(old['pole']-new['pole'])
            errors[f'residue_matrix{i}']=float(np.max(np.abs(old['residue_matrix']-new['residue_matrix'])))
    budget=128*np.finfo(float).eps*p.volume*max(1.,a['eigenvalue_bound'])
    passed=bool(same_roots and max(errors.values())<=budget
                and len(streamed.delta)==len(direct.delta)
                and streamed.original_count==direct.original_count)
    return dict(passed=passed,errors=errors,roundoff_budget=budget,parameters=p.__dict__,
        raw_count=streamed.original_count,active_count=len(streamed.delta),
        gram_factorization_error=streamed.gram_factorization_error,
        source='Complete direct Bloch RationalResponse versus exact chunk sum, including derivative and full exterior scan')


def main(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('New empty evidence directory required')
    reference=ROOT/'exterior_extension'/'diagnostic.json'
    old=json.loads(reference.read_text(encoding='utf-8'))
    if old['status']!='EXTERIOR_EXTENSION_CONVERGED' or old['settings']!=SETTINGS:
        raise RuntimeError('Unchanged successful L72/Nk192 reference required')
    for name,value in old['source_sha256'].items():
        if sha(ROOT/name)!=value: raise RuntimeError('Reference source changed: '+name)
    for name,value in old['output_sha256'].items():
        if sha(reference.parent/name)!=value: raise RuntimeError('Reference output changed: '+name)
    sources=[ROOT/name for name in ('lattice_response.py','bloch_response.py','pole_zero_analysis.py',
        'run_pole_zero_experiment.py','extend_transverse.py','extend_exterior_pole.py','stream_exterior_pole.py')]
    inputs=[reference,ROOT/'exterior_extension'/'exterior_search.json',
            ROOT/'exterior_kz_continuation'/'diagnostic.json']
    hashes={path.name:sha(path) for path in sources}
    input_hashes={str(path.relative_to(ROOT)):sha(path) for path in inputs}
    output.mkdir(parents=True,exist_ok=True)
    def log(message):
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as f:f.write(message+'\n')
    started=time.perf_counter()
    p=Parameters(Lx=72,Ly=72,Nkz=384)
    raw=p.volume*16
    # All raw slots upper-bound retained chunk arrays(80B) plus global support,
    # cc,aa arrays(24B), with 256MiB reserved for bounded chunk temporaries.
    estimate=104*raw+256*1024**2
    admission=int(1.5*estimate+512*1024**2)
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),settings=SETTINGS,
        parameters=p.__dict__,source_sha256=hashes,input_sha256=input_hashes,memory_before=memory_status(),
        estimated_peak_bytes=estimate,memory_admission_required_bytes=admission,
        method='Exact chunked RationalResponse sum; Gram-preserving exterior bound; unchanged64-sample exterior scan',
        passes_declared_tolerances=False,low_energy_hybrid_gap_certified=False,
        continuum_landau_sheet_constructed=False,limitations=old['limitations'])
    def checkpoint():write_json(output/'diagnostic.json',report)
    checkpoint()
    try:
        if report['memory_before']['available_physical_bytes']<admission:
            raise MemoryError('Insufficient memory even for the conservative chunked budget')
        validation=validate_stream()
        write_json(output/'path_validation.json',validation)
        report['path_validation_pass']=validation['passed']
        if not validation['passed']:raise RuntimeError('Exact streamed path does not match direct response')
        log(f'[streamed exterior] raw slots={raw}, expected peak={estimate}, admission={admission}')
        parts=[]
        vertex_error,density_value=0.,0.
        block_size=128
        for first in range(0,p.Lx*p.Ly,block_size):
            available=memory_status()['available_physical_bytes']
            # Already-stored chunks are accounted for by available RAM; only
            # fixed temporary+final concatenation headroom is required here.
            if available<24*raw+768*1024**2:
                raise MemoryError('Insufficient remaining headroom for chunk scratch and final support arrays')
            tr=tile_transitions(p,first,min(first+block_size,p.Lx*p.Ly),.5,.04,16)
            vertex_error=max(vertex_error,tr.vertex_error)
            density_value+=tr.density_value
            parts.append(RationalResponse(tr))
            del tr
            if first==0 or len(parts)%8==0 or first+block_size>=p.Lx*p.Ly:
                log(f'Chunks {len(parts)}/41, elapsed={time.perf_counter()-started:.1f}s')
        response=StreamedRationalResponse(parts)
        u=interaction_matrix(2*np.sin(SETTINGS['q']/2),.8,8.)
        report['memory_before_search']=memory_status()
        log(f'Search: active transitions={len(response.delta)}, elapsed={time.perf_counter()-started:.1f}s')
        scan=exterior_real_poles(response,u,samples=64)
        current=dict(L=72,Nkz=384,q=SETTINGS['q'],Ua=8.,stage='streamed_exterior_continuation',
            density=density_value,raw_band_pair_slots=response.original_count,active_transitions=len(response.delta),
            exactly_zero_weights_removed=response.zero_weight_count,vertex_error=vertex_error,
            transition_factorization_error=response.transition_factorization_error,
            gram_factorization_error=response.gram_factorization_error,
            moment0_max=float(np.max(np.abs(response.moment0))),**scan)
        report['calculation']=current
        write_json(output/'exterior_search.json',current)
        coarse=old['calculation']
        comparison=dict(coarse_L=72,fine_L=72,coarse_Nkz=192,fine_Nkz=384,
            pole_relative_tolerance=1e-5,residue_relative_tolerance=.01,passes_declared_tolerances=False)
        good=lambda r:r.get('residue_status')=='isolated_nullspace_checked' and not r.get('nearby_root_cluster',True)
        if len(coarse['roots'])==len(current['roots'])==1 and good(coarse['roots'][0]) and good(current['roots'][0]):
            a,b=coarse['roots'][0],current['roots'][0]
            ra,rb=decode_matrix(a['residue_matrix']),np.asarray(b['residue_matrix'])
            comparison.update(pole_relative_error=abs(a['pole']-b['pole'])/abs(b['pole']),
                charge_residue_relative_error=float(abs(ra[0,0]-rb[0,0])/abs(rb[0,0])),
                axial_residue_relative_error=float(abs(ra[1,1]-rb[1,1])/abs(rb[1,1])))
            comparison['passes_declared_tolerances']=bool(comparison['pole_relative_error']<=1e-5
                and max(comparison['charge_residue_relative_error'],comparison['axial_residue_relative_error'])<=.01)
            report['matched_root']=dict(pole=b['pole'],charge_residue=rb[0,0],axial_residue=rb[1,1],
                residue_matrix=rb,nearest_bare_distance=b['nearest_bare_distance'],
                nullspace_eigenvalue_residual=b['nullspace_eigenvalue_residual'])
        else:comparison['unresolved_reason']='Not one isolated matched exterior root'
        report['comparison']=comparison
        report['transverse_comparison']=old['comparison']
        report['passes_declared_tolerances']=comparison['passes_declared_tolerances']
        report['both_directions_converged']=bool(comparison['passes_declared_tolerances'] and old['passes_declared_tolerances'])
        write_csv(output/'comparison.csv',[old['comparison'],comparison])
        if vertex_error>=2e-11 or response.original_count!=raw:
            raise RuntimeError('Vertex or raw-transition retention check failed')
        report['status']=('EXTERIOR_BOTH_DIRECTIONS_CONVERGED' if report['both_directions_converged']
                          else 'STREAMED_EXTERIOR_COMPLETED_UNCONVERGED')
        log('Comparison: '+json.dumps(comparison))
    except Exception as exc:
        report['status']='STREAMED_EXTERIOR_FAILED'
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
        log(traceback.format_exc())
    report['sources_unchanged']=hashes=={path.name:sha(path) for path in sources}
    report['inputs_unchanged']=input_hashes=={str(path.relative_to(ROOT)):sha(path) for path in inputs}
    if not report['sources_unchanged'] or not report['inputs_unchanged']:report['status']='SOURCE_OR_INPUT_CHANGED'
    report['elapsed_seconds']=time.perf_counter()-started
    report['completed_utc']=datetime.now(timezone.utc).isoformat()
    log('Final status: '+report['status'])
    report['output_sha256']={path.name:sha(path) for path in output.iterdir() if path.is_file() and path.name!='diagnostic.json'}
    checkpoint()
    if report['status'] in ('STREAMED_EXTERIOR_FAILED','SOURCE_OR_INPUT_CHANGED'):raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.output.resolve())