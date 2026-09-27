"""Verify restored numerical records, including declared failures; no fits run."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(root, path):
    return json.loads((root/path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',choices=['epjp','prb'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    checks=[]
    if args.paper=='epjp':
        base='pole_zero_work/data_prl/full_band_reproduction/'
        old=read(root,base+'historical_comparison/reproduction_metadata.json')
        new=read(root,base+'reproduction_metadata.json')
        assert old['status']=='FAIL' and new['status']=='PASS'
        assert not new['convergence_reconstruction']['archived_aggregator_present']
        checks.append('Historical FAIL and current repeat PASS retained; aggregation remains a reconstruction')
        for summary in [old,new]:
            for name,digest in summary['source_sha256'].items():
                assert sha(root/'pole_zero_work'/name)==digest,name
        checks.append('Both archived full-band reports reference the unchanged production code')
        for name,digest in old['reference_sha256'].items():
            assert sha(root/base/'archived_reference'/name)==digest,name
        for name,digest in new['reference_sha256'].items():
            assert sha(root/'pole_zero_work/data_prl/full_bz_completion'/name)==digest,name
        checks.append('Historical reference and current active tables match their separate frozen hashes')
        folder=root/'research/prl_physical_discrimination_20260912'
        result=json.loads((folder/'production_v2/diagnostic.json').read_text())
        assert result['rows']==11280 and result['conditional_boxes']==11280
        for name,digest in result['file_sha256'].items():
            assert sha(folder/'production_v2'/name)==digest,name
        with (folder/'production_v2/trials.csv').open(newline='') as handle:
            assert sum(1 for _ in csv.DictReader(handle))==11280
        controls=json.loads((folder/'controls/diagnostic.json').read_text())
        assert controls['weyl_rows']==594 and controls['structural_rows']==18
        checks.append('11280 conditional oscillator boxes and separate 594 Weyl controls retained')
    else:
        base=root/'research/prb_lattice_response'
        report=json.loads((base/'zero_full_input_followup/delivery_review/summary.json').read_text())
        assert report['new_input_verified_local_zeros']==751
        assert report['original_fixed_float_verified_count']==1901
        assert report['increment_to_original_fixed_float_count']==0
        assert not report['original_751_one_to_one_correspondence_certified']
        for old,digest in report['source_input_sha256'].items():
            relative=old.replace('\\','/').split('/weyl-collective-modes/',1)[1]
            assert sha(root/relative)==digest,relative
        checks.append('20 full-input code/data/log hashes and new-versus-old zero lineage preserved')
        result=json.loads((base/'exterior_kz_retry/diagnostic.json').read_text())
        assert result['comparison']['passes_declared_tolerances']
        assert result['calculation']['near_edge_unresolved']
        assert not result['low_energy_hybrid_gap_certified']
        assert (base/'exterior_kz_streamed/execution.log').exists()
        assert (base/'zero_full_input_followup/preflight_01/failure.txt').exists()
        checks.append('Exterior retry and preserved streaming/preflight failures; no gap certificate')
    print(json.dumps({'status':'PASS','paper':args.paper,'checks':checks,'no_scientific_recomputation':True},indent=2))


if __name__=='__main__': main()
