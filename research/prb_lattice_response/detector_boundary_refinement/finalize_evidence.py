"""Summarize preserved boundary-refinement evidence after independent review."""
import argparse,csv,json,sys
from pathlib import Path
from datetime import datetime,timezone
SCIENCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCIENCE))
import parameter_sector_sampling as sampler
from extend_transverse import sha,csv_write
from run_pole_zero_experiment import write_json
ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def table(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def check_hashes(mapping):
 for path,expected in mapping.items():
  if sha(Path(path))!=expected:raise ValueError('Hash mismatch: '+path)

def run(review_path):
 out=ROOT/'delivery'
 if out.exists():raise FileExistsError('New delivery output required')
 review=read(review_path)
 if review['status']!='PASS' or review.get('profiles')!=72 or not review['inputs_unchanged']:
  raise ValueError('Need completed independent 72-profile review')
 ids={r['dataset'] for r in review['alignment_rows']}
 if ids!={'B0_baseline','B0_qhalf','B0_qdouble','Bplus_eta_low'} or len(review['alignment_rows'])!=24:
  raise ValueError('Need all four datasets, two windows and three levels aligned')
 check_hashes(review['input_sha256'])
 profiles=ROOT/'higher_profile_comparison';profile=read(profiles/'diagnostic.json')
 if profile['status']!='TARGETED_PROFILE_REFINEMENT_COMPLETE' or profile['shared_higher_profile_count']!=36:
  raise ValueError('Incomplete common-reference comparison')
 linear=read(ROOT/'linear_bias_comparison/diagnostic.json')
 if linear['status']!='LINEAR_MESH_BIAS_COMPARISON_COMPLETE':raise ValueError('Incomplete bias comparison')
 check_hashes(linear['source_input_sha256'])
 small=read(ROOT/'small_validation.json')
 if small['status']!='PASS':raise ValueError('Small checks not passed')
 if small['source_sha256']!=sha(SCIENCE/'detector_boundary_refinement.py'):raise ValueError('Tested source differs')
 index=read(SCIENCE/'parameter_sector_sampling/calibration_inputs.json')
 rows=[];samples={}
 for field in ('B0','Bplus'):
  samples[field]=read(ROOT/(field+'_higher')/'diagnostic.json')
  if samples[field]['status']!='DETECTOR_BOUNDARY_SAMPLING_COMPLETE':raise ValueError('Incomplete sample')
  for case in samples[field]['cases']:
   item=next(r for r in index['datasets'] if r['field']==field and r['case']==case)
   coarse=SCIENCE/'parameter_sector_sampling'/item['convergence']['coarse_csv']
   fine=SCIENCE/'parameter_sector_sampling'/item['csv']
   higher=ROOT/(field+'_higher')/(case+'.csv')
   for a,b,al,bl in ((coarse,fine,'coarse','fine'),(fine,higher,'fine','higher')):
    comparison=sampler.compare_outputs(a,b)
    rows.append(dict(dataset=item['id'],coarse_level=al,fine_level=bl,
       original_grid_status=item['convergence_status'],coarse_sha256=sha(a),fine_sha256=sha(b),
       **comparison['scaled_errors'],tolerance=comparison['tolerance'],matches_historical_scaled_criterion=comparison['pass_all'],
       detector_relative_tolerance_certified=False,comparison_definition=comparison['definition']))
 out.mkdir()
 csv_write(out/'five_kernel_comparisons.csv',rows)
 own=table(profiles/'own_mesh_truth_profiles.csv');shared=table(profiles/'shared_higher_truth_profiles.csv')
 changes=table(profiles/'profile_grid_changes.csv')
 original_ua={'Bplus_eta_low_Ua1':1.,'B0_qscan_Ua20':20.,'B0_qdouble_scaled_Ua17':17.}
 selected=lambda data:[r for r in data if float(r['ua'])==original_ua[r['target']]]
 result=dict(status='BOUNDARY_REFINEMENT_EVIDENCE_COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),
  source_sha256=sha(SCIENCE/'detector_boundary_refinement.py'),independent_review=str(review_path.resolve()),
  independent_review_sha256=sha(review_path),independent_profile_count=72,
  maximum_independent_rms_error=review['maximum_rms_difference'],
  sample_elapsed_seconds={f:r['elapsed_seconds'] for f,r in samples.items()},
  original_hypothesis_own_truth=selected(own),original_hypothesis_shared_higher_truth=selected(shared),
  original_hypothesis_grid_changes=selected(changes),
  shared_reference_Ua8_controls=[r for r in shared if float(r['ua'])==8.],
  detector_truth_grid_differences=table(profiles/'detector_truth_grid_differences.csv'),
  first_order_bias_comparison=table(ROOT/'linear_bias_comparison/linear_g_bias_comparison.csv'),
  historical_scaled_grid_comparisons=rows,
  scope=dict(new_collective_gap=False,magnetic_transverse_convergence=False,rigorous_error_bound=False,
             global_g_optimum=False,statistical_confidence_intervals=False,original_evidence_overwritten=False))
 write_json(out/'summary.json',result)
 files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
 manifest=dict(schema_version=1,root=str(ROOT),files_sha256=files,
  external_source_sha256={str(p):sha(p) for p in (SCIENCE/'detector_boundary_refinement.py',SCIENCE/'test_detector_boundary_refinement.py',review_path)},
  note='Includes preserved failures, partial pilots and completed results. File presence is not a scientific PASS.')
 write_json(out/'manifest.json',manifest)
 print(json.dumps(dict(status=result['status'],file_count=len(files),summary_sha256=sha(out/'summary.json'),manifest_sha256=sha(out/'manifest.json')),indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review-report',type=Path,required=True)
 run(ap.parse_args().review_report)