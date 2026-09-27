"""Read-only independent alignment and 2x2 response audit of boundary refinement.

No production modules are imported. Only this review directory receives output.
No spectral sums, grid-error bounds or global optimization certificates are made.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, io, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DATA=STUDY/'detector_boundary_refinement'
INDEX=STUDY/'parameter_sector_sampling/calibration_inputs.json'
ATOL=2e-14
EXPECTED={
 'B0':dict(Lx=96,Ly=96,Nkz=768,numerator=0,denominator=1,cases=('qhalf','baseline','qdouble')),
 'Bplus':dict(Lx=72,Ly=13,Nkz=384,numerator=1,denominator=36,cases=('eta_low',))}
TARGETS={
 'Bplus_eta_low_Ua1':('fixed_lab',('Bplus_eta_low',),1.,.01),
 'B0_qscan_Ua20':('fixed_lab',('B0_qhalf','B0_baseline','B0_qdouble'),20.,.001),
 'B0_qdouble_scaled_Ua17':('scaled_q',('B0_qdouble',),17.,.001)}

class Evidence:
 def __init__(self):self.hashes={}
 def read(self,path,expected=None,binary=False):
  path=Path(path).resolve();b=path.read_bytes();h=hashlib.sha256(b).hexdigest()
  if expected is not None and h!=expected:raise ValueError('Hash mismatch '+str(path))
  if str(path) in self.hashes and self.hashes[str(path)]!=h:raise ValueError('Changed input '+str(path))
  self.hashes[str(path)]=h;return b if binary else b.decode('utf-8-sig')
 def check(self):
  for p,h in self.hashes.items():
   if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed input '+p)

def close(a,b,label):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a,b,rtol=0,atol=ATOL):
  raise ValueError('Alignment mismatch: '+label)

def metadata_alignment(meta,item,report):
 field=item['field'];case=item['case'];spec=EXPECTED[field]
 if meta['field']!=field or meta['case']!=case or report['field']!=field:raise ValueError('Field/case identity mismatch')
 for key in ('q','mu','T','eta'):close(meta[key],item[key],key)
 close(meta['q'],np.pi/12*({'qhalf':.5,'qdouble':2}.get(case,1.)),'declared physical q')
 close(meta['mu'],.5,'fixed mu');close(meta['T'],.04,'temperature')
 close(meta['eta'],.0075 if case=='eta_low' else .015,'declared eta')
 p=meta['parameters']
 for key in ('Lx','Ly','Nkz'):
  if p[key]!=spec[key] or report['parameters'][key]!=spec[key]:raise ValueError('Mesh mismatch')
 for key,value in dict(mass=0.,wilson=1.,hopping=1.,exchange=1.,theta_x=0.,theta_y=0.,theta_z=0.).items():
  close(p[key],value,key);close(report['parameters'][key],value,key)
 close(report['physical_B'],2*np.pi*spec['numerator']/spec['denominator'],'magnetic field')
 if report['numerator']!=spec['numerator'] or report['denominator']!=spec['denominator']:raise ValueError('Flux mismatch')
 close(2*np.pi*meta['qshift']/p['Nkz'],meta['q'],'qshift')
 for window in ('fixed_lab','scaled_q'):
  values=meta[window+'_frequencies']
  if len(values)!=45 or len(set(values))!=45:raise ValueError('45 unique frequencies required')
  close(values,item[window+'_frequencies'],window+' frequency alignment')
 if meta['complete_torus'] is not True or meta['checks']['identity_checks_pass'] is not True:raise ValueError('Incomplete case')
 blocks=spec['Lx']//spec['denominator']*spec['Ly']
 pairs=blocks*spec['Nkz']*(4*spec['denominator'])**2
 c=meta['checks']
 if c['raw_pairs']!=pairs or c['expected_raw_pairs']!=pairs or c['retains_all_raw_pairs'] is not True:raise ValueError('Incomplete pair inventory')
 if report['blocks_completed']!=blocks or report['expected_total_blocks']!=blocks or report['complete_torus'] is not True:raise ValueError('Incomplete blocks')
 for key,tol in [('vertex_error',2e-11),('full_moment0_max',2e-10),('total_charge_ward_bare',2e-10),('total_charge_ward_rpa',2e-10),('dynamic_matching_identity_max_error',2e-12)]:
  val=float(c[key])
  if not np.isfinite(val) or abs(val)>=tol:raise ValueError('Declared identity value failed '+key)
 val=c['minimum_total_rpa_weighted_loss_eigenvalue']
 if not np.isfinite(val) or val < -2e-10:raise ValueError('Declared weighted passivity failed')
 return blocks,pairs

def matrix(rows,prefix):
 a=np.array([[[complex(float(r[f'{prefix}_{i}{j}_re']),float(r[f'{prefix}_{i}{j}_im'])) for j in range(2)] for i in range(2)] for r in rows])
 if not np.all(np.isfinite(a)):raise ValueError('Nonfinite matrix')
 return a

def response(pi,q,g,ua):
 # Explicit adjugate of I-Pi U, followed by multiplication by Pi.
 u=np.array([g/(2*np.sin(q/2))**2,ua]);k=np.eye(2)-pi*u
 det=k[:,0,0]*k[:,1,1]-k[:,0,1]*k[:,1,0]
 if not np.all(np.isfinite(det)) or np.any(det==0):raise ValueError('Invalid dielectric denominator')
 adj=np.empty_like(k);adj[:,0,0]=k[:,1,1];adj[:,1,1]=k[:,0,0]
 adj[:,0,1]=-k[:,0,1];adj[:,1,0]=-k[:,1,0]
 value=(adj@pi)/det[:,None,None]
 if not np.all(np.isfinite(value)):raise ValueError('Nonfinite response')
 return value

def rows_csv(text):return list(csv.DictReader(io.StringIO(text)))

def select(rows,omega):
 actual=np.array([float(r['omega']) for r in rows])
 if not np.all(np.isfinite(actual)) or len(set(actual))!=len(actual):raise ValueError('Invalid CSV frequencies')
 ids=[int(np.argmin(abs(actual-w))) for w in omega]
 close(actual[ids],omega,'CSV frequency selection')
 if len(set(ids))!=45:raise ValueError('Repeated CSV rows')
 return [rows[i] for i in ids]

def verify(fields,evidence):
 index=json.loads(evidence.read(INDEX,'fdfd29e3e83a7adf3329d22df0988d43d81ad39c8a018ff002b962ecaff274db'))
 items={i['id']:i for i in index['datasets']};data={};checks=[]
 for field in fields:
  folder=DATA/(field+'_higher');report=json.loads(evidence.read(folder/'diagnostic.json'))
  if report['status']!='DETECTOR_BOUNDARY_SAMPLING_COMPLETE' or report.get('sources_unchanged') is not True or report.get('inputs_unchanged') is not True:raise ValueError('Incomplete or changed production '+field)
  if set(report['cases'])!=set(EXPECTED[field]['cases']):raise ValueError('Case inventory mismatch')
  for p,h in {**report['source_sha256'],**report['input_sha256']}.items():evidence.read(p,h)
  for p,h in report['output_sha256'].items():evidence.read(folder/p,h,binary=True)
  for case,meta in report['cases'].items():
   item=items[field+'_'+case];blocks,pairs=metadata_alignment(meta,item,report)
   separate=json.loads(evidence.read(folder/(case+'_metadata.json')))
   if separate!=meta:raise ValueError('Separate case metadata differs')
   rows=rows_csv(evidence.read(folder/meta['csv']))
   if not rows or not all(np.isfinite(float(v)) for row in rows for v in row.values()):raise ValueError('Nonfinite CSV scalar')
   constants={k:meta[k] for k in ('q','mu','T','eta','density')}
   constants.update({k:meta['parameters'][k] for k in ('Lx','Ly','Nkz')});constants['B']=report['physical_B']
   for k,v in constants.items():close([float(r[k]) for r in rows],np.full(len(rows),v),'CSV '+k)
   for level,rel,h in [('fine',item['csv'],item['csv_sha256']),('coarse',item['convergence']['coarse_csv'],item['convergence']['coarse_sha256']),('higher',None,None)]:
    rr=rows if level=='higher' else rows_csv(evidence.read(INDEX.parent/rel,h))
    # Each grid's density is allowed to differ: this is the same fixed-mu family.
    for k in ('q','mu','T','eta','B'):close([float(r[k]) for r in rr],np.full(len(rr),constants[k]),level+' physical '+k)
    for window in ('fixed_lab','scaled_q'):
     chosen=select(rr,item[window+'_frequencies']);pi=matrix(chosen,'bare');saved=matrix(chosen,'rpa')
     value=response(pi,item['q'],.8,8.);error=float(np.max(abs(value-saved)))
     if error>=2e-12:raise ValueError('Adjugate/direct stored RPA mismatch')
     data[(window,level,item['id'])]=(pi,item['q'])
     checks.append(dict(dataset=item['id'],window=window,level=level,points=45,stored_rpa_error=error,blocks_higher=blocks,raw_pairs_higher=pairs))
 return data,checks

def profile_review(path,data,evidence):
 diag=json.loads(evidence.read(path/'diagnostic.json'))
 if diag['status']!='TARGETED_PROFILE_REFINEMENT_COMPLETE':raise ValueError('Incomplete profiles')
 for p,h in diag['input_source_sha256'].items():evidence.read(p,h)
 for p,h in diag['output_sha256'].items():evidence.read(path/p,h,binary=True)
 out=[]
 for filename,kind in [('own_mesh_truth_profiles.csv','own_mesh'),('shared_higher_truth_profiles.csv','shared_higher')]:
  for row in rows_csv(evidence.read(path/filename)):
   window,names,original_ua,tol=TARGETS[row['target']]
   if row['window']!=window or row['truth_reference']!=kind or row['mode']!='charge_only':raise ValueError('Profile definition mismatch')
   g=float(row['g']);ua=float(row['ua']);level=row['model_level'];truth_level=level if kind=='own_mesh' else 'higher'
   squares=[]
   for name in names:
    pi,q=data[(window,level,name)];pt,qt=data[(window,truth_level,name)]
    x=response(pi,q,g,ua)[:,0,0];y=response(pt,qt,.8,8.)[:,0,0]
    squares.append(float(np.vdot(x-y,x-y).real/np.vdot(y,y).real))
   rms=float(np.sqrt(np.mean(squares)));saved=float(row['normalized_mismatch'])
   if abs(rms-saved)>2e-11:raise ValueError('RMS mismatch')
   close(float(row['threshold']),tol,'profile threshold')
   if (row['within_declared_threshold']=='True')!=(rms<=tol):raise ValueError('Classification mismatch')
   if abs(float(row['threshold_minus_rms'])-(tol-rms))>2e-11:raise ValueError('Margin sign mismatch')
   out.append(dict(target=row['target'],window=window,truth_reference=kind,model_level=level,ua=ua,g=g,rms=rms,saved_rms=saved,absolute_rms_error=abs(rms-saved),threshold=tol,margin=tol-rms,within_threshold=rms<=tol))
 if len(out)!=72 or len({(r['target'],r['truth_reference'],r['model_level'],r['ua']) for r in out})!=72:raise ValueError('Expected 72 unique profiles')
 return out

def controls():
 # Independent 2x2 complex response and normalization checks, no model modules.
 p=np.array([[[.1-.03j,.02+.01j],[.03+.04j,-.4-.01j]]]);q=.3;u=np.diag([.8/(2*np.sin(q/2))**2,8.])
 got=response(p,q,.8,8.);direct=np.linalg.solve(np.eye(2)-p[0]@u,p[0])
 if np.max(abs(got[0]-direct))>2e-14:raise AssertionError('Response control')
 wrong=p.copy();wrong[:,1,0]=wrong[:,0,1].conj()
 if np.max(abs(response(wrong,q,.8,8.)-got))<1e-4:raise AssertionError('Phase negative control too weak')
 rejected=[]
 for label,a,b in [('q',.4,.3),('window',[.1,.2],[.1,.3]),('nan',[.1,np.nan],[.1,.2]),('B',.1745,0.)]:
  try:close(a,b,label)
  except ValueError:rejected.append(label)
  else:raise AssertionError('Alignment control accepted '+label)
 # Equal dataset weighting must not turn into pooled intensity weighting.
 per=np.sqrt(np.mean([1.,0.]));pooled=np.sqrt(1/(1+100**2))
 if abs(per-pooled)<.5:raise AssertionError('Norm control too weak')
 return dict(complex_adjugate_vs_solve=True,offdiagonal_conjugation_rejected=True,alignment_rejections=rejected,per_dataset_vs_pooled_separation=float(per-pooled))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--field',choices=('B0','Bplus'));ap.add_argument('--profiles',type=Path);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();out=args.output.resolve()
 if not out.is_relative_to(HERE):raise ValueError('Review output must remain under review directory')
 if out.exists():raise FileExistsError('New review output required')
 out.mkdir(parents=True);ev=Evidence();ev.read(__file__)
 result={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING','science_recomputed':False,'global_g_optimum_certified':False,'physical_convergence_certified':False}
 try:
  result['small_controls']=controls();data,checks=verify((args.field,) if args.field else tuple(EXPECTED),ev)
  result['alignment_rows']=checks
  if args.profiles:
   rows=profile_review(args.profiles.resolve(),data,ev)
   with (out/'profile_cofactor_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   result['profiles']=len(rows);result['maximum_rms_difference']=max(r['absolute_rms_error'] for r in rows)
  ev.check();result['status']='PASS';result['inputs_unchanged']=True
 except Exception as exc:
  result['status']='FAIL';result['failure']=dict(type=type(exc).__name__,message=str(exc));raise
 finally:
  result['input_sha256']=ev.hashes;result['completed_utc']=datetime.now(timezone.utc).isoformat()
  (out/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
  print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','alignment_rows')},indent=2))

if __name__=='__main__':main()