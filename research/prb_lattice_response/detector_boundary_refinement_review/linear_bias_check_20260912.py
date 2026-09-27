"""Independent nine-row real-g projection check; no production-module imports.

Uses the already independent adjugate reader, complex-vector projection,
charge cofactor derivative, and a five-point finite-difference cross-check.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import independent_gate as gate

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'detector_boundary_refinement'
ALGEBRA_TOL=2e-11
FD_TOL=5e-8

def vector(values,truth,names):
 return np.concatenate([values[n]/np.linalg.norm(truth[n]) for n in names])/np.sqrt(len(names))

def check():
 out=HERE/'linear_bias_check_20260912.json'
 if out.exists():raise FileExistsError('Preserve existing independent check')
 ev=gate.Evidence();ev.read(__file__);ev.read(gate.__file__)
 data,alignment=gate.verify(tuple(gate.EXPECTED),ev)
 original=json.loads(ev.read(BASE/'linear_bias_comparison/diagnostic.json'))
 if original['status']!='LINEAR_MESH_BIAS_COMPARISON_COMPLETE':raise ValueError('Incomplete source analysis')
 for path,h in original['source_input_sha256'].items():ev.read(path,h)
 saved=gate.rows_csv(ev.read(BASE/'linear_bias_comparison/linear_g_bias_comparison.csv',original['output_sha256']['linear_g_bias_comparison.csv']))
 expected={(name,level) for name in gate.TARGETS for level in ('coarse','fine','higher')}
 if len(saved)!=9 or {(r['target'],r['model_level']) for r in saved}!=expected:raise ValueError('Expected exact nine-row inventory')
 rows=[]
 for row in saved:
  window,names,_,_=gate.TARGETS[row['target']];level=row['model_level']
  truth={};model={};derivative={};first_mesh={};finite={h:{} for h in (1e-5,5e-6)}
  for name in names:
   p,q=data[(window,level,name)];ph,qh=data[(window,'higher',name)]
   chi=gate.response(p,q,.8,8.);chi_h=gate.response(ph,qh,.8,8.)
   truth[name]=chi_h[:,0,0];model[name]=chi[:,0,0]
   Q=2*np.sin(q/2)
   # Different route from the producer's full chi @ diag @ chi product.
   derivative[name]=chi[:,0,0]**2/Q**2
   u=np.diag([.8/Q**2,8.]);delta=p-ph
   # L=I+chi U, R=I+U chi: no production inverse-response routine used.
   first_mesh[name]=((np.eye(2)+chi_h@u)@delta@(np.eye(2)+u@chi_h))[:,0,0]
   for h in finite:
    fm2=gate.response(p,q,.8-2*h,8.)[:,0,0];fm1=gate.response(p,q,.8-h,8.)[:,0,0]
    fp1=gate.response(p,q,.8+h,8.)[:,0,0];fp2=gate.response(p,q,.8+2*h,8.)[:,0,0]
    finite[h][name]=(fm2-8*fm1+8*fp1-fp2)/(12*h)
  e=vector({n:model[n]-truth[n] for n in names},truth,names)
  j=vector(derivative,truth,names);jj=float(np.vdot(j,j).real)
  if jj<=0 or not np.isfinite(jj):raise ValueError('Invalid derivative norm')
  dg=-float(np.vdot(j,e).real)/jj;rp=e+dg*j;predg=.8+dg
  residual_at_pred={}
  for name in names:
   p,q=data[(window,level,name)];residual_at_pred[name]=gate.response(p,q,predg,8.)[:,0,0]-truth[name]
  pred_exact=vector(residual_at_pred,truth,names)
  mesh_error=float(np.linalg.norm(vector(first_mesh,truth,names)-e));norme=float(np.linalg.norm(e))
  derived=dict(unfitted_shared_reference_rms=norme,coulomb_derivative_norm=float(np.sqrt(jj)),
    predicted_delta_g=dg,predicted_g=predg,linear_projected_rms=float(np.linalg.norm(rp)),
    exact_rms_at_linearly_predicted_g=float(np.linalg.norm(pred_exact)),
    first_order_mesh_response_absolute_error=mesh_error,
    first_order_mesh_response_relative_error=mesh_error/norme if norme else None)
  errors={}
  for key,val in derived.items():
   if val is None:
    if row[key]!='':raise ValueError('Undefined relative error must remain absent')
    continue
   errors[key]=abs(val-float(row[key]))
   if errors[key]>ALGEBRA_TOL:raise ValueError('Saved scalar disagreement '+key)
  orth=float(abs(np.vdot(j,rp).real));pyth=abs(float(np.vdot(e,e).real)-(dg*dg*jj+float(np.vdot(rp,rp).real)))
  if orth>2e-14 or pyth>2e-14:raise ValueError('Real projection identity failed')
  fd_errors={}
  for h,value in finite.items():
   jf=vector(value,truth,names)
   fd_errors[str(h)]=float(np.linalg.norm(jf-j)/np.linalg.norm(j))
   if fd_errors[str(h)]>FD_TOL:raise ValueError('Finite-difference derivative disagreement')
  rows.append(dict(target=row['target'],level=level,window=window,actual_ua=8.,**derived,
    scalar_absolute_errors=errors,real_orthogonality_error=orth,pythagorean_error=pyth,
    finite_difference_relative_j_errors=fd_errors))
 ev.check()
 result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),rows=rows,row_count=len(rows),
  maximum_scalar_absolute_error=max(v for r in rows for v in r['scalar_absolute_errors'].values()),
  maximum_finite_difference_relative_j_error=max(v for r in rows for v in r['finite_difference_relative_j_errors'].values()),
  maximum_real_orthogonality_error=max(r['real_orthogonality_error'] for r in rows),
  maximum_pythagorean_error=max(r['pythagorean_error'] for r in rows),
  algebra_tolerance=ALGEBRA_TOL,finite_difference_tolerance=FD_TOL,
  finite_difference_rule='Five-point central derivative with h=1e-5 and 5e-6 in real g',
  independent_routes=['2x2 adjugate RPA','complex vectors and Re(j_dagger e)','scalar charge derivative chi_cc^2/Q^2','L=I+chi U and R=I+U chi for mesh linearization'],
  scientific_production_imported=False,new_electronic_structure_computation=False,
  global_optimization_repeated=False,physical_error_bound_certified=False,
  original_diagnostic_sha256=ev.hashes[str((BASE/'linear_bias_comparison/diagnostic.json').resolve())],
  input_sha256=ev.hashes,inputs_unchanged=True)
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256')},indent=2))

if __name__=='__main__':check()