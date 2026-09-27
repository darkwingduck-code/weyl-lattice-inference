"""First-order explanation of mesh-induced g bias against a common reference.

No diagonalization. This is a deterministic local projection, not an error
bound, covariance, confidence interval, or independent data-generation check.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
SCIENCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCIENCE))
import detector_boundary_refinement as experiment
from parameter_calibration_independent_review import Snapshot,load_windows,models_for,truths_for,cofactor_response
from extend_transverse import csv_write,sha
from run_pole_zero_experiment import write_json


def linear_response_shift(pi,delta,u):
    return np.linalg.solve(np.eye(2)-pi@u,delta)@np.linalg.inv(np.eye(2)-u@pi)


def real_charge_vector(values,truth,names):
    z=np.concatenate([values[n][:,0,0]/np.linalg.norm(truth[n][:,0,0]) for n in names])/np.sqrt(len(names))
    return np.r_[z.real,z.imag]


def projection(data,reference,names):
    truth=truths_for(reference);model=truths_for(data)
    error={n:model[n]-truth[n] for n in names}
    derivative={n:model[n]@np.diag([1/(2*np.sin(data[n]['q']/2))**2,0.])@model[n] for n in names}
    r0=real_charge_vector(error,truth,names);s=real_charge_vector(derivative,truth,names)
    ds=float(s@s)
    if ds<=0:raise ValueError('Zero Coulomb derivative norm')
    dg=-float(s@r0)/ds
    projected=r0+dg*s
    prediction=models_for(data,names,.8+dg,8.)
    exact_at_pred=real_charge_vector({n:prediction[n]-truth[n] for n in names},truth,names)
    approx={}
    for n in names:
        q=data[n]['q'];u=np.diag([.8/(2*np.sin(q/2))**2,8.])
        approx[n]=linear_response_shift(reference[n]['pi'],data[n]['pi']-reference[n]['pi'],u)
    predicted_mesh=real_charge_vector(approx,truth,names)
    return dict(unfitted_shared_reference_rms=float(np.linalg.norm(r0)),
        coulomb_derivative_norm=float(np.linalg.norm(s)),predicted_delta_g=dg,predicted_g=.8+dg,
        predicted_g_within_declared_bounds=.2<=.8+dg<=1.6,
        linear_projected_rms=float(np.linalg.norm(projected)),
        exact_rms_at_linearly_predicted_g=float(np.linalg.norm(exact_at_pred)),
        first_order_mesh_response_absolute_error=float(np.linalg.norm(predicted_mesh-r0)),
        first_order_mesh_response_relative_error=(float(np.linalg.norm(predicted_mesh-r0)/np.linalg.norm(r0)) if np.linalg.norm(r0)>0 else None))


def controls():
    pi=np.array([[[.12-.04j,.02+.01j],[.03-.02j,-.47-.03j]],[[.21-.06j,.01-.03j],[.02+.04j,-.36-.02j]]])
    perturb=np.array([[[.01+.02j,.003-.004j],[.006+.008j,-.02+.01j]],[[.003-.007j,.009+.002j],[-.004+.001j,.008-.005j]]])
    q=.3;u=np.diag([.8/(2*np.sin(q/2))**2,8.]);eps=1e-5
    exact=cofactor_response(pi+eps*perturb,q,.8,8.)-cofactor_response(pi,q,.8,8.)
    first=linear_response_shift(pi,eps*perturb,u)
    error=float(np.max(abs(exact-first)))
    delta=eps*perturb
    left=np.linalg.inv(np.eye(2)-pi@u)
    right=np.linalg.inv(np.eye(2)-u@pi)
    left_shifted=np.linalg.inv(np.eye(2)-(pi+delta)@u)
    remainder=left@delta@u@left_shifted@delta@right
    remainder_error=float(np.max(abs(exact-first-remainder)))
    if remainder_error>2e-14:raise AssertionError('Exact second-order resolvent identity failed')
    diagonal=np.zeros_like(pi);diagonal[:,0,0]=pi[:,0,0];diagonal[:,1,1]=pi[:,1,1]
    delta_v=.005;coarse=diagonal.copy();coarse[:,0,0]/=1-delta_v*diagonal[:,0,0]
    reference={'a':dict(pi=diagonal,q=q)};data={'a':dict(pi=coarse,q=q)}
    fit=experiment.profile(data,truths_for(reference),('a',),8.)
    expected=.8-(2*np.sin(q/2))**2*delta_v
    if abs(fit['g']-expected)>2e-8 or fit['normalized_mismatch']>2e-8:
        raise AssertionError('Exactly confounded scalar input control failed')
    return dict(first_order_response_remainder=error,exact_second_order_identity_error=remainder_error,
        exactly_confounded_scalar_expected_g=expected,scalar_fitted_g=fit['g'],scalar_fit_rms=fit['normalized_mismatch'],
        interpretation='A scalar input perturbation can be exactly absorbed by biased g; this fixture is not the four-band production result.')


def run(folder,profile_output,output):
    output=experiment.new_output(output);snapshot=Snapshot();snapshot.read(Path(__file__))
    for name in experiment.SOURCE_NAMES:snapshot.read(SCIENCE/name)
    _,windows,_=load_windows(experiment.INDEX,snapshot);experiment.add_higher(windows,folder,snapshot)
    rows=list(csv.DictReader(snapshot.read(profile_output/'shared_higher_truth_profiles.csv').splitlines()))
    results=[]
    for t in experiment.TARGETS:
        for level in ('coarse','fine','higher'):
            record=projection(windows[t['window']][level],windows[t['window']]['higher'],t['names'])
            actual=[r for r in rows if r['target']==t['id'] and r['model_level']==level and float(r['ua'])==8.]
            if len(actual)!=1:raise ValueError('Missing shared-reference truth control')
            actual=actual[0];g=float(actual['g']);rms=float(actual['normalized_mismatch'])
            results.append(dict(target=t['id'],window=t['window'],model_level=level,**record,
                nonlinear_profiled_g=g,nonlinear_profiled_rms=rms,
                predicted_minus_nonlinear_g=record['predicted_g']-g,
                predicted_linear_rms_minus_nonlinear_rms=record['linear_projected_rms']-rms,
                rigorous_error_bound=False,statistical_interpretation=False))
    test=controls();snapshot.check();csv_write(output/'linear_g_bias_comparison.csv',results)
    report=dict(status='LINEAR_MESH_BIAS_COMPARISON_COMPLETE',row_count=len(results),controls=test,
        source_input_sha256=snapshot.hashes,inputs_sources_unchanged=True,
        equation='delta chi = (I-Pi U)^-1 delta Pi (I-U Pi)^-1 + O(delta Pi^2)',
        projection='delta g = -s dot r0/(s dot s), using real/imag stacked charge vectors with common higher truth normalization',
        limitation='Local first-order explanation; not a mesh error bound or statistical calibration model',
        output_sha256={'linear_g_bias_comparison.csv':sha(output/'linear_g_bias_comparison.csv')})
    write_json(output/'diagnostic.json',report)
    print(json.dumps({'status':report['status'],'controls':test,'rows':results},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--higher-root',type=Path);ap.add_argument('--profile-output',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.self_test:print(json.dumps(controls(),indent=2))
    else:
        if any(x is None for x in (args.higher_root,args.profile_output,args.output)):ap.error('All three paths required')
        run(args.higher_root.resolve(),args.profile_output.resolve(),args.output)