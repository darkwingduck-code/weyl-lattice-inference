"""Frozen-design validation: full profiles, common higher references, and noise."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.optimize import least_squares
from observation_design import *

def raw_fit(record,observed,scale,ua=None,degree=0,starts=None):
    """Full complex matrix LS with factorized axial source/readout transfer.
    Fits d(x),s(x) as complex polynomials. No elementwise independent gains.
    scale is fixed per raw entry/sample; zero/noise floors are never refitted.
    """
    x=(record['omega']-np.mean(record['omega']))/np.ptp(record['omega'])
    basis=np.column_stack([x**k for k in range(degree+1)])
    count=degree+1;free=ua is None;offset=2 if free else 1
    def unpack(v):
        g=v[0];u=v[1] if free else ua
        z=v[offset:].reshape(2,2,count)
        d=basis@(z[0,0]+1j*z[0,1]);s=basis@(z[1,0]+1j*z[1,1])
        return g,u,d,s
    def fun(v):
        g,u,d,s=unpack(v);m=transfer(response(record['pi'],record['q'],g,u),d,s)
        delta=(m-observed)/scale
        return np.concatenate([delta.real.ravel(),delta.imag.ravel()])/np.sqrt(observed.size)
    starts=starts or ([ (.6,4.),(.8,8.),(1.,16.) ] if free else [(.6,ua),(.8,ua),(1.,ua)])
    fits=[]
    for g,u in starts:
        m=response(record['pi'],record['q'],g,u)
        z=np.zeros((2,2,count))
        for k,(a,b) in enumerate([(1,0),(0,1)]):
            design=m[:,a,b,None]*basis
            gain=np.linalg.lstsq(design,observed[:,a,b],rcond=None)[0]
            z[k,0]=gain.real;z[k,1]=gain.imag
        v=np.r_[g,u,z.ravel()] if free else np.r_[g,z.ravel()]
        low=np.r_[.2,0.,np.full(z.size,-np.inf)] if free else np.r_[.2,np.full(z.size,-np.inf)]
        high=np.r_[1.6,24.,np.full(z.size,np.inf)] if free else np.r_[1.6,np.full(z.size,np.inf)]
        fit=least_squares(fun,v,bounds=(low,high),ftol=2e-11,xtol=2e-11,gtol=2e-11,max_nfev=350)
        fits.append(fit)
    fit=min(fits,key=lambda f:np.linalg.norm(f.fun));g,u,d,s=unpack(fit.x)
    return dict(g=float(g),ua=float(u),rms=float(np.linalg.norm(fit.fun)),success=bool(fit.success),nfev=int(fit.nfev),all_starts_success=all(f.success for f in fits),global_optimum_certified=False)

def fit_products(record,observed,normalizers,ua=None):
    n='d';ds={n:record};obs={n:observed};sc={n:normalizers}
    def fun(v):return residual(models(ds,[n],v[0],v[1] if ua is None else ua),obs,[n],'cross_product',normalizers=sc)
    fits=[]
    for g,u in [(.6,4.),(.8,8.),(1.,16.)]:
        start=[g,u] if ua is None else [g]
        fit=least_squares(fun,start,bounds=([.2,0.],[1.6,24.]) if ua is None else ([.2],[1.6]),xtol=1e-12,gtol=1e-12,ftol=1e-12,max_nfev=250)
        fits.append(fit)
    fit=min(fits,key=lambda f:np.linalg.norm(f.fun))
    return dict(g=float(fit.x[0]),ua=float(fit.x[1] if ua is None else ua),rms=float(np.linalg.norm(fit.fun)),success=bool(fit.success),all_starts_success=all(f.success for f in fits))

def noisy_transfer(record,g=.8,u=8.):
    m=response(record['pi'],record['q'],g,u)
    x=(record['omega']-np.mean(record['omega']))/np.ptp(record['omega'])
    d=.7*np.exp(.4j)*(1+.1*x);s=1.3*np.exp(-.25j)*(1+.1j*x)
    return m,transfer(m,d,s)

def main():
    out=ROOT/'validation';out.mkdir(exist_ok=False);start=time.perf_counter()
    source={str(p):sha(p) for p in ROOT.glob('*.py')};source[str(ROOT/'frozen_design.json')]=sha(ROOT/'frozen_design.json')
    data,snap,err=load_all();profile_rows=[];mesh_rows=[];held=[]
    def save(name,rows):table(out/name,rows)
    policies=['diag_separate_complex','diag_separate_tilt','closure','cross_product']
    for window,levels in data.items():
        for name in levels['fine']:
            if window=='scaled_q' and data[window]['fine'][name]['item']['case'] not in ('qhalf','qdouble'):continue
            ds=levels['fine'];truth=models(ds,[name],.8,8.)
            for u in np.arange(0.,24.01,.5):
                for policy in policies:
                    fit=profile(ds,truth,[name],u,policy)
                    profile_rows.append(dict(dataset=name,window=window,ua=float(u),policy=policy,**fit))
            print('full profiles',window,name,len(profile_rows),flush=True)
    save('full_profiles.csv',profile_rows)
    for window,levels in data.items():
        for name in levels['higher']:
            if window=='scaled_q' and name not in ('B0_qhalf','B0_qdouble'):continue
            truth=models(levels['higher'],[name],.8,8.)
            for level in ('coarse','fine','higher'):
                for u in (1.,4.,7.5,8.,8.5,12.,16.,20.,24.):
                    for policy in ('closure','cross_product'):
                        fit=profile(levels[level],truth,[name],u,policy)
                        finecheck=profile(levels[level],truth,[name],u,policy,ng=129)
                        mesh_rows.append(dict(dataset=name,window=window,level=level,ua=u,policy=policy,denser_g=finecheck['g'],denser_rms=finecheck['rms'],rms_grid_difference=abs(fit['rms']-finecheck['rms']),**fit))
    save('common_higher_profiles.csv',mesh_rows)
    cases=['Bplus_baseline','Bplus_qhalf','Bplus_qdouble','Bplus_mu_low','Bplus_mu_high','B0_baseline','B0_mu_low']
    ds=data['fixed_lab']['fine']
    for name in cases:
        for g0 in (.6,1.):
            for u0 in (4.,12.,16.):
                truth=models(ds,[name],g0,u0)
                for u in sorted(set([u0,u0/2,min(24.,u0*1.5)])):
                    for policy in ('diag_separate_tilt','closure','cross_product'):
                        fit=profile(ds,truth,[name],u,policy)
                        held.append(dict(dataset=name,g_truth=g0,ua_truth=u0,ua=u,policy=policy,**fit))
    save('heldout_parameters.csv',held)
    print('profiles and heldout complete',time.perf_counter()-start,flush=True)
    raw=[]
    for name in ['Bplus_baseline','B0_baseline','B0_mu_low','Bplus_qdouble']:
        record=ds[name];physical,y=noisy_transfer(record);sigma=.001*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
        for u in (4.,8.,16.):
            for degree in (0,1):
                fit=raw_fit(record,y,sigma,ua=u,degree=degree)
                raw.append(dict(dataset=name,ua_trial=u,degree=degree,sigma_complex_rms=sigma,normalization='L2(all raw complex residuals)/(sigma sqrt(4N)); common absolute floor',**fit))
    save('raw_matrix_hypotheses.csv',raw)
    bias=[]
    for window,levels in data.items():
        for name,h in levels['higher'].items():
            if window=='scaled_q' and name not in ('B0_qhalf','B0_qdouble'):continue
            physical,y=noisy_transfer(h);sigma=.001*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
            for level in ('coarse','fine','higher'):
                record=levels[level][name]
                fit=raw_fit(record,y,sigma,degree=1)
                bias.append(dict(window=window,dataset=name,model_level=level,reference_level='higher',sigma_complex_rms=sigma,g_bias=fit['g']-.8,ua_bias=fit['ua']-8.,**fit))
    save('common_higher_joint_bias.csv',bias)
    noise=[];rng=np.random.default_rng(20260912)
    configs=[('Bplus_baseline','fine','absolute',s) for s in (.0001,.001,.01)]
    configs +=[(n,'fine','absolute',.001) for n in ['B0_baseline','B0_mu_low','Bplus_qdouble','Bplus_mu_high','Bplus_eta_high']]
    configs +=[('Bplus_eta_low','higher','absolute',.001),('Bplus_baseline','fine','relative',.01)]
    for name,level,kind,s in configs:
        record=data['fixed_lab'][level][name];physical,y=noisy_transfer(record)
        if kind=='absolute':sigma=s*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
        else:sigma=s*np.sqrt(np.mean(abs(y)**2,axis=0))[None,:,:]
        scales=[np.linalg.norm(y[:,0,0]),np.linalg.norm(y[:,0,1]*y[:,1,0])]
        for rep in range(100):
            eps=(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))*sigma/np.sqrt(2)
            obs=y+eps;fit=fit_products(record,obs,scales)
            row=dict(dataset=name,level=level,noise_kind=kind,sigma_multiplier=s,replica=rep,method='cross_product',g_bias=fit['g']-.8,ua_bias=fit['ua']-8.,**fit)
            noise.append(row)
            # The primary operating regime and weak-signal counterexample have an independent raw matrix fit.
            if name in ('Bplus_baseline','B0_mu_low'):
                f=raw_fit(record,obs,sigma,degree=1)
                noise.append(dict(dataset=name,level=level,noise_kind=kind,sigma_multiplier=s,replica=rep,method='raw_matrix_affine',g_bias=f['g']-.8,ua_bias=f['ua']-8.,g=f['g'],ua=f['ua'],rms=f['rms'],success=f['success'],all_starts_success=f['all_starts_success']))
        save('noise_replicas.csv',noise)
        print('noise',name,kind,s,'rows',len(noise),'elapsed',round(time.perf_counter()-start,1),flush=True)
    violation=[]
    for name in ['Bplus_baseline','B0_baseline','B0_mu_low']:
        record=ds[name];physical,y=noisy_transfer(record)
        sigma=.001*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
        for kind in ['ca_entry_gain','aa_to_ca_leakage']:
            for amp in (0.,.001,.01):
                obs=y.copy()
                if kind=='ca_entry_gain':obs[:,0,1]*=1+amp
                else:obs[:,0,1]+=amp*obs[:,1,1]
                f=raw_fit(record,obs,sigma,degree=1)
                violation.append(dict(dataset=name,perturbation=kind,amplitude=amp,g_bias=f['g']-.8,ua_bias=f['ua']-8.,**f))
    save('nonfactorizable_controls.csv',violation)
    snap.check();assert all(sha(Path(p))==h for p,h in source.items())
    summary=dict(status='OBSERVATION_DESIGN_VALIDATION_COMPLETE',elapsed_seconds=time.perf_counter()-start,profile_rows=len(profile_rows),higher_profile_rows=len(mesh_rows),heldout_rows=len(held),raw_hypothesis_rows=len(raw),higher_joint_bias_rows=len(bias),noise_rows=len(noise),independent_noise_realizations=1000,nonfactorizable_control_rows=len(violation),sources_unchanged=True,inputs_unchanged=True,source_sha256=source,input_sha256=snap.hashes,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},global_minimum_certified=False,material_probe_demonstrated=False,continuum_error_certified=False)
    dump(out/'diagnostic.json',summary);print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)},indent=2),flush=True)
if __name__=='__main__':main()