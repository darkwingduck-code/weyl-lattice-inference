"""Additional declared nuisance and covariance checks, preserving primary sweep."""
import time,json
import numpy as np
from scipy.optimize import least_squares
from observation_design import *
from run_validation import noisy_transfer,raw_fit,fit_products

def fit_frequency(record,y,sigma,ua):
    """Independent per-frequency complex axial source/readout gains, analytic Jacobian."""
    N=len(y);size=y.size;rows=np.arange(N)*4;Q=2*np.sin(record['q']/2)
    ec=np.diag([1.,0.]);scale=float(sigma)*np.sqrt(size)
    def calculate(v,jac=False):
        g=v[0];d=v[1:1+N]+1j*v[1+N:1+2*N];s=v[1+2*N:1+3*N]+1j*v[1+3*N:]
        chi=response(record['pi'],record['q'],g,ua);m=transfer(chi,d,s)
        delta=(m-y).ravel()/scale
        if not jac:return np.r_[delta.real,delta.imag]
        J=np.zeros((size,len(v)),complex)
        J[:,0]=transfer(chi@ec@chi/Q**2,d,s).ravel()
        for base,factor in [(1,1.),(1+N,1j)]:
            J[rows+2,base+np.arange(N)]=factor*chi[:,1,0]
            J[rows+3,base+np.arange(N)]=factor*s*chi[:,1,1]
        for base,factor in [(1+2*N,1.),(1+3*N,1j)]:
            J[rows+1,base+np.arange(N)]=factor*chi[:,0,1]
            J[rows+3,base+np.arange(N)]=factor*d*chi[:,1,1]
        J/=scale
        return np.r_[J.real,J.imag]
    candidates=[]
    for g in (.6,.8,1.):
        chi=response(record['pi'],record['q'],g,ua)
        d=y[:,1,0]/chi[:,1,0];s=y[:,0,1]/chi[:,0,1]
        v=np.r_[g,d.real,d.imag,s.real,s.imag]
        fit=least_squares(calculate,v,jac=lambda v:calculate(v,True),bounds=(np.r_[.2,np.full(4*N,-np.inf)],np.r_[1.6,np.full(4*N,np.inf)]),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        candidates.append(fit)
    fit=min(candidates,key=lambda f:np.linalg.norm(f.fun));v=fit.x
    # Direct directional derivative check at final fit, not the optimizer's own finite difference.
    rng=np.random.default_rng(42);direction=rng.normal(size=len(v));direction/=np.linalg.norm(direction);step=1e-6
    fd=(calculate(v+step*direction)-calculate(v-step*direction))/(2*step)
    err=np.linalg.norm(fd-calculate(v,True)@direction)/(1+np.linalg.norm(fd))
    gains=np.column_stack([v[1:1+N],v[1+N:1+2*N],v[1+2*N:1+3*N],v[1+3*N:]])
    return dict(g=float(v[0]),rms=float(np.linalg.norm(fit.fun)),success=bool(fit.success),all_starts_success=all(f.success for f in candidates),analytic_jacobian_directional_error=float(err)),gains

def main():
    out=ROOT/'nuisance_robustness';out.mkdir(exist_ok=False);started=time.perf_counter()
    data,snap,_=load_all();sources={str(p):sha(p) for p in [Path(__file__),ROOT/'observation_design.py',ROOT/'run_validation.py']}
    rows=[];fits=[];rng=np.random.default_rng(9020260912)
    # Exact variance of W e_aa-y_ac e_ca-y_ca e_ac-e_ca e_ac under independent circular Gaussian entry noise.
    for name in ['Bplus_baseline','B0_mu_low']:
        rec=data['fixed_lab']['fine'][name];physical,y=noisy_transfer(rec);W=closure(y)
        for mult in (.001,.01):
            sigma=mult*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
            total=np.zeros(len(y));mean=np.zeros(len(y),complex);R=100000
            predicted=(abs(W)**2+abs(y[:,1,0])**2+abs(y[:,0,1])**2)*sigma**2+sigma**4
            for _ in range(100):
                e=(rng.normal(size=(1000,len(y),3))+1j*rng.normal(size=(1000,len(y),3)))*sigma/np.sqrt(2)
                F=W[None,:]*e[:,:,2]-y[None,:,1,0]*e[:,:,0]-y[None,:,0,1]*e[:,:,1]-e[:,:,0]*e[:,:,1]
                total+=np.sum(abs(F)**2,axis=0);mean+=np.sum(F,axis=0)
            for j,w in enumerate(rec['omega']):rows.append(dict(dataset=name,sigma_multiplier=mult,omega=float(w),samples=R,predicted_variance=float(predicted[j]),empirical_second_moment=float(total[j]/R),variance_relative_difference=float(total[j]/R/predicted[j]-1),normalized_mean_abs=float(abs(mean[j]/R)/np.sqrt(predicted[j]))))
        sigma=.001*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
        for u in (4.,8.,16.):
            f,gains=fit_frequency(rec,y,sigma,u)
            fits.append(dict(dataset=name,ua_trial=u,complex_sigma=sigma,**f))
            table(out/f'{name}_Ua{u:g}_frequency_gains.csv',[dict(omega=float(w),d_re=float(z[0]),d_im=float(z[1]),s_re=float(z[2]),s_im=float(z[3])) for w,z in zip(rec['omega'],gains)])
        print(name,'covariance and arbitrary frequency transfer done',time.perf_counter()-started,flush=True)
    table(out/'product_covariance.csv',rows);table(out/'frequency_gain_raw_profiles.csv',fits)
    rec=data['fixed_lab']['fine']['Bplus_baseline'];physical,y=noisy_transfer(rec);sigma=.001*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y))
    base=[]
    for level in ['coarse','fine']:
        f=raw_fit(data['fixed_lab'][level]['Bplus_baseline'],y,sigma,degree=1)
        base.append(dict(dataset='Bplus_baseline',model_level=level,reference_level='fine',g_bias=f['g']-.8,ua_bias=f['ua']-8.,**f))
    table(out/'baseline_same_reference_grid.csv',base)
    # Predeclared noise amplitude .01, now crossed with weak-mixing mu=.25 to expose the failure regime.
    rec=data['fixed_lab']['fine']['B0_mu_low'];physical,y=noisy_transfer(rec);sigma=.01*np.linalg.norm(physical[:,1,1])/np.sqrt(len(y));scales=[np.linalg.norm(y[:,0,0]),np.linalg.norm(y[:,0,1]*y[:,1,0])]
    failures=[]
    for i in range(100):
        e=(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))*sigma/np.sqrt(2);obs=y+e
        for method,f in [('cross_product',fit_products(rec,obs,scales)),('raw_matrix_affine',raw_fit(rec,obs,sigma,degree=1))]:
            failures.append(dict(replica=i,method=method,g=f['g'],ua=f['ua'],rms=f['rms'],ua_bias=f['ua']-8.,success=f['success'],all_starts_success=f['all_starts_success']))
    table(out/'weak_signal_noise.csv',failures)
    snap.check();assert all(sha(Path(p))==h for p,h in sources.items())
    dump(out/'diagnostic.json',dict(status='NUISANCE_ROBUSTNESS_COMPLETE',elapsed_seconds=time.perf_counter()-started,source_sha256=sources,input_sha256=snap.hashes,independent_additional_noise_realizations=100,product_covariance_realizations=400000,source_inputs_unchanged=True,all_frequency_fits_success=all(r['success'] for r in fits),maximum_jacobian_check_error=max(r['analytic_jacobian_directional_error'] for r in fits),output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},global_minimum_certified=False))
    print('COMPLETE',time.perf_counter()-started,flush=True)
if __name__=='__main__':main()