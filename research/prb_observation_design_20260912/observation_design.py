"""New observation design on immutable finite-grid four-band RPA data."""
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'prb_lattice_response'
sys.path.insert(0,str(OLD))
from parameter_calibration_independent_review import Snapshot,load_windows
from parameter_calibration_audit import load_datasets
from detector_boundary_refinement import add_higher

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_all():
    snap=Snapshot();idx=OLD/'parameter_sector_sampling/calibration_inputs.json'
    # Frozen full metadata/hash validator, followed by independent cofactor reader.
    load_datasets(idx,'fixed_lab',snap.hashes)
    _,data,err=load_windows(idx,snap)
    add_higher(data,OLD/'detector_boundary_refinement',snap)
    for window,levels in data.items():
        for name,h in levels['higher'].items():
            f=levels['fine'][name]
            for key in ('q','mu','eta','field'):
                a=h['q'] if key=='q' else h['item'][key]
                b=f['q'] if key=='q' else f['item'][key]
                if a!=b:raise ValueError('Higher alignment mismatch '+key)
            if np.max(abs(h['omega']-f['omega']))>2e-14:raise ValueError('Higher omega mismatch')
    snap.check();return data,snap,err

def response(p,q,g,u):
    # Vectorized direct matrix solve; independent scalar cofactors are tested.
    interaction=np.diag([g/(2*np.sin(q/2))**2,u])
    return np.linalg.solve(np.eye(2)-p@interaction,p)
def models(data,names,g,u):return {n:response(data[n]['pi'],data[n]['q'],g,u) for n in names}
def closure(x):
    if np.any(x[:,1,1]==0):raise ValueError('Zero aa: direct closure undefined')
    return x[:,0,1]*x[:,1,0]/x[:,1,1]
def transfer(x,d,s):
    y=x.copy();y[:,1,:]*=np.asarray(d)[...,None] if np.ndim(d) else d
    y[:,:,1]*=np.asarray(s)[...,None] if np.ndim(s) else s
    return y

def residual(model,truth,names,policy,omega=None,normalizers=None):
    """Relative L2 by dataset/channel. Closure adds a *derived* channel."""
    out=[];gains={}
    if policy.startswith('diag_'):
        separate='separate' in policy
        groups=[[n] for n in names] if separate else [names]
        degree=1 if 'tilt' in policy else 0
        for group in groups:
            xs=[];ys=[]
            for n in group:
                scale=np.linalg.norm(truth[n][:,1,1]);x=model[n][:,1,1]/scale
                t=(omega[n]-np.mean(omega[n]))/np.ptp(omega[n]) if degree else np.zeros(len(x))
                xs.append(np.column_stack([x]+[x*t] if degree else [x]));ys.append(truth[n][:,1,1]/scale)
            X=np.concatenate(xs);y=np.concatenate(ys)
            if 'real' in policy:
                a=np.array([max(0.,np.vdot(X[:,0],y).real/np.vdot(X[:,0],X[:,0]).real)])
            else:a=np.linalg.lstsq(X,y,rcond=None)[0]
            for n in group:
                t=(omega[n]-np.mean(omega[n]))/np.ptp(omega[n]) if degree else np.zeros(len(model[n]))
                gains[n]=a[0]+(a[1]*t if degree else 0.)
    for n in names:
        x,y=model[n],truth[n]
        ccscale=np.linalg.norm(y[:,0,0]) if normalizers is None else normalizers[n][0]
        out.append((x[:,0,0]-y[:,0,0])/ccscale)
        if policy=='charge':continue
        if policy.startswith('diag_'):
            out.append((gains[n]*x[:,1,1]-y[:,1,1])/np.linalg.norm(y[:,1,1]))
        elif policy=='closure':
            scale=np.linalg.norm(closure(y)) if normalizers is None else normalizers[n][1]
            out.append((closure(x)-closure(y))/scale)
        elif policy=='cross_product':
            # No measured aa division; normalizer is recorded and held fixed in noise runs.
            scale=np.linalg.norm(y[:,0,1]*y[:,1,0]) if normalizers is None else normalizers[n][1]
            out.append((closure(x)*y[:,1,1]-y[:,0,1]*y[:,1,0])/scale)
        else:raise ValueError(policy)
    v=np.concatenate(out)/np.sqrt(len(out))
    return np.concatenate([v.real,v.imag])

def profile(data,truth,names,ua,policy,ng=65,normalizers=None):
    omega={n:data[n]['omega'] for n in names}
    def cost(g):
        r=residual(models(data,names,g,ua),truth,names,policy,omega,normalizers)
        return float(r@r)
    grid=np.linspace(.2,1.6,ng);vals=np.array([cost(g) for g in grid])
    candidates=[(float(vals[i]),float(grid[i]),True) for i in [0,len(grid)-1,int(vals.argmin())]]
    candidates.append((cost(.8),.8,True))
    for i in range(1,len(grid)-1):
        if vals[i]<=vals[i-1] and vals[i]<=vals[i+1]:
            fit=minimize_scalar(cost,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
            candidates.append((float(fit.fun),float(fit.x),bool(fit.success)))
    best=min([c for c in candidates if c[2]],key=lambda a:a[0])
    return dict(g=best[1],rms=float(np.sqrt(best[0])),all_local_success=all(c[2] for c in candidates),global_optimum_certified=False)

def invariant_checks(data):
    errors=[];amps=[]
    for w,levels in data.items():
        for level,ds in levels.items():
            for n,v in ds.items():
                x=response(v['pi'],v['q'],.8,8);inv=np.linalg.inv(v['pi'])
                target=1/(inv[:,0,0]-.8/(2*np.sin(v['q']/2))**2)
                om=v['omega'];d=(.7+.4j)*(1+.13j*om);s=(1.4-.3j)*(1-.17*om)
                y=transfer(x,d,s)
                errors.extend([np.max(abs(x[:,0,0]-closure(x)-target)),np.max(abs(closure(x)-closure(y)))])
                norms=[np.linalg.norm(x[:,a,b]) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
                amps.append(dict(window=w,level=level,dataset=n,cc_l2=norms[0],ca_l2=norms[1],ac_l2=norms[2],aa_l2=norms[3],W_l2=float(np.linalg.norm(closure(x))),cross_over_aa=min(norms[1:3])/norms[3],cross_over_cc=min(norms[1:3])/norms[0],min_abs_aa=float(np.min(abs(x[:,1,1])))))
    return float(max(errors)),amps

def pilot():
    out=ROOT/'pilot';out.mkdir(exist_ok=False);start=time.perf_counter();data,snap,cache=load_all()
    errs,amps=invariant_checks(data);table(out/'signal_amplitudes.csv',amps)
    policies=['charge','diag_shared_real','diag_separate_real','diag_shared_complex','diag_separate_complex','diag_separate_tilt','closure','cross_product']
    groups={f:[f] for f in ['B0_baseline','Bplus_baseline']}
    groups['B_pair']=['B0_baseline','Bplus_baseline']
    for axis,cases in [('q',['qhalf','baseline','qdouble']),('mu',['mu_low','baseline','mu_high'])]:
        for field in ['B0','Bplus']:
            groups[axis+'_'+field]=[field+'_'+c for c in cases]
        groups[axis+'_both']=[f+'_'+c for f in ['B0','Bplus'] for c in cases]
    rows=[]
    for window,levels in data.items():
        for name,names in groups.items():
            if window=='scaled_q' and not name.startswith('q_'):continue
            ds=levels['fine'];truth=models(ds,names,.8,8.)
            for u in [4.,16.]:
                for policy in policies:
                    fit=profile(ds,truth,names,u,policy)
                    rows.append(dict(group=name,window=window,policy=policy,ua=u,**fit))
            print(window,name,len(rows),'profiles',round(time.perf_counter()-start,1),'s',flush=True)
    table(out/'profiles.csv',rows);snap.check()
    dump(out/'diagnostic.json',dict(status='PILOT_COMPLETE',profiles=len(rows),elapsed_seconds=time.perf_counter()-start,algebra_max_abs=errs,cached_response_max_abs=cache,input_sha256=snap.hashes,source_sha256={str(Path(__file__)):sha(__file__)},inputs_unchanged=True,outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':pilot()