"""Charge zeros of the complete serialized 110-digit finite transition input.

The primary input preserves every stored cc/aa/ca digit, including its tiny
rank-one serialization defect. A separately labeled coherent-vertex input is
available. Neither result is merged with the old 1901 fixed-float zero checks.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parent
PRECISION=160
ZERO_TOL=mp.mpf('1e-50')
POLE_RATIO_GUARD=mp.mpf('1e-50')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exact(x):
    a,b=float(x).as_integer_ratio();return mp.mpf(a)/b

def determinant(p):return p[0]*p[1]-p[2]*p[2]-p[3]*p[3]
def determinant_derivative(p,dp):return dp[0]*p[1]+p[0]*dp[1]-2*(p[2]*dp[2]+p[3]*dp[3])
def matrix(p):return mp.matrix([[p[0],mp.mpc(p[2],p[3])],[mp.mpc(p[2],-p[3]),p[1]]])

class FullInput:
    def __init__(self,delta,weights,vc,ua,kind='serialized'):
        self.delta=list(delta);self.weights=[list(x) for x in weights];self.vc,self.ua=vc,ua;self.kind=kind
        if len(self.weights)!=4 or any(len(x)!=len(delta) for x in self.weights):raise ValueError('Four real weight columns required')
        if vc<0 or ua<0:raise ValueError('Repulsive diagonal interaction required')
    @classmethod
    def load(cls,path,kind='serialized'):
        records=json.loads(path.read_text());delta=[];columns=[[],[],[],[]]
        for row in records:
            delta.append(mp.mpf(row['delta']))
            if kind=='serialized':
                w=[mp.mpf(row['cc']),mp.mpf(row['aa']),mp.mpf(row['ca']['real']),mp.mpf(row['ca']['imag'])]
            elif kind=='coherent_vertices':
                df=mp.mpf(row['df_over_volume'])
                c=mp.mpc(row['charge_vertex']['real'],row['charge_vertex']['imag'])
                a=mp.mpc(row['axial_vertex']['real'],row['axial_vertex']['imag'])
                ca=df*c*mp.conj(a);w=[df*abs(c)**2,df*abs(a)**2,ca.real,ca.imag]
            else:raise ValueError('Unknown input interpretation')
            for column,value in zip(columns,w):column.append(value)
        Q=2*np.sin((2*np.pi/24)/2)
        return cls(delta,columns,exact(.8)/exact(Q)**2,mp.mpf(8),kind)
    def background(self,index,t,derivative=True):
        center=self.delta[index];pairs=[(j,center-d+t) for j,d in enumerate(self.delta) if j!=index]
        if any(d==0 for _,d in pairs):raise ValueError('Another bare transition reached')
        factors=[(j,1/d) for j,d in pairs]
        p=[mp.fsum(c[j]*f for j,f in factors) for c in self.weights]
        dp=[-mp.fsum(c[j]*f*f for j,f in factors) for c in self.weights] if derivative else None
        return p,dp
    def scaled(self,index,t,derivative=True):
        p,dp=self.background(index,t,derivative);w=[c[index] for c in self.weights]
        det=determinant(p);dw=determinant(w)
        cross=w[0]*p[1]+w[1]*p[0]-2*(w[2]*p[2]+w[3]*p[3])
        a=p[0]-self.ua*det;b=w[0]-self.ua*cross;c=-self.ua*dw
        ad=1-self.vc*p[0]-self.ua*p[1]+self.vc*self.ua*det
        bd=-self.vc*w[0]-self.ua*w[1]+self.vc*self.ua*cross;cd=self.vc*self.ua*dw
        n=t*t*a+t*b+c;d=t*t*ad+t*bd+cd
        result=dict(N2=n,D2=d,N_terms=[t*t*a,t*b,c],D_terms=[t*t*ad,t*bd,cd],
            N_background=a,N_weight=b,D_background=ad,D_weight=bd,weight_determinant=dw)
        if derivative:
            dd=determinant_derivative(p,dp)
            dcross=w[0]*dp[1]+w[1]*dp[0]-2*(w[2]*dp[2]+w[3]*dp[3])
            da=dp[0]-self.ua*dd;db=-self.ua*dcross
            dad=-self.vc*dp[0]-self.ua*dp[1]+self.vc*self.ua*dd;dbd=self.vc*self.ua*dcross
            result.update(N_prime=2*t*a+t*t*da+b+t*db,D_prime=2*t*ad+t*t*dad+bd+t*dbd)
        return result
    def root(self,index,kind='N'):
        center=self.delta[index]
        separation=min(abs(center-d) for j,d in enumerate(self.delta) if j!=index)
        if separation==0:raise ValueError('Bare degeneracy needs a summed-rank analysis')
        w=[c[index] for c in self.weights];scale=max(abs(x) for x in w)
        if scale==0:raise ValueError('Exactly zero target weight; no local branch')
        defect=abs(determinant(w))/scale**2
        if defect>mp.mpf('1e-95'):raise ValueError('Target is not coherent rank one to input precision')
        initial=self.scaled(index,mp.mpf(0),False)
        a,b=initial[kind+'_background'],initial[kind+'_weight']
        if a==0 or b==0:raise ValueError('No nonzero isolated first-order offset seed')
        t=-b/a
        if t==0 or abs(t)>=separation/4:raise ValueError('Offset seed outside protected local component')
        sign=mp.sign(t);converged=False
        for iteration in range(12):
            values=self.scaled(index,t)
            derivative=values[kind+'_prime']
            if derivative==0:raise ValueError('Non-simple cofactor root')
            step=values[kind+'2']/derivative;candidate=t-step
            for _ in range(50):
                if mp.sign(candidate)==sign and abs(candidate)<separation/4:break
                step/=2;candidate=t-step
            if not (mp.sign(candidate)==sign and abs(candidate)<separation/4):raise ValueError('Newton could not stay off bare boundary')
            t=candidate
            if abs(step)<=abs(t)*mp.mpf('1e-100'):
                converged=True;break
        if not converged:raise ValueError('Local cofactor root did not converge')
        if t==0 or center+t==0:raise ValueError('Bare or origin multiplication root rejected')
        values=self.scaled(index,t,False)
        scale_value=mp.fsum(abs(x) for x in values[kind+'_terms'])
        relative=abs(values[kind+'2'])/scale_value if scale_value else mp.inf
        if relative>ZERO_TOL:raise ValueError('Scaled cofactor residual failed')
        half=abs(t)*mp.mpf('1e-25');ends=[t-half,t+half]
        fv=[self.scaled(index,x,False)[kind+'2'] for x in ends]
        if fv[0]*fv[1]>=0:raise ValueError('No numerical two-sided local sign change')
        return t,dict(iterations=iteration+1,protected_radius=mp.nstr(separation/4,50),
            nearest_other_bare_distance=mp.nstr(separation,50),rank_one_relative_determinant=mp.nstr(defect,20),
            normalized_scaled_residual=mp.nstr(relative,25),offset=mp.nstr(t,PRECISION),
            center=mp.nstr(center,PRECISION),frequency=mp.nstr(center+t,PRECISION),
            offset_bracket=[mp.nstr(x,PRECISION) for x in ends],two_sided_sign_change=True,
            outward_rounding_certificate=False)
    def direct(self,index,t):
        if t==0:raise ValueError('Cannot test a bare multiplication root')
        z=self.delta[index]+t
        if z==0:raise ValueError('Origin excluded from this follow-up')
        factors=[1/(self.delta[index]-d+t) for d in self.delta]
        p=[mp.fsum(cj*f for cj,f in zip(c,factors)) for c in self.weights]
        det=determinant(p);n=p[0]-self.ua*det
        d=1-self.vc*p[0]-self.ua*p[1]+self.vc*self.ua*det
        term_scales=[mp.fsum(abs(cj*f) for cj,f in zip(c,factors)) for c in self.weights]
        nscale=term_scales[0]+self.ua*(term_scales[0]*term_scales[1]+term_scales[2]**2+term_scales[3]**2)
        dscale=1+abs(self.vc*p[0])+abs(self.ua*p[1])+abs(self.vc*self.ua*det)
        normalized_n=abs(n)/(nscale if nscale else 1)
        denominator_ratio=abs(d)/dscale
        if denominator_ratio<POLE_RATIO_GUARD:raise ValueError('D=0 or unresolved dielectric cancellation at candidate')
        pm=matrix(p);chi=(mp.eye(2)-pm*mp.diag([self.vc,self.ua]))**-1*pm
        chi_normalized=abs(chi[0,0])/(1+mp.norm(chi,p=2))
        if max(normalized_n,chi_normalized)>ZERO_TOL:raise ValueError('Unscaled cofactor/direct inverse check failed')
        return dict(numerator=mp.nstr(n,50),denominator=mp.nstr(d,50),
            normalized_numerator=mp.nstr(normalized_n,25),normalized_dielectric_denominator=mp.nstr(denominator_ratio,25),
            unscaled_charge_response=mp.nstr(abs(chi[0,0]),50),normalized_charge_response=mp.nstr(chi_normalized,25),
            cofactor_inverse_difference=mp.nstr(abs(n/d-chi[0,0]),25),
            direct_original_3456_slot_sum_checked=True)

def verify_zero(model,index,old,old_bounds,coherent=None):
    zero,zn=model.root(index,'N');pole,pd=model.root(index,'D')
    separation=abs(zero-pole)
    if separation<=mp.mpf('1e-70')*max(abs(zero),abs(pole)):raise ValueError('Pole-zero separation unresolved at declared precision')
    direct=model.direct(index,zero)
    center=model.delta[index];left,right=old_bounds
    distance=abs((center+pole)-exact(old['modal_endpoint']))
    mapping=('OLD_ENDPOINT_SHIFT_EXCEEDS_NEW_POLE_ZERO_SEPARATION' if distance>separation
             else 'COMPATIBLE_LOCAL_SCALE_WITHOUT_CERTIFIED_ONE_TO_ONE_MAPPING')
    result=dict(status='VERIFIED_NEW_SERIALIZED_INPUT_LOCAL_CHARGE_ZERO',old_interval_index=old['index'],
        raw_index=index,input_kind=model.kind,zero=zn,nearby_pole=pd,direct=direct,
        pole_zero_separation=mp.nstr(separation,70),old_endpoint_movement=mp.nstr(distance,50),
        new_zero_inside_old_numeric_interval=bool(left<center+zero<right),old_mapping_status=mapping,
        one_to_one_old_interval_mapping_certified=False,old_1901_count_increment=0,
        data_precision_digits=110,arithmetic_precision_digits=PRECISION,
        new_zero_separation_to_1e_100_input_scale=mp.nstr(separation/(mp.mpf('1e-100')*max(1,abs(center))),25),
        input_scale_is_rigorous_bound=False)
    if coherent is not None:
        cz,_=coherent.root(index,'N');coherent.direct(index,cz)
        result['coherent_vertex_crosscheck']=dict(status='VERIFIED_SEPARATE_COHERENT_INPUT',
            zero_offset_difference=mp.nstr(abs(cz-zero),70),
            difference_over_pole_zero_separation=mp.nstr(abs(cz-zero)/separation,25),
            inputs_identical=False,weight_cutoff_used=False)
    return result

def main(output,selection,limit,start_index,coherent_check):
    if output.exists() and any(output.iterdir()):raise RuntimeError('New empty output directory required')
    output.mkdir(parents=True,exist_ok=True);mp.mp.dps=PRECISION;started=time.perf_counter()
    source=ROOT/'band_precision_audit/local_poles_110/all_3456_transitions_110.json'
    paths=[Path(__file__).resolve(),ROOT/'test_zero_full_input_followup.py',source,
        ROOT/'band_precision_audit/local_poles_110/selected_local_poles.json',
        ROOT/'band_precision_audit/run_01/all_751_associations.json',ROOT/'mode_analysis/finite_charge_zeros.csv',
        ROOT/'band_precision_audit/manifest.json']
    frozen={str(p):sha(p) for p in paths}
    for rel,value in json.loads((ROOT/'band_precision_audit/manifest.json').read_text()).items():
        if sha(ROOT/rel)!=value:raise RuntimeError('Frozen band evidence changed: '+rel)
    model=FullInput.load(source);coherent=FullInput.load(source,'coherent_vertices') if coherent_check else None
    allcases=json.loads((ROOT/'band_precision_audit/run_01/all_751_associations.json').read_text())
    representative_ids={x['interval_index'] for x in json.loads((ROOT/'band_precision_audit/local_poles_110/selected_local_poles.json').read_text())}
    cases=[x for x in allcases if selection=='all' or (x['index'] in representative_ids)==(selection=='representatives')]
    cases=cases[start_index:(None if limit is None else start_index+limit)]
    with (ROOT/'mode_analysis/finite_charge_zeros.csv').open(newline='',encoding='utf-8') as f:old_rows=list(csv.DictReader(f))
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),source_input_sha256=frozen,
        selection=selection,start_index=start_index,requested_cases=len(cases),data_precision_digits=110,
        arithmetic_precision_digits=PRECISION,raw_transition_slots=len(model.delta),zero_residual_tolerance='1e-50',
        denominator_ratio_guard='1e-50',coherent_input_crosscheck=coherent_check,
        original_verified_zero_count_remains=1901,original_751_one_to_one_mapping_certified=False,
        complete_new_input_zero_catalog_certified=False,submission_ready=False)
    records=[]
    def checkpoint():
        report['elapsed_seconds']=time.perf_counter()-started
        report['completed_cases']=len(records);report['status_counts']=dict(Counter(x['status'] for x in records))
        dump(output/'cases.json',records);dump(output/'diagnostic.json',report)
    checkpoint()
    for position,old in enumerate(cases):
        case_start=time.perf_counter();index=old['nearest_raw_index'];bounds=old_rows[old['index']]
        try:
            result=verify_zero(model,index,old,(exact(bounds['left_pole']),exact(bounds['right_pole'])),coherent)
        except Exception as error:
            result=dict(status='UNRESOLVED_NEW_INPUT_LOCAL_ZERO',old_interval_index=old['index'],raw_index=index,
                error_type=type(error).__name__,reason=str(error),old_1901_count_increment=0,
                one_to_one_old_interval_mapping_certified=False)
        result['elapsed_seconds']=time.perf_counter()-case_start;records.append(result)
        checkpoint()
        message=f"{position+1}/{len(cases)} old interval {old['index']}: {result['status']} ({result['elapsed_seconds']:.3f}s)"
        print(message,flush=True)
        with (output/'execution.log').open('a',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
    if frozen!={str(p):sha(p) for p in paths}:raise RuntimeError('Source/input changed during computation')
    report.update(status='BOUNDED_NEW_FULL_INPUT_ZERO_FOLLOWUP_COMPLETE',source_inputs_unchanged=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['Stored 110-digit data are reused, not a new Hamiltonian calculation.',
            '160-digit arithmetic does not add physical input accuracy.',
            'Primary serialized weights retain their tiny rank-one serialization defect; no weight cutoff.',
            'Local roots are checked numerically, not by directed-rounding interval arithmetic.',
            'Old endpoint motion can exceed a true pole-zero separation; nearest-frequency association is not identity.',
            'The old 1901 checks and unresolved 751 labels are not rewritten or merged with this new-input count.'])
    checkpoint()
    dump(output/'manifest.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',choices=('representatives','remaining','all'),default='representatives')
    parser.add_argument('--start-index',type=int,default=0);parser.add_argument('--limit',type=int)
    parser.add_argument('--coherent-check',action='store_true');args=parser.parse_args()
    destination=args.output.resolve()
    try:main(destination,args.selection,args.limit,args.start_index,args.coherent_check)
    except Exception as error:
        if destination.exists():dump(destination/'failure.json',dict(type=type(error).__name__,reason=str(error)))
        raise
