"""Independent output/hash audit; numerical integrity is not a physics certificate."""
import csv,json
from pathlib import Path
import numpy as np
from extend_transverse import sha
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'uv_matching_audit'


def matrix(row,prefix):
    return np.array([[complex(float(row[f'{prefix}_{a}{b}_re']),float(row[f'{prefix}_{a}{b}_im'])) for b in range(2)] for a in range(2)])


def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def sectors(path):
    data=read(path)
    return np.array([[matrix(r,'bare') for r in sorted([x for x in data if x['sector']==name],key=lambda x:float(x['omega']))]
        for name in ('total','positive_positive')])


def responses(a):
    u=np.diag([.8/(2*np.sin(np.pi/24))**2,8.]);eye=np.eye(2)
    h=a[0,0]-a[1,0];left=np.linalg.inv(eye-h@u);right=np.linalg.inv(eye-u@h);effective=u@left
    return dict(bare_total=a[0],bare_positive_positive=a[1],
        full=np.array([np.linalg.solve(eye-p@u,p) for p in a[0]]),
        naive_low_bare_U=np.array([np.linalg.solve(eye-p@u,p) for p in a[1]]),
        matched_constant=np.array([left@h+left@np.linalg.solve(eye-p@effective,p)@right for p in a[1]]))


def main():
    folders=('cached_reference','sector_sampling','Bplus_zero_real_reference','B0_zero_real_reference')
    hashes=[];metricchecks=[];count=0
    for folder in folders:
        report=json.loads((BASE/folder/'diagnostic.json').read_text(encoding='utf-8'))
        for key,base in [('source_sha256',ROOT),('input_sha256',ROOT),('output_sha256',BASE/folder)]:
            for name,expected in report[key].items():
                actual=sha(base/name)
                hashes.append(dict(report=folder,path=str((base/name).relative_to(ROOT)),expected=expected,actual=actual,passed=expected==actual))
        if folder=='sector_sampling':continue
        rows=read(BASE/folder/'responses.csv');count+=len(rows)
        truth=np.array([matrix(r,'response') for r in rows if r['method']=='full'])
        for method in report['metrics']:
            predicted=np.array([matrix(r,'response') for r in rows if r['method']==method])
            for detector,a,b in [('cc',0,0),('aa',1,1),('ca',0,1),('ac',1,0)]:
                target=truth[:,a,b];got=predicted[:,a,b]
                actual=float(np.max(np.abs(got-target))/np.max(np.abs(target)))
                expected=report['metrics'][method][detector]['normalized_complex_sup_error']
                metricchecks.append(dict(folder=folder,method=method,detector=detector,recorded=expected,recomputed=actual,
                    passed=bool(np.isclose(actual,expected,rtol=1e-13,atol=1e-15))))
    sample=json.loads((BASE/'sector_sampling/diagnostic.json').read_text(encoding='utf-8'))
    convchecks=[]
    for label,coarse,fine in [('B0_transverse_L48_to72_Nk192','B0_L48_Nk192','B0_L72_Nk192'),
        ('B0_longitudinal_L72_Nk192_to384','B0_L72_Nk192','B0_L72_Nk384')]:
        ca,fa=responses(sectors(BASE/f'sector_sampling/{coarse}.csv')),responses(sectors(BASE/f'sector_sampling/{fine}.csv'))
        for key in ca:
            actual=float(np.max(np.abs(fa[key]-ca[key])/(1+np.abs(fa[key]))))
            expected=sample['comparisons'][label]['scaled_errors'][key]
            convchecks.append(dict(comparison=label,response=key,recorded=expected,recomputed=actual,
                passed=bool(np.isclose(actual,expected,rtol=1e-10,atol=1e-14))))
    passed=all(x['passed'] for x in hashes+metricchecks+convchecks)
    result=dict(status='UV_MATCHING_INDEPENDENT_VALIDATION_PASS' if passed else 'UV_MATCHING_INDEPENDENT_VALIDATION_FAIL',
        all_checks_pass=passed,hash_count=len(hashes),hash_checks=hashes,response_csv_rows_checked=count,
        detector_metric_checks=metricchecks,convergence_checks=convchecks,
        source_sha256={Path(__file__).name:sha(Path(__file__))},
        scope='Integrity and arithmetic checks only; not a universal constant-background approximation theorem, pole/gap certificate, or calibrated experimental measurement claim.')
    output=BASE/'independent_validation.json'
    if output.exists():raise RuntimeError('Preserve existing audit')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'],len(hashes),'hashes;',count,'response rows;',len(metricchecks),'detector metrics;',len(convchecks),'convergence metrics')
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
