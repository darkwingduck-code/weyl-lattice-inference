"""Read-only numerical/provenance audit of completed parameter-sector evidence.

The frozen-background result is independently reconstructed as RPA(Pi_L+H_ref),
not by calling the elimination formula used by the production sampler.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from extend_transverse import sha
from run_pole_zero_experiment import write_json
ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    matrices={}
    for prefix in ('bare','low','rpa','naive','matched'):
        matrices[prefix]=np.array([[[complex(float(r[f'{prefix}_{a}{b}_re']),float(r[f'{prefix}_{a}{b}_im'])) for b in range(2)] for a in range(2)] for r in rows])
    return rows,matrices


def solve(pi,u):return np.linalg.solve(np.eye(2)-pi@u,pi)


def audit(directory,output):
    errors=[];checks=[];hash_count=0;row_count=0;metadata_count=0
    def require(condition,message):
        checks.append(dict(check=message,pass_check=bool(condition)))
        if not condition:errors.append(message)
    index=json.loads((directory/'calibration_inputs.json').read_text(encoding='utf-8'))
    require(len(index['datasets'])==14,'Exactly 14 final physical datasets')
    require(len({x['id'] for x in index['datasets']})==14,'Unique final dataset identities')
    report_paths=[directory/(field+'_'+mesh)/'diagnostic.json' for field in ('B0','Bplus') for mesh in ('coarse','final')]
    maximum=dict(full_rpa=0.,naive_rpa=0.,matched_rpa=0.,reported_detector_metric=0.,reported_convergence=0.)
    for reportpath in report_paths:
        report=json.loads(reportpath.read_text(encoding='utf-8'));folder=reportpath.parent
        require(report['status']=='PARAMETER_SECTOR_SAMPLING_COMPLETE',str(folder.name)+' completed')
        require(report['sources_unchanged'] and report['inputs_unchanged'],str(folder.name)+' source/input unchanged')
        for path,digest in report['source_sha256'].items():
            require(sha(ROOT/path)==digest,'source '+path+' '+folder.name);hash_count+=1
        for path,digest in report['input_sha256'].items():
            require(sha(ROOT/path)==digest,'input '+path+' '+folder.name);hash_count+=1
        for path,digest in report['output_sha256'].items():
            require(sha(folder/path)==digest,'output '+path+' '+folder.name);hash_count+=1
        for name,metadata in report['cases'].items():
            saved=json.loads((folder/(name+'_metadata.json')).read_text(encoding='utf-8'))
            require(saved==metadata,'standalone metadata matches diagnostic '+folder.name+'/'+name)
            rows,values=read(folder/metadata['csv']);row_count+=len(rows);metadata_count+=1
            omega=np.array([float(r['omega']) for r in rows]);q=metadata['q'];Q=2*np.sin(q/2);u=np.diag([.8/Q**2,8.])
            require(np.array_equal(omega,np.unique(np.r_[0.,metadata['fixed_lab_frequencies'],metadata['scaled_q_frequencies']])),folder.name+'/'+name+' actual frequency union')
            for key in ('q','mu','T','eta','density'):
                require(all(float(r[key])==metadata[key] for r in rows),folder.name+'/'+name+' row '+key)
            require(all(float(r['B'])==(0. if metadata['field']=='B0' else 2*np.pi/36) for r in rows),folder.name+'/'+name+' actual B')
            require(metadata['checks']['identity_checks_pass'],folder.name+'/'+name+' stored full-band identities')
            h=values['bare'][0]-values['low'][0]
            truth={key:np.array([solve(p,u) for p in pi]) for key,pi in
                (('rpa',values['bare']),('naive',values['low']),('matched',values['low']+h))}
            for key,label in (('rpa','full_rpa'),('naive','naive_rpa'),('matched','matched_rpa')):
                error=float(np.max(abs(truth[key]-values[key])));maximum[label]=max(maximum[label],error)
                require(error<=2e-12,folder.name+'/'+name+' independent '+key)
            for window in ('fixed_lab','scaled_q'):
                ix=[int(np.argmin(abs(omega-x))) for x in metadata[window+'_frequencies']]
                full=truth['rpa'][ix]
                for key,method in (('naive','naive_low_bare_U'),('matched','matched_constant')):
                    for detector,a,b in (('cc',0,0),('aa',1,1),('ca',0,1),('ac',1,0)):
                        amplitude=float(np.max(abs(full[:,a,b])))
                        error=float(np.max(abs(truth[key][ix,a,b]-full[:,a,b])))/amplitude if amplitude else None
                        stored=metadata['metrics'][window][method][detector]['normalized_complex_sup_error']
                        delta=abs(error-stored) if error is not None and stored is not None else (0. if error is stored else np.inf)
                        maximum['reported_detector_metric']=max(maximum['reported_detector_metric'],delta)
                        require(delta<=2e-11*max(1.,abs(error or 0.)),folder.name+'/'+name+'/'+window+'/'+method+'/'+detector+' metric')
    for item in index['datasets']:
        fine=directory/item['csv'];require(sha(fine)==item['csv_sha256'],'index csv '+item['id']);hash_count+=1
        comparison=item.get('convergence')
        require(comparison is not None,'convergence present '+item['id'])
        if comparison is None:continue
        coarse=directory/comparison['coarse_csv'];require(sha(coarse)==comparison['coarse_sha256'],'coarse hash '+item['id']);hash_count+=1
        _,c=read(coarse);_,f=read(fine);actual={key:float(np.max(abs(f[key]-c[key])/(1+abs(f[key])))) for key in f}
        for key,value in actual.items():
            delta=abs(value-comparison['scaled_errors'][key]);maximum['reported_convergence']=max(maximum['reported_convergence'],delta)
            require(delta<2e-15,'independent convergence '+item['id']+'/'+key)
        passed=max(actual.values())<=.001
        require(comparison['tolerance']==.001 and comparison['pass_all']==passed,'unchanged comparison criterion '+item['id'])
        require(item['convergence_status']==('pass' if passed else 'fail'),'honest convergence status '+item['id'])
    for path,digest in index['input_sha256'].items():
        require(sha(directory/path)==digest,'index report '+path);hash_count+=1
    result=dict(status='PARAMETER_EVIDENCE_AUDIT_PASS' if not errors else 'PARAMETER_EVIDENCE_AUDIT_FAIL',
        all_checks_pass=not errors,checks_count=len(checks),sha256_checks=hash_count,response_rows=row_count,physical_case_metadata=metadata_count,
        numerical_maximum_errors=maximum,failed_checks=errors,
        final_convergence_status={x['id']:x['convergence_status'] for x in index['datasets']},
        scope='Evidence validity does not imply every physical case converged. Failed convergence must remain failed.',
        source_sha256={Path(__file__).name:sha(Path(__file__)), 'extend_transverse.py':sha(ROOT/'extend_transverse.py'),
                       'run_pole_zero_experiment.py':sha(ROOT/'run_pole_zero_experiment.py')},
        index_sha256=sha(directory/'calibration_inputs.json'),input_diagnostic_sha256={str(p.relative_to(directory)):sha(p) for p in report_paths})
    write_json(output,result);print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.directory.resolve(),a.output.resolve())
