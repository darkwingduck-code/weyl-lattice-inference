"""Report detector-normalized mesh differences without inventing a new pass rule.

The production 1e-3 criterion uses 1+|response|. It does not imply relative
accuracy for a weak detector or stability of a separately normalized metric.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_parameter_sector_outputs import read
from extend_transverse import sha,csv_write
from run_pole_zero_experiment import write_json


def audit(directory,output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh metric-stability directory required')
    output.mkdir(parents=True,exist_ok=True)
    ip=directory/'calibration_inputs.json';index=json.loads(ip.read_text(encoding='utf-8'));inputs={str(ip):sha(ip)};records=[]
    for item in index['datasets']:
        fine=directory/item['csv'];coarse=directory/item['convergence']['coarse_csv']
        fm=fine.with_name(fine.stem+'_metadata.json');cm=coarse.with_name(coarse.stem+'_metadata.json')
        for p in (fine,coarse,fm,cm):inputs[str(p)]=sha(p)
        fr,f=read(fine);cr,c=read(coarse);fmeta=json.loads(fm.read_text(encoding='utf-8'));cmeta=json.loads(cm.read_text(encoding='utf-8'))
        w=np.array([float(x['omega']) for x in fr])
        for window in ('fixed_lab','scaled_q'):
            ix=[int(np.argmin(abs(w-x))) for x in item[window+'_frequencies']]
            for detector,a,b in (('cc',0,0),('aa',1,1),('ca',0,1),('ac',1,0)):
                row=dict(id=item['id'],field=item['field'],case=item['case'],window=window,detector=detector,
                         original_convergence_status=item['convergence_status'])
                for prefix in ('bare','low','rpa','naive','matched'):
                    amplitude=float(np.max(abs(f[prefix][ix,a,b])));error=float(np.max(abs(f[prefix][ix,a,b]-c[prefix][ix,a,b])))
                    row[prefix+'_mesh_difference_over_fine_detector_sup']=error/amplitude if amplitude else None
                for key,method in (('naive','naive_low_bare_U'),('matched','matched_constant')):
                    fine_metric=fmeta['metrics'][window][method][detector]['normalized_complex_sup_error']
                    coarse_metric=cmeta['metrics'][window][method][detector]['normalized_complex_sup_error']
                    row[key+'_coarse_approximation_error']=coarse_metric;row[key+'_fine_approximation_error']=fine_metric
                    row[key+'_approximation_error_change']=fine_metric-coarse_metric
                records.append(row)
    path=output/'detector_metric_stability.csv';csv_write(path,records)
    write_json(output/'diagnostic.json',dict(status='DETECTOR_METRIC_STABILITY_REPORTED',records=len(records),
        source_sha256={Path(__file__).name:sha(Path(__file__)), 'audit_parameter_sector_outputs.py':sha(Path(__file__).with_name('audit_parameter_sector_outputs.py'))},
        input_sha256=inputs,output_sha256={path.name:sha(path)},
        detector_relative_convergence_certified=False,
        note='No new threshold was chosen. Original 1e-3 scaled criterion is unchanged; this records a distinct detector normalization and stability of approximation-error estimates.'))
    print('DETECTOR_METRIC_STABILITY_REPORTED',len(records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.directory.resolve(),a.output.resolve())
