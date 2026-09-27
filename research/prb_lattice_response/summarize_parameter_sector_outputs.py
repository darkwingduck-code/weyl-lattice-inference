"""Postprocess frozen parameter samples: window-specific background dispersion.

This uses the standard resolvent difference identity only as an algebra check;
finite-window matching failures and detector errors are empirical observations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_parameter_sector_outputs import read
from extend_transverse import csv_write,sha
from run_pole_zero_experiment import write_json


def summarize(directory,output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh summary directory required')
    output.mkdir(parents=True,exist_ok=True)
    ip=directory/'calibration_inputs.json';index=json.loads(ip.read_text(encoding='utf-8'))
    rows=[];inputs={str(ip):sha(ip)};maximum=0.;case_reports={}
    for item in index['datasets']:
        cp=directory/item['csv'];mp=cp.with_name(cp.stem+'_metadata.json')
        raw,v=read(cp);metadata=json.loads(mp.read_text(encoding='utf-8'));inputs[str(cp)]=sha(cp);inputs[str(mp)]=sha(mp)
        omega=np.array([float(r['omega']) for r in raw]);h=v['bare']-v['low'];href=h[0]
        q=item['q'];u=np.diag([.8/(2*np.sin(q/2))**2,8.]);identity=[];bounds=[]
        for i in range(len(omega)):
            p=v['bare'][i];p0=v['low'][i]+href;dh=h[i]-href
            left=np.linalg.inv(np.eye(2)-p@u);right=np.linalg.inv(np.eye(2)-u@p0)
            error=v['rpa'][i]-v['matched'][i]
            identity.append(float(np.max(abs(error-left@dh@right))))
            bounds.append(float(np.linalg.norm(left,2)*np.linalg.norm(dh,2)*np.linalg.norm(right,2)))
        maximum=max(maximum,max(identity));case_reports[item['id']]=dict(
            reference_excluded=href,resolvent_difference_identity_max_error=max(identity),
            resolvent_difference_note='Full and matched responses differ by L_full [H(omega)-H_ref] R_matched. This standard identity is not new physics.')
        for window in ('fixed_lab','scaled_q'):
            ix=np.array([int(np.argmin(abs(omega-x))) for x in item[window+'_frequencies']])
            for detector,a,b in (('cc',0,0),('aa',1,1),('ca',0,1),('ac',1,0)):
                metric=metadata['metrics'][window]
                high=h[ix,a,b];dh=high-href[a,b]
                rows.append(dict(id=item['id'],field=item['field'],case=item['case'],window=window,detector=detector,
                    q=item['q'],mu=item['mu'],T=item['T'],eta=item['eta'],density=item['density'],
                    convergence_status=item['convergence_status'],
                    naive_complex_sup_error=metric['naive_low_bare_U'][detector]['normalized_complex_sup_error'],
                    matched_complex_sup_error=metric['matched_constant'][detector]['normalized_complex_sup_error'],
                    excluded_reference_re=float(href[a,b].real),excluded_reference_im=float(href[a,b].imag),
                    excluded_reference_magnitude=float(abs(href[a,b])),
                    excluded_variation_max_absolute=float(np.max(abs(dh))),
                    excluded_variation_over_reference=(float(np.max(abs(dh))/abs(href[a,b])) if abs(href[a,b]) else None),
                    excluded_max_abs_imag=float(np.max(abs(high.imag))),
                    excluded_real_min=float(high.real.min()),excluded_real_max=float(high.real.max()),
                    maximum_total_matrix_error=float(np.max(abs(v['rpa'][ix]-v['matched'][ix]))),
                    maximum_resolvent_spectral_norm_bound=max(bounds[i] for i in ix)))
    csv_write(output/'window_detector_background.csv',rows)
    report=dict(status='PARAMETER_BACKGROUND_SUMMARY_COMPLETE' if maximum<2e-12 else 'RESOLVENT_IDENTITY_FAILURE',
        cases=case_reports,maximum_resolvent_identity_error=maximum,records=len(rows),
        limitations=['The low sector is energy-sign selected, not a certified UV energy-transfer separation.',
            'Background dispersion alone does not prove an interband continuum threshold or analytic pole.',
            'Fixed q finite eta matching does not certify current/torque/contact/high-frequency identities.',
            'Convergence status inherits the explicit finite-grid comparison only.'],
        source_sha256={Path(__file__).name:sha(Path(__file__)), 'audit_parameter_sector_outputs.py':sha(Path(__file__).with_name('audit_parameter_sector_outputs.py'))},
        input_sha256=inputs,output_sha256={'window_detector_background.csv':sha(output/'window_detector_background.csv')})
    write_json(output/'diagnostic.json',report)
    print(report['status'], 'rows',len(rows),'identity',maximum)
    if maximum>=2e-12:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.directory.resolve(),a.output.resolve())
