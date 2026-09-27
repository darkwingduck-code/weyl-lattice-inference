"""Independent numeric summary/hash audit for the frozen low-energy run."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from low_energy_mode_audit import analyze
from extend_transverse import sha
from run_pole_zero_experiment import write_json
ROOT=Path(__file__).resolve().parent

def main(folder):
    report=json.loads((folder/'diagnostic.json').read_text(encoding='utf-8'))
    if report['status']!='LOW_ENERGY_FINITE_RESPONSE_AUDIT_COMPLETE':raise RuntimeError('Complete first-stage run required')
    checks=[]
    for key,base in [('source_sha256',ROOT),('input_sha256',ROOT),('output_sha256',folder)]:
        for relative,expected in report[key].items():
            actual=sha(base/relative)
            checks.append(dict(kind=key,path=relative,expected=expected,actual=actual,passed=actual==expected))
    def principal(case,channel):
        peaks=[p for p in case['peaks'][channel] if .025<=p['omega']<=.4]
        return max(peaks,key=lambda x:x['height']) if peaks else None
    cached=[dict(dataset=c['dataset'],case=c['case'],axial=c['axial'],coulomb=c['coulomb'],remove_mixing=c['remove_mixing'],
        minimum_dielectric_singular_value=c['minimum_dielectric_singular_value'],
        charge_peak=principal(c,'charge'),axial_peak=principal(c,'axial'),
        maximum_bare_axial_real=c['maximum_bare_axial_real'],
        maximum_offdiagonal_determinant_fraction=c['maximum_offdiagonal_determinant_fraction']) for c in report['cached_cases']]
    eta_cases=[]
    for c in report.get('eta_cases',[]):
        if c['case']!=2:continue
        cp,ap=principal(c,'charge'),principal(c,'axial')
        eta_cases.append(dict(eta=c['eta'],charge_peak=cp,axial_peak=ap,
            charge_width_over_2eta=cp['half_height_width']/(2*c['eta']) if cp and cp['half_height_width'] else None,
            minimum_dielectric_singular_value=c['minimum_dielectric_singular_value'],
            minimum_at_omega=c['minimum_at_omega'],maximum_bare_axial_real=c['maximum_bare_axial_real']))
    rows=list(csv.DictReader((folder/'transition_sectors.csv').open(newline='',encoding='utf-8')))
    baseline=next(c for c in eta_cases if c['eta']==.015)
    wp=baseline['charge_peak']['omega']
    selected={r['sector']:r for r in rows if abs(float(r['omega'])-wp)<1e-12}
    total=-float(selected['total']['bare_00_im'])
    ratios={name:(-float(r['bare_00_im'])/total) for name,r in selected.items()}
    # Two independent bare oscillators have two loss peaks even though the
    # interaction-free dielectric matrix is identically I (no RPA zeros).
    w=np.linspace(.02,1.2,601);z=w+.004j;pi=np.zeros((len(w),2,2),complex)
    pi[:,0,0]=.02/(z-.3)-.02/(z+.3);pi[:,1,1]=.03/(z-.7)-.03/(z+.7)
    fixture=analyze(w,pi,.2,.004,coulomb=0,axial=0)
    negative=dict(charge_peak_count=len(fixture['summary']['peaks']['charge']),
        axial_peak_count=len(fixture['summary']['peaks']['axial']),
        minimum_dielectric_singular_value=float(fixture['singular'].min()),
        dielectric_identity_error=float(np.max(np.abs(fixture['kernel']-np.eye(2)))),
        two_peaks_are_not_a_hybrid_gap=bool(fixture['summary']['peaks']['charge'] and fixture['summary']['peaks']['axial']
            and np.max(np.abs(fixture['kernel']-np.eye(2)))<1e-14))
    minima={}
    for filename in ('cached_controls.csv','eta_controls.csv'):
        data=list(csv.DictReader((folder/filename).open(newline='',encoding='utf-8')))
        for row in data:
            if int(row['case'])!=2:continue
            key=row['dataset']+'_eta'+row['eta']
            value=min(float(row['eigenchannel0_re']),float(row['eigenchannel1_re']))
            minima[key]=min(minima.get(key,float('inf')),value)
    summary=dict(status='LOW_ENERGY_AUDIT_INDEPENDENT_SUMMARY_PASS',all_hashes_pass=all(x['passed'] for x in checks),
        hash_checks=checks,hash_count=len(checks),input_report_sha256=sha(folder/'diagnostic.json'),
        summary_source_sha256=sha(Path(__file__)),cached_principal_peak_controls=cached,eta_principal_peak_controls=eta_cases,
        sector_frequency=wp,charge_bare_loss_sector_ratios=ratios,
        minimum_sampled_real_dielectric_eigenvalue=minima,two_bare_peaks_negative_control=negative,
        interpretation_scope='Positive sampled singular values and finite-eta peak tracking do not prove absence of a collective pole. The audit does not establish two analytic low-energy poles, residues, intrinsic widths, or a stable hybrid gap.',
        remaining_requirements=report['unresolved_requirements'])
    if not summary['all_hashes_pass'] or not negative['two_peaks_are_not_a_hybrid_gap']:
        summary['status']='LOW_ENERGY_AUDIT_INDEPENDENT_SUMMARY_FAIL'
    target=folder/'independent_summary.json'
    if target.exists():raise RuntimeError('Do not overwrite prior independent summary')
    write_json(target,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='hash_checks'},indent=2))
    if summary['status'].endswith('FAIL'):raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-folder',type=Path,required=True)
    main(parser.parse_args().output_folder.resolve())
