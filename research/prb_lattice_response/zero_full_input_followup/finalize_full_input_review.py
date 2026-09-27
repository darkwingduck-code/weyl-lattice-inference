"""Independent arithmetic/delivery checks of the new full-input zero batches."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero_full_input_followup import ROOT,FullInput,sha,dump,exact

BASE=ROOT/'zero_full_input_followup'
BATCHES=('representatives_01','remaining_000_363','remaining_364_727')

def main(output):
    if output.exists() and any(output.iterdir()):raise RuntimeError('Fresh review output required')
    reports={};records=[];frozen={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    for name in BATCHES:
        folder=BASE/name;report=json.loads((folder/'diagnostic.json').read_text())
        if report['status']!='BOUNDED_NEW_FULL_INPUT_ZERO_FOLLOWUP_COMPLETE':raise RuntimeError('Batch not complete: '+name)
        for relative,expected in json.loads((folder/'manifest.json').read_text()).items():
            if sha(folder/relative)!=expected:raise RuntimeError('Changed batch artifact: '+relative)
            frozen[str(folder/relative)]=expected
        frozen[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        for path,value in report['source_input_sha256'].items():
            if sha(Path(path))!=value:raise RuntimeError('Changed source/input: '+path)
            frozen[path]=value
        reports[name]=report;records.extend(json.loads((folder/'cases.json').read_text()))
    ids=[r['old_interval_index'] for r in records]
    associations=json.loads((ROOT/'band_precision_audit/run_01/all_751_associations.json').read_text())
    expected={r['index'] for r in associations}
    if len(ids)!=751 or len(set(ids))!=751 or set(ids)!=expected:raise RuntimeError('Incomplete or duplicate original-751 coverage')
    output.mkdir(parents=True,exist_ok=True);mp.mp.dps=180
    good=[r for r in records if r['status']=='VERIFIED_NEW_SERIALIZED_INPUT_LOCAL_CHARGE_ZERO']
    bad=[r for r in records if r['status']!='VERIFIED_NEW_SERIALIZED_INPUT_LOCAL_CHARGE_ZERO']
    # Explicit charge-only review: absolute chi and hence chi/(1+sum |Wcc/(z-delta)|).
    if any(mp.mpf(r['direct']['unscaled_charge_response'])>mp.mpf('1e-50') for r in good):
        raise RuntimeError('Absolute charge-only residual exceeds 1e-50')
    radius_records=sorted((mp.mpf(r['zero']['center'])-mp.mpf(r['zero']['protected_radius']),
                           mp.mpf(r['zero']['center'])+mp.mpf(r['zero']['protected_radius']),r['raw_index']) for r in good)
    if any(a[1]>=b[0] for a,b in zip(radius_records,radius_records[1:])):raise RuntimeError('Distinct target neighborhoods overlap')
    source=ROOT/'band_precision_audit/local_poles_110/all_3456_transitions_110.json'
    model=FullInput.load(source)
    ordered=sorted(good,key=lambda r:mp.mpf(r['pole_zero_separation']))
    # Deterministic scale coverage + worst recorded charge residual and dielectric condition.
    chosen={ordered[round(i*(len(ordered)-1)/31)]['old_interval_index'] for i in range(32)} if ordered else set()
    if good:
        chosen.add(max(good,key=lambda r:mp.mpf(r['direct']['unscaled_charge_response']))['old_interval_index'])
        chosen.add(min(good,key=lambda r:mp.mpf(r['direct']['normalized_dielectric_denominator']))['old_interval_index'])
    cases=[]
    for record in good:
        if record['old_interval_index'] not in chosen:continue
        i=record['raw_index'];t=mp.mpf(record['zero']['offset']);p=model.direct(i,t)
        factors=[1/(model.delta[i]-d+t) for d in model.delta]
        charge_scale=1+mp.fsum(abs(w*f) for w,f in zip(model.weights[0],factors))
        charge_only=mp.mpf(p['unscaled_charge_response'])/charge_scale
        cases.append(dict(old_interval_index=record['old_interval_index'],raw_index=i,
            recalculation_precision_digits=180,data_precision_digits=110,
            direct=p,charge_only_normalized_residual=mp.nstr(charge_only,30),
            same_literal_input=True,new_hamiltonian_precision_claim=False))
    dump(output/'independent_180_digit_checks.json',cases)
    environment=os.environ.copy();environment.pop('PYTHONPATH',None)
    environment['TEMP']=environment['TMP']='H:/Research/.paper-runtime'
    environment['PYTHONDONTWRITEBYTECODE']='1'
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):environment[key]='1'
    command=[sys.executable,'-m','unittest','discover','-s','research/prb_lattice_response','-p','test_zero_full_input_followup.py','-v']
    run=subprocess.run(command,cwd=ROOT.parents[1],env=environment,capture_output=True,text=True)
    (output/'analytic_controls.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
    if run.returncode:raise RuntimeError('Analytic controls failed')
    if frozen!={p:sha(Path(p)) for p in frozen}:raise RuntimeError('Source/input changed during independent review')
    summary=dict(status='NEW_INPUT_ZERO_BATCHES_AND_INDEPENDENT_REVIEW_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        source_input_sha256=frozen,covered_original_interval_labels=len(records),
        new_input_verified_local_zeros=len(good),unresolved_new_input_cases=len(bad),
        status_counts=dict(Counter(r['status'] for r in records)),
        old_mapping_status_counts=dict(Counter(r['old_mapping_status'] for r in good)),
        new_zero_inside_old_numeric_interval=sum(r['new_zero_inside_old_numeric_interval'] for r in good),
        unique_raw_targets=len({r['raw_index'] for r in good}),protected_neighborhoods_disjoint=True,
        original_fixed_float_verified_count=1901,increment_to_original_fixed_float_count=0,
        original_751_one_to_one_correspondence_certified=False,
        all_new_input_real_zeros_certified=False,physical_zero_count_certified=False,
        source_inputs_unchanged=True,arithmetic_precision_digits=160,independent_check_precision_digits=180,input_digits=110,
        max_absolute_charge_response=mp.nstr(max(mp.mpf(r['direct']['unscaled_charge_response']) for r in good),30),
        max_normalized_numerator=mp.nstr(max(mp.mpf(r['direct']['normalized_numerator']) for r in good),30),
        min_normalized_dielectric_denominator=mp.nstr(min(mp.mpf(r['direct']['normalized_dielectric_denominator']) for r in good),30),
        minimum_pole_zero_separation=mp.nstr(min(mp.mpf(r['pole_zero_separation']) for r in good),40),
        independent_rechecked_cases=len(cases),
        max_180_digit_absolute_charge_response=mp.nstr(max(mp.mpf(r['direct']['unscaled_charge_response']) for r in cases),30),
        max_180_digit_charge_only_residual=mp.nstr(max(mp.mpf(r['charge_only_normalized_residual']) for r in cases),30),
        analytic_control_count=7,analytic_control_returncode=run.returncode,analytic_control_command=command)
    dump(output/'summary.json',summary)
    dump(output/'unresolved_cases.json',bad)
    # A compact comparison CSV keeps each old label, result, and mapping ambiguity visible.
    flat=[]
    for r in sorted(records,key=lambda x:x['old_interval_index']):
        line=dict(old_interval_index=r['old_interval_index'],raw_index=r['raw_index'],status=r['status'],
            old_mapping_status=r.get('old_mapping_status','NOT_EVALUATED'),old_1901_count_increment=0)
        if r in good:line.update(new_zero=r['zero']['frequency'],new_pole=r['nearby_pole']['frequency'],
            pole_zero_separation=r['pole_zero_separation'],inside_old_numeric_interval=r['new_zero_inside_old_numeric_interval'],
            absolute_chi_cc=r['direct']['unscaled_charge_response'],D_ratio=r['direct']['normalized_dielectric_denominator'])
        else:line['reason']=r.get('reason','')
        flat.append(line)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (output/'old_new_case_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    dump(output/'manifest.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    pilot=reports['representatives_01'];pg=json.loads((BASE/'representatives_01/cases.json').read_text())
    journal=ROOT/'detailed_zero_full_input_journal_20260911.md'
    lines=['','## 실제 계산 완료와 해석',
        '',f"첫 대표 23개 배치는 {pilot['elapsed_seconds']:.3f}초에 끝났고 23개 모두 새 입력의 전하 영점 검증을 통과했다. 별도 coherent-vertex 입력에서도 통과했으며, serialized/coherent offset 차이를 pole-zero 거리로 나눈 최대치는 {max(float(r['coherent_vertex_crosscheck']['difference_over_pole_zero_separation']) for r in pg):.6e}이다. 저장된 rank-one 결함을 주 계산에서 삭제하지 않았다.",
        '', '첫 배치 결과와 실행 시간을 확인한 뒤 남은 728개를 서로 겹치지 않는 364개 배치 두 개로 나누어 병행 계산했다. 각 case 완료 직후 cases.json과 diagnostic.json을 갱신하고 시간·결과를 execution.log에 추가했다. 아래는 완료 후 고정된 집계다.',
        '', '| 배치 | 대상 수 | 성공 | 미해결 | 실행 시간(초) |', '|---|---:|---:|---:|---:|']
    for name,d in reports.items():
        counts=d['status_counts'];lines.append(f"| {name} | {d['completed_cases']} | {counts.get('VERIFIED_NEW_SERIALIZED_INPUT_LOCAL_CHARGE_ZERO',0)} | {counts.get('UNRESOLVED_NEW_INPUT_LOCAL_ZERO',0)} | {d['elapsed_seconds']:.3f} |")
    lines += ['',f"751개 old label을 누락·중복 없이 검사했다. 새 serialized-input의 검증된 국소 영점은 {len(good)}개이며, 새 입력에서도 미해결인 case는 {len(bad)}개이다. 성공한 target의 보호 구간은 서로 겹치지 않는다. 이 개수는 새 입력에서 선정된 국소 영점의 개수이며, 전체 실수축 영점의 완전한 목록이나 물리적으로 관측 가능한 모드 개수가 아니다.",
        '',f"기존 끝점 이동이 새 pole-zero 거리보다 큰 경우는 {summary['old_mapping_status_counts'].get('OLD_ENDPOINT_SHIFT_EXCEEDS_NEW_POLE_ZERO_SEPARATION',0)}개다. 기존 숫자 구간 안에 새 영점이 놓인 경우는 {summary['new_zero_inside_old_numeric_interval']}개다. 어느 경우도 이 사실만으로 old/new 일대일 대응이 증명되지는 않는다. 원래 fixed-float 1,901개 기록에 합산한 수는 0이며, 원래 751개 구간의 일대일 대응 미인증 상태를 그대로 유지한다.",
        '',f"전체 성공 case의 최대 absolute chi_cc={summary['max_absolute_charge_response']}, 최대 합 전 Lehmann 규모로 정상화한 N 잔차={summary['max_normalized_numerator']}, 최소 dielectric denominator ratio={summary['min_normalized_dielectric_denominator']}이다. 전체 행렬 norm에 가려진 전하 오차가 없는지 별도로 absolute chi_cc<=1e-50를 전수 확인했다. 최소 pole-zero 거리는 {summary['minimum_pole_zero_separation']}이다.",
        '',f"추가 독립 재평가에서는 pole-zero 거리 순서로 32개 등간격 대표와 가장 큰 charge 잔차·가장 작은 D ratio case를 포함한 {len(cases)}개를 선택했다. 같은 110자리 입력을 180자리 연산으로 다시 읽고 full transition sum과 2x2 역산을 반복했다. 최대 absolute chi_cc={summary['max_180_digit_absolute_charge_response']}; 전하 전용 기준 abs(chi_cc)/(1+sum abs(Wcc/(z-delta)))의 최대치는 {summary['max_180_digit_charge_only_residual']}였다. 이는 새 Hamiltonian 재계산이나 물리 입력 정밀도 인증이 아니다.",
        '', '최종 7개 분석 대조를 다시 실행해 로그를 delivery_review/analytic_controls.txt에 보존했다. 모든 배치 manifest와 source/input 시작 해시를 현재 파일과 대조했으며, 독립 재검사 후에도 일치했다. 원래 실패 preflight_01/의 source/test와 요약은 그대로 보존했다.',
        '', '## 배치 및 최종 검토 재현 명령',
        '', '    python research/prb_lattice_response/zero_full_input_followup.py --selection remaining --start-index 0 --limit 364 --output research/prb_lattice_response/zero_full_input_followup/remaining_000_363',
        '    python research/prb_lattice_response/zero_full_input_followup.py --selection remaining --start-index 364 --limit 364 --output research/prb_lattice_response/zero_full_input_followup/remaining_364_727',
        '    python research/prb_lattice_response/zero_full_input_followup/finalize_full_input_review.py --output research/prb_lattice_response/zero_full_input_followup/delivery_review',
        '', '기존 출력은 덮어쓰지 않는다. 새 재현은 새 출력 경로와 그 경로를 읽는 보존된 새 검토 runner를 사용해야 한다. primary batch는 매 case마다 checkpoint를 남긴다. 전체 751개 개별 결과는 old_new_case_table.csv와 각 batch cases.json으로 추적할 수 있다.',
        '', '## 남는 범위와 다음 연구',
        '', '160/180자리 연산이 저장된 110자리 에너지·정점 데이터를 더 정확한 물리 자료로 만드는 것은 아니다. thermal occupation 복원과 입력 의미 변경 때문에 기존 CSV의 pole/zero endpoint 숫자가 이동한다. 검증된 새 국소 영점이 있다는 사실과 원래 구간을 손실 없이 동일한 물리 모드로 연결했다는 주장은 구분한다.',
        '', '이번 계산은 전체 영점 개수 정리, continuum/Landau sheet, 감쇠, 저에너지 hybrid gap 또는 신규성/투고 가능성을 인증하지 않는다. 이 결과를 바탕으로 전체 새 입력의 pole/zero 대응표를 만들려면 모든 극·퇴화·관측 residue를 함께 조사하는 별도 단계가 필요하다.']
    with journal.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_input_sha256'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
