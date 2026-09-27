"""Portable replay wrapper; frozen production code and data remain unchanged.

A fresh --output receives copies of the same source bytes and new outputs.
Historical absolute provenance paths are resolved against this extracted repo,
then checked against their original SHA256; no references are rewritten.
"""
import argparse,json,shutil,sys,time
from pathlib import Path
import observation_design as obs
from observation_design import sha,dump
BUNDLE_STUDY=Path(__file__).resolve().parent
BUNDLE_REPO=BUNDLE_STUDY.parents[1]
BaseSnapshot=obs.Snapshot
class RebasedSnapshot(BaseSnapshot):
    def read(self,path,expected=None):
        p=Path(path).resolve()
        try:p.relative_to(BUNDLE_REPO)
        except ValueError:
            text=str(p).replace('\\','/')
            if '/research/' not in text:raise ValueError('Cannot rebase provenance path: '+text)
            p=BUNDLE_REPO/'research'/text.split('/research/',1)[1]
        return super().read(p,expected)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['checks','pilot','validation','robustness','all'],default='checks')
    a=ap.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output folder')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    for p in BUNDLE_STUDY.glob('*.py'):shutil.copy2(p,out/p.name)
    shutil.copy2(BUNDLE_STUDY/'frozen_design.json',out/'frozen_design.json')
    obs.Snapshot=RebasedSnapshot
    data,snap,err=obs.load_all()
    invariance,_=obs.invariant_checks(data)
    if a.stage=='checks':
        import csv
        source=list(csv.DictReader((BUNDLE_STUDY/'validation/full_profiles.csv').open(encoding='utf-8')))
        diffs=[]
        for u in (4.,8.,16.):
            for policy in ('diag_separate_complex','closure','cross_product'):
                ds=data['fixed_lab']['fine'];names=['Bplus_baseline'];truth=obs.models(ds,names,.8,8.)
                fit=obs.profile(ds,truth,names,u,policy)
                ref=next(r for r in source if r['dataset']=='Bplus_baseline' and r['window']=='fixed_lab' and float(r['ua'])==u and r['policy']==policy)
                diffs.append(abs(fit['rms']-float(ref['rms'])))
        if max(diffs)>2e-11:raise AssertionError('Representative profile mismatch')
        status='PORTABLE_READER_AND_PROFILE_CHECK_PASS'
    else:
        obs.ROOT=out
        if a.stage in ('all','pilot'):obs.pilot()
        if a.stage in ('all','validation'):
            import run_validation as v
            v.ROOT=out;v.load_all=obs.load_all;v.main()
        if a.stage in ('all','robustness'):
            import nuisance_robustness as r
            r.ROOT=out;r.load_all=obs.load_all;r.main()
        status='PORTABLE_REPLAY_COMPLETE';diffs=[]
    snap.check()
    dump(out/'portable_reproduction.json',dict(status=status,stage=a.stage,elapsed_seconds=time.perf_counter()-start,bundle_root=str(BUNDLE_REPO),output_root=str(out),source_sha256={p.name:sha(p) for p in BUNDLE_STUDY.glob('*.py')},input_sha256=snap.hashes,all_sources_resolved_within_bundle=all(Path(p).is_relative_to(BUNDLE_REPO) for p in snap.hashes),profile_check_count=len(diffs),maximum_profile_rms_difference=max(diffs,default=0.),algebra_max_abs=invariance,cached_response_max_abs=err,original_files_modified=False))
    print(status,flush=True)
if __name__=='__main__':main()