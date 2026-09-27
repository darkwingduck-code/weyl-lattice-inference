"""Independent 2x2-cofactor audit of frozen parameter/calibration profiles.

No production or lattice modules are imported. Coarse comparisons hold the
final fitted g fixed and refit only the declared nonnegative real gain(s).
They are not coarse g-profiles, uncertainty bounds, or confidence intervals.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
POLICIES = ("charge_only", "calibrated", "shared_axial_gain", "separate_axial_gains")
CHECK_TOL = 2e-11  # numerical algebra agreement, not a physical tolerance
FD_TOL = 5e-8     # absolute/scaled finite-difference comparison only
THRESHOLDS = (.001, .01)  # copied from the declared hypothetical RMS study

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Snapshot:
    def __init__(self):
        self.hashes = {}
    def read(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        value = hashlib.sha256(data).hexdigest()
        if expected is not None and value != expected:
            raise ValueError("Frozen file hash mismatch: " + str(path))
        if str(path) in self.hashes and self.hashes[str(path)] != value:
            raise ValueError("File changed during review: " + str(path))
        self.hashes[str(path)] = value
        return data.decode("utf-8")
    def check(self):
        bad = [p for p,h in self.hashes.items() if digest(p) != h]
        if bad:
            raise ValueError("Files changed during review: " + repr(bad))

def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))

def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty review table: " + str(path))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

def extract_matrix(rows, prefix):
    matrix = np.asarray([[
        [complex(float(r[f"{prefix}_{a}{b}_re"]), float(r[f"{prefix}_{a}{b}_im"]))
         for b in range(2)] for a in range(2)] for r in rows])
    if matrix.shape != (len(rows), 2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid matrix CSV")
    return matrix

def cofactor_response(pi, q, g, ua):
    """Direct scalar cofactors; no matrix inverse or production formula calls."""
    vc = float(g) / (2 * np.sin(float(q) / 2))**2
    a, b, c, d = pi[:,0,0], pi[:,0,1], pi[:,1,0], pi[:,1,1]
    det_pi = a*d - b*c
    det = 1 - vc*a - ua*d + vc*ua*det_pi
    if not np.all(np.isfinite(det)) or np.any(det == 0):
        raise ValueError("Singular/nonfinite RPA denominator at review sample")
    value = np.empty_like(pi, dtype=complex)
    value[:,0,0], value[:,1,1] = (a - ua*det_pi)/det, (d - vc*det_pi)/det
    value[:,0,1], value[:,1,0] = b/det, c/det
    if not np.all(np.isfinite(value)):
        raise ValueError("Nonfinite cofactor response")
    return value

def real_gain(model, truth, names):
    # Euclidean projection after normalizing each data set by its own truth.
    x = np.concatenate([model[n][:,1,1] / np.linalg.norm(truth[n][:,1,1]) for n in names])
    y = np.concatenate([truth[n][:,1,1] / np.linalg.norm(truth[n][:,1,1]) for n in names])
    denom = float(np.vdot(x,x).real)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Zero/nonfinite gain-design norm")
    return max(0., float(np.vdot(x,y).real) / denom)

def residual(model, truth, names, policy):
    if policy not in POLICIES or not names:
        raise ValueError("Invalid detector policy/group")
    channels = (0,) if policy == "charge_only" else (0,1)
    gains = {}
    if policy == "shared_axial_gain":
        gain = real_gain(model, truth, names)
        gains = {n:gain for n in names}
    elif policy == "separate_axial_gains":
        gains = {n:real_gain(model, truth, (n,)) for n in names}
    chunks = []
    for name in names:
        for channel in channels:
            reference = truth[name][:,channel,channel]
            normalizer = np.linalg.norm(reference)
            if not np.isfinite(normalizer) or normalizer <= 0:
                raise ValueError("Zero/nonfinite reference-channel norm")
            gain = gains.get(name,1.) if channel == 1 else 1.
            delta = (gain*model[name][:,channel,channel]-reference)/normalizer
            chunks.append(delta)
    # One real block and one imaginary block: a harmless orthogonal permutation
    # of the production layout, with the same L2 inner products and singulars.
    values = np.concatenate(chunks)/np.sqrt(len(names)*len(channels))
    vector = np.concatenate((values.real, values.imag))
    if not np.all(np.isfinite(vector)):
        raise ValueError("Nonfinite residual")
    return vector, gains

def models_for(data, names, g, ua):
    return {n:cofactor_response(data[n]["pi"],data[n]["q"],g,ua) for n in names}

def truths_for(data):
    return {n:cofactor_response(v["pi"],v["q"],.8,8.) for n,v in data.items()}


def load_windows(index_path, snapshot):
    index = json.loads(snapshot.read(index_path))
    if index.get("complete_14_final_datasets") is not True or len(index["datasets"]) != 14:
        raise ValueError("Need the completed 14-dataset index")
    data = {w:{"fine":{}, "coarse":{}} for w in ("fixed_lab","scaled_q")}
    cached_error = 0.
    for item in index["datasets"]:
        name = item["id"]
        for level, rel, expected in (
            ("fine",item["csv"],item["csv_sha256"]),
            ("coarse",item["convergence"]["coarse_csv"],item["convergence"]["coarse_sha256"])):
            path = (index_path.parent/rel).resolve()
            rows = read_csv(snapshot.read(path,expected))
            omega = np.asarray([float(r["omega"]) for r in rows])
            if not np.all(np.isfinite(omega)):
                raise ValueError("Nonfinite frequency")
            for window in data:
                wanted = np.asarray(item[window+"_frequencies"])
                ids = [int(np.argmin(abs(omega-w))) for w in wanted]
                if len(ids) != 45 or len(set(ids)) != 45 or max(abs(omega[i]-w) for i,w in zip(ids,wanted)) > 2e-14:
                    raise ValueError("Missing independent review frequency")
                selected = [rows[i] for i in ids]
                pi = extract_matrix(selected,"bare")
                saved = extract_matrix(selected,"rpa")
                q = float(item["q"])
                got = cofactor_response(pi,q,.8,8.)
                cached_error = max(cached_error,float(np.max(abs(saved-got))))
                data[window][level][name] = {
                    "pi":pi,"q":q,"omega":wanted,"item":item,
                    "source_csv":str(path),"saved_rpa":saved}
    if cached_error > CHECK_TOL:
        raise ValueError("Stored truth differs from independent cofactors")
    return index,data,cached_error

def local_fd(data, truth, names, policy, step=1e-4):
    # Differentiate the fully reprofiled residual. No analytical matrix
    # derivatives or nuisance QR from the production module are reused.
    columns = []
    for coordinate in (0,1):
        samples = {}
        for multiple in (-2,-1,1,2):
            g = .8*np.exp(multiple*step) if coordinate == 0 else .8
            ua = 8.*np.exp(multiple*step) if coordinate == 1 else 8.
            samples[multiple] = residual(models_for(data,names,g,ua),truth,names,policy)[0]
        columns.append((samples[-2]-8*samples[-1]+8*samples[1]-samples[2])/(12*step))
    jac = np.column_stack(columns)
    return np.linalg.svd(jac,compute_uv=False), float(np.linalg.norm(jac[:,1]))

def truth_grid_table(windows):
    rows = []
    for window, levels in windows.items():
        fine, coarse = truths_for(levels["fine"]), truths_for(levels["coarse"])
        for name in sorted(fine):
            item = levels["fine"][name]["item"]
            row = dict(window=window,dataset=name,field=item["field"],case=item["case"],
                q=item["q"],mu=item["mu"],eta=item["eta"],
                declared_five_kernel_grid_comparison=item["convergence_status"],
                normalization="L2(fine-coarse)/L2(fine); no new physical tolerance")
            for channel,label in ((0,"cc"),(1,"aa")):
                den = np.linalg.norm(fine[name][:,channel,channel])
                row["rpa_"+label+"_relative_l2"] = float(np.linalg.norm(
                    fine[name][:,channel,channel]-coarse[name][:,channel,channel])/den)
            row["rpa_max_abs_over_one_plus_fine"] = float(np.max(
                abs(fine[name]-coarse[name])/(1+abs(fine[name]))))
            rows.append(row)
    return rows

def self_test():
    checks = {}
    pi = np.array([[[.12-.04j,.02+.01j],[.03-.02j,-.47-.03j]],
                   [[.21-.06j,.01-.03j],[.02+.04j,-.36-.02j]]])
    q,g,ua = .3,.8,8.
    got = cofactor_response(pi,q,g,ua)
    u = np.diag([g/(2*np.sin(q/2))**2,ua])
    direct = np.stack([np.linalg.solve(np.eye(2)-p@u,p) for p in pi])
    assert np.max(abs(got-direct)) < 2e-14
    checks["complex_offdiagonal_cofactor_vs_direct_linear_solve"] = True
    diagonal = pi.copy();diagonal[:,0,1]=0;diagonal[:,1,0]=0
    d = cofactor_response(diagonal,q,g,ua)
    assert np.max(abs(d[:,1,1]-diagonal[:,1,1]/(1-ua*diagonal[:,1,1]))) < 1e-14
    checks["independent_decoupled_scalar_screening"] = True
    truth = {"a":got,"b":2*got}
    model = {"a":got.copy(),"b":2*got.copy()}
    for name in model:model[name][:,1,1]*=2
    vector,gains = residual(model,truth,("a","b"),"shared_axial_gain")
    assert np.linalg.norm(vector)<1e-14 and abs(gains["a"]-.5)<1e-14
    checks["known_shared_gain_and_dataset_normalization"] = True
    negative = {"a":got.copy()};negative["a"][:,1,1]*=-1
    vector,gains = residual(negative,{"a":got},("a",),"shared_axial_gain")
    assert gains["a"]==0 and abs(np.linalg.norm(vector)-1/np.sqrt(2))<1e-14
    checks["negative_gain_boundary_preserves_mismatch"] = True
    pi0 = np.array([np.diag([.1-.03j,-.47]),np.diag([.12-.04j,-.47])])
    data = {"a":{"pi":pi0,"q":.2},"b":{"pi":pi0,"q":.4}}
    truth = truths_for(data)
    singular,norm = local_fd(data,truth,("a","b"),"shared_axial_gain")
    assert norm<1e-10
    checks["finite_difference_removes_exact_constant_background_nuisance"] = True
    return checks


def main(production, index_path, output):
    started = time.perf_counter()
    snapshot = Snapshot()
    snapshot.read(Path(__file__).resolve())
    production = production.resolve();index_path=index_path.resolve();output=output.resolve()
    report = json.loads(snapshot.read(production/"diagnostic.json"))
    if report.get("status")!="PARAMETER_CALIBRATION_COMPLETE":
        raise ValueError("Production is not complete")
    if report.get("inputs_sources_unchanged") is not True:
        raise ValueError("Production did not preserve its snapshot")
    manifest = json.loads(snapshot.read(production/"manifest.json"))
    for name,expected in manifest.items():
        snapshot.read(production/name,expected)
    for path,expected in report["source_input_sha256"].items():
        snapshot.read(path,expected)
    tests = self_test()
    groups = json.loads(snapshot.read(production/"groups.json"))
    profiles = read_csv(snapshot.read(production/"profiles.csv"))
    sensitivities = read_csv(snapshot.read(production/"local_sensitivity.csv"))
    tolerance_sets = json.loads(snapshot.read(production/"tolerance_sets.json"))
    if (len(groups),len(profiles),len(sensitivities),len(tolerance_sets)) != (36,6174,126,252):
        raise ValueError("Unexpected completed production inventory")
    if report["profile_count"]!=len(profiles) or report["group_window_count"]!=len(groups):
        raise ValueError("Production counts disagree")
    index,windows,cache_error = load_windows(index_path,snapshot)
    # All report artifacts already exist and are complete before output creation.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty independent-review output")
    output.mkdir(parents=True,exist_ok=True)
    group_map = {(g["window"],g["id"]):g for g in groups}
    if len(group_map)!=36:
        raise ValueError("Duplicate group/window records")
    truths = {w:{level:truths_for(data) for level,data in levels.items()}
              for w,levels in windows.items()}
    expected_rows = set()
    for group in groups:
        names = tuple(group["names"])
        if len(set(names))!=len(names) or not names:
            raise ValueError("Repeated or empty dataset group")
        items = [windows[group["window"]]["fine"][n]["item"] for n in names]
        if len({v["eta"] for v in items})!=1:
            raise ValueError("Production combined different numerical regulators")
        policies = POLICIES[:3] if len(names)==1 else POLICIES
        for policy in policies:
            for i in range(49):
                expected_rows.add((group["window"],group["id"],policy,.5*i))
    if len(expected_rows)!=6174:
        raise ValueError("Group membership does not imply the declared inventory")
    max_cost_error=max_gain_error=max_gain_scaled_error=0.
    checks=[];coarse_rows=[];seen=set();accepted={}
    representative_groups = {
        "single_B0_baseline","single_Bplus_baseline","field_pair_baseline",
        "q_scan_B0","q_scan_Bplus","q_scan_both",
        "mu_scan_B0","mu_scan_Bplus","mu_scan_both"}
    for i,row in enumerate(profiles):
        window,group_id,policy=row["window"],row["group"],row["mode"]
        ua,g=float(row["ua"]),float(row["best_g"])
        key=(window,group_id,policy,ua)
        if key in seen or key not in expected_rows:
            raise ValueError("Duplicate/unexpected profile")
        seen.add(key)
        names=tuple(group_map[window,group_id]["names"])
        model=models_for(windows[window]["fine"],names,g,ua)
        vector,gains=residual(model,truths[window]["fine"],names,policy)
        cost=float(np.linalg.norm(vector))
        stored=float(row["normalized_mismatch"])
        error=abs(cost-stored)
        max_cost_error=max(max_cost_error,error)
        saved_gains=json.loads(row["axial_gains"])
        if set(saved_gains)!=set(gains):
            raise ValueError("Gain membership mismatch")
        gain_error=max((abs(float(saved_gains[n])-v) for n,v in gains.items()),default=0.)
        gain_scaled=max((abs(float(saved_gains[n])-v)/(1+abs(v)) for n,v in gains.items()),default=0.)
        max_gain_error=max(max_gain_error,gain_error)
        max_gain_scaled_error=max(max_gain_scaled_error,gain_scaled)
        if error/(1+abs(stored))>CHECK_TOL or gain_scaled>CHECK_TOL:
            raise ValueError("Independent cofactor/gain/RMS mismatch at "+repr(key))
        declared_pass=all(windows[window]["fine"][n]["item"]["convergence_status"]=="pass" for n in names)
        if row["all_declared_grid_comparisons_pass"]!=str(declared_pass):
            raise ValueError("Grid-comparison flag mismatch")
        checks.append(dict(window=window,group=group_id,mode=policy,ua=ua,final_fitted_g=g,
            stored_mismatch=stored,independent_mismatch=cost,mismatch_absolute_error=error,
            gain_absolute_error=gain_error,independent_axial_gains=json.dumps(gains,sort_keys=True),
            all_declared_grid_comparisons_pass=declared_pass))
        for threshold in THRESHOLDS:
            accepted.setdefault((window,group_id,policy,threshold),[])
            if cost<=threshold:accepted[window,group_id,policy,threshold].append(ua)
        coarse_model=models_for(windows[window]["coarse"],names,g,ua)
        cv,cg=residual(coarse_model,truths[window]["coarse"],names,policy)
        ccost=float(np.linalg.norm(cv))
        coarse_rows.append(dict(window=window,group=group_id,mode=policy,ua=ua,final_fitted_g=g,
            fine_independent_mismatch=cost,coarse_at_final_g_mismatch=ccost,
            coarse_minus_fine_mismatch=ccost-cost,
            fine_axial_gains=json.dumps(gains,sort_keys=True),coarse_refitted_axial_gains=json.dumps(cg,sort_keys=True),
            all_declared_grid_comparisons_pass=declared_pass,
            sampled_classification_differs_at_0p1percent=bool((cost<=.001)!=(ccost<=.001)),
            sampled_classification_differs_at_1percent=bool((cost<=.01)!=(ccost<=.01)),
            coarse_g_reoptimized=False,statistical_confidence_interval=False))
        if (i+1)%1000==0:
            print(f"Reviewed {i+1}/{len(profiles)} fixed-fit hypotheses",flush=True)
    if seen!=expected_rows:
        raise ValueError("Profile inventory incomplete")
    tolerance_checks=[]
    tkeys=set()
    for item in tolerance_sets:
        key=(item["window"],item["group"],item["mode"],float(item["threshold"]))
        if key in tkeys or key not in accepted:
            raise ValueError("Duplicate/unexpected tolerance set")
        tkeys.add(key)
        got=sorted(accepted[key])
        if got!=item["accepted_sampled_ua"]:
            raise ValueError("Independent sampled-tolerance set differs: "+repr(key))
        tolerance_checks.append(dict(window=key[0],group=key[1],mode=key[2],threshold=key[3],
            accepted_sampled_ua=got,matches=True,interpretation="Sampled grid only; not a confidence interval"))
    if tkeys!=set(accepted):
        raise ValueError("Tolerance inventory incomplete")


    fd_rows=[];max_fd_error=max_fd_step_change=0.;skeys=set()
    for item in sensitivities:
        key=(item["window"],item["group"],item["mode"])
        if key in skeys:
            raise ValueError("Duplicate sensitivity record")
        skeys.add(key)
        window,group_id,policy=key
        names=tuple(group_map[window,group_id]["names"])
        data=windows[window]["fine"];truth=truths[window]["fine"]
        singular,after=local_fd(data,truth,names,policy,1e-4)
        finer,after_finer=local_fd(data,truth,names,policy,5e-5)
        before_policy="charge_only" if policy=="charge_only" else "calibrated"
        _,before=local_fd(data,truth,names,before_policy,5e-5)
        expected=np.array([float(item["largest_singular"]),float(item["smallest_singular"]),
                           float(item["norm_Ua_column_before"]),float(item["norm_Ua_column_after"])])
        got=np.array([finer[0],finer[1],before,after_finer])
        error=float(np.max(abs(got-expected)/(1+abs(expected))))
        step_change=float(max(np.max(abs(finer-singular)),abs(after_finer-after)))
        max_fd_error=max(max_fd_error,error);max_fd_step_change=max(max_fd_step_change,step_change)
        if error>FD_TOL:
            raise ValueError("Finite-difference local sensitivity mismatch at "+repr(key))
        fd_rows.append(dict(window=window,group=group_id,mode=policy,
            stored_largest_singular=expected[0],fd_largest_singular=got[0],
            stored_smallest_singular=expected[1],fd_smallest_singular=got[1],
            stored_Ua_column_before=expected[2],fd_Ua_column_before=got[2],
            stored_Ua_column_after=expected[3],fd_Ua_column_after=got[3],
            maximum_scaled_agreement_error=error,
            maximum_step_change=step_change,coarse_fd_step=1e-4,fine_fd_step=5e-5,
            exact_rank_certified=False))
    expected_skeys={(k[0],k[1],k[2]) for k in expected_rows}
    if skeys!=expected_skeys:
        raise ValueError("Sensitivity inventory incomplete")
    truth_rows=truth_grid_table(windows)
    representatives=[r for r in coarse_rows if r["ua"]==4. and r["group"] in representative_groups]
    snapshot.check()
    write_csv(output/"profile_checks.csv",checks)
    write_json(output/"tolerance_set_checks.json",tolerance_checks)
    write_csv(output/"local_sensitivity_finite_difference.csv",fd_rows)
    write_csv(output/"coarse_at_final_fit.csv",coarse_rows)
    write_csv(output/"truth_grid_relative_differences.csv",truth_rows)
    write_csv(output/"representative_ua4_comparisons.csv",representatives)
    snapshot.check()
    result=dict(status="INDEPENDENT_PARAMETER_CALIBRATION_REVIEW_PASS",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,
        self_tests=tests,profile_count=len(checks),tolerance_set_count=len(tolerance_checks),
        local_sensitivity_count=len(fd_rows),truth_grid_rows=len(truth_rows),
        representative_ua4_rows=len(representatives),coarse_at_final_fit_count=len(coarse_rows),
        maximum_cached_rpa_cofactor_absolute_error=cache_error,
        maximum_profile_rms_absolute_error=max_cost_error,
        maximum_gain_absolute_error=max_gain_error,
        maximum_gain_scaled_error=max_gain_scaled_error,
        maximum_local_finite_difference_scaled_error=max_fd_error,
        maximum_local_finite_difference_step_change=max_fd_step_change,
        algebra_check_tolerance=CHECK_TOL,finite_difference_check_tolerance=FD_TOL,
        declared_grid_comparison_counts={s:sum(i["convergence_status"]==s for i in index["datasets"])
                                        for s in ("pass","fail","not_certified")},
        source_input_sha256=snapshot.hashes,inputs_sources_unchanged=True,
        production_global_minimum_certified=False,coarse_g_reoptimized=False,
        coarse_gain_reoptimized_at_final_g=True,statistical_confidence_intervals=False,
        scope=[
            "Independent cofactors recompute response, real gain and RMS only at saved final fitted g and Ua.",
            "No production, sampler, lattice, gain or derivative module is imported.",
            "Coarse comparisons refit gain using coarse truth at the same saved final g; not coarse g-profiles or rigorous bounds.",
            "Truth-grid relative errors divide by the fine truth channel norm; no new physical tolerance is imposed.",
            "Declared 0.001 maximum-entry/(1+fine) grid comparison pass does not certify relative detector precision.",
            "Finite-difference singular values test local numerical derivatives, not exact rank, Fisher coverage, or global inverse uniqueness.",
            "Different eta cases remain separate numerical studies; no different eta values are combined.",
            "All outputs are finite-grid, finite-regulator model calculations; no material detector or low-energy gap is established."])
    write_json(output/"diagnostic.json",result)
    write_json(output/"manifest.json",{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production",type=Path,default=ROOT/"parameter_calibration_audit/run_01")
    parser.add_argument("--index",type=Path,default=ROOT/"parameter_sector_sampling/calibration_inputs.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2))
    else:
        if args.output is None:parser.error("--output is required for a review")
        main(args.production,args.index,args.output)

