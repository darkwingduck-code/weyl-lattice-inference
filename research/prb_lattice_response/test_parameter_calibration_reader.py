"""Reader controls use explicitly synthetic 14-dataset fixtures, never production."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import parameter_calibration_audit as audit

class ParameterReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="prb-calibration-reader-")
        self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        with audit.BASE_FREQUENCIES_SOURCE.open(encoding="utf-8") as f:
            base=np.array([float(r["omega"]) for r in csv.DictReader(f) if r["sector"]=="total"])
        self.reports={};self.index=dict(complete_14_final_datasets=True,datasets=[],input_sha256={})
        for field in ("B0","Bplus"):
            folder=self.folder/(field+"_final");folder.mkdir()
            mesh=dict(Lx=72,Ly=72 if field=="B0" else 13,Nkz=384 if field=="B0" else 192,
                      mass=0.,wilson=1.,hopping=1.,exchange=1.,theta_x=0.,theta_y=0.,theta_z=0.)
            report=dict(status="PARAMETER_SECTOR_SAMPLING_COMPLETE",complete_torus=True,
                sources_unchanged=True,inputs_unchanged=True,field=field,mesh="final",parameters=mesh,
                source_sha256={"lattice_response.py":audit.sha(audit.ROOT/"lattice_response.py")},
                input_sha256={"low_energy_mode_audit/transition_sectors.csv":audit.sha(audit.BASE_FREQUENCIES_SOURCE)},
                cases={},output_sha256={})
            self.reports[field]=report
            for case,(factor,mu,eta) in audit.CASE_DESIGN.items():
                q=factor*np.pi/12;omega=np.unique(np.r_[0.,base,base*factor])
                record=dict(q=q,mu=mu,T=.04,eta=eta,density=2.,parameters=mesh,
                    fixed_lab_frequencies=base.tolist(),scaled_q_frequencies=(base*factor).tolist(),
                    checks={"identity_checks_pass":True})
                report["cases"][case]=record
                item=dict(**{k:v for k,v in record.items() if k not in ("parameters","checks")},
                    id=field+"_"+case,field=field,case=case,mesh=mesh,
                    csv=field+"_final/"+case+".csv",convergence_status="not_certified")
                rows=[]
                for w in omega:
                    pi=np.array([[.1-.03j,.001+.002j],[.001+.002j,-.47-.01j]])
                    chi=audit.prediction(pi,q,.8,8.)
                    row=dict(omega=float(w),q=q,mu=mu,T=.04,eta=eta,B=0. if field=="B0" else 2*np.pi/36,
                             density=2.,Lx=mesh["Lx"],Ly=mesh["Ly"],Nkz=mesh["Nkz"])
                    for prefix,matrix in (("bare",pi),("rpa",chi)):
                        for a in range(2):
                            for b in range(2):
                                row[f"{prefix}_{a}{b}_re"]=float(matrix[a,b].real)
                                row[f"{prefix}_{a}{b}_im"]=float(matrix[a,b].imag)
                    rows.append(row)
                audit.csv_write(folder/(case+".csv"),rows)
                self.index["datasets"].append(item)
        self.refresh()
    def refresh(self):
        for item in self.index["datasets"]:
            digest=audit.sha(self.folder/item["csv"]);item["csv_sha256"]=digest
            self.reports[item["field"]]["output_sha256"][item["case"]+".csv"]=digest
        for field,report in self.reports.items():
            path=self.folder/(field+"_final")/"diagnostic.json"
            path.write_text(json.dumps(report),encoding="utf-8")
            self.index["input_sha256"][field+"_final/diagnostic.json"]=audit.sha(path)
        self.path=self.folder/"calibration_inputs.json"
        self.path.write_text(json.dumps(self.index),encoding="utf-8")
    def mutate_csv(self,name,key,value):
        path=self.folder/(name+".csv")
        with path.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        for row in rows:row[key]=value
        audit.csv_write(path,rows);self.refresh()
    def test_valid_synthetic_design_both_windows(self):
        hashes={}
        for window in ("fixed_lab","scaled_q"):
            bare,truth,meta,inputs=audit.load_datasets(self.path,window,hashes)
            self.assertEqual(len(bare),14);self.assertTrue(hashes)
            self.assertEqual(len(audit.dataset_groups(meta)),27)
    def test_field_swap_with_valid_hashes_is_rejected(self):
        self.mutate_csv("Bplus_final/baseline","B",0.)
        with self.assertRaisesRegex(ValueError,"CSV B"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_wrong_qhalf_design_is_rejected(self):
        item=next(x for x in self.index["datasets"] if x["id"]=="B0_qhalf")
        item["q"]=np.pi/12;self.refresh()
        with self.assertRaisesRegex(ValueError,"design"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_wrong_scaled_window_is_rejected(self):
        item=next(x for x in self.index["datasets"] if x["id"]=="B0_qhalf")
        item["scaled_q_frequencies"]=item["fixed_lab_frequencies"];self.refresh()
        with self.assertRaisesRegex(ValueError,"scaled_q_frequencies"):
            audit.load_datasets(self.path,"scaled_q")
    def test_nonfinite_metadata_does_not_bypass_comparison(self):
        self.index["datasets"][0]["T"]=float("nan");self.refresh()
        with self.assertRaisesRegex(ValueError,"design"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_nonfinite_frequency_is_rejected_even_with_valid_hash(self):
        self.mutate_csv("B0_final/baseline","omega",float("nan"))
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_partial_torus_cannot_be_declared_final(self):
        self.reports["B0"]["complete_torus"]=False;self.refresh()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_stale_csv_hash_is_rejected(self):
        path=self.folder/"B0_final/baseline.csv"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError,"hash"):
            audit.load_datasets(self.path,"fixed_lab")
    def test_changed_file_cannot_replace_captured_digest(self):
        hashes={};path=self.folder/"probe.txt";path.write_text("old",encoding="utf-8")
        audit.read_checked(path,hashes);path.write_text("new",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"changed during run"):
            audit.read_checked(path,hashes)
    def test_corrupted_rpa_is_rejected_with_matching_manifest(self):
        self.mutate_csv("B0_final/baseline","rpa_00_re",999.)
        with self.assertRaisesRegex(ValueError,"truth"):
            audit.load_datasets(self.path,"fixed_lab")
    def make_coarse(self,field,case):
        import copy
        item=next(x for x in self.index["datasets"] if x["id"]==field+"_"+case)
        fp=self.folder/item["csv"]
        with fp.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        # Synthetic five-kernel fixture: only reader provenance is tested.
        for row in rows:
            for prefix,origin in (("low","bare"),("naive","rpa"),("matched","rpa")):
                for a in range(2):
                    for b in range(2):
                        for part in ("re","im"):
                            row[f"{prefix}_{a}{b}_{part}"]=row[f"{origin}_{a}{b}_{part}"]
        audit.csv_write(fp,rows)
        cr=copy.deepcopy(self.reports[field]);cr["mesh"]="coarse"
        mesh=copy.deepcopy(cr["parameters"])
        mesh.update(Lx=48 if field=="B0" else 72,Ly=48 if field=="B0" else 13,
                    Nkz=192 if field=="B0" else 96)
        cr["parameters"]=mesh
        record=copy.deepcopy(cr["cases"][case]);record["parameters"]=mesh;record["density"]=2.01
        cr["cases"]={case:record}
        cp=self.folder/(field+"_coarse")/(case+".csv");cp.parent.mkdir(exist_ok=True)
        for row in rows:
            for key in ("Lx","Ly","Nkz"):row[key]=mesh[key]
            row["density"]=record["density"]
        audit.csv_write(cp,rows)
        item["convergence_status"]="pass"
        item["convergence"]=dict(coarse_csv=str(cp.relative_to(self.folder)),coarse_sha256=audit.sha(cp),
            fine_sha256=audit.sha(fp),scaled_errors={k:0. for k in ("bare","low","rpa","naive","matched")},
            tolerance=.001,pass_all=True)
        self.refresh_coarse(cp,cr,item)
        return cp,cr,item

    def refresh_coarse(self,cp,cr,item):
        digest=audit.sha(cp);cr["output_sha256"]={cp.name:digest}
        item["convergence"]["coarse_sha256"]=digest
        (cp.parent/"diagnostic.json").write_text(json.dumps(cr),encoding="utf-8")
        self.refresh()

    def test_windows_manifest_keys_and_conflicting_aliases(self):
        self.index["input_sha256"]={str(Path(k)):v for k,v in self.index["input_sha256"].items()}
        self.path.write_text(json.dumps(self.index),encoding="utf-8")
        self.assertEqual(len(audit.load_datasets(self.path,"fixed_lab")[0]),14)
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            audit.normalized_manifest_keys({"B0_final/diagnostic.json":"a","B0_final\\diagnostic.json":"b"})

    def test_valid_coarse_contract_both_fields_and_different_density(self):
        self.make_coarse("B0","mu_low")
        self.make_coarse("Bplus","qhalf")
        hashes={}
        for window in ("fixed_lab","scaled_q"):
            _,_,meta,_=audit.load_datasets(self.path,window,hashes)
            self.assertEqual(meta["B0_mu_low"]["convergence_status"],"pass")
            self.assertEqual(meta["Bplus_qhalf"]["convergence_status"],"pass")
        self.assertTrue(any("B0_coarse" in k for k in hashes))

    def test_coarse_eta_mismatch_rejected_with_updated_hashes(self):
        cp,cr,item=self.make_coarse("Bplus","baseline")
        with cp.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        for row in rows:row["eta"]=.03
        audit.csv_write(cp,rows);self.refresh_coarse(cp,cr,item)
        with self.assertRaisesRegex(ValueError,"coarse CSV eta"):
            audit.load_datasets(self.path,"fixed_lab")

    def test_final_data_misfiled_as_coarse_rejected(self):
        cp,cr,item=self.make_coarse("B0","baseline")
        cr["mesh"]="final";self.refresh_coarse(cp,cr,item)
        with self.assertRaisesRegex(ValueError,"coarse diagnostic"):
            audit.load_datasets(self.path,"fixed_lab")

    def test_coarse_provenance_and_identity_failure_rejected(self):
        cp,cr,item=self.make_coarse("B0","baseline")
        cr["sources_unchanged"]=False;self.refresh_coarse(cp,cr,item)
        with self.assertRaisesRegex(ValueError,"coarse diagnostic"):
            audit.load_datasets(self.path,"fixed_lab")
        cr["sources_unchanged"]=True;cr["cases"]["baseline"]["checks"]["identity_checks_pass"]=False
        self.refresh_coarse(cp,cr,item)
        with self.assertRaisesRegex(ValueError,"Coarse full-band identity"):
            audit.load_datasets(self.path,"fixed_lab")


if __name__=="__main__":unittest.main()
