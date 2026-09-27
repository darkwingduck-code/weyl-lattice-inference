import unittest
import numpy as np
from parameter_calibration_audit import prediction,profiled_residual,sensitivity,nuisance_columns,dataset_groups
class ParameterCalibrationTests(unittest.TestCase):
    def fixture(self,aa):
        return np.array([np.diag([.1-.03j,aa]),np.diag([.12-.04j,aa])])
    def test_constant_background_has_exact_gain_ambiguity_across_q(self):
        bare={n:self.fixture(-.47) for n in ("a","b")};meta={"a":{"q":.2},"b":{"q":.4}}
        truth={n:prediction(p,meta[n]["q"],.8,8) for n,p in bare.items()}
        model={n:prediction(p,meta[n]["q"],.8,4) for n,p in bare.items()}
        v,g=profiled_residual(model,truth,tuple(bare),"shared_axial_gain")
        self.assertLess(np.linalg.norm(v),1e-14)
        self.assertAlmostEqual(g["a"],(1+4*.47)/(1+8*.47),places=14)
        s=sensitivity(bare,truth,meta,tuple(bare),"shared_axial_gain")
        self.assertTrue(s["numerically_rank_deficient"])
        self.assertLess(s["norm_Ua_column_after"],1e-14)
    def test_changing_background_only_helps_with_shared_calibration(self):
        bare={"a":self.fixture(-.2),"b":self.fixture(-.8)}
        truth={n:prediction(p,.3,.8,8) for n,p in bare.items()}
        model={n:prediction(p,.3,.8,4) for n,p in bare.items()}
        shared,_=profiled_residual(model,truth,tuple(bare),"shared_axial_gain")
        separate,_=profiled_residual(model,truth,tuple(bare),"separate_axial_gains")
        self.assertGreater(np.linalg.norm(shared),.02)
        self.assertLess(np.linalg.norm(separate),1e-14)
    def test_projected_jacobian_matches_profiled_finite_difference(self):
        bare={"a":np.array([[[.2-.08j,.02+.01j],[.015+.03j,-.47-.03j]],
                            [[.1-.04j,.01+.03j],[.03+.02j,-.45-.02j]]])}
        truth={"a":prediction(bare["a"],.3,.8,8)}
        step=1e-5
        plus={"a":prediction(bare["a"],.3,.8,8*np.exp(step))}
        minus={"a":prediction(bare["a"],.3,.8,8*np.exp(-step))}
        vp,_=profiled_residual(plus,truth,("a",),"shared_axial_gain")
        vm,_=profiled_residual(minus,truth,("a",),"shared_axial_gain")
        expected=sensitivity(bare,truth,{"a":{"q":.3}},("a",),"shared_axial_gain")["norm_Ua_column_after"]
        self.assertAlmostEqual(np.linalg.norm((vp-vm)/(2*step)),expected,places=9)
    def test_zero_reference_and_negative_gain_boundary(self):
        truth={"a":self.fixture(-.4)};model={"a":truth["a"].copy()}
        model["a"][:,1,1]*=-1
        v,g=profiled_residual(model,truth,("a",),"shared_axial_gain")
        self.assertEqual(g["a"],0);self.assertGreater(np.linalg.norm(v),.5)
        with self.assertRaises(ValueError):
            profiled_residual({"a":np.zeros((2,2,2),complex)},{"a":np.zeros((2,2,2),complex)},("a",),"shared_axial_gain")
    def test_duplicate_dataset_does_not_invent_precision_gain(self):
        truth={"a":self.fixture(-.3)};model={"a":self.fixture(-.4)}
        v,_=profiled_residual(model,truth,("a",),"calibrated")
        truth["b"]=truth["a"].copy();model["b"]=model["a"].copy()
        vv,_=profiled_residual(model,truth,("a","b"),"calibrated")
        self.assertAlmostEqual(np.linalg.norm(v),np.linalg.norm(vv),places=14)
    def test_group_design_never_combines_regulator_values(self):
        metadata={}
        cases=("baseline","qhalf","qdouble","mu_low","mu_high","eta_low","eta_high")
        for field in ("B0","Bplus"):
            for case in cases:
                metadata[field+"_"+case]={"field":field,"case":case,"eta":.0075 if case=="eta_low" else .03 if case=="eta_high" else .015}
        groups=dataset_groups(metadata)
        self.assertEqual(len(groups),27)
        self.assertTrue(all(len({metadata[n]["eta"] for n in g["names"]})==1 for g in groups))
        metadata["B0_qhalf"]["eta"]=.2
        with self.assertRaisesRegex(ValueError,"eta"):
            dataset_groups(metadata)

    def test_distinct_nuisance_columns_are_orthogonal(self):
        truth={"a":self.fixture(-.3),"b":self.fixture(-.8)}
        c=nuisance_columns(truth,("a","b"),"separate_axial_gains")
        self.assertAlmostEqual(float(c[:,0]@c[:,1]),0.,places=15)
        self.assertEqual(c.shape[1],2)
if __name__=="__main__":unittest.main()
