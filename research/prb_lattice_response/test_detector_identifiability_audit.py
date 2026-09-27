"""Independent controls for calibrated detector profiling."""
import unittest
import numpy as np
from detector_identifiability_audit import prediction,derivatives,residual_vector,numerical_profile

class DetectorIdentifiabilityTests(unittest.TestCase):
    def test_matrix_derivative_uses_ordered_complex_product(self):
        pi=np.array([[[.2+.1j,.03+.04j],[.07+.02j,-.1+.2j]]])
        q,g,ua=.37,.8,1.2;chi=prediction(pi,q,g,ua)
        dc,da=derivatives(chi,q,g,ua);step=1e-5
        finite=(prediction(pi,q,g,ua*np.exp(step))-prediction(pi,q,g,ua*np.exp(-step)))/(2*step)
        np.testing.assert_allclose(finite,da,rtol=1e-8,atol=1e-10)
        self.assertAlmostEqual(da[0,0,0],ua*chi[0,0,1]*chi[0,1,0],places=14)
        self.assertGreater(abs(da[0,0,0]-ua*abs(chi[0,0,1])**2),1e-6)

    def test_decoupled_charge_cannot_measure_axial_interaction(self):
        pi=np.array([np.diag([.1-.1j,-.3-.05j]),np.diag([.12-.13j,-.32-.09j])])
        a=prediction(pi,.3,.8,4);b=prediction(pi,.3,.8,12)
        self.assertEqual(np.linalg.norm(residual_vector(a,b,"charge_complex")),0.)
        self.assertGreater(np.linalg.norm(residual_vector(a,b,"charge_and_axial_complex")),.1)

    def test_phase_loss_and_joint_calibration_are_distinct(self):
        truth=np.array([np.diag([1+2j,3+4j]),np.diag([2+1j,4+3j])])
        model=truth.copy();model[:,0,0]+=1
        self.assertEqual(np.linalg.norm(residual_vector(model,truth,"charge_loss")),0.)
        self.assertGreater(np.linalg.norm(residual_vector(model,truth,"charge_complex")),.1)
        with self.assertRaises(ValueError):residual_vector(np.zeros_like(truth),np.zeros_like(truth),"charge_complex")

    def test_profile_records_endpoints_and_multiple_wells(self):
        def cost(x):return ((x-.4)*(x-1.2))**2+.001*(x-.4)**2
        p=numerical_profile(cost,np.linspace(.2,1.6,65))
        self.assertLess(abs(p["g"]-.4),1e-6);self.assertTrue(p["success"])
        self.assertGreaterEqual(p["refinement_candidates"],2);self.assertFalse(p["global_minimum_certified"])
        p=numerical_profile(lambda x:x,np.linspace(.2,1.6,65))
        self.assertEqual(p["g"],.2)
if __name__=="__main__":unittest.main()
