import unittest
import numpy as np
from low_energy_mode_audit import analyze,track_channels,validate_sector_path,loss_peaks

class LowEnergyAuditTests(unittest.TestCase):
    def test_two_bare_peaks_are_not_dielectric_zeros(self):
        w=np.linspace(.02,1.2,601);z=w+.004j
        pi=np.zeros((len(w),2,2),complex)
        pi[:,0,0]=.02/(z-.3)-.02/(z+.3)
        pi[:,1,1]=.03/(z-.7)-.03/(z+.7)
        result=analyze(w,pi,.2,.004,coulomb=0,axial=0)
        self.assertTrue(result['summary']['peaks']['charge'])
        self.assertTrue(result['summary']['peaks']['axial'])
        np.testing.assert_allclose(result['singular'],1.,rtol=0,atol=1e-14)
        self.assertFalse(result['summary']['hybrid_gap_certified'])
        np.testing.assert_allclose(result['response'],pi,rtol=0,atol=1e-14)

    def test_offdiagonal_negative_control_changes_mixed_response(self):
        w=np.linspace(.02,.8,81);z=w+.02j
        pi=np.einsum('n,ab->nab',.01/(z-.4)-.01/(z+.4),np.array([[1,.3],[.3,.7]]))
        mixed=analyze(w,pi,.2,.02);removed=analyze(w,pi,.2,.02,remove_mixing=True)
        self.assertGreater(np.max(np.abs(mixed['response']-removed['response'])),1e-4)
        np.testing.assert_array_equal(removed['kernel'][:,0,1],0)

    def test_eigenvalue_labels_ignore_arbitrary_solver_order(self):
        angles=np.linspace(0,2*np.pi,60)
        matrices=np.array([[[2+np.cos(x),np.sin(x)],[np.sin(x),2-np.cos(x)]] for x in angles],complex)
        values,separation=track_channels(matrices)
        np.testing.assert_allclose(values,np.tile([1,3],(60,1)),atol=2e-14)
        self.assertAlmostEqual(separation,2)

    def test_sector_sum_matches_independent_complex_resolvents(self):
        self.assertTrue(validate_sector_path()['all_checks_pass'])

    def test_width_is_unresolved_at_sample_boundary(self):
        peaks=loss_peaks(np.arange(5.),np.array([4.,4.5,5.,4.5,4.]))
        self.assertEqual(len(peaks),1)
        self.assertIsNone(peaks[0]['half_height_width'])

if __name__=='__main__':unittest.main()
