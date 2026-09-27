import unittest
import numpy as np
from uv_matching_audit import match_response,solve_response,method_responses,detector_metrics

class UVMatchingTests(unittest.TestCase):
    def test_noncommuting_complex_matrix_identity(self):
        low=np.array([[-.02-.03j,.006+.004j],[-.003+.007j,-.01-.05j]])
        high=np.array([[-.03-.001j,.009-.004j],[.002+.005j,-.45-.008j]])
        u=np.array([[11.,.2+.1j],[.2-.1j,8.]])
        result=match_response(low,high,u)
        np.testing.assert_allclose(result['response'],solve_response(low+high,u),rtol=2e-13,atol=2e-14)
        self.assertGreater(np.max(abs(result['effective_interaction'].imag)),.001)

    def test_zero_excluded_sector_returns_unmodified_response(self):
        low=np.array([[-.01-.04j,.003j],[.003j,-.02-.06j]])
        u=np.diag([10.,8.]);got=match_response(low,np.zeros((2,2)),u)
        np.testing.assert_allclose(got['response'],solve_response(low,u),atol=1e-14)
        np.testing.assert_array_equal(got['effective_interaction'],u)

    def test_external_vertices_are_indispensable(self):
        low=np.diag([-.01-.02j,-.03-.04j]);high=np.diag([-.002,-.47]);u=np.diag([12.,8.])
        got=match_response(low,high,u)
        self.assertGreater(abs(got['dressed_low'][1,1].imag/got['response'][1,1].imag),20)
        np.testing.assert_allclose(got['response'],solve_response(low+high,u),atol=1e-14)

    def test_constant_background_is_exact_when_background_is_constant(self):
        high=np.array([[-.003,.002],[.002,-.46]],complex)
        low=np.array([np.diag([-.01-.01j,-.02-.03j]),np.diag([.02-.04j,-.01-.02j])])
        values,_,_=method_responses(low+high,low,np.diag([12.,8.]),high)
        np.testing.assert_allclose(values['matched_constant'],values['full'],atol=2e-14)
        self.assertGreater(np.max(abs(values['naive_low_bare_U']-values['full'])),.01)

    def test_singular_background_is_not_regularized(self):
        with self.assertRaises(np.linalg.LinAlgError):match_response(np.eye(2)*.1,np.diag([.1,0.]),np.diag([10.,8.]))

if __name__=='__main__':unittest.main()
