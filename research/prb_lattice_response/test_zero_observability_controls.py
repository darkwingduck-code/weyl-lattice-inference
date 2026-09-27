"""Independent algebra and failure controls for zero observability."""
import unittest
import numpy as np
from zero_observability_controls import components,scalar_response,regular_zero
class ZeroObservabilityTests(unittest.TestCase):
    def test_complex_cross_product_and_derivative(self):
        pi=np.array([[.2+.1j,.03+.04j],[.07+.02j,-.1+.2j]])
        dp=np.array([[.3-.2j,-.02+.05j],[.01+.03j,.2-.1j]])
        vc,ua=1.3,.8;actual,derivative,_,_=scalar_response(pi,dp,vc,ua)
        u=np.diag([vc,ua]);l=np.eye(2)-pi@u;r=np.eye(2)-u@pi
        self.assertAlmostEqual(actual,np.linalg.solve(l,pi)[0,0],places=14)
        self.assertAlmostEqual(derivative,(np.linalg.solve(l,dp)@np.linalg.inv(r))[0,0],places=14)
        wrong=pi[0,0]-ua*(pi[0,0]*pi[1,1]-abs(pi[0,1])**2)
        self.assertGreater(abs(wrong-components(pi,dp,ua)[0]),1e-5)
    def test_zero_and_slope_coulomb_invariance(self):
        pi=np.array([[-.1,np.sqrt(.075)],[np.sqrt(.075),.5]])
        dp=np.array([[.3,.07],[.07,-.2]])
        values=[scalar_response(pi,dp,v,.8) for v in [0.,.2,1.,30.]]
        for v in [0.,.2,1.,30.]: self.assertTrue(regular_zero(pi,dp,v,.8))
        self.assertLess(max(abs(x[0]) for x in values),1e-15)
        self.assertLess(max(abs(x[1]-values[0][1]) for x in values),1e-14)
    def test_coulomb_derivative_and_offdiagonal_counterexample(self):
        pi=np.array([[.3+.1j,.02+.07j],[.05-.01j,.12+.04j]])
        dp=np.zeros((2,2),complex);v,ua=.7,.8;h=1e-5
        cc=scalar_response(pi,dp,v,ua)[0]
        finite=(scalar_response(pi,dp,v+h,ua)[0]-scalar_response(pi,dp,v-h,ua)[0])/(2*h)
        self.assertLess(abs(finite-cc*cc),1e-10)
        full=np.linalg.solve(np.eye(2)-pi@np.array([[v,.2],[.2,ua]]),pi)[0,0]
        self.assertGreater(abs(full-cc),1e-4)
    def test_shared_factor_is_not_regular_zero(self):
        pi=np.diag([.2,1.25]).astype(complex);dp=np.eye(2)
        n,a,_,_=components(pi,dp,.8);self.assertEqual(abs(n)+abs(a),0.)
        for vc in [0.,.7]:
            self.assertFalse(regular_zero(pi,dp,vc,.8))
            self.assertGreater(abs(pi[0,0]/(1-vc*pi[0,0])),.19)
if __name__=="__main__": unittest.main()
