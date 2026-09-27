import unittest
import numpy as np
from observation_design import response,closure,transfer,residual,profile
from run_validation import raw_fit

class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.w=np.linspace(.05,.35,17);w=self.w
        self.pi=np.zeros((len(w),2,2),complex)
        self.pi[:,0,0]=-.03+.005*w-.009j*w
        self.pi[:,1,1]=-.47+.08*w-.03j*w
        self.pi[:,0,1]=.016+.018j*w
        self.pi[:,1,0]=.011-.024j*w
        self.q=np.pi/12
        self.x=response(self.pi,self.q,.8,8.)
    def test_scalar_cofactor_nonhermitian(self):
        p=self.pi;a,b,c,d=p[:,0,0],p[:,0,1],p[:,1,0],p[:,1,1]
        v=.8/(2*np.sin(self.q/2))**2;detp=a*d-b*c;D=1-v*a-8*d+v*8*detp
        got=np.empty_like(p);got[:,0,0]=(a-8*detp)/D;got[:,1,1]=(d-v*detp)/D;got[:,0,1]=b/D;got[:,1,0]=c/D
        np.testing.assert_allclose(self.x,got,rtol=2e-14,atol=2e-16)
        self.assertGreater(np.max(abs(b-c.conj())),.003)
    def test_frequency_complex_transfer_cancels(self):
        d=.7*np.exp(.4j)*(1+.2j*self.w);s=1.3*np.exp(-.3j)*(1-.15*self.w)
        y=transfer(self.x,d,s)
        np.testing.assert_allclose(closure(y),closure(self.x),rtol=2e-14)
    def test_inverse_schur_and_reconstruct(self):
        ip=np.linalg.inv(self.pi);k=self.x[:,0,0]-closure(self.x)
        vc=ip[:,0,0]-1/k;g=vc*(2*np.sin(self.q/2))**2
        u=ip[:,1,1]-ip[:,1,0]*ip[:,0,1]/(ip[:,0,0]-vc-1/self.x[:,0,0])
        np.testing.assert_allclose(g,.8,rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(u,8.,rtol=3e-12,atol=3e-12)
    def test_cross_product_no_measured_aa_division(self):
        y=self.x.copy();y[:,1,1]=0
        v=residual({'d':self.x},{'d':y},['d'],'cross_product')
        self.assertTrue(np.all(np.isfinite(v)))
        with self.assertRaises(ValueError):closure(y)
    def test_independent_entry_transfer_does_not_cancel(self):
        y=transfer(self.x,.7+.3j,1.5-.2j);y[:,0,1]*=1.01
        np.testing.assert_allclose(closure(y),1.01*closure(self.x),rtol=2e-14)
        self.assertGreater(np.linalg.norm(closure(y)-closure(self.x)),1e-6)
    def test_complex_diagonal_nuisance(self):
        y=self.x.copy();y[:,1,1]*=.7+.3j
        v=residual({'d':self.x},{'d':y},['d'],'diag_separate_complex',{'d':self.w})
        self.assertLess(np.linalg.norm(v),1e-14)
        v=residual({'d':self.x},{'d':y},['d'],'diag_separate_real',{'d':self.w})
        self.assertGreater(np.linalg.norm(v),.1)
    def test_zero_mixing_information_failure(self):
        p=self.pi.copy();p[:,0,1]=p[:,1,0]=0;p[:,1,1]=-.47
        a=response(p,self.q,.8,4);b=response(p,self.q,.8,8)
        np.testing.assert_allclose(a[:,0,0],b[:,0,0],rtol=2e-14)
        np.testing.assert_array_equal(closure(a),0.)
        gain=(1-4*(-.47))/(1-8*(-.47))
        np.testing.assert_allclose(gain*a[:,1,1],b[:,1,1])
    def test_full_matrix_fit_factorized_complex_affine(self):
        t=(self.w-self.w.mean())/np.ptp(self.w)
        y=transfer(self.x,(.7+.3j)*(1+.1*t),(1.2-.4j)*(1+.1j*t))
        record=dict(pi=self.pi,q=self.q,omega=self.w)
        fit=raw_fit(record,y,.001,degree=1)
        self.assertTrue(fit['success']);self.assertLess(fit['rms'],1e-8)
        self.assertAlmostEqual(fit['g'],.8,places=7);self.assertAlmostEqual(fit['ua'],8.,places=6)
    def test_profile_common_anchor(self):
        data={'d':dict(pi=self.pi,q=self.q,omega=self.w)}
        fit=profile(data,{'d':self.x},['d'],8.,'cross_product')
        self.assertEqual(fit['g'],.8);self.assertLess(fit['rms'],1e-14)
if __name__=='__main__':unittest.main()