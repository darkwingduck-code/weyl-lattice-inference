"""Independent negative controls for the bounded precision follow-up."""
import unittest
from decimal import Decimal,Context,localcontext
from types import SimpleNamespace
import numpy as np
from zero_precision_followup import DecimalRational,modal_bracket,deflated_sign_interval,D,PREC

class PrecisionControls(unittest.TestCase):
    def model(self):
        tr=SimpleNamespace(delta=np.array([.7,-.7,1.3,-1.3]),
                           cc=np.array([.08,-.08,.04,-.04]),
                           aa=np.zeros(4),ca=np.zeros(4,complex))
        return DecimalRational(tr,np.diag([.4,.3]))

    def test_laurent_matches_full_rational_at_very_small_offset(self):
        model=self.model()
        with localcontext(Context(prec=PREC)):
            center=D(.7);x=center+Decimal("1e-35")
            n,d,_,_=model.evaluate(x)
            ns,ds,_,_=model.evaluate(x,center)
            self.assertLess(abs(n/d-ns/ds),Decimal("1e-65"))

    def test_bare_multiplication_zero_is_rejected(self):
        model=self.model()
        with self.assertRaises(ValueError):
            model.evaluate(D(.7),D(.7))
        with self.assertRaises(ValueError):
            model.root(D(.7),laurent=True)

    def test_tiny_nonzero_weight_has_certified_sub_ulp_zero(self):
        with localcontext(Context(prec=PREC)):
            p=list(map(Decimal,("-2","-1","1","2")))
            w=list(map(Decimal,(".01",".02",".02","1e-50")))
            regular=sum((w[i]/(p[3]-p[i]) for i in range(3)),Decimal(0))
            estimate=-w[3]/regular
            bracket=modal_bracket(p,w,3,estimate,p[2],p[3])
            self.assertEqual(bracket["status"],"CERTIFIED_ROUNDED_WEIGHTED_MODAL_BRACKET")
            self.assertLess(abs(estimate),D(np.spacing(2.)))
            self.assertNotEqual(estimate,0)

    def test_dropping_tiny_weight_invalidates_certificate(self):
        p=list(map(Decimal,("-2","-1","1","2")))
        w=list(map(Decimal,(".01",".02",".02","0")))
        with self.assertRaises(ValueError):
            deflated_sign_interval(p,w,3,Decimal("-1e-50"))

    def test_raw_rank_rounding_is_not_silently_identified_with_factor(self):
        a=.1;b=a+4*np.spacing(a)
        tr=SimpleNamespace(delta=np.array([1.,-1.]),cc=np.array([a,-a]),
                           aa=np.array([b,-b]),ca=np.array([a,-a],complex))
        raw=DecimalRational(tr,np.diag([.4,.3]),"raw")
        factored=DecimalRational(tr,np.diag([.4,.3]),"factored")
        with localcontext(Context(prec=PREC)):
            n,d,_,_=raw.evaluate(Decimal(".3"))
            nf,df,_,_=factored.evaluate(Decimal(".3"))
            self.assertGreater(abs(n/d-nf/df),Decimal("1e-22"))

    def test_direct_refinement_preserves_true_zero(self):
        model=self.model()
        with localcontext(Context(prec=PREC)):
            x,record=model.root(Decimal("1.055"),bounds=(Decimal(".8"),Decimal("1.2")),laurent=True)
            n,d,_,_=model.evaluate(x)
            self.assertLess(abs(n/d),Decimal("1e-70"))
            self.assertTrue(Decimal("1.05")<x<Decimal("1.06"))

if __name__=="__main__":unittest.main()
