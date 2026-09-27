"""Analytic controls independent of the lattice cofactor implementation."""
import unittest
import mpmath as mp
from zero_full_input_followup import FullInput,determinant,matrix

class FullInputZeroControls(unittest.TestCase):
    def setUp(self):self.previous=mp.mp.dps;mp.mp.dps=160
    def tearDown(self):mp.mp.dps=self.previous
    def oscillators(self,energies,strengths,vc=1,ua=8,phase=True):
        alpha=mp.mpc('.3','.4') if phase else mp.mpf(0)
        d=[];w=[[],[],[],[]]
        for e,a in zip(energies,strengths):
            for s in (1,-1):
                d.append(s*e);ca=s*a*mp.conj(alpha)
                for column,x in zip(w,(s*a,s*a*abs(alpha)**2,ca.real,ca.imag)):column.append(x)
        return FullInput(d,w,mp.mpf(vc),mp.mpf(ua))
    def test_two_oscillator_zero_has_known_closed_form(self):
        e1,e2=mp.mpf('.6'),mp.mpf('1.4');a1,a2=mp.mpf('1e-30'),mp.mpf('.07')
        model=self.oscillators([e1,e2],[a1,a2]);t,detail=model.root(0)
        expected=mp.sqrt((e1*a1*e2**2+e2*a2*e1**2)/(e1*a1+e2*a2))
        self.assertLess(abs(e1+t-expected),mp.mpf('1e-140'))
        self.assertTrue(detail['two_sided_sign_change']);model.direct(0,t)
    def test_charge_zero_independent_of_vc_in_complex_vertex_control(self):
        e=[mp.mpf('.6'),mp.mpf('1.4')];a=[mp.mpf('1e-30'),mp.mpf('.07')]
        low=self.oscillators(e,a,vc='.1');high=self.oscillators(e,a,vc='100')
        self.assertEqual(low.root(0)[0],high.root(0)[0])
    def test_one_oscillator_has_no_spurious_local_zero(self):
        model=self.oscillators([mp.mpf('.6')],[mp.mpf('1e-30')])
        with self.assertRaises(ValueError):model.root(0)
        with self.assertRaisesRegex(ValueError,'bare'):model.direct(0,mp.mpf(0))
    def test_literal_rank_defect_is_not_silently_projected_out(self):
        model=self.oscillators([mp.mpf('.6'),mp.mpf('1.4')],[mp.mpf('1e-30'),mp.mpf('.07')])
        model.weights[1][0]+=mp.mpf('1e-140')
        value=model.scaled(0,mp.mpf(0),False)
        self.assertNotEqual(value['weight_determinant'],0)
        self.assertEqual(value['N2'],-model.ua*value['weight_determinant'])
    def test_degenerate_bare_target_is_rejected(self):
        model=self.oscillators([mp.mpf('.6'),mp.mpf('.6')],[mp.mpf('.01'),mp.mpf('.02')])
        with self.assertRaisesRegex(ValueError,'degeneracy'):model.root(0)
    def test_D_N_common_zero_not_accepted_as_regular_charge_zero(self):
        e,a,u=mp.mpf('.6'),mp.mpf('.04'),mp.mpf(4)
        model=FullInput([e,-e,e,-e],[[a,-a,0,0],[0,0,a,-a],[0,0,0,0],[0,0,0,0]],u,u)
        pole=mp.sqrt(e**2+2*e*a*u)
        with self.assertRaisesRegex(ValueError,'dielectric'):model.direct(0,pole-e)
    def test_scaled_analytic_derivative_matches_complex_step(self):
        model=self.oscillators([mp.mpf('.6'),mp.mpf('1.4')],[mp.mpf('1e-30'),mp.mpf('.07')])
        t=mp.mpf('2e-30');h=mp.mpf('1e-80');value=model.scaled(0,t)
        shifted=model.scaled(0,t+1j*h,False)
        for key in ('N','D'):
            expected=shifted[key+'2'].imag/h
            self.assertLess(abs(expected-value[key+'_prime'])/max(abs(expected),mp.mpf('1e-300')),mp.mpf('1e-90'))

if __name__=='__main__':unittest.main()
