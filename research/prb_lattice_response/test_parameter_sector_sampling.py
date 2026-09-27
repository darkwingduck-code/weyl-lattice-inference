"""Independent numeric controls for the new parameter-sector sampler."""
from dataclasses import replace
import unittest
import mpmath as mp
import numpy as np
from lattice_response import Parameters,diagonalize,transitions,rpa_response
from magnetic_bloch_response import (magnetic_bloch_spectrum,magnetic_bloch_transitions,
                                     iter_magnetic_bloch_spectra)
from parameter_sector_sampling import (stable_fermi_difference,transition_vertices,
    weighted_transitions,two_sector_response,case_specs,BASE_Q)


class ParameterSectorSamplingTests(unittest.TestCase):
    def test_stable_df_matches_arbitrary_precision_saturation_and_sign(self):
        pairs=[(-700.,-699.),(-40.,-39.),(40.,41.),(699.,700.),
               (-1.e-10,1.e-10),(-8.,7.),(0.,0.),(41.,40.)]
        with mp.workdps(400):
            for x,y in pairs:
                ref=float(1/(1+mp.exp(mp.mpf(x)))-1/(1+mp.exp(mp.mpf(y))))
                got=float(stable_fermi_difference(x,y))
                self.assertLessEqual(abs(got-ref),8e-14*abs(ref)+1e-320)
                self.assertEqual(got,-float(stable_fermi_difference(y,x)))
                self.assertEqual(np.sign(got),np.sign(y-x))
        from scipy.special import expit
        self.assertEqual(float(expit(40.)-expit(39.)),0.)
        self.assertGreater(float(stable_fermi_difference(-40.,-39.)),0.)
        self.assertEqual(float(stable_fermi_difference(10.,10.)),0.)
        with self.assertRaises(ValueError):stable_fermi_difference(np.inf,0.)

    def test_all_pairs_vertices_and_legacy_weights_match_frozen_provider(self):
        for numerator,denominator in ((0,1),(1,3),(-1,3)):
            p=Parameters(Lx=3,Ly=3,Nkz=8,theta_x=.27,theta_y=-.39,theta_z=.18)
            spec=magnetic_bloch_spectrum(p,numerator,denominator)
            for shift in (-1,1):
                v=transition_vertices(spec,shift);got,stats=weighted_transitions(spec,v,.5,.04,legacy=True)
                frozen=magnetic_bloch_transitions(spec,.5,.04,shift)
                expected=spec.total_blocks*p.Nkz*(4*denominator)**2
                self.assertEqual(stats['raw_pairs'],expected)
                for key in ('delta','cc','aa','ca','conduction'):
                    np.testing.assert_allclose(getattr(got,key),getattr(frozen,key),rtol=2e-15,atol=2e-17)
                self.assertLess(v.vertex_error,2e-13)
                stable,_=weighted_transitions(spec,v,.5,.04)
                np.testing.assert_allclose(stable.density_response(.21+.015j),frozen.density_response(.21+.015j),rtol=2e-12,atol=2e-14)
            for invalid in (0,.5,8,-8,True):
                with self.assertRaises(ValueError):transition_vertices(spec,invalid)

    def test_batched_two_sector_response_matches_complex_direct_sums(self):
        p=Parameters(Lx=3,Ly=4,Nkz=8,theta_x=.21,theta_y=-.29,theta_z=.13)
        spec=magnetic_bloch_spectrum(p,0,1);tr,_=weighted_transitions(spec,transition_vertices(spec,-1),.25,.04)
        low=replace(tr,cc=tr.cc*tr.conduction,aa=tr.aa*tr.conduction,ca=tr.ca*tr.conduction)
        omega=np.array([0.,.05,.19,.37,.7])
        for eta in (.0075,.015,.03):
            for batch in (1,2,4):
                got=two_sector_response(tr,omega,eta,batch)
                expected=np.array([[t.density_response(complex(w,eta)) for w in omega] for t in (tr,low)])
                np.testing.assert_allclose(got,expected,rtol=2e-12,atol=2e-14)

    def test_streamed_bare_sum_matches_dense_then_single_rpa(self):
        p=Parameters(Lx=3,Ly=3,Nkz=8,theta_x=.27,theta_y=-.39,theta_z=.18)
        u=np.array([[.4,.07+.02j],[.07-.02j,.7]]);z=.29+.04j
        for numerator,denominator in ((0,1),(1,3),(-1,3)):
            bare=np.zeros((2,2),complex);wrong=np.zeros((2,2),complex);moment=np.zeros((2,2),complex)
            for spec in iter_magnetic_bloch_spectra(p,numerator,denominator,blocks_per_chunk=2):
                tr,_=weighted_transitions(spec,transition_vertices(spec,1),.5,.04)
                part=tr.density_response(z);bare+=part;wrong+=rpa_response(part,u);moment+=tr.moments()[0]
            dense=transitions(diagonalize(p,numerator*p.Lx*p.Ly//denominator),.5,.04,1)
            np.testing.assert_allclose(bare,dense.density_response(z),rtol=2e-11,atol=2e-13)
            np.testing.assert_allclose(rpa_response(bare,u),rpa_response(dense.density_response(z),u),rtol=2e-11,atol=2e-13)
            self.assertLess(np.max(abs(moment)),3e-14)
            self.assertGreater(np.max(abs(wrong-rpa_response(bare,u))),1e-5)

    def test_physical_q_and_both_windows_are_explicit(self):
        base=np.array([.05,.11,.2,.35]);specs=case_specs(base,192)
        self.assertEqual(len(specs),7)
        for name in ('qhalf','qdouble'):
            s=specs[name];factor=.5 if name=='qhalf' else 2.
            np.testing.assert_array_equal(s['fixed_lab_frequencies'],base)
            np.testing.assert_array_equal(s['scaled_q_frequencies'],base*factor)
            np.testing.assert_array_equal(s['omega'],np.unique(np.r_[0.,base,base*factor]))
            self.assertEqual(s['q'],factor*BASE_Q)
        with self.assertRaises(ValueError):case_specs(base,25)


if __name__=='__main__':unittest.main()
