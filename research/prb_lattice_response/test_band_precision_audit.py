"""Independent negative controls for band/occupation precision diagnostics."""
import unittest
import mpmath as mp
import numpy as np
from band_precision_audit import stable_df, exact, projector_strength, block_hamiltonian, float_matrices
from bloch_response import bloch_spectrum
from lattice_response import Parameters

class BandPrecisionTests(unittest.TestCase):
    def test_saturated_occupation_difference_is_not_zero(self):
        with mp.workdps(80):
            ei,ej,mu,temp=map(mp.mpf,('-12','-11','.5','.04'))
            self.assertEqual(1/(1+mp.exp((ei-mu)/temp))-1/(1+mp.exp((ej-mu)/temp)),0)
            stable=stable_df(ei,ej,mu,temp)
            self.assertGreater(stable,0)
            self.assertLess(stable,mp.mpf('1e-100'))
            self.assertLess(abs(stable+stable_df(ej,ei,mu,temp))/stable,mp.mpf('1e-70'))
    def test_opposite_energy_occupation_identity(self):
        with mp.workdps(80):
            a,b,mu,t=map(mp.mpf,('-7','-6','.5','.04'))
            self.assertLess(abs(stable_df(a,b,mu,t)-stable_df(-b,-a,-mu,t))/abs(stable_df(a,b,mu,t)),mp.mpf('1e-70'))
    def test_equal_energies_exact_zero_and_sign(self):
        with mp.workdps(80):
            self.assertEqual(stable_df(mp.mpf(1),mp.mpf(1),mp.mpf(0),mp.mpf('.04')),0)
            self.assertLess(stable_df(mp.mpf(2),mp.mpf(1),mp.mpf(0),mp.mpf('.04')),0)
    def test_literal_float_is_not_decimal_literal(self):
        with mp.workdps(80):
            self.assertNotEqual(exact(.1),mp.mpf('.1'))
            self.assertEqual(exact(.1),mp.mpf(3602879701896397)/36028797018963968)
    def test_degenerate_projector_sum_survives_basis_rotation(self):
        with mp.workdps(80):
            identity=mp.eye(4); op=mp.diag([1,2,3,4]); r=mp.eye(4)
            angle=mp.mpf('.413'); r[0,0]=r[1,1]=mp.cos(angle); r[0,1]=-mp.sin(angle); r[1,0]=mp.sin(angle)
            reference=projector_strength(identity,identity,op,[0,1],[0,1])
            rotated=projector_strength(r,identity,op,[0,1],[0,1])
            self.assertLess(abs(reference-rotated),mp.mpf('1e-70'))
            self.assertNotEqual(abs((r[:,0].H*op*identity[:,0])[0])**2,1)
    def test_trig_rounding_and_k_rounding_are_distinct_inputs(self):
        with mp.workdps(80):
            spec=bloch_spectrum(Parameters(Lx=3,Ly=3,Nkz=24,theta_x=.21,theta_y=.37,theta_z=.19))
            arrays=float_matrices(spec)
            a=block_hamiltonian(spec,arrays,0,0,'literal_trig')
            b=block_hamiltonian(spec,arrays,0,0,'literal_k')
            self.assertGreater(mp.norm(a-b,p=2),0)
            self.assertLess(mp.norm(a-b,p=2),mp.mpf('1e-12'))
            self.assertEqual(mp.norm(a-a.H,p=2),0)

if __name__=='__main__':unittest.main()
