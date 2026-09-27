"""Exact magnetic-cell decomposition checked against the dense Peierls torus."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import block_diag

from lattice_response import (
    Parameters, density, diagonalize, fixed_density_mu, hamiltonian,
    rpa_response, transverse_hamiltonian, transitions,
)
from magnetic_bloch_response import (
    magnetic_bloch_hamiltonian, magnetic_bloch_spectrum,
    iter_magnetic_bloch_spectra, magnetic_bloch_density, magnetic_bloch_mu,
    magnetic_bloch_transitions,
)
from test_lattice_response import direct_current_kernel, operator_commutator_moments

EPS = np.finfo(float).eps


def fourier_isometry(p, denominator, ix, iy):
    """Independent torus-to-cell map; theta belongs to H, not this Fourier map."""
    nx = p.Lx//denominator
    result = np.zeros((p.dim,4*denominator),complex)
    for x in range(p.Lx):
        for y in range(p.Ly):
            phase = np.exp(2j*np.pi*(ix*x/p.Lx+iy*y/p.Ly))/np.sqrt(nx*p.Ly)
            site = 4*(x*p.Ly+y)
            cell = 4*(x%denominator)
            result[site:site+4,cell:cell+4] = phase*np.eye(4)
    return result


class MagneticBlochResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = .08
        cls.target = 2.12
        cls.z = .61+.08j
        cls.interaction = np.array([[.38,.06+.04j],[.06-.04j,.27]])
        cls.cases = []
        for lx,ly,numerator,denominator in (
            (3,3,1,3),(3,3,-1,3),(3,6,1,3),
            (6,3,-1,3),(6,6,1,3),(6,6,-1,3),
            (6,3,1,2),(3,4,0,1),(6,3,2,6),
        ):
            p = Parameters(Lx=lx,Ly=ly,Nkz=8,theta_x=.29,theta_y=-.43,
                           theta_z=.19,mass=.14,hopping=.93,wilson=1.13,exchange=.79)
            magnetic = magnetic_bloch_spectrum(p,numerator,denominator)
            torus = diagonalize(p,magnetic.nphi)
            mu = magnetic_bloch_mu(magnetic,cls.target,cls.T)
            dense_mu = fixed_density_mu(torus,cls.target,cls.T)
            cls.cases.append((p,magnetic,torus,mu,dense_mu))

    def assert_roundoff_equal(self, actual, expected, p):
        actual,expected = np.asarray(actual),np.asarray(expected)
        scale = max(1.,float(np.max(np.abs(actual))),float(np.max(np.abs(expected))))
        budget = 128*EPS*p.volume*scale
        self.assertLessEqual(float(np.max(np.abs(actual-expected))),budget,
                             f"roundoff budget {budget:.3e}")

    def test_full_fourier_transform_block_diagonalizes_dense_hamiltonian(self):
        for p,s,torus,_,_ in self.cases:
            with self.subTest(Lx=p.Lx,Ly=p.Ly,flux=(s.numerator,s.denominator)):
                maps = [fourier_isometry(p,s.denominator,ix,iy)
                        for ix in range(p.Lx//s.denominator) for iy in range(p.Ly)]
                unitary = np.concatenate(maps,axis=1)
                self.assert_roundoff_equal(unitary.conj().T@unitary,np.eye(p.dim),p)
                for i in (0,p.Nkz-1):
                    expected = block_diag(*[
                        magnetic_bloch_hamiltonian(p,s.numerator,s.denominator,
                                                    kx,ky,s.kz[i])
                        for kx,ky in zip(s.kx,s.ky)])
                    self.assert_roundoff_equal(unitary.conj().T@torus.hamiltonians[i]@unitary,
                                              expected,p)

    def test_spectrum_density_mu_and_flux_sign_match_dense_torus(self):
        for p,s,torus,mu,dense_mu in self.cases:
            with self.subTest(Lx=p.Lx,Ly=p.Ly,flux=(s.numerator,s.denominator)):
                self.assertTrue(s.complete)
                self.assertEqual(s.energies.shape,(s.total_blocks,p.Nkz,4*s.denominator))
                self.assertEqual(s.vectors.shape,
                                 (s.total_blocks,p.Nkz,4*s.denominator,4*s.denominator))
                assembled = np.sort(s.energies.transpose(1,0,2).reshape(p.Nkz,p.dim),axis=1)
                self.assert_roundoff_equal(assembled,torus.energies,p)
                self.assert_roundoff_equal(s.field,p.field(s.nphi),p)
                self.assert_roundoff_equal(mu,dense_mu,p)
                for probe_mu in (-.7,.5,1.2):
                    self.assert_roundoff_equal(magnetic_bloch_density(s,probe_mu,self.T),
                                              density(torus,probe_mu,self.T),p)
                self.assert_roundoff_equal(magnetic_bloch_density(s,mu,self.T),self.target,p)
        plus,minus = self.cases[0],self.cases[1]
        # Fixed magnetic exchange breaks T: silently dropping the flux sign
        # must not pass simply because a model happened to have an even spectrum.
        self.assertGreater(np.max(np.abs(plus[2].energies-minus[2].energies)),1e-4)

    def test_complex_bare_rpa_current_and_moments_match_for_both_probe_signs(self):
        for p,s,torus,mu,dense_mu in self.cases:
            for shift in (-1,1):
                with self.subTest(Lx=p.Lx,Ly=p.Ly,flux=(s.numerator,s.denominator),q=shift):
                    actual = magnetic_bloch_transitions(s,mu,self.T,shift)
                    dense = transitions(torus,dense_mu,self.T,shift)
                    self.assert_roundoff_equal(actual.vertex_error,0.,p)
                    for z in (self.z,1.19+.13j):
                        self.assert_roundoff_equal(actual.full_response(z),dense.full_response(z),p)
                        self.assert_roundoff_equal(rpa_response(actual.full_response(z),self.interaction),
                                                  rpa_response(dense.full_response(z),self.interaction),p)
                        self.assert_roundoff_equal(actual.ward_errors(z),[0.,0.],p)
                        self.assert_roundoff_equal(actual.ward_errors(z,self.interaction),[0.,0.],p)
                    direct,_,contact = direct_current_kernel(torus,dense_mu,self.T,self.z,shift)
                    self.assert_roundoff_equal(actual.full_response(self.z),direct,p)
                    self.assert_roundoff_equal(np.dot(actual.cc,actual.delta)/actual.Q**2,contact,p)
                    independent = operator_commutator_moments(torus,dense_mu,self.T,shift)
                    for value,expected in zip(actual.moments(),independent):
                        self.assert_roundoff_equal(value,expected,p)

    def test_periodic_cell_gauge_and_arbitrary_torus_gauge_consistency(self):
        p,s,torus,mu,_ = self.cases[4]
        phases = np.array([.31,-.87,.22])
        gauged = magnetic_bloch_spectrum(p,s.numerator,s.denominator,cell_phases=phases)
        site_phases = np.array([phases[x%s.denominator]
                               for x in range(p.Lx) for y in range(p.Ly)])
        dense_gauged = diagonalize(p,s.nphi,site_phases=site_phases)
        self.assert_roundoff_equal(gauged.energies,s.energies,p)
        for block,index in enumerate(s.block_indices):
            ix,iy = divmod(int(index),p.Ly)
            mapping = fourier_isometry(p,s.denominator,ix,iy)
            transformed = mapping.conj().T@dense_gauged.hamiltonians[2]@mapping
            expected = magnetic_bloch_hamiltonian(p,s.numerator,s.denominator,
                s.kx[block],s.ky[block],s.kz[2],cell_phases=phases)
            self.assert_roundoff_equal(transformed,expected,p)
        arbitrary = diagonalize(p,s.nphi,site_phases=np.random.default_rng(953).normal(size=p.Lx*p.Ly))
        for shift in (-1,1):
            reference = magnetic_bloch_transitions(s,mu,self.T,shift).full_response(self.z)
            self.assert_roundoff_equal(magnetic_bloch_transitions(
                gauged,mu,self.T,shift).full_response(self.z),reference,p)
            self.assert_roundoff_equal(transitions(arbitrary,mu,self.T,shift)
                                      .full_response(self.z),reference,p)

    def test_all_bands_zero_weights_and_streamed_bare_sum_are_preserved(self):
        p,s,_,mu,_ = self.cases[4]
        whole = magnetic_bloch_transitions(s,mu,self.T)
        expected_count = p.Nkz*s.total_blocks*(4*s.denominator)**2
        for name in ("delta","cc","aa","ca","conduction"):
            self.assertEqual(getattr(whole,name).size,expected_count)
        self.assertGreater(np.count_nonzero(whole.cc==0.),0)
        summed = np.zeros((3,3),complex)
        moments = np.zeros((2,2,2),complex)
        density_sum,count = 0.,0
        for chunk in iter_magnetic_bloch_spectra(p,s.numerator,s.denominator,blocks_per_chunk=5):
            self.assertFalse(chunk.complete)
            with self.assertRaises(ValueError):
                magnetic_bloch_mu(chunk,self.target,self.T)
            part = magnetic_bloch_transitions(chunk,mu,self.T)
            summed += part.full_response(self.z)
            moments += np.array(part.moments())
            density_sum += part.density_value
            count += part.delta.size
        self.assertEqual(count,expected_count)
        self.assert_roundoff_equal(summed,whole.full_response(self.z),p)
        self.assert_roundoff_equal(moments,np.array(whole.moments()),p)
        self.assert_roundoff_equal(density_sum,self.target,p)
        self.assert_roundoff_equal(rpa_response(summed,self.interaction),
                                  rpa_response(whole.full_response(self.z),self.interaction),p)

    def test_explicit_sample_counts_refine_size_at_fixed_physical_field(self):
        base = self.cases[0][0]
        for nx,ny in ((1,3),(1,6),(2,3),(2,6)):
            s = magnetic_bloch_spectrum(base,1,3,Nkx=nx,Nky=ny)
            self.assertEqual((s.parameters.Lx,s.parameters.Ly),(3*nx,ny))
            self.assertEqual(s.total_blocks,nx*ny)
            self.assertEqual(s.nphi,nx*ny)
            self.assert_roundoff_equal(s.field,2*np.pi/3,s.parameters)

    def test_denominator_36_block_matches_independent_peierls_projection(self):
        p = Parameters(Lx=36,Ly=3,Nkz=4,theta_x=.21,theta_y=-.37,theta_z=.13)
        mapping = fourier_isometry(p,36,0,1)
        for numerator in (-1,1):
            dense = hamiltonian(p,transverse_hamiltonian(p,3*numerator),p.kz_grid()[1])
            block = magnetic_bloch_hamiltonian(p,numerator,36,-p.theta_x/p.Lx,
                                               (2*np.pi-p.theta_y)/p.Ly,p.kz_grid()[1])
            self.assert_roundoff_equal(mapping.conj().T@dense@mapping,block,p)
            self.assert_roundoff_equal(dense@mapping,mapping@block,p)
            subset = magnetic_bloch_spectrum(p,numerator,36,block_range=(1,2))
            self.assertEqual(subset.energies.shape,(1,4,144))
            self.assert_roundoff_equal(subset.energies[0,1],
                                      np.linalg.eigvalsh(mapping.conj().T@dense@mapping),p)
            for shift in (-1,1):
                part = magnetic_bloch_transitions(subset,.5,.04,shift)
                self.assertEqual(part.delta.size,4*144**2)
                self.assert_roundoff_equal(part.vertex_error,0.,p)
                self.assert_roundoff_equal(part.ward_errors(.45+.015j),[0.,0.],p)

    def test_invalid_flux_grids_phases_and_probe_boundaries(self):
        p,s,torus,mu,dense_mu = self.cases[0]
        for numerator,denominator in ((1,0),(1,-3),(.5,3),(1,2.5),(True,3),(1,2)):
            with self.assertRaises(ValueError):
                magnetic_bloch_spectrum(p,numerator,denominator)
        for kwargs in ({"Nkx":0},{"Nky":2},{"block_range":(0,0)},
                       {"block_range":(-1,2)},{"block_range":(0,4)},
                       {"cell_phases":[0,1]},{"cell_phases":[0,1,np.nan]}):
            with self.assertRaises(ValueError):
                magnetic_bloch_spectrum(p,1,3,**kwargs)
        for shift in (0,p.Nkz,-p.Nkz,1.5,True):
            with self.assertRaises(ValueError):
                magnetic_bloch_transitions(s,mu,self.T,shift)
        for shift in (p.Nkz-1,1-p.Nkz):
            actual = magnetic_bloch_transitions(s,mu,self.T,shift)
            dense = transitions(torus,dense_mu,self.T,shift)
            self.assert_roundoff_equal(actual.full_response(self.z),dense.full_response(self.z),p)
            self.assert_roundoff_equal(actual.vertex_error,0.,p)
        with self.assertRaises(ValueError):
            magnetic_bloch_density(s,mu,0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)