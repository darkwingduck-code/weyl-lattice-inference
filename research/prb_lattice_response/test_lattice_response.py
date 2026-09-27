"""Independent finite-lattice checks for the prospective PRB calculation.

These tests check the stated Wilson-Dirac model and its finite-q retarded
resolvent. They do not establish a continuum limit, a collective pole gap,
or conservation of the local axial bilinear. Existing NumPy/SciPy suffice.

Run from this directory with: python -m unittest -v test_lattice_response
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from lattice_response import (
    AXIAL, GAMMA, MASS, Parameters, bloch_hamiltonian, diagonalize,
    fixed_density_mu, loss_matrix, rpa_response, transitions,
)


EPS = np.finfo(float).eps


def analytic_bloch_energies(k, p):
    """Closed eigenvalues, independently of a four-by-four diagonalization."""
    k = np.asarray(k)
    mass = p.mass + p.wilson * np.sum(1 - np.cos(k))
    longitudinal = np.sqrt(mass**2 + (p.hopping * np.sin(k[2]))**2)
    transverse = p.hopping**2 * np.sum(np.sin(k[:2])**2)
    positive = np.sqrt(transverse + (longitudinal + np.array([-1, 1])
                                      * p.exchange)**2)
    return np.sort(np.concatenate((-positive, positive)))


def fermi_projectors(s, mu, temperature):
    # The independent tanh representation avoids the core occupation helper.
    f = 0.5 * (1 + np.tanh((mu - s.energies) / (2 * temperature)))
    matrices = np.array([(u * fi[None, :]) @ u.conj().T
                         for u, fi in zip(s.vectors, f)])
    return f, matrices


def direct_current_kernel(s, mu, temperature, z, qshift):
    """Unsimplified Kubo vertices and the independent hopping-curvature contact.

    The longitudinal vertex is the analytic Peierls vertex at the midpoint.
    It is not replaced by delta*charge/Q or a Hamiltonian finite difference.
    """
    p = s.parameters
    q = 2 * np.pi * qshift / p.Nkz
    f, projectors = fermi_projectors(s, mu, temperature)
    eye_sites = np.eye(p.Lx * p.Ly)
    axial = np.kron(eye_sites, AXIAL)
    result = np.zeros((3, 3), complex)
    contact = 0j
    for i, k in enumerate(s.kz):
        j = (i + qshift) % p.Nkz
        u, v = s.vectors[i], s.vectors[j]
        midpoint = k + q / 2
        current_operator = np.kron(
            eye_sites, p.wilson * np.sin(midpoint) * MASS
            + p.hopping * np.cos(midpoint) * GAMMA[2])
        vertices = np.array([u.conj().T @ v,
                             u.conj().T @ axial @ v,
                             u.conj().T @ current_operator @ v])
        energies = s.energies[j][None, :] - s.energies[i][:, None]
        weight = (f[i][:, None] - f[j][None, :]) / (z - energies)
        result += np.einsum("ij,aij,bij->ab", weight, vertices,
                            vertices.conj(), optimize=True) / p.volume
        curvature = np.kron(
            eye_sites, p.wilson * np.cos(k) * MASS
            - p.hopping * np.sin(k) * GAMMA[2])
        contact += np.trace(projectors[i] @ curvature) / p.volume
    paramagnetic = result.copy()
    result[2, 2] += contact
    return result, paramagnetic, contact


def operator_commutator_moments(s, mu, temperature, qshift):
    """Equal-time commutators evaluated in the orbital basis, block by block.

    With A_q mapping k+q to k, the first moment is
    Tr f(H) [[A_q,H],B_q^dagger]/V. No transition weights enter here.
    """
    p = s.parameters
    _, f = fermi_projectors(s, mu, temperature)
    operators = (np.eye(p.dim), np.kron(np.eye(p.Lx * p.Ly), AXIAL))
    zero = np.zeros((2, 2), complex)
    first = np.zeros((2, 2), complex)
    for i, h in enumerate(s.hamiltonians):
        forward = s.hamiltonians[(i + qshift) % p.Nkz]
        backward = s.hamiltonians[(i - qshift) % p.Nkz]
        for a, left in enumerate(operators):
            for b, right in enumerate(operators):
                zero[a, b] += np.trace(f[i] @ (left @ right - right @ left))
                nested = ((left @ forward - h @ left) @ right
                          - right @ (left @ h - backward @ left))
                first[a, b] += np.trace(f[i] @ nested)
    return zero / p.volume, first / p.volume


class LatticeResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Parameters(Lx=3, Ly=3, Nkz=12, theta_x=.13,
                           theta_y=-.17, theta_z=.11)
        cls.temperature = .08
        cls.target = 2.12
        cls.plus = diagonalize(cls.p, 1)
        cls.minus = diagonalize(cls.p, -1)
        cls.mu_plus = fixed_density_mu(cls.plus, cls.target, cls.temperature)
        cls.mu_minus = fixed_density_mu(cls.minus, cls.target, cls.temperature)
        cls.transition = transitions(cls.plus, cls.mu_plus, cls.temperature)
        cls.z = .61 + .08j
        cls.interaction = np.array([[.38, .06 + .04j],
                                    [.06 - .04j, .27]])

    def assert_roundoff_equal(self, actual, expected, size=None):
        """A size-scaled floating-point budget, rather than fitted tolerances."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        if size is None:
            size = self.p.volume
        scale = max(1., float(np.max(np.abs(actual))),
                    float(np.max(np.abs(expected))))
        tolerance = 128 * EPS * size * scale
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), tolerance,
                             f"floating-point budget {tolerance:.3e}")

    def test_analytic_nodes_and_projected_axial_charge(self):
        p = Parameters(Lx=3, Ly=3, Nkz=12)
        # A zero requires sin(kx)=sin(ky)=0 and sqrt(M^2+sin(kz)^2)=1.
        # For transverse Wilson offsets c=0,2,4 this reduces to
        # cos(kz)=(c+1)/2. Only c=0 is admissible, giving +/- pi/3.
        allowed = []
        for kx in (0., np.pi):
            for ky in (0., np.pi):
                c = (1 - np.cos(kx)) + (1 - np.cos(ky))
                cosine = (c + 1) / 2
                if abs(cosine) <= 1:
                    allowed.extend([(kx, ky, sign * np.arccos(cosine))
                                    for sign in (-1, 1)])
        self.assertEqual(len(allowed), 2)
        for k in allowed:
            self.assert_roundoff_equal(abs(k[2]), np.pi / 3, size=4)
            h = bloch_hamiltonian(k, p)
            values, vectors = np.linalg.eigh(h)
            zero = np.abs(values) < 128 * EPS * 4 * np.linalg.norm(h)
            self.assertEqual(int(zero.sum()), 2)
            basis = vectors[:, zero]
            expected_axial = -np.sign(k[2]) * np.sqrt(3) / 2 * np.eye(2)
            self.assert_roundoff_equal(basis.conj().T @ AXIAL @ basis,
                                      expected_axial, size=4)
        rng = np.random.default_rng(1401)
        for k in rng.uniform(-np.pi, np.pi, size=(16, 3)):
            self.assert_roundoff_equal(np.linalg.eigvalsh(bloch_hamiltonian(k, p)),
                                      analytic_bloch_energies(k, p), size=4)

    def test_zero_field_torus_spectrum_matches_bloch_grids(self):
        # Rectangular torus, twists, and non-default couplings expose boundary,
        # Fourier-sign and missing-band errors without copying the real-space code.
        p = Parameters(Lx=3, Ly=4, Nkz=12, theta_x=.29, theta_y=-.43,
                       theta_z=.19, mass=.14, hopping=.93,
                       wilson=1.13, exchange=.79)
        s = diagonalize(p, 0)
        kxs = (2 * np.pi * np.arange(p.Lx) - p.theta_x) / p.Lx
        kys = (2 * np.pi * np.arange(p.Ly) - p.theta_y) / p.Ly
        for i, kz in enumerate(s.kz):
            expected = np.sort(np.concatenate([
                analytic_bloch_energies((kx, ky, kz), p)
                for kx in kxs for ky in kys]))
            self.assert_roundoff_equal(s.energies[i], expected, size=p.dim)

    def test_arbitrary_local_gauge_covariance_of_spectrum_and_response(self):
        rng = np.random.default_rng(2402)
        phase = rng.uniform(-np.pi, np.pi, self.p.Lx * self.p.Ly)
        gauged = diagonalize(self.p, 1, site_phases=phase)
        g = np.repeat(np.exp(1j * phase), 4)
        transformed = (g[None, :, None] * self.plus.hamiltonians
                       * g.conj()[None, None, :])
        self.assert_roundoff_equal(gauged.hamiltonians, transformed)
        self.assert_roundoff_equal(gauged.energies, self.plus.energies)
        mu = fixed_density_mu(gauged, self.target, self.temperature)
        self.assert_roundoff_equal(mu, self.mu_plus)
        response = transitions(gauged, mu, self.temperature).full_response(self.z)
        bare = self.transition.full_response(self.z)
        self.assert_roundoff_equal(response, bare)
        self.assert_roundoff_equal(rpa_response(response, self.interaction),
                                  rpa_response(bare, self.interaction))

    def test_independent_eigenvector_phases_leave_responses_invariant(self):
        rng = np.random.default_rng(3403)
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi,
                                        self.plus.energies.shape))
        rephased = replace(self.plus,
                           vectors=self.plus.vectors * phases[:, None, :])
        response = transitions(rephased, self.mu_plus, self.temperature)
        self.assert_roundoff_equal(response.full_response(self.z),
                                  self.transition.full_response(self.z))
        for actual, expected in zip(response.moments(), self.transition.moments()):
            self.assert_roundoff_equal(actual, expected)

    def test_degenerate_subspace_rotations_leave_responses_invariant(self):
        p = Parameters(Lx=3, Ly=3, Nkz=12)
        s = diagonalize(p, 0)
        vectors = s.vectors.copy()
        rng = np.random.default_rng(4404)
        rotated_groups = 0
        for i, values in enumerate(s.energies):
            threshold = 128 * EPS * p.dim * max(1., np.max(np.abs(values)))
            starts = np.r_[0, np.flatnonzero(np.diff(values) > threshold) + 1,
                           p.dim]
            for lo, hi in zip(starts[:-1], starts[1:]):
                if hi - lo < 2:
                    continue
                # Only roundoff-degenerate eigenvalues may share a basis rotation.
                self.assertLessEqual(values[hi - 1] - values[lo], threshold)
                n = hi - lo
                unitary, _ = np.linalg.qr(rng.normal(size=(n, n))
                                         + 1j * rng.normal(size=(n, n)))
                vectors[i, :, lo:hi] = s.vectors[i, :, lo:hi] @ unitary
                rotated_groups += 1
        self.assertGreater(rotated_groups, 0)
        rotated = replace(s, vectors=vectors)
        for h, v, e in zip(rotated.hamiltonians, vectors, rotated.energies):
            self.assert_roundoff_equal(h @ v, v * e[None, :])
        mu = fixed_density_mu(s, self.target, self.temperature)
        reference = transitions(s, mu, self.temperature).full_response(self.z)
        response = transitions(rotated, mu, self.temperature).full_response(self.z)
        self.assert_roundoff_equal(response, reference)

    def test_finite_q_charge_ward_identity_before_and_after_rpa(self):
        for shift in (-1, 1, 2):
            t = transitions(self.plus, self.mu_plus, self.temperature, shift)
            self.assert_roundoff_equal(t.vertex_error, 0.)
            for z in (self.z, 1.19 + .13j):
                contraction = np.array([z, 0., -t.Q])
                bare = t.full_response(z)
                for response in (bare, rpa_response(bare, self.interaction)):
                    with self.subTest(qshift=shift, z=z):
                        self.assert_roundoff_equal(contraction @ response,
                                                  np.zeros(3))
                        self.assert_roundoff_equal(response @ contraction,
                                                  np.zeros(3))

    def test_full_response_matches_unsimplified_current_and_contact(self):
        for shift in (-1, 1, 2):
            direct, paramagnetic, contact = direct_current_kernel(
                self.plus, self.mu_plus, self.temperature, self.z, shift)
            t = transitions(self.plus, self.mu_plus, self.temperature, shift)
            self.assert_roundoff_equal(t.full_response(self.z), direct)
            self.assert_roundoff_equal(contact.imag, 0.)
            self.assertGreater(abs(contact), 1e4 * EPS)
            # Omitting the diamagnetic contact must visibly fail the Ward check.
            contraction = np.array([self.z, 0., -t.Q])
            self.assertGreater(np.max(np.abs(contraction @ paramagnetic)),
                               abs(t.Q * contact) / 2)

    def test_density_is_fixed_separately_for_both_flux_signs(self):
        for s, mu in ((self.plus, self.mu_plus), (self.minus, self.mu_minus)):
            occupancy = .5 * (1 + np.tanh((mu - s.energies)
                                          / (2 * self.temperature)))
            actual = np.sum(occupancy) / self.p.volume
            self.assert_roundoff_equal(actual, self.target)
        # This geometry gives different chemical potentials. Reusing mu(+B)
        # at -B would therefore violate the declared ensemble.
        self.assertGreater(abs(self.mu_plus - self.mu_minus), 1e4 * EPS)
        wrong = np.sum(.5 * (1 + np.tanh(
            (self.mu_plus - self.minus.energies) / (2 * self.temperature))))
        self.assertGreater(abs(wrong / self.p.volume - self.target), 1e4 * EPS)

    def test_finite_q_commutators_and_large_frequency_moments(self):
        for shift in (-1, 1, 2):
            t = transitions(self.plus, self.mu_plus, self.temperature, shift)
            zero, first = operator_commutator_moments(
                self.plus, self.mu_plus, self.temperature, shift)
            actual_zero, actual_first = t.moments()
            self.assert_roundoff_equal(actual_zero, zero)
            self.assert_roundoff_equal(actual_first, first)
            self.assert_roundoff_equal(zero, np.zeros((2, 2)))
            # The tail bound follows from the geometric resolvent expansion.
            # It is checked against independent equal-time moments, not a fit.
            largest_delta = np.max(np.abs(t.delta))
            weights = ((t.cc, t.ca), (t.ca.conj(), t.aa))
            second_absolute = np.array([
                [np.sum(np.abs(w) * t.delta**2) for w in row]
                for row in weights])
            for z in (400j, 800j):
                remainder = z**2 * t.density_response(z) - z * zero - first
                bound = second_absolute / (abs(z) - largest_delta)
                cancellation_budget = (128 * EPS * self.p.volume * abs(z)
                                       * max(1., np.max(np.abs(first))))
                self.assertTrue(np.all(np.abs(remainder)
                                       <= bound + cancellation_budget))
                self.assertLess(np.max(np.abs(remainder)),
                                .02 * max(1., np.max(np.abs(first))))

    def test_rpa_dyson_identity_and_full_matrix_loss_congruence(self):
        for dimension in (2, 3):
            bare = self.transition.full_response(self.z)[:dimension, :dimension]
            coupling = np.zeros((dimension, dimension), complex)
            coupling[:2, :2] = self.interaction
            left_inverse = np.linalg.inv(np.eye(dimension) - bare @ coupling)
            actual = rpa_response(bare, self.interaction)
            self.assert_roundoff_equal(actual, left_inverse @ bare)
            self.assert_roundoff_equal(actual, bare + bare @ coupling @ actual)
            # Use the anti-Hermitian matrix, not elementwise imaginary parts.
            independent_loss = -(actual - actual.conj().T) / (2j)
            bare_loss = -(bare - bare.conj().T) / (2j)
            expected_loss = left_inverse @ bare_loss @ left_inverse.conj().T
            self.assert_roundoff_equal(independent_loss, expected_loss)
            self.assert_roundoff_equal(loss_matrix(actual), independent_loss)
            self.assert_roundoff_equal(independent_loss, independent_loss.conj().T)
            # Positivity of the frequency-weighted density response is checked
            # separately. No PSD claim is made for the current-contact 3x3.

    def test_frequency_weighted_density_passivity_at_finite_eta(self):
        t = self.transition
        z = self.z
        bare = t.density_response(z)
        actual = rpa_response(bare, self.interaction)
        self.assert_roundoff_equal(t.moments()[0], np.zeros((2, 2)))
        # z/(z-delta) has imaginary part -eta*delta/|z-delta|^2.
        # Each df*delta is nonnegative for monotone Fermi occupations.
        weight = z.imag * t.delta / np.abs(z - t.delta)**2
        independent_bare = np.array([
            [np.dot(t.cc, weight), np.dot(t.ca, weight)],
            [np.dot(t.ca.conj(), weight), np.dot(t.aa, weight)]])
        weighted_bare_loss = -(z * bare - (z * bare).conj().T) / (2j)
        self.assert_roundoff_equal(weighted_bare_loss, independent_bare)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.interaction).min(), 0.)
        left = np.linalg.inv(np.eye(2) - bare @ self.interaction)
        correction = z.imag * bare @ self.interaction @ bare.conj().T
        expected = left @ (independent_bare + correction) @ left.conj().T
        weighted_rpa_loss = -(z * actual - (z * actual).conj().T) / (2j)
        self.assert_roundoff_equal(weighted_rpa_loss, expected)
        for matrix in (independent_bare, correction, weighted_rpa_loss):
            self.assert_roundoff_equal(matrix, matrix.conj().T)
            tolerance = 128 * EPS * self.p.volume * max(1., np.linalg.norm(matrix))
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -tolerance)
        # The finite-eta interaction correction is measurably nonzero here.
        self.assertGreater(np.linalg.norm(correction), 1e4 * EPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
