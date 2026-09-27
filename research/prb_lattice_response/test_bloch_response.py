"""Bloch acceleration checked against the full zero-flux Peierls torus.

Roundoff budgets scale with the reference matrix size and summed volume;
these tests establish algebraic equivalence, not thermodynamic convergence.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from bloch_response import (
    BlochSpectrum, bloch_density, bloch_mu, bloch_spectrum, bloch_transitions,
)
from lattice_response import (
    Parameters, density, diagonalize, fixed_density_mu, rpa_response, transitions,
)
from test_lattice_response import (
    direct_current_kernel, operator_commutator_moments,
)


EPS = np.finfo(float).eps


class BlochResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temperature = .08
        cls.target = 2.12
        cls.cases = []
        for p in (
            Parameters(Lx=3, Ly=3, Nkz=12, theta_x=.13, theta_y=-.17,
                       theta_z=.11),
            Parameters(Lx=4, Ly=4, Nkz=12, theta_x=.29, theta_y=-.43,
                       theta_z=.19, mass=.14, hopping=.93,
                       wilson=1.13, exchange=.79),
            Parameters(Lx=3, Ly=4, Nkz=8),
        ):
            bloch = bloch_spectrum(p)
            torus = diagonalize(p, 0)
            mu = bloch_mu(bloch, cls.target, cls.temperature)
            torus_mu = fixed_density_mu(torus, cls.target, cls.temperature)
            cls.cases.append((p, bloch, torus, mu, torus_mu))
        cls.interaction = np.array([[.38, .06 + .04j],
                                    [.06 - .04j, .27]])

    def assert_roundoff_equal(self, actual, expected, p):
        actual, expected = np.asarray(actual), np.asarray(expected)
        scale = max(1., float(np.max(np.abs(actual))),
                    float(np.max(np.abs(expected))))
        tolerance = 128 * EPS * p.volume * scale
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), tolerance,
                             f"floating-point budget {tolerance:.3e}")

    def test_bloch_shapes_twists_and_full_torus_eigenvalues(self):
        for p, bloch, torus, _, _ in self.cases:
            with self.subTest(Lx=p.Lx, Ly=p.Ly):
                self.assertIsInstance(bloch, BlochSpectrum)
                self.assertEqual(bloch.energies.shape, (p.Lx * p.Ly, p.Nkz, 4))
                self.assertEqual(bloch.vectors.shape,
                                 (p.Lx * p.Ly, p.Nkz, 4, 4))
                self.assert_roundoff_equal(
                    bloch.kx, np.repeat((2 * np.pi * np.arange(p.Lx)
                                        - p.theta_x) / p.Lx, p.Ly), p)
                self.assert_roundoff_equal(
                    bloch.ky, np.tile((2 * np.pi * np.arange(p.Ly)
                                      - p.theta_y) / p.Ly, p.Lx), p)
                self.assert_roundoff_equal(bloch.kz, torus.kz, p)
                assembled = np.sort(bloch.energies.transpose(1, 0, 2)
                                    .reshape(p.Nkz, p.dim), axis=1)
                self.assert_roundoff_equal(assembled, torus.energies, p)

    def test_density_and_fixed_density_mu_match_full_torus(self):
        for p, bloch, torus, mu, torus_mu in self.cases:
            self.assert_roundoff_equal(mu, torus_mu, p)
            self.assert_roundoff_equal(
                bloch_density(bloch, mu, self.temperature), self.target, p)
            for probe_mu in (-.7, .5, 1.2):
                self.assert_roundoff_equal(
                    bloch_density(bloch, probe_mu, .13),
                    density(torus, probe_mu, .13), p)

    def test_all_channels_and_rpa_match_full_torus_for_both_q_signs(self):
        for p, bloch, torus, mu, torus_mu in self.cases:
            for shift in (-1, 1):
                with self.subTest(Lx=p.Lx, Ly=p.Ly, qshift=shift):
                    actual = bloch_transitions(bloch, mu, self.temperature, shift)
                    expected = transitions(torus, torus_mu, self.temperature, shift)
                    self.assert_roundoff_equal(actual.q, expected.q, p)
                    self.assert_roundoff_equal(actual.Q, expected.Q, p)
                    self.assert_roundoff_equal(actual.density_value, self.target, p)
                    self.assert_roundoff_equal(actual.vertex_error, 0., p)
                    for z in (.61 + .08j, 1.19 + .13j):
                        self.assert_roundoff_equal(actual.full_response(z),
                                                  expected.full_response(z), p)
                        self.assert_roundoff_equal(
                            rpa_response(actual.full_response(z), self.interaction),
                            rpa_response(expected.full_response(z), self.interaction), p)
                        self.assert_roundoff_equal(actual.ward_errors(z), [0., 0.], p)
                        self.assert_roundoff_equal(
                            actual.ward_errors(z, self.interaction), [0., 0.], p)

    def test_independent_peierls_current_contact_and_commutator_moments(self):
        for p, bloch, torus, mu, torus_mu in self.cases:
            for shift in (-1, 1):
                with self.subTest(Lx=p.Lx, Ly=p.Ly, qshift=shift):
                    actual = bloch_transitions(bloch, mu, self.temperature, shift)
                    direct, _, _ = direct_current_kernel(
                        torus, torus_mu, self.temperature, .61 + .08j, shift)
                    self.assert_roundoff_equal(actual.full_response(.61 + .08j),
                                              direct, p)
                    independent = operator_commutator_moments(
                        torus, torus_mu, self.temperature, shift)
                    for result, expected in zip(actual.moments(), independent):
                        self.assert_roundoff_equal(result, expected, p)

    def test_all_bands_and_zero_weight_pairs_are_retained(self):
        for p, bloch, _, mu, _ in self.cases:
            for shift in (-1, 1):
                actual = bloch_transitions(bloch, mu, self.temperature, shift)
                count = p.volume * 16
                for name in ("delta", "cc", "aa", "ca", "conduction"):
                    self.assertEqual(getattr(actual, name).shape, (count,))
                self.assertEqual(actual.conduction.dtype, np.dtype(bool))
                positive = bloch.energies > 0
                expected_conduction = sum(
                    np.sum(positive[:, i, :, None]
                           & positive[:, (i + shift) % p.Nkz, None, :])
                    for i in range(p.Nkz))
                self.assertEqual(np.count_nonzero(actual.conduction),
                                 expected_conduction)
                # Deep filled pairs have exactly equal floating-point occupations.
                # They remain in the returned arrays instead of being discarded.
                self.assertGreater(np.count_nonzero(actual.cc == 0.), 0)
                self.assertGreater(np.count_nonzero(~actual.conduction), 0)

    def test_independent_current_vertex_detects_changed_hopping(self):
        p, bloch, _, mu, _ = self.cases[0]
        stale = replace(bloch, parameters=replace(p, hopping=1.01 * p.hopping))
        for shift in (-1, 1):
            actual = bloch_transitions(bloch, mu, self.temperature, shift)
            bad = bloch_transitions(stale, mu, self.temperature, shift)
            self.assert_roundoff_equal(actual.vertex_error, 0., p)
            self.assertGreater(bad.vertex_error, 1e-4)

    def test_q_boundaries_and_positive_temperature_contract(self):
        p, bloch, torus, mu, torus_mu = self.cases[0]
        for shift in (0, p.Nkz, -p.Nkz, p.Nkz + 1, 1.5, -1.5):
            with self.subTest(invalid_qshift=shift):
                with self.assertRaises(ValueError):
                    bloch_transitions(bloch, mu, self.temperature, shift)
        # Largest allowed shifts wrap the kz grid and test midpoint periodicity.
        for shift in (p.Nkz - 1, 1 - p.Nkz):
            actual = bloch_transitions(bloch, mu, self.temperature, shift)
            expected = transitions(torus, torus_mu, self.temperature, shift)
            self.assert_roundoff_equal(actual.full_response(.61 + .08j),
                                      expected.full_response(.61 + .08j), p)
            self.assert_roundoff_equal(actual.vertex_error, 0., p)
        for temperature in (0., -.1):
            with self.assertRaises(ValueError):
                bloch_density(bloch, mu, temperature)
            with self.assertRaises(ValueError):
                bloch_mu(bloch, self.target, temperature)
            with self.assertRaises(ValueError):
                bloch_transitions(bloch, mu, temperature)


if __name__ == "__main__":
    unittest.main(verbosity=2)