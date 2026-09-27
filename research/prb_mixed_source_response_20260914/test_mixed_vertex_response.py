import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parents[1] / "research" / "prb_lattice_response"
for path in (HERE, LEGACY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bloch_response import bloch_spectrum, bloch_transitions
from lattice_response import AXIAL, GAMMA, Parameters, rpa_response
from magnetic_bloch_response import magnetic_bloch_spectrum
from mixed_vertex_response import (
    accumulate_spectrum, apply_transfer, closure, contract_mixed,
    dress_response, local_operators, scaled_max_error, ward_errors,
    weighted_passivity_minimum, process_rss_bytes,
)
from noise_sensitivity import (
    complex_noise, distribution_summary, parameter_summary, transfers,
)
from analyze_inference import raw_affine_fit


class AlgebraTests(unittest.TestCase):
    def test_raw_fit_reports_complex_rms_noise_units(self):
        # A zero response cannot fit this fixed residual, whatever the gains.
        omega = np.linspace(.05, .35, 9)
        observed = np.full((len(omega), 2, 2), 1 + 2j)
        fit = raw_affine_fit(np.zeros((len(omega), 3, 3)), omega, .3,
                             observed, np.sqrt(5.), .7)
        self.assertAlmostEqual(fit["rms_noise_units"], 1., places=13)

    def test_absolute_complex_noise_has_declared_rms(self):
        rng = np.random.default_rng(2026091501)
        noise = complex_noise(rng, (200000,), 2.5)
        empirical = np.sqrt(np.mean(np.abs(noise)**2))
        self.assertLess(abs(empirical/2.5-1), 0.01)

    def test_noise_summary_keeps_and_counts_boundary_values(self):
        values = np.array([.2, .200001, .8, 1.6])
        result = parameter_summary(values, .8, [.2, 1.6], 1e-5)
        self.assertEqual(result["finite_count"], 4)
        self.assertEqual(result["lower_bound_hits"], 2)
        self.assertEqual(result["upper_bound_hits"], 1)
        expected = np.sqrt(np.mean((values-.8)**2))
        self.assertAlmostEqual(result["rmse"], expected)

    def test_transfer_definition_matches_raw_affine_truth(self):
        omega = np.linspace(.05, .35, 45)
        readout, source = transfers(omega)
        x = (omega-omega.mean())/np.ptp(omega)
        np.testing.assert_allclose(readout,
            .7*np.exp(.4j)*(1+.1*x), rtol=0, atol=0)
        np.testing.assert_allclose(source,
            1.3*np.exp(-.25j)*(1+.1j*x), rtol=0, atol=0)
        summary = distribution_summary([.7, .8, .9], .8)
        self.assertAlmostEqual(summary["bias"], 0.)
    def test_contract_mixed_matches_manual_formula(self):
        rng = np.random.default_rng(14)
        response = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
        lr, ls = 0.37, -0.24
        actual = contract_mixed(response, lr, ls)
        manual = np.empty((5, 2, 2), complex)
        manual[:, 0, 0] = response[:, 0, 0]
        manual[:, 0, 1] = response[:, 0, 1] + ls * response[:, 0, 2]
        manual[:, 1, 0] = response[:, 1, 0] + lr * response[:, 2, 0]
        manual[:, 1, 1] = (response[:, 1, 1] + ls * response[:, 1, 2]
                           + lr * response[:, 2, 1]
                           + lr * ls * response[:, 2, 2])
        self.assertLess(scaled_max_error(actual, manual), 2e-15)

    def test_embedded_dressing_matches_established_rpa(self):
        rng = np.random.default_rng(9)
        bare = 0.01 * (rng.normal(size=(4, 3, 3))
                       + 1j * rng.normal(size=(4, 3, 3)))
        Q, g, ua = 0.31, 0.8, 8.0
        actual = dress_response(bare, Q, g, ua)
        interaction = np.diag([g / Q**2, ua])
        for i in range(len(bare)):
            expected = rpa_response(bare[i, :2, :2], interaction)
            self.assertLess(scaled_max_error(actual[i, :2, :2], expected), 2e-14)

    def test_transfer_closure_is_invariant(self):
        rng = np.random.default_rng(21)
        response = rng.normal(size=(11, 2, 2)) + 1j * rng.normal(size=(11, 2, 2))
        response[:, 1, 1] += 2.0
        x = np.linspace(-0.5, 0.5, len(response))
        transformed = apply_transfer(response,
            0.8 * np.exp(0.3j) * (1 + 0.1*x),
            1.2 * np.exp(-0.2j) * (1 - 0.05j*x))
        self.assertLess(scaled_max_error(closure(transformed), closure(response)),
                        3e-15)

    def test_windows_process_memory_probe(self):
        if sys.platform == "win32":
            value = process_rss_bytes()
            self.assertIsInstance(value, int)
            self.assertGreater(value, 0)


class SmallLatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Parameters(Lx=3, Ly=3, Nkz=24)
        cls.mu, cls.temperature, cls.eta = 0.37, 0.2, 0.04
        cls.qshift = 2
        cls.omega = np.array([0.09, 0.21, 0.33])
        cls.spec0 = magnetic_bloch_spectrum(cls.p, 0, 1)
        cls.acc3 = accumulate_spectrum(cls.spec0, cls.qshift, cls.mu,
            cls.temperature, cls.omega, cls.eta, frequency_batch=2)

    def test_zero_field_agrees_with_established_two_channel_core(self):
        legacy = bloch_transitions(bloch_spectrum(self.p), self.mu,
                                   self.temperature, self.qshift)
        expected = np.asarray([legacy.density_response(w + 1j*self.eta)
                               for w in self.omega])
        self.assertLess(scaled_max_error(self.acc3.bare[:, :2, :2], expected),
                        2e-12)
        self.assertAlmostEqual(self.acc3.density, legacy.density_value, places=13)

    def test_contracted_three_channel_equals_direct_mixed_vertex(self):
        lam = 0.7
        charge, gamma5, gamma3 = local_operators(1)
        direct = accumulate_spectrum(self.spec0, self.qshift, self.mu,
            self.temperature, self.omega, self.eta,
            operators=(charge, gamma5 + lam*gamma3), frequency_batch=2)
        contracted = contract_mixed(self.acc3.bare, lam)
        self.assertLess(scaled_max_error(contracted, direct.bare), 3e-12)

    def test_response_identities_and_passivity(self):
        q = 2*np.pi*self.qshift/self.p.Nkz
        errors = ward_errors(self.acc3, self.omega, self.eta, 2*np.sin(q/2))
        self.assertLess(self.acc3.charge_vertex_error, 2e-12)
        self.assertLess(np.max(np.abs(self.acc3.moment0)), 2e-12)
        self.assertLess(errors["left"], 3e-12)
        self.assertLess(errors["right"], 3e-12)
        self.assertGreater(weighted_passivity_minimum(
            self.acc3.bare, self.omega, self.eta), -3e-12)
        self.assertEqual(self.acc3.raw_pairs, self.p.volume * 16)

    def test_complete_magnetic_torus_is_cell_phase_invariant(self):
        pm = Parameters(Lx=3, Ly=3, Nkz=12)
        phases = np.array([0.17, -0.31, 0.52])
        a = accumulate_spectrum(magnetic_bloch_spectrum(pm, 1, 3), 1,
            self.mu, self.temperature, self.omega[:1], self.eta)
        b = accumulate_spectrum(magnetic_bloch_spectrum(
            pm, 1, 3, cell_phases=phases), 1, self.mu, self.temperature,
            self.omega[:1], self.eta)
        self.assertLess(scaled_max_error(a.bare, b.bare), 3e-12)
        self.assertLess(abs(a.density-b.density), 3e-13)


if __name__ == "__main__":
    unittest.main()
