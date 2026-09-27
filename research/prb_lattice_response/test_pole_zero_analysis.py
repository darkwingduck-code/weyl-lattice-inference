"""Independent checks of finite-system rational poles, zeros, and residues.

The lower-half-plane evaluations below concern a finite rational resolvent.
They do not construct a continuum Landau sheet or establish a physical
linewidth, a thermodynamic gap, or an experimentally identifiable mode.
"""
from __future__ import annotations

import unittest

import numpy as np

from bloch_response import bloch_spectrum, bloch_transitions
from lattice_response import Parameters, Transitions, diagonalize, rpa_response
from pole_zero_analysis import (
    RationalResponse, charge_zero_intervals, exterior_real_poles,
    regular_pole_residue, transition_modes,
)
from test_lattice_response import (
    direct_current_kernel, operator_commutator_moments,
)

EPS = np.finfo(float).eps


def paired_oscillators(energies, weights, directions=None):
    """Rank-one +/- spectral pairs with exactly cancelling zeroth moment."""
    energies, weights = np.asarray(energies), np.asarray(weights)
    if directions is None:
        directions = np.eye(2)
    delta, matrices = [], []
    for energy, weight, vector in zip(energies, weights, np.asarray(directions).T):
        projector = np.outer(vector, vector.conj())
        for sign in (1, -1):
            delta.append(sign * energy)
            matrices.append(sign * weight * projector)
    matrices = np.asarray(matrices)
    return Transitions(
        delta=np.asarray(delta), cc=matrices[:, 0, 0].real,
        aa=matrices[:, 1, 1].real, ca=matrices[:, 0, 1],
        conduction=np.zeros(len(delta), dtype=bool), Q=.37, q=.372,
        vertex_error=0., density_value=2.)


def channel_rotation(theta=.43, phase=.61):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s * np.exp(1j * phase)],
                     [s * np.exp(-1j * phase), c]])


def oscillator_polarization(z, energies, weights, directions):
    """Closed oscillator functions, not a sum over signed transition slots."""
    result, derivative = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for e, w, vector in zip(energies, weights, directions.T):
        projector = np.outer(vector, vector.conj())
        result += 2 * e * w / (z**2 - e**2) * projector
        derivative += -4 * e * w * z / (z**2 - e**2)**2 * projector
    return result, derivative


def modal_sum(modes, z):
    poles, residues = np.asarray(modes["poles"]), np.asarray(modes["residues"])
    return np.sum(residues / (z - poles)[:, None, None], axis=0)


class PoleZeroAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.energies = np.array([1.1, 2.3])
        self.weights = np.array([.6, .4])
        self.rotation = channel_rotation()
        self.interaction = np.diag([.7, .35])
        self.decoupled = paired_oscillators(self.energies, self.weights)
        self.mixed = paired_oscillators(self.energies, self.weights, self.rotation)

    def assert_scaled_close(self, actual, expected, operations=4):
        actual, expected = np.asarray(actual), np.asarray(expected)
        scale = max(1., float(np.max(np.abs(actual))),
                    float(np.max(np.abs(expected))))
        tolerance = 512 * EPS * max(1, operations) * scale
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), tolerance,
                             f"roundoff budget {tolerance:.3e}")

    def test_rational_polarization_and_derivative_match_closed_oscillators(self):
        response = RationalResponse(self.mixed)
        for z in (.43 + .19j, 1.53 + .21j, -.62 - .13j, 3.1):
            expected, derivative = oscillator_polarization(
                z, self.energies, self.weights, self.rotation)
            self.assert_scaled_close(response.polarization(z), expected)
            self.assert_scaled_close(response.polarization(z, derivative=1), derivative)
        self.assert_scaled_close(self.mixed.moments()[0], np.zeros((2, 2)))

    def test_full_cofactor_and_derivatives_match_oscillator_algebra(self):
        response = RationalResponse(self.mixed)
        uc, ua = np.diag(self.interaction)
        for z in (.47 + .23j, 1.64 - .19j, -2.71 + .11j):
            pi, dpi = oscillator_polarization(
                z, self.energies, self.weights, self.rotation)
            f = 2 * self.energies * self.weights / (z**2 - self.energies**2)
            df = -4 * self.energies * self.weights * z / (z**2 - self.energies**2)**2
            # Orthogonal channel projectors give det Pi = f1*f2.
            det_pi = f[0] * f[1]
            ddet_pi = df[0] * f[1] + f[0] * df[1]
            expected = dict(
                determinant=1 - uc * pi[0, 0] - ua * pi[1, 1]
                            + uc * ua * det_pi,
                numerator=pi[0, 0] - ua * det_pi,
                determinant_derivative=-uc * dpi[0, 0] - ua * dpi[1, 1]
                                       + uc * ua * ddet_pi,
                numerator_derivative=dpi[0, 0] - ua * ddet_pi)
            actual = response.cofactor(z, self.interaction)
            for key, value in expected.items():
                self.assert_scaled_close(actual[key], value)
            direct = np.linalg.solve(np.eye(2) - pi @ self.interaction, pi)
            self.assert_scaled_close(response.dressed(z, self.interaction), direct)
            self.assert_scaled_close(actual["numerator"] / actual["determinant"],
                                     direct[0, 0])

    def test_decoupled_rpa_poles_and_signed_residues_are_analytic(self):
        modes = transition_modes(self.decoupled, self.interaction)
        omega = np.sqrt(self.energies**2
                        + 2 * self.energies * self.weights
                        * np.diag(self.interaction))
        expected_poles, expected_residues = [], []
        for channel, (e, w, frequency) in enumerate(zip(
                self.energies, self.weights, omega)):
            for sign in (-1, 1):
                residue = np.zeros((2, 2))
                residue[channel, channel] = sign * e * w / frequency
                expected_poles.append(sign * frequency)
                expected_residues.append(residue)
        order = np.argsort(expected_poles)
        actual_order = np.argsort(modes["poles"])
        self.assertEqual(len(modes["poles"]), 4)
        self.assert_scaled_close(np.asarray(modes["poles"])[actual_order],
                                 np.asarray(expected_poles)[order])
        self.assert_scaled_close(np.asarray(modes["residues"])[actual_order],
                                 np.asarray(expected_residues)[order])
        expected_weights = (np.asarray(modes["poles"])
                            * np.asarray(modes["residues"])[:, 0, 0])
        self.assert_scaled_close(modes["weighted_charge_weights"], expected_weights)
        self.assert_scaled_close(np.sum(modes["residues"], axis=0), np.zeros((2, 2)))
        for z in (.43 + .19j, 2.02 + .07j):
            expected = np.diag(2 * self.energies * self.weights / (z**2 - omega**2))
            self.assert_scaled_close(modal_sum(modes, z), expected)

    def test_complex_channel_phase_is_preserved_in_modal_residues(self):
        # Scalar U is invariant under this complex channel rotation.
        interaction = np.eye(2) * .53
        base = transition_modes(self.decoupled, interaction)
        rotated = transition_modes(self.mixed, interaction)
        ibase, irotated = np.argsort(base["poles"]), np.argsort(rotated["poles"])
        self.assert_scaled_close(np.asarray(rotated["poles"])[irotated],
                                 np.asarray(base["poles"])[ibase])
        expected = np.array([self.rotation @ r @ self.rotation.conj().T
                             for r in np.asarray(base["residues"])[ibase]])
        self.assertGreater(np.max(np.abs(expected[:, 0, 1].imag)), .01)
        self.assert_scaled_close(np.asarray(rotated["residues"])[irotated], expected)
        for residue in rotated["residues"]:
            self.assert_scaled_close(residue, residue.conj().T)

    def test_charge_hidden_axial_pole_cancels_in_full_scalar_cofactor(self):
        response = RationalResponse(self.decoupled)
        modes = transition_modes(self.decoupled, self.interaction)
        e, w = self.energies, self.weights
        uc, ua = np.diag(self.interaction)
        omega_axial = np.sqrt(e[1]**2 + 2 * e[1] * w[1] * ua)
        result = response.cofactor(omega_axial, self.interaction)
        self.assert_scaled_close(result["determinant"], 0.)
        self.assert_scaled_close(result["numerator"], 0.)
        # L'Hopital gives the regular charge value at the hidden matrix pole.
        charge_value = 2 * e[0] * w[0] / (
            omega_axial**2 - e[0]**2 - 2 * e[0] * w[0] * uc)
        self.assert_scaled_close(result["numerator_derivative"]
                                 / result["determinant_derivative"], charge_value)
        selected = int(np.argmin(np.abs(np.asarray(modes["poles"]) - omega_axial)))
        residue = np.asarray(modes["residues"])[selected]
        self.assert_scaled_close(residue[0, 0], 0.)
        self.assertGreater(abs(residue[1, 1]), .1)
        matrix_result = regular_pole_residue(response, self.interaction, omega_axial)
        self.assertEqual(matrix_result["nullity"], 1)
        self.assert_scaled_close(matrix_result["residue_matrix"], residue)

    def test_degenerate_determinant_zero_retains_a_true_simple_charge_pole(self):
        energy, weight, coupling = 1.3, .7, .5
        tr = paired_oscillators([energy, energy], [weight, weight])
        interaction = np.eye(2) * coupling
        response = RationalResponse(tr)
        pole = np.sqrt(energy**2 + 2 * energy * weight * coupling)
        cofactor = response.cofactor(pole, interaction)
        # D=(1-u*p)^2 and N=p*(1-u*p): their common zero does not
        # cancel the scalar pole, because the denominator has order two.
        for key in ("determinant", "numerator", "determinant_derivative"):
            self.assert_scaled_close(cofactor[key], 0.)
        self.assertGreater(abs(cofactor["numerator_derivative"]), .1)
        expected = energy * weight / pole * np.eye(2)
        result = regular_pole_residue(response, interaction, pole, expected_nullity=2)
        self.assertEqual(result["nullity"], 2)
        self.assert_scaled_close(result["residue_matrix"], expected)
        self.assert_scaled_close(result["eigenvalue_residual"], 0.)
        self.assertGreater(result["derivative_minimum"], 0.)
        modes = transition_modes(tr, interaction)
        selected = np.abs(np.asarray(modes["poles"]) - pole) < 512 * EPS * pole
        self.assertEqual(int(np.count_nonzero(selected)), 2)
        self.assert_scaled_close(np.asarray(modes["residues"])[selected].sum(axis=0),
                                 expected)

    def test_full_zero_differs_from_positive_two_pole_principal_part_zero(self):
        response = RationalResponse(self.mixed)
        modes = transition_modes(self.mixed, self.interaction)
        a = 2 * self.energies * self.weights
        charge_projectors = np.abs(self.rotation[0, :])**2
        ua = self.interaction[1, 1]
        zero_squared = (a[0] * charge_projectors[0] * self.energies[1]**2
                        + a[1] * charge_projectors[1] * self.energies[0]**2
                        + ua * a[0] * a[1]) / np.dot(a, charge_projectors)
        zero = np.sqrt(zero_squared)
        value = response.cofactor(zero, self.interaction)
        self.assert_scaled_close(value["numerator"], 0.)
        self.assertGreater(abs(value["determinant"]), 1e-3)
        self.assert_scaled_close(response.dressed(zero, self.interaction)[0, 0], 0.)
        selected = np.asarray(modes["poles"]) > 0
        poles = np.asarray(modes["poles"])[selected]
        residues = np.asarray(modes["residues"])[selected, 0, 0]
        self.assertEqual(len(poles), 2)
        principal_zero = (residues[0] * poles[1] + residues[1] * poles[0]) / sum(residues)
        self.assertGreater(abs(principal_zero - zero), 1e4 * EPS)
        self.assertGreater(abs(np.sum(residues / (zero - poles))), 1e4 * EPS)
        self.assert_scaled_close(modal_sum(modes, zero)[0, 0], 0.)

    def test_simple_charge_residue_matches_full_cofactor_derivative(self):
        response = RationalResponse(self.mixed)
        modes = transition_modes(self.mixed, self.interaction)
        for pole, residue in zip(modes["poles"], modes["residues"]):
            value = response.cofactor(pole, self.interaction)
            self.assert_scaled_close(value["determinant"], 0.)
            self.assertGreater(abs(value["determinant_derivative"]), .01)
            self.assert_scaled_close(value["numerator"]
                                     / value["determinant_derivative"], residue[0, 0])

    def test_finite_rational_reflection_is_matrix_hermitian_reflection(self):
        response = RationalResponse(self.mixed)
        for z in (.43 + .19j, 1.53 + .21j):
            self.assert_scaled_close(response.polarization(z.conjugate()),
                                     response.polarization(z).conj().T)
            self.assert_scaled_close(response.polarization(z.conjugate(), derivative=1),
                                     response.polarization(z, derivative=1).conj().T)
            self.assert_scaled_close(response.dressed(z.conjugate(), self.interaction),
                                     response.dressed(z, self.interaction).conj().T)
            upper, lower = (response.cofactor(point, self.interaction)
                            for point in (z, z.conjugate()))
            for key in ("determinant", "numerator", "determinant_derivative",
                        "numerator_derivative"):
                self.assert_scaled_close(lower[key], upper[key].conjugate())

    def test_exact_zero_slots_removed_but_arbitrarily_small_weights_retained(self):
        directions = np.column_stack((np.eye(2), np.array([1., 1j]) / np.sqrt(2)))
        tiny = paired_oscillators([1.1, 2.3, 3.7], [.6, .4, 1e-24], directions)
        # A zero-weight slot at a probe frequency is not an active bare pole.
        zero = Transitions(
            delta=np.r_[tiny.delta, .43], cc=np.r_[tiny.cc, 0.],
            aa=np.r_[tiny.aa, 0.], ca=np.r_[tiny.ca, 0j],
            conduction=np.zeros(7, bool), Q=tiny.Q, q=tiny.q,
            vertex_error=0., density_value=2.)
        response = RationalResponse(zero)
        expected, _ = oscillator_polarization(
            .43, np.array([1.1, 2.3, 3.7]), np.array([.6, .4, 1e-24]), directions)
        self.assert_scaled_close(response.polarization(.43), expected)
        modes = transition_modes(zero, self.interaction)
        self.assertEqual(len(modes["poles"]), 6)
        self.assertEqual(np.asarray(modes["residues"]).shape, (6, 2, 2))

    def test_twisted_small_lattice_modes_reconstruct_independent_orbital_kubo(self):
        p = Parameters(Lx=3, Ly=3, Nkz=8, theta_x=.17, theta_y=-.23,
                       theta_z=.37, mass=.09, hopping=1.07,
                       wilson=.96, exchange=.91)
        mu, temperature = .53, .19
        tr = bloch_transitions(bloch_spectrum(p), mu, temperature, qshift=1)
        reference = diagonalize(p, 0)
        response = RationalResponse(tr)
        interaction = np.diag([.28, .41])
        modes = transition_modes(tr, interaction, max_transitions=2500)
        active = np.count_nonzero((tr.cc != 0) | (tr.aa != 0) | (tr.ca != 0))
        self.assertEqual(len(modes["poles"]), active)
        self.assert_scaled_close(np.asarray(modes["poles"]).imag, 0., operations=active)
        for z in (.43 + .19j, .82 + .31j, -.58 + .27j):
            independent, _, _ = direct_current_kernel(reference, mu, temperature, z, 1)
            bare = independent[:2, :2]
            dressed = np.linalg.solve(np.eye(2) - bare @ interaction, bare)
            self.assert_scaled_close(response.polarization(z), bare, operations=active)
            self.assert_scaled_close(response.dressed(z, interaction), dressed,
                                     operations=active)
            self.assert_scaled_close(modal_sum(modes, z), dressed, operations=active)
            self.assert_scaled_close(rpa_response(tr.density_response(z), interaction),
                                     dressed, operations=active)
        zero, first = operator_commutator_moments(reference, mu, temperature, 1)
        residues, poles = np.asarray(modes["residues"]), np.asarray(modes["poles"])
        self.assert_scaled_close(np.sum(residues, axis=0), zero, operations=active)
        weighted = poles[:, None, None] * residues
        self.assert_scaled_close(np.sum(weighted, axis=0), first, operations=active)
        self.assert_scaled_close(modes["weighted_charge_weights"], weighted[:, 0, 0],
                                 operations=active)
        self.assert_scaled_close(modes["moment0_error"], np.max(np.abs(residues.sum(axis=0))),
                                 operations=active)
        tolerance = 512 * EPS * active * max(1., np.max(np.abs(weighted)))
        self.assertGreaterEqual(np.linalg.eigvalsh(weighted).min(), -tolerance)


    def test_nearby_distinct_poles_are_not_merged_into_a_nullspace(self):
        energy, weight, coupling = 1.3, .7, .5
        first_pole = np.sqrt(energy**2 + 2 * energy * weight * coupling)
        target_second = first_pole + 1e-10
        second_energy = -weight * coupling + np.sqrt(
            (weight * coupling)**2 + target_second**2)
        energies = np.array([energy, second_energy])
        expected_poles = np.sqrt(energies**2 + 2 * energies * weight * coupling)
        self.assertGreater(expected_poles[1] - expected_poles[0], 100 * EPS)
        self.assertLess(expected_poles[1] - expected_poles[0], 1e-8)
        response = RationalResponse(paired_oscillators(energies, [weight, weight]))
        interaction = np.eye(2) * coupling
        with self.assertRaisesRegex(ValueError, "Unresolved nearby dielectric roots"):
            regular_pole_residue(response, interaction, first_pole)
        with self.assertRaisesRegex(ValueError, "No declared RPA nullspace"):
            regular_pole_residue(response, interaction, first_pole, expected_nullity=2)
        exterior = exterior_real_poles(response, interaction)
        self.assertEqual(len(exterior["roots"]), 2)
        self.assert_scaled_close([r["pole"] for r in exterior["roots"]], expected_poles)
        self.assertTrue(all(r["nearby_root_cluster"] for r in exterior["roots"]))
        self.assertTrue(all(r["residue_status"] == "unresolved"
                            for r in exterior["roots"]))

    def test_charge_zero_interlacing_requires_zero_density_commutator(self):
        valid_modes = transition_modes(self.mixed, self.interaction)
        records = charge_zero_intervals(valid_modes)
        central = [r for r in records if r["status"] == "multiplication_zero"]
        self.assertEqual(len(central), 1)
        self.assert_scaled_close(central[0]["zero"], 0.)
        response = RationalResponse(self.mixed)
        self.assertGreater(abs(response.dressed(0., self.interaction)[0, 0]), .01)
        resolved = [r for r in records if r["status"] == "resolved_charge_zero"]
        self.assertEqual(len(resolved), 2)
        for record in resolved:
            self.assert_scaled_close(
                response.dressed(record["zero"], self.interaction)[0, 0], 0.)
        # An unbalanced, but individually valid rank-one transition set has
        # M0 != 0. Its weighted pole sum is z*chi-M0, so its zeros are not
        # the zeros asserted by the commuting-density interlacing identity.
        broken = paired_oscillators(self.energies, self.weights)
        broken.cc[0] *= 1.25
        invalid_modes = transition_modes(broken, self.interaction)
        self.assertGreater(invalid_modes["input_moment0_error"], .01)
        with self.assertRaisesRegex(ValueError, "M0=0"):
            charge_zero_intervals(invalid_modes)
        with self.assertRaisesRegex(ValueError, "M0=0"):
            exterior_real_poles(RationalResponse(broken), self.interaction)

    def test_exterior_search_starts_at_positive_not_absolute_bare_edge(self):
        # Pi_cc=4w/[(z-1)(z+3)] has M0=0 but asymmetric signed support.
        # Its positive RPA root lies above +1 and below |-3|.
        weight, coupling = .4, .2
        tr = Transitions(
            delta=np.array([1., -3.]), cc=np.array([weight, -weight]),
            aa=np.zeros(2), ca=np.zeros(2, complex),
            conduction=np.zeros(2, bool), Q=.37, q=.372,
            vertex_error=0., density_value=2.)
        response = RationalResponse(tr)
        interaction = np.diag([coupling, .7])
        expected_pole = -1 + np.sqrt(4 + 4 * weight * coupling)
        expected_residue = 2 * weight / (expected_pole + 1)
        self.assertGreater(expected_pole, 1.)
        self.assertLess(expected_pole, 3.)
        result = exterior_real_poles(response, interaction)
        self.assertEqual(result["status"], "bounded_exterior_search_completed")
        self.assertEqual(result["positive_bare_edge"], 1.)
        self.assertEqual(len(result["roots"]), 1)
        root = result["roots"][0]
        self.assert_scaled_close(root["pole"], expected_pole)
        self.assert_scaled_close(root["residue_matrix"][0, 0], expected_residue)
        self.assertEqual(root["residue_status"], "isolated_nullspace_checked")
        self.assertTrue(result["near_edge_unresolved"])
        self.assertEqual(result["excluded_edge_interval"][0], 1.)
        self.assertGreater(result["excluded_edge_interval"][1], 1.)


    def test_zero_interaction_has_bare_response_poles_but_no_dielectric_roots(self):
        interaction = np.zeros((2, 2))
        response = RationalResponse(self.mixed)
        modes = transition_modes(self.mixed, interaction)
        expected_poles, expected_residues = [], []
        for energy, weight, vector in zip(
                self.energies, self.weights, self.rotation.T):
            projector = np.outer(vector, vector.conj())
            for sign in (-1, 1):
                expected_poles.append(sign * energy)
                expected_residues.append(sign * weight * projector)
        expected_order = np.argsort(expected_poles)
        actual_order = np.argsort(modes["poles"])
        self.assertEqual(len(modes["poles"]), 4)
        self.assert_scaled_close(np.asarray(modes["poles"])[actual_order],
                                 np.asarray(expected_poles)[expected_order])
        self.assert_scaled_close(np.asarray(modes["residues"])[actual_order],
                                 np.asarray(expected_residues)[expected_order])
        self.assertTrue(np.all(np.abs(np.asarray(modes["residues"])[:, 0, 0]) > .01))
        for z in (.43 + .19j, -.62 - .13j, 3.1):
            expected, _ = oscillator_polarization(
                z, self.energies, self.weights, self.rotation)
            cofactor = response.cofactor(z, interaction)
            self.assert_scaled_close(cofactor["determinant"], 1.)
            self.assert_scaled_close(cofactor["determinant_derivative"], 0.)
            self.assert_scaled_close(cofactor["numerator"], expected[0, 0])
            self.assert_scaled_close(response.dressed(z, interaction), expected)
            self.assert_scaled_close(modal_sum(modes, z), expected)
        exterior = exterior_real_poles(response, interaction)
        self.assertEqual(exterior["roots"], [])
        self.assertEqual(exterior["status"],
                         "zero_interaction_no_exterior_determinant_roots")
        self.assertIs(exterior["near_edge_unresolved"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
