import unittest
import numpy as np
from run_study import clean_observations, fit, real_residual
from run_study import dress_response, physical
from response_derivatives import contracted_derivatives
from recover_source_class import recover
from run_study import apply_transfer, contract_mixed


class SourceProtocolTests(unittest.TestCase):
    def test_unordered_recovery_with_arbitrary_complex_gains(self):
        rng = np.random.default_rng(994)
        response = rng.normal(size=(9, 3, 3)) + 1j*rng.normal(size=(9, 3, 3))
        response = response + response.swapaxes(-1, -2)
        lr, ls = .7, -.3
        measured = apply_transfer(contract_mixed(response, lr, ls),
            np.exp(1j*rng.normal(size=9)), 2*np.exp(1j*rng.normal(size=9)))
        result = recover({"B": response}, {"B": measured}, ["B"])
        self.assertEqual(result["rank"], 2)
        np.testing.assert_allclose([result["p"], result["q"]], [lr+ls, lr*ls], atol=1e-13)
        np.testing.assert_allclose(result["roots"], [ls, lr], atol=1e-13)

    def test_analytic_derivatives_against_independent_centered_differences(self):
        rng = np.random.default_rng(98)
        bare = -.1*np.eye(3) + .02*(rng.normal(size=(5, 3, 3)) +
                                   1j*rng.normal(size=(5, 3, 3)))
        data = dict(bare=bare, Q=.3)
        parameters = np.array([.8, 8., .7, -.3])
        derivatives = contracted_derivatives(dress_response(bare, .3, .8, 8.), .3, .7, -.3)
        for index, analytic in enumerate(derivatives):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical = (physical(data, *plus)-physical(data, *minus))/(2e-5)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-11)

    def test_complex_rms_and_diagonal_selection(self):
        truth = {"B": np.zeros((7, 2, 2), complex)}
        model = {"B": np.full((7, 2, 2), 1+2j)}
        self.assertAlmostEqual(np.linalg.norm(real_residual(
            model, truth, ["B"], "full", np.sqrt(5.))), 1.)
        model["B"][:, 0, 1] = 1000.
        self.assertAlmostEqual(np.linalg.norm(real_residual(
            model, truth, ["B"], "diagonal", np.sqrt(5.))), 1.)
        with self.assertRaises(ValueError):
            real_residual(model, truth, ["B"], "full", 0.)

    def test_independent_directions_recover_asymmetric_synthetic_response(self):
        rng = np.random.default_rng(2709)
        bare = -.1*np.eye(3) + .025*(rng.normal(size=(12, 3, 3))
                                    + 1j*rng.normal(size=(12, 3, 3)))
        data = {"B": dict(bare=bare, omega=np.linspace(.05, .35, 12), Q=.3)}
        observations = clean_observations(data, (.7, -.3))
        result = fit(data, observations, ["B"], "full", "independent",
                     (.7, -.3), .001, nstarts=2)
        self.assertLess(result["rms_noise_units"], 1e-7)
        np.testing.assert_allclose([result[k] for k in ("g", "Ua", "lambda_r", "lambda_s")],
                                   [.8, 8., .7, -.3], rtol=1e-6, atol=1e-7)
        self.assertEqual(len(result["jacobian_singular_values"]), 4)


if __name__ == "__main__":
    unittest.main()
