import unittest
import numpy as np
from run_fixed_budget import PLAN, Problem, clean_observations


class FixedBudgetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(171)
        self.item = dict(omega=np.linspace(.05, .35, 9), Q=.3,
            bare=-.1*np.eye(3)+.02*(rng.normal(size=(9, 3, 3))+1j*rng.normal(size=(9, 3, 3))))
        self.data = {"B0": self.item, "Bplus": self.item}
        self.truth = [.8, 8., .7, -.3]

    def test_weighted_complex_rms_and_analytic_jacobian(self):
        fractions = {"B0": .75, "Bplus": .25}
        clean = clean_observations(self.data, self.truth[2:])
        observed = {f: x-PLAN["sigma0"]/np.sqrt(fractions[f])*(1+2j)/np.sqrt(5.)
                    for f, x in clean.items()}
        problem = Problem(self.data, observed, fractions)
        v = problem.start(self.truth, True)
        residual, jac = problem.evaluate(v)
        self.assertAlmostEqual(np.linalg.norm(residual), 1., places=10)
        for column in (0, 1, 2, 3, 4, 7, 8, 11, 12, 19):
            plus, minus = v.copy(), v.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            numerical = (problem.evaluate(plus)[0]-problem.evaluate(minus)[0])/(2e-6)
            np.testing.assert_allclose(jac[:, column], numerical, atol=2e-7, rtol=1e-6)

    def test_splitting_identical_fields_does_not_create_information(self):
        clean = clean_observations(self.data, self.truth[2:])
        single = Problem(self.data, clean, {"B0": 1.}).information(self.truth)
        split = Problem(self.data, clean, {"B0": .75, "Bplus": .25}).information(self.truth)
        np.testing.assert_allclose(single["interaction_fisher"], split["interaction_fisher"], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
