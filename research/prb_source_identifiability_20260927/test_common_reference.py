"""Integration regression for a higher-mesh case with competing local minima."""
import unittest

import numpy as np

from run_study import (DESIGN, ROOT, clean_observations, fit, inputs, load,
                       real_residual)


class CommonReferenceTests(unittest.TestCase):
    def test_five_frozen_starts_find_feasible_B0_basin(self):
        reference_path = (ROOT.parent / "prb_refinement_20260927" /
                          "B0_higher" / "response_3channel.npz")
        if not reference_path.exists():
            self.skipTest("The separate higher-resolution B0 artifact is required")
        _, data = inputs()
        all_truths = [clean_observations(data, pair)
                      for pair in DESIGN["truths"].values()]
        sigma = DESIGN["epsilon"] * np.sqrt(np.mean(np.concatenate([
            abs(value).ravel()**2 for truth in all_truths for value in truth.values()])))
        model = {"B0": data["B0"]}
        reference = {"B0": load(reference_path)}
        directions = DESIGN["truths"]["symmetric"]
        observed = clean_observations(reference, directions)
        truth_rms = np.linalg.norm(real_residual(clean_observations(model, directions),
            observed, ["B0"], "full", sigma))
        result = fit(model, observed, ["B0"], "full", "independent",
                     directions, sigma, nstarts=len(DESIGN["multistarts"]))
        self.assertTrue(result["success"])
        self.assertLess(result["rms_noise_units"], .1)
        self.assertLessEqual(result["rms_noise_units"], truth_rms + 1e-7)
        self.assertLess(abs(result["Ua"] - 8.), .01)


if __name__ == "__main__":
    unittest.main()
