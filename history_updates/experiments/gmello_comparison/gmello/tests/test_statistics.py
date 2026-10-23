from pathlib import Path
import importlib.util
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comparison_evaluator", ROOT / "evaluate.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class StatisticsTests(unittest.TestCase):
    def test_linked_seed_mean_and_registered_interval(self):
        ids = list(range(128))
        complete = {endpoint: {i: True for i in ids} for endpoint in module.ENDPOINTS}
        complete["REPAIR_0"][0] = False
        result = module.bootstrap(complete, ids)
        rng = np.random.default_rng(2026101211); indices = rng.integers(0, 128, size=(10000,128))
        differences = np.zeros(128); differences[0] = -.5
        expected = np.quantile(differences[indices].mean(axis=1), [.0125,.9875])
        self.assertEqual(result["G_CORE"]["interval"], expected.tolist())
        self.assertEqual(result["G_CORE"]["repair_mean_minus_comparator"], -.5/128)
        self.assertEqual(result["G_FULL"]["interval"], expected.tolist())
        self.assertEqual(result["G_CORE"]["confidence"], .975)
        self.assertFalse(result["G_CORE"]["equivalence_claim"])
    def test_zero_difference_not_equivalence(self):
        ids = list(range(128)); complete = {endpoint: {i: i % 2 == 0 for i in ids} for endpoint in module.ENDPOINTS}
        result = module.bootstrap(complete, ids)
        self.assertEqual(result["G_CORE"]["interval"], [0.,0.])
        self.assertFalse(result["G_CORE"]["equivalence_claim"])

if __name__ == "__main__": unittest.main()
