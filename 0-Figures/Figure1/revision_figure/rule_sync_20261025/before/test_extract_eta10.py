import unittest
from extract_eta10 import select


class ExtractionTests(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertEqual(select([0, 100], [0, -20])[:3], ('eta10_crossing', 50, -10))

    def test_largest_crossing(self):
        self.assertEqual(select([0, 100, 200, 300], [0, -20, 0, -20])[1], 250)

    def test_terminal_gate_overrides_earlier_crossing(self):
        self.assertEqual(select([0, 100, 200], [0, -20, -5])[:2], ('scan_limit', 200))

    def test_never_reaches(self):
        self.assertEqual(select([0, 100], [-1, -9])[:2], ('scan_limit', 100))

    def test_all_above(self):
        self.assertEqual(select([0, 100], [-12, -20])[:2], ('above_target_no_crossing', None))

    def test_start_above_then_crosses(self):
        self.assertEqual(select([0, 100, 200], [-20, 0, -20])[1], 150)

    def test_plateau(self):
        result = select([0, 100, 200, 300], [0, -10, -10, -20])
        self.assertEqual(result[1], 200)
        self.assertEqual(len(result[-1]), 1)

    def test_reversal_uses_largest_eta_not_last_index(self):
        self.assertEqual(select([0, 300, 100, 200], [0, -20, 0, -20])[1], 200)

    def test_final_exact_target(self):
        self.assertEqual(select([0, 100], [0, -10])[:2], ('eta10_crossing', 100))

    def test_boundary_is_scan_maximum(self):
        self.assertEqual(select([0, 300, 100], [0, -8, -5])[:2], ('scan_limit', 300))


if __name__ == '__main__':
    unittest.main()
