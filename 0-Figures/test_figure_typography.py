"""Typography-only invariants; no plotting or model fits."""
import unittest
from copy import deepcopy
from lxml import etree as ET
from figure_typography import normalize_tree


class SuperscriptTests(unittest.TestCase):
    def tree(self, rise=7, suffix='2'):
        return ET.ElementTree(ET.fromstring(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
          <path d="M 0 0 L 10 20"/>
          <text><tspan x="0" y="0" style="font-size: 10px; font-family: Arial">R</tspan>
          <tspan x="8" y="{-rise}" style="font-size: 7px; font-family: Arial">{suffix}</tspan>
          <tspan x="14" y="3" style="font-size: 7px; font-family: Arial">1</tspan></text></svg>'''))

    def test_only_excessive_superscript_y_changes(self):
        tree = self.tree()
        before = deepcopy(tree)
        changed = normalize_tree(tree)
        self.assertEqual(len(changed), 1)
        for old, new in zip(before.iter(), tree.iter()):
            self.assertEqual(old.tag, new.tag)
            self.assertEqual(old.text, new.text)
            self.assertEqual(old.tail, new.tail)
            if old.get('y') == '-7':
                self.assertEqual(new.get('y'), '-3.400000')
                old.set('y', new.get('y'))
            self.assertEqual(dict(old.attrib), dict(new.attrib))
        self.assertEqual(normalize_tree(tree), [])

    def test_correct_superscript_and_subscript_untouched(self):
        tree = self.tree(rise=3.4)
        before = ET.tostring(tree)
        self.assertEqual(normalize_tree(tree), [])
        self.assertEqual(ET.tostring(tree), before)

    def test_negative_exponent(self):
        self.assertEqual(len(normalize_tree(self.tree(suffix='&#8722;2'))), 1)

    def test_non_numeric_math_not_repositioned(self):
        self.assertEqual(normalize_tree(self.tree(suffix='x')), [])


if __name__ == '__main__':
    unittest.main()
