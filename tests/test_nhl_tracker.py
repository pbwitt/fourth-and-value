"""A scheduled page rebuild must retain NHL tracking controls and asset order."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class TrackerPages(unittest.TestCase):
    def test_generated_pages_load_tracker_before_board(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            refresh = types.ModuleType('nhl.refresh')
            refresh.ROOT = output
            spec = importlib.util.spec_from_file_location('tracker_page_builder', ROOT / 'scripts/nhl/site.py')
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {'nhl.refresh': refresh}):
                spec.loader.exec_module(module)
            module.build({'season': 20262027, 'last_success_at': None})
            for name in ['props/index.html', 'totals/index.html', 'top.html', 'picks.html']:
                text = (output / 'docs/nhl' / name).read_text()
                board = 'nhl-candidates.js?v=9' if name == 'picks.html' else 'nhl.js?v=5'
                self.assertEqual(text.count('offer-tracker.js?v=1'), 1)
                self.assertIn('offer-tracker.css?v=1', text)
                self.assertLess(text.index('offer-tracker.js?v=1'), text.index(board))
            for name in ['index.html', 'methods.html']:
                self.assertNotIn('offer-tracker.js', (output / 'docs/nhl' / name).read_text())


if __name__ == '__main__':
    unittest.main()
