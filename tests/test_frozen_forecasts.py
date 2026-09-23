"""Saved-model evidence must describe the delivered bytes and original lists."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from freeze_forecasts import fingerprint, selected_keys

ROOT = Path(__file__).resolve().parents[1]


class FrozenForecastTests(unittest.TestCase):
    def test_saved_artifact_hashes(self):
        folder = ROOT / 'growth_outputs/frozen_forecasts'
        manifest = json.loads((folder/'manifest.json').read_text())
        for name, expected in manifest['artifact_hashes'].items():
            self.assertEqual(fingerprint(folder/name), expected)
        for name, expected in manifest['source_hashes'].items():
            self.assertEqual(fingerprint(ROOT/name), expected)
        for result in manifest['verification'].values():
            self.assertEqual(result['reload_max_difference'], 0)
            self.assertTrue(result['original_list_membership_identical'])

    def test_expanded_scores_preserve_original_forecasts(self):
        old = pd.read_csv(ROOT/'growth_outputs/masterclass/predictions.csv', parse_dates=['date'])
        expanded = pd.read_csv(ROOT/'growth_outputs/frozen_forecasts/expanded_predictions.csv', parse_dates=['date'])
        self.assertEqual(len(expanded), 13618)
        self.assertEqual(expanded.growth.isna().sum(), 86)
        joined = old[['CERT','date']].merge(expanded, on=['CERT','date'], validate='one_to_one')
        for method in ['Ridge','MLP']:
            np.testing.assert_allclose(old[method], joined[method], rtol=1e-6, atol=1e-8)
            self.assertEqual(selected_keys(old, old[method]), selected_keys(joined, joined[method]))
