"""Independent checks on the reader-facing claims and saved evidence."""
import hashlib
import json
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'growth_outputs/masterclass'


class FollowupTests(unittest.TestCase):
    def test_cash_before_models_and_full_timeline(self):
        for edition in ['Masterclass', 'Submission']:
            nb = nbformat.read(ROOT / f'FDIC_Deep_Learning_{edition}.ipynb', 4)
            lead = nb.cells[0].source
            self.assertLess(lead.index('Why examine'), lead.index('Ridge'))
            self.assertLess(lead.index('withdraw'), lead.index('Ridge'))
            self.assertIn('reused historical evaluation period', lead)
            self.assertIn('38 quarters', lead)
            self.assertIn('48 quarters', lead)
        splits = pd.read_csv(OUT / 'splits.csv')
        self.assertEqual(splits.Rows.tolist(), [214425, 13834, 13532])
        self.assertEqual(splits['Predictor quarters'].tolist(), [38, 3, 3])
        early = pd.read_csv(OUT / 'opening_periods.csv')
        self.assertEqual(early['Examples'].tolist(), splits.Rows.tolist())
        raw = pd.read_csv(ROOT / 'data/fdic_financials_2013_2024.csv')
        self.assertEqual(raw.REPDTE.nunique(), 48)

    def test_followup_editions_agree(self):
        for filename in ['followup_predictions.csv', 'followup_scores.csv', 'followup_ranking.csv',
                         'paired_intervals.csv', 'structural_sensitivity.csv']:
            a = pd.read_csv(OUT / filename)
            b = pd.read_csv(ROOT / 'growth_outputs/submission' / filename)
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-8)

    def test_medians_independently_from_earlier_reports(self):
        raw = pd.read_csv(ROOT / 'data/fdic_financials_2013_2024.csv')
        metadata = pd.read_csv(ROOT / 'data/fdic_reporting_2010_2024.csv')
        frame = raw.merge(metadata, on=['CERT', 'REPDTE'], validate='one_to_one')
        frame = frame.loc[frame.BKCLASS.isin(['N', 'NM', 'SM', 'SB', 'SI', 'SL']) & frame.CALLFORM.isin([31, 41, 51])].copy()
        frame['date'] = pd.to_datetime(frame.REPDTE.astype(str))
        frame = frame.sort_values(['CERT', 'date'])
        frame['q'] = frame.date.dt.to_period('Q').astype('int64')
        group = frame.groupby('CERT')
        previous_q, next_q = group.q.shift(), group.q.shift(-1)
        previous_deposits, next_deposits = group.DEPDOM.shift(), group.DEPDOM.shift(-1)
        keep = ((frame.q - previous_q == 1) & (next_q - frame.q == 1)
                & frame.DEPDOM.gt(0) & previous_deposits.gt(0) & next_deposits.gt(0)
                & frame.ASSET.gt(0) & frame.date.le('2022-09-30'))
        train = frame.loc[keep].copy()
        train['log_growth'] = np.log(next_deposits.loc[keep] / train.DEPDOM)
        self.assertEqual(len(train), 214425)
        expected = train.groupby(train.date.dt.quarter).log_growth.median()
        predictions = pd.read_csv(OUT / 'followup_predictions.csv')
        quarters = pd.to_datetime(predictions.date).dt.quarter
        np.testing.assert_allclose(predictions['Seasonal median'], np.expm1(quarters.map(expected)))
        np.testing.assert_allclose(predictions['Training median'], np.expm1(train.log_growth.median()))
        design = json.loads((OUT / 'followup_design.json').read_text())
        self.assertIn('after examining 2024', design['status'])

    def test_rankings_and_exact_random_reference(self):
        p = pd.read_csv(OUT / 'followup_predictions.csv')
        r = pd.read_csv(OUT / 'followup_ranking.csv')
        for row in r.to_dict('records'):
            q = p.loc[p.date.eq(row['Quarter'])]
            k = int(np.ceil(len(q) / 10))
            actual = set(q.sort_values(['growth', 'CERT']).head(k).CERT)
            selected = set(q.sort_values([row['Model'], 'CERT']).head(k).CERT)
            hits = len(actual & selected)
            self.assertEqual(hits, row['Hits'])
            self.assertAlmostEqual(row['Precision'], hits / k)
            self.assertAlmostEqual(row['Expected random hits'], k * len(actual) / len(q))
            self.assertAlmostEqual(row['Lift'], (hits / k) / (len(actual) / len(q)))
        means = pd.read_csv(OUT / 'mean_ranking.csv').set_index('Model')
        np.testing.assert_allclose(means.Lift.sort_index(), r.groupby('Model').Lift.mean().sort_index())
        original = pd.read_csv(OUT / 'ranking.csv')
        zero = original.loc[original.Model.eq('Zero growth')]
        self.assertTrue(zero.Hits.isna().all())
        march = r.loc[r.Quarter.eq('2024-03-31')].set_index('Model')
        self.assertEqual((march.loc['Ridge', 'Banks'], march.loc['Ridge', 'Selected']), (4536, 454))
        self.assertEqual((march.loc['Ridge', 'Hits'], march.loc['MLP', 'Hits']), (112, 109))

    def test_errors_and_bootstrap_receipts(self):
        p = pd.read_csv(OUT / 'followup_predictions.csv')
        s = pd.read_csv(OUT / 'extended_scores.csv')
        for row in s.to_dict('records'):
            errors = 100 * (p[row['Model']] - p.growth)
            self.assertAlmostEqual(row['MAE (pp)'], errors.abs().mean())
            self.assertAlmostEqual(row['RMSE (pp)'], np.sqrt(np.mean(errors**2)))
        self.assertEqual(s.iloc[0].Model, 'Seasonal median')
        intervals = pd.read_csv(OUT / 'paired_intervals.csv')
        samples = pd.read_csv(OUT / 'paired_bootstrap_samples.csv')
        for row in intervals.to_dict('records'):
            sample = samples.loc[samples.Metric.eq(row['Metric']) & samples.Comparison.eq(row['Comparison']), 'Difference']
            self.assertEqual(len(sample), 1000)
            np.testing.assert_allclose(np.quantile(sample, [.025, .975]), [row['95% lower'], row['95% upper']])
            self.assertEqual(row['Banks'], p.CERT.nunique())

    def test_history_groups_and_sensitivity(self):
        history = pd.read_csv(OUT / 'history_length_scores.csv')
        self.assertEqual(set(history['History quarters']), {'2–3', '4–7', '8–15', '16–31', '32+'})
        self.assertTrue(history.groupby('History quarters').Rows.nunique().eq(1).all())
        self.assertTrue(history.groupby('Model').Rows.sum().eq(13532).all())
        events = pd.read_csv(OUT / 'verified_event_rows.csv')
        self.assertEqual(len(events), 4)
        self.assertTrue(pd.to_datetime(events.date).dt.year.eq(2023).all())
        source = json.loads((ROOT / 'sources/provenance.json').read_text())
        self.assertIn('data/fdic_financials_2013_2024.csv', source['files'])
        for name, receipt in source['files'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), receipt['sha256'])


if __name__ == '__main__':
    unittest.main()
