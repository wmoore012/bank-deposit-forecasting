"""Regression checks for concrete examples and preservation of exploratory evidence."""
import ast
import json
import unittest
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'growth_outputs/masterclass'
EDA={'reporting_banks_time','median_deposits_time','next_report','decline_share_time',
     'training_seasonality','cash_by_outcome_box','feature_correlations','persistence_eda',
     'target_growth_side_by_side','small_denominator_growth','missingness','coverage_time'}
class ConcreteTests(unittest.TestCase):
    def test_eda_and_reading_guides_survive(self):
        for edition in ['Masterclass','Submission']:
            nb=nbformat.read(ROOT/f'FDIC_Deep_Learning_{edition}.ipynb',4)
            source='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
            for name in EDA:
                self.assertIn(name,source)
                self.assertIn('READING_GUIDES['+repr(name)+']',source)
            for inspection in ['post_conversion.info(', 'post_conversion.isna().sum()',
                               '.describe()', 'wm_render_micro_profile_cards(', 'display_data_chips(']:
                self.assertIn(inspection,source)
            self.assertIn('cash_shortfall = withdrawal_request - cash_available',source)
            self.assertIn('assert cash_shortfall == 5',source)
    def test_miniature_errors_explain_reversed_ordering(self):
        s=pd.read_csv(OUT/'metric_illustration.csv',index_col=0)
        self.assertAlmostEqual(s.loc['Forecast A','MAE (pp)'],3)
        self.assertAlmostEqual(s.loc['Forecast A','RMSE (pp)'],np.sqrt(17))
        self.assertLess(s.loc['Forecast A','MAE (pp)'],s.loc['Forecast B','MAE (pp)'])
        self.assertGreater(s.loc['Forecast A','RMSE (pp)'],s.loc['Forecast B','RMSE (pp)'])
    def test_capacity_denominators_and_matching_original(self):
        r=pd.read_csv(OUT/'capacity_results.csv')
        original=pd.read_csv(OUT/'followup_ranking.csv')
        self.assertEqual(len(r),36)
        np.testing.assert_array_equal(r.Selected,np.ceil(r.Capacity*r.Banks))
        np.testing.assert_array_equal(r['Outcome count'],np.ceil(.1*r.Banks))
        np.testing.assert_allclose(r.Precision,r.Hits/r.Selected)
        np.testing.assert_allclose(r.Recall,r.Hits/r['Outcome count'])
        np.testing.assert_allclose(r.Lift,r.Precision/(r['Outcome count']/r.Banks))
        np.testing.assert_allclose(r['Expected random hits'],r.Selected*r['Outcome count']/r.Banks)
        a=r.loc[r.Capacity.eq(.1)].sort_values(['Quarter','Model'])
        b=original.sort_values(['Quarter','Model'])
        np.testing.assert_array_equal(a.Hits,b.Hits)
        self.assertTrue(r.sort_values('Capacity').groupby(['Quarter','Model']).Hits.apply(lambda s:s.is_monotonic_increasing).all())
    def test_real_report_maps_to_five_inputs(self):
        bank=pd.read_csv(OUT/'real_bank_receipt.csv').iloc[0]
        inputs=pd.read_csv(OUT/'real_bank_inputs.csv')
        self.assertEqual(int(bank.CERT),902)
        self.assertEqual(int(bank.REPDTE),20240331)
        expected=[np.log(bank.DEPDOM),bank.CHBAL/bank.ASSET,bank.LNLSNET/bank.ASSET,
                  bank.EQ/bank.ASSET,np.log(bank.DEPDOM/bank['Previous deposits'])]
        np.testing.assert_allclose(inputs['Value before scaling'],expected)
        self.assertGreaterEqual(bank.ASSET-bank.CHBAL-bank.LNLSNET,0)
        other=pd.read_csv(ROOT/'growth_outputs/submission/real_bank_inputs.csv')
        pd.testing.assert_frame_equal(inputs,other)

    def test_merger_date_and_currency(self):
        e=pd.read_csv(OUT/'merger_timeline.csv').iloc[0]
        self.assertEqual(e['Effective merger'],'2023-10-01')
        self.assertEqual((pd.Timestamp(e['Effective merger'])-pd.Timestamp(e['Predictor date'])).days,1)
        self.assertAlmostEqual(e['Reported growth (%)'],100*(e['Next deposits USD']/e['Starting deposits USD']-1))
        for name in ['cash_balance_sheet','withdrawal_shortfall','merger_timeline']:
            f=json.loads((OUT/'charts'/f'{name}.json').read_text())
            self.assertNotIn('$',f['layout']['title']['text'])

if __name__=='__main__':unittest.main()

class ChartSemanticsTests(unittest.TestCase):
    def test_calendar_split_uses_dates_and_all_predictor_quarters(self):
        import json
        chart = json.loads((ROOT / 'growth_outputs/masterclass/charts/split.json').read_text())
        self.assertEqual(chart['layout']['xaxis']['type'], 'date')
        self.assertEqual([len(t['x']) for t in chart['data']], [38, 3, 3])
        self.assertTrue(all(str(x).startswith(('201','202')) for t in chart['data'] for x in t['x']))

    def test_charts_have_titles_and_evidence(self):
        import json
        nb = nbformat.read(ROOT / 'FDIC_Deep_Learning_Masterclass.ipynb', 4)
        names = set()
        for cell in nb.cells:
            if cell.cell_type == 'code':
                for node in ast.walk(ast.parse(cell.source)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'chart' and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        names.add(node.args[1].value)
        for path in [ROOT / 'growth_outputs/masterclass/charts' / (name + '.json') for name in names]:
            figure = json.loads(path.read_text())
            title = figure['layout'].get('title', {}).get('text', '')
            self.assertTrue(title.strip(), path.name)
            self.assertTrue(figure['data'], path.name)
