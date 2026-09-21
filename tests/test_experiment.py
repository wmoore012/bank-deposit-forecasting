"""Evidence tests run against freshly executed notebook outputs."""
import json, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1]
class ExperimentTests(unittest.TestCase):
    def test_two_editions_agree_on_predictions(self):
        a=pd.read_csv(ROOT/'growth_outputs/masterclass/predictions.csv')
        b=pd.read_csv(ROOT/'growth_outputs/submission/predictions.csv')
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-6,atol=1e-8)
    def test_target_arithmetic_and_consecutive_quarters(self):
        p=pd.read_csv(ROOT/'growth_outputs/masterclass/predictions.csv')
        np.testing.assert_allclose(p.growth,(p.next_deposits-p.DEPDOM)/p.DEPDOM)
        before=pd.to_datetime(p.date).dt.to_period('Q').astype('int64')
        after=pd.to_datetime(p.target_date).dt.to_period('Q').astype('int64')
        self.assertTrue(after.sub(before).eq(1).all())
        self.assertFalse(p.duplicated(['CERT','date']).any())
    def test_score_units_and_tail_sample_sizes(self):
        out=ROOT/'growth_outputs/masterclass'
        p=pd.read_csv(out/'predictions.csv');s=pd.read_csv(out/'historical_scores.csv').set_index('Model')
        for n in ['Zero growth','Persistence','Ridge','MLP']:
            e=p[n]-p.growth
            self.assertAlmostEqual(s.loc[n,'MAE (pp)'],100*abs(e).mean(),places=6)
            self.assertAlmostEqual(s.loc[n,'RMSE (pp)'],100*np.sqrt(np.mean(e**2)),places=6)
        ranking=pd.read_csv(out/'ranking.csv')
        np.testing.assert_allclose(ranking.Precision,ranking.Hits/ranking.Selected)
    def test_frozen_design_and_temporal_boundary(self):
        out=ROOT/'growth_outputs/masterclass'
        f=json.loads((out/'frozen_plan.json').read_text())
        gate=json.loads((out/'comparability_gate.json').read_text())
        self.assertEqual(f['architecture'],[5,32,16,1])
        self.assertEqual(f['seed'],42)
        self.assertEqual(f['target']['target'],'log_growth')
        self.assertEqual(gate['selected_start'],2013)
        self.assertTrue(gate['post_2012_extract_exact'])
        self.assertTrue(f['selection_complete_before_holdout_predictions'])
        splits=pd.read_csv(out/'splits.csv').set_index('Period')
        self.assertLess(splits.loc['Training','Last outcome'],splits.loc['Validation','First predictor'])
        self.assertLess(splits.loc['Validation','Last outcome'],splits.loc['Reused holdout','First predictor'])
    def test_rendering_contract_and_execution(self):
        for name in ['Masterclass','Submission']:
            nb=nbformat.read(ROOT/f'FDIC_Deep_Learning_{name}.ipynb',as_version=4)
            source='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
            self.assertIn('justify-content: center !important',source)
            for c in nb.cells:
                if c.cell_type=='code':
                    self.assertIsNotNone(c.execution_count)
                    self.assertFalse(any(o.output_type=='error' for o in c.outputs))
            html='\n'.join(o.get('data',{}).get('text/html','') for c in nb.cells for o in c.get('outputs',[]))
            self.assertIn('Plotly.newPlot',html)
            self.assertIn('"responsive": false',html)
            self.assertNotIn('src="https://cdn.plot.ly/',html)
            self.assertGreaterEqual(html.count('wm-plot-shell-wrap'),8)
            self.assertIn('justify-content:center',html)
    def test_histograms_remain_numeric_and_scatter_axes_match(self):
        out=ROOT/'growth_outputs/masterclass/charts'
        fig=json.loads((out/'target_full_growth.json').read_text())
        self.assertNotEqual(fig['layout']['xaxis'].get('type'),'category')
        fig=json.loads((out/'actual_predicted.json').read_text())
        self.assertEqual(fig['layout']['hoverlabel']['bgcolor'],'white')
        self.assertEqual(fig['layout']['hoverlabel']['font']['color'],'#182E3A')
        self.assertEqual(fig['layout']['yaxis']['scaleanchor'],'x')
        self.assertEqual(fig['layout']['xaxis']['range'],fig['layout']['yaxis']['range'])

        comparison=json.loads((out/'comparison.json').read_text())
        self.assertTrue(all(trace['type']=='bar' for trace in comparison['data']))
        for trace in comparison['data']:
            self.assertEqual(trace['marker']['color'].count('#007F89'),1)

    def test_eda_inspection_and_time_series_are_saved(self):
        nb=nbformat.read(ROOT/'FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
        source='\n'.join(cell.source for cell in nb.cells if cell.cell_type=='code')
        for check in ['post_conversion.head()', 'post_conversion.info(', 'post_conversion.isna().sum()', '.describe()', 'display_data_chips(', 'style_describe_wm(', 'wm_render_micro_profile_cards(', 'wm_compare_fields(']:
            self.assertIn(check, source)

        charts=ROOT/'growth_outputs/masterclass/charts'
        for name, field in [('reporting_banks_time','Reporting banks'), ('median_deposits_time','Median deposits (USD million)')]:
            fig=json.loads((charts/f'{name}.json').read_text())
            self.assertEqual(fig['data'][0]['type'],'scatter')
            self.assertEqual(fig['data'][0]['mode'],'lines+markers')
            self.assertGreaterEqual(len(fig['data'][0]['x']),48)
            self.assertIn(field, json.dumps(fig['layout']))

        html='\n'.join(output.get('data',{}).get('text/html','') for cell in nb.cells for output in cell.get('outputs',[]))
        self.assertIn('Largest training changes: balances in USD thousands', html)
        self.assertIn('What do the five training inputs look like?', html)
        self.assertNotIn('214,425.0000', html)

        box=json.loads((charts/'cash_by_outcome_box.json').read_text())
        self.assertEqual(len(box['data']),2)
        self.assertTrue(all(trace['type']=='box' for trace in box['data']))
        self.assertEqual({trace['name'] for trace in box['data']},
                         {'Deposits fell','No decline'})
        decline=json.loads((charts/'decline_share_time.json').read_text())
        self.assertEqual(decline['data'][0]['mode'],'lines+markers')

    def test_diagnostic_charts_answer_their_questions(self):
        out=ROOT/'growth_outputs/masterclass/charts'

        tail=json.loads((out/'tail_errors.json').read_text())
        self.assertTrue(all(trace['type']=='scatter' for trace in tail['data']))
        self.assertTrue(all(trace.get('mode')=='lines+markers+text' for trace in tail['data']))

        ranking=json.loads((out/'ranking.json').read_text())
        self.assertTrue(all(trace['type']=='bar' for trace in ranking['data']))
        self.assertEqual(ranking['layout']['yaxis']['range'],[0,0.31])
        labels=[label for trace in ranking['data'] for label in trace['text']]
        self.assertTrue(all(' of ' in label for label in labels))

    def test_saved_tables_keep_human_reading_order(self):
        out=ROOT/'growth_outputs/masterclass'
        model_order=['Zero growth','Persistence','Ridge','MLP']

        scores=pd.read_csv(out/'historical_scores.csv')
        self.assertEqual(scores.Model.tolist(),model_order)

        quarter=pd.read_csv(out/'quarter_scores.csv')
        expected=[]
        for date in sorted(quarter.Quarter.unique()):
            expected.extend((date,model) for model in model_order)
        self.assertEqual(list(zip(quarter.Quarter,quarter.Model)),expected)

        tails=pd.read_csv(out/'tail_errors.csv')
        slice_order=['All','Realized bottom 25%','Realized bottom 10%']
        expected=[]
        for slice_name in slice_order:
            expected.extend((slice_name,model) for model in model_order)
        self.assertEqual(list(zip(tails.Slice,tails.Model)),expected)
if __name__=='__main__':unittest.main()
