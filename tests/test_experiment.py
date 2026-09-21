"""Evidence tests run against freshly executed notebook outputs."""
import base64, json, unittest
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
            self.assertIn('.wm-table-card:has(td.col4)',source)
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
        fig=json.loads((out/'target_growth_side_by_side.json').read_text())
        self.assertNotEqual(fig['layout']['xaxis'].get('type'),'category')
        self.assertEqual(len(fig['data']),2)
        small=json.loads((out/'small_denominator_growth.json').read_text())
        self.assertEqual(small['layout']['xaxis']['type'],'log')
        self.assertEqual(small['layout']['yaxis']['type'],'log')
        extreme=small['data'][1]
        xs=np.frombuffer(base64.b64decode(extreme['x']['bdata']),dtype='f8')
        ys=np.frombuffer(base64.b64decode(extreme['y']['bdata']),dtype='f8')
        for annotation, x, y in zip(
            small['layout']['annotations'], xs, ys
        ):
            self.assertAlmostEqual(annotation['x'], np.log10(x))
            self.assertAlmostEqual(annotation['y'], np.log10(y))
        fig=json.loads((out/'actual_predicted.json').read_text())
        self.assertEqual(fig['layout']['hoverlabel']['bgcolor'],'white')
        self.assertEqual(fig['layout']['hoverlabel']['font']['color'],'#182E3A')
        self.assertEqual(fig['layout']['yaxis']['scaleanchor'],'x')
        self.assertEqual(fig['layout']['xaxis']['range'],fig['layout']['yaxis']['range'])

        comparison=json.loads((out/'comparison.json').read_text())
        self.assertEqual(len(comparison['data']),8)
        self.assertTrue(all(trace['type']=='scatter' for trace in comparison['data']))
        winners=[trace for trace in comparison['data'] if trace.get('marker',{}).get('color')=='#007F89']
        self.assertEqual(len(winners),2)
        self.assertEqual({trace['y'][0] for trace in winners},{'Zero growth','MLP'})
        scores=pd.read_csv(ROOT/'growth_outputs/masterclass/historical_scores.csv').set_index('Model')
        for panel, column in ((comparison['data'][:4],'MAE (pp)'),
                              (comparison['data'][4:8],'RMSE (pp)')):
            for trace in panel:
                self.assertAlmostEqual(trace['x'][0],scores.loc[trace['y'][0],column],places=10)

        nb=nbformat.read(ROOT/'FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
        scorecard_cell=next(cell for cell in nb.cells if cell.cell_type=='code'
                            and "'comparison'," in cell.source)
        rendered='\n'.join(output.get('data',{}).get('text/html','')
                           for output in scorecard_cell.outputs)
        self.assertNotIn('Not tested',rendered)

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
        self.assertIn('Six largest training changes', html)
        self.assertIn('What do the five training inputs look like?', html)
        self.assertNotIn('214,425.0000', html)

        box=json.loads((charts/'cash_by_outcome_box.json').read_text())
        self.assertEqual(len(box['data']),2)
        self.assertTrue(all(trace['type']=='box' for trace in box['data']))
        self.assertEqual({trace['name'] for trace in box['data']},
                         {'Deposits fell','No decline'})
        decline=json.loads((charts/'decline_share_time.json').read_text())
        self.assertEqual(decline['data'][0]['mode'],'lines+markers')
        seasonality=pd.read_csv(ROOT/'growth_outputs/masterclass/training_seasonality.csv')
        self.assertEqual(set(seasonality.Quarter),{'Q1','Q2','Q3','Q4'})
        self.assertLessEqual(seasonality.Year.max(),2022)
        self.assertGreaterEqual(seasonality.Year.min(),2013)
        self.assertEqual(seasonality['Quarter position'].nunique(),len(seasonality))
        self.assertLess(box['layout']['xaxis']['range'][1],1)

    def test_diagnostic_charts_answer_their_questions(self):
        out=ROOT/'growth_outputs/masterclass/charts'

        tail=json.loads((out/'tail_errors.json').read_text())
        self.assertTrue(all(trace['type']=='scatter' for trace in tail['data']))
        self.assertTrue(all(trace.get('mode')=='lines+markers+text' for trace in tail['data']))

        ranking=json.loads((out/'ranking.json').read_text())
        self.assertEqual(len(ranking['data']),3)
        self.assertTrue(all(trace['type']=='scatter' for trace in ranking['data']))
        self.assertTrue(any(shape['x0']==.1 and shape['x1']==.1
                            for shape in ranking['layout']['shapes']))
        self.assertTrue(all(len(trace['customdata'])==3 for trace in ranking['data']))

    def test_conclusion_and_main_flow_do_not_duplicate_evidence(self):
        out=ROOT/'growth_outputs/masterclass'
        scores=pd.read_csv(out/'historical_scores.csv').set_index('Model')
        self.assertEqual(scores['MAE (pp)'].idxmin(),'Zero growth')
        self.assertEqual(scores['RMSE (pp)'].idxmin(),'MLP')
        quantiles=pd.read_csv(out/'large_error_quantiles.csv').set_index('Model')
        self.assertLess(quantiles.loc['MLP'].iloc[0],quantiles.loc['Zero growth'].iloc[0])
        summary=json.loads((out/'run_summary.json').read_text())
        self.assertIn('did not improve typical forecast error',summary['conclusion'])
        self.assertIn('lowest RMSE',summary['conclusion'])

        nb=nbformat.read(ROOT/'FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
        titles=[cell.source.splitlines()[0] for cell in nb.cells if cell.cell_type=='markdown']
        conclusion_index=next(i for i,title in enumerate(titles) if 'What did we learn' in title)
        audit_index=next(i for i,title in enumerate(titles) if 'What changed in older reports' in title)
        self.assertLess(conclusion_index,audit_index)
        core_source='\n'.join(cell.source for cell in nb.cells[:next(
            i for i,cell in enumerate(nb.cells) if cell.cell_type=='markdown'
            and 'What did we learn' in cell.source
        )] if cell.cell_type=='code')
        self.assertNotIn("table(\n    metrics,",core_source)
        self.assertNotIn("table(\n    ranking_display,",core_source)

    def test_error_tradeoff_and_untested_candidates_are_explicit(self):
        out=ROOT/'growth_outputs/masterclass'
        tradeoff=pd.read_csv(out/'error_tradeoff.csv').set_index('Band')
        scores=pd.read_csv(out/'historical_scores.csv').set_index('Model')

        self.assertEqual(int(tradeoff.Rows.sum()),13532)
        self.assertGreater(tradeoff.loc['Smallest 90%','MLP minus zero (pp)'],0)
        self.assertLess(tradeoff.loc['Largest 1%','MLP minus zero (pp)'],0)
        weighted=np.average(tradeoff['MLP minus zero (pp)'],weights=tradeoff.Rows)
        self.assertAlmostEqual(
            weighted,
            scores.loc['MLP','MAE (pp)']-scores.loc['Zero growth','MAE (pp)'],
            places=6,
        )

        chart=json.loads((out/'charts/error_tradeoff.json').read_text())
        self.assertEqual(chart['data'][0]['marker']['color'],
                         ['#AF7721','#007F89','#007F89'])
        self.assertTrue(any(shape['x0']==0 and shape['x1']==0
                            for shape in chart['layout']['shapes']))

        nb=nbformat.read(ROOT/'FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
        source='\n'.join(cell.source for cell in nb.cells)
        self.assertIn('timesfm_comparison',source)
        self.assertIn('What does the 4.3% improvement buy us?',source)
        self.assertIn('Would anomaly detection give us a better review list?',source)
        self.assertIn('Future-quarter balances cannot enter a score',source)
        self.assertNotIn('TimesFM,', (out/'historical_scores.csv').read_text())

    def test_timesfm_uses_full_holdout_and_separate_extension(self):
        predicted = pd.read_csv(ROOT/'growth_outputs/timesfm_zero_shot_predictions.csv')
        reference = pd.read_csv(ROOT/'growth_outputs/submission/predictions.csv')
        joined = predicted.merge(reference[['CERT','date','growth']], on=['CERT','date'],
                                 how='outer', validate='one_to_one', indicator=True)
        self.assertTrue(joined['_merge'].eq('both').all())
        self.assertEqual(len(joined), 13532)
        error = 100*(joined.TimesFM-joined.growth)
        scores = pd.read_csv(ROOT/'growth_outputs/masterclass/timesfm_comparison.csv').set_index('Model')
        self.assertAlmostEqual(scores.loc['TimesFM','MAE (pp)'], error.abs().mean())
        self.assertAlmostEqual(scores.loc['TimesFM','RMSE (pp)'], np.sqrt(np.square(error).mean()))
        from benchmark_timesfm import build_contexts
        raw = pd.read_csv(ROOT/'data/fdic_financials_2013_2024.csv')
        probe = reference.iloc[[0]].copy()
        before,_ = build_contexts(raw,probe)
        dates = pd.to_datetime(raw.REPDTE.astype(str),format='%Y%m%d')
        raw.loc[dates > pd.Timestamp(probe.iloc[0].date),'DEPDOM'] = 999999999999
        after,_ = build_contexts(raw,probe)
        np.testing.assert_array_equal(before[0],after[0])
        master = nbformat.read(ROOT/'FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
        titles = [c.source for c in master.cells if c.cell_type=='markdown']
        bonus = next(i for i,t in enumerate(titles) if '## Bonus' in t)
        self.assertFalse(any(line.startswith('    ') for line in titles[bonus].splitlines() if line.strip()))
        core = next(i for i,t in enumerate(titles) if '## 14 · What did we learn?' in t)
        self.assertGreater(bonus,core)
        submission = nbformat.read(ROOT/'FDIC_Deep_Learning_Submission.ipynb',as_version=4)
        self.assertFalse(any('## Bonus' in c.source for c in submission.cells))

    def test_chronos_and_source_state_denominators(self):
        reference = pd.read_csv(ROOT / 'growth_outputs/submission/predictions.csv')
        predicted = pd.read_csv(ROOT / 'growth_outputs/chronos_zero_shot_predictions.csv')
        joined = reference.merge(predicted, on=['CERT', 'date', 'DEPDOM', 'target_date'],
                                 how='outer', validate='one_to_one', indicator=True)
        self.assertTrue(joined['_merge'].eq('both').all())
        self.assertEqual(len(joined), 13532)
        timesfm = pd.read_csv(ROOT / 'growth_outputs/timesfm_zero_shot_predictions.csv')
        pd.testing.assert_frame_equal(
            predicted[['CERT', 'date', 'Context start', 'Context quarters']],
            timesfm[['CERT', 'date', 'Context start', 'Context quarters']],
        )
        error = 100 * (joined['Chronos-Bolt'] - joined['growth'])
        scores = pd.read_csv(ROOT / 'growth_outputs/masterclass/foundation_model_comparison.csv').set_index('Model')
        self.assertAlmostEqual(scores.loc['Chronos-Bolt', 'MAE (pp)'], error.abs().mean())
        self.assertAlmostEqual(scores.loc['Chronos-Bolt', 'RMSE (pp)'], np.sqrt(np.square(error).mean()))
        states = pd.read_csv(ROOT / 'growth_outputs/masterclass/uninsured_states.csv')
        np.testing.assert_allclose(states['Missing'] + states['Zero'] + states['Nonzero'], states['Rows'])
        populated = states['API populated'].gt(0)
        np.testing.assert_allclose(states.loc[populated, 'Zero share'],
                                   states.loc[populated, 'Zero values'] / states.loc[populated, 'API populated'])
        chart = json.loads((ROOT / 'growth_outputs/masterclass/charts/uninsured_states.json').read_text())
        self.assertEqual(len(chart['data']), 12)

    def test_finetuning_selection_respects_time_and_validation(self):
        reference = pd.read_csv(ROOT / 'growth_outputs/submission/predictions.csv')
        for key in ['chronos', 'timesfm']:
            path = ROOT / f'growth_outputs/{key}_finetuned_predictions.csv'
            meta = json.loads(path.with_suffix('.json').read_text())
            self.assertTrue(meta['candidate_weights_changed'])
            self.assertLess(meta['trainable_parameters'], meta['total_parameters'])
            self.assertLess(meta['training_unique_rows_seen'], meta['training_rows'])
            self.assertLess(pd.Timestamp(meta['train_last_outcome']), pd.Timestamp('2023-03-31'))
            self.assertLess(pd.Timestamp(meta['validation_last_outcome']), pd.Timestamp('2024-03-31'))
            winner = min(meta['validation'], key=lambda row: row['validation_log_MAE'])
            self.assertEqual(meta['selected_step'], winner['step'])
            selected = pd.read_csv(path)
            joined = reference.merge(selected, on=['CERT', 'date', 'DEPDOM', 'target_date'],
                                     how='outer', validate='one_to_one', indicator=True)
            self.assertTrue(joined['_merge'].eq('both').all())
            self.assertEqual(len(joined), 13532)
            error = 100 * (joined.Prediction - joined.growth)
            self.assertAlmostEqual(meta['MAE (pp)'], error.abs().mean())
            self.assertAlmostEqual(meta['RMSE (pp)'], np.sqrt(np.square(error).mean()))

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
