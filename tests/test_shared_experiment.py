"""Behavior and evidence checks for the shared experiment and missing outcomes."""
import ast
import inspect
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import deposit_experiment as core

ROOT = Path(__file__).resolve().parents[1]


class SharedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / 'data/fdic_financials_2013_2024.csv')
        cls.meta = pd.read_csv(ROOT / 'data/fdic_reporting_2010_2024.csv')
        cls.scoring, cls.observed = core.prepare_population(cls.raw, cls.meta)

    def test_coverage_and_original_population(self):
        train, valid, test = core.chronological_splits(self.observed)
        self.assertEqual([len(train), len(valid), len(test)], [214425, 13834, 13532])
        inputs = self.scoring.loc[self.scoring.date.between('2024-01-01', '2024-09-30')]
        coverage = core.outcome_coverage(inputs)
        self.assertEqual(coverage.Unresolved.tolist(), [30, 22, 34])
        self.assertEqual(coverage.Eligible.sum(), 13618)
        self.assertTrue(inputs.loc[inputs['Outcome status'].ne('Usable adjacent outcome'), 'growth'].isna().all())
        expected = pd.read_csv(ROOT / 'growth_outputs/masterclass/predictions.csv')
        np.testing.assert_array_equal(test.CERT, expected.CERT)
        np.testing.assert_allclose(test.growth, expected.growth)

    def test_future_outcomes_cannot_change_inputs_or_preprocessing(self):
        # Change June and later balances; March input eligibility/features must stay identical.
        changed = self.raw.copy()
        changed.loc[changed.REPDTE.ge(20240630), 'DEPDOM'] = 0
        scoring, _ = core.prepare_population(changed, self.meta)
        original_march = self.scoring.loc[self.scoring.date.eq('2024-03-31')]
        changed_march = scoring.loc[scoring.date.eq('2024-03-31')]
        pd.testing.assert_frame_equal(original_march[['CERT', *core.FEATURES]], changed_march[['CERT', *core.FEATURES]])
        train, _, _ = core.chronological_splits(self.observed)
        imputer, scaler = core.fit_preprocessor(train)
        medians, means = imputer.statistics_.copy(), scaler.mean_.copy()
        core.transform_features(changed_march, imputer, scaler)
        np.testing.assert_array_equal(medians, imputer.statistics_)
        np.testing.assert_array_equal(means, scaler.mean_)
        # Removing later reports cannot remove otherwise scoreable March banks either.
        shortened, _ = core.prepare_population(self.raw.loc[self.raw.REPDTE.le(20240331)], self.meta)
        truncated_march = shortened.loc[shortened.date.eq('2024-03-31')]
        pd.testing.assert_frame_equal(original_march[['CERT', *core.FEATURES]].reset_index(drop=True),
                                      truncated_march[['CERT', *core.FEATURES]].reset_index(drop=True))

    def test_shared_code_is_visible_and_runner_imports_normally(self):
        import nbformat
        for edition in ['Masterclass', 'Submission']:
            nb = nbformat.read(ROOT / f'FDIC_Deep_Learning_{edition}.ipynb', 4)
            source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
            tree = ast.parse(source)
            functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            for name in ['fit_network', 'scoring_eligibility', 'chronological_splits', 'compare_training_windows']:
                expected = ast.parse(inspect.getsource(getattr(core, name))).body[0]
                self.assertEqual(ast.dump(functions[name]), ast.dump(expected))
            self.assertIn('But wait.', '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown'))
        tree = ast.parse((ROOT / 'experiments/training_window_sensitivity/run.py').read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in {'exec', 'eval', 'compile'} for n in ast.walk(tree)))

    def test_window_scores_and_prediction_receipts(self):
        reference = pd.read_csv(ROOT / 'experiments/training_window_sensitivity/scores.csv')
        for edition in ['masterclass', 'submission']:
            scores = pd.read_csv(ROOT / f'growth_outputs/{edition}/training_window_scores.csv')
            columns = ['training_start', 'model', 'seed', 'training_rows', 'MAE_pp', 'RMSE_pp', 'bottom_decile_precision']
            pd.testing.assert_frame_equal(scores[columns], reference[columns], check_exact=False, rtol=1e-6, atol=1e-8)
        predictions = pd.read_csv(ROOT / 'experiments/training_window_sensitivity/predictions.csv')
        self.assertEqual(len(predictions), 3 * 3 * 13532)
        for (start, seed), frame in predictions.groupby(['training_start', 'seed']):
            row = reference.loc[reference.training_start.eq(start) & reference.seed.eq(seed)].iloc[0]
            measured = core.scores(frame.growth, frame.prediction)
            self.assertAlmostEqual(row.MAE_pp, measured['MAE (pp)'], places=6)
            self.assertAlmostEqual(row.RMSE_pp, measured['RMSE (pp)'], places=6)

    def test_single_row_final_training_batch(self):
        # 513 rows leaves one example after the first 512-row batch.
        core.tf.config.threading.set_inter_op_parallelism_threads(2)
        core.tf.config.threading.set_intra_op_parallelism_threads(2)
        core.tf.config.experimental.enable_op_determinism()
        x = np.zeros((513, 5), dtype='float32')
        y = np.zeros(513, dtype='float32')
        model, history, _ = core.fit_network(42, x, y, x[:4], y[:4])
        self.assertTrue(np.isfinite(history['loss']).all())
        self.assertEqual(core.predict_log_growth(model, x[:1]).shape, (1,))

    def test_target_units_use_separate_axes(self):
        f = json.loads((ROOT / 'growth_outputs/masterclass/charts/target_ranges.json').read_text())
        axes = {t.get('xaxis', 'x') for t in f['data']}
        self.assertEqual(axes, {'x', 'x2', 'x3'})
        labels = [f['layout'][a]['title']['text'] for a in ['xaxis', 'xaxis2', 'xaxis3']]
        self.assertEqual(len(set(labels)), 3)
        self.assertTrue(all(not f['layout'][a].get('matches') for a in ['xaxis', 'xaxis2', 'xaxis3']))
