"""Verify history coverage, score provenance, and matched review comparisons."""

import ast
import inspect
import json
from pathlib import Path
import unittest
import joblib
import nbformat
import numpy as np
import pandas as pd
import tensorflow as tf
from deposit_experiment import FEATURES, prepare_population
from freeze_forecasts import fingerprint
import review_history as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/review_history/outputs"


class ReviewHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/fdic_financials_2013_2024.csv")
        cls.metadata = pd.read_csv(ROOT / "data/fdic_reporting_2010_2024.csv")
        cls.inputs, _ = prepare_population(cls.raw, cls.metadata)
        cls.rows, cls.values, cls.audit = study.feature_histories(cls.inputs)
        mask = cls.rows.date.between("2024-01-01", "2024-09-30")
        cls.test = cls.rows.loc[mask].reset_index(drop=True)
        cls.histories = cls.values[mask]
        cls.saved = pd.read_csv(OUT / "scores.csv", parse_dates=["date"])
        cls.scores = cls.saved.drop(columns=["CERT", "NAME", "date"])

    def test_audited_population_and_report_order(self):
        self.assertEqual(len(self.test), 13576)
        self.assertEqual(self.test.growth.notna().sum(), 13490)
        unknown = self.audit.loc[
            self.audit.date.between("2024-01-01", "2024-09-30") & ~self.audit.history_eligible
        ]
        self.assertEqual(len(unknown), 42)
        self.assertTrue(unknown.CERT.eq(59283).any())
        for cert in [14, 27010, 27330, 57083]:
            pos = self.test.index[self.test.CERT.eq(cert) & self.test.date.eq("2024-03-31")]
            if len(pos):
                source = self.inputs.loc[
                    self.inputs.CERT.eq(cert) & self.inputs.date.le("2024-03-31")
                ].tail(8)
                self.assertTrue(source.quarter_number.diff().iloc[1:].eq(1).all())
                np.testing.assert_array_equal(
                    self.histories[pos[0]], source[list(FEATURES)].to_numpy()
                )
        original = pd.read_csv(OUT / "original_list_audit.csv")
        missing = original.loc[~original.anomaly_score_available]
        self.assertEqual(missing["Ridge original selected"].sum(), 5)
        self.assertEqual(missing["MLP original selected"].sum(), 7)

    def test_future_outcomes_and_event_labels_cannot_change_histories_or_scores(self):
        changed = self.raw.copy()
        changed.loc[changed.REPDTE.gt(20240331), "DEPDOM"] = 0
        inputs, _ = prepare_population(changed, self.metadata)
        inputs["future_failure"] = True
        rows, values, _ = study.feature_histories(inputs)
        original = self.rows.date.le("2024-03-31")
        retained = rows.date.le("2024-03-31")
        pd.testing.assert_frame_equal(
            self.rows.loc[original, ["CERT", "date"]].reset_index(drop=True),
            rows.loc[retained, ["CERT", "date"]].reset_index(drop=True),
        )
        np.testing.assert_array_equal(self.values[original], values[retained])
        training = self.values[self.rows.date.le("2022-09-30")].reshape(-1, 40)
        scaler = joblib.load(OUT / "models/full_scaler.joblib")
        np.testing.assert_allclose(scaler.mean_, training.mean(axis=0), rtol=1e-12, atol=1e-12)
        pca = joblib.load(OUT / "models/full_pca.joblib")
        before = scaler.transform(self.values[original][-20:].reshape(-1, 40))
        after = scaler.transform(values[retained][-20:].reshape(-1, 40))
        np.testing.assert_array_equal(pca.transform(before), pca.transform(after))

    def test_every_saved_model_reproduces_scores(self):
        for representation, indices in [
            ("full", list(range(5))),
            ("without_size", list(range(1, 5))),
        ]:
            suffix = "" if representation == "full" else " without size"
            scaler = joblib.load(OUT / f"models/{representation}_scaler.joblib")
            X = scaler.transform(self.histories[:, :, indices].reshape(len(self.test), -1))
            pca = joblib.load(OUT / f"models/{representation}_pca.joblib")
            training = self.values[self.rows.date.le("2022-09-30")][:, :, indices]
            standardized = scaler.transform(training.reshape(len(training), -1))
            variances = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
            minimum = int(np.searchsorted(np.cumsum(variances) / variances.sum(), 0.9) + 1)
            self.assertEqual(pca.n_components_, minimum)
            predicted = pca.inverse_transform(pca.transform(X))
            error, shares = study.reconstruction_error(X, predicted, len(indices))
            np.testing.assert_allclose(error, self.scores["PCA" + suffix], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(shares.sum(axis=1), 1)
            for seed in [42, 7, 99]:
                model = joblib.load(OUT / f"models/{representation}_isolation_{seed}.joblib")
                np.testing.assert_allclose(
                    -model.score_samples(X),
                    self.scores[f"Isolation Forest {seed}" + suffix],
                    rtol=1e-10,
                )
                auto = tf.keras.models.load_model(
                    OUT / f"models/{representation}_autoencoder_{seed}.keras"
                )
                reconstructed = np.asarray(auto(X.astype("float32"), training=False))
                error, _ = study.reconstruction_error(X, reconstructed, len(indices))
                np.testing.assert_allclose(
                    error, self.scores[f"Autoencoder {seed}" + suffix], rtol=1e-6, atol=1e-8
                )

    def test_capacities_overlap_and_outcome_denominators(self):
        for population, mask in [
            ("expanded", np.ones(len(self.test), dtype=bool)),
            ("original", self.test.growth.notna().to_numpy()),
        ]:
            rows = self.test.loc[mask].reset_index(drop=True)
            scores = self.scores.loc[mask].reset_index(drop=True)
            flags = study.review_flags(rows, scores)
            for _, quarter in rows.groupby("date"):
                self.assertTrue(
                    flags.loc[quarter.index].sum().eq(int(np.ceil(0.1 * len(quarter)))).all()
                )
            overlap = study.list_overlaps(rows, flags)
            saved = pd.read_csv(OUT / f"{population}_overlap.csv")
            pd.testing.assert_frame_equal(overlap, saved, check_exact=False, rtol=1e-10)
            if population == "original":
                measured = study.outcome_capture(rows, flags)
                pd.testing.assert_frame_equal(
                    measured,
                    pd.read_csv(OUT / "outcome_capture.csv"),
                    check_exact=False,
                    rtol=1e-10,
                )
        tied = pd.DataFrame(
            {"CERT": [9, 3, 1, 4, 7, 8, 6, 5, 2, 10, 11], "date": pd.Timestamp("2024-03-31")}
        )
        selected = study.review_flags(tied, pd.DataFrame({"constant": np.zeros(11)}))
        self.assertEqual(set(tied.loc[selected.constant, "CERT"]), {1, 2})

    def test_score_fingerprint_visible_code_and_execution(self):
        manifest = json.loads((OUT / "score_manifest.json").read_text())
        self.assertEqual(fingerprint(OUT / "scores.csv"), manifest["scores_sha256"])
        self.assertEqual(fingerprint(OUT / "plan.json"), manifest["plan_sha256"])
        for name, digest in manifest["models"].items():
            self.assertEqual(fingerprint(OUT / "models" / name), digest)
        nb = nbformat.read(ROOT / "FDIC_Review_History.ipynb", 4)
        definitions = {
            node.name: node
            for cell in nb.cells
            if cell.cell_type == "code"
            for node in ast.parse(cell.source).body
            if isinstance(node, ast.FunctionDef)
        }
        for name in [
            "feature_histories",
            "standardize_histories",
            "fit_pca",
            "build_autoencoder",
            "fit_autoencoder",
            "review_flags",
            "outcome_capture",
        ]:
            self.assertEqual(
                ast.dump(definitions[name]),
                ast.dump(ast.parse(inspect.getsource(getattr(study, name))).body[0]),
            )
        for cell in nb.cells:
            if cell.cell_type == "code":
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(o.output_type == "error" for o in cell.outputs))

    def test_event_coverage_remains_separate(self):
        folder = ROOT / "sources/failure_coverage"
        provenance = json.loads((folder / "provenance.json").read_text())
        self.assertEqual(fingerprint(folder / "fdic_failure_register.json"), provenance["sha256"])
        ledger = pd.read_csv(folder / "unresolved_outcome_annotations.csv")
        self.assertEqual(len(ledger), 86)
        self.assertEqual(ledger.loc[ledger.CERT.eq(4134), "later_failure_records"].iloc[0], 1)
        self.assertNotIn("failure", " ".join(self.scores.columns).lower())
