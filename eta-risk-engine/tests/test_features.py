import unittest

import numpy as np
import pandas as pd

from etarisk.features import (
    FEATURE_COLUMNS,
    TargetEncoder,
    build_features,
    temporal_folds,
    temporal_split,
)
from etarisk.generate import GeneratorConfig, generate_shipments


def _noise_frame(n=4000, n_keys=800, seed=0):
    """A frame where the key carries *no* real signal about the target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lane": [f"L{i}" for i in rng.integers(0, n_keys, n)],
            "planned_transit_h": np.full(n, 100.0),
            "actual_transit_h": 100.0 * rng.normal(1.0, 0.2, n),
        }
    )


class TestTemporalSplitting(unittest.TestCase):
    def test_every_validation_index_is_after_every_training_index(self):
        # This is the test that fails the moment a random split is reintroduced.
        for tr, va in temporal_folds(1000, n_folds=5):
            self.assertLess(int(tr.max()), int(va.min()))

    def test_folds_partition_the_tail_without_overlap(self):
        folds = temporal_folds(1000, n_folds=4, min_train_frac=0.3)
        val = np.concatenate([va for _, va in folds])
        self.assertEqual(val.size, len(set(val.tolist())))
        self.assertEqual(int(val.max()), 999)

    def test_split_rejects_an_unsorted_frame(self):
        df = generate_shipments(GeneratorConfig(n_shipments=800, days=120, seed=1))
        with self.assertRaises(ValueError):
            temporal_split(df.iloc[::-1])

    def test_split_blocks_are_contiguous_in_time(self):
        df = generate_shipments(GeneratorConfig(n_shipments=2000, days=200, seed=1))
        tr, cal, te = temporal_split(df)
        self.assertLessEqual(tr["ship_ts"].max(), cal["ship_ts"].min())
        self.assertLessEqual(cal["ship_ts"].max(), te["ship_ts"].min())
        self.assertEqual(len(tr) + len(cal) + len(te), len(df))


class TestTargetEncoding(unittest.TestCase):
    def test_in_sample_encoding_leaks_but_out_of_fold_does_not(self):
        df = _noise_frame()
        y = (df["actual_transit_h"] / df["planned_transit_h"]).to_numpy()
        enc = TargetEncoder(("lane",), smoothing=0.0)
        in_sample = enc.fit(df).transform(df)
        oof = TargetEncoder(("lane",), smoothing=0.0).oof_transform(df, n_folds=5)
        leak = abs(float(np.corrcoef(in_sample, y)[0, 1]))
        honest = abs(float(np.corrcoef(oof, y)[0, 1]))
        self.assertGreater(leak, 0.4)
        self.assertLess(honest, 0.10)
        self.assertLess(honest, leak)

    def test_unseen_categories_fall_back_to_the_prior(self):
        df = _noise_frame()
        enc = TargetEncoder(("lane",)).fit(df)
        unseen = pd.DataFrame({"lane": ["NOT_A_LANE"], "planned_transit_h": [1.0], "actual_transit_h": [1.0]})
        self.assertAlmostEqual(float(enc.transform(unseen)[0]), enc.prior_, places=9)

    def test_smoothing_pulls_small_groups_toward_the_prior(self):
        df = _noise_frame(n=600, n_keys=300, seed=4)
        low = TargetEncoder(("lane",), smoothing=0.0).fit(df).transform(df)
        high = TargetEncoder(("lane",), smoothing=500.0).fit(df).transform(df)
        prior = float((df["actual_transit_h"] / df["planned_transit_h"]).mean())
        self.assertLess(np.abs(high - prior).mean(), np.abs(low - prior).mean())


class TestBuildFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_shipments(GeneratorConfig(n_shipments=4000, days=300, seed=2))
        cls.tr, cls.cal, cls.te = temporal_split(cls.df)
        cls.X_tr, (cls.X_cal, cls.X_te), cls.encoders = build_features(cls.tr, [cls.cal, cls.te])

    def test_columns_and_shapes(self):
        self.assertEqual(list(self.X_tr.columns), FEATURE_COLUMNS)
        self.assertEqual(list(self.X_te.columns), FEATURE_COLUMNS)
        self.assertEqual(len(self.X_te), len(self.te))

    def test_no_nulls_or_infinities(self):
        for X in (self.X_tr, self.X_cal, self.X_te):
            arr = X.to_numpy(dtype=float)
            self.assertTrue(np.isfinite(arr).all())

    def test_no_target_or_latent_column_reaches_the_design_matrix(self):
        banned = ("actual", "delay", "is_late", "latent", "arrival")
        for col in FEATURE_COLUMNS:
            self.assertFalse(any(b in col for b in banned), col)

    def test_downstream_encodings_come_only_from_the_training_block(self):
        # Refitting the encoder on train alone must reproduce the test encoding
        # exactly; if any test-block label had been used it would not.
        enc = TargetEncoder(("carrier",)).fit(self.tr)
        self.assertTrue(np.allclose(enc.transform(self.te), self.X_te["te_carrier"].to_numpy()))


if __name__ == "__main__":
    unittest.main()
