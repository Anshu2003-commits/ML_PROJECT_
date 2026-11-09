import pandas as pd
import pytest

from fxforecast.evaluation.splits import make_folds, replication_folds, walk_forward_folds

ORIGINS = pd.date_range("1995-04-01", "2022-12-01", freq="MS")


def months(n):
    return pd.DateOffset(months=n)


def assert_no_leakage(folds, horizon):
    for f in folds:
        # every training target (origin + h) is observed by the refit date
        assert f.train.max() + months(horizon) <= f.refit_date
        # forecasting starts at or after the refit date
        assert f.test.min() >= f.refit_date
        assert f.train.intersection(f.test).empty


@pytest.mark.parametrize("h", [1, 3, 6])
def test_walk_forward_has_no_leakage(h):
    origins = ORIGINS[: len(ORIGINS) - h]          # rows with an observed target
    folds = walk_forward_folds(origins, pd.Timestamp("2010-01-01"), 12, h)
    assert_no_leakage(folds, h)


@pytest.mark.parametrize("h", [1, 3, 6])
def test_walk_forward_test_blocks_cover_period_once(h):
    origins = ORIGINS[: len(ORIGINS) - h]
    folds = walk_forward_folds(origins, pd.Timestamp("2010-01-01"), 12, h)
    tested = pd.DatetimeIndex([d for f in folds for d in f.test])
    assert tested.is_unique and tested.is_monotonic_increasing
    assert tested.equals(origins[origins >= "2010-01-01"])


def test_walk_forward_window_expands():
    folds = walk_forward_folds(ORIGINS, pd.Timestamp("2010-01-01"), 12, 1)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[1] - sizes[0] == 12
    assert len(folds) == 13                         # refits 2010-01 .. 2022-01
    assert all(f.train[0] == ORIGINS[0] for f in folds)


@pytest.mark.parametrize("h", [1, 3])
def test_replication_split_matches_paper_dates(h):
    origins = ORIGINS[: len(ORIGINS) - h]
    (fold,) = replication_folds(origins, pd.Timestamp("2016-07-01"), pd.Timestamp("2018-12-01"), h)
    assert_no_leakage([fold], h)
    first_target = fold.test[0] + months(h)
    last_target = fold.test[-1] + months(h)
    assert fold.test[0] == pd.Timestamp("2016-07-01")
    assert first_target == pd.Timestamp("2016-07-01") + months(h)
    assert last_target == pd.Timestamp("2018-12-01")


def test_replication_h1_has_29_test_months():
    (fold,) = replication_folds(ORIGINS, pd.Timestamp("2016-07-01"), pd.Timestamp("2018-12-01"), 1)
    assert len(fold.test) == 29                     # Aug 2016 .. Dec 2018, as in the paper


def test_no_training_data_raises():
    with pytest.raises(ValueError, match="No training rows"):
        make_folds(ORIGINS, [ORIGINS[0]], 1)


def test_refit_after_data_raises():
    with pytest.raises(ValueError, match="No folds"):
        make_folds(ORIGINS, [pd.Timestamp("2030-01-01")], 1)
