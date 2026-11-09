import numpy as np
import pandas as pd
import pytest

from fxforecast.features.scaling import scale_train_test


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(5, 2, (50, 2)), columns=["a", "b"])
    test = pd.DataFrame(rng.normal(5, 2, (10, 2)), columns=["a", "b"], index=range(50, 60))
    return train, test


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_scaler_ignores_test_data(data, method):
    train, test = data
    tr1, _, s1 = scale_train_test(train, test, method)
    tr2, te2, s2 = scale_train_test(train, test * 1000, method)   # wildly different test set
    pd.testing.assert_frame_equal(tr1, tr2)
    assert np.allclose(s1.transform(train.to_numpy()), s2.transform(train.to_numpy()))
    assert te2.index.equals(test.index) and list(te2.columns) == ["a", "b"]


def test_standard_scaling_uses_train_moments(data):
    train, test = data
    tr, te, _ = scale_train_test(train, test, "standard")
    assert np.allclose(tr.mean(), 0) and np.allclose(tr.std(ddof=0), 1)
    expected = (test["a"] - train["a"].mean()) / train["a"].std(ddof=0)
    assert np.allclose(te["a"], expected)


def test_unknown_method(data):
    with pytest.raises(ValueError):
        scale_train_test(*data, method="robust")
