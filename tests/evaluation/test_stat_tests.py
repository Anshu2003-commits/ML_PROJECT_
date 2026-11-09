import numpy as np
import pytest

from fxforecast.evaluation.stat_tests import clark_west, diebold_mariano


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_dm_detects_clearly_better_model(rng):
    e_bad, e_good = rng.normal(scale=2.0, size=300), rng.normal(scale=1.0, size=300)
    r = diebold_mariano(e_bad, e_good, horizon=1)
    assert r.statistic > 0 and r.p_value < 0.001 and r.mean_loss_diff > 0


def test_dm_is_symmetric(rng):
    e1, e2 = rng.normal(size=200), rng.normal(scale=1.3, size=200)
    a, b = diebold_mariano(e1, e2, 3), diebold_mariano(e2, e1, 3)
    assert a.statistic == pytest.approx(-b.statistic) and a.p_value == pytest.approx(b.p_value)


def test_dm_hand_computed_h1():
    e1 = np.array([1.0, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    e2 = np.ones(10)
    d = e1 ** 2 - e2 ** 2                     # 0,3,0,3,...
    expected = d.mean() / np.sqrt(d.var() / 10) * np.sqrt((10 + 1 - 2) / 10)
    assert diebold_mariano(e1, e2, 1).statistic == pytest.approx(expected)


def test_dm_size_is_about_right_under_the_null():
    rng = np.random.default_rng(1)
    rejections = [diebold_mariano(rng.normal(size=150), rng.normal(size=150), 1).p_value < 0.05
                  for _ in range(400)]
    assert 0.02 < np.mean(rejections) < 0.09


def test_identical_forecasts_give_no_rejection(rng):
    e = rng.normal(size=50)
    assert diebold_mariano(e, e, 2).p_value == 1.0


def test_clark_west_rejects_when_extra_variable_matters(rng):
    n = 300
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    restricted = np.zeros(n)                   # ignores x
    unrestricted = 0.45 * x                    # uses x
    r = clark_west(y, restricted, unrestricted, horizon=1)
    assert r.p_value < 0.001 and r.statistic > 0


def test_clark_west_does_not_reject_when_extra_variable_is_noise(rng):
    n = 300
    y = rng.normal(size=n)
    restricted = np.zeros(n)
    unrestricted = 0.3 * rng.normal(size=n)    # adds noise only
    assert clark_west(y, restricted, unrestricted, horizon=1).p_value > 0.2


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.ones(5), 1)
    with pytest.raises(ValueError):
        clark_west(np.ones(20), np.ones(20), np.ones(19), 1)
