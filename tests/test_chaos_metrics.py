from __future__ import annotations

import numpy as np

from astrooracle.chaos_metrics import compute_chaos_metrics


def test_compute_chaos_metrics_bounds():
    rng = np.random.default_rng(7)
    series = rng.normal(size=200)
    m = compute_chaos_metrics(series)
    assert 0.0 <= m.score <= 1.0
    assert isinstance(m.lyapunov_proxy, float)
    assert isinstance(m.rqa_determinism, float)


def test_compute_chaos_metrics_short_series():
    m = compute_chaos_metrics(np.array([1.0, 2.0, 3.0]))
    assert m.score == 0.0
