"""The statistical methods: exact results against SciPy, and frequentist properties
(interval coverage, p-value calibration, unbiasedness) by Monte Carlo with fixed seeds.

Monte Carlo tolerances are set at about four binomial standard errors, so a correct method
fails a test with probability well below one in ten thousand."""
import numpy as np
import pytest
from scipy import stats as sps

from mtex_workbench import stats
from mtex_workbench.ops.image_ops import region_table
from mtex_workbench.ops.stats_ops import group_median


def mc_tolerance(p, reps, k=4.0):
  return k * np.sqrt(p * (1 - p) / reps)


# -- lognormal fit ---------------------------------------------------------------------------

def test_lognormal_mle_matches_scipy():
  rng = np.random.default_rng(1)
  d = rng.lognormal(1.2, 0.5, 400)
  f = stats.lognormal_fit(d)
  s, loc, scale = sps.lognorm.fit(d, floc=0)
  assert f.sigma == pytest.approx(s, rel=1e-6)
  assert f.median == pytest.approx(scale, rel=1e-6)
  assert f.area_weighted_mean == pytest.approx(np.exp(f.mu + 2.5 * f.s ** 2))


def test_area_weighted_mean_identity():
  # for log d ~ N(mu, s^2): E[d^3] / E[d^2] = exp(mu + 5 s^2 / 2)
  mu, s = 0.7, 0.45
  d = np.random.default_rng(2).lognormal(mu, s, 2_000_000)
  assert (d ** 3).mean() / (d ** 2).mean() == pytest.approx(np.exp(mu + 2.5 * s ** 2), rel=5e-3)


def test_lognormal_interval_coverage():
  rng = np.random.default_rng(3)
  mu, sigma, n, reps = 1.0, 0.6, 40, 2000
  true_mean = np.exp(mu + sigma ** 2 / 2)
  hit_mu = hit_sigma = hit_mean = 0
  for _ in range(reps):
    f = stats.lognormal_fit(rng.lognormal(mu, sigma, n))
    hit_mu += f.mu_ci[0] <= mu <= f.mu_ci[1]
    hit_sigma += f.sigma_ci[0] <= sigma <= f.sigma_ci[1]
    hit_mean += f.mean_ci[0] <= true_mean <= f.mean_ci[1]
  tol = mc_tolerance(0.95, reps)
  # mu and sigma intervals are exact
  assert abs(hit_mu / reps - 0.95) < tol
  assert abs(hit_sigma / reps - 0.95) < tol
  # Cox's interval is first-order accurate: close to nominal at n = 40
  assert abs(hit_mean / reps - 0.95) < tol + 0.015


def test_lognormal_fit_rejects_bad_input():
  with pytest.raises(ValueError):
    stats.lognormal_fit([1.0, 2.0])
  with pytest.raises(ValueError):
    stats.lognormal_fit([1.0, 2.0, 0.0, 3.0])


# -- goodness of fit -------------------------------------------------------------------------

def test_gof_statistics_match_scipy():
  rng = np.random.default_rng(4)
  y = np.sort(rng.normal(size=60))
  u = sps.norm.cdf(y)
  assert stats.ks_statistic(u) == pytest.approx(sps.kstest(y, "norm").statistic, rel=1e-12)
  # A^2 for a fully specified F, from the definition
  n = y.size
  i = np.arange(1, n + 1)
  a2 = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
  assert stats.ad_statistic(u) == pytest.approx(a2, rel=1e-12)


def test_gof_pvalues_calibrated_under_null():
  """with parameters estimated, the bootstrap p-values are uniform under H0"""
  rng = np.random.default_rng(5)
  reps = 300
  ks_p = np.empty(reps)
  ad_p = np.empty(reps)
  for r in range(reps):
    g = stats.lognormal_gof(rng.lognormal(0.3, 0.8, 50), n_boot=199, seed=r)
    ks_p[r], ad_p[r] = g["ks_p"], g["ad_p"]
  tol = mc_tolerance(0.05, reps)
  assert abs(np.mean(ks_p <= 0.05) - 0.05) < tol
  assert abs(np.mean(ad_p <= 0.05) - 0.05) < tol
  # the naive KS p-value with estimated parameters is conservative: far fewer rejections
  logs = (np.log(rng.lognormal(0.3, 0.8, 50)) for _ in range(reps))
  naive = [sps.kstest((y - y.mean()) / y.std(), sps.norm.cdf).pvalue for y in logs]
  assert np.mean(np.asarray(naive) <= 0.05) < 0.02


def test_gof_has_power_against_weibull():
  rng = np.random.default_rng(6)
  rejections = sum(stats.lognormal_gof(rng.weibull(1.0, 300), n_boot=199, seed=k)["ad_p"] <= 0.05
                   for k in range(30))
  assert rejections >= 27


# -- bootstrap -------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bca", "percentile"])
def test_bootstrap_mean_coverage_skewed(method):
  rng = np.random.default_rng(7)
  reps, n = 600, 40
  hits = 0
  for r in range(reps):
    b = stats.bootstrap_ci(rng.exponential(2.0, n), "mean", n_boot=999, seed=r, method=method)
    hits += b["ci"][0] <= 2.0 <= b["ci"][1]
  # bootstrap intervals undercover slightly for skewed data at small n; BCa less so
  lo = 0.90 if method == "bca" else 0.89
  assert lo - mc_tolerance(0.95, reps) < hits / reps < 0.95 + mc_tolerance(0.95, reps)


def test_bootstrap_weighted_mean_and_jackknife():
  rng = np.random.default_rng(8)
  x, w = rng.lognormal(1, 0.5, 80), rng.uniform(0.5, 2, 80)
  b = stats.bootstrap_ci(x, "weighted_mean", n_boot=999, weights=w)
  assert b["estimate"] == pytest.approx((x * w).sum() / w.sum())
  assert b["ci"][0] < b["estimate"] < b["ci"][1]
  # the closed-form jackknife equals the brute-force one, so the acceleration does too
  jack = np.array([np.delete(x * w, i).sum() / np.delete(w, i).sum() for i in range(x.size)])
  d = jack.mean() - jack
  assert b["acceleration"] == pytest.approx(np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5), rel=1e-9)


def test_bootstrap_is_reproducible():
  x = np.random.default_rng(9).normal(size=50)
  assert stats.bootstrap_ci(x, seed=11)["ci"] == stats.bootstrap_ci(x, seed=11)["ci"]


# -- proportions, reference CDFs -------------------------------------------------------------

def test_wilson_known_values():
  lo, hi = stats.wilson_interval(0, 10)
  assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)
  lo, hi = stats.wilson_interval(5, 10)
  assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_wilson_coverage():
  rng = np.random.default_rng(10)
  p, n, reps = 0.08, 120, 4000
  ks = rng.binomial(n, p, reps)
  hits = sum(lo <= p <= hi for lo, hi in (stats.wilson_interval(k, n) for k in ks))
  assert hits / reps > 0.95 - mc_tolerance(0.95, reps) - 0.01


def test_ks_against_tabulated_reference():
  grid = np.linspace(0, 1, 2001)
  cdf = stats.tabulated_cdf(grid, 2 * grid)  # density 2x on [0, 1], CDF x^2
  x = np.sqrt(np.random.default_rng(11).uniform(size=300))
  r = stats.ks_against(x, cdf)
  ref = sps.kstest(x, lambda t: np.clip(t, 0, 1) ** 2)
  assert r["D"] == pytest.approx(ref.statistic, abs=2e-6)
  assert r["p"] == pytest.approx(ref.pvalue, rel=1e-3)


def test_group_median():
  rng = np.random.default_rng(12)
  keys = rng.integers(0, 30, 500)
  vals = rng.normal(size=500)
  uniq, med, cnt = group_median(keys, vals)
  for k, m, c in zip(uniq, med, cnt):
    assert m == pytest.approx(np.median(vals[keys == k]))
    assert c == np.sum(keys == k)


def test_group_means_ignores_background():
  ids, means, counts = stats.group_means([1, 2, 3, np.nan, 5], [0, 1, 1, 2, 2])
  assert list(ids) == [1, 2] and list(means) == [2.5, 5.0] and list(counts) == [2, 1]


# -- images ----------------------------------------------------------------------------------

def test_otsu_bimodal():
  rng = np.random.default_rng(13)
  img = np.concatenate([rng.normal(0.25, 0.05, 5000), rng.normal(0.75, 0.05, 5000)])
  t, eta = stats.otsu_threshold(img)
  assert 0.45 < t < 0.55
  assert eta > 0.85
  # a unimodal image is poorly separated
  _, eta_uni = stats.otsu_threshold(rng.normal(0.5, 0.1, 10000))
  assert eta_uni < 0.75


def test_astm_grain_size_number():
  # 100 grains per mm^2 at 1x: G = 3.321928 * 2 - 2.954
  assert stats.astm_grain_size_number(1e4) == pytest.approx(3.321928 * 2 - 2.954)


def test_miles_lantuejoul_weights_remove_border_bias():
  """grains of two sizes, equally frequent, placed uniformly: the interior grains are
  biased towards the small ones, the weighted fraction is not"""
  rng = np.random.default_rng(14)
  W = H = 100
  sizes = (6, 50)
  w_small = w_large = n_small = n_large = 0.0
  for _ in range(12000):
    s = sizes[rng.integers(2)]
    r, c = rng.integers(-s + 1, H, 2)  # every placement that intersects the window
    lab = np.zeros((H, W), np.int32)
    lab[max(r, 0):min(r + s, H), max(c, 0):min(c + s, W)] = 1
    t = region_table(lab, 1.0)
    if t["touches_border"][0]:
      continue
    small = s == sizes[0]
    n_small += small
    n_large += not small
    w_small += t["ml_weight"][0] * small
    w_large += t["ml_weight"][0] * (not small)
  # at equal intensities, large grains intersect the window more often, by the ratio of the
  # numbers of intersecting placements (H + s - 1)^2; the sizes were drawn 1:1, so the large
  # ones are reweighted by that ratio to represent the population
  bias = ((H + sizes[1] - 1) / (H + sizes[0] - 1)) ** 2
  raw_fraction_small = n_small / (n_small + n_large * bias)
  weighted_fraction_small = w_small / (w_small + w_large * bias)
  assert raw_fraction_small > 0.6  # border loss removes large grains
  assert weighted_fraction_small == pytest.approx(0.5, abs=0.04)


def test_region_table_geometry():
  lab = np.zeros((20, 30), np.int32)
  lab[5:9, 5:9] = 1      # interior 4 x 4
  lab[0:3, 10:13] = 2    # touches the top border
  t = region_table(lab, pixel_size=0.5)
  assert list(t["pixels"]) == [16, 9]
  assert t["area"][0] == pytest.approx(16 * 0.25)
  assert t["ecd"][0] == pytest.approx(2 * np.sqrt(4 / np.pi))
  assert list(t["touches_border"]) == [False, True]
  assert t["ml_weight"][0] == pytest.approx(600 / ((30 - 4 - 1) * (20 - 4 - 1)))
  assert np.isnan(t["ml_weight"][1])
