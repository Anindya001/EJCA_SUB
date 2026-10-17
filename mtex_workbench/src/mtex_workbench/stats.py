"""Statistical methods of the workbench, in NumPy and SciPy only.

Conventions: every interval is two-sided at level `confidence`; every random procedure takes
a `seed` and is reproducible; every function documents its sampling unit, because the
choice of unit (pixel, segment, grain, grain pair) decides whether an interval means what
it claims. Neighbouring EBSD pixels are strongly correlated, so pixel-level intervals are
never reported as if the pixels were independent.

References
  Cox's interval for a lognormal mean: Land (1972) Technometrics 14, 145; Zhou & Gao (1997)
    Stat. Med. 16, 783.
  BCa bootstrap: Efron (1987) JASA 82, 171; Efron & Tibshirani (1993) ch. 14.
  Parametric bootstrap goodness of fit with estimated parameters (the Lilliefors problem):
    Stute, Gonzalez Manteiga & Presedo Quindimil (1993) Test 2, 113.
  Anderson-Darling statistic: Anderson & Darling (1954) JASA 49, 765.
  Wilson score interval: Wilson (1927) JASA 22, 209.
  Otsu threshold: Otsu (1979) IEEE Trans. SMC 9, 62.
  Planimetric grain size number: ASTM E112-13, eq. (2).
  Random misorientation (Mackenzie) distribution: Mackenzie (1958) Biometrika 45, 229.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import special, stats as sps


def _z(confidence: float) -> float:
  if not 0 < confidence < 1:
    raise ValueError("confidence must lie in (0, 1)")
  return float(sps.norm.ppf(0.5 + confidence / 2))


def _positive(x, name="data") -> np.ndarray:
  x = np.asarray(x, float).ravel()
  x = x[np.isfinite(x)]
  if x.size < 3:
    raise ValueError(f"{name}: at least 3 finite values are needed, got {x.size}")
  if np.any(x <= 0):
    raise ValueError(f"{name}: a lognormal fit needs strictly positive values")
  return x


# -- the lognormal grain size model ----------------------------------------------------------

@dataclass
class LognormalFit:
  n: int
  mu: float            # mean of log d (MLE)
  sigma: float         # sd of log d, MLE (divisor n)
  s: float             # sd of log d, unbiased variance (divisor n - 1)
  mu_ci: tuple         # exact t interval
  sigma_ci: tuple      # exact chi-square interval for the population sd of log d
  median: float        # exp(mu)
  median_ci: tuple
  mean: float          # number-weighted mean, exp(mu + s^2 / 2)
  mean_ci: tuple       # Cox's interval
  area_weighted_mean: float  # exp(mu + 5 s^2 / 2), the mean diameter of the grain that a random point falls in
  confidence: float

  def to_dict(self) -> dict:
    return asdict(self)


def lognormal_fit(d, confidence: float = 0.95) -> LognormalFit:
  """maximum likelihood fit of log d ~ N(mu, sigma^2), with intervals that are exact for
  lognormal data (mu, sigma) or accurate to O(1/n) (the mean, by Cox's method)

  If log d ~ N(mu, sigma^2) then the area-weighted mean diameter, with area proportional to
  d^2, is E[d^3] / E[d^2] = exp(mu + 5 sigma^2 / 2)."""
  x = _positive(d, "grain sizes")
  n = x.size
  y = np.log(x)
  mu = float(y.mean())
  var_mle = float(np.mean((y - mu) ** 2))
  s2 = var_mle * n / (n - 1)
  s = np.sqrt(s2)
  a = 1 - confidence
  t = sps.t.ppf(1 - a / 2, n - 1)
  mu_ci = (mu - t * s / np.sqrt(n), mu + t * s / np.sqrt(n))
  chi_lo, chi_hi = sps.chi2.ppf([a / 2, 1 - a / 2], n - 1)
  sigma_ci = (float(np.sqrt((n - 1) * s2 / chi_hi)), float(np.sqrt((n - 1) * s2 / chi_lo)))
  z = _z(confidence)
  centre = mu + s2 / 2
  half = z * np.sqrt(s2 / n + s2 ** 2 / (2 * (n - 1)))
  return LognormalFit(
    n=n, mu=mu, sigma=float(np.sqrt(var_mle)), s=float(s),
    mu_ci=tuple(map(float, mu_ci)), sigma_ci=sigma_ci,
    median=float(np.exp(mu)), median_ci=(float(np.exp(mu_ci[0])), float(np.exp(mu_ci[1]))),
    mean=float(np.exp(centre)), mean_ci=(float(np.exp(centre - half)), float(np.exp(centre + half))),
    area_weighted_mean=float(np.exp(mu + 2.5 * s2)), confidence=confidence)


def ks_statistic(sorted_u: np.ndarray) -> np.ndarray:
  """sup |F_n - F| from F evaluated at the sorted sample, along the last axis"""
  n = sorted_u.shape[-1]
  i = np.arange(1, n + 1)
  return np.maximum((i / n - sorted_u).max(-1), (sorted_u - (i - 1) / n).max(-1))


def ad_statistic(sorted_u: np.ndarray) -> np.ndarray:
  """Anderson-Darling A^2 from F at the sorted sample, along the last axis"""
  n = sorted_u.shape[-1]
  u = np.clip(sorted_u, 1e-300, 1 - 1e-16)
  i = np.arange(1, n + 1)
  return -n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[..., ::-1])), axis=-1)


def lognormal_gof(d, n_boot: int = 999, seed: int = 0) -> dict:
  """Kolmogorov-Smirnov and Anderson-Darling tests of a lognormal model whose parameters
  are estimated from the same data

  The textbook KS p-value assumes a fully specified F and is far too large when mu and
  sigma are fitted. The p-values here are calibrated by the parametric bootstrap: samples
  of the same size are drawn from the fitted model, refitted, and their statistics form the
  null distribution. p = (1 + #{T* >= T}) / (B + 1)."""
  x = _positive(d, "grain sizes")
  y = np.sort(np.log(x))
  n = y.size

  def stat(ys):  # ys sorted along the last axis
    m = ys.mean(-1, keepdims=True)
    sd = ys.std(-1, keepdims=True)  # the MLE, as in the fit
    u = special.ndtr((ys - m) / sd)
    return ks_statistic(u), ad_statistic(u)

  ks, ad = stat(y)
  rng = np.random.default_rng(seed)
  ks_b = np.empty(n_boot)
  ad_b = np.empty(n_boot)
  chunk = max(1, int(2e6 // n))
  for s in range(0, n_boot, chunk):
    m = min(chunk, n_boot - s)
    sim = np.sort(rng.standard_normal((m, n)), axis=-1)  # location-scale invariant: N(0,1) suffices
    ks_b[s:s + m], ad_b[s:s + m] = stat(sim)
  return {
    "n": n, "n_boot": n_boot, "seed": seed,
    "ks": float(ks), "ks_p": float((1 + np.sum(ks_b >= ks)) / (n_boot + 1)),
    "ad": float(ad), "ad_p": float((1 + np.sum(ad_b >= ad)) / (n_boot + 1)),
  }


# -- the bootstrap ---------------------------------------------------------------------------

_VECTOR_STATS = {
  "mean": lambda a: a.mean(-1),
  "median": lambda a: np.median(a, -1),
  "std": lambda a: a.std(-1, ddof=1),
}


def bootstrap_ci(x, statistic: str = "mean", n_boot: int = 4999, confidence: float = 0.95,
                 seed: int = 0, method: str = "bca", weights=None) -> dict:
  """a nonparametric bootstrap interval for a statistic of iid observations

  method 'bca' is the bias-corrected and accelerated interval, second-order accurate;
  'percentile' is the plain percentile interval. statistic is 'mean', 'median', 'std' or,
  with weights, 'weighted_mean' (sum w x / sum w, the pairs (x, w) resampled together; with
  w the grain areas this is the area-weighted mean)."""
  x = np.asarray(x, float).ravel()
  w = None if weights is None else np.asarray(weights, float).ravel()
  ok = np.isfinite(x) if w is None else np.isfinite(x) & np.isfinite(w)
  x = x[ok]
  w = None if w is None else w[ok]
  n = x.size
  if n < 3:
    raise ValueError(f"the bootstrap needs at least 3 observations, got {n}")
  if statistic == "weighted_mean":
    if w is None:
      raise ValueError("weighted_mean needs weights")
    f = lambda idx: (x[idx] * w[idx]).sum(-1) / w[idx].sum(-1)
  else:
    if statistic not in _VECTOR_STATS:
      raise ValueError(f"unknown statistic {statistic!r}")
    g = _VECTOR_STATS[statistic]
    f = lambda idx: g(x[idx])
  theta = float(f(np.arange(n)))
  rng = np.random.default_rng(seed)
  boot = np.empty(n_boot)
  chunk = max(1, int(4e6 // n))
  for s in range(0, n_boot, chunk):
    m = min(chunk, n_boot - s)
    boot[s:s + m] = f(rng.integers(0, n, size=(m, n)))
  a = 1 - confidence
  if method == "percentile":
    lo, hi = np.quantile(boot, [a / 2, 1 - a / 2])
    z0 = acc = 0.0
  elif method == "bca":
    # bias correction; ties counted half so that a discrete statistic is not biased by them
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n_boot
    prop = np.clip(prop, 1 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = float(sps.norm.ppf(prop))
    # acceleration from the jackknife; closed forms for the (weighted) mean
    if statistic == "mean":
      jack = (x.sum() - x) / (n - 1)
    elif statistic == "weighted_mean":
      jack = ((x * w).sum() - x * w) / (w.sum() - w)
    else:
      jack = np.empty(n)
      full = np.arange(n)
      for i in range(n):
        jack[i] = f(np.delete(full, i))
    d = jack.mean() - jack
    den = 6 * (np.sum(d ** 2)) ** 1.5
    acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    zq = sps.norm.ppf([a / 2, 1 - a / 2])
    adj = sps.norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
    lo, hi = np.quantile(boot, adj)
  else:
    raise ValueError(f"unknown method {method!r}")
  return {"statistic": statistic, "estimate": theta, "ci": (float(lo), float(hi)),
          "se": float(boot.std(ddof=1)), "bias": float(boot.mean() - theta), "n": n,
          "n_boot": n_boot, "confidence": confidence, "method": method, "z0": z0,
          "acceleration": acc, "seed": seed}


# -- proportions and reference distributions -------------------------------------------------

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
  """the Wilson score interval for a binomial proportion k / n"""
  if n <= 0:
    return (float("nan"), float("nan"))
  z = _z(confidence)
  p = k / n
  den = 1 + z ** 2 / n
  c = (p + z ** 2 / (2 * n)) / den
  h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
  return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def tabulated_cdf(grid, density):
  """the CDF of a density tabulated on an increasing grid (trapezoidal rule, normalised)"""
  grid = np.asarray(grid, float)
  density = np.clip(np.asarray(density, float), 0, None)
  cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
  if cum[-1] <= 0:
    raise ValueError("the density integrates to zero")
  cum /= cum[-1]
  return lambda t: np.interp(t, grid, cum, left=0.0, right=1.0)


def ks_against(x, cdf) -> dict:
  """the one-sample KS test against a fully specified continuous CDF (exact p-value)"""
  x = np.sort(np.asarray(x, float)[np.isfinite(x)])
  n = x.size
  if n < 1:
    raise ValueError("no finite observations")
  D = float(ks_statistic(cdf(x)))
  return {"n": n, "D": D, "p": float(sps.kstwo.sf(D, n))}


def histogram_density(x, edges):
  """a histogram as a density (integrates to one), with the count per bin"""
  counts, edges = np.histogram(np.asarray(x, float), bins=edges)
  width = np.diff(edges)
  total = counts.sum()
  dens = counts / (total * width) if total else np.zeros_like(width)
  return dens, counts, edges


# -- robust summaries ------------------------------------------------------------------------

def robust_summary(x) -> dict:
  x = np.asarray(x, float).ravel()
  fin = x[np.isfinite(x)]
  out = {"n_total": int(x.size), "n_finite": int(fin.size)}
  if fin.size == 0:
    return out
  q = np.quantile(fin, [0.05, 0.25, 0.5, 0.75, 0.95])
  out.update({"mean": float(fin.mean()), "sd": float(fin.std(ddof=1)) if fin.size > 1 else float("nan"),
              "median": float(q[2]), "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]),
              "q95": float(q[4]), "iqr": float(q[3] - q[1]),
              "mad": float(1.482602218505602 * np.median(np.abs(fin - q[2]))),
              "min": float(fin.min()), "max": float(fin.max())})
  return out


def group_means(values, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(group ids, mean value, count) over finite values, groups <= 0 ignored"""
  v = np.asarray(values, float).ravel()
  g = np.asarray(groups).ravel().astype(np.int64)
  ok = np.isfinite(v) & (g > 0)
  ids, inv, cnt = np.unique(g[ok], return_inverse=True, return_counts=True)
  sums = np.bincount(inv, weights=v[ok], minlength=ids.size)
  return ids, sums / cnt, cnt


# -- images ----------------------------------------------------------------------------------

def otsu_threshold(image, bins: int = 256) -> tuple[float, float]:
  """(threshold, effectiveness) maximising the between-class variance

  With class probabilities w0(t), w1(t) and means m0(t), m1(t) the between-class variance is
  w0 w1 (m0 - m1)^2; effectiveness eta = sigma_B^2 / sigma_T^2 lies in [0, 1] and tells
  how bimodal the histogram is. The threshold returned is the upper edge of the last bin
  of the dark class, so `image > t` is the bright class."""
  x = np.asarray(image, float).ravel()
  x = x[np.isfinite(x)]
  lo, hi = float(x.min()), float(x.max())
  if hi <= lo:
    raise ValueError("the image is constant; there is nothing to threshold")
  counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
  p = counts / counts.sum()
  centres = 0.5 * (edges[1:] + edges[:-1])
  w0 = np.cumsum(p)
  mu_k = np.cumsum(p * centres)
  mu_t = mu_k[-1]
  w1 = 1 - w0
  with np.errstate(divide="ignore", invalid="ignore"):
    sb = (mu_t * w0 - mu_k) ** 2 / (w0 * w1)
  sb[~np.isfinite(sb)] = 0
  k = int(np.argmax(sb[:-1]))
  st = float(np.sum(p * (centres - mu_t) ** 2))
  return float(edges[k + 1]), float(sb[k] / st) if st > 0 else 0.0


def astm_grain_size_number(mean_area_um2: float) -> float:
  """ASTM E112 planimetric grain size number G = 3.321928 log10(N_A) - 2.954, with N_A the
  number of grains per mm^2 at 1x, i.e. 1 / (mean grain area in mm^2)"""
  if not mean_area_um2 > 0:
    raise ValueError("the mean grain area must be positive")
  return float(3.321928 * np.log10(1e6 / mean_area_um2) - 2.954)
