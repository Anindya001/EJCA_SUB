#!/usr/bin/env python3
"""Core numerics for two-sided sampling-period design windows.

This module holds the reusable, side-effect-free part of the reproducibility
package: process models, exact sampled-data frequency responses, hardened
phase-margin evaluation, the individual sampling-period bounds, and the
feasible-window assembly with active-constraint reporting.

``validation_ejc.py`` imports this module to build every figure, table and
LaTeX macro used by the manuscript; ``test_validation_ejc.py`` exercises it
directly.

Conventions
-----------
* ``Omega`` (``Om``) denotes digital frequency in rad/sample; continuous
  frequency is ``omega = Omega / T``.
* Phase margins are returned in degrees on a branch-consistent (unwrapped)
  phase, so a loop that has wrapped past -180 deg is reported as a negative
  margin rather than being aliased back into the stable range.
* Every bound carries the assumptions under which it was derived; see
  ``Bound.validity``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

__all__ = [
    "FOPTD", "SOPTD", "PIParams", "PIDParams", "Timing", "Budgets",
    "Bound", "Window",
    "IMMEDIATE", "NEXT_SCAN",
    "simc_pi", "simc_pid_cancelling",
    "zoh_foptd_frequency", "zoh_soptd_frequency",
    "digital_pi_frequency", "tustin_pid_frequency",
    "margin_from_callable", "MarginResult",
    "foptd_pi_margin", "soptd_pid_margin", "soptd_pid_continuous_margin",
    "continuous_pm_deg", "surrogate_pm_deg",
    "foptd_pi_is_stable", "T_min_sampling_zero", "window_is_feasible",
    "foptd_pi_closed_loop_roots", "foptd_pi_stability_limit",
    "sampling_zero_exact", "sampling_zero_first_order",
    "cost_root_update", "cost_root_noise", "T_cost_update", "T_cost_noise",
    "build_window", "REMEDY",
]

# --------------------------------------------------------------------------
# Model, controller and implementation descriptions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FOPTD:
    """First-order-plus-time-delay process ``K exp(-theta s)/(tau s + 1)``."""

    K: float
    tau: float
    theta: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.theta < 0.0:
            raise ValueError("theta must be non-negative")


@dataclass(frozen=True)
class SOPTD:
    """Second-order-plus-time-delay process with natural frequency ``omega_n``."""

    K: float
    zeta: float
    omega_n: float
    theta: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.omega_n <= 0.0 or self.zeta <= 0.0:
            raise ValueError("omega_n and zeta must be positive")
        if self.theta < 0.0:
            raise ValueError("theta must be non-negative")

    @property
    def omega_d(self) -> float:
        """Damped natural frequency; zero for critically/overdamped plants."""
        return self.omega_n * math.sqrt(max(0.0, 1.0 - self.zeta**2))

    @property
    def bandwidth(self) -> float:
        """-3 dB bandwidth of the delay-free second-order factor."""
        u = 1.0 - 2.0 * self.zeta**2
        return self.omega_n * math.sqrt(u + math.sqrt(u * u + 1.0))


@dataclass(frozen=True)
class PIParams:
    Kc: float
    tau_i: float


@dataclass(frozen=True)
class PIDParams:
    Kc: float
    tau_i: float
    tau_d: float


@dataclass(frozen=True)
class Timing:
    """Delay architecture.

    ``kappa`` collects every delay that scales with the sampling period
    (1/2 for the hold, ``c_comp`` for computation, ``c_filt`` for a filter
    whose corner tracks the Nyquist frequency).  ``T_c0`` collects the fixed
    transport delays in seconds.  ``extra_sample_delays`` is the integer
    number of whole samples added in the *exact* discrete model, which must
    be consistent with ``kappa``: immediate update is (0, kappa=1/2), next
    scan is (1, kappa=3/2).
    """

    kappa: float
    T_c0: float = 0.0
    extra_sample_delays: int = 0
    label: str = ""

    def effective_delay(self, theta: float, T: float) -> float:
        return theta + self.kappa * T + self.T_c0


IMMEDIATE = Timing(kappa=0.5, T_c0=0.0, extra_sample_delays=0, label="immediate")
NEXT_SCAN = Timing(kappa=1.5, T_c0=0.0, extra_sample_delays=1, label="next scan")


@dataclass(frozen=True)
class Budgets:
    """Performance and implementation budgets that generate the window.

    Any field left as ``None`` disables the corresponding constraint, which is
    then reported as inactive rather than silently omitted.
    """

    dphi_deg: float | None = 10.0        # allowed phase-margin loss [deg]
    eps_iae: float | None = 0.25         # allowed relative IAE degradation
    mode_fraction: float | None = 0.25   # omega_d*T <= mode_fraction * pi
    bits: int | None = 12                # fractional bits of the stored pole
    eps_coef: float | None = 0.02        # allowed relative coefficient error
    sigma_n: float | None = None         # measurement-noise std for PID
    V_umax: float | None = None          # command-variance budget
    z0_tol: float | None = None          # |z0| tolerance if a zero is cancelled
    T_hardware: tuple[float, ...] = ()   # implementable periods (PLC scan, 1/fsw)


# --------------------------------------------------------------------------
# Tuning rules
# --------------------------------------------------------------------------


def simc_pi(model: FOPTD, lam: float) -> PIParams:
    """SIMC PI settings for an FOPTD model.

    Validity: FOPTD plant, series-form PI, single closed-loop knob ``lam``.
    """
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    Kc = model.tau / (model.K * (lam + model.theta))
    tau_i = min(model.tau, 4.0 * (lam + model.theta))
    return PIParams(Kc=Kc, tau_i=tau_i)


def simc_pid_cancelling(model: SOPTD, lam: float) -> PIDParams:
    """Ideal PID whose zero pair cancels the delay-free SOPTD denominator.

    Validity: benchmark construction only.  Cancelling a lightly damped mode
    is deliberately *not* recommended for design; it is used here because it
    makes the analytical crossover and margin transparent.
    """
    tau_i = 2.0 * model.zeta / model.omega_n
    tau_d = 1.0 / (2.0 * model.zeta * model.omega_n)
    wc = 1.0 / (lam + model.theta)
    Kc = tau_i * wc / model.K
    return PIDParams(Kc=Kc, tau_i=tau_i, tau_d=tau_d)


# --------------------------------------------------------------------------
# Exact sampled-data frequency responses
# --------------------------------------------------------------------------


def _split_delay(theta: float, T: float) -> tuple[int, float]:
    """Split ``theta = m*T + delta`` with ``0 <= delta < T``, robustly."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    m = int(math.floor(theta / T + 1e-13))
    delta = theta - m * T
    if delta < 1e-12 * max(1.0, T):
        delta = 0.0
    if delta >= T - 1e-12 * max(1.0, T):
        m += 1
        delta = 0.0
    return m, delta


def zoh_foptd_frequency(
    omega_d: np.ndarray | float,
    *,
    K: float,
    tau: float,
    theta: float,
    T: float,
    extra_sample_delays: int = 0,
) -> np.ndarray:
    """Exact sampled FOPTD frequency response under a zero-order hold.

    Implements ``G_T(z) = z^-(m+1) (b0 + b1 z^-1)/(1 - a z^-1)`` with the
    arbitrary dead time split as ``theta = m T + delta``.  Exact at the
    sampling instants; it is not a modified-z approximation.
    """
    if tau <= 0.0 or theta < 0.0:
        raise ValueError("tau must be positive and theta non-negative")
    m, delta = _split_delay(theta, T)
    a = math.exp(-T / tau)
    b0 = K * (1.0 - math.exp(-(T - delta) / tau))
    b1 = K * (math.exp(-(T - delta) / tau) - a)
    q = np.exp(-1j * np.asarray(omega_d, dtype=float))
    return q ** (m + 1 + extra_sample_delays) * (b0 + b1 * q) / (1.0 - a * q)


def zoh_soptd_frequency(
    omega_d: np.ndarray | float,
    *,
    K: float,
    zeta: float,
    omega_n: float,
    theta: float,
    T: float,
    extra_sample_delays: int = 0,
) -> np.ndarray | complex:
    """Exact ZOH-sampled SOPTD frequency response with fractional dead time.

    For ``0 < delta < T`` the previously held input acts over the first
    ``delta`` seconds of the interval and the current delayed input over the
    remainder.  The 2x2 resolvent is inverted in closed form so the whole
    frequency grid is evaluated without a Python loop.
    """
    if T <= 0.0 or K <= 0.0 or zeta <= 0.0 or omega_n <= 0.0 or theta < 0.0:
        raise ValueError("invalid SOPTD parameters")
    m, delta = _split_delay(theta, T)

    A = np.array([[0.0, 1.0], [-omega_n**2, -2.0 * zeta * omega_n]])
    B = np.array([[0.0], [K * omega_n**2]])
    I = np.eye(2)
    Phi = expm(A * T)

    def gamma(duration: float) -> np.ndarray:
        if duration <= 1e-14:
            return np.zeros((2, 1))
        return np.linalg.solve(A, (expm(A * duration) - I) @ B)

    gamma_new = gamma(T - delta)
    gamma_old = gamma(T) - gamma_new

    scalar = np.ndim(omega_d) == 0
    Om = np.atleast_1d(np.asarray(omega_d, dtype=float))
    q = np.exp(-1j * Om)
    z = np.exp(1j * Om)

    qn = q ** (m + extra_sample_delays)
    g0 = qn * (gamma_new[0, 0] + q * gamma_old[0, 0])
    g1 = qn * (gamma_new[1, 0] + q * gamma_old[1, 0])

    # (zI - Phi)^-1 for a 2x2 matrix, first row only (C = [1, 0]).
    m00 = z - Phi[0, 0]
    m01 = -Phi[0, 1]
    m10 = -Phi[1, 0]
    m11 = z - Phi[1, 1]
    det = m00 * m11 - m01 * m10
    H = (m11 * g0 - m01 * g1) / det
    return complex(H[0]) if scalar else H


def digital_pi_frequency(
    omega_d: np.ndarray | float,
    *,
    Kc: float,
    tau_i: float,
    T: float,
    method: str,
) -> np.ndarray:
    """Emulated PI controller: backward-Euler or Tustin integration."""
    q = np.exp(-1j * np.asarray(omega_d, dtype=float))
    if method == "backward_euler":
        return Kc * ((1.0 + T / tau_i) - q) / (1.0 - q)
    if method == "tustin":
        r = T / (2.0 * tau_i)
        return Kc * ((1.0 + r) + (-1.0 + r) * q) / (1.0 - q)
    raise ValueError(f"unknown PI discretisation: {method!r}")


def tustin_pid_frequency(
    omega_d: np.ndarray | float,
    *,
    Kc: float,
    tau_i: float,
    tau_d: float,
    T: float,
    N: float | None = None,
) -> np.ndarray:
    """Ideal (``N is None``) or filtered PID under the bilinear transform."""
    q = np.exp(-1j * np.asarray(omega_d, dtype=float))
    s = (2.0 / T) * (1.0 - q) / (1.0 + q)
    deriv = tau_d * s if N is None else tau_d * s / (1.0 + tau_d * s / N)
    return Kc * (1.0 + 1.0 / (tau_i * s) + deriv)


# --------------------------------------------------------------------------
# Hardened phase-margin evaluation
# --------------------------------------------------------------------------


@dataclass
class MarginResult:
    """Worst-case phase margin over every downward unity-gain crossing.

    ``regime`` disambiguates the two very different situations in which no
    crossing exists, which must never be conflated:

    ``"crossings"``     at least one downward unity-gain crossing was found;
                        ``pm_deg`` is the worst of them.
    ``"gain_below"``    ``|L| < 1`` throughout the Nyquist band, so the loop
                        never reaches unity gain.  Phase margin is undefined
                        (reported as ``nan``) but this is the benign case.
    ``"gain_above"``    ``|L| > 1`` throughout the band.  Phase margin is
                        again ``nan``, but here the loop has no margin at all.

    Callers that compare ``pm_deg`` against a requirement inherit the
    conservative behaviour automatically: ``nan >= x`` is ``False``, so a
    degenerate loop is never counted as satisfying a margin constraint.
    """

    pm_deg: float
    omega_c: float            # continuous crossover [rad/s] of the worst crossing
    n_crossings: int
    crossings_deg: tuple[float, ...] = ()
    omega_crossings: tuple[float, ...] = ()
    regime: str = "crossings"

    @property
    def ok(self) -> bool:
        return math.isfinite(self.pm_deg)


def margin_from_callable(
    Lfun: Callable[[np.ndarray | float], np.ndarray | complex],
    *,
    T: float,
    n_grid: int = 4000,
    Om_lo: float = 1e-6,
    Om_hi: float | None = None,
) -> MarginResult:
    """Phase margin of a discrete loop given as a function of ``Omega``.

    All downward crossings of ``|L| = 1`` inside the Nyquist band are located,
    each is refined by Brent's method, and the phase at every refined
    crossover is taken on the *unwrapped* branch so that a loop which has
    already wrapped past -180 deg is reported as a negative margin.  The
    minimum margin over all crossings is returned, together with the crossing
    multiplicity.
    """
    if Om_hi is None:
        Om_hi = math.pi - 1e-9
    Om = np.geomspace(Om_lo, Om_hi, int(n_grid))
    L = np.asarray(Lfun(Om), dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        logmag = np.log(np.abs(L))
    phase = np.unwrap(np.angle(L))

    finite = np.isfinite(logmag)
    if not finite.all():
        logmag = np.where(finite, logmag, np.nan)

    idx = np.where((logmag[:-1] >= 0.0) & (logmag[1:] < 0.0))[0]
    if idx.size == 0:
        finite_lm = logmag[np.isfinite(logmag)]
        if finite_lm.size and np.all(finite_lm > 0.0):
            regime = "gain_above"
        elif finite_lm.size and np.all(finite_lm < 0.0):
            regime = "gain_below"
        else:
            regime = "no_downward_crossing"
        return MarginResult(float("nan"), float("nan"), 0, regime=regime)

    def f(x: float) -> float:
        return float(np.log(np.abs(Lfun(x))))

    margins: list[float] = []
    omegas: list[float] = []
    for i in idx:
        lo, hi = float(Om[i]), float(Om[i + 1])
        try:
            Oc = brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16, maxiter=200)
        except (ValueError, RuntimeError):
            # Fall back to linear interpolation in log-magnitude.
            y1, y2 = logmag[i], logmag[i + 1]
            Oc = lo - y1 * (hi - lo) / (y2 - y1)
        # Branch-consistent phase: pick the 2*pi lap implied by the unwrapped
        # grid phase, then use the exact angle at the refined crossover.
        w = (Oc - lo) / (hi - lo) if hi > lo else 0.0
        phase_ref = (1.0 - w) * phase[i] + w * phase[i + 1]
        phase_exact = float(np.angle(Lfun(Oc)))
        lap = round((phase_ref - phase_exact) / (2.0 * math.pi))
        phase_c = phase_exact + 2.0 * math.pi * lap
        margins.append(math.degrees(math.pi + phase_c))
        omegas.append(float(Oc) / T)

    worst = int(np.argmin(margins))
    return MarginResult(
        pm_deg=float(margins[worst]),
        omega_c=float(omegas[worst]),
        n_crossings=len(margins),
        crossings_deg=tuple(float(v) for v in margins),
        omega_crossings=tuple(omegas),
    )


def foptd_pi_margin(
    *,
    plant: FOPTD,
    ctrl: PIParams,
    T: float,
    method: str,
    timing: Timing = IMMEDIATE,
    n_grid: int = 4000,
) -> MarginResult:
    """Exact discrete phase margin of an FOPTD plant with an emulated PI.

    The plant may differ from the model used to tune ``ctrl`` (frozen
    controller under parameter uncertainty).  ``timing.T_c0`` is folded into
    the process delay and ``timing.extra_sample_delays`` adds whole samples;
    the hold is represented exactly, so ``kappa`` is *not* added again here.
    """
    theta_tot = plant.theta + timing.T_c0

    def Lfun(Om):
        return zoh_foptd_frequency(
            Om, K=plant.K, tau=plant.tau, theta=theta_tot, T=T,
            extra_sample_delays=timing.extra_sample_delays,
        ) * digital_pi_frequency(Om, Kc=ctrl.Kc, tau_i=ctrl.tau_i, T=T, method=method)

    return margin_from_callable(Lfun, T=T, n_grid=n_grid)


def soptd_pid_margin(
    *,
    plant: SOPTD,
    ctrl: PIDParams,
    T: float,
    timing: Timing = IMMEDIATE,
    N: float | None = None,
    n_grid: int = 4000,
) -> MarginResult:
    """Exact discrete phase margin of an SOPTD plant with a Tustin PID."""
    theta_tot = plant.theta + timing.T_c0

    def Lfun(Om):
        return zoh_soptd_frequency(
            Om, K=plant.K, zeta=plant.zeta, omega_n=plant.omega_n,
            theta=theta_tot, T=T,
            extra_sample_delays=timing.extra_sample_delays,
        ) * tustin_pid_frequency(
            Om, Kc=ctrl.Kc, tau_i=ctrl.tau_i, tau_d=ctrl.tau_d, T=T, N=N
        )

    return margin_from_callable(Lfun, T=T, n_grid=n_grid)


# --------------------------------------------------------------------------
# Analytical surrogates
# --------------------------------------------------------------------------


def soptd_pid_continuous_margin(
    *, plant: SOPTD, ctrl: PIDParams, N: float | None = None,
    n_grid: int = 8000,
) -> MarginResult:
    """Continuous-time margin of the *actual* SOPTD/PID loop.

    This is the reference against which the sampled-data phase loss is
    measured.  Using it, rather than the nominal closed-form margin, keeps
    controller/plant modal mismatch out of the reported sampling error: with
    a mismatched PID the loop no longer cancels, and its continuous margin
    already differs from the nominal value by several degrees.
    """

    def Lfun(w):
        jw = 1j * np.asarray(w, dtype=float)
        G = (plant.K * plant.omega_n**2 * np.exp(-plant.theta * jw)
             / (jw**2 + 2.0 * plant.zeta * plant.omega_n * jw + plant.omega_n**2))
        deriv = ctrl.tau_d * jw if N is None else ctrl.tau_d * jw / (
            1.0 + ctrl.tau_d * jw / N)
        Cc = ctrl.Kc * (1.0 + 1.0 / (ctrl.tau_i * jw) + deriv)
        return G * Cc

    scale = max(plant.omega_n, 1.0 / max(ctrl.tau_i, 1e-12))
    return margin_from_callable(
        Lfun, T=1.0, n_grid=n_grid, Om_lo=1e-5 * scale, Om_hi=1e3 * scale
    )


def continuous_pm_deg(theta_actual: float, theta_design: float, lam: float) -> float:
    """Nominal SIMC phase margin, valid when ``tau_i = tau``."""
    return 90.0 - math.degrees(theta_actual / (lam + theta_design))


def surrogate_pm_deg(
    theta_actual: float,
    theta_design: float,
    lam: float,
    T: float,
    kappa: float,
    fixed_delay: float = 0.0,
) -> float:
    """Delay-equivalent sampled-data phase margin (the design surrogate)."""
    return 90.0 - math.degrees(
        (theta_actual + kappa * T + fixed_delay) / (lam + theta_design)
    )


# --------------------------------------------------------------------------
# Exact closed-loop roots and the stability limit
# --------------------------------------------------------------------------


def foptd_pi_closed_loop_roots(
    *, plant: FOPTD, ctrl: PIParams, T: float, method: str,
    timing: Timing = IMMEDIATE,
) -> np.ndarray:
    """Roots in ``q = z^-1`` of the exact FOPTD/PI characteristic polynomial.

    ``1 + L(q) = 0`` is cleared of denominators, giving a finite-order
    polynomial in ``q``.  The loop is stable iff every root satisfies
    ``|q| > 1``.  This avoids any reliance on frequency-grid resolution.
    """
    theta_tot = plant.theta + timing.T_c0
    m, delta = _split_delay(theta_tot, T)
    m += timing.extra_sample_delays
    a = math.exp(-T / plant.tau)
    b0 = plant.K * (1.0 - math.exp(-(T - delta) / plant.tau))
    b1 = plant.K * (math.exp(-(T - delta) / plant.tau) - a)

    if method == "backward_euler":
        c = np.array([ctrl.Kc * (1.0 + T / ctrl.tau_i), -ctrl.Kc])
    elif method == "tustin":
        r = T / (2.0 * ctrl.tau_i)
        c = np.array([ctrl.Kc * (1.0 + r), ctrl.Kc * (-1.0 + r)])
    else:
        raise ValueError(f"unknown PI discretisation: {method!r}")

    # Polynomials in q, ascending powers.
    den = np.polynomial.polynomial.polymul([1.0, -a], [1.0, -1.0])
    num = np.polynomial.polynomial.polymul([b0, b1], c)
    shifted = np.concatenate([np.zeros(m + 1), num])
    n = max(den.size, shifted.size)
    poly = np.zeros(n)
    poly[: den.size] += den
    poly[: shifted.size] += shifted
    return np.polynomial.polynomial.polyroots(poly)


def foptd_pi_is_stable(**kwargs) -> bool:
    roots = foptd_pi_closed_loop_roots(**kwargs)
    if roots.size == 0:
        return True
    return bool(np.min(np.abs(roots)) > 1.0 + 1e-12)


def foptd_pi_stability_limit(
    *, model: FOPTD, lam: float, method: str, timing: Timing = IMMEDIATE,
    T_lo: float = 1e-3, T_hi: float = 60.0, tol: float = 1e-4,
) -> float:
    """Largest sampling period for which the exact discrete loop is stable.

    The controller is tuned once on ``model`` and then held fixed while ``T``
    varies, matching the emulation setting of the paper.  Returns ``nan`` if
    the loop is already unstable at ``T_lo`` or still stable at ``T_hi``.
    """
    ctrl = simc_pi(model, lam)

    def stable(T: float) -> bool:
        return foptd_pi_is_stable(
            plant=model, ctrl=ctrl, T=T, method=method, timing=timing
        )

    if not stable(T_lo):
        return float("nan")
    if stable(T_hi):
        return float("nan")
    lo, hi = T_lo, T_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Sampling zeros
# --------------------------------------------------------------------------


#: Below this value of ``omega_n * T`` the direct numerator formulas lose too
#: many significant digits to be usable and the series form is used instead.
_ZERO_SERIES_CUTOFF = 1e-3


def sampling_zero_exact(zeta: float, omega_n: float, T: float) -> float:
    """Finite zero of the ZOH-discretised delay-free second-order factor.

    Both numerator coefficients are ``O((omega_n T)^2)`` differences of terms
    of order one, so evaluating them directly loses about
    ``2 log10(1/(omega_n T))`` significant digits and underflows to exactly
    zero for very fast sampling.  For ``omega_n T`` below
    ``_ZERO_SERIES_CUTOFF`` the common factor ``(omega_n T)^2 / 2`` is divided
    out analytically and the cancellation-free expansion

    ``b1 / (u^2/2) = 1 - (2/3)x + x^2/3 - u^2/12 + O(T^3)``
    ``b0 / (u^2/2) = 1 - (4/3)x + x^2   - u^2/12 + O(T^3)``

    with ``x = zeta*omega_n*T`` and ``u = omega_n*T`` is used, which reproduces
    ``z0 = -1 + (2/3) zeta omega_n T + O(T^2)`` to first order and is accurate
    to about ``1e-9`` at the cutoff.
    """
    if T <= 0.0:
        raise ValueError("T must be positive")
    a = zeta * omega_n
    u = omega_n * T
    x = a * T

    if u < _ZERO_SERIES_CUTOFF:
        B1 = 1.0 - (2.0 / 3.0) * x + x * x / 3.0 - u * u / 12.0
        B0 = 1.0 - (4.0 / 3.0) * x + x * x - u * u / 12.0
        return -B0 / B1

    E = math.exp(-x)
    wd = omega_n * math.sqrt(max(0.0, 1.0 - zeta * zeta))
    if wd <= 0.0:  # critically damped limit: sin(phi)/phi -> 1
        b1 = 1.0 - E * (1.0 + x)
        b0 = E * E - E * (1.0 - x)
    else:
        phi = wd * T
        b1 = 1.0 - E * (math.cos(phi) + (a / wd) * math.sin(phi))
        b0 = E * E - E * (math.cos(phi) - (a / wd) * math.sin(phi))
    if b1 == 0.0:
        raise FloatingPointError(
            "sampling-zero numerator underflowed; this indicates omega_n*T "
            "below the supported range"
        )
    return -b0 / b1


def sampling_zero_first_order(zeta: float, omega_n: float, T: float) -> float:
    """First-order expansion ``z0 = -1 + (2/3) zeta omega_n T``."""
    return -1.0 + (2.0 / 3.0) * zeta * omega_n * T


def T_min_sampling_zero(zeta: float, omega_n: float, tol: float) -> float:
    """Smallest ``T`` with ``|z0(T)| <= tol`` using the exact zero.

    Validity: only binding for controllers that *cancel* the sampling zero
    (deadbeat, minimum-variance, naive pole placement).  Returns ``0.0`` when
    the tolerance is already met at arbitrarily small ``T`` (it is not, for
    ``tol < 1``, but the guard keeps the caller total).
    """
    if not (0.0 < tol < 1.0):
        raise ValueError("z0 tolerance must lie in (0, 1)")

    def f(T: float) -> float:
        return abs(sampling_zero_exact(zeta, omega_n, T)) - tol

    lo = 1e-9 / max(omega_n, 1e-12)
    hi = 1.0 / max(zeta * omega_n, 1e-12)
    for _ in range(200):
        if f(hi) <= 0.0:
            break
        hi *= 1.5
    else:
        return float("nan")
    if f(lo) <= 0.0:
        return 0.0
    return float(brentq(f, lo, hi, xtol=1e-12))


# --------------------------------------------------------------------------
# Surrogate costs
# --------------------------------------------------------------------------


def cost_root_update(rho_u: float) -> float:
    """Unique positive ``y`` solving ``y^2 (1 + y) = rho_u``."""
    if rho_u <= 0.0:
        raise ValueError("rho_u must be positive")
    hi = max(1.0, rho_u ** (1.0 / 3.0) + 1.0)
    return float(brentq(lambda y: y * y * (1.0 + y) - rho_u, 0.0, hi, xtol=1e-14))


def cost_root_noise(rho_d: float) -> float:
    """Unique positive ``y`` solving ``y^3 (1 + y) = rho_d``."""
    if rho_d <= 0.0:
        raise ValueError("rho_d must be positive")
    hi = max(1.0, rho_d ** 0.25 + 1.0)
    return float(brentq(lambda y: y**3 * (1.0 + y) - rho_d, 0.0, hi, xtol=1e-14))


def T_cost_update(*, theta_0: float, kappa: float, rho_u: float) -> float:
    """Unconstrained minimiser of ``J_u = alpha (theta0 + kappa T)^2 + gamma/T``."""
    return cost_root_update(rho_u) * theta_0 / kappa


def T_cost_noise(*, theta_0: float, kappa: float, rho_d: float) -> float:
    """Unconstrained minimiser of ``J_d = alpha (theta0 + kappa T)^2 + beta/T^2``."""
    return cost_root_noise(rho_d) * theta_0 / kappa


# --------------------------------------------------------------------------
# Bounds and the feasible window
# --------------------------------------------------------------------------

#: Engineering remedy attached to each constraint, used when a window is empty
#: or when the selected period sits on an endpoint.
REMEDY: dict[str, str] = {
    "phase margin": "relax the phase-loss allowance, increase lambda, or reduce "
                    "computation/communication delay (lower kappa or T_c0)",
    "load disturbance": "relax the IAE degradation budget or accept a slower "
                        "closed loop (larger lambda)",
    "mode resolution": "sample faster, or accept that the oscillatory mode is "
                       "not resolved between samples",
    "coefficient resolution": "use more fractional bits, a delta-operator "
                              "realisation, or relax the coefficient tolerance",
    "derivative noise": "add or tighten the derivative filter, reduce tau_D, or "
                        "raise the command-variance budget",
    "sampling zero": "do not cancel the sampling zero; redesign the controller "
                     "so the constraint disappears",
}


@dataclass(frozen=True)
class Bound:
    """One endpoint candidate, with the assumptions it was derived under."""

    name: str
    kind: str            # "lower" or "upper"
    value: float
    validity: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("lower", "upper"):
            raise ValueError("kind must be 'lower' or 'upper'")


@dataclass
class Window:
    """Assembled design window with active-constraint bookkeeping."""

    bounds: list[Bound]
    T_min: float
    T_max: float
    active_lower: str
    active_upper: str
    T_cost: float = float("nan")
    T_sel: float = float("nan")
    T_hardware_feasible: tuple[float, ...] = ()
    T_hardware_selected: float = float("nan")

    @property
    def feasible(self) -> bool:
        return bool(self.T_min <= self.T_max)

    @property
    def width_ratio(self) -> float:
        """``T_max / T_min``; < 1 means the window is empty."""
        if self.T_min <= 0.0:
            return float("inf")
        return self.T_max / self.T_min

    def upper(self, name: str) -> float:
        for b in self.bounds:
            if b.name == name and b.kind == "upper":
                return b.value
        return float("nan")

    def lower(self, name: str) -> float:
        for b in self.bounds:
            if b.name == name and b.kind == "lower":
                return b.value
        return float("nan")

    def diagnosis(self) -> str:
        """Human-readable statement of what governs this window."""
        if not self.feasible:
            return (
                f"infeasible: T_min = {self.T_min:.4g} s ({self.active_lower}) "
                f"exceeds T_max = {self.T_max:.4g} s ({self.active_upper}). "
                f"Remedies: {REMEDY.get(self.active_lower, 'n/a')}; or "
                f"{REMEDY.get(self.active_upper, 'n/a')}."
            )
        where = "interior"
        if math.isfinite(self.T_sel):
            if abs(self.T_sel - self.T_min) <= 1e-12 * max(1.0, self.T_min):
                where = f"clamped to T_min ({self.active_lower})"
            elif abs(self.T_sel - self.T_max) <= 1e-12 * max(1.0, self.T_max):
                where = f"clamped to T_max ({self.active_upper})"
        return (
            f"feasible: T in [{self.T_min:.4g}, {self.T_max:.4g}] s, "
            f"binding upper = {self.active_upper}, binding lower = "
            f"{self.active_lower}, selection {where}."
        )


def _pm_upper(omega_c: float, dphi_deg: float, timing: Timing) -> float:
    """Margin-erosion upper bound; ``nan`` if the fixed delay already spends it."""
    budget = math.radians(dphi_deg) / omega_c - timing.T_c0
    return budget / timing.kappa


def _iae_upper(
    *, model: FOPTD, lam: float, eps: float, timing: Timing
) -> tuple[float, str]:
    """Two-regime load-disturbance upper bound.

    The regime is selected at the no-sampling baseline ``theta_0``, as in
    Proposition 2; the branch label is returned so the caller can report it.
    """
    theta_0 = model.theta + timing.T_c0
    if model.tau <= 4.0 * (lam + theta_0):
        return eps * theta_0 / timing.kappa, "linear"
    return theta_0 * (math.sqrt(1.0 + eps) - 1.0) / timing.kappa, "quadratic"


def _pole_lower(tau: float, bits: int, eps_coef: float) -> float:
    """Model-coefficient resolution lower bound ``tau 2^-(B+1) / eps_c``."""
    return tau * 2.0 ** (-(bits + 1)) / eps_coef


def _deriv_lower(Kc: float, tau_d: float, sigma_n: float, V_umax: float) -> float:
    """Derivative-noise lower bound from an unfiltered backward difference."""
    return math.sqrt(2.0) * Kc * tau_d * sigma_n / math.sqrt(V_umax)


def build_window(
    *,
    model: FOPTD | SOPTD,
    lam: float,
    timing: Timing,
    budgets: Budgets,
    omega_c: float | None = None,
    Kc: float | None = None,
    tau_d: float = 0.0,
    rho_u: float | None = None,
    rho_d: float | None = None,
) -> Window:
    """Assemble ``[T_min, T_max]``, the cost candidate and the projection.

    Every applicable constraint from Table 3 of the paper is evaluated; the
    binding ones are recorded by name so that an empty window can be explained
    and repaired.  ``omega_c`` defaults to the SIMC crossover
    ``1/(lam + theta)``, which is what makes the closed-form margin bound
    valid; supply it explicitly for any other tuning rule.
    """
    is_soptd = isinstance(model, SOPTD)
    theta = model.theta
    if omega_c is None:
        omega_c = 1.0 / (lam + theta)

    bounds: list[Bound] = []

    # ---- upper bounds -----------------------------------------------------
    if budgets.dphi_deg is not None:
        bounds.append(Bound(
            "phase margin", "upper", _pm_upper(omega_c, budgets.dphi_deg, timing),
            "delay-equivalent surrogate at the nominal crossover; exact within "
            "the surrogate because a pure delay does not move the crossover",
        ))

    if budgets.eps_iae is not None and not is_soptd:
        val, regime = _iae_upper(
            model=model, lam=lam, eps=budgets.eps_iae, timing=timing
        )
        bounds.append(Bound(
            "load disturbance", "upper", val,
            f"SIMC retuned on theta_eff, {regime} regime, one-signed error so "
            "IAE = |IE|",
        ))

    if is_soptd and budgets.mode_fraction is not None and model.omega_d > 0.0:
        bounds.append(Bound(
            "mode resolution", "upper",
            budgets.mode_fraction * math.pi / model.omega_d,
            "fidelity rule omega_d T <= f*pi; f = 1 is the strict aliasing "
            "boundary, f = 1/4 is the validated accuracy region",
        ))

    # ---- lower bounds -----------------------------------------------------
    if budgets.bits is not None and budgets.eps_coef is not None:
        tau_rep = model.tau if isinstance(model, FOPTD) else 1.0 / (
            model.zeta * model.omega_n
        )
        bounds.append(Bound(
            "coefficient resolution", "lower",
            _pole_lower(tau_rep, budgets.bits, budgets.eps_coef),
            "representation bound on a near-unity stored pole; not a "
            "controller-level fixed-point stability result",
        ))

    if (
        budgets.sigma_n is not None
        and budgets.V_umax is not None
        and tau_d > 0.0
        and Kc is not None
    ):
        bounds.append(Bound(
            "derivative noise", "lower",
            _deriv_lower(Kc, tau_d, budgets.sigma_n, budgets.V_umax),
            "unfiltered backward difference on white noise; a derivative "
            "filter caps the growth and relaxes this bound",
        ))

    if budgets.z0_tol is not None and is_soptd:
        bounds.append(Bound(
            "sampling zero", "lower",
            T_min_sampling_zero(model.zeta, model.omega_n, budgets.z0_tol),
            "optional; binding only if the digital controller cancels the "
            "sampling zero",
        ))

    uppers = [b for b in bounds if b.kind == "upper" and math.isfinite(b.value)]
    lowers = [b for b in bounds if b.kind == "lower" and math.isfinite(b.value)]

    if uppers:
        ub = min(uppers, key=lambda b: b.value)
        T_max, active_upper = ub.value, ub.name
    else:
        T_max, active_upper = float("inf"), "none"
    if lowers:
        lb = max(lowers, key=lambda b: b.value)
        T_min, active_lower = lb.value, lb.name
    else:
        T_min, active_lower = 0.0, "none"

    win = Window(
        bounds=bounds, T_min=T_min, T_max=T_max,
        active_lower=active_lower, active_upper=active_upper,
    )

    # ---- interior candidate and projection --------------------------------
    theta_0 = theta + timing.T_c0
    T_cost = float("nan")
    if rho_d is not None:
        T_cost = T_cost_noise(theta_0=theta_0, kappa=timing.kappa, rho_d=rho_d)
    elif rho_u is not None:
        T_cost = T_cost_update(theta_0=theta_0, kappa=timing.kappa, rho_u=rho_u)
    win.T_cost = T_cost

    if win.feasible:
        if math.isfinite(T_cost):
            win.T_sel = min(T_max, max(T_min, T_cost))
        else:
            win.T_sel = T_max
        if budgets.T_hardware:
            ok = tuple(
                T for T in sorted(budgets.T_hardware)
                if T_min - 1e-12 <= T <= T_max + 1e-12
            )
            win.T_hardware_feasible = ok
            if ok:
                # Largest admissible hardware period at or below the cost
                # candidate; otherwise the smallest admissible one.
                below = [T for T in ok if T <= win.T_sel + 1e-12]
                win.T_hardware_selected = below[-1] if below else ok[0]
    return win


def window_is_feasible(T_min_candidates: Iterable[float],
                       T_max_candidates: Iterable[float]) -> bool:
    """Feasibility test ``max_i T_min,i <= min_j T_max,j`` (Proposition 5)."""
    lo = max(list(T_min_candidates) or [0.0])
    hi = min(list(T_max_candidates) or [math.inf])
    return lo <= hi
