#!/usr/bin/env python3
"""Reproducible mathematical and numerical checks for the EJC manuscript.

The script generates:
- exact FOPTD pulse-transfer phase-margin validation;
- a multi-case phase-budget benchmark;
- an exact SOPTD sampled-data check with fractional delay;
- a synthetic residual-bootstrap dead-time uncertainty study;
- tables and LaTeX macros used by the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import least_squares, brentq

OUT = Path(__file__).resolve().parent
RNG_SEED = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def zoh_foptd_frequency(
    omega_d: np.ndarray | float,
    *,
    K: float,
    tau: float,
    theta: float,
    T: float,
    extra_sample_delays: int = 0,
) -> np.ndarray:
    """Exact sampled FOPTD frequency response under a ZOH.

    omega_d is the digital frequency in rad/sample. Output is sampled at kT.
    The arbitrary delay is split as theta = m T + delta, 0 <= delta < T.
    """
    if T <= 0 or tau <= 0 or theta < 0:
        raise ValueError("T and tau must be positive; theta must be non-negative")
    m = int(math.floor(theta / T + 1e-13))
    delta = theta - m * T
    if delta < 1e-12:
        delta = 0.0
    if delta >= T - 1e-12:
        m += 1
        delta = 0.0
    a = math.exp(-T / tau)
    b0 = K * (1.0 - math.exp(-(T - delta) / tau))
    b1 = K * (math.exp(-(T - delta) / tau) - a)
    q = np.exp(-1j * np.asarray(omega_d))
    return q ** (m + 1 + extra_sample_delays) * (b0 + b1 * q) / (1.0 - a * q)


def digital_pi_frequency(
    omega_d: np.ndarray | float,
    *,
    Kc: float,
    tau_i: float,
    T: float,
    method: str,
) -> np.ndarray:
    q = np.exp(-1j * np.asarray(omega_d))
    if method == "backward_euler":
        return Kc * ((1.0 + T / tau_i) - q) / (1.0 - q)
    if method == "tustin":
        r = T / (2.0 * tau_i)
        return Kc * ((1.0 + r) + (-1.0 + r) * q) / (1.0 - q)
    raise ValueError(f"unknown method: {method}")


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
    """Exact ZOH sampled SOPTD frequency response.

    The delay is split as theta = m T + delta.  For 0 < delta < T, the
    previous held input acts over the first delta seconds of the sample
    interval and the current delayed input acts over the remaining interval.
    """
    if T <= 0 or K <= 0 or zeta <= 0 or omega_n <= 0 or theta < 0:
        raise ValueError("invalid SOPTD parameters")
    m = int(math.floor(theta / T + 1e-13))
    delta = theta - m * T
    if delta < 1e-12:
        delta = 0.0
    if delta >= T - 1e-12:
        m += 1
        delta = 0.0

    A = np.array([[0.0, 1.0], [-omega_n**2, -2.0*zeta*omega_n]])
    B = np.array([[0.0], [K*omega_n**2]])
    C = np.array([[1.0, 0.0]])
    I = np.eye(2)

    Phi = expm(A*T)

    def gamma(duration: float) -> np.ndarray:
        if duration <= 1e-14:
            return np.zeros((2, 1))
        return np.linalg.solve(A, (expm(A*duration) - I) @ B)

    gamma_new = gamma(T - delta)
    gamma_old = gamma(T) - gamma_new

    scalar = np.ndim(omega_d) == 0
    Om = np.atleast_1d(np.asarray(omega_d, dtype=float))
    q = np.exp(-1j * Om)
    z = np.exp(1j * Om)
    H = np.empty(Om.shape, dtype=complex)
    for idx, (zi, qi) in enumerate(zip(z, q)):
        delayed_input = (qi ** (m + extra_sample_delays)) * gamma_new
        delayed_input += (qi ** (m + 1 + extra_sample_delays)) * gamma_old
        H[idx] = (C @ np.linalg.solve(zi*I - Phi, delayed_input))[0, 0]
    return H[0] if scalar else H


def tustin_pid_frequency(
    omega_d: np.ndarray | float,
    *,
    Kc: float,
    tau_i: float,
    tau_d: float,
    T: float,
) -> np.ndarray:
    """Ideal PID frequency response under the bilinear transform."""
    q = np.exp(-1j * np.asarray(omega_d))
    s_tustin = (2.0/T) * (1.0 - q) / (1.0 + q)
    return Kc * (1.0 + 1.0/(tau_i*s_tustin) + tau_d*s_tustin)


def exact_phase_margin_deg(
    *,
    K: float,
    tau: float,
    theta_actual: float,
    theta_design: float,
    lam: float,
    T: float,
    method: str,
    extra_sample_delays: int = 0,
    fixed_delay: float = 0.0,
    n_grid: int = 9000,
) -> tuple[float, float]:
    """Return phase margin [deg] and crossover [rad/s].

    Controller parameters are fixed using theta_design. fixed_delay is folded
    into the actual process delay for the exact pulse-transfer calculation.
    """
    tau_i = min(tau, 4.0 * (lam + theta_design))
    Kc = tau / (K * (lam + theta_design))
    Om = np.geomspace(1e-7, math.pi - 1e-6, n_grid)
    L = zoh_foptd_frequency(
        Om, K=K, tau=tau, theta=theta_actual + fixed_delay, T=T,
        extra_sample_delays=extra_sample_delays,
    ) * digital_pi_frequency(Om, Kc=Kc, tau_i=tau_i, T=T, method=method)
    logmag = np.log(np.abs(L))
    crossings = np.where((logmag[:-1] >= 0.0) & (logmag[1:] < 0.0))[0]
    if crossings.size == 0:
        return float("nan"), float("nan")
    i = int(crossings[0])
    x1, x2 = Om[i], Om[i + 1]
    y1, y2 = logmag[i], logmag[i + 1]
    Oc = x1 - y1 * (x2 - x1) / (y2 - y1)
    Lv = zoh_foptd_frequency(
        Oc, K=K, tau=tau, theta=theta_actual + fixed_delay, T=T,
        extra_sample_delays=extra_sample_delays,
    ) * digital_pi_frequency(Oc, Kc=Kc, tau_i=tau_i, T=T, method=method)
    phase = float(np.angle(Lv))
    if phase > 0.0:
        phase -= 2.0 * math.pi
    pm = math.pi + phase
    return math.degrees(pm), Oc / T


def exact_soptd_pid_phase_margin_deg(
    *,
    K: float,
    zeta: float,
    omega_n: float,
    theta_actual: float,
    theta_design: float,
    lam: float,
    T: float,
    fixed_delay: float = 0.0,
    extra_sample_delays: int = 0,
    n_grid: int = 9000,
) -> tuple[float, float]:
    """Phase margin for an underdamped SOPTD loop with cancelling ideal PID.

    The continuous PID zero pair is chosen to cancel the delay-free SOPTD
    denominator, leaving an integrator and the process delay in the continuous
    surrogate.  The exact discrete check uses the ZOH plant with fractional
    delay and a Tustin digital PID.
    """
    tau_i = 2.0*zeta/omega_n
    tau_d = 1.0/(2.0*zeta*omega_n)
    wc = 1.0/(lam + theta_design)
    Kc = tau_i * wc / K
    Om = np.geomspace(1e-7, math.pi - 1e-6, n_grid)
    L = zoh_soptd_frequency(
        Om, K=K, zeta=zeta, omega_n=omega_n,
        theta=theta_actual + fixed_delay, T=T,
        extra_sample_delays=extra_sample_delays,
    ) * tustin_pid_frequency(Om, Kc=Kc, tau_i=tau_i, tau_d=tau_d, T=T)
    logmag = np.log(np.abs(L))
    crossings = np.where((logmag[:-1] >= 0.0) & (logmag[1:] < 0.0))[0]
    if crossings.size == 0:
        return float("nan"), float("nan")
    i = int(crossings[0])
    x1, x2 = Om[i], Om[i + 1]
    y1, y2 = logmag[i], logmag[i + 1]
    Oc = x1 - y1 * (x2 - x1) / (y2 - y1)
    Lv = zoh_soptd_frequency(
        Oc, K=K, zeta=zeta, omega_n=omega_n,
        theta=theta_actual + fixed_delay, T=T,
        extra_sample_delays=extra_sample_delays,
    ) * tustin_pid_frequency(Oc, Kc=Kc, tau_i=tau_i, tau_d=tau_d, T=T)
    phase = float(np.angle(Lv))
    if phase > 0.0:
        phase -= 2.0 * math.pi
    pm = math.pi + phase
    return math.degrees(pm), Oc / T


def continuous_pm_deg(theta_actual: float, theta_design: float, lam: float) -> float:
    return 90.0 - math.degrees(theta_actual / (lam + theta_design))


def surrogate_pm_deg(
    theta_actual: float,
    theta_design: float,
    lam: float,
    T: float,
    kappa: float,
    fixed_delay: float = 0.0,
) -> float:
    return 90.0 - math.degrees(
        (theta_actual + kappa * T + fixed_delay) / (lam + theta_design)
    )


def cost_root_update(rho_u: float) -> float:
    """Unique positive y solving y^2(1+y)=rho_u."""
    return brentq(lambda y: y * y * (1.0 + y) - rho_u, 0.0, max(1.0, rho_u ** (1 / 3) + 1.0))


def cost_root_noise(rho_d: float) -> float:
    """Unique positive y solving y^3(1+y)=rho_d."""
    return brentq(lambda y: y ** 3 * (1.0 + y) - rho_d, 0.0, max(1.0, rho_d ** 0.25 + 1.0))


def generate_margin_validation() -> dict[str, float]:
    K, tau, theta, lam = 1.0, 10.0, 2.0, 2.0
    wc = 1.0 / (lam + theta)
    pm0 = continuous_pm_deg(theta, theta, lam)
    T = np.linspace(0.08, 1.55, 80)
    surrogate_im = np.array([surrogate_pm_deg(theta, theta, lam, t, 0.5) for t in T])
    surrogate_ns = np.array([surrogate_pm_deg(theta, theta, lam, t, 1.5) for t in T])
    curves: dict[tuple[str, str], np.ndarray] = {}
    for arch, extra in (("immediate", 0), ("next_scan", 1)):
        for method in ("backward_euler", "tustin"):
            curves[(arch, method)] = np.array([
                exact_phase_margin_deg(
                    K=K, tau=tau, theta_actual=theta, theta_design=theta,
                    lam=lam, T=t, method=method, extra_sample_delays=extra,
                    n_grid=7000,
                )[0]
                for t in T
            ])

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)
    ax[0].plot(T, surrogate_im, "k--", linewidth=1.8, label="delay surrogate, immediate update")
    ax[0].plot(T, curves[("immediate", "backward_euler")], linewidth=1.5, label="exact pulse model, backward Euler")
    ax[0].plot(T, curves[("immediate", "tustin")], linewidth=1.5, label="exact pulse model, Tustin")
    ax[0].axhline(pm0 - 10.0, linestyle=":", linewidth=1.2, label="10 degree loss target")
    ax[0].set_xlabel("sampling period, T [s]")
    ax[0].set_ylabel("phase margin [degree]")
    ax[0].set_title("(a) Immediate-update implementation")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="best", ncol=2)

    err_be_im = curves[("immediate", "backward_euler")] - surrogate_im
    err_tu_im = curves[("immediate", "tustin")] - surrogate_im
    err_be_ns = curves[("next_scan", "backward_euler")] - surrogate_ns
    err_tu_ns = curves[("next_scan", "tustin")] - surrogate_ns
    x = wc * T
    ax[1].plot(x, err_be_im, label="backward Euler, immediate")
    ax[1].plot(x, err_tu_im, label="Tustin, immediate")
    ax[1].plot(x, err_be_ns, label="backward Euler, next scan")
    ax[1].plot(x, err_tu_ns, label="Tustin, next scan")
    ax[1].axhline(0.0, linestyle="--", linewidth=1.0)
    ax[1].set_xlabel(r"normalised period, $\omega_c T$")
    ax[1].set_ylabel("exact minus surrogate margin [degree]")
    ax[1].set_title("(b) Approximation error")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="best", ncol=2)
    fig.savefig(OUT / "fig2_exact_margin_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    mask = x <= math.radians(15.0) * 2.0  # approximately the studied 15-degree immediate budget
    return {
        "running_immediate_be_max_abs_error": float(np.max(np.abs(err_be_im[mask]))),
        "running_immediate_tustin_max_abs_error": float(np.max(np.abs(err_tu_im[mask]))),
        "running_next_be_max_abs_error": float(np.max(np.abs(err_be_ns[x <= 0.18]))),
        "running_next_tustin_max_abs_error": float(np.max(np.abs(err_tu_ns[x <= 0.18]))),
    }


@dataclass
class BenchmarkRow:
    theta_tau: float
    lambda_theta: float
    architecture: str
    method: str
    T_bound: float
    target_pm: float
    exact_pm: float
    error_deg: float
    exact_loss: float


@dataclass
class SoptdRow:
    zeta: float
    omega_n: float
    theta: float
    lam: float
    T_bound: float
    omega_d_T: float
    target_pm: float
    exact_pm: float
    error_deg: float


def generate_benchmark() -> tuple[list[BenchmarkRow], dict[str, dict[str, float]]]:
    K, tau = 1.0, 10.0
    dphi_deg = 10.0
    dphi = math.radians(dphi_deg)
    ratios = [0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    rows: list[BenchmarkRow] = []
    for ratio in ratios:
        theta = ratio * tau
        for lr in (1.0, 2.0):
            lam = lr * theta
            # All selected cases satisfy tau_i=tau; assert the proposition's domain.
            assert tau <= 4.0 * (lam + theta) + 1e-12
            pm0 = continuous_pm_deg(theta, theta, lam)
            for architecture, kappa, extra in (
                ("immediate", 0.5, 0),
                ("next scan", 1.5, 1),
            ):
                T_bound = dphi * (lam + theta) / kappa
                target_pm = pm0 - dphi_deg
                for method in ("backward Euler", "Tustin"):
                    key = "backward_euler" if method == "backward Euler" else "tustin"
                    exact_pm, _ = exact_phase_margin_deg(
                        K=K, tau=tau, theta_actual=theta, theta_design=theta,
                        lam=lam, T=T_bound, method=key,
                        extra_sample_delays=extra, n_grid=9000,
                    )
                    rows.append(BenchmarkRow(
                        theta_tau=ratio,
                        lambda_theta=lr,
                        architecture=architecture,
                        method=method,
                        T_bound=T_bound,
                        target_pm=target_pm,
                        exact_pm=exact_pm,
                        error_deg=exact_pm-target_pm,
                        exact_loss=pm0-exact_pm,
                    ))

    with (OUT / "benchmark_phase_budget.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BenchmarkRow.__annotations__.keys())
        for r in rows:
            w.writerow([getattr(r, k) for k in BenchmarkRow.__annotations__.keys()])

    groups: dict[str, dict[str, float]] = {}
    for arch in ("immediate", "next scan"):
        for method in ("backward Euler", "Tustin"):
            vals = np.array([r.error_deg for r in rows if r.architecture == arch and r.method == method])
            groups[f"{arch}; {method}"] = {
                "mean_error": float(vals.mean()),
                "median_abs_error": float(np.median(np.abs(vals))),
                "max_abs_error": float(np.max(np.abs(vals))),
                "min_error": float(vals.min()),
                "max_error": float(vals.max()),
            }

    # Large, readable stacked benchmark figure for a single-column review PDF.
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.6), constrained_layout=True)
    markers = {"backward Euler": "o", "Tustin": "s"}
    linestyles = {"immediate": "-", "next scan": "--"}
    for architecture in ("immediate", "next scan"):
        for method in ("backward Euler", "Tustin"):
            for lr in (1.0, 2.0):
                rs = [r for r in rows if r.architecture == architecture and r.method == method and r.lambda_theta == lr]
                ax[0].plot(
                    [r.theta_tau for r in rs], [r.error_deg for r in rs],
                    marker=markers[method], linestyle=linestyles[architecture],
                    label=f"{method}, {architecture}, lambda/theta={lr:g}",
                )
    ax[0].axhline(0.0, linestyle=":", linewidth=1.0)
    ax[0].set_xlabel(r"dead-time ratio, $\theta/\tau$")
    ax[0].set_ylabel("exact minus target margin [degree]")
    ax[0].set_title("(a) Error at the analytical 10 degree bound")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8.0, ncol=2)

    labels = list(groups)
    xpos = np.arange(len(labels))
    med = [groups[k]["median_abs_error"] for k in labels]
    mx = [groups[k]["max_abs_error"] for k in labels]
    ax[1].bar(xpos - 0.18, med, width=0.36, label="median absolute error")
    ax[1].bar(xpos + 0.18, mx, width=0.36, label="maximum absolute error")
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([s.replace("; ", "\n") for s in labels], rotation=0)
    ax[1].set_ylabel("margin error [degree]")
    ax[1].set_title("(b) Aggregate over 12 plant/tuning cases")
    ax[1].grid(True, axis="y", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.savefig(OUT / "fig12_phase_budget_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    with (OUT / "benchmark_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    return rows, groups


def generate_soptd_validation() -> tuple[list[SoptdRow], dict[str, float]]:
    K = 1.0
    kappa = 0.5
    dphi_deg = 5.0
    dphi = math.radians(dphi_deg)
    rows: list[SoptdRow] = []
    for zeta in (0.25, 0.40, 0.70):
        for omega_n in (0.70, 1.00):
            theta = 0.50 / omega_n
            lam = 2.00 / omega_n
            T_bound = dphi * (lam + theta) / kappa
            omega_damped = omega_n * math.sqrt(1.0 - zeta*zeta)
            target_pm = continuous_pm_deg(theta, theta, lam) - dphi_deg
            exact_pm, _ = exact_soptd_pid_phase_margin_deg(
                K=K, zeta=zeta, omega_n=omega_n,
                theta_actual=theta, theta_design=theta, lam=lam,
                T=T_bound, n_grid=9000,
            )
            rows.append(SoptdRow(
                zeta=zeta,
                omega_n=omega_n,
                theta=theta,
                lam=lam,
                T_bound=T_bound,
                omega_d_T=omega_damped*T_bound,
                target_pm=target_pm,
                exact_pm=exact_pm,
                error_deg=exact_pm-target_pm,
            ))

    with (OUT / "soptd_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SoptdRow.__annotations__.keys())
        for r in rows:
            w.writerow([getattr(r, k) for k in SoptdRow.__annotations__.keys()])

    vals = np.array([r.error_deg for r in rows])
    summary = {
        "n_cases": len(rows),
        "median_abs_error": float(np.median(np.abs(vals))),
        "max_abs_error": float(np.max(np.abs(vals))),
        "min_error": float(np.min(vals)),
        "max_error": float(np.max(vals)),
        "max_omega_d_T": float(max(r.omega_d_T for r in rows)),
    }
    with (OUT / "soptd_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return rows, summary


def step_response(t: np.ndarray, K: float, tau: float, theta: float) -> np.ndarray:
    x = np.maximum(t - theta, 0.0)
    return np.where(t > theta, K * (1.0 - np.exp(-x / tau)), 0.0)


def fit_foptd_step(t: np.ndarray, y: np.ndarray, x0: np.ndarray) -> np.ndarray:
    fit = least_squares(
        lambda p: step_response(t, p[0], p[1], p[2]) - y,
        x0=x0,
        bounds=([0.2, 1.0, 0.01], [2.0, 50.0, 8.0]),
        max_nfev=2000,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k/n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den
    return centre-half, centre+half


def generate_bootstrap_chance() -> dict[str, float]:
    rng = np.random.default_rng(RNG_SEED)
    true = np.array([1.0, 10.0, 2.0])
    t = np.arange(0.0, 30.0 + 1e-12, 0.5)
    sigma = 0.03
    y_clean = step_response(t, *true)
    y = y_clean + rng.normal(0.0, sigma, t.size)
    estimate = fit_foptd_step(t, y, true.copy())
    residual = y - step_response(t, *estimate)
    n_boot = 1000
    boot = np.empty((n_boot, 3))
    for i in range(n_boot):
        yb = step_response(t, *estimate) + rng.choice(residual, size=t.size, replace=True)
        boot[i] = fit_foptd_step(t, yb, estimate)
    theta_b = boot[:, 2]

    alpha = 0.05
    q95 = float(np.quantile(theta_b, 1.0-alpha))
    theta_mean = float(theta_b.mean())
    theta_design = float(estimate[2])
    lam = theta_design
    fixed_delay = 0.30
    kappa = 0.5
    pm_min = 50.0
    A = math.pi/2.0 - math.radians(pm_min)
    Tcc = (A*(lam + theta_design) - q95 - fixed_delay)/kappa
    Tmean = (A*(lam + theta_design) - theta_mean - fixed_delay)/kappa
    if Tcc <= 0:
        raise RuntimeError("selected bootstrap example is infeasible")

    pm_sur_cc = np.array([
        surrogate_pm_deg(th, theta_design, lam, Tcc, kappa, fixed_delay)
        for th in theta_b
    ])
    pm_sur_mean = np.array([
        surrogate_pm_deg(th, theta_design, lam, Tmean, kappa, fixed_delay)
        for th in theta_b
    ])

    # Exact pulse-transfer checks with the fixed controller. The fixed delay is
    # folded into the process delay; the ZOH is represented exactly, so kappa's
    # half-sample term is not added separately.
    pm_be_cc = np.array([
        exact_phase_margin_deg(
            K=1.0, tau=10.0, theta_actual=th, theta_design=theta_design,
            lam=lam, T=Tcc, method="backward_euler", fixed_delay=fixed_delay,
            n_grid=6000,
        )[0] for th in theta_b
    ])
    pm_tu_cc = np.array([
        exact_phase_margin_deg(
            K=1.0, tau=10.0, theta_actual=th, theta_design=theta_design,
            lam=lam, T=Tcc, method="tustin", fixed_delay=fixed_delay,
            n_grid=6000,
        )[0] for th in theta_b
    ])

    def coverage(vals: np.ndarray) -> float:
        return float(np.mean(vals >= pm_min))

    # Numerical guard period for backward Euler: largest 0.005-s grid point
    # that reaches at least the nominal 95% empirical coverage.
    guard_grid = np.arange(max(0.05, Tcc-0.20), Tcc+0.001, 0.005)
    guard_cov = []
    for Tg in guard_grid:
        vals = np.array([
            exact_phase_margin_deg(
                K=1.0, tau=10.0, theta_actual=th, theta_design=theta_design,
                lam=lam, T=float(Tg), method="backward_euler",
                fixed_delay=fixed_delay, n_grid=4500,
            )[0] for th in theta_b
        ])
        guard_cov.append(coverage(vals))
    feasible_guard = [float(Tg) for Tg, c in zip(guard_grid, guard_cov) if c >= 0.95]
    Tguard = max(feasible_guard) if feasible_guard else float(guard_grid[0])
    pm_be_guard = np.array([
        exact_phase_margin_deg(
            K=1.0, tau=10.0, theta_actual=th, theta_design=theta_design,
            lam=lam, T=Tguard, method="backward_euler",
            fixed_delay=fixed_delay, n_grid=6000,
        )[0] for th in theta_b
    ])

    arrays = {
        "surrogate, quantile period": pm_sur_cc,
        "exact BE, quantile period": pm_be_cc,
        "exact Tustin, quantile period": pm_tu_cc,
        "exact BE, guarded period": pm_be_guard,
        "surrogate, mean period": pm_sur_mean,
    }
    cov = {k: coverage(v) for k, v in arrays.items()}

    with (OUT / "bootstrap_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "tau", "theta"])
        w.writerows(boot.tolist())

    # Readable 2x2 figure.
    fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    ax[0,0].plot(t, y_clean, linewidth=1.5, label="ground truth")
    ax[0,0].scatter(t, y, s=13, label="synthetic observations")
    ax[0,0].plot(t, step_response(t, *estimate), linestyle="--", linewidth=1.6, label="FOPTD fit")
    ax[0,0].set_xlabel("time [s]")
    ax[0,0].set_ylabel("step response")
    ax[0,0].set_title("(a) Synthetic identification data")
    ax[0,0].grid(True, alpha=0.3)
    ax[0,0].legend()

    ax[0,1].hist(theta_b, bins=32, density=True, alpha=0.75)
    ax[0,1].axvline(theta_mean, linestyle="--", linewidth=1.4, label="bootstrap mean")
    ax[0,1].axvline(q95, linestyle=":", linewidth=1.8, label="upper 95% quantile")
    ax[0,1].axvline(true[2], linewidth=1.2, label="ground truth")
    ax[0,1].set_xlabel(r"dead-time estimate, $\theta$ [s]")
    ax[0,1].set_ylabel("density")
    ax[0,1].set_title("(b) Residual-bootstrap uncertainty")
    ax[0,1].grid(True, alpha=0.3)
    ax[0,1].legend()

    xsort = np.linspace(0.0, 1.0, n_boot, endpoint=False) + 1.0/n_boot
    for label in (
        "surrogate, quantile period",
        "exact BE, quantile period",
        "exact Tustin, quantile period",
        "exact BE, guarded period",
    ):
        ax[1,0].plot(np.sort(arrays[label]), xsort, label=label)
    ax[1,0].axvline(pm_min, linestyle="--", linewidth=1.2, label="required margin")
    ax[1,0].set_xlabel("phase margin [degree]")
    ax[1,0].set_ylabel("empirical CDF")
    ax[1,0].set_title("(c) Fixed-controller chance validation")
    ax[1,0].grid(True, alpha=0.3)
    ax[1,0].legend(fontsize=8)

    labels = list(cov)
    vals = [100*cov[k] for k in labels]
    xpos = np.arange(len(labels))
    ax[1,1].bar(xpos, vals)
    ax[1,1].axhline(95.0, linestyle="--", linewidth=1.2, label="target")
    ax[1,1].set_ylim(0, 105)
    ax[1,1].set_xticks(xpos)
    ax[1,1].set_xticklabels([
        "surrogate\nquantile", "BE\nquantile", "Tustin\nquantile",
        "BE\nguarded", "surrogate\nmean"
    ])
    ax[1,1].set_ylabel("empirical satisfaction probability [%]")
    ax[1,1].set_title("(d) Achieved probability")
    ax[1,1].grid(True, axis="y", alpha=0.3)
    ax[1,1].legend()
    fig.savefig(OUT / "fig10_bootstrap_chance.pdf", bbox_inches="tight")
    plt.close(fig)

    wilson = {}
    for label, vals in arrays.items():
        k = int(np.sum(vals >= pm_min))
        wilson[label] = wilson_interval(k, n_boot)
    result = {
        "seed": RNG_SEED,
        "n_bootstrap": n_boot,
        "estimate_K": float(estimate[0]),
        "estimate_tau": float(estimate[1]),
        "estimate_theta": theta_design,
        "theta_boot_mean": theta_mean,
        "theta_q95": q95,
        "theta_boot_sd": float(theta_b.std(ddof=1)),
        "T_cc": float(Tcc),
        "T_mean": float(Tmean),
        "T_guard_BE": float(Tguard),
        "coverage_surrogate_cc": cov["surrogate, quantile period"],
        "coverage_exact_BE_cc": cov["exact BE, quantile period"],
        "coverage_exact_Tustin_cc": cov["exact Tustin, quantile period"],
        "coverage_exact_BE_guard": cov["exact BE, guarded period"],
        "coverage_surrogate_mean": cov["surrogate, mean period"],
        "wilson_low_surrogate_cc": wilson["surrogate, quantile period"][0],
        "wilson_high_surrogate_cc": wilson["surrogate, quantile period"][1],
        "wilson_low_exact_BE_cc": wilson["exact BE, quantile period"][0],
        "wilson_high_exact_BE_cc": wilson["exact BE, quantile period"][1],
        "wilson_low_exact_Tustin_cc": wilson["exact Tustin, quantile period"][0],
        "wilson_high_exact_Tustin_cc": wilson["exact Tustin, quantile period"][1],
        "wilson_low_exact_BE_guard": wilson["exact BE, guarded period"][0],
        "wilson_high_exact_BE_guard": wilson["exact BE, guarded period"][1],
        "wilson_low_surrogate_mean": wilson["surrogate, mean period"][0],
        "wilson_high_surrogate_mean": wilson["surrogate, mean period"][1],
    }
    with (OUT / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_latex_outputs(
    running: dict[str, float],
    groups: dict[str, dict[str, float]],
    soptd_rows: list[SoptdRow],
    soptd: dict[str, float],
    boot: dict[str, float],
) -> None:
    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}\n"
    lines = ["% Generated by validation_ejc.py; do not edit manually.\n"]
    lines += [
        cmd("BenchN", "48"),
        cmd("BenchMaxBEImmediate", f"{groups['immediate; backward Euler']['max_abs_error']:.2f}"),
        cmd("BenchMaxTustinImmediate", f"{groups['immediate; Tustin']['max_abs_error']:.2f}"),
        cmd("BenchMaxBENext", f"{groups['next scan; backward Euler']['max_abs_error']:.2f}"),
        cmd("BenchMaxTustinNext", f"{groups['next scan; Tustin']['max_abs_error']:.2f}"),
        cmd("BootThetaHat", f"{boot['estimate_theta']:.3f}"),
        cmd("BootThetaMean", f"{boot['theta_boot_mean']:.3f}"),
        cmd("BootThetaQ", f"{boot['theta_q95']:.3f}"),
        cmd("BootTcc", f"{boot['T_cc']:.3f}"),
        cmd("BootTmean", f"{boot['T_mean']:.3f}"),
        cmd("BootTguard", f"{boot['T_guard_BE']:.3f}"),
        cmd("BootCovSur", f"{100*boot['coverage_surrogate_cc']:.1f}"),
        cmd("BootCovBE", f"{100*boot['coverage_exact_BE_cc']:.1f}"),
        cmd("BootCovTu", f"{100*boot['coverage_exact_Tustin_cc']:.1f}"),
        cmd("BootCovGuard", f"{100*boot['coverage_exact_BE_guard']:.1f}"),
        cmd("BootCovMean", f"{100*boot['coverage_surrogate_mean']:.1f}"),
        cmd("BootCISurLo", f"{100*boot['wilson_low_surrogate_cc']:.1f}"),
        cmd("BootCISurHi", f"{100*boot['wilson_high_surrogate_cc']:.1f}"),
        cmd("BootCIBELo", f"{100*boot['wilson_low_exact_BE_cc']:.1f}"),
        cmd("BootCIBEHi", f"{100*boot['wilson_high_exact_BE_cc']:.1f}"),
        cmd("BootCITuLo", f"{100*boot['wilson_low_exact_Tustin_cc']:.1f}"),
        cmd("BootCITuHi", f"{100*boot['wilson_high_exact_Tustin_cc']:.1f}"),
        cmd("BootCIGuardLo", f"{100*boot['wilson_low_exact_BE_guard']:.1f}"),
        cmd("BootCIGuardHi", f"{100*boot['wilson_high_exact_BE_guard']:.1f}"),
        cmd("SoptdN", f"{int(soptd['n_cases'])}"),
        cmd("SoptdMedianAbs", f"{soptd['median_abs_error']:.2f}"),
        cmd("SoptdMaxAbs", f"{soptd['max_abs_error']:.2f}"),
        cmd("SoptdWorstShortfall", f"{min(soptd['min_error'],0.0):.2f}"),
        cmd("SoptdMaxExcess", f"{max(soptd['max_error'],0.0):.2f}"),
        cmd("SoptdMaxModePhase", f"{soptd['max_omega_d_T']:.2f}"),
    ]
    (OUT / "validation_macros.tex").write_text("".join(lines), encoding="utf-8")

    # Compact table for the main text.
    table = r"""% Generated by validation_ejc.py
\begin{table*}[htbp]
\centering
\caption{Exact pulse-transfer error at the analytical $10^\circ$ phase-loss
bound over 12 plant/tuning cases per implementation. A negative shortfall
means that the exact phase margin fell below the analytical target.}
\label{tab:benchmark-aggregate}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Implementation & Median $|e_{\rm PM}|$ & Worst shortfall & Max. excess\\
 & [$^\circ$] & [$^\circ$] & [$^\circ$]\\
\midrule
"""
    labels = {
        "immediate; backward Euler": "Immediate, backward Euler",
        "immediate; Tustin": "Immediate, Tustin",
        "next scan; backward Euler": "Next scan, backward Euler",
        "next scan; Tustin": "Next scan, Tustin",
    }
    for key in labels:
        g = groups[key]
        table += (f"{labels[key]} & {g['median_abs_error']:.2f} & "
                  f"{min(g['min_error'],0.0):.2f} & {max(g['max_error'],0.0):.2f}"
                  + r"\\" + "\n")
    table += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    (OUT / "benchmark_summary_table.tex").write_text(table, encoding="utf-8")

    soptd_table = r"""% Generated by validation_ejc.py
\begin{table}[htbp]
\centering
\caption{SOPTD exact sampled-data check at the analytical $5^\circ$ phase-loss
period. The loop uses the cancellable continuous PID construction described
in the text and a Tustin digital PID in the exact check.}
\label{tab:soptd-validation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}ccccc@{}}
\toprule
$\zeta$ & $\omega_n$ & $T$ & $\omega_dT$ & Error\\
 & [rad/s] & [s] &  & [$^\circ$]\\
\midrule
"""
    for r in soptd_rows:
        soptd_table += (
            f"{r.zeta:.2f} & {r.omega_n:.2f} & {r.T_bound:.3f} & "
            f"{r.omega_d_T:.2f} & {r.error_deg:.2f}" + r"\\" + "\n"
        )
    soptd_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "soptd_validation_table.tex").write_text(soptd_table, encoding="utf-8")


def main() -> None:
    running = generate_margin_validation()
    _, groups = generate_benchmark()
    soptd_rows, soptd = generate_soptd_validation()
    boot = generate_bootstrap_chance()
    write_latex_outputs(running, groups, soptd_rows, soptd, boot)
    summary = {
        "running": running,
        "benchmark": groups,
        "soptd_validation": soptd,
        "bootstrap": boot,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
