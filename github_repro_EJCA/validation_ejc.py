#!/usr/bin/env python3
"""Reproducible numerical checks and figures for the EJC manuscript.

Running this script regenerates *every* numerical figure, CSV file, JSON
summary and generated LaTeX input used by the manuscript:

  fig1_zoh_half_delay            zero-order hold as a half-sample delay
  fig2_exact_margin_validation   exact pulse-transfer check of the surrogate
  fig3_iae_law                   hybrid load-disturbance IAE check
  fig5_hidden_oscillations       aliasing of a lightly damped mode
  fig6_sampling_zero             sampling-zero migration towards z = -1
  fig7_noise_montecarlo          derivative-noise amplification
  fig8_wordlength                coefficient-resolution lower bound
  fig9_actuator_variation        command activity of a cancelled zero
  fig10_bootstrap_chance         split-bootstrap fixed-controller study
  fig12_phase_budget_benchmark   multi-case FOPTD phase-budget benchmark
  fig_arch_delay_budget          where each delay enters the loop
  fig_window_map                 the two-sided window and what closes it
  fig_design_chart               dimensionless design chart + robustness bands
  fig_applications               process-industry and power-electronics cases

The core numerics live in ``ejc_window.py``; ``test_validation_ejc.py``
locks the headline numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import least_squares

from ejc_window import (
    FOPTD, SOPTD, PIParams, Timing, Budgets, IMMEDIATE, NEXT_SCAN,
    simc_pi, simc_pid_cancelling,
    zoh_foptd_frequency, digital_pi_frequency,
    foptd_pi_margin, soptd_pid_margin, soptd_pid_continuous_margin,
    foptd_pi_stability_limit,
    sampling_zero_exact, sampling_zero_first_order,
    continuous_pm_deg, surrogate_pm_deg,
    T_cost_update, T_cost_noise,
    build_window, Window, REMEDY,
)

OUT = Path(__file__).resolve().parent

# Deterministic, study-specific seeds derived from one master seed.
RNG_SEED = 10
SEEDS = {
    "bootstrap": RNG_SEED,
    "noise": RNG_SEED + 101,
    "robustness": RNG_SEED + 202,
}

# --------------------------------------------------------------------------
# Plot style: one consistent, colour-blind-safe system for every figure.
# --------------------------------------------------------------------------
# Okabe-Ito palette; distinguishable in colour, in greyscale and to the most
# common forms of colour-vision deficiency.
C = {
    "ink": "#1a1a1a",
    "grey": "#7f7f7f",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}
FEASIBLE_FILL = "#009E73"
INFEASIBLE_FILL = "#D55E00"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def _finish(fig, name: str) -> None:
    """Write a vector PDF and close the figure."""
    for ax in fig.get_axes():
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def _band(ax, x, lo, hi, *, label=None, color=FEASIBLE_FILL, alpha=0.18):
    return ax.fill_between(x, lo, hi, where=np.asarray(hi) >= np.asarray(lo),
                           color=color, alpha=alpha, linewidth=0, label=label)


# ==========================================================================
# Running example used throughout the paper
# ==========================================================================
RUN_MODEL = FOPTD(K=1.0, tau=10.0, theta=2.0, label="running example")
RUN_LAM = 2.0
RUN_RHO_U = 0.005          # gamma*kappa/(2 alpha theta0^3) for the ideal case
LAG_MODEL = FOPTD(K=1.0, tau=20.0, theta=1.0, label="lag-dominant example")


# ==========================================================================
# 1. Zero-order hold as a half-sample delay
# ==========================================================================
def generate_zoh_figure() -> dict[str, float]:
    wT = np.linspace(1e-4, math.pi, 800)
    mag_db = 20.0 * np.log10(np.abs(np.sinc(wT / (2.0 * math.pi))))
    phase_deg = -np.degrees(wT / 2.0)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    ax[0].plot(wT, mag_db, color=C["blue"])
    ax[0].axhline(-0.1, color=C["grey"], ls=":", lw=1.0)
    ax[0].axvline(0.5, color=C["orange"], ls="--", lw=1.2)
    ax[0].annotate(r"$\omega T=0.5$: $<0.1$ dB",
                   xy=(0.5, -0.1), xycoords="data",
                   xytext=(0.22, 0.52), textcoords="axes fraction",
                   fontsize=7.6, color=C["orange"], ha="left",
                   arrowprops=dict(arrowstyle="->", color=C["orange"], lw=0.8))
    ax[0].annotate(f"{mag_db[-1]:.1f} dB at Nyquist",
                   xy=(math.pi, mag_db[-1]), xycoords="data",
                   xytext=(0.42, 0.14), textcoords="axes fraction",
                   fontsize=7.6, color=C["ink"], ha="left",
                   arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.8))
    ax[0].set_xlabel(r"normalised frequency, $\omega T$ [rad]")
    ax[0].set_ylabel(r"$|H(j\omega)|/T$ [dB]")
    ax[0].set_title("(a) Hold magnitude droop", fontsize=9)

    ax[1].plot(wT, phase_deg, color=C["blue"], label=r"exact hold phase")
    ax[1].plot(wT, phase_deg, color=C["vermillion"], ls="--", lw=1.0,
               label=r"delay model $-\omega T/2$")
    ax[1].set_xlabel(r"normalised frequency, $\omega T$ [rad]")
    ax[1].set_ylabel("phase [degree]")
    ax[1].set_title("(b) Phase is exactly a half sample", fontsize=9)
    ax[1].legend(loc="lower left")
    _finish(fig, "fig01_zoh_half_delay")
    return {
        "droop_db_at_wT_0p5": float(20.0 * math.log10(np.sinc(0.5 / (2 * math.pi)))),
        "droop_db_at_nyquist": float(mag_db[-1]),
    }


# ==========================================================================
# 2. Exact pulse-transfer validation of the delay surrogate
# ==========================================================================
def generate_margin_validation() -> dict[str, float]:
    model, lam = RUN_MODEL, RUN_LAM
    ctrl = simc_pi(model, lam)
    wc = 1.0 / (lam + model.theta)
    pm0 = continuous_pm_deg(model.theta, model.theta, lam)

    T = np.linspace(0.08, 1.55, 80)
    sur = {
        "immediate": np.array([
            surrogate_pm_deg(model.theta, model.theta, lam, t, IMMEDIATE.kappa)
            for t in T]),
        "next scan": np.array([
            surrogate_pm_deg(model.theta, model.theta, lam, t, NEXT_SCAN.kappa)
            for t in T]),
    }
    curves: dict[tuple[str, str], np.ndarray] = {}
    max_crossings = 0
    for timing in (IMMEDIATE, NEXT_SCAN):
        for method in ("backward_euler", "tustin"):
            vals = []
            for t in T:
                r = foptd_pi_margin(plant=model, ctrl=ctrl, T=float(t),
                                    method=method, timing=timing, n_grid=6000)
                max_crossings = max(max_crossings, r.n_crossings)
                vals.append(r.pm_deg)
            curves[(timing.label, method)] = np.array(vals)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax[0].plot(T, sur["immediate"], color=C["ink"], ls="--", lw=1.8,
               label="delay surrogate")
    ax[0].plot(T, curves[("immediate", "backward_euler")], color=C["blue"],
               label="exact pulse model, backward Euler")
    ax[0].plot(T, curves[("immediate", "tustin")], color=C["vermillion"],
               label="exact pulse model, Tustin")
    ax[0].axhline(pm0 - 10.0, color=C["green"], ls=":", lw=1.3,
                  label=r"$10^\circ$ loss target")
    T10 = math.radians(10.0) / (wc * IMMEDIATE.kappa)
    ax[0].axvline(T10, color=C["green"], ls=":", lw=1.0)
    ax[0].annotate(rf"$T_{{\max}}^{{\rm PM}}={T10:.2f}$ s",
                   xy=(T10, pm0 - 10.0), xycoords="data",
                   xytext=(0.62, 0.86), textcoords="axes fraction",
                   fontsize=8, color=C["green"], ha="left",
                   arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8))
    ax[0].set_xlabel("sampling period, $T$ [s]")
    ax[0].set_ylabel("phase margin [degree]")
    ax[0].set_title("(a) Immediate-update implementation", fontsize=9)
    ax[0].legend(loc="lower left", fontsize=6.4)

    x = wc * T
    err = {
        ("backward Euler", "immediate"):
            curves[("immediate", "backward_euler")] - sur["immediate"],
        ("Tustin", "immediate"):
            curves[("immediate", "tustin")] - sur["immediate"],
        ("backward Euler", "next scan"):
            curves[("next scan", "backward_euler")] - sur["next scan"],
        ("Tustin", "next scan"):
            curves[("next scan", "tustin")] - sur["next scan"],
    }
    styles = {
        ("backward Euler", "immediate"): (C["blue"], "-"),
        ("Tustin", "immediate"): (C["vermillion"], "-"),
        ("backward Euler", "next scan"): (C["blue"], "--"),
        ("Tustin", "next scan"): (C["vermillion"], "--"),
    }
    ax[1].axhspan(-1.0, 1.0, color=C["green"], alpha=0.12, lw=0,
                  label=r"$\pm1^\circ$ acceptance band")
    for key, e in err.items():
        col, ls = styles[key]
        ax[1].plot(x, e, color=col, ls=ls, label=f"{key[0]}, {key[1]}")
    ax[1].axhline(0.0, color=C["grey"], ls=":", lw=1.0)
    ax[1].set_xlabel(r"normalised period, $\omega_c T$")
    ax[1].set_ylabel("exact $-$ surrogate margin [degree]")
    ax[1].set_title("(b) Approximation error", fontsize=9)
    ax[1].legend(loc="lower left", fontsize=6.4)
    _finish(fig, "fig03_exact_margin_validation")

    # Exact discrete stability limits for the same frozen controller.
    T_stab = {
        m: foptd_pi_stability_limit(model=model, lam=lam, method=m,
                                    timing=IMMEDIATE)
        for m in ("backward_euler", "tustin")
    }
    mask = x <= 2.0 * math.radians(15.0)
    return {
        "running_max_crossings": int(max_crossings),
        "running_immediate_be_max_abs_error":
            float(np.max(np.abs(err[("backward Euler", "immediate")][mask]))),
        "running_immediate_tu_max_abs_error":
            float(np.max(np.abs(err[("Tustin", "immediate")][mask]))),
        "running_next_be_max_abs_error":
            float(np.max(np.abs(err[("backward Euler", "next scan")][x <= 0.18]))),
        "running_next_tu_max_abs_error":
            float(np.max(np.abs(err[("Tustin", "next scan")][x <= 0.18]))),
        "T_stability_backward_euler": float(T_stab["backward_euler"]),
        "T_stability_tustin": float(T_stab["tustin"]),
        "T_max_pm_immediate": float(T10),
        "T_max_pm_next_scan": float(math.radians(10.0) / (wc * NEXT_SCAN.kappa)),
        "T_zero_margin_immediate":
            float((0.5 * math.pi * (lam + model.theta) - model.theta)
                  / IMMEDIATE.kappa),
    }


# ==========================================================================
# 3. Load-disturbance IAE law (hybrid simulation)
# ==========================================================================
def _simulate_load_disturbance(
    *, plant: FOPTD, ctrl: PIParams, T: float, timing: Timing,
    horizon: float, n_sub: int = 40,
) -> tuple[float, float, bool]:
    """Hybrid simulation of a unit input load step; returns (IAE, IE, one_signed).

    The plant is integrated on a sub-grid ``h = T/n_sub`` with the exact
    fractional-delay update, so the intersample error is captured.  The
    controller is a backward-Euler PI updated once per sampling period.
    """
    h = T / n_sub
    theta_tot = plant.theta + timing.T_c0
    m = int(round(theta_tot / h))
    a = math.exp(-h / plant.tau)
    b = plant.K * (1.0 - a)

    n_steps = int(round(horizon / h))
    buf = [0.0] * (m + 1)          # delayed plant input history
    x = 0.0
    integ = 0.0
    u = 0.0
    e_hist = np.empty(n_steps + 1)
    y = 0.0
    e_hist[0] = -y
    hold_delay = timing.extra_sample_delays
    pending = [0.0] * (hold_delay + 1)

    for k in range(n_steps):
        if k % n_sub == 0:                       # controller update instant
            e = -y                               # setpoint 0, load rejection
            integ += (T / ctrl.tau_i) * e
            u_new = ctrl.Kc * (e + integ)
            pending.append(u_new)
            u = pending.pop(0)
        d = 1.0                                  # unit load step at plant input
        buf.append(u + d)
        x = a * x + b * buf.pop(0)
        y = x
        e_hist[k + 1] = -y

    t = np.arange(n_steps + 1) * h
    IE = float(np.trapezoid(e_hist, t))
    IAE = float(np.trapezoid(np.abs(e_hist), t))
    one_signed = bool(np.all(e_hist <= 1e-12) or np.all(e_hist >= -1e-12))
    return IAE, IE, one_signed


def generate_iae_law() -> dict[str, float]:
    """Check the quadratic IAE law for a lag-dominant FOPTD loop."""
    plant = LAG_MODEL
    timing = IMMEDIATE
    Ts = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5])
    sim, ana = [], []
    for T in Ts:
        theta_eff = timing.effective_delay(plant.theta, float(T))
        lam = theta_eff                       # robust SIMC choice, retuned
        ctrl = simc_pi(FOPTD(plant.K, plant.tau, theta_eff), lam)
        assert plant.tau > 4.0 * (lam + plant.theta), "not lag-dominant"
        iae, ie, one_signed = _simulate_load_disturbance(
            plant=plant, ctrl=ctrl, T=float(T), timing=timing,
            horizon=40.0 * plant.tau,
        )
        if not one_signed:
            raise RuntimeError(
                f"error changed sign at T={T}: IAE = |IE| no longer holds"
            )
        sim.append(iae)
        ana.append(16.0 * plant.K * theta_eff**2 / plant.tau)
    sim = np.array(sim)
    ana = np.array(ana)
    ratio = sim / ana

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    ax[0].plot(Ts, ana, color=C["ink"], ls="--",
               label=r"analytical $16K\theta_{\rm eff}^2/\tau$")
    ax[0].plot(Ts, sim, "o", color=C["blue"], ms=5, mfc="none",
               label="hybrid simulation")
    ax[0].set_xlabel("sampling period, $T$ [s]")
    ax[0].set_ylabel("load-disturbance IAE")
    ax[0].set_title("(a) Quadratic degradation law", fontsize=9)
    ax[0].legend(loc="upper left")

    ax[1].plot(Ts, 100.0 * (ratio - 1.0), "o-", color=C["vermillion"], ms=4)
    ax[1].axhline(0.0, color=C["grey"], ls=":", lw=1.0)
    ax[1].set_ylim(-0.2, 0.2)
    ax[1].set_xlabel("sampling period, $T$ [s]")
    ax[1].set_ylabel("simulated / analytical $-1$ [%]")
    ax[1].set_title("(b) Deviation on a fixed $\\pm0.2\\%$ scale", fontsize=9)
    _finish(fig, "fig04_iae_law")

    return {
        "iae_ratio_min": float(ratio.min()),
        "iae_ratio_max": float(ratio.max()),
        "iae_n_periods": int(Ts.size),
    }


# ==========================================================================
# 5. Hidden oscillations
# ==========================================================================
def generate_hidden_oscillations() -> dict[str, float]:
    zeta, wn = 0.15, 1.0
    wd = wn * math.sqrt(1.0 - zeta**2)
    t = np.linspace(0.0, 60.0, 4000)
    phi = math.acos(zeta)
    y = 1.0 - np.exp(-zeta * wn * t) / math.sqrt(1 - zeta**2) * np.sin(wd * t + phi)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True,
                           sharey=True)
    for i, (T, title) in enumerate((
        (2.0 * math.pi / wd, r"(a) $\omega_d T = 2\pi$: ringing hidden"),
        (2.0 * math.pi / (10.0 * wd), "(b) ten samples per damped period"),
    )):
        tk = np.arange(0.0, 60.0, T)
        yk = 1.0 - np.exp(-zeta * wn * tk) / math.sqrt(1 - zeta**2) * np.sin(
            wd * tk + phi)
        ax[i].plot(t, y, color=C["grey"], lw=1.0, label="continuous response")
        ax[i].plot(tk, yk, "o-", color=C["blue"], ms=3.8, lw=1.2,
                   label="sampled sequence")
        ax[i].set_xlabel("time [s]")
        ax[i].set_title(title, fontsize=9)
    ax[0].set_ylabel("output")
    ax[0].legend(loc="lower right", fontsize=7.2)
    _finish(fig, "fig05_hidden_oscillations")
    return {"hidden_zeta": zeta, "hidden_omega_n": wn}


# ==========================================================================
# Sampling zeros and the actuator activity they cause
# ==========================================================================
def generate_sampling_zero() -> dict[str, float]:
    """Zero migration towards z = -1 and the command activity it produces.

    Panels (b) and (c) quantify why cancelling the zero is the mechanism that
    makes *faster* sampling harmful: both the command travel and the reversal
    count grow as 1/T while the leading-order ringing time does not shrink.
    """
    wn = 1.0
    T = np.linspace(0.01, 1.2, 400)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    a = ax[0]
    for zeta, col in ((0.2, C["blue"]), (0.4, C["vermillion"]),
                      (0.7, C["green"])):
        exact = np.array([sampling_zero_exact(zeta, wn, float(t)) for t in T])
        first = np.array([sampling_zero_first_order(zeta, wn, float(t))
                          for t in T])
        a.plot(T, exact, color=col, label=rf"$\zeta={zeta}$")
        a.plot(T, first, color=col, ls=":", lw=1.1)
    a.axhline(-1.0, color=C["ink"], ls="--", lw=1.0)
    a.annotate(r"$z_0\to-1$ as $T\to0$", xy=(0.02, -1.0), xycoords="data",
               xytext=(0.40, 0.86), textcoords="axes fraction", fontsize=6.8,
               color=C["ink"], ha="left",
               arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.8))
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("sampling zero, $z_0$")
    a.set_title("(a) Zero migration", fontsize=9)
    a.legend(loc="lower right", fontsize=6.6, title="dotted: expansion",
             title_fontsize=6.2)

    zeta = 0.4
    horizon = 20.0
    Ts = np.geomspace(0.02, 0.5, 25)
    tv, rev = [], []
    for t in Ts:
        z0 = sampling_zero_exact(zeta, wn, float(t))
        n = int(horizon / t)
        u = z0 ** np.arange(n + 1)
        tv.append(float(np.sum(np.abs(np.diff(u)))))
        rev.append(float(np.sum(np.diff(np.sign(u)) != 0)))
    tv, rev = np.array(tv), np.array(rev)

    a = ax[1]
    a.loglog(Ts, tv, "o", color=C["blue"], ms=4.5, mfc="none",
             label="cancelled zero")
    a.loglog(Ts, tv[0] * Ts[0] / Ts, color=C["ink"], ls="--", lw=1.2,
             label=r"$\propto 1/T$")
    a.axhline(2.0, color=C["green"], ls=":", lw=1.3, label="bounded variation")
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("total command variation")
    a.set_title(rf"(b) Command travel, $\zeta={zeta}$", fontsize=9)
    a.legend(loc="lower left", fontsize=6.6)
    a.grid(True, which="both", alpha=0.2)

    a = ax[2]
    a.loglog(Ts, rev, "s", color=C["vermillion"], ms=4.5, mfc="none",
             label="reversals")
    a.loglog(Ts, rev[0] * Ts[0] / Ts, color=C["ink"], ls="--", lw=1.2,
             label=r"$\propto 1/T$")
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("command reversals")
    a.set_title("(c) Reversal count", fontsize=9)
    a.legend(loc="lower left", fontsize=6.6)
    a.grid(True, which="both", alpha=0.2)
    # Log minor-tick labels collide on these narrow panels; label a few
    # decade-friendly values explicitly instead.
    for a_ in (ax[1], ax[2]):
        a_.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
        a_.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _finish(fig, "fig06_sampling_zero")

    z_ex = sampling_zero_exact(0.4, 1.0, 0.35)
    z_fo = sampling_zero_first_order(0.4, 1.0, 0.35)
    return {
        "zero_exact_ref": float(z_ex),
        "zero_first_order_ref": float(z_fo),
        "zero_rel_error_pct": float(100.0 * abs(z_fo - z_ex) / abs(z_ex)),
        "actuator_tv_slope": float(np.polyfit(np.log(Ts), np.log(tv), 1)[0]),
        "actuator_rev_slope": float(np.polyfit(np.log(Ts), np.log(rev), 1)[0]),
    }


# ==========================================================================
# The fast-sampling lower bounds, in one figure
# ==========================================================================
def _first_order_iir(x: np.ndarray, a: float) -> np.ndarray:
    """y[k] = a y[k-1] + x[k], vectorised via scipy.signal.lfilter."""
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -a], x)


def generate_lower_bounds() -> dict[str, float]:
    """Coefficient resolution and derivative noise: the two lower endpoints.

    Panels (a)-(b) are the representation bound on a near-unity stored pole;
    panel (c) is the derivative-noise Monte Carlo.  Both mechanisms make the
    *small*-T side of the window inadmissible, which is what a one-sided
    bandwidth rule cannot express.
    """
    tau, eps_c = 10.0, 0.02
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    # ---- (a) exact rounded coefficient error vs the half-step envelope ----
    T = np.geomspace(1e-3, 1.0, 400)
    a = ax[0]
    for B, col in ((12, C["blue"]), (16, C["vermillion"])):
        pole = np.exp(-T / tau)
        step = 2.0 ** (-B)
        p_q = np.round(pole / step) * step
        with np.errstate(divide="ignore"):
            tau_hat = -T / np.log(np.clip(p_q, 1e-16, 1 - 1e-16))
        rel = np.abs(tau_hat - tau) / tau
        env = (tau / T) * 2.0 ** (-(B + 1))
        a.loglog(T, np.maximum(rel, 1e-9), color=col, lw=1.0, alpha=0.85,
                 label=f"rounded, $B={B}$")
        a.loglog(T, env, color=col, ls="--", lw=1.3,
                 label=f"envelope, $B={B}$")
    a.axhline(eps_c, color=C["green"], ls=":", lw=1.3)
    a.text(0.62, eps_c * 1.6, r"$\varepsilon_c=2\%$", fontsize=6.8,
           color=C["green"], ha="right")
    a.set_ylim(1e-6, 1e3)
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel(r"lag error, $|\hat\tau-\tau|/\tau$")
    a.set_title("(a) Near-unity pole storage", fontsize=9)
    a.legend(loc="lower left", fontsize=6.0, ncol=2, columnspacing=0.7,
             handlelength=1.3)
    a.grid(True, which="both", alpha=0.2)

    # ---- (b) resulting lower endpoint vs word length ---------------------
    bits = np.arange(8, 25)
    Tmin = tau * 2.0 ** (-(bits + 1.0)) / eps_c
    a = ax[1]
    a.semilogy(bits, Tmin, "o-", color=C["blue"], ms=3.6)
    for b_, dy in ((12, 1), (16, -1)):
        v = tau * 2.0 ** (-(b_ + 1)) / eps_c
        a.annotate(f"$B={b_}$: {1e3*v:.0f} ms" if v < 1 else f"$B={b_}$: {v:.2f} s",
                   xy=(b_, v), xycoords="data",
                   xytext=(0.35, 0.80 if dy > 0 else 0.16),
                   textcoords="axes fraction", fontsize=6.8,
                   arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.8))
    a.set_xlabel("fractional bits, $B$")
    a.set_ylabel(r"$T_{\min}^{\rm pole}$ [s]")
    a.set_title("(b) Lower endpoint", fontsize=9)

    # ---- (c) derivative-noise Monte Carlo --------------------------------
    rng = np.random.default_rng(SEEDS["noise"])
    Kc, tau_d, sigma = 2.0, 1.0, 0.01
    N_filt = 10.0
    Ts = np.geomspace(0.01, 1.0, 12)
    n_rep, n_samp = 40, 4000
    var_raw, var_filt = [], []
    for t in Ts:
        vr, vf = [], []
        # First-order derivative filter with time constant tau_D/N: the pole
        # is exp(-N T/tau_D) and the high-frequency gain saturates at Kc*N
        # instead of growing as 1/T.
        pole = math.exp(-N_filt * t / tau_d)
        g = Kc * N_filt
        for _ in range(n_rep):
            n = rng.normal(0.0, sigma, n_samp)
            dn = np.diff(n, prepend=n[0])
            vr.append(float(np.var(Kc * tau_d * dn[1:] / t)))
            d_f = _first_order_iir(g * (1.0 - pole) * dn, pole)
            vf.append(float(np.var(d_f[1:])))
        var_raw.append(np.mean(vr))
        var_filt.append(np.mean(vf))
    var_raw, var_filt = np.array(var_raw), np.array(var_filt)
    slope = float(np.polyfit(np.log(Ts), np.log(var_raw), 1)[0])
    theory = 2.0 * (Kc * tau_d * sigma / Ts) ** 2

    a = ax[2]
    a.loglog(Ts, var_raw, "o", color=C["blue"], ms=4.5, mfc="none",
             label=f"unfiltered (slope {slope:.2f})")
    a.loglog(Ts, theory, color=C["ink"], ls="--", lw=1.3,
             label=r"$2(K_c\tau_D\sigma_n/T)^2$")
    a.loglog(Ts, var_filt, "s", color=C["vermillion"], ms=4.5, mfc="none",
             label=rf"filtered, $N={N_filt:g}$")
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("command variance")
    a.set_title("(c) Derivative noise", fontsize=9)
    a.legend(loc="lower left", fontsize=6.2)
    a.grid(True, which="both", alpha=0.2)
    _finish(fig, "fig07_lower_bounds")

    return {
        "word_Tmin_B12": float(tau * 2.0 ** (-13) / eps_c),
        "word_Tmin_B16": float(tau * 2.0 ** (-17) / eps_c),
        "noise_slope": slope,
        "noise_replications": n_rep,
    }


# ==========================================================================
# 12. Multi-case FOPTD phase-budget benchmark
# ==========================================================================
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
    n_crossings: int


def generate_benchmark() -> tuple[list[BenchmarkRow], dict[str, dict[str, float]],
                                  dict[str, float]]:
    K, tau = 1.0, 10.0
    dphi_deg = 10.0
    dphi = math.radians(dphi_deg)
    rows: list[BenchmarkRow] = []
    for ratio in (0.15, 0.20, 0.30, 0.50, 0.75, 1.00):
        theta = ratio * tau
        for lr in (1.0, 2.0):
            lam = lr * theta
            model = FOPTD(K=K, tau=tau, theta=theta)
            ctrl = simc_pi(model, lam)
            # The closed-form margin uses tau_i = tau; assert the domain.
            assert tau <= 4.0 * (lam + theta) + 1e-12
            assert abs(ctrl.tau_i - tau) < 1e-12
            pm0 = continuous_pm_deg(theta, theta, lam)
            for timing in (IMMEDIATE, NEXT_SCAN):
                T_bound = dphi * (lam + theta) / timing.kappa
                target = pm0 - dphi_deg
                for label, method in (("backward Euler", "backward_euler"),
                                      ("Tustin", "tustin")):
                    r = foptd_pi_margin(plant=model, ctrl=ctrl, T=T_bound,
                                        method=method, timing=timing,
                                        n_grid=9000)
                    rows.append(BenchmarkRow(
                        theta_tau=ratio, lambda_theta=lr,
                        architecture=timing.label, method=label,
                        T_bound=T_bound, target_pm=target, exact_pm=r.pm_deg,
                        error_deg=r.pm_deg - target, exact_loss=pm0 - r.pm_deg,
                        n_crossings=r.n_crossings,
                    ))

    fields = list(BenchmarkRow.__annotations__)
    with (OUT / "benchmark_phase_budget.csv").open("w", newline="",
                                                   encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k) for k in fields])

    groups: dict[str, dict[str, float]] = {}
    for arch in ("immediate", "next scan"):
        for method in ("backward Euler", "Tustin"):
            v = np.array([r.error_deg for r in rows
                          if r.architecture == arch and r.method == method])
            groups[f"{arch}; {method}"] = {
                "n": int(v.size),
                "mean_error": float(v.mean()),
                "median_abs_error": float(np.median(np.abs(v))),
                "max_abs_error": float(np.max(np.abs(v))),
                "min_error": float(v.min()),
                "max_error": float(v.max()),
            }
    overall = {
        "n_cases": len(rows),
        "worst_shortfall": float(min(0.0, min(r.error_deg for r in rows))),
        "max_crossings": int(max(r.n_crossings for r in rows)),
    }

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    styles = {
        ("backward Euler", "immediate"): (C["blue"], "-", "o"),
        ("Tustin", "immediate"): (C["vermillion"], "-", "s"),
        ("backward Euler", "next scan"): (C["blue"], "--", "^"),
        ("Tustin", "next scan"): (C["vermillion"], "--", "v"),
    }
    ax[0].axhspan(-1.0, 0.0, color=INFEASIBLE_FILL, alpha=0.10, lw=0)
    for (method, arch), (col, ls, mk) in styles.items():
        for lr, alpha in ((1.0, 1.0), (2.0, 0.45)):
            rs = [r for r in rows if r.architecture == arch
                  and r.method == method and r.lambda_theta == lr]
            # Only the lambda/theta = 1 series carries a legend entry; the
            # faded series is explained by a single proxy handle below.
            ax[0].plot([r.theta_tau for r in rs], [r.error_deg for r in rs],
                       color=col, ls=ls, marker=mk, ms=4.2, alpha=alpha,
                       label=(f"{method}, {arch}" if lr == 1.0 else None))
    ax[0].axhline(0.0, color=C["ink"], ls=":", lw=1.0)
    ax[0].text(0.02, 0.06, "shortfall", transform=ax[0].transAxes, fontsize=8,
               color=INFEASIBLE_FILL)
    ax[0].set_xlabel(r"dead-time ratio, $\theta/\tau$")
    ax[0].set_ylabel("exact $-$ target margin [degree]")
    ax[0].set_title(r"(a) Error at the $10^\circ$ bound", fontsize=9)
    handles, labels = ax[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D(
        [], [], color=C["grey"], alpha=0.45, lw=1.6,
        label=r"faded: $\lambda/\theta=2$"))
    labels.append(r"faded: $\lambda/\theta=2$")
    ax[0].legend(handles, labels, fontsize=5.8, ncol=1, loc="upper left")

    labels = list(groups)
    xpos = np.arange(len(labels))
    med = [groups[k]["median_abs_error"] for k in labels]
    mx = [groups[k]["max_abs_error"] for k in labels]
    ax[1].bar(xpos - 0.19, med, width=0.36, color=C["blue"],
              label="median absolute error")
    ax[1].bar(xpos + 0.19, mx, width=0.36, color=C["sky"],
              label="maximum absolute error")
    for xi, (m1, m2) in enumerate(zip(med, mx)):
        ax[1].text(xi - 0.19, m1, f"{m1:.2f}", ha="center", va="bottom",
                   fontsize=6.4)
        ax[1].text(xi + 0.19, m2, f"{m2:.2f}", ha="center", va="bottom",
                   fontsize=6.4)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(
        [s.replace("immediate", "imm.").replace("backward Euler", "BE")
          .replace("; ", "\n") for s in labels], fontsize=7.0)
    ax[1].set_ylabel("margin error [degree]")
    ax[1].set_title("(b) Aggregate, 12 cases each", fontsize=9)
    ax[1].legend(loc="upper right", fontsize=6.6)
    _finish(fig, "fig10_phase_budget_benchmark")

    with (OUT / "benchmark_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump({"groups": groups, "overall": overall}, f, indent=2)
    return rows, groups, overall


# ==========================================================================
# SOPTD sampled-data check with timing and modal mismatch
# ==========================================================================
@dataclass
class SoptdRow:
    zeta: float
    omega_n_theta: float
    omega_n_lam: float
    architecture: str
    mismatch_pct: float
    T_bound: float
    omega_d_T: float
    active_upper: str
    target_pm: float
    exact_pm: float
    error_deg: float
    n_crossings: int


def generate_soptd_validation() -> tuple[list[SoptdRow], dict[str, float],
                                         dict[str, dict[str, float]]]:
    """Dimensionless SOPTD benchmark with timing and modal mismatch.

    The tested period is the *window* value ``min(T_max^PM, T_max^mode)``,
    not the phase-margin bound alone, so every case stays inside the stated
    mode-resolution region.
    """
    K, omega_n = 1.0, 1.0
    dphi_deg = 5.0
    dphi = math.radians(dphi_deg)
    rows: list[SoptdRow] = []
    for zeta in (0.25, 0.40, 0.70):
        for wn_theta in (0.25, 0.50, 1.00):
            for wn_lam in (1.0, 2.0, 4.0):
                theta = wn_theta / omega_n
                lam = wn_lam / omega_n
                plant = SOPTD(K=K, zeta=zeta, omega_n=omega_n, theta=theta)
                for timing in (IMMEDIATE, NEXT_SCAN):
                    win = build_window(
                        model=plant, lam=lam, timing=timing,
                        budgets=Budgets(dphi_deg=dphi_deg, eps_iae=None,
                                        mode_fraction=0.25, bits=None,
                                        eps_coef=None),
                    )
                    T_bound = win.T_max
                    for mm in (-0.10, 0.0, 0.10):
                        ctrl_model = SOPTD(K=K, zeta=zeta * (1 + mm),
                                           omega_n=omega_n * (1 + mm),
                                           theta=theta)
                        ctrl = simc_pid_cancelling(ctrl_model, lam)
                        # Reference = continuous margin of the *same*
                        # (possibly mismatched) loop, less the delay-surrogate
                        # loss at its own crossover.  This isolates the
                        # sampled-data approximation error from the modal
                        # mismatch, which is a modelling error, not a
                        # sampling error.
                        cont = soptd_pid_continuous_margin(
                            plant=plant, ctrl=ctrl, n_grid=8000)
                        loss = math.degrees(
                            cont.omega_c * (timing.kappa * T_bound + timing.T_c0))
                        target = cont.pm_deg - loss
                        r = soptd_pid_margin(plant=plant, ctrl=ctrl, T=T_bound,
                                             timing=timing, n_grid=9000)
                        rows.append(SoptdRow(
                            zeta=zeta, omega_n_theta=wn_theta,
                            omega_n_lam=wn_lam, architecture=timing.label,
                            mismatch_pct=100.0 * mm, T_bound=T_bound,
                            omega_d_T=plant.omega_d * T_bound,
                            active_upper=win.active_upper,
                            target_pm=target, exact_pm=r.pm_deg,
                            error_deg=r.pm_deg - target,
                            n_crossings=r.n_crossings,
                        ))

    fields = list(SoptdRow.__annotations__)
    with (OUT / "soptd_validation.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k) for k in fields])

    v = np.array([r.error_deg for r in rows])
    summary = {
        "n_cases": len(rows),
        "n_distinct": len({(r.zeta, r.omega_n_theta, r.omega_n_lam)
                           for r in rows}),
        "median_abs_error": float(np.median(np.abs(v))),
        "max_abs_error": float(np.max(np.abs(v))),
        "min_error": float(v.min()),
        "max_error": float(v.max()),
        "worst_shortfall": float(min(0.0, v.min())),
        "max_omega_d_T": float(max(r.omega_d_T for r in rows)),
        "max_crossings": int(max(r.n_crossings for r in rows)),
        "n_mode_limited": int(sum(1 for r in rows
                                  if r.active_upper == "mode resolution")),
    }
    groups: dict[str, dict[str, float]] = {}
    for arch in ("immediate", "next scan"):
        for mm in (-10.0, 0.0, 10.0):
            sel = np.array([r.error_deg for r in rows
                            if r.architecture == arch and r.mismatch_pct == mm])
            groups[f"{arch}; {mm:+.0f}%"] = {
                "n": int(sel.size),
                "median_abs_error": float(np.median(np.abs(sel))),
                "worst_shortfall": float(min(0.0, sel.min())),
                "max_excess": float(max(0.0, sel.max())),
            }
    with (OUT / "soptd_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "groups": groups}, f, indent=2)
    return rows, summary, groups


# ==========================================================================
# 10. Split-bootstrap fixed-controller uncertainty study
# ==========================================================================
def step_response(t: np.ndarray, K: float, tau: float, theta: float) -> np.ndarray:
    x = np.maximum(t - theta, 0.0)
    return np.where(t > theta, K * (1.0 - np.exp(-x / tau)), 0.0)


def fit_foptd_step(t: np.ndarray, y: np.ndarray, x0: np.ndarray) -> np.ndarray:
    fit = least_squares(
        lambda p: step_response(t, p[0], p[1], p[2]) - y,
        x0=x0, bounds=([0.2, 1.0, 0.01], [2.0, 50.0, 8.0]), max_nfev=2000,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return centre - half, centre + half


def _joint_coverage(
    draws: np.ndarray, *, ctrl: PIParams, T: float, method: str,
    timing: Timing, pm_min: float, n_grid: int,
) -> tuple[float, np.ndarray]:
    """Empirical satisfaction probability of the frozen controller."""
    pm = np.empty(draws.shape[0])
    for i, (Kb, taub, thb) in enumerate(draws):
        pm[i] = foptd_pi_margin(
            plant=FOPTD(K=float(Kb), tau=float(taub), theta=float(thb)),
            ctrl=ctrl, T=T, method=method, timing=timing, n_grid=n_grid,
        ).pm_deg
    return float(np.mean(pm >= pm_min)), pm


def _calibrate_joint_period(
    draws: np.ndarray, *, ctrl: PIParams, method: str, timing: Timing,
    pm_min: float, target: float, T_hi: float, n_grid: int = 2500,
    conservative: bool = False,
) -> tuple[float, bool, list[tuple[float, float]]]:
    """Largest calibration period meeting the satisfaction requirement.

    With ``conservative=False`` the acceptance test is the plain empirical
    frequency ``p_hat >= target``.  Because the same finite calibration
    sample both selects and scores the period, that test is optimistic: the
    selected period is the one where ``p_hat`` first dips, so held-out
    coverage sits slightly below nominal.  With ``conservative=True`` the
    test is instead the *lower* Wilson bound of the calibration estimate,
    which prices in calibration-sample uncertainty and is what we recommend
    in practice.

    A coarse scan first checks that the acceptance probability is
    non-increasing in ``T`` -- what makes the bisection meaningful.  The flag
    is returned so the caller can report a violation rather than silently
    trusting the root.
    """
    n = draws.shape[0]

    def accept(T: float) -> tuple[bool, float]:
        p, _ = _joint_coverage(draws, ctrl=ctrl, T=T, method=method,
                               timing=timing, pm_min=pm_min, n_grid=n_grid)
        if conservative:
            lo, _hi = wilson_interval(int(round(p * n)), n)
            return lo >= target, p
        return p >= target, p

    coarse_T = np.linspace(0.15 * T_hi, T_hi, 9)
    coarse: list[tuple[float, float]] = []
    flags: list[bool] = []
    for t in coarse_T:
        ok, p = accept(float(t))
        coarse.append((float(t), p))
        flags.append(ok)
    probs = [p for _, p in coarse]
    monotone = all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1))

    if not flags[0]:                      # infeasible even at the smallest T
        return float("nan"), monotone, coarse
    if flags[-1]:
        return float(coarse_T[-1]), monotone, coarse

    lo, hi = float(coarse_T[0]), float(coarse_T[-1])
    for t, ok in zip(coarse_T, flags):
        if ok:
            lo = float(t)
        else:
            hi = float(t)
            break
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        ok, _p = accept(mid)
        if ok:
            lo = mid
        else:
            hi = mid
    return float(lo), monotone, coarse


def generate_bootstrap_chance() -> dict[str, float]:
    rng = np.random.default_rng(SEEDS["bootstrap"])
    truth = np.array([1.0, 10.0, 2.0])
    t = np.arange(0.0, 30.0 + 1e-12, 0.5)
    sigma = 0.03
    y_clean = step_response(t, *truth)
    y = y_clean + rng.normal(0.0, sigma, t.size)
    estimate = fit_foptd_step(t, y, truth.copy())

    # Centred residual bootstrap: removing the residual mean keeps the
    # resampling distribution unbiased for the fitted response.
    residual = y - step_response(t, *estimate)
    residual = residual - residual.mean()

    n_boot = 2000
    boot = np.empty((n_boot, 3))
    base = step_response(t, *estimate)
    for i in range(n_boot):
        yb = base + rng.choice(residual, size=t.size, replace=True)
        boot[i] = fit_foptd_step(t, yb, estimate)

    # Split before anything is selected, so validation is genuinely held out.
    perm = rng.permutation(n_boot)
    cal, val = boot[perm[: n_boot // 2]], boot[perm[n_boot // 2:]]

    K_hat, tau_hat, theta_hat = (float(v) for v in estimate)
    lam = theta_hat
    design = FOPTD(K=K_hat, tau=tau_hat, theta=theta_hat)
    ctrl = simc_pi(design, lam)
    timing = Timing(kappa=0.5, T_c0=0.30, extra_sample_delays=0,
                    label="immediate + 0.30 s transport")
    pm_min = 50.0
    alpha = 0.05
    A = math.pi / 2.0 - math.radians(pm_min)

    q95 = float(np.quantile(cal[:, 2], 1.0 - alpha))
    theta_mean_cal = float(cal[:, 2].mean())
    T_theta = (A * (lam + theta_hat) - q95 - timing.T_c0) / timing.kappa
    T_mean = (A * (lam + theta_hat) - theta_mean_cal - timing.T_c0) / timing.kappa
    if T_theta <= 0.0:
        raise RuntimeError(
            "conditional chance bound is infeasible: the required margin "
            "cannot be met even as T -> 0"
        )

    # ---- held-out checks of the conditional (dead-time only) design -------
    pm_sur_val = np.array([
        surrogate_pm_deg(th, theta_hat, lam, T_theta, timing.kappa, timing.T_c0)
        for th in val[:, 2]])
    pm_sur_mean_val = np.array([
        surrogate_pm_deg(th, theta_hat, lam, T_mean, timing.kappa, timing.T_c0)
        for th in val[:, 2]])
    scalar_val = np.column_stack([
        np.full(val.shape[0], K_hat), np.full(val.shape[0], tau_hat), val[:, 2]])
    cov_scalar_tu, pm_scalar_tu = _joint_coverage(
        scalar_val, ctrl=ctrl, T=T_theta, method="tustin", timing=timing,
        pm_min=pm_min, n_grid=6000)

    # ---- the same period, but with joint (K, tau, theta) uncertainty ------
    cov_joint_tu_at_scalar, pm_joint_tu_at_scalar = _joint_coverage(
        val, ctrl=ctrl, T=T_theta, method="tustin", timing=timing,
        pm_min=pm_min, n_grid=6000)
    cov_joint_be_at_scalar, _ = _joint_coverage(
        val, ctrl=ctrl, T=T_theta, method="backward_euler", timing=timing,
        pm_min=pm_min, n_grid=6000)

    # ---- implementation-specific calibration on the calibration half ------
    calibrated: dict[str, dict[str, float]] = {}
    pm_joint_val: dict[str, np.ndarray] = {}
    for method, key in (("tustin", "Tu"), ("backward_euler", "BE")):
        entry: dict[str, float] = {}
        for cons, tag in ((False, ""), (True, "_cons")):
            T_j, monotone, _coarse = _calibrate_joint_period(
                cal, ctrl=ctrl, method=method, timing=timing, pm_min=pm_min,
                target=1.0 - alpha, T_hi=T_theta, conservative=cons)
            if not math.isfinite(T_j):
                raise RuntimeError(
                    f"joint calibration infeasible for {method} "
                    f"(conservative={cons})"
                )
            cov, pm = _joint_coverage(val, ctrl=ctrl, T=T_j, method=method,
                                      timing=timing, pm_min=pm_min, n_grid=6000)
            lo, hi = wilson_interval(int(np.sum(pm >= pm_min)), pm.size)
            entry.update({
                f"T{tag}": T_j, f"coverage{tag}": cov,
                f"wilson_lo{tag}": lo, f"wilson_hi{tag}": hi,
                f"monotone_calibration{tag}": bool(monotone),
            })
            if not cons:
                pm_joint_val[key] = pm
            else:
                pm_joint_val[key + "_cons"] = pm
        calibrated[key] = entry

    with (OUT / "bootstrap_parameters.csv").open("w", newline="",
                                                 encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "K", "tau", "theta"])
        for row in cal:
            w.writerow(["calibration", *row.tolist()])
        for row in val:
            w.writerow(["validation", *row.tolist()])

    # ---------------------- figure ----------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    a = ax[0, 0]
    a.plot(t, y_clean, color=C["grey"], lw=1.2, label="ground truth")
    a.plot(t, y, "o", color=C["blue"], ms=3.2, mfc="none",
           label="synthetic observations")
    a.plot(t, step_response(t, *estimate), color=C["vermillion"], ls="--",
           label="FOPTD fit")
    a.set_xlabel("time [s]")
    a.set_ylabel("step response")
    a.set_title("(a) Identification experiment", fontsize=9)
    a.legend(loc="lower right")

    a = ax[0, 1]
    bins = np.linspace(min(cal[:, 2].min(), val[:, 2].min()),
                       max(cal[:, 2].max(), val[:, 2].max()), 36)
    a.hist(cal[:, 2], bins=bins, density=True, color=C["blue"], alpha=0.55,
           label=f"calibration ($n={cal.shape[0]}$)")
    a.hist(val[:, 2], bins=bins, density=True, histtype="step",
           color=C["vermillion"], lw=1.4,
           label=f"validation ($n={val.shape[0]}$)")
    a.axvline(q95, color=C["green"], ls=":", lw=1.6,
              label="calibration upper 95% quantile")
    a.axvline(truth[2], color=C["ink"], lw=1.0, label="ground truth")
    a.set_xlabel(r"dead-time estimate, $\theta$ [s]")
    a.set_ylabel("density")
    a.set_title("(b) Split residual bootstrap", fontsize=9)
    a.legend(loc="upper right", fontsize=7.2)

    a = ax[1, 0]
    xs = (np.arange(val.shape[0]) + 1.0) / val.shape[0]
    for arr, lab, col, ls in (
        (pm_sur_val, "scalar design, surrogate", C["ink"], "--"),
        (pm_scalar_tu, "scalar design, exact Tustin", C["sky"], "-"),
        (pm_joint_tu_at_scalar, "joint draws at scalar period", C["orange"], "-"),
        (pm_joint_val["Tu"], "joint calibrated, Tustin", C["green"], "-"),
        (pm_joint_val["BE"], "joint calibrated, backward Euler", C["blue"], "-"),
    ):
        a.plot(np.sort(arr), xs, color=col, ls=ls, lw=1.4, label=lab)
    a.axvline(pm_min, color=C["vermillion"], ls=":", lw=1.5,
              label="required margin")
    a.axhline(alpha, color=C["grey"], ls=":", lw=1.0)
    a.text(pm_min - 0.15, alpha + 0.012, r"$\alpha=5\%$", fontsize=6.8,
           color=C["grey"], ha="right")
    # The decision is made entirely in the lower tail; zoom onto it.
    a.set_xlim(46.5, 54.0)
    a.set_ylim(0.0, 0.42)
    a.set_xlabel("held-out phase margin [degree]")
    a.set_ylabel("empirical CDF")
    a.set_title("(c) Lower tail of the held-out margin", fontsize=9)
    a.legend(loc="upper left", fontsize=6.4)

    a = ax[1, 1]
    # Seven categories with multi-word names: horizontal bars keep every
    # label legible without rotation.
    names = ["scalar, surrogate", "scalar, exact Tustin",
             "joint draws at scalar $T$", "joint calibrated, Tustin",
             "joint calibrated, BE", "Wilson calibrated, Tustin",
             r"mean-$\theta$ surrogate"]
    vals = [100 * np.mean(pm_sur_val >= pm_min), 100 * cov_scalar_tu,
            100 * cov_joint_tu_at_scalar, 100 * calibrated["Tu"]["coverage"],
            100 * calibrated["BE"]["coverage"],
            100 * calibrated["Tu"]["coverage_cons"],
            100 * np.mean(pm_sur_mean_val >= pm_min)]
    cols = [C["ink"], C["sky"], C["orange"], C["green"], C["blue"],
            C["purple"], C["grey"]]
    ypos = np.arange(len(names))[::-1]
    a.barh(ypos, vals, color=cols, height=0.68)
    a.axvline(100 * (1 - alpha), color=C["vermillion"], ls="--", lw=1.3,
              label="95% target")
    for yv, v in zip(ypos, vals):
        a.text(v + 1.5, yv, f"{v:.1f}", va="center", fontsize=7.0)
    a.set_xlim(0, 118)
    a.set_yticks(ypos)
    a.set_yticklabels(names, fontsize=6.8)
    a.set_xlabel("held-out satisfaction [%]")
    a.set_title("(d) Achieved probability", fontsize=9)
    a.legend(loc="lower right", fontsize=6.8)
    _finish(fig, "fig11_bootstrap_chance")

    result = {
        "seed": SEEDS["bootstrap"],
        "n_bootstrap": n_boot,
        "n_calibration": int(cal.shape[0]),
        "n_validation": int(val.shape[0]),
        "estimate_K": K_hat,
        "estimate_tau": tau_hat,
        "estimate_theta": theta_hat,
        "theta_cal_mean": theta_mean_cal,
        "theta_cal_q95": q95,
        "theta_boot_sd": float(boot[:, 2].std(ddof=1)),
        "T_theta": float(T_theta),
        "T_mean": float(T_mean),
        "T_joint_Tu": calibrated["Tu"]["T"],
        "T_joint_BE": calibrated["BE"]["T"],
        "T_joint_Tu_cons": calibrated["Tu"]["T_cons"],
        "T_joint_BE_cons": calibrated["BE"]["T_cons"],
        "cov_joint_tustin_cons": calibrated["Tu"]["coverage_cons"],
        "cov_joint_be_cons": calibrated["BE"]["coverage_cons"],
        "wilson_joint_tustin_cons_lo": calibrated["Tu"]["wilson_lo_cons"],
        "wilson_joint_tustin_cons_hi": calibrated["Tu"]["wilson_hi_cons"],
        "wilson_joint_be_cons_lo": calibrated["BE"]["wilson_lo_cons"],
        "wilson_joint_be_cons_hi": calibrated["BE"]["wilson_hi_cons"],
        "cov_surrogate_val": float(np.mean(pm_sur_val >= pm_min)),
        "cov_scalar_tustin_val": float(cov_scalar_tu),
        "cov_joint_tustin_at_scalar": float(cov_joint_tu_at_scalar),
        "cov_joint_be_at_scalar": float(cov_joint_be_at_scalar),
        "cov_joint_tustin": calibrated["Tu"]["coverage"],
        "cov_joint_be": calibrated["BE"]["coverage"],
        "cov_surrogate_mean": float(np.mean(pm_sur_mean_val >= pm_min)),
        "wilson_joint_tustin_lo": calibrated["Tu"]["wilson_lo"],
        "wilson_joint_tustin_hi": calibrated["Tu"]["wilson_hi"],
        "wilson_joint_be_lo": calibrated["BE"]["wilson_lo"],
        "wilson_joint_be_hi": calibrated["BE"]["wilson_hi"],
        "monotone_calibration_tustin": calibrated["Tu"]["monotone_calibration"],
        "monotone_calibration_be": calibrated["BE"]["monotone_calibration"],
    }
    with (OUT / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


# ==========================================================================
# Implementation architecture: where each delay enters
# ==========================================================================
def generate_architecture_figure() -> dict[str, float]:
    """Where every term of the effective-delay budget physically enters."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.2)
    ax.grid(False)

    blocks = [
        ("Process", r"$Ke^{-\theta s}/(\tau s+1)$", C["grey"], r"$\theta$"),
        ("Sensor,\nanti-alias filter", "group delay", C["sky"],
         r"$c_{\rm filt}T$ or $T_{\rm filt,0}$"),
        ("Sampler,\nADC", "quantisation", C["sky"], "--"),
        ("PI/PID\n(digital)", r"$B$ fractional bits", C["blue"], "--"),
        ("Computation,\ncommunication", "scan / bus", C["orange"],
         r"$c_{\rm comp}T+T_{\rm comm}$"),
        ("Hold, PWM,\nactuator", "zero-order hold", C["vermillion"], r"$T/2$"),
    ]
    w, h, y0, gap = 1.42, 0.86, 1.72, 0.28
    xs = [0.18 + i * (w + gap) for i in range(len(blocks))]
    for x, (title, sub, col, _dly) in zip(xs, blocks):
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.2, edgecolor=col, facecolor=col + "20"))
        ax.text(x + w / 2, y0 + 0.56, title, ha="center", va="center",
                fontsize=7.8, weight="bold")
        ax.text(x + w / 2, y0 + 0.20, sub, ha="center", va="center",
                fontsize=6.9, color="#333333")
    for i in range(len(blocks) - 1):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=C["ink"]))

    # Feedback path routed cleanly below the chain, not through it.
    y_fb = y0 - 0.42
    x_end, x_start = xs[-1] + w / 2, xs[0] + w / 2
    ax.plot([x_end, x_end, x_start], [y0, y_fb, y_fb], color=C["ink"], lw=1.0)
    ax.add_patch(FancyArrowPatch((x_start, y_fb), (x_start, y0),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color=C["ink"]))

    # Delay contributions written under each block.
    for x, (_t, _s, col, dly) in zip(xs, blocks):
        if dly != "--":
            ax.text(x + w / 2, y_fb - 0.30, dly, ha="center", va="center",
                    fontsize=7.2, color=col)

    ax.text(5.2, 3.02,
            r"Upper endpoint $T_{\max}$: phase margin $\cdot$ load disturbance "
            r"$\cdot$ mode resolution", ha="center", fontsize=8.2,
            color=C["vermillion"], weight="bold")
    ax.text(5.2, 0.62,
            r"Lower endpoint $T_{\min}$: coefficient resolution $\cdot$ "
            r"derivative noise $\cdot$ actuator activity", ha="center",
            fontsize=8.2, color=C["blue"], weight="bold")
    ax.text(5.2, 0.20,
            r"$\theta_{\rm eff}=\theta+\kappa T+T_{c0}$,     "
            r"$\kappa=\frac{1}{2}+c_{\rm comp}+c_{\rm filt}$,     "
            r"$T_{c0}=T_{\rm comp,0}+T_{\rm comm}+T_{\rm filt,0}$",
            ha="center", fontsize=8.4)
    _finish(fig, "fig02_delay_architecture")
    return {"architecture_blocks": len(blocks)}


# ==========================================================================
# Application cases
# ==========================================================================
@dataclass
class CaseResult:
    name: str
    window: Window
    exact_pm: float
    target_pm: float
    omega_c: float
    T_used: float
    notes: str = ""


PROCESS_CASE = dict(
    model=FOPTD(K=1.8, tau=180.0, theta=40.0, label="jacketed-vessel temperature"),
    lam=40.0,
    # Synchronous-scan PLC: command applied at the next scan (c_comp = 1),
    # fixed 2.0 s analogue filter group delay + 0.15 s fieldbus transport.
    timing=Timing(kappa=1.5, T_c0=2.15, extra_sample_delays=1,
                  label="PLC next scan"),
    budgets=Budgets(dphi_deg=10.0, eps_iae=0.25, mode_fraction=None,
                    bits=12, eps_coef=0.02,
                    T_hardware=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)),
    rho_u=0.02,
)

# Digitally controlled buck converter, outer voltage loop.  L = 220 uH,
# C = 220 uF give omega_n = 1/sqrt(LC); zeta comes from ESR and load damping.
_L, _Cf = 220e-6, 220e-6
_WN = 1.0 / math.sqrt(_L * _Cf)
POWER_CASE = dict(
    model=SOPTD(K=1.0, zeta=0.20, omega_n=_WN, theta=20e-6,
                label="LC-filtered converter voltage loop"),
    lam=1.0e-3,
    # Duty updated on the next PWM boundary (c_comp = 1) plus 20 us of
    # ADC/averaging transport.
    timing=Timing(kappa=1.5, T_c0=20e-6, extra_sample_delays=1,
                  label="next PWM update"),
    budgets=Budgets(dphi_deg=10.0, eps_iae=None, mode_fraction=0.25,
                    bits=16, eps_coef=0.02, sigma_n=5e-3, V_umax=1e-4,
                    T_hardware=tuple(1.0 / f for f in
                                     (10e3, 20e3, 50e3, 100e3))),
    rho_d=None,
)


def _process_case() -> CaseResult:
    spec = PROCESS_CASE
    model, lam, timing = spec["model"], spec["lam"], spec["timing"]
    win = build_window(model=model, lam=lam, timing=timing,
                       budgets=spec["budgets"], rho_u=spec["rho_u"])
    ctrl = simc_pi(model, lam)
    T_used = win.T_hardware_selected if math.isfinite(win.T_hardware_selected) \
        else win.T_sel
    r = foptd_pi_margin(plant=model, ctrl=ctrl, T=T_used,
                        method="backward_euler", timing=timing, n_grid=9000)
    target = surrogate_pm_deg(model.theta, model.theta, lam, T_used,
                              timing.kappa, timing.T_c0)
    return CaseResult(
        name="Process loop (jacketed-vessel temperature, PLC)",
        window=win, exact_pm=r.pm_deg, target_pm=target,
        omega_c=1.0 / (lam + model.theta), T_used=T_used,
        notes=f"tau_I = {ctrl.tau_i:.1f} s, Kc = {ctrl.Kc:.3f}",
    )


def _power_case() -> CaseResult:
    spec = POWER_CASE
    model, lam, timing = spec["model"], spec["lam"], spec["timing"]
    ctrl = simc_pid_cancelling(model, lam)
    win = build_window(model=model, lam=lam, timing=timing,
                       budgets=spec["budgets"], Kc=ctrl.Kc, tau_d=ctrl.tau_d)
    T_used = win.T_hardware_selected if math.isfinite(win.T_hardware_selected) \
        else win.T_sel
    r = soptd_pid_margin(plant=model, ctrl=ctrl, T=T_used, timing=timing,
                         N=10.0, n_grid=9000)
    target = surrogate_pm_deg(model.theta, model.theta, lam, T_used,
                              timing.kappa, timing.T_c0)
    return CaseResult(
        name="Power-electronics loop (LC-filtered converter voltage)",
        window=win, exact_pm=r.pm_deg, target_pm=target,
        omega_c=1.0 / (lam + model.theta), T_used=T_used,
        notes=f"omega_n = {model.omega_n:.0f} rad/s "
              f"({model.omega_n/2/math.pi:.0f} Hz), tau_D = {ctrl.tau_d*1e6:.0f} us",
    )


def _bound_interval_panel(ax, case: CaseResult, *, scale: float, unit: str,
                          hw_labels: dict[float, str] | None = None) -> None:
    """Horizontal 'what bounds what' panel: the visual core of the method."""
    win = case.window
    lowers = [b for b in win.bounds if b.kind == "lower"]
    uppers = [b for b in win.bounds if b.kind == "upper"]
    names = [b.name for b in lowers] + [b.name for b in uppers]
    vals = [b.value * scale for b in lowers] + [b.value * scale for b in uppers]
    kinds = ["lower"] * len(lowers) + ["upper"] * len(uppers)
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    kinds = [kinds[i] for i in order]

    n_b = len(names)
    ax.set_xscale("log")
    lo_lim = min(min(vals), win.T_min * scale) / 3.5
    hi_lim = max(max(vals), win.T_max * scale) * 3.5
    ax.set_xlim(lo_lim, hi_lim)
    # Reserve a strip above the bounds for the hardware-period labels.
    ax.set_ylim(-0.6, n_b - 1 + (1.9 if hw_labels else 0.5))
    ax.axvspan(win.T_min * scale, win.T_max * scale, color=FEASIBLE_FILL,
               alpha=0.16, lw=0, label="feasible window")

    tick_labels = []
    for y, (n, v, k) in enumerate(zip(names, vals, kinds)):
        col = C["blue"] if k == "lower" else C["vermillion"]
        binding = ((k == "lower" and n == win.active_lower)
                   or (k == "upper" and n == win.active_upper))
        # A faint half-line showing which side the constraint permits.
        ax.plot([v, hi_lim if k == "lower" else lo_lim], [y, y],
                color=col, lw=1.1, alpha=0.28, solid_capstyle="butt")
        ax.plot([v], [y], marker=">" if k == "lower" else "<", color=col,
                ms=9 if binding else 6,
                mfc=col if binding else "white", mew=1.3)
        ax.text(v, y + 0.16, f"{v:.3g}", fontsize=7.0, color=col, ha="center")
        tick_labels.append(("$\\bf{" + n.replace(' ', r'\ ') + "}$")
                           if binding else n)
    ax.set_yticks(np.arange(n_b))
    ax.set_yticklabels(tick_labels, fontsize=7.4)

    if hw_labels:
        y_lab = n_b - 1 + 0.42
        # Stop the vertical rules just below the label row so the two never
        # overprint each other.
        y_lo, y_hi = ax.get_ylim()
        ymax_frac = (y_lab - 0.10 - y_lo) / (y_hi - y_lo)
        for T, lab in hw_labels.items():
            x = T * scale
            inside = win.T_min - 1e-15 <= T <= win.T_max + 1e-15
            ax.axvline(x, ymax=ymax_frac,
                       color=C["ink"] if inside else C["grey"],
                       ls="-" if inside else ":",
                       lw=1.3 if inside else 0.9,
                       alpha=0.85 if inside else 0.55)
            ax.text(x, y_lab, lab, rotation=90, fontsize=6.6, ha="center",
                    va="bottom", color=C["ink"] if inside else C["grey"],
                    weight="bold" if inside else "normal", clip_on=False)
    # The selected period often coincides with a hardware rule; draw it on
    # top and slightly wider so it stays visible.
    ax.axvline(case.T_used * scale, color=C["green"], lw=3.0, alpha=0.55,
               zorder=6)
    ax.set_xlabel(f"sampling period, $T$ [{unit}]")


def generate_applications() -> tuple[CaseResult, CaseResult, dict[str, float]]:
    proc = _process_case()
    powr = _power_case()

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    # ---- (a) process bounds ---------------------------------------------
    _bound_interval_panel(
        ax[0, 0], proc, scale=1.0, unit="s",
        hw_labels={T: f"{T:g} s" for T in PROCESS_CASE["budgets"].T_hardware})
    ax[0, 0].set_title("(a) Process loop: PLC scan periods")
    ax[0, 0].legend(loc="lower right", fontsize=7.2)

    # ---- (b) process exact check ----------------------------------------
    model, lam, timing = (PROCESS_CASE["model"], PROCESS_CASE["lam"],
                          PROCESS_CASE["timing"])
    ctrl = simc_pi(model, lam)
    Ts = np.geomspace(0.3, 14.0, 40)
    exact = np.array([foptd_pi_margin(plant=model, ctrl=ctrl, T=float(t),
                                      method="backward_euler", timing=timing,
                                      n_grid=6000).pm_deg for t in Ts])
    sur = np.array([surrogate_pm_deg(model.theta, model.theta, lam, float(t),
                                     timing.kappa, timing.T_c0) for t in Ts])
    a = ax[0, 1]
    a.axvspan(proc.window.T_min, proc.window.T_max, color=FEASIBLE_FILL,
              alpha=0.16, lw=0, label="feasible window")
    a.plot(Ts, sur, color=C["ink"], ls="--", label="design surrogate")
    a.plot(Ts, exact, color=C["blue"], label="exact pulse model, BE PI")
    pm0 = continuous_pm_deg(model.theta, model.theta, lam)
    a.axhline(pm0 - 10.0, color=C["vermillion"], ls=":", lw=1.3,
              label=r"$10^\circ$ loss target")
    a.axvline(proc.T_used, color=C["green"], lw=2.0, alpha=0.85)
    a.annotate(f"selected {proc.T_used:g} s\nexact PM {proc.exact_pm:.1f}$^\\circ$",
               xy=(proc.T_used, proc.exact_pm), xycoords="data",
               xytext=(0.60, 0.72), textcoords="axes fraction",
               fontsize=7.4, color=C["green"], ha="left",
               arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.9))
    a.set_xscale("log")
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("phase margin [degree]")
    a.set_title("(b) Process loop: exact check")
    a.legend(loc="lower left", fontsize=7.0)

    # ---- (c) converter bounds -------------------------------------------
    _bound_interval_panel(
        ax[1, 0], powr, scale=1e6, unit=r"$\mu$s",
        hw_labels={1.0 / f: f"{f/1e3:g} kHz" for f in (10e3, 20e3, 50e3, 100e3)})
    ax[1, 0].set_title("(c) Converter loop: PWM-tied periods")

    # ---- (d) converter exact check --------------------------------------
    pmodel, plam, ptiming = (POWER_CASE["model"], POWER_CASE["lam"],
                             POWER_CASE["timing"])
    pctrl = simc_pid_cancelling(pmodel, plam)
    Tp = np.geomspace(5e-6, 300e-6, 40)
    exact_p = np.array([soptd_pid_margin(plant=pmodel, ctrl=pctrl, T=float(t),
                                         timing=ptiming, N=10.0,
                                         n_grid=6000).pm_deg for t in Tp])
    sur_p = np.array([surrogate_pm_deg(pmodel.theta, pmodel.theta, plam,
                                       float(t), ptiming.kappa, ptiming.T_c0)
                      for t in Tp])
    a = ax[1, 1]
    a.axvspan(powr.window.T_min * 1e6, powr.window.T_max * 1e6,
              color=FEASIBLE_FILL, alpha=0.16, lw=0, label="feasible window")
    a.plot(Tp * 1e6, sur_p, color=C["ink"], ls="--", label="design surrogate")
    a.plot(Tp * 1e6, exact_p, color=C["blue"], label="exact ZOH model, Tustin PID")
    T_mode = 0.25 * math.pi / pmodel.omega_d
    a.axvline(T_mode * 1e6, color=C["orange"], ls="-.", lw=1.2,
              label=r"$\omega_d T=\pi/4$")
    a.axvline(powr.T_used * 1e6, color=C["green"], lw=2.0, alpha=0.85)
    a.annotate(f"selected {powr.T_used*1e6:.0f} $\\mu$s "
               f"({1e-3/powr.T_used:.0f} kHz)\nPM {powr.exact_pm:.1f}$^\\circ$",
               xy=(powr.T_used * 1e6, powr.exact_pm), xycoords="data",
               xytext=(0.05, 0.30), textcoords="axes fraction",
               fontsize=7.4, color=C["green"], ha="left",
               arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.9))
    a.set_xscale("log")
    a.set_xlabel(r"sampling period, $T$ [$\mu$s]")
    a.set_ylabel("phase margin [degree]")
    a.set_title("(d) Converter loop: exact check")
    a.legend(loc="lower left", fontsize=7.2)
    _finish(fig, "fig12_applications")

    summary = {
        "process": {
            "T_min": proc.window.T_min, "T_max": proc.window.T_max,
            "T_cost": proc.window.T_cost, "T_sel": proc.window.T_sel,
            "T_used": proc.T_used, "exact_pm": proc.exact_pm,
            "target_pm": proc.target_pm,
            "active_lower": proc.window.active_lower,
            "active_upper": proc.window.active_upper,
            "hardware_feasible": list(proc.window.T_hardware_feasible),
            "bounds": {b.name: b.value for b in proc.window.bounds},
            "diagnosis": proc.window.diagnosis(),
        },
        "power_electronics": {
            "T_min": powr.window.T_min, "T_max": powr.window.T_max,
            "T_sel": powr.window.T_sel, "T_used": powr.T_used,
            "exact_pm": powr.exact_pm, "target_pm": powr.target_pm,
            "active_lower": powr.window.active_lower,
            "active_upper": powr.window.active_upper,
            "hardware_feasible": list(powr.window.T_hardware_feasible),
            "f_switching_hz": 1.0 / powr.T_used,
            "omega_d_T": POWER_CASE["model"].omega_d * powr.T_used,
            "bounds": {b.name: b.value for b in powr.window.bounds},
            "diagnosis": powr.window.diagnosis(),
        },
    }
    with (OUT / "application_cases.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return proc, powr, summary


# ==========================================================================
# Window map: what closes the window
# ==========================================================================
def generate_window_map(proc: CaseResult) -> dict[str, float]:
    spec = PROCESS_CASE
    model, lam0, timing = spec["model"], spec["lam"], spec["timing"]
    budgets = spec["budgets"]

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.85), constrained_layout=True)

    # ---- (a) window against dead-time ratio ------------------------------
    ratios = np.linspace(0.05, 0.6, 90)
    lo, hi = [], []
    per_bound: dict[str, list[float]] = {}
    for r in ratios:
        m = FOPTD(K=model.K, tau=model.tau, theta=r * model.tau)
        w = build_window(model=m, lam=m.theta, timing=timing, budgets=budgets,
                         rho_u=spec["rho_u"])
        lo.append(w.T_min)
        hi.append(w.T_max)
        for b in w.bounds:
            per_bound.setdefault(b.name, []).append(b.value)
    lo, hi = np.array(lo), np.array(hi)
    a = ax[0]
    _band(a, ratios, lo, hi, label="feasible window")
    styles = {"phase margin": (C["vermillion"], "-"),
              "load disturbance": (C["orange"], "-"),
              "coefficient resolution": (C["blue"], "-")}
    for name, series in per_bound.items():
        col, ls = styles.get(name, (C["grey"], "-"))
        a.plot(ratios, series, color=col, ls=ls, lw=1.3, label=name)
    a.axvline(model.theta / model.tau, color=C["green"], lw=1.4, ls="-",
              alpha=0.75)
    a.set_yscale("log")
    a.set_ylim(0.5, 60.0)
    a.set_xlabel(r"dead-time ratio, $\theta/\tau$")
    a.set_ylabel("sampling period, $T$ [s]")
    a.set_title(r"(a) Window vs. dead time", fontsize=9)
    a.legend(loc="upper left", fontsize=6.2)

    # ---- (b) window closure against word length --------------------------
    bits = np.arange(6, 21)
    lo_b, hi_b = [], []
    for B in bits:
        w = build_window(model=model, lam=lam0, timing=timing,
                         budgets=Budgets(**{**asdict_budgets(budgets),
                                            "bits": int(B)}),
                         rho_u=spec["rho_u"])
        lo_b.append(w.T_min)
        hi_b.append(w.T_max)
    lo_b, hi_b = np.array(lo_b), np.array(hi_b)
    a = ax[1]
    closed = lo_b > hi_b
    _band(a, bits, lo_b, hi_b, label="feasible window")
    if closed.any():
        a.fill_between(bits, hi_b, lo_b, where=closed, color=INFEASIBLE_FILL,
                       alpha=0.20, lw=0, hatch="///", label="infeasible")
    a.plot(bits, lo_b, color=C["blue"], label=r"$T_{\min}$ (coefficients)")
    a.plot(bits, hi_b, color=C["vermillion"], label=r"$T_{\max}$ (performance)")
    B_close = int(bits[closed].max()) if closed.any() else -1
    if B_close > 0:
        a.axvline(B_close + 0.5, color=C["ink"], ls=":", lw=1.2)
        a.annotate(f"window opens\nat $B={B_close+1}$",
                   xy=(B_close + 0.5, 3.0), xycoords="data",
                   xytext=(0.52, 0.05), textcoords="axes fraction",
                   fontsize=6.8, ha="left",
                   arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.8))
    a.set_yscale("log")
    a.set_ylim(0.3, 300.0)
    a.set_xlabel("fractional bits, $B$")
    a.set_ylabel("sampling period, $T$ [s]")
    a.set_title("(b) Window vs. precision", fontsize=9)
    a.legend(loc="upper right", fontsize=6.2)

    # ---- (c) active-constraint map ---------------------------------------
    rr = np.linspace(0.05, 0.8, 70)
    ll = np.linspace(0.4, 3.0, 70)
    code = np.zeros((ll.size, rr.size))
    names = ["phase margin", "load disturbance", "infeasible"]
    for i, lr in enumerate(ll):
        for j, r in enumerate(rr):
            m = FOPTD(K=model.K, tau=model.tau, theta=r * model.tau)
            w = build_window(model=m, lam=lr * m.theta, timing=timing,
                             budgets=budgets, rho_u=spec["rho_u"])
            if not w.feasible:
                code[i, j] = 2
            else:
                code[i, j] = names.index(w.active_upper)
    a = ax[2]
    # Strongly separated fills: the two upper constraints must not read as
    # one colour when the map is printed.
    fills = [C["vermillion"], "#FBE0A6", "#c9c9c9"]
    cmap = matplotlib.colors.ListedColormap(fills)
    a.pcolormesh(rr, ll, code, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
    a.contour(rr, ll, code, levels=[0.5, 1.5], colors=C["ink"],
              linewidths=0.7)
    a.plot(model.theta / model.tau, lam0 / model.theta, "o", color=C["green"],
           ms=7, mec="white", mew=1.4, zorder=5)
    a.annotate("case study", xy=(model.theta / model.tau, lam0 / model.theta),
               xytext=(0.42, 0.12), textcoords="axes fraction", fontsize=6.8,
               color=C["green"],
               arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.9))
    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor=C["ink"],
                                        linewidth=0.5, label=n)
               for c, n in zip(fills, names)]
    a.legend(handles=handles, loc="upper right", fontsize=6.2)
    a.set_xlabel(r"dead-time ratio, $\theta/\tau$")
    a.set_ylabel(r"tuning ratio, $\lambda/\theta$")
    a.set_title("(c) Binding constraint", fontsize=9)
    a.grid(False)
    _finish(fig, "fig08_window_map")

    return {
        "window_closes_below_bits": int(B_close + 1) if B_close > 0 else 0,
        "map_fraction_iae_limited": float(np.mean(code == 1)),
        "map_fraction_infeasible": float(np.mean(code == 2)),
    }


def asdict_budgets(b: Budgets) -> dict:
    return {
        "dphi_deg": b.dphi_deg, "eps_iae": b.eps_iae,
        "mode_fraction": b.mode_fraction, "bits": b.bits,
        "eps_coef": b.eps_coef, "sigma_n": b.sigma_n, "V_umax": b.V_umax,
        "z0_tol": b.z0_tol, "T_hardware": b.T_hardware,
    }


# ==========================================================================
# Dimensionless design chart and local robustness bands
# ==========================================================================
def generate_design_chart() -> dict[str, float]:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    # ---- (a) dimensionless chart -----------------------------------------
    # x = theta/(lambda+theta) in (0,1); y = kappa T/(lambda+theta).
    x = np.linspace(0.02, 0.98, 200)
    a = ax[0]
    # The phase-margin bound is a horizontal line in these coordinates, so
    # label the three budgets inline rather than spending legend rows on them.
    for dphi, col in ((5.0, C["sky"]), (10.0, C["vermillion"]),
                      (15.0, C["purple"])):
        yv = math.radians(dphi)
        a.axhline(yv, color=col, lw=1.4)
        a.text(0.015, yv + 0.006, rf"$\Delta\varphi_a={dphi:g}^\circ$",
               fontsize=6.8, color=col, ha="left", va="bottom")
    eps = 0.25
    a.plot(x, eps * x, color=C["orange"], lw=1.4,
           label=rf"IAE linear, $\varepsilon={eps:g}$")
    a.plot(x, (math.sqrt(1 + eps) - 1) * x, color=C["green"], lw=1.4,
           label=rf"IAE quadratic, $\varepsilon={eps:g}$")
    upper = np.minimum(math.radians(10.0), eps * x)
    _band(a, x, np.zeros_like(x), upper,
          label=r"feasible ($10^\circ$, $25\%$)")
    a.set_xlabel(r"delay fraction, $\theta/(\lambda+\theta)$")
    a.set_ylabel(r"normalised period, $\kappa T/(\lambda+\theta)$")
    a.set_ylim(0, 0.30)
    a.set_title("(a) Unit-free design chart", fontsize=9)
    a.legend(loc="lower right", fontsize=6.4)

    # ---- (b) local robustness bands --------------------------------------
    rng = np.random.default_rng(SEEDS["robustness"])
    model, lam, timing = (PROCESS_CASE["model"], PROCESS_CASE["lam"],
                          PROCESS_CASE["timing"])
    ctrl = simc_pi(model, lam)          # frozen on the identified model
    Ts = np.geomspace(0.5, 12.0, 26)
    n_draw = 200
    spread = 0.20
    draws = np.column_stack([
        model.K * (1 + spread * rng.uniform(-1, 1, n_draw)),
        model.tau * (1 + spread * rng.uniform(-1, 1, n_draw)),
        model.theta * (1 + spread * rng.uniform(-1, 1, n_draw)),
    ])
    lo, med, hi, nom = [], [], [], []
    for T in Ts:
        pm = np.array([
            foptd_pi_margin(plant=FOPTD(float(k), float(tt), float(th)),
                            ctrl=ctrl, T=float(T), method="backward_euler",
                            timing=timing, n_grid=4000).pm_deg
            for k, tt, th in draws])
        lo.append(np.nanmin(pm))
        med.append(np.nanmedian(pm))
        hi.append(np.nanmax(pm))
        nom.append(foptd_pi_margin(plant=model, ctrl=ctrl, T=float(T),
                                   method="backward_euler", timing=timing,
                                   n_grid=4000).pm_deg)
    lo, med, hi, nom = map(np.array, (lo, med, hi, nom))

    a = ax[1]
    a.fill_between(Ts, lo, hi, color=C["blue"], alpha=0.18, lw=0,
                   label=r"$\pm20\%$ in $K,\tau,\theta$ (range)")
    a.plot(Ts, med, color=C["blue"], label="median over draws")
    a.plot(Ts, nom, color=C["ink"], ls="--", label="nominal model")
    a.axhline(45.0, color=C["vermillion"], ls=":", lw=1.3,
              label=r"$\mathrm{PM}_{\min}=45^\circ$")
    win = build_window(model=model, lam=lam, timing=timing,
                       budgets=PROCESS_CASE["budgets"],
                       rho_u=PROCESS_CASE["rho_u"])
    a.axvspan(win.T_min, win.T_max, color=FEASIBLE_FILL, alpha=0.14, lw=0,
              label="nominal window")
    a.set_xscale("log")
    a.set_xlabel("sampling period, $T$ [s]")
    a.set_ylabel("exact phase margin [degree]")
    a.set_title("(b) Robustness band", fontsize=9)
    a.legend(loc="lower left", fontsize=6.2)
    _finish(fig, "fig09_design_chart")

    # Robustness summary at the selected hardware period.
    T_sel = 2.0
    pm_sel = np.array([
        foptd_pi_margin(plant=FOPTD(float(k), float(tt), float(th)), ctrl=ctrl,
                        T=T_sel, method="backward_euler", timing=timing,
                        n_grid=6000).pm_deg for k, tt, th in draws])
    return {
        "rob_spread": spread,
        "rob_draws": n_draw,
        "rob_T": T_sel,
        "rob_worst_pm": float(np.nanmin(pm_sel)),
        "rob_median_pm": float(np.nanmedian(pm_sel)),
        "rob_sat_prob_45": float(np.mean(pm_sel >= 45.0)),
        "rob_nominal_pm": float(foptd_pi_margin(
            plant=model, ctrl=ctrl, T=T_sel, method="backward_euler",
            timing=timing, n_grid=6000).pm_deg),
    }


# ==========================================================================
# Running-example scalar checks used verbatim in the manuscript text
# ==========================================================================
def running_example_numbers() -> dict[str, float]:
    model, lam = RUN_MODEL, RUN_LAM
    theta, tau = model.theta, model.tau
    wc = 1.0 / (lam + theta)

    T_cost = T_cost_update(theta_0=theta, kappa=0.5, rho_u=RUN_RHO_U)
    T_min_pole = tau * 2.0 ** (-13) / 0.02
    T_max_iae = 2.0 * 0.25 * theta                     # linear regime, ideal
    lag = LAG_MODEL
    T_max_iae_lag = 2.0 * lag.theta * (math.sqrt(1.25) - 1.0)

    # Bandwidth-rule comparison
    T10 = 2.0 * math.pi / (10.0 * wc)
    T20 = 2.0 * math.pi / (20.0 * wc)

    # SOPTD cost example from the text: the stated weight ratio is
    # beta/(alpha theta^4) = 0.01, and rho_d = beta kappa^2/(alpha theta_0^4),
    # so the dimensionless parameter carries the kappa^2 factor.
    T_cost_soptd = T_cost_noise(theta_0=0.5, kappa=0.5, rho_d=0.01 * 0.5**2)

    return {
        "run_wc": wc,
        "run_pm0": continuous_pm_deg(theta, theta, lam),
        "run_T_cost": T_cost,
        "run_T_min_pole": T_min_pole,
        "run_T_max_iae": T_max_iae,
        "lag_T_max_iae": T_max_iae_lag,
        "bandwidth_T10": T10,
        "bandwidth_loss10_deg": math.degrees(0.5 * T10 * wc),
        "bandwidth_T20": T20,
        "bandwidth_loss20_deg": math.degrees(0.5 * T20 * wc),
        "soptd_T_cost": T_cost_soptd,
        "soptd_zero_at_cost": sampling_zero_exact(0.4, 1.0, T_cost_soptd),
    }


# ==========================================================================
# LaTeX outputs
# ==========================================================================
def write_latex_outputs(ctx: dict) -> None:
    run = ctx["running"]
    groups = ctx["benchmark_groups"]
    bench = ctx["benchmark_overall"]
    soptd = ctx["soptd"]
    soptd_groups = ctx["soptd_groups"]
    boot = ctx["bootstrap"]
    apps = ctx["applications"]
    rob = ctx["robustness"]
    scal = ctx["scalars"]
    wmap = ctx["window_map"]
    misc = ctx["misc"]

    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}\n"

    L = ["% Generated by validation_ejc.py; do not edit manually.\n"]
    L += [
        # --- benchmark ---
        cmd("BenchN", f"{bench['n_cases']}"),
        cmd("BenchWorstShortfall", f"{abs(bench['worst_shortfall']):.2f}"),
        cmd("BenchMaxCrossings", f"{bench['max_crossings']}"),
        # --- SOPTD ---
        cmd("SoptdN", f"{int(soptd['n_cases'])}"),
        cmd("SoptdDistinct", f"{int(soptd['n_distinct'])}"),
        cmd("SoptdMedianAbs", f"{soptd['median_abs_error']:.2f}"),
        cmd("SoptdMaxAbs", f"{soptd['max_abs_error']:.2f}"),
        cmd("SoptdWorstShortfall", f"{abs(soptd['worst_shortfall']):.2f}"),
        cmd("SoptdMaxModePhase", f"{soptd['max_omega_d_T']:.2f}"),
        cmd("SoptdModeLimited", f"{int(soptd['n_mode_limited'])}"),
        # --- bootstrap ---
        cmd("BootThetaHat", f"{boot['estimate_theta']:.3f}"),
        cmd("BootThetaQ", f"{boot['theta_cal_q95']:.3f}"),
        cmd("BootTtheta", f"{boot['T_theta']:.3f}"),
        cmd("BootTmean", f"{boot['T_mean']:.3f}"),
        cmd("BootTjointTu", f"{boot['T_joint_Tu']:.3f}"),
        cmd("BootTjointBE", f"{boot['T_joint_BE']:.3f}"),
        cmd("BootCovSurVal", f"{100*boot['cov_surrogate_val']:.1f}"),
        cmd("BootCovScalarTuVal", f"{100*boot['cov_scalar_tustin_val']:.1f}"),
        cmd("BootCovJointTuAtScalar",
            f"{100*boot['cov_joint_tustin_at_scalar']:.1f}"),
        cmd("BootCovJointBEAtScalar",
            f"{100*boot['cov_joint_be_at_scalar']:.1f}"),
        cmd("BootCovJointTu", f"{100*boot['cov_joint_tustin']:.1f}"),
        cmd("BootCovJointBE", f"{100*boot['cov_joint_be']:.1f}"),
        cmd("BootCovMean", f"{100*boot['cov_surrogate_mean']:.1f}"),
        cmd("BootJointTuCILo", f"{100*boot['wilson_joint_tustin_lo']:.1f}"),
        cmd("BootJointTuCIHi", f"{100*boot['wilson_joint_tustin_hi']:.1f}"),
        cmd("BootJointBECILo", f"{100*boot['wilson_joint_be_lo']:.1f}"),
        cmd("BootJointBECIHi", f"{100*boot['wilson_joint_be_hi']:.1f}"),
        cmd("BootNCal", f"{boot['n_calibration']}"),
        cmd("BootNVal", f"{boot['n_validation']}"),
        cmd("BootTjointTuCons", f"{boot['T_joint_Tu_cons']:.3f}"),
        cmd("BootTjointBECons", f"{boot['T_joint_BE_cons']:.3f}"),
        cmd("BootCovJointTuCons", f"{100*boot['cov_joint_tustin_cons']:.1f}"),
        cmd("BootCovJointBECons", f"{100*boot['cov_joint_be_cons']:.1f}"),
        cmd("BootJointTuConsCILo",
            f"{100*boot['wilson_joint_tustin_cons_lo']:.1f}"),
        cmd("BootJointTuConsCIHi",
            f"{100*boot['wilson_joint_tustin_cons_hi']:.1f}"),
        # --- supporting studies ---
        cmd("NoiseSlope", f"{misc['noise_slope']:.3f}"),
        cmd("IaeRatioLo", f"{misc['iae_ratio_min']:.3f}"),
        cmd("IaeRatioHi", f"{misc['iae_ratio_max']:.3f}"),
        cmd("RunMaxCrossings", f"{run['running_max_crossings']}"),
        cmd("RunTstabBE", f"{run['T_stability_backward_euler']:.1f}"),
        cmd("RunTstabTU", f"{run['T_stability_tustin']:.1f}"),
        cmd("RunTmaxPM", f"{run['T_max_pm_immediate']:.2f}"),
        cmd("RunTmaxPMNext", f"{run['T_max_pm_next_scan']:.2f}"),
        cmd("RunTzeroMargin", f"{run['T_zero_margin_immediate']:.2f}"),
        cmd("RunMarginRatio",
            f"{run['T_zero_margin_immediate']/run['T_max_pm_immediate']:.1f}"),
        cmd("RunStabRatioBE",
            f"{run['T_stability_backward_euler']/run['T_max_pm_immediate']:.1f}"),
        cmd("RunStabRatioTU",
            f"{run['T_stability_tustin']/run['T_max_pm_immediate']:.1f}"),
        cmd("RunTcost", f"{scal['run_T_cost']:.3f}"),
        cmd("RunTminPole", f"{scal['run_T_min_pole']:.3f}"),
        cmd("RunTmaxIAE", f"{scal['run_T_max_iae']:.2f}"),
        cmd("LagTmaxIAE", f"{scal['lag_T_max_iae']:.2f}"),
        cmd("BandTten", f"{scal['bandwidth_T10']:.2f}"),
        cmd("BandLossTen", f"{scal['bandwidth_loss10_deg']:.0f}"),
        cmd("BandTtwenty", f"{scal['bandwidth_T20']:.2f}"),
        cmd("BandLossTwenty", f"{scal['bandwidth_loss20_deg']:.0f}"),
        cmd("WordTwelve", f"{1e3*misc['word_Tmin_B12']:.0f}"),
        cmd("WordSixteen", f"{1e3*misc['word_Tmin_B16']:.1f}"),
        cmd("ZeroExact", f"{misc['zero_exact_ref']:.4f}"),
        cmd("ZeroFirst", f"{misc['zero_first_order_ref']:.4f}"),
        cmd("ZeroPct", f"{misc['zero_rel_error_pct']:.1f}"),
        cmd("SoptdTcost", f"{scal['soptd_T_cost']:.2f}"),
        cmd("SoptdZeroAtCost", f"{scal['soptd_zero_at_cost']:.3f}"),
        cmd("ActTVSlope", f"{misc['actuator_tv_slope']:.2f}"),
        cmd("ActRevSlope", f"{misc['actuator_rev_slope']:.2f}"),
        # --- applications ---
        cmd("ProcTmin", f"{apps['process']['T_min']:.2f}"),
        cmd("ProcTmax", f"{apps['process']['T_max']:.2f}"),
        cmd("ProcTcost", f"{apps['process']['T_cost']:.2f}"),
        cmd("ProcTsel", f"{apps['process']['T_used']:g}"),
        cmd("ProcActiveUpper", apps["process"]["active_upper"]),
        cmd("ProcActiveLower", apps["process"]["active_lower"]),
        cmd("ProcPMexact", f"{apps['process']['exact_pm']:.1f}"),
        cmd("ProcPMtarget", f"{apps['process']['target_pm']:.1f}"),
        cmd("ProcGridAdmissible",
            ", ".join(f"{v:g}" for v in apps["process"]["hardware_feasible"])),
        cmd("PowTmin", f"{1e6*apps['power_electronics']['T_min']:.1f}"),
        cmd("PowTmax", f"{1e6*apps['power_electronics']['T_max']:.1f}"),
        cmd("PowTsel", f"{1e6*apps['power_electronics']['T_used']:.0f}"),
        cmd("PowFsw", f"{1e-3/apps['power_electronics']['T_used']:.0f}"),
        cmd("PowActiveUpper", apps["power_electronics"]["active_upper"]),
        cmd("PowActiveLower", apps["power_electronics"]["active_lower"]),
        cmd("PowPMexact", f"{apps['power_electronics']['exact_pm']:.1f}"),
        cmd("PowPMtarget", f"{apps['power_electronics']['target_pm']:.1f}"),
        cmd("PowModePhase", f"{apps['power_electronics']['omega_d_T']:.2f}"),
        cmd("PowFres", f"{POWER_CASE['model'].omega_n/(2*math.pi):.0f}"),
        cmd("PowTmodeRes",
            f"{1e6*apps['power_electronics']['bounds']['mode resolution']:.0f}"),
        cmd("PowTcoefRes",
            f"{1e6*apps['power_electronics']['bounds']['coefficient resolution']:.2f}"),
        cmd("ProcTmaxPM",
            f"{apps['process']['bounds']['phase margin']:.2f}"),
        # --- window map and robustness ---
        cmd("WindowOpensBits", f"{wmap['window_closes_below_bits']}"),
        cmd("MapFracIAE", f"{100*wmap['map_fraction_iae_limited']:.0f}"),
        cmd("RobSpread", f"{100*rob['rob_spread']:.0f}"),
        cmd("RobDraws", f"{rob['rob_draws']}"),
        cmd("RobWorstPM", f"{rob['rob_worst_pm']:.1f}"),
        cmd("RobMedianPM", f"{rob['rob_median_pm']:.1f}"),
        cmd("RobNominalPM", f"{rob['rob_nominal_pm']:.1f}"),
        cmd("RobSatProb", f"{100*rob['rob_sat_prob_45']:.1f}"),
    ]
    (OUT / "validation_macros.tex").write_text("".join(L), encoding="utf-8")

    # ------------------------------------------------------------------
    # Benchmark aggregate table
    # ------------------------------------------------------------------
    tbl = r"""% Generated by validation_ejc.py
\begin{table}[pos=!htb]
\centering
\caption{Exact pulse-transfer error at the analytical $10^\circ$ phase-loss
period over 12 plant/tuning cases per implementation. Negative values denote
margin shortfall; positive values denote conservatism.}
\label{tab:benchmark-aggregate}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Implementation & Median $|e_{\rm PM}|$ & Worst shortfall & Maximum excess\\
 & [$^\circ$] & [$^\circ$] & [$^\circ$]\\
\midrule
"""
    labels = {
        "immediate; backward Euler": "Immediate, backward Euler",
        "immediate; Tustin": "Immediate, Tustin",
        "next scan; backward Euler": "Next scan, backward Euler",
        "next scan; Tustin": "Next scan, Tustin",
    }
    for key, lab in labels.items():
        g = groups[key]
        tbl += (f"{lab} & {g['median_abs_error']:.2f} & "
                f"{min(g['min_error'], 0.0):.2f} & "
                f"{max(g['max_error'], 0.0):.2f}" + r"\\" + "\n")
    tbl += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT / "benchmark_summary_table.tex").write_text(tbl, encoding="utf-8")

    # ------------------------------------------------------------------
    # SOPTD table
    # ------------------------------------------------------------------
    st = r"""% Generated by validation_ejc.py
\begin{table}[pos=!htb]
\centering
\caption{SOPTD exact sampled-data check at the window period
$\min\{T_{\max}^{\rm PM},T_{\max}^{\rm mode}\}$. Modal mismatch is applied to
the $\zeta$ and $\omega_n$ used for the cancelling PID settings.}
\label{tab:soptd-validation}
\small
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Case & $n$ & Median $|e|$ & Worst shortfall & Maximum excess\\
 & & [$^\circ$] & [$^\circ$] & [$^\circ$]\\
\midrule
"""
    for key, g in soptd_groups.items():
        arch, mm = key.split("; ")
        name = ("Immediate" if arch == "immediate" else "Next scan")
        mm_lab = "exact" if mm in ("+0%", "-0%") else mm.replace("%", r"\%")
        st += (f"{name}, {mm_lab} & {int(g['n'])} & {g['median_abs_error']:.2f} & "
               f"{g['worst_shortfall']:.2f} & {g['max_excess']:.2f}"
               + r"\\" + "\n")
    st += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT / "soptd_validation_table.tex").write_text(st, encoding="utf-8")

    # ------------------------------------------------------------------
    # Application worked-example table
    # ------------------------------------------------------------------
    p, q = apps["process"], apps["power_electronics"]
    at = r"""% Generated by validation_ejc.py
\begin{table}[pos=!htb]
\centering
\caption{Two illustrative engineering case studies. Both windows are
two-sided: the process loop is bounded below by coefficient resolution and
above by load-disturbance degradation, the converter loop below by derivative
noise and above by phase-margin erosion. Neither case is a hardware
validation.}
\label{tab:applications}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.34\linewidth}XX@{}}
\toprule
Quantity & Process loop & Converter loop\\
\midrule
"""
    rows = [
        ("Plant", r"FOPTD, $K=1.8$, $\tau=\SI{180}{\second}$, "
                  r"$\theta=\SI{40}{\second}$",
         r"SOPTD, $\zeta=0.20$, $f_n=\PowFres\,$Hz, "
         r"$\theta=\SI{20}{\micro\second}$"),
        ("Timing", r"next scan, $\kappa=3/2$, $T_{c0}=\SI{2.15}{\second}$",
         r"next PWM update, $\kappa=3/2$, $T_{c0}=\SI{20}{\micro\second}$"),
        ("Controller", r"SIMC PI, backward Euler",
         r"PID with filtered derivative ($N=10$), Tustin"),
        ("Coefficient word length", r"$B=12$ fractional bits",
         r"$B=16$ fractional bits"),
        (r"$T_{\min}$", r"\ProcTmin\,s (\ProcActiveLower)",
         r"\PowTmin\,$\mu$s (\PowActiveLower)"),
        (r"$T_{\max}$", r"\ProcTmax\,s (\ProcActiveUpper)",
         r"\PowTmax\,$\mu$s (\PowActiveUpper)"),
        ("Implementable periods in window",
         r"\ProcGridAdmissible\,s (of 0.1--10\,s scan set)",
         r"100 and 50\,$\mu$s (of 10--100\,kHz set)"),
        ("Selected period", r"\ProcTsel\,s",
         r"\PowTsel\,$\mu$s ($f_s=\PowFsw$\,kHz)"),
        ("Surrogate margin at selection", r"\ProcPMtarget$^\circ$",
         r"\PowPMtarget$^\circ$"),
        ("Exact sampled-data margin", r"\ProcPMexact$^\circ$",
         r"\PowPMexact$^\circ$"),
    ]
    for a_, b_, c_ in rows:
        at += f"{a_} & {b_} & {c_}" + r"\\" + "\n"
    at += "\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    (OUT / "application_table.tex").write_text(at, encoding="utf-8")


# ==========================================================================
# Driver
# ==========================================================================
def main() -> None:
    steps = [
        "hold / half-sample delay",
        "delay architecture diagram",
        "exact margin validation and stability limits",
        "load-disturbance IAE law",
        "hidden oscillations",
        "sampling zeros and actuator activity",
        "fast-sampling lower bounds",
        "FOPTD phase-budget benchmark",
        "SOPTD sampled-data benchmark",
        "application case studies",
        "window map and design chart",
        "split-bootstrap uncertainty study",
    ]
    n = len(steps)

    def step(i: int) -> None:
        print(f"[{i}/{n}] {steps[i-1]}", flush=True)

    step(1); zoh = generate_zoh_figure()
    step(2); arch = generate_architecture_figure()
    step(3); running = generate_margin_validation()
    step(4); iae = generate_iae_law()
    step(5); hidden = generate_hidden_oscillations()
    step(6); zero = generate_sampling_zero()
    step(7); lower = generate_lower_bounds()
    step(8); _, groups, bench_overall = generate_benchmark()
    step(9); _, soptd, soptd_groups = generate_soptd_validation()
    step(10); proc, powr, apps = generate_applications()
    step(11); wmap = generate_window_map(proc); rob = generate_design_chart()
    step(12); boot = generate_bootstrap_chance()
    scalars = running_example_numbers()

    misc = {**zoh, **iae, **hidden, **zero, **lower, **arch}
    ctx = {
        "running": running,
        "benchmark_groups": groups,
        "benchmark_overall": bench_overall,
        "soptd": soptd,
        "soptd_groups": soptd_groups,
        "bootstrap": boot,
        "applications": apps,
        "robustness": rob,
        "scalars": scalars,
        "window_map": wmap,
        "misc": misc,
    }
    write_latex_outputs(ctx)
    (OUT / "validation_summary.json").write_text(
        json.dumps(ctx, indent=2, default=float), encoding="utf-8")

    print("\n--- headline results ---")
    print(f"FOPTD benchmark   : worst shortfall "
          f"{bench_overall['worst_shortfall']:+.2f} deg over "
          f"{bench_overall['n_cases']} cases, max crossings "
          f"{bench_overall['max_crossings']}")
    print(f"SOPTD benchmark   : worst shortfall "
          f"{soptd['worst_shortfall']:+.2f} deg over {soptd['n_cases']} cases, "
          f"max omega_d*T = {soptd['max_omega_d_T']:.2f}")
    print(f"bootstrap (Tustin): scalar {boot['T_theta']:.3f} s -> "
          f"{100*boot['cov_joint_tustin_at_scalar']:.1f}% | calibrated "
          f"{boot['T_joint_Tu']:.3f} s -> {100*boot['cov_joint_tustin']:.1f}% | "
          f"Wilson {boot['T_joint_Tu_cons']:.3f} s -> "
          f"{100*boot['cov_joint_tustin_cons']:.1f}%")
    print(f"process loop      : {proc.window.diagnosis()}")
    print(f"converter loop    : {powr.window.diagnosis()}")


if __name__ == "__main__":
    main()
