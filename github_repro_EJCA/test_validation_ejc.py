#!/usr/bin/env python3
"""Regression tests for the sampling-period design-window package.

Two layers:

* property tests on ``ejc_window`` that must hold for any input -- exactness
  of the pulse-transfer model, branch consistency of the phase, agreement
  between the analytical bounds and the exact sampled-data loop, and the
  algebra of the feasible window;
* value tests that lock the headline numbers quoted in the manuscript, read
  back from the generated JSON so that code, data and text cannot drift
  apart silently.

Run with ``pytest -q`` (or ``python test_validation_ejc.py`` for a summary).
The value tests need ``python validation_ejc.py`` to have been run first.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ejc_window import (
    FOPTD, SOPTD, Budgets, Timing, IMMEDIATE, NEXT_SCAN,
    simc_pi, simc_pid_cancelling,
    zoh_foptd_frequency, digital_pi_frequency, zoh_soptd_frequency,
    margin_from_callable, foptd_pi_margin, soptd_pid_margin,
    soptd_pid_continuous_margin,
    foptd_pi_closed_loop_roots, foptd_pi_is_stable, foptd_pi_stability_limit,
    sampling_zero_exact, sampling_zero_first_order, T_min_sampling_zero,
    cost_root_update, cost_root_noise, T_cost_update, T_cost_noise,
    build_window, window_is_feasible, continuous_pm_deg, surrogate_pm_deg,
)

OUT = Path(__file__).resolve().parent
RUN = FOPTD(K=1.0, tau=10.0, theta=2.0)
RUN_LAM = 2.0


# ==========================================================================
# Exactness of the sampled-data models
# ==========================================================================
def _integrate_foptd_ode(K, tau, theta, T, u, n, sub=4000):
    """Reference solution of the FOPTD ODE with a ZOH input, by fine stepping.

    dx/dt = -x/tau + (K/tau) u(t - theta), u held constant on each period.
    Integrated with an exact exponential step on a sub-grid, so the only
    approximation is the resolution of the delayed input edge.
    """
    h = T / sub
    steps = n * sub
    ah = math.exp(-h / tau)
    bh = K * (1.0 - ah)
    x = 0.0
    out = np.zeros(n)
    for k in range(steps):
        if k % sub == 0:
            out[k // sub] = x
        t_delayed = k * h - theta
        if t_delayed < 0.0:
            ud = 0.0
        else:
            j = int(math.floor(t_delayed / T + 1e-12))
            ud = u[j] if j < n else 0.0
        x = ah * x + bh * ud
    return out


@pytest.mark.parametrize("theta,T", [(2.0, 0.7), (2.0, 2.0), (1.3, 0.4),
                                     (0.0, 0.5), (0.25, 1.0)])
def test_exact_update_matches_the_continuous_ode(theta, T):
    """The b0/b1/m/delta recursion must reproduce the ODE at sample instants.

    This is the claim that Eq. (8) is exact rather than a modified-z
    approximation, so it is tested against the differential equation itself.
    """
    K, tau, n = 1.7, 6.0, 25
    rng = np.random.default_rng(3)
    u = rng.normal(size=n)

    m = int(math.floor(theta / T + 1e-13))
    delta = theta - m * T
    a = math.exp(-T / tau)
    b0 = K * (1.0 - math.exp(-(T - delta) / tau))
    b1 = K * (math.exp(-(T - delta) / tau) - a)
    x = 0.0
    rec = np.zeros(n)
    for k in range(n):
        rec[k] = x
        um = u[k - m] if k - m >= 0 else 0.0
        um1 = u[k - m - 1] if k - m - 1 >= 0 else 0.0
        x = a * x + b0 * um + b1 * um1

    ref = _integrate_foptd_ode(K, tau, theta, T, u, n)
    assert np.max(np.abs(rec - ref)) < 2e-3


def test_frequency_response_is_the_transform_of_the_recursion():
    """zoh_foptd_frequency must equal the recursion's transfer function."""
    K, tau, theta, T = 1.7, 6.0, 1.3, 0.4
    m = int(math.floor(theta / T + 1e-13))
    delta = theta - m * T
    a = math.exp(-T / tau)
    b0 = K * (1.0 - math.exp(-(T - delta) / tau))
    b1 = K * (math.exp(-(T - delta) / tau) - a)
    Om = np.array([0.02, 0.4, 1.1, 2.0, 3.0])
    q = np.exp(-1j * Om)
    expected = q ** (m + 1) * (b0 + b1 * q) / (1.0 - a * q)
    got = zoh_foptd_frequency(Om, K=K, tau=tau, theta=theta, T=T)
    assert np.allclose(got, expected, rtol=0, atol=1e-14)


def test_delay_split_is_consistent_across_the_boundary():
    """theta = m*T exactly and theta = m*T - eps must agree in the limit."""
    T, tau = 0.5, 10.0
    Om = np.linspace(0.01, 3.0, 50)
    a = zoh_foptd_frequency(Om, K=1.0, tau=tau, theta=2.0, T=T)
    b = zoh_foptd_frequency(Om, K=1.0, tau=tau, theta=2.0 - 1e-10, T=T)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_soptd_frequency_matches_scalar_and_vector_calls():
    kw = dict(K=1.0, zeta=0.3, omega_n=2.0, theta=0.35, T=0.2)
    Om = np.array([0.1, 0.7, 2.0])
    vec = zoh_soptd_frequency(Om, **kw)
    for i, w in enumerate(Om):
        assert abs(zoh_soptd_frequency(float(w), **kw) - vec[i]) < 1e-12


def test_soptd_zero_delay_dc_gain():
    """At omega = 0 the ZOH-sampled plant must have the plant's DC gain."""
    H = zoh_soptd_frequency(1e-7, K=2.5, zeta=0.5, omega_n=3.0, theta=0.0,
                            T=0.05)
    assert abs(H.real - 2.5) < 1e-4
    assert abs(H.imag) < 1e-3


# ==========================================================================
# Phase-margin evaluation
# ==========================================================================
def test_margin_is_branch_consistent_for_a_wrapped_loop():
    """A loop wrapped past -180 deg must report a negative margin.

    Naive use of ``angle()`` would alias the true phase of -376 deg back to
    -16 deg and report a comfortable +164 deg margin for a loop that is in
    fact far past the stability boundary.
    """
    def Lfun(Om):
        w = np.asarray(Om, dtype=float)
        return (1.0 / (0.5 + w)) * np.exp(-1j * (0.5 * np.pi + 10.0 * w))

    r = margin_from_callable(Lfun, T=1.0, n_grid=20000)
    assert r.n_crossings == 1
    # |L| = 1 at w = 0.5; true phase there is -(pi/2 + 5) rad = -376.5 deg.
    assert r.omega_crossings[0] == pytest.approx(0.5, rel=1e-6)
    assert r.pm_deg == pytest.approx(180.0 - math.degrees(0.5 * math.pi + 5.0),
                                     abs=1e-3)
    assert r.pm_deg < -100.0


def test_no_crossing_regimes_are_distinguished():
    """|L|<1 everywhere and |L|>1 everywhere must not be conflated."""
    below = margin_from_callable(lambda Om: 0.1 * np.exp(-1j * np.asarray(Om)),
                                 T=1.0, n_grid=500)
    above = margin_from_callable(lambda Om: 9.0 * np.exp(-1j * np.asarray(Om)),
                                 T=1.0, n_grid=500)
    assert below.regime == "gain_below" and math.isnan(below.pm_deg)
    assert above.regime == "gain_above" and math.isnan(above.pm_deg)
    # A degenerate loop must never be counted as meeting a margin requirement.
    assert not (below.pm_deg >= 45.0)
    assert not (above.pm_deg >= 45.0)


def test_loop_beyond_the_stability_limit_has_no_crossover():
    """Past T_stab the FOPTD/PI loop gain exceeds one across the whole band."""
    ctrl = simc_pi(RUN, RUN_LAM)
    T_stab = foptd_pi_stability_limit(model=RUN, lam=RUN_LAM,
                                      method="backward_euler")
    below = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=0.9 * T_stab,
                            method="backward_euler", n_grid=12000)
    assert below.pm_deg > 0.0 and below.regime == "crossings"
    above = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=1.1 * T_stab,
                            method="backward_euler", n_grid=12000)
    assert math.isnan(above.pm_deg)
    assert not (above.pm_deg >= 0.0)


def test_margin_sign_agrees_with_the_characteristic_roots():
    """Positive phase margin and root-based stability must never disagree."""
    ctrl = simc_pi(RUN, RUN_LAM)
    for T in (0.2, 1.0, 3.0, 6.0, 8.0, 9.0, 10.5, 12.0):
        pm = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=T,
                             method="backward_euler", n_grid=8000).pm_deg
        stable = foptd_pi_is_stable(plant=RUN, ctrl=ctrl, T=T,
                                    method="backward_euler", timing=IMMEDIATE)
        if abs(pm) > 0.5:            # skip the immediate neighbourhood of 0
            assert (pm > 0.0) == stable, f"T={T}: pm={pm}, stable={stable}"


def test_all_crossings_are_found_and_worst_is_reported():
    """A hand-built loop with three crossings must report the worst margin."""
    # |L| dips below 1 and comes back twice; phase is a plain delay.
    def Lfun(Om):
        w = np.asarray(Om, dtype=float)
        mag = 1.6 + 1.2 * np.cos(9.0 * w)      # oscillating magnitude
        return mag * np.exp(-1j * (0.5 * np.pi + 2.2 * w))

    r = margin_from_callable(Lfun, T=1.0, n_grid=20000)
    assert r.n_crossings >= 3
    assert r.pm_deg == pytest.approx(min(r.crossings_deg))


def test_brent_refinement_beats_the_grid():
    """The refined crossover must satisfy |L| = 1 to near machine precision."""
    ctrl = simc_pi(RUN, RUN_LAM)

    def Lfun(Om):
        return (zoh_foptd_frequency(Om, K=RUN.K, tau=RUN.tau, theta=RUN.theta,
                                    T=0.8)
                * digital_pi_frequency(Om, Kc=ctrl.Kc, tau_i=ctrl.tau_i,
                                       T=0.8, method="tustin"))

    r = margin_from_callable(Lfun, T=0.8, n_grid=400)   # deliberately coarse
    Oc = r.omega_crossings[0] * 0.8
    assert abs(abs(Lfun(Oc)) - 1.0) < 1e-10


def test_next_scan_costs_exactly_one_sample_of_phase():
    """The extra z^-1 must show up as omega_c*T more phase lag."""
    ctrl = simc_pi(RUN, RUN_LAM)
    T = 0.6
    imm = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=T, method="tustin",
                          timing=IMMEDIATE, n_grid=9000)
    nxt = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=T, method="tustin",
                          timing=Timing(kappa=1.5, extra_sample_delays=1),
                          n_grid=9000)
    # Same magnitude, so the same crossover; the loss is exactly omega_c*T.
    assert nxt.omega_c == pytest.approx(imm.omega_c, rel=1e-9)
    assert (imm.pm_deg - nxt.pm_deg) == pytest.approx(
        math.degrees(imm.omega_c * T), rel=1e-6)


# ==========================================================================
# The analytical bounds against the exact loop
# ==========================================================================
@pytest.mark.parametrize("ratio", [0.15, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("method", ["backward_euler", "tustin"])
def test_analytical_margin_bound_is_not_optimistic_by_much(ratio, method):
    """At the analytical 10 deg period, the exact shortfall stays under 1 deg."""
    tau = 10.0
    theta = ratio * tau
    model = FOPTD(K=1.0, tau=tau, theta=theta)
    lam = theta
    ctrl = simc_pi(model, lam)
    T = math.radians(10.0) * (lam + theta) / IMMEDIATE.kappa
    target = continuous_pm_deg(theta, theta, lam) - 10.0
    exact = foptd_pi_margin(plant=model, ctrl=ctrl, T=T, method=method,
                            n_grid=9000).pm_deg
    assert exact - target > -1.0


def test_surrogate_matches_exact_as_T_goes_to_zero():
    """Both models must agree in the continuous limit."""
    ctrl = simc_pi(RUN, RUN_LAM)
    for T in (0.05, 0.02, 0.01):
        exact = foptd_pi_margin(plant=RUN, ctrl=ctrl, T=T, method="tustin",
                                n_grid=9000).pm_deg
        sur = surrogate_pm_deg(RUN.theta, RUN.theta, RUN_LAM, T,
                               IMMEDIATE.kappa)
        assert abs(exact - sur) < 0.1


def test_soptd_continuous_reference_matches_nominal_when_cancelling():
    """With exact cancellation the loop is an integrator plus delay."""
    plant = SOPTD(K=1.0, zeta=0.4, omega_n=1.0, theta=0.5)
    lam = 2.0
    ctrl = simc_pid_cancelling(plant, lam)
    r = soptd_pid_continuous_margin(plant=plant, ctrl=ctrl)
    assert r.omega_c == pytest.approx(1.0 / (lam + plant.theta), rel=1e-4)
    assert r.pm_deg == pytest.approx(
        continuous_pm_deg(plant.theta, plant.theta, lam), abs=0.05)


# ==========================================================================
# Stability limit
# ==========================================================================
def test_characteristic_polynomial_has_no_root_at_the_origin():
    ctrl = simc_pi(RUN, RUN_LAM)
    roots = foptd_pi_closed_loop_roots(plant=RUN, ctrl=ctrl, T=1.0,
                                       method="tustin")
    assert np.min(np.abs(roots)) > 1e-9


def test_stability_limit_brackets_the_transition():
    for method in ("backward_euler", "tustin"):
        T = foptd_pi_stability_limit(model=RUN, lam=RUN_LAM, method=method)
        ctrl = simc_pi(RUN, RUN_LAM)
        assert foptd_pi_is_stable(plant=RUN, ctrl=ctrl, T=T * 0.999,
                                  method=method, timing=IMMEDIATE)
        assert not foptd_pi_is_stable(plant=RUN, ctrl=ctrl, T=T * 1.001,
                                      method=method, timing=IMMEDIATE)


def test_margin_bound_lies_well_inside_the_stability_limit():
    """The design window must sit strictly inside the stability region."""
    T_pm = math.radians(10.0) * (RUN_LAM + RUN.theta) / IMMEDIATE.kappa
    for method in ("backward_euler", "tustin"):
        T_stab = foptd_pi_stability_limit(model=RUN, lam=RUN_LAM,
                                          method=method)
        assert T_stab > 4.0 * T_pm


# ==========================================================================
# Sampling zeros
# ==========================================================================
def test_sampling_zero_expansion_is_first_order_accurate():
    for zeta in (0.2, 0.4, 0.7):
        errs = []
        for T in (0.2, 0.1, 0.05):
            e = abs(sampling_zero_exact(zeta, 1.0, T)
                    - sampling_zero_first_order(zeta, 1.0, T))
            errs.append(e)
        # Halving T must cut the error by at least ~3x for an O(T^2) residual.
        assert errs[0] / errs[1] > 3.0
        assert errs[1] / errs[2] > 3.0


def test_sampling_zero_tends_to_minus_one():
    for zeta in (0.1, 0.5, 0.9):
        for T in (1e-4, 1e-6, 1e-9):
            assert sampling_zero_exact(zeta, 1.0, T) == pytest.approx(
                -1.0, abs=1e-3)


def _z0_direct(zeta, wn, T):
    """Textbook numerator formulas, valid where cancellation is tolerable."""
    a = zeta * wn
    E = math.exp(-a * T)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    phi = wd * T
    b1 = 1.0 - E * (math.cos(phi) + (a / wd) * math.sin(phi))
    b0 = E * E - E * (math.cos(phi) - (a / wd) * math.sin(phi))
    return -b0 / b1


def test_sampling_zero_series_branch_matches_the_direct_formula():
    """Just above the cutoff both forms are valid and must agree."""
    from ejc_window import _ZERO_SERIES_CUTOFF as cut
    for zeta in (0.1, 0.4, 0.9):
        for wn in (1.0, 4525.0):
            for mult in (1.0, 3.0, 10.0):
                T = mult * cut / wn
                assert sampling_zero_exact(zeta, wn, T) == pytest.approx(
                    _z0_direct(zeta, wn, T), abs=1e-9)


def test_sampling_zero_is_continuous_across_the_branch_cutoff():
    """No jump at the switchover: the slope must be the analytic one.

    Across the cutoff the finite difference of z0 must equal the first-order
    slope (2/3) zeta omega_n, which is only true if the two branches join.
    """
    from ejc_window import _ZERO_SERIES_CUTOFF as cut
    for zeta in (0.1, 0.4, 0.9):
        for wn in (1.0, 4525.0):
            T_lo, T_hi = 0.98 * cut / wn, 1.02 * cut / wn
            slope = ((sampling_zero_exact(zeta, wn, T_hi)
                      - sampling_zero_exact(zeta, wn, T_lo)) / (T_hi - T_lo))
            assert slope == pytest.approx((2.0 / 3.0) * zeta * wn, rel=1e-3)


def test_sampling_zero_is_monotone_and_bounded():
    """|z0| must decrease from 1 as T grows, for every damping ratio."""
    for zeta in (0.2, 0.5, 0.8):
        vals = [abs(sampling_zero_exact(zeta, 1.0, T))
                for T in (1e-5, 1e-3, 0.01, 0.1, 0.3, 0.6)]
        assert all(v <= 1.0 + 1e-12 for v in vals)
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_T_min_sampling_zero_hits_the_tolerance():
    for tol in (0.85, 0.9, 0.95):
        T = T_min_sampling_zero(0.4, 1.0, tol)
        assert abs(sampling_zero_exact(0.4, 1.0, T)) == pytest.approx(tol,
                                                                     abs=1e-8)


# ==========================================================================
# Cost roots and projection
# ==========================================================================
@pytest.mark.parametrize("rho", [1e-4, 0.005, 0.02, 1.0, 50.0])
def test_cost_roots_solve_their_equations(rho):
    y = cost_root_update(rho)
    assert y * y * (1 + y) == pytest.approx(rho, rel=1e-10)
    y = cost_root_noise(rho)
    assert y**3 * (1 + y) == pytest.approx(rho, rel=1e-10)


def test_cost_root_asymptotics():
    # y ~ sqrt(rho) for rho << 1 and y ~ cbrt(rho) for rho >> 1; the
    # tolerances reflect the size of the neglected correction, not precision.
    assert cost_root_update(1e-8) == pytest.approx(1e-4, rel=1e-3)
    assert cost_root_update(1e6) == pytest.approx(1e2, rel=5e-3)


def test_running_example_cost_matches_the_manuscript():
    assert T_cost_update(theta_0=2.0, kappa=0.5, rho_u=0.005) == \
        pytest.approx(0.274, abs=5e-4)
    assert T_cost_noise(theta_0=0.5, kappa=0.5, rho_d=0.01 * 0.25) == \
        pytest.approx(0.13, abs=5e-3)


# ==========================================================================
# Window assembly
# ==========================================================================
def test_window_feasibility_matches_the_proposition():
    assert window_is_feasible([1.0, 2.0], [3.0, 5.0])
    assert window_is_feasible([1.0], [1.0])
    assert not window_is_feasible([1.0, 4.0], [3.0, 5.0])


def test_process_case_window_is_two_sided_and_correctly_attributed():
    model = FOPTD(K=1.8, tau=180.0, theta=40.0)
    timing = Timing(kappa=1.5, T_c0=2.15, extra_sample_delays=1)
    w = build_window(model=model, lam=40.0, timing=timing,
                     budgets=Budgets(dphi_deg=10.0, eps_iae=0.25,
                                     mode_fraction=None, bits=12,
                                     eps_coef=0.02,
                                     T_hardware=(0.1, 0.2, 0.5, 1.0, 2.0,
                                                 5.0, 10.0)),
                     rho_u=0.02)
    assert w.feasible
    assert w.active_upper == "load disturbance"
    assert w.active_lower == "coefficient resolution"
    assert w.T_hardware_feasible == (2.0, 5.0)
    assert w.T_hardware_selected == 2.0
    # The endpoints must satisfy the closed forms they came from.
    assert w.T_min == pytest.approx(180.0 * 2.0 ** -13 / 0.02)
    assert w.T_max == pytest.approx(0.25 * (40.0 + 2.15) / 1.5)


def test_converter_case_window_is_bounded_below_by_derivative_noise():
    wn = 1.0 / math.sqrt(220e-6 * 220e-6)
    model = SOPTD(K=1.0, zeta=0.20, omega_n=wn, theta=20e-6)
    timing = Timing(kappa=1.5, T_c0=20e-6, extra_sample_delays=1)
    ctrl = simc_pid_cancelling(model, 1.0e-3)
    w = build_window(model=model, lam=1.0e-3, timing=timing,
                     budgets=Budgets(dphi_deg=10.0, eps_iae=None,
                                     mode_fraction=0.25, bits=16,
                                     eps_coef=0.02, sigma_n=5e-3,
                                     V_umax=1e-4),
                     Kc=ctrl.Kc, tau_d=ctrl.tau_d)
    assert w.feasible
    assert w.active_upper == "phase margin"
    assert w.active_lower == "derivative noise"
    assert w.upper("mode resolution") > w.upper("phase margin")
    assert w.lower("coefficient resolution") < w.lower("derivative noise")


def test_window_closes_as_precision_falls_and_only_then():
    model = FOPTD(K=1.8, tau=180.0, theta=40.0)
    timing = Timing(kappa=1.5, T_c0=2.15, extra_sample_delays=1)
    feas = {}
    for bits in range(6, 21):
        w = build_window(model=model, lam=40.0, timing=timing,
                         budgets=Budgets(dphi_deg=10.0, eps_iae=0.25,
                                         mode_fraction=None, bits=bits,
                                         eps_coef=0.02), rho_u=0.02)
        feas[bits] = w.feasible
    # Feasibility must be monotone in precision: once open, it stays open.
    opened = [b for b in sorted(feas) if feas[b]]
    assert opened == list(range(min(opened), 21))
    assert not feas[min(opened) - 1]


def test_projection_clamps_to_the_binding_endpoint():
    model = FOPTD(K=1.0, tau=10.0, theta=2.0)
    # Tiny update weight -> cost candidate below T_min -> clamp to T_min.
    w = build_window(model=model, lam=2.0, timing=IMMEDIATE,
                     budgets=Budgets(dphi_deg=10.0, eps_iae=0.25,
                                     mode_fraction=None, bits=8,
                                     eps_coef=0.02), rho_u=1e-9)
    assert w.feasible and w.T_sel == pytest.approx(w.T_min)
    assert "clamped to T_min" in w.diagnosis()
    # Huge update weight -> candidate above T_max -> clamp to T_max.
    w = build_window(model=model, lam=2.0, timing=IMMEDIATE,
                     budgets=Budgets(dphi_deg=10.0, eps_iae=0.25,
                                     mode_fraction=None, bits=16,
                                     eps_coef=0.02), rho_u=1e6)
    assert w.feasible and w.T_sel == pytest.approx(w.T_max)


def test_infeasible_window_reports_both_active_constraints():
    model = FOPTD(K=1.0, tau=1000.0, theta=1.0)
    w = build_window(model=model, lam=1.0, timing=IMMEDIATE,
                     budgets=Budgets(dphi_deg=1.0, eps_iae=0.01,
                                     mode_fraction=None, bits=6,
                                     eps_coef=0.001), rho_u=0.01)
    assert not w.feasible
    d = w.diagnosis()
    assert "infeasible" in d
    assert w.active_lower in d and w.active_upper in d
    assert math.isnan(w.T_sel)


def test_pm_bound_scales_inversely_with_kappa():
    model = FOPTD(K=1.0, tau=10.0, theta=2.0)
    b = Budgets(dphi_deg=10.0, eps_iae=None, mode_fraction=None, bits=None,
                eps_coef=None)
    a = build_window(model=model, lam=2.0, timing=IMMEDIATE, budgets=b)
    c = build_window(model=model, lam=2.0, timing=NEXT_SCAN, budgets=b)
    assert a.T_max / c.T_max == pytest.approx(3.0, rel=1e-12)


# ==========================================================================
# Headline numbers quoted in the manuscript
# ==========================================================================
def _load(name: str):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} missing; run validation_ejc.py first")
    return json.loads(path.read_text())


def test_benchmark_headline_numbers():
    d = _load("benchmark_aggregate.json")
    assert d["overall"]["n_cases"] == 48
    assert d["overall"]["max_crossings"] == 1
    assert d["overall"]["worst_shortfall"] == pytest.approx(-0.80, abs=0.05)
    g = d["groups"]
    assert g["immediate; backward Euler"]["max_abs_error"] == \
        pytest.approx(6.00, abs=0.05)
    assert g["next scan; Tustin"]["max_abs_error"] == pytest.approx(0.12,
                                                                    abs=0.02)


def test_soptd_headline_numbers():
    d = _load("soptd_validation_summary.json")["summary"]
    assert d["n_cases"] == 162 and d["n_distinct"] == 27
    assert d["max_crossings"] == 1
    assert d["max_omega_d_T"] <= math.pi / 4 + 1e-9
    # Every case conservative: measured against the same loop's continuous
    # margin, the surrogate never overstates the achievable margin.
    assert d["worst_shortfall"] == pytest.approx(0.0, abs=1e-9)
    assert d["max_abs_error"] < 1.0


def test_bootstrap_headline_numbers():
    d = _load("bootstrap_summary.json")
    assert d["n_bootstrap"] == 2000
    assert d["n_calibration"] == d["n_validation"] == 1000
    assert d["monotone_calibration_tustin"] and d["monotone_calibration_be"]
    # Ordering of the three uncertainty treatments is the paper's claim.
    assert d["cov_joint_be_at_scalar"] < d["cov_joint_be"] < \
        d["cov_joint_be_cons"]
    assert d["T_joint_BE_cons"] < d["T_joint_BE"] < d["T_theta"]
    assert d["cov_joint_tustin_cons"] >= 0.95
    assert d["cov_joint_be_cons"] >= 0.95
    assert d["cov_surrogate_mean"] < 0.6      # mean-theta design fails badly


def test_application_case_numbers():
    d = _load("application_cases.json")
    p = d["process"]
    assert p["active_upper"] == "load disturbance"
    assert p["active_lower"] == "coefficient resolution"
    assert p["T_used"] == 2.0
    assert p["hardware_feasible"] == [2.0, 5.0]
    assert abs(p["exact_pm"] - p["target_pm"]) < 1.0
    q = d["power_electronics"]
    assert q["active_upper"] == "phase margin"
    assert q["active_lower"] == "derivative noise"
    assert q["T_used"] == pytest.approx(1e-4)
    assert q["omega_d_T"] < math.pi / 4
    assert abs(q["exact_pm"] - q["target_pm"]) < 1.0


def test_supporting_studies():
    d = _load("validation_summary.json")
    misc, run = d["misc"], d["running"]
    assert misc["noise_slope"] == pytest.approx(-2.0, abs=0.05)
    assert misc["iae_ratio_min"] == pytest.approx(1.0, abs=2e-3)
    assert misc["iae_ratio_max"] == pytest.approx(1.0, abs=2e-3)
    assert misc["actuator_tv_slope"] == pytest.approx(-1.0, abs=0.05)
    assert misc["actuator_rev_slope"] == pytest.approx(-1.0, abs=0.05)
    assert misc["zero_rel_error_pct"] < 1.0
    assert run["T_stability_backward_euler"] == pytest.approx(8.7, abs=0.1)
    assert run["T_stability_tustin"] == pytest.approx(11.2, abs=0.1)
    assert run["running_max_crossings"] == 1


def test_generated_macros_cover_every_command_the_manuscript_uses():
    """Every \\newcommand the paper calls must be emitted by the code."""
    macros = OUT / "validation_macros.tex"
    if not macros.exists():
        pytest.skip("validation_macros.tex missing; run validation_ejc.py")
    defined = set()
    for line in macros.read_text().splitlines():
        if line.startswith(r"\newcommand{\\"[:12]):
            pass
    import re
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros.read_text()))
    tex = OUT.parent / "paper" / "main_sc.tex"
    if not tex.exists():
        pytest.skip("manuscript not present next to the package")
    body = tex.read_text()
    # Commands the manuscript defines itself are not the code's job.
    local = set(re.findall(r"\\newcommand\{\\(\w+)\}", body))
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)\b", body))
    generated_prefixes = ("Bench", "Soptd", "Boot", "Run", "Proc", "Pow",
                          "Rob", "Band", "Word", "Zero", "Noise", "Iae",
                          "Act", "Window", "Map")
    referenced = {u for u in used
                  if u.startswith(generated_prefixes) and u not in local}
    missing = referenced - defined
    assert not missing, f"manuscript uses undefined macros: {sorted(missing)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
