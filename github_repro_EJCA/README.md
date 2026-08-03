# Two-sided sampling-period design windows: reproducibility package

Scripts, generated data and figures for the European Journal of Control
manuscript

> **Two-sided sampling-period design windows for digital PI/PID control under
> dead-time and identification uncertainty**, A. Bhattacharyya

Everything in the paper that is a number, a figure or a generated table comes
from `validation_ejc.py`. Nothing is transcribed by hand.

## Contents

| File | Role |
| --- | --- |
| `ejc_window.py` | Core numerics: plant models, exact sampled-data frequency responses, hardened phase-margin evaluation, the individual bounds and the feasible-window assembly. Side-effect free and importable. |
| `validation_ejc.py` | Driver: runs every study, writes every figure, CSV, JSON and generated LaTeX input. |
| `test_validation_ejc.py` | Regression suite: property tests on the numerics plus value tests that lock the headline numbers. |
| `requirements.txt` | Minimum dependency versions. |
| `fig01`–`fig12` (PDF) | The manuscript figures, in document order. |
| `benchmark_phase_budget.csv`, `benchmark_aggregate.json` | FOPTD phase-budget benchmark. |
| `soptd_validation.csv`, `soptd_validation_summary.json` | SOPTD sampled-data benchmark. |
| `bootstrap_parameters.csv`, `bootstrap_summary.json` | Split-bootstrap uncertainty study. |
| `application_cases.json` | The two worked engineering case studies. |
| `validation_summary.json` | Combined machine-readable summary of every study. |
| `validation_macros.tex`, `benchmark_summary_table.tex`, `soptd_validation_table.tex`, `application_table.tex` | Generated LaTeX inputs. **Do not edit by hand.** |

The LaTeX sources of the manuscript live in `../paper`.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python validation_ejc.py        # ~80 s; regenerates every output file
python -m pytest test_validation_ejc.py -q
```

The run is deterministic: every seed derives from the master seed `10`, and
repeated runs reproduce all 167 numeric summary fields bit for bit.

To rebuild the manuscript:

```bash
cd ../paper
cp ../github_repro_EJCA/fig*.pdf ../github_repro_EJCA/*_table.tex \
   ../github_repro_EJCA/validation_macros.tex .
pdflatex main_sc && bibtex main_sc && pdflatex main_sc && pdflatex main_sc
```

## What the code computes

**Design window.** `build_window` evaluates every applicable constraint from
Table 3 of the paper, records which one attains each endpoint, and returns a
diagnosis. An empty window is not an exception: it is returned with both
active constraints named, since those are the only two whose relaxation can
open it (Proposition 4).

**Exact sampled-data models.** FOPTD loops use the arbitrary-dead-time pulse
transfer model `G_T(z) = z^-(m+1) (b0 + b1 z^-1)/(1 - a z^-1)`; SOPTD loops
use a ZOH state-space model with fractional delay. Both are exact at the
sampling instants, and `test_exact_update_matches_the_continuous_ode` checks
the FOPTD recursion against the differential equation itself.

**Phase margins.** `margin_from_callable` locates *every* downward unity-gain
crossing in the Nyquist band, refines each with Brent's method, and evaluates
the phase on the unwrapped branch, so a loop that has wrapped past -180 deg is
reported as a negative margin rather than aliased back into the stable range.
The worst margin over all crossings is returned together with the crossing
multiplicity. When no crossing exists, `regime` distinguishes `gain_below`
(benign) from `gain_above` (no margin at all); `pm_deg` is `nan` in both, so
comparisons against a requirement fail conservatively.

**Stability limits** come from the roots of the exact characteristic
polynomial in `q = z^-1` (stable iff every `|q| > 1`), not from a frequency
grid.

**Sampling zeros.** Both numerator coefficients are `O((omega_n T)^2)`
differences of order-one terms, so the direct formulas lose roughly
`2 log10(1/(omega_n T))` digits and underflow for fast sampling. Below
`omega_n T = 1e-3` the code divides out the common factor analytically and
uses a cancellation-free expansion instead; the two branches agree to about
`1e-9` where both are valid.

**Uncertainty.** The bootstrap is centred and split into disjoint calibration
and validation halves *before* any period is chosen. Three treatments are
reported: the scalar dead-time quantile, empirical joint calibration, and
joint calibration against the lower Wilson bound. The last is the recommended
one — the empirical criterion is optimistic by construction, because the same
finite sample both selects and scores the period. Monotonicity of the
acceptance probability in `T` is checked before the bisection and recorded in
`bootstrap_summary.json`.

## Scope

Analytical and simulation-based. There is no hardware data and no
controller-level fixed-point experiment. The two case studies in
`application_cases.json` are design calculations on published component
values, not measurements. The window is a performance and implementation
rule, not a sampled-data stability certificate.

After archiving this repository, cite the DOI issued by the archive service.
