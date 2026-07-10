# Sampling-period design windows: reproducibility package

This repository contains the scripts, generated data files and figures used for
the European Journal of Control manuscript:

**Sampling-period design windows for digital PI/PID control under dead-time uncertainty**

Author: A. Bhattacharyya

## Contents

- `validation_ejc.py`: reproduces the numerical validation, benchmark tables,
  bootstrap study, SOPTD sampled-data check and figures.
- `requirements.txt`: Python dependencies.
- `benchmark_phase_budget.csv`, `benchmark_aggregate.json`: FOPTD phase-budget
  benchmark outputs.
- `bootstrap_parameters.csv`, `bootstrap_summary.json`: residual-bootstrap
  uncertainty outputs.
- `soptd_validation.csv`, `soptd_validation_summary.json`: SOPTD sampled-data
  validation outputs.
- `validation_summary.json`: combined machine-readable summary.
- `fig*.pdf`: manuscript figures generated or carried by the validation set.
- `benchmark_summary_table.tex`, `soptd_validation_table.tex`,
  `validation_macros.tex`: generated LaTeX inputs used by the manuscript.
- `manuscript_review.pdf`, `manuscript_compact_check.pdf`: PDFs matching the
  EJC manuscript build at the time of this package.

## Reproduce

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```bash
python validation_ejc.py
```

The random seed is fixed at `10`. Running the script regenerates the CSV, JSON,
PDF figure and generated LaTeX output files in this repository root.

## Notes

The package is analytical and simulation-based. It does not contain hardware
data or controller-level fixed-point experiments. After archiving this
repository, cite the DOI issued by the archive service.
