"""Shared path helpers for the normalized research repository layout.

Scripts were originally written with a single `realdata_out/` directory. In this repo, inputs live in
data/, machine-readable outputs in data/results/, and generated plots in data/figures/. To keep the
original scripts working with minimal edits, DATA/RESULTS/FIGURES resolve relative to the repo root, and the
legacy `OUT` variable points at data/results/ (most scripts both read cached inputs and write outputs there).
Input files that live in data/ are resolved by name via `inp()`.
"""
import os

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
ROOT = _ROOT
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(DATA, "results")
FIGURES = os.path.join(DATA, "figures")
for _d in (RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

# input files that were moved to data/ (everything else read/written under data/results/)
_DATA_NAMES = {"market_snapshot.json", "cohort_classic_factors.csv", "cohort_snapshots_multi.json",
               "equity_snapshots_multi.json", "cohort_prices.csv"}


def inp(name):
    """Resolve an input file: prefer data/, fall back to data/results/, else data/."""
    if name in _DATA_NAMES or name.startswith("cohort_agents") or name.startswith("equity_classic_factors"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            return p
    p = os.path.join(RESULTS, name)
    return p if os.path.exists(p) else os.path.join(DATA, name)
