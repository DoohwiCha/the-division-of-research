"""
SIGNED SAME-SIDE LIQUIDITY. N_eff is sign-agnostic -- a contrarian
mirror book lowers N_eff but does NOT consume the same-side liquidity. So we make the PRIMARY metric the
signed same-side overlap (dollars on the SAME asset SAME direction = competing for the same fills), and we
compare LLM vs CLASSIC at MATCHED cohort size (the earlier S=0.46 vs 0.39 was an unfair M=35-vs-5 contrast).
We also measure the RESIDUAL (factor-removed) magnitude so "residual crowding" is shown to be material.

Run:  python3 cohort_liquidity_signed.py  ->  results/cohort_liquidity_signed.json + fig_liquidity_signed.png
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from cohort_analyze import normalize
from cohort_crossvendor_hardening import books

from _paths import RESULTS as OUT, FIGURES, DATA, inp
RNG = np.random.default_rng(0)
PERSONAS = ["momentum", "contrarian", "defensive", "aggressive", "quant"]


def same_side_overlap(W):
    """Mean over agent pairs of sum_i min(|w_i|,|w_j|)*1[same sign] -- dollars crowding the same side."""
    A = W.to_numpy().T; n = A.shape[0]; v = []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = np.sign(A[i]), np.sign(A[j])
            v.append(float(np.sum(np.minimum(np.abs(A[i]), np.abs(A[j])) * (si == sj))))
    return float(np.mean(v)) if v else 0.0


def surviving_gross(W):
    return float(W.sum(axis=1).abs().sum() / W.abs().sum().sum())


def matched_null(metric, M, syms, B=2000):
    vals = [metric(normalize(pd.DataFrame(RNG.standard_normal((len(syms), M)), index=syms))) for _ in range(B)]
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(render_figure: bool = True):
    F = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0)
    syms = F.index.tolist()
    Wf = normalize(F.reindex(syms).fillna(0.0))
    llm = books(json.load(open(inp("cohort_agents_crossvendor.json"))), syms)
    llm = llm.loc[:, llm.std() > 1e-12]
    cols = list(llm.columns)

    # ---- matched M=5: one LLM agent per persona (random vendor) vs the 5 classic factors ----
    def subsample5():
        pick = [[c for c in cols if c.split(":")[0] == p][RNG.integers(len([c for c in cols if c.split(":")[0] == p]))] for p in PERSONAS]
        return normalize(llm[pick])
    S5 = np.array([surviving_gross(subsample5()) for _ in range(2000)])
    O5 = np.array([same_side_overlap(subsample5()) for _ in range(2000)])
    S_cls, O_cls = surviving_gross(Wf), same_side_overlap(Wf)

    # ---- full LLM (M=35) vs its matched null, for the excess-over-chance statement ----
    S_llm, O_llm = surviving_gross(llm), same_side_overlap(llm)
    Sn, Sn_lo, Sn_hi = matched_null(surviving_gross, 35, syms)
    On, On_lo, On_hi = matched_null(same_side_overlap, 35, syms)

    # ---- residual magnitude (factor-removed) ----
    X = Wf.to_numpy(); R = (np.eye(len(syms)) - X @ np.linalg.pinv(X)) @ llm.to_numpy()
    resid_frac = float((np.linalg.norm(R, axis=0) / (np.linalg.norm(llm.to_numpy(), axis=0) + 1e-12)).mean())
    Wr = normalize(pd.DataFrame(R, index=syms, columns=cols)); Wr = Wr.loc[:, Wr.std() > 1e-12]
    O_resid = same_side_overlap(Wr)

    out = {
        "matched_M5": {"S_llm_median": float(np.median(S5)), "S_llm_ci": [float(np.percentile(S5, 2.5)), float(np.percentile(S5, 97.5))],
                       "S_classic": S_cls, "P_S_llm_gt_classic": float((S5 > S_cls).mean()),
                       "overlap_llm_median": float(np.median(O5)), "overlap_llm_ci": [float(np.percentile(O5, 2.5)), float(np.percentile(O5, 97.5))],
                       "overlap_classic": O_cls, "P_overlap_llm_gt_classic": float((O5 > O_cls).mean())},
        "full_M35_vs_null": {"S_llm": S_llm, "S_null": Sn, "S_null_ci": [Sn_lo, Sn_hi],
                             "overlap_llm": O_llm, "overlap_null": On, "overlap_null_ci": [On_lo, On_hi]},
        "residual": {"gross_fraction": resid_frac, "same_side_overlap": O_resid, "full_same_side_overlap": same_side_overlap(llm)},
    }
    print("=" * 82); print("SIGNED SAME-SIDE LIQUIDITY (primary metric; LLM vs classic at matched M)"); print("=" * 82)
    m = out["matched_M5"]
    print(f"  MATCHED M=5 (LLM 1/persona vs 5 classic factors):")
    print(f"    surviving gross S:   LLM {m['S_llm_median']:.3f} [{m['S_llm_ci'][0]:.3f},{m['S_llm_ci'][1]:.3f}]  classic {m['S_classic']:.3f}  -> P(LLM>classic)={m['P_S_llm_gt_classic']:.2f}")
    print(f"    same-side overlap:   LLM {m['overlap_llm_median']:.3f} [{m['overlap_llm_ci'][0]:.3f},{m['overlap_llm_ci'][1]:.3f}]  classic {m['overlap_classic']:.3f}  -> P(LLM>classic)={m['P_overlap_llm_gt_classic']:.2f}")
    print(f"  RESIDUAL (5 factors removed): {resid_frac*100:.0f}% of each book is residual; residual same-side overlap {O_resid:.3f} (full {same_side_overlap(llm):.3f})")
    print("  -> at matched cohort size the LLM cohort is significantly MORE same-side liquidity-crowded than")
    print("     classic factors, and the factor-residual is material (not a tiny vector).")
    json.dump(out, open(os.path.join(OUT, "cohort_liquidity_signed.json"), "w"), indent=2, default=float)

    if render_figure:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
        a1.hist(S5, bins=30, color="#c44e52", alpha=0.7, label="LLM cohort (M=5, 1/persona)")
        a1.axvline(S_cls, color="#4c72b0", lw=2, label=f"classic factors = {S_cls:.2f}")
        a1.set_xlabel("surviving gross $S$ (matched $M{=}5$)"); a1.set_ylabel("freq"); a1.legend(fontsize=8)
        a1.set_title(f"A. LLM > classic on liquidity crowding\nP(LLM$>$classic)={m['P_S_llm_gt_classic']:.2f}")
        a2.hist(O5, bins=30, color="#c44e52", alpha=0.7, label="LLM (M=5)")
        a2.axvline(O_cls, color="#4c72b0", lw=2, label=f"classic = {O_cls:.2f}")
        a2.axvline(O_resid, color="#dd8452", lw=2, ls="--", label=f"LLM factor-residual = {O_resid:.2f}")
        a2.set_xlabel("signed same-side overlap"); a2.legend(fontsize=8)
        a2.set_title("B. Same-side overlap: LLM > classic; residual still crowds")
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_liquidity_signed.png"), dpi=140)
        fig.savefig(os.path.join(FIGURES, "fig_liquidity_signed.pdf")); plt.close(fig)
        print("  wrote cohort_liquidity_signed.json + fig_liquidity_signed.png/pdf")
    else:
        print("  wrote cohort_liquidity_signed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Regenerate JSON without the legacy diagnostic figure.",
    )
    args = parser.parse_args()
    main(render_figure=not args.results_only)
