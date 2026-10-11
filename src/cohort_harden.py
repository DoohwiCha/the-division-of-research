"""
COHORT VALIDATION — test whether position homogeneity remains after common factor and market-beta
exposures are removed.

Two pre-registered controls (single-snapshot keystone + multi-regime robustness):
  (1) RANDOM-NULL: N_eff / mean|corr| of B random dollar-neutral panels of the SAME size. The LLM
      collapse must sit far BELOW the chance distribution to be real (not a small-sample artifact).
  (2) FACTOR-RESIDUALIZATION: project each agent's position vector OUT of the span of the 5 classic
      factor portfolios (momentum, reversal, low_vol, trend, carry), then recompute N_eff / |corr| on
      the RESIDUALS. Market/beta is already removed by the dollar-neutral construction; this removes
      the standard factor tilts too. If residual homogeneity PERSISTS (still well below the residual
      random-null), the crowding is LLM-specific, not shared factor exposure.

Inputs:  results/cohort_agents.json + cohort_classic_factors.csv  (single snapshot)
         results/cohort_agents_multi.json + cohort_snapshots_multi.json  (4 regimes)
Run:  python3 cohort_harden.py   ->  results/{cohort_residual.json, fig_cohort_residual.png}
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from cohort_analyze import normalize, n_eff

from _paths import RESULTS as OUT, FIGURES, DATA, inp


def _abs_rho(C: np.ndarray) -> float:
    M = C.shape[0]; iu = np.triu_indices(M, 1)
    return float(np.abs(C[iu]).mean())


def neff_absrho(W: pd.DataFrame):
    """N_eff and mean|corr| of a position book (assets x members); drop degenerate members."""
    W = W.loc[:, W.std() > 1e-12]
    C = np.corrcoef(W.to_numpy().T)
    return n_eff(C), _abs_rho(C)


def llm_matrix(agents, syms) -> pd.DataFrame:
    A = pd.DataFrame(0.0, index=syms, columns=[f"{a['persona']}:{a['model']}" for a in agents])
    for a in agents:
        c = f"{a['persona']}:{a['model']}"
        for w in a["weights"]:
            if w["symbol"] in A.index:
                A.loc[w["symbol"], c] = float(w["weight"])
    return normalize(A)


def residualize(W: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Project each column of W out of the span of the factor portfolios F (assets x n_factors)."""
    X = F.to_numpy()
    P = X @ np.linalg.pinv(X)                       # projection onto factor span (handles collinearity)
    R = (np.eye(X.shape[0]) - P) @ W.to_numpy()
    return normalize(pd.DataFrame(R, index=W.index, columns=W.columns))


def random_null(n_members: int, syms, F: pd.DataFrame | None = None, B: int = 3000, seed: int = 0):
    """Null distribution of N_eff / |corr| for B random dollar-neutral panels (optionally residualized
    on F, the fair null for the residual comparison)."""
    rng = np.random.default_rng(seed)
    d = len(syms); neffs, rhos = [], []
    for _ in range(B):
        Wr = normalize(pd.DataFrame(rng.standard_normal((d, n_members)), index=syms))
        if F is not None:
            Wr = residualize(Wr, F)
        ne, ar = neffs_one = neff_absrho(Wr)
        neffs.append(ne); rhos.append(ar)
    neffs, rhos = np.array(neffs), np.array(rhos)
    return {"neff_mean": float(neffs.mean()), "neff_p2_5": float(np.percentile(neffs, 2.5)),
            "neff_p97_5": float(np.percentile(neffs, 97.5)),
            "absrho_mean": float(rhos.mean()), "absrho_p2_5": float(np.percentile(rhos, 2.5)),
            "absrho_p97_5": float(np.percentile(rhos, 97.5))}


def main():
    agents = json.load(open(inp("cohort_agents.json")))
    F = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0)
    syms = F.index.tolist()
    Wl = llm_matrix(agents, syms)
    Wf = normalize(F.reindex(syms).fillna(0.0))
    M = Wl.shape[1]

    ne_raw, ar_raw = neff_absrho(Wl)
    Wl_res = residualize(Wl, Wf)
    ne_res, ar_res = neff_absrho(Wl_res)
    null_raw = random_null(M, syms, F=None)
    null_res = random_null(M, syms, F=Wf)

    print("=" * 80); print("COHORT HARDENING — random-null + factor-residualization (single snapshot)"); print("=" * 80)
    print(f"  LLM panel: M={M} agents over {len(syms)} assets")
    print(f"\n  (1) RANDOM-NULL (raw, dollar-neutral, {M} members):")
    print(f"      random N_eff = {null_raw['neff_mean']:.1f}  [95% {null_raw['neff_p2_5']:.1f}, {null_raw['neff_p97_5']:.1f}]"
          f"   random |corr| = {null_raw['absrho_mean']:.2f}")
    print(f"      LLM    N_eff = {ne_raw:.2f}  |corr| = {ar_raw:.2f}   -> "
          f"{'FAR BELOW null (real collapse)' if ne_raw < null_raw['neff_p2_5'] else 'within null'}")
    print(f"\n  (2) FACTOR-RESIDUALIZED (project out the 5 classic factors; market/beta already out via dollar-neutral):")
    print(f"      residual random-null N_eff = {null_res['neff_mean']:.1f}  [95% {null_res['neff_p2_5']:.1f}, {null_res['neff_p97_5']:.1f}]")
    print(f"      LLM residual        N_eff = {ne_res:.2f}  |corr| = {ar_res:.2f}")
    surv = ne_res < null_res["neff_p2_5"]
    print(f"      -> residual homogeneity {'PERSISTS below the residual null => LLM-SPECIFIC crowding (not just factor exposure)' if surv else 'gone => was common-factor exposure'}")
    print(f"      (raw N_eff {ne_raw:.2f} -> residual {ne_res:.2f}; {100*(ne_res-ne_raw)/max(null_res['neff_mean']-ne_raw,1e-9):.0f}% of the gap to the null is recovered by removing factors)")

    # multi-regime residual robustness
    multi = json.load(open(inp("cohort_agents_multi.json")))
    snaps = json.load(open(inp("cohort_snapshots_multi.json")))
    regime = []
    for s in snaps:
        Wfd = normalize(pd.DataFrame({k: v for k, v in s["classic"].items()}).reindex(syms).fillna(0.0))
        a_d = [a for a in multi if a["date"] == s["date"]]
        Wld = llm_matrix(a_d, syms)
        r0, _ = neff_absrho(Wld)
        r1, ar1 = neff_absrho(residualize(Wld, Wfd))
        regime.append({"date": s["date"], "neff_raw": r0, "neff_resid": r1, "absrho_resid": ar1})
    nullr_multi = random_null(15, syms, F=Wf)["neff_mean"]
    print(f"\n  MULTI-REGIME residual N_eff (residual random-null ~{nullr_multi:.1f}):")
    for r in regime:
        print(f"    {r['date']}: raw {r['neff_raw']:.2f} -> residual {r['neff_resid']:.2f}  (|corr|_resid {r['absrho_resid']:.2f})")
    mean_resid = float(np.mean([r["neff_resid"] for r in regime]))
    print(f"    mean residual N_eff over regimes = {mean_resid:.2f}  vs residual null ~{nullr_multi:.1f}")

    out = {"snapshot": {"M": M, "n_assets": len(syms),
                        "neff_raw": ne_raw, "neff_resid": ne_res, "absrho_raw": ar_raw, "absrho_resid": ar_res,
                        "null_raw": null_raw, "null_resid": null_res, "residual_homogeneity_persists": bool(surv)},
           "multi_regime": regime, "multi_residual_null_neff": nullr_multi,
           "multi_mean_resid_neff": mean_resid}
    json.dump(out, open(os.path.join(OUT, "cohort_residual.json"), "w"), indent=2, default=float)

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # panel 1: N_eff bars (LLM raw, LLM residual) vs null bands
    xs = [0, 1]
    a1.bar(xs, [ne_raw, ne_res], 0.5, color=["#c44e52", "#dd8452"], zorder=3,
           label="LLM agents (15)")
    a1.errorbar([0], [null_raw["neff_mean"]], yerr=[[null_raw["neff_mean"] - null_raw["neff_p2_5"]],
                [null_raw["neff_p97_5"] - null_raw["neff_mean"]]], fmt="o", color="k", capsize=6, zorder=5,
                label="random-null 95%")
    a1.errorbar([1], [null_res["neff_mean"]], yerr=[[null_res["neff_mean"] - null_res["neff_p2_5"]],
                [null_res["neff_p97_5"] - null_res["neff_mean"]]], fmt="o", color="k", capsize=6, zorder=5)
    a1.set_xticks(xs); a1.set_xticklabels(["raw\npositions", "factor-residualized\n(classic 5 removed)"])
    a1.set_ylabel(r"effective dimensions $N_{eff}$")
    a1.set_title("LLM homogeneity survives factor removal\n(bars = LLM, dots = random-null)")
    a1.legend(fontsize=8); a1.axhline(0, color="k", lw=0.5)
    # panel 2: multi-regime residual N_eff vs null
    dates = [r["date"] for r in regime]
    x = np.arange(len(dates))
    a2.bar(x - 0.2, [r["neff_raw"] for r in regime], 0.4, color="#c44e52", label="raw")
    a2.bar(x + 0.2, [r["neff_resid"] for r in regime], 0.4, color="#dd8452", label="factor-residualized")
    a2.axhline(nullr_multi, color="k", ls="--", lw=1.2, label=f"residual random-null ~{nullr_multi:.1f}")
    a2.set_xticks(x); a2.set_xticklabels(dates, rotation=20, fontsize=8)
    a2.set_ylabel(r"LLM $N_{eff}$"); a2.set_title("Residual homogeneity across regimes"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_cohort_residual.png"), dpi=130); plt.close(fig)
    print(f"\n  wrote cohort_residual.json + fig_cohort_residual.png")


if __name__ == "__main__":
    main()
