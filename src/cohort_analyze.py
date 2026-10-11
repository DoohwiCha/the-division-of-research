"""
ENHANCEMENT keystone — analyze the LLM-agent homogeneity panel: do diverse LLM agents (varied
model x persona) take MORE correlated positions than genuinely-diverse classic factor strategies?
If so, the LLM cohort has a larger effective size kappa -> smaller per-fund capacity (A*_f = A*_single/kappa).

Inputs:  results/cohort_agents.json  (from the llm-cohort-homogeneity workflow)
         results/cohort_classic_factors.csv  (the fair classic baseline)
Run:  python3 cohort_analyze.py
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

from _paths import RESULTS as OUT, FIGURES, DATA, inp


def normalize(W: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral, gross-1 per column (each agent/strategy is a directional bet vector)."""
    W = W.sub(W.mean(axis=0), axis=1)                       # dollar-neutral
    g = W.abs().sum(axis=0).replace(0, np.nan)
    return (W / g).fillna(0.0)


def n_eff(C: np.ndarray) -> float:
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 1e-10]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def mean_pair_corr(C: np.ndarray) -> float:
    M = C.shape[0]
    iu = np.triu_indices(M, 1)
    return float(C[iu].mean())


def panel_stats(W: pd.DataFrame, label: str) -> dict:
    """W: assets x members. Returns correlation matrix + diversity stats."""
    C = np.corrcoef(W.to_numpy().T)                         # member x member
    M = C.shape[0]
    rho = mean_pair_corr(C)
    iu = np.triu_indices(M, 1)
    abs_rho = float(np.abs(C[iu]).mean())                  # axis concentration (sign-agnostic)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    ne = n_eff(C)
    return {"label": label, "M": M, "C": C, "rho_bar": rho, "abs_rho": abs_rho,
            "eig1_frac": float(lam[0] / lam.sum()), "N_eff": ne,
            "N_eff_frac": ne / M, "kappa": 1 + (M - 1) * rho}


def main():
    agents = json.load(open(inp("cohort_agents.json")))
    syms = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0).index.tolist()
    # LLM agent weight matrix: assets x agents
    A = pd.DataFrame(0.0, index=syms, columns=[f"{a['persona']}:{a['model']}" for a in agents])
    meta = []
    for a in agents:
        col = f"{a['persona']}:{a['model']}"
        for w in a["weights"]:
            if w["symbol"] in A.index:
                A.loc[w["symbol"], col] = float(w["weight"])
        meta.append({"col": col, "persona": a["persona"], "model": a["model"]})
    md = pd.DataFrame(meta)
    Wl = normalize(A)
    F = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0)
    Wf = normalize(F.reindex(syms))

    llm = panel_stats(Wl, "LLM agents")
    cls = panel_stats(Wf, "classic factors")

    print("=" * 76); print("LLM-AGENT HOMOGENEITY — position cross-correlation"); print("=" * 76)
    print(f"  LLM panel   (M={llm['M']}): signed rho={llm['rho_bar']:+.2f}  |rho|={llm['abs_rho']:.2f}  "
          f"top-eig {llm['eig1_frac']:.0%} of variance  N_eff={llm['N_eff']:.2f} ({llm['N_eff_frac']:.0%} of M)")
    print(f"  classic fac (M={cls['M']}): signed rho={cls['rho_bar']:+.2f}  |rho|={cls['abs_rho']:.2f}  "
          f"top-eig {cls['eig1_frac']:.0%} of variance  N_eff={cls['N_eff']:.2f} ({cls['N_eff_frac']:.0%} of M)")
    print(f"  -> 15 LLM agents collapse to ~{llm['N_eff']:.1f} effective positions ({llm['eig1_frac']:.0%} of "
          f"variance on ONE axis); 5 classic factors span {cls['N_eff']:.1f}.")
    print(f"     LLM |corr|={llm['abs_rho']:.2f} vs classic {cls['abs_rho']:.2f}: agents trade the SAME NAMES; "
          f"signed rho lower ({llm['rho_bar']:+.2f}) only because contrarians are the mirror image.")

    # decomposition: model-induced vs persona-induced homogeneity
    C = pd.DataFrame(llm["C"], index=md["col"], columns=md["col"])
    same_persona, same_model, diff_both = [], [], []
    cols = md["col"].tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pi, mi = md.iloc[i]["persona"], md.iloc[i]["model"]
            pj, mj = md.iloc[j]["persona"], md.iloc[j]["model"]
            v = C.iloc[i, j]
            if pi == pj: same_persona.append(v)
            elif mi == mj: same_model.append(v)
            else: diff_both.append(v)
    print(f"\n  decomposition of LLM pairwise correlation:")
    print(f"    same persona, diff model : {np.mean(same_persona):.3f}  (model-convergence)")
    print(f"    same model, diff persona : {np.mean(same_model):.3f}  (persona barely diversifies?)")
    print(f"    diff persona & diff model: {np.mean(diff_both):.3f}  (residual 'crowded by construction')")

    # capacity implication: per-fund capacity A*_f = A*_single / kappa, kappa=1+(N-1)rho
    print(f"\n  CAPACITY IMPLICATION (per-fund A*_f = A*_single / kappa, kappa=1+(N-1)*rho_bar):")
    for N in (10, 50, 200):
        kL = 1 + (N - 1) * llm["rho_bar"]; kC = 1 + (N - 1) * max(cls["rho_bar"], 0)
        print(f"    N={N:>3} funds: kappa_LLM={kL:>6.1f}  kappa_classic={kC:>5.1f}  -> LLM per-fund "
              f"capacity {kC/kL:.2f}x that of a classic-diverse cohort")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    order = md.sort_values(["persona", "model"]).index
    Cl = pd.DataFrame(llm["C"], index=md["col"], columns=md["col"]).iloc[order, order]
    im0 = axes[0].imshow(Cl, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(cols))); axes[0].set_xticklabels(Cl.columns, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(cols))); axes[0].set_yticklabels(Cl.index, fontsize=6)
    axes[0].set_title(f"LLM agents (rho_bar={llm['rho_bar']:.2f}, N_eff={llm['N_eff']:.1f})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(cls["C"], vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(cls["M"])); axes[1].set_xticklabels(Wf.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(cls["M"])); axes[1].set_yticklabels(Wf.columns, fontsize=7)
    axes[1].set_title(f"classic factors (rho_bar={cls['rho_bar']:.2f}, N_eff={cls['N_eff']:.1f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    Ns = np.arange(2, 201)
    axes[2].plot(Ns, 1 / (1 + (Ns - 1) * llm["rho_bar"]), label="LLM cohort", color="C3")
    axes[2].plot(Ns, 1 / (1 + (Ns - 1) * max(cls["rho_bar"], 1e-3)), label="classic cohort", color="C0")
    axes[2].set_xlabel("cohort size N (funds)"); axes[2].set_ylabel(r"per-fund capacity / $A^*_{single}$")
    axes[2].set_title("Per-fund capacity vs cohort size"); axes[2].legend(); axes[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_cohort.png"), dpi=130); plt.close(fig)

    json.dump({"llm_rho_bar": llm["rho_bar"], "llm_abs_rho": llm["abs_rho"],
               "llm_eig1_frac": llm["eig1_frac"], "llm_N_eff": llm["N_eff"], "llm_M": llm["M"],
               "classic_rho_bar": cls["rho_bar"], "classic_abs_rho": cls["abs_rho"],
               "classic_N_eff": cls["N_eff"], "classic_M": cls["M"],
               "same_persona_corr": float(np.mean(same_persona)),
               "diff_both_corr": float(np.mean(diff_both)),
               "capacity_ratio_N50": float((1 + 49 * max(cls["rho_bar"], 0)) / (1 + 49 * llm["rho_bar"]))},
              open(os.path.join(OUT, "cohort_fit.json"), "w"), indent=2, default=float)
    print(f"\n  wrote fig_cohort.png and cohort_fit.json")


if __name__ == "__main__":
    main()
