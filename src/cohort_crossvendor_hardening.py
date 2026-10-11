"""
CROSS-VENDOR VALIDATION — re-run the mechanism results (factor-residualization, realized co-movement,
capacity, matched random null) on the 35-agent 7-vendor panel so the headline and mechanism use the
same panel.

Reuses the exact estimators from cohort_analyze / cohort_harden / cohort_capacity_delta / cohort_predictive,
just re-pointed at cohort_agents_crossvendor.json (single snapshot, 35 agents) and
cohort_agents_crossvendor_multi.json (4 regimes x 3 replicates x 35 agents).

Run:  python3 cohort_crossvendor_hardening.py  ->  results/cohort_crossvendor_hardening.json
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from cohort_analyze import normalize, n_eff, mean_pair_corr
from cohort_harden import neff_absrho, residualize
from cohort_capacity_delta import gross_survival

from _paths import RESULTS as OUT, FIGURES, DATA, inp
SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT"]
H = 30


def books(agents, syms):
    cols = [f"{a['persona']}:{a.get('vendor') or a.get('model')}" for a in agents]
    A = pd.DataFrame(0.0, index=syms, columns=cols)
    for a, c in zip(agents, cols):
        for w in a["weights"]:
            if w["symbol"] in A.index:
                A.loc[w["symbol"], c] = float(w["weight"])
    return normalize(A)


def family_average_diagnostic(agents, W):
    """Audit the seven model-family averages on the primary market snapshot."""
    groups = {}
    for agent, column in zip(agents, W.columns):
        groups.setdefault(agent.get("vendor") or agent.get("model"), []).append(column)
    family_books = pd.DataFrame(
        {family: W[columns].mean(axis=1) for family, columns in sorted(groups.items())},
        index=W.index,
    )
    family_books = normalize(family_books)
    ne, absrho = neff_absrho(family_books)
    C = np.corrcoef(family_books.to_numpy(dtype=float).T)
    return {
        "source": "cohort_agents_crossvendor.json",
        "normalization": "mean five unit-gross persona books within family, then renormalize each family average to unit gross",
        "n_family_books": int(family_books.shape[1]),
        "N_eff": float(ne),
        "mean_abs_corr": float(absrho),
        "mean_signed_corr": float(mean_pair_corr(C)),
        "same_side_overlap": float(same_side_overlap(family_books)),
    }


def matched_null(M, syms, F=None, B=3000, seed=0):
    rng = np.random.default_rng(seed); d = len(syms); ne = []
    for _ in range(B):
        Wr = normalize(pd.DataFrame(rng.standard_normal((d, M)), index=syms))
        if F is not None:
            Wr = residualize(Wr, F)
        ne.append(neff_absrho(Wr)[0])
    ne = np.array(ne)
    return {"mean": float(ne.mean()), "p2_5": float(np.percentile(ne, 2.5)), "p97_5": float(np.percentile(ne, 97.5))}


def same_side_overlap(W: pd.DataFrame) -> float:
    arr = normalize(W).to_numpy(dtype=float).T
    vals = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            same = np.sign(arr[i]) == np.sign(arr[j])
            vals.append(float((np.minimum(np.abs(arr[i]), np.abs(arr[j])) * same).sum()))
    return float(np.mean(vals)) if vals else 0.0


def book_permutation_null(W: pd.DataFrame, B: int = 3000, seed: int = 20260710) -> dict:
    """Preserve every book's empirical weights while independently permuting asset identity."""
    rng = np.random.default_rng(seed)
    arr = normalize(W).to_numpy(dtype=float)
    neff_draws = []
    overlap_draws = []
    for _ in range(B):
        permuted = np.column_stack([rng.permutation(arr[:, j]) for j in range(arr.shape[1])])
        draw = pd.DataFrame(permuted, index=W.index, columns=W.columns)
        neff_draws.append(n_eff(np.corrcoef(draw.to_numpy(dtype=float).T)))
        overlap_draws.append(same_side_overlap(draw))
    neff_draws = np.asarray(neff_draws, dtype=float)
    overlap_draws = np.asarray(overlap_draws, dtype=float)
    return {
        "draws": B,
        "seed": seed,
        "N_eff": {
            "mean": float(neff_draws.mean()),
            "p2_5": float(np.percentile(neff_draws, 2.5)),
            "p97_5": float(np.percentile(neff_draws, 97.5)),
        },
        "same_side_overlap": {
            "mean": float(overlap_draws.mean()),
            "p2_5": float(np.percentile(overlap_draws, 2.5)),
            "p97_5": float(np.percentile(overlap_draws, 97.5)),
        },
    }


def main():
    agents = json.load(open(inp("cohort_agents_crossvendor.json")))
    F = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0)
    syms = F.index.tolist()
    Wl = books(agents, syms); Wl = Wl.loc[:, Wl.std() > 1e-12]
    Wf = normalize(F.reindex(syms).fillna(0.0))
    M = Wl.shape[1]
    delta = json.load(open(os.path.join(OUT, "cross_venue_fit.json")))["pooled_delta"]

    print("=" * 84); print(f"CROSS-VENDOR HARDENING — all mechanism results on the {M}-agent 7-vendor panel"); print("=" * 84)

    # (0) matched M=35 null + rank ceiling
    ne_raw, ar_raw = neff_absrho(Wl)
    family_average = family_average_diagnostic(agents, Wl)
    null35 = matched_null(M, syms)
    permutation_null = book_permutation_null(Wl)
    rank_ceiling = min(M, len(syms) - 1)            # dollar-neutral on D assets -> rank <= D-1
    print(f"\n  (0) headline N_eff={ne_raw:.2f} on M={M};  matched M={M} random null={null35['mean']:.2f} "
          f"[{null35['p2_5']:.2f},{null35['p97_5']:.2f}];  rank ceiling (12 dollar-neutral assets)={rank_ceiling}")
    print(f"      book-permutation null={permutation_null['N_eff']['mean']:.2f} "
          f"[{permutation_null['N_eff']['p2_5']:.2f},{permutation_null['N_eff']['p97_5']:.2f}], "
          f"overlap={permutation_null['same_side_overlap']['mean']:.3f}")
    print(
        f"      primary family averages: N_eff={family_average['N_eff']:.2f} on "
        f"M={family_average['n_family_books']}; |rho|={family_average['mean_abs_corr']:.3f}; "
        f"overlap={family_average['same_side_overlap']:.3f}"
    )

    # (1) factor-residualization on the cross-vendor panel
    ne_res, absrho_res = neff_absrho(residualize(Wl, Wf))
    null35_res = matched_null(M, syms, F=Wf)
    print(f"  (1) factor-residualized cross-vendor N_eff: {ne_raw:.2f} -> {ne_res:.2f}  (residual null {null35_res['mean']:.2f}"
          f" [{null35_res['p2_5']:.2f},{null35_res['p97_5']:.2f}]) -> {'orthogonal to factors' if ne_res<null35_res['p2_5'] else 'partly factor'}")

    # (2) model-free surviving gross on the cross-vendor panel
    S = gross_survival(Wl)
    rng = np.random.default_rng(0)
    Snull = np.array([gross_survival(normalize(pd.DataFrame(rng.standard_normal((len(syms), M)), index=syms))) for _ in range(3000)])
    print(f"  (2) surviving-gross S(cross-vendor)={S:.3f}  vs null {Snull.mean():.3f} [{np.percentile(Snull,2.5):.3f},{np.percentile(Snull,97.5):.3f}]"
          f"  (1/sqrt(M)={1/np.sqrt(M):.3f}) -> {S/Snull.mean():.1f}x")

    # (3) capacity band on the cross-vendor rho
    rho_l = mean_pair_corr(np.corrcoef(Wl.to_numpy().T)); rho_c = mean_pair_corr(np.corrcoef(Wf.to_numpy().T))
    def cr_band(N):
        kl = 1 + (N - 1) * rho_l; kc = 1 + (N - 1) * max(rho_c, 0.0)
        co = kc / kl; coef = (kc / kl) ** (1.0 / delta)
        return min(co, coef), max(co, coef)
    cap = {N: cr_band(N) for N in (10, 50, 200)}
    print(f"  (3) capacity band (cross-vendor signed rho_l={rho_l:.3f}, classic clamped): "
          f"N=50 CR=[{cap[50][0]*100:.1f}%,{cap[50][1]*100:.1f}%]  N=200 [{cap[200][0]*100:.1f}%,{cap[200][1]*100:.1f}%]")

    # (4) realized co-movement on the cross-vendor MULTI panel (factor-residualized + raw), with replicate error bars
    multi = json.load(open(inp("cohort_agents_crossvendor_multi.json")))
    snaps = json.load(open(inp("cohort_snapshots_multi.json")))
    prices = pd.read_csv(inp("cohort_prices.csv"), index_col=0); prices.index = prices.index.astype(int)
    rets = prices.pct_change()
    daydate = pd.to_datetime(np.array(prices.index), unit="ms").date
    today = "2026-05-31"
    realized_rows = []
    for s in snaps:
        date = s["date"]
        if date == today:
            continue
        hits = np.where([str(d) == date for d in daydate])[0]
        if len(hits) == 0:
            continue
        i0 = int(hits[0]); win = rets.iloc[i0 + 1:i0 + 1 + H][SYMS]
        if len(win) < 10:
            continue
        Wfd = normalize(pd.DataFrame({k: v for k, v in s["classic"].items()}).reindex(SYMS).fillna(0.0))
        for r in sorted({a["replicate"] for a in multi if a["date"] == date}):
            panel = [a for a in multi if a["date"] == date and a["replicate"] == r]
            W = books(panel, SYMS); W = W.loc[:, W.std() > 1e-12]
            pos_ne = neff_absrho(W)[0]
            ar = win.to_numpy() @ W.reindex(SYMS).fillna(0.0).to_numpy()
            ar = ar[:, ar.std(axis=0) > 1e-12]
            real_ne = n_eff(np.corrcoef(ar.T))
            # factor-residualized realized: strip the classic-factor return axes from agent returns
            fr = win.to_numpy() @ Wfd.reindex(SYMS).fillna(0.0).to_numpy()       # factor return series (H x 5)
            fr = fr[:, fr.std(axis=0) > 1e-12]
            P = fr @ np.linalg.pinv(fr)
            ar_res = ar - P @ ar
            ar_res = ar_res[:, ar_res.std(axis=0) > 1e-12]
            real_ne_res = n_eff(np.corrcoef(ar_res.T)) if ar_res.shape[1] > 1 else np.nan
            realized_rows.append({"date": date, "replicate": r, "pos_neff": pos_ne,
                                  "realized_neff": real_ne, "realized_neff_factor_resid": float(real_ne_res)})
    rr = pd.DataFrame(realized_rows)
    print(f"\n  (4) REALIZED co-movement on cross-vendor multi panel ({len(rr)} panels, factor-residualized too):")
    for date, g in rr.groupby("date"):
        print(f"      {date}: pos N_eff {g['pos_neff'].mean():.2f} | realized {g['realized_neff'].mean():.2f}+/-{g['realized_neff'].std(ddof=1):.2f}"
              f" | realized FACTOR-RESID {g['realized_neff_factor_resid'].mean():.2f}+/-{g['realized_neff_factor_resid'].std(ddof=1):.2f}")
    print(f"      MEAN: pos {rr['pos_neff'].mean():.2f} -> realized {rr['realized_neff'].mean():.2f} "
          f"-> realized factor-residualized {rr['realized_neff_factor_resid'].mean():.2f}  (null ~ {null35_res['mean']:.1f})")

    out = {"panel": "cross-vendor", "M": M, "rank_ceiling": rank_ceiling, "delta": delta,
           "static": {"neff_raw": float(ne_raw), "absrho_raw": float(ar_raw),
                      "family_average_diagnostic": family_average, "null_M35": null35,
                      "book_permutation_null": permutation_null},
           "factor_residualized": {"neff": float(ne_res), "absrho": float(absrho_res), "null_M35_resid": null35_res,
                                   "orthogonal": bool(ne_res < null35_res["p2_5"])},
           "surviving_gross": {"S": S, "null_mean": float(Snull.mean()),
                               "null_p2_5": float(np.percentile(Snull, 2.5)), "null_p97_5": float(np.percentile(Snull, 97.5)),
                               "one_over_sqrtM": float(1 / np.sqrt(M)), "ratio": float(S / Snull.mean())},
           "rho": {"llm_signed": rho_l, "classic_signed": rho_c},
           "capacity_band": {str(N): {"cr_low": cap[N][0], "cr_high": cap[N][1]} for N in cap},
           "realized": {"mean_pos_neff": float(rr["pos_neff"].mean()),
                        "mean_realized_neff": float(rr["realized_neff"].mean()),
                        "mean_realized_neff_factor_resid": float(rr["realized_neff_factor_resid"].mean()),
                        "n_panels": len(rr),
                        "per_regime": rr.groupby("date").agg(
                            pos=("pos_neff", "mean"), realized=("realized_neff", "mean"),
                            realized_sd=("realized_neff", "std"),
                            realized_fr=("realized_neff_factor_resid", "mean")).reset_index().to_dict("records")}}
    json.dump(out, open(os.path.join(OUT, "cohort_crossvendor_hardening.json"), "w"), indent=2, default=float)
    print("\n  wrote cohort_crossvendor_hardening.json")


if __name__ == "__main__":
    main()
