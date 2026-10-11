"""
Estimate capacity implications of measured cohort dependence.

Three upgrades, all from the cached single-snapshot panel (no new data, fully reproducible):

 (1) delta-CONSISTENT capacity BAND instead of a single linear-impact point. The per-fund capacity
     compression depends on HOW correlated cohort trading enters the impact law:
       - co-impact / sqrt-law: kappa scales the effective co-moving ORDER SIZE inside the law,
         drag_f = k*(kappa*A_f)^delta  =>  A*_f = A*_single / kappa            (delta-INVARIANT)
       - coefficient-scaling:  kappa scales the drag COEFFICIENT,
         drag_f = kappa*k*A_f^delta    =>  A*_f = A*_single * kappa^(-1/delta)
     So the honest object is a BAND  CR in [ (kc/kl)^(1/delta) , kc/kl ]  at the measured delta.
     (The original note reported only the upper, delta-invariant endpoint as "9%".)

 (2) MODEL-FREE crowding statistic (NO kappa, NO delta, NO impact model, NO extrapolation):
     aggregate-book gross SURVIVAL  S = gross(sum_i b_i) / sum_i gross(b_i),  b_i dollar-neutral unit-gross.
     Diversified cohort -> positions cancel on aggregation -> S ~ 1/sqrt(M); crowded cohort -> S ~ O(1).
     This is the robust headline the panel asked us to lead with.

 (3) signed-vs-|rho| x classic-clamp SURFACE for CR, so the headline does not rest on one choice.

Run:  python3 cohort_capacity_delta.py  ->  results/cohort_capacity_band.json
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from cohort_analyze import normalize, mean_pair_corr

from _paths import RESULTS as OUT, FIGURES, DATA, inp


def books(agents, syms):
    A = pd.DataFrame(0.0, index=syms, columns=[f"{a['persona']}:{a['model']}" for a in agents])
    for a in agents:
        for w in a["weights"]:
            if w["symbol"] in A.index:
                A.loc[w["symbol"], f"{a['persona']}:{a['model']}"] = float(w["weight"])
    return normalize(A)                                  # dollar-neutral, unit-gross columns


def gross_survival(W: pd.DataFrame) -> float:
    """S = gross of the aggregate (summed) book / total gross deployed. ~1/sqrt(M) if diversified,
    ~O(1) if crowded. Model-free: no kappa/delta/impact assumption."""
    agg = W.sum(axis=1)
    return float(agg.abs().sum() / W.abs().sum().sum())


def main():
    delta = json.load(open(os.path.join(OUT, "cross_venue_fit.json")))["pooled_delta"]
    agents = json.load(open(inp("cohort_agents.json")))
    F = pd.read_csv(inp("cohort_classic_factors.csv"), index_col=0)
    syms = F.index.tolist()
    Wl = books(agents, syms)
    Wf = normalize(F.reindex(syms).fillna(0.0))
    M = Wl.shape[1]

    rho_l_signed = mean_pair_corr(np.corrcoef(Wl.to_numpy().T))
    rho_c_signed = mean_pair_corr(np.corrcoef(Wf.to_numpy().T))
    iu = lambda C: C[np.triu_indices(C.shape[0], 1)]
    rho_l_abs = float(np.abs(iu(np.corrcoef(Wl.to_numpy().T))).mean())
    rho_c_abs = float(np.abs(iu(np.corrcoef(Wf.to_numpy().T))).mean())

    print("=" * 80); print(f"COHORT CAPACITY — delta-consistent BAND (measured delta={delta:.3f})"); print("=" * 80)

    def cr_band(N, rho_c, rho_l):
        kl = 1 + (N - 1) * rho_l
        kc = 1 + (N - 1) * max(rho_c, 0.0)             # clamp: a diversified cohort gets no capacity bonus
        co = kc / kl                                    # delta-invariant (co-impact / effective size)
        coef = (kc / kl) ** (1.0 / delta)               # coefficient-scaling
        return kl, kc, min(co, coef), max(co, coef)

    print("\n  (1) delta-consistent per-fund capacity ratio CR = A*_f,LLM / A*_f,classic (signed rho, classic clamped >=0):")
    print(f"      {'N':>4} | {'kappa_LLM':>9} {'kappa_cls':>9} | {'CR band (coeff..co-impact)':>30}")
    bands = {}
    for N in (10, 50, 200):
        kl, kc, lo, hi = cr_band(N, rho_c_signed, rho_l_signed)
        bands[N] = {"kappa_llm": kl, "kappa_classic": kc, "cr_low_coeff": lo, "cr_high_coimpact": hi}
        print(f"      {N:>4} | {kl:>9.2f} {kc:>9.2f} | {lo*100:>6.1f}% .. {hi*100:>5.1f}%")
    print("      (note's original '9% @N=50 / 2% @N=200' = the delta-invariant co-impact ENDPOINT;")
    print("       the band's lower endpoint is the coefficient-scaling reading. Both = severe compression.)")

    # (2) model-free gross survival
    S_llm = gross_survival(Wl)
    S_cls = gross_survival(Wf)
    rng = np.random.default_rng(0); B = 3000
    S_null = []
    for _ in range(B):
        Wr = normalize(pd.DataFrame(rng.standard_normal((len(syms), M)), index=syms))
        S_null.append(gross_survival(Wr))
    S_null = np.array(S_null)
    print("\n  (2) MODEL-FREE crowding statistic — aggregate-book gross survival S (no kappa/delta):")
    print(f"      S_LLM     = {S_llm:.3f}   (positions REINFORCE on aggregation -> liquidity crowding)")
    print(f"      S_classic = {S_cls:.3f}")
    print(f"      random-null S = {S_null.mean():.3f}  [95% {np.percentile(S_null,2.5):.3f}, {np.percentile(S_null,97.5):.3f}]"
          f"  (1/sqrt(M)={1/np.sqrt(M):.3f})")
    print(f"      -> LLM is {'FAR ABOVE' if S_llm > np.percentile(S_null,97.5) else 'within'} the null:"
          f" {S_llm/S_null.mean():.1f}x the random aggregate gross.")

    # (3) signed vs abs x clamp surface at N=50
    print("\n  (3) CR sensitivity surface at N=50 (co-impact endpoint shown):")
    surf = {}
    for tag, rc, rl in [("signed", rho_c_signed, rho_l_signed), ("abs", rho_c_abs, rho_l_abs)]:
        for clamp in (True, False):
            rcc = max(rc, 0.0) if clamp else rc
            kl = 1 + 49 * rl; kc = 1 + 49 * rcc
            cr = kc / kl if kl != 0 else float("nan")
            surf[f"{tag}_{'clamped' if clamp else 'raw'}"] = cr
            print(f"      rho={tag:>6}, classic {'clamped>=0' if clamp else 'raw':>10}: CR = {cr*100:>6.1f}%   (kc={kc:.2f}, kl={kl:.2f})")

    json.dump({"delta": delta, "M": M,
               "rho": {"llm_signed": rho_l_signed, "llm_abs": rho_l_abs,
                       "classic_signed": rho_c_signed, "classic_abs": rho_c_abs},
               "capacity_band_signed_clamped": bands,
               "gross_survival": {"llm": S_llm, "classic": S_cls,
                                  "null_mean": float(S_null.mean()),
                                  "null_p2_5": float(np.percentile(S_null, 2.5)),
                                  "null_p97_5": float(np.percentile(S_null, 97.5)),
                                  "one_over_sqrt_M": float(1 / np.sqrt(M))},
               "cr_surface_N50": surf},
              open(os.path.join(OUT, "cohort_capacity_band.json"), "w"), indent=2, default=float)
    print("\n  wrote cohort_capacity_band.json")


if __name__ == "__main__":
    main()
