"""
EQUITY MULTI-REGIME cohort: run 7 vendors x 5 personas on each of 3 PAST equity snapshots (uptrend /
bounce / crash) to test whether the cross-vendor collapse holds across regimes in EQUITIES too (not just
crypto, and not just one equity date). Reports per-regime N_eff, same-side overlap, vs matched null and the
classic-factor baseline.

Needs OPENROUTER_API_KEY; cached per date. Run: OPENROUTER_API_KEY=$(cat .orkey) python3 cohort_equity_multiregime.py
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from cohort_analyze import normalize, n_eff
from cohort_crossvendor import PERSONAS, OPENROUTER_MODELS, call_provider, _extract_json
from cohort_crossvendor_hardening import books, matched_null
from cohort_liquidity_signed import same_side_overlap

from _paths import RESULTS as OUT, FIGURES, DATA, inp


EQUITY_PROMPT_SPEC_VERSION = "equity-roster-locked-v1"


def build_equity_prompt(persona, syms, table, market_name="US"):
    return (f"You are a {persona} equity portfolio manager. Cross-section of {len(syms)} large-cap {market_name} stocks "
            f"(% returns 7/30/90-day, annualized 30-day vol %, drawdown from 90-day high %):\n\n{table}\n\n"
            f"Style ({persona}): {PERSONAS[persona]}\n\nAllocate a long/short book over EXACTLY: "
            f"{', '.join(syms)}. Reply ONLY JSON: {{\"weights\":[{{\"symbol\":\"AAPL\",\"weight\":0.5}}, ...]}}. "
            f"Positive=long, negative=short; cover all.")


# Backward-compatible name used by the original three-date collector.
prompt = build_equity_prompt


def collect(syms, table):
    jobs = [(v, m, p) for v, m in OPENROUTER_MODELS for p in PERSONAS]
    def one(j):
        v, m, p = j
        prov = {"name": v, "kind": "openrouter", "key": "OPENROUTER_API_KEY", "model": m}
        for attempt in range(3):
            try:
                obj = _extract_json(call_provider(prov, prompt(p, syms, table)))
                return {"vendor": v, "persona": p, "weights": obj["weights"]}
            except Exception as e:
                if attempt == 2:
                    print(f"    FAIL {v} {p}: {repr(e)[:40]}"); return None
                time.sleep(2 * (attempt + 1))
    out = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            r = f.result()
            if r:
                out.append(r)
    return out


def main():
    snaps = json.load(open(inp("equity_snapshots_multi.json")))
    rows = []
    for s in snaps:
        date, regime, syms, table = s["date"], s["regime"], s["syms"], s["snapshot"]
        cache = inp(f"cohort_agents_equity_{date}.json")
        if os.environ.get("OPENROUTER_API_KEY") and not os.path.exists(cache):
            print(f"collecting equity {date} ({regime})...")
            ag = collect(syms, table); json.dump(ag, open(cache, "w"), indent=1)
        elif os.path.exists(cache):
            ag = json.load(open(cache))
        else:
            print("set OPENROUTER_API_KEY"); return
        F = pd.read_csv(inp(f"equity_classic_factors_{date}.csv"), index_col=0)
        W = books(ag, syms); W = W.loc[:, W.std() > 1e-12]
        C = np.corrcoef(W.to_numpy().T); M = W.shape[1]
        ne, O = n_eff(C), same_side_overlap(W)
        null = matched_null(M, syms)
        cls = n_eff(np.corrcoef(normalize(F.reindex(syms).fillna(0.0)).to_numpy().T))
        rows.append({"date": date, "regime": regime, "M": M, "N_eff": float(ne), "overlap": float(O),
                     "abs_rho": float(np.abs(C[np.triu_indices(M, 1)]).mean()),
                     "null_mean": null["mean"], "null_p2_5": null["p2_5"], "classic_N_eff": float(cls)})

    print("\n" + "=" * 80); print("EQUITY MULTI-REGIME cross-vendor cohort (7 vendors x 5 personas per date)"); print("=" * 80)
    print(f"  {'date':>12} {'regime':>16} | {'N_eff':>6} {'|rho|':>6} {'overlap':>8} | {'null':>6} {'classic':>8}")
    for r in rows:
        print(f"  {r['date']:>12} {r['regime']:>16} | {r['N_eff']:>6.2f} {r['abs_rho']:>6.2f} {r['overlap']:>8.3f} | "
              f"{r['null_mean']:>6.2f} {r['classic_N_eff']:>8.2f}")
    mean_ne = float(np.mean([r["N_eff"] for r in rows])); mean_O = float(np.mean([r["overlap"] for r in rows]))
    allcollapse = all(r["N_eff"] < r["null_p2_5"] for r in rows)
    print(f"\n  MEAN over {len(rows)} equity regimes: N_eff={mean_ne:.2f}, overlap={mean_O:.3f}; "
          f"collapses below matched null in {sum(r['N_eff']<r['null_p2_5'] for r in rows)}/{len(rows)} regimes.")
    print(f"  (2026-05-31 single-snapshot equity reference: N_eff=1.78). "
          f"-> equity crowding {'is regime-robust, not a single-date artifact' if allcollapse else 'varies by regime'}.")
    json.dump({"per_regime": rows, "mean_N_eff": mean_ne, "mean_overlap": mean_O,
               "all_below_null": bool(allcollapse), "single_snapshot_ref_2026_05_31": 1.78},
              open(os.path.join(OUT, "cohort_equity_multiregime.json"), "w"), indent=2, default=float)
    print("  wrote cohort_equity_multiregime.json")


if __name__ == "__main__":
    main()
