"""
Collect a cross-vendor cohort panel under a common portfolio task.

The keystone's most-quotable claim ("model diversity does almost nothing", corr 0.88) is currently
established only WITHIN the Anthropic family (Opus/Sonnet/Haiku). This harness re-runs the SAME 5
personas on the SAME market snapshot across MULTIPLE PROVIDERS (OpenAI, Google, Meta-Llama, Alibaba-
Qwen, DeepSeek, Mistral, Anthropic) so we can test whether convergence holds ACROSS vendors — turning
"Anthropic siblings converge" into "LLMs converge across providers", the population-level claim the
market-structure warning needs.

It calls each provider's public REST API with stdlib urllib only (no extra deps), parses each agent's
long/short allocation, MERGES the multi-vendor panel, and reports cross-vendor N_eff plus the
same-vendor-vs-cross-vendor correlation decomposition. Providers with no API key in the environment
are skipped (so it runs with whatever keys are available).

ENV KEYS (set the ones you have): OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), GROQ_API_KEY,
TOGETHER_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY.

Run:  python3 cohort_crossvendor.py        # queries every provider whose key is set
Output: results/{cohort_agents_crossvendor.json, cohort_crossvendor.json, fig_cohort_crossvendor.png}
"""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.error
import numpy as np
import pandas as pd
from cohort_analyze import normalize, n_eff, mean_pair_corr

from _paths import RESULTS as OUT, FIGURES, DATA, inp

PERSONAS = {
    "momentum": "Trend-follower: long the strongest positive-momentum names in the supplied snapshot, short the weakest; conviction scales with observed momentum.",
    "contrarian": "Mean-reversion: long the most beaten-down / largest-drawdown names, short the recent winners; you deliberately fade the crowd.",
    "defensive": "Capital preservation: tilt long low-volatility, shallow-drawdown names; underweight/short high-vol deep-drawdown names; keep gross modest.",
    "aggressive": "High-conviction directional: concentrate in a few strong bets, embrace high vol and strong momentum; large long and short where the signal is clearest.",
    "quant": "Systematic multi-factor: blend the return, volatility, and drawdown features supplied in the snapshot into a diversified dollar-neutral book; spread risk across many names.",
}

PROMPT_SPEC_VERSION = "cohort-primary-v2"

# OpenAI-compatible providers share one schema; Gemini and Anthropic have their own.
PROVIDERS = [
    {"name": "openai",    "kind": "openai", "key": "OPENAI_API_KEY",    "url": "https://api.openai.com/v1/chat/completions",        "model": "gpt-4o"},
    {"name": "groq",      "kind": "openai", "key": "GROQ_API_KEY",      "url": "https://api.groq.com/openai/v1/chat/completions",     "model": "llama-3.3-70b-versatile"},
    {"name": "together",  "kind": "openai", "key": "TOGETHER_API_KEY",  "url": "https://api.together.xyz/v1/chat/completions",        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo"},
    {"name": "deepseek",  "kind": "openai", "key": "DEEPSEEK_API_KEY",  "url": "https://api.deepseek.com/chat/completions",           "model": "deepseek-chat"},
    {"name": "mistral",   "kind": "openai", "key": "MISTRAL_API_KEY",   "url": "https://api.mistral.ai/v1/chat/completions",          "model": "mistral-large-latest"},
    {"name": "google",    "kind": "gemini", "key": "GEMINI_API_KEY",    "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", "model": "gemini-1.5-pro", "alt_key": "GOOGLE_API_KEY"},
    {"name": "anthropic", "kind": "anthropic", "key": "ANTHROPIC_API_KEY", "url": "https://api.anthropic.com/v1/messages",            "model": "claude-opus-4-8"},
]

# PREFERRED PATH: one OpenRouter key (OPENROUTER_API_KEY) reaches every vendor through an
# OpenAI-compatible endpoint -> true cross-vendor with a single credential. One model per distinct
# vendor keeps cost minimal; edit this list to add/swap models.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    ("openai",    "openai/gpt-4o"),
    ("google",    "google/gemini-2.5-flash"),
    ("meta",      "meta-llama/llama-3.1-70b-instruct"),
    ("alibaba",   "qwen/qwen-2.5-72b-instruct"),
    ("mistral",   "mistralai/mistral-large"),
    ("deepseek",  "deepseek/deepseek-chat"),
    ("anthropic", "anthropic/claude-sonnet-4.5"),
]


def _http_timeout() -> float:
    raw = os.environ.get("LLM_HTTP_TIMEOUT_SECONDS") or os.environ.get("OPENROUTER_TIMEOUT_SECONDS")
    if not raw:
        return 90.0
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 90.0


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_http_timeout()) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _extract_json(text):
    """Decode the first complete JSON object, ignoring trailing commentary."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("first JSON value is not an object")
    return value


def call_provider(p, prompt):
    key = os.environ.get(p["key"]) or (os.environ.get(p.get("alt_key", "")) if p.get("alt_key") else None)
    if not key:
        return None
    if p["kind"] == "openrouter":
        d = _post(OPENROUTER_URL, {"model": p["model"], "temperature": 0.7, "max_tokens": 1500,
                                   "messages": [{"role": "user", "content": prompt}]},
                  {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://icaif-cohort.local",
                   "X-Title": "cohort-crossvendor"})
        return d["choices"][0]["message"]["content"]
    if p["kind"] == "openai":
        d = _post(p["url"], {"model": p["model"], "temperature": 0.7, "max_tokens": 700,
                             "messages": [{"role": "user", "content": prompt}]},
                  {"Authorization": f"Bearer {key}"})
        return d["choices"][0]["message"]["content"]
    if p["kind"] == "gemini":
        url = p["url"].format(model=p["model"]) + f"?key={key}"
        d = _post(url, {"contents": [{"parts": [{"text": prompt}]}]}, {})
        return d["candidates"][0]["content"]["parts"][0]["text"]
    if p["kind"] == "anthropic":
        d = _post(p["url"], {"model": p["model"], "max_tokens": 1024, "temperature": 0.7,
                             "messages": [{"role": "user", "content": prompt}]},
                  {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return d["content"][0]["text"]
    raise ValueError(p["kind"])


def snapshot_table():
    snap = json.load(open(inp("market_snapshot.json")))
    syms = [r["symbol"] for r in snap]
    hdr = "asset    ret_30d  vol_30d_ann  drawdown  price"
    rows = [f"{r['symbol']:<8} {r['ret_30d']:>7.1f} {r['vol_30d_ann']:>11.1f} {r['drawdown']:>9.1f} {r['price']:>10.2f}" for r in snap]
    return syms, hdr + "\n" + "\n".join(rows)


def build_prompt(persona, style, syms, table):
    return (f"You are a {persona} crypto portfolio manager. Cross-section of 12 majors "
            f"(% 30-day return, annualized 30-day vol %, drawdown %, price):\n\n{table}\n\n"
            f"Your style — {persona}: {style}\n\n"
            "Use only the columns shown in the snapshot; do not assume unavailable style features.\n\n"
            f"Allocate a long/short book over EXACTLY these symbols: {', '.join(syms)}. Reply ONLY with JSON: "
            f'{{\"weights\":[{{\"symbol\":\"BTCUSDT\",\"weight\":0.5}}, ...], \"thesis\":\"one sentence\"}}. '
            f"Positive=long, negative=short, magnitude=conviction; cover all 12.")


def matrix(agents, syms):
    A = pd.DataFrame(0.0, index=syms, columns=[f"{a['persona']}:{a['model']}" for a in agents])
    for a in agents:
        for w in a["weights"]:
            if w["symbol"] in A.index:
                A.loc[w["symbol"], f"{a['persona']}:{a['model']}"] = float(w["weight"])
    return normalize(A)


def main():
    syms, table = snapshot_table()
    if os.environ.get("OPENROUTER_API_KEY"):                       # preferred: one key, all vendors
        available = [{"name": v, "kind": "openrouter", "key": "OPENROUTER_API_KEY", "model": m}
                     for v, m in OPENROUTER_MODELS]
    else:                                                          # fallback: per-provider keys
        available = [p for p in PROVIDERS if (os.environ.get(p["key"]) or (p.get("alt_key") and os.environ.get(p["alt_key"])))]
    print("=" * 80); print("CROSS-VENDOR LLM COHORT — vendors this run:",
                            ", ".join(p["name"] for p in available) or "(NONE)"); print("=" * 80)
    if not available:
        print("No API key found. EASIEST: set an OpenRouter key (one key -> all vendors):\n"
              "  ! export OPENROUTER_API_KEY=sk-or-...   then re-run `python3 cohort_crossvendor.py`\n"
              "Or set per-provider keys (OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY\n"
              "/ MISTRAL_API_KEY / ANTHROPIC_API_KEY). Harness is ready; this run is a no-op without keys.")
        return

    agents = []
    for p in available:
        for persona, style in PERSONAS.items():
            try:
                txt = call_provider(p, build_prompt(persona, style, syms, table))
                obj = _extract_json(txt)
                agents.append({"persona": persona, "model": f"{p['name']}:{p['model']}",
                               "vendor": p["name"], "weights": obj["weights"], "thesis": obj.get("thesis", ""),
                               "prompt_spec_version": PROMPT_SPEC_VERSION})
                print(f"  ok  {p['name']:<9} {persona}")
            except Exception as e:
                print(f"  FAIL {p['name']:<9} {persona}: {e}")
            time.sleep(1.0)                                   # gentle rate limiting

    if len(agents) < 4:
        print(f"\nOnly {len(agents)} allocations collected — need >=2 vendors to test cross-vendor. Stopping.")
        json.dump(agents, open(inp("cohort_agents_crossvendor.json"), "w"), indent=1)
        return

    json.dump(agents, open(inp("cohort_agents_crossvendor.json"), "w"), indent=1)
    W = matrix(agents, syms)
    W = W.loc[:, W.std() > 1e-12]
    C = np.corrcoef(W.to_numpy().T)
    cols = list(W.columns)
    vend = {f"{a['persona']}:{a['model']}": a["vendor"] for a in agents}
    pers = {f"{a['persona']}:{a['model']}": a["persona"] for a in agents}
    same_v, cross_v, same_p_cross_v = [], [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C[i, j]
            sv = vend.get(cols[i]) == vend.get(cols[j]); sp = pers.get(cols[i]) == pers.get(cols[j])
            (same_v if sv else cross_v).append(v)
            if sp and not sv:
                same_p_cross_v.append(v)

    ne = n_eff(C); M = len(cols)
    res = {"n_vendors": len(set(vend.values())), "vendors": sorted(set(vend.values())), "M": M,
           "N_eff": ne, "N_eff_frac": ne / M, "mean_abs_corr": float(np.abs(C[np.triu_indices(M, 1)]).mean()),
           "same_vendor_corr": float(np.mean(same_v)) if same_v else None,
           "cross_vendor_corr": float(np.mean(cross_v)) if cross_v else None,
           "same_persona_cross_vendor_corr": float(np.mean(same_p_cross_v)) if same_p_cross_v else None}
    print(f"\n  CROSS-VENDOR PANEL: {res['n_vendors']} vendors, M={M} agents")
    print(f"    N_eff = {ne:.2f} ({100*ne/M:.0f}% of M)   mean|corr| = {res['mean_abs_corr']:.2f}")
    print(f"    same-vendor corr   = {res['same_vendor_corr']}")
    print(f"    CROSS-vendor corr  = {res['cross_vendor_corr']}   <-- the load-bearing number")
    print(f"    same-persona/cross-vendor corr = {res['same_persona_cross_vendor_corr']}")
    print("    -> if CROSS-vendor corr stays high, convergence is an LLM-population property, not an Anthropic-family artifact.")
    json.dump(res, open(os.path.join(OUT, "cohort_crossvendor.json"), "w"), indent=2, default=float)
    print("\n  wrote cohort_agents_crossvendor.json + cohort_crossvendor.json")


if __name__ == "__main__":
    main()
