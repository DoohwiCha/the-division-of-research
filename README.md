# The Division of Research: Information Access and Independent-Book Capacity in LLM Investment Teams

Code, frozen caches, and reproducibility record for the ICAIF '26 paper
*The Division of Research: Information Access and Independent-Book Capacity in LLM Investment Teams*
(Doohwi Cha, Minjae Lee, Minsuk Sung, Juyeong Lee, Donghwa Seo, Seunghan Son, Yoontae Hwang).

- `The_Division_of_Research_ICAIF26.pdf` — camera-ready paper (9 pages).
- `paper/` — LaTeX source of the camera-ready paper (`acmart`, `sigconf`). Generated number macros live in `paper/tables/`.
- `reproducibility/` — executed-protocol record referenced in §4.1 of the paper:
  - `reproducibility_review.pdf` — rendered record (routes, observed providers, both frozen 7×32 access matrices, all 32 field definitions and transformations, the full executed prompt template, and the API output schema).
  - `repro_protocol.json` — machine-readable protocol (routes, symbols, fields, matrices, request settings, schema hash).
  - `repro_prompt_template.txt`, `repro_prompt_examples.json` — the executed user-message template and three fully rendered prompts (shared / mixed / coherent) with packet and prompt hashes.
  - `repro_response_format.json` — the exact `response_format` sent to the API.
- `src/` — analysis and collection scripts. Every quantity reported in the paper is regenerated from frozen caches by deterministic scripts; no script in this repository calls a model API unless invoked explicitly with collection flags.
- `data/` — frozen packets, price/volume inputs, cached model responses (`data/cohort_agents_information_capacity_*.json`), experiment plans, and result files (`data/results/`).

## Reproducing reported quantities

Python ≥ 3.10 with `pip install -r requirements.txt`. Run scripts from the repository root; they read
frozen caches under `data/` and write to `data/results/`. Analysis scripts do not call model APIs.

Main entry points (one per reported panel; see each script's module docstring and `--help` for options):

| Reported object | Script |
|---|---|
| Effective-Book Curve, pooled Table 2 rows, 22-date complementary row | `src/cohort_information_capacity_audited_curve_analysis.py` |
| Order diagnostics and audited architecture results | `src/cohort_information_capacity_audited_analysis.py` |
| Moving-block bootstrap intervals | `src/cohort_information_capacity_block_bootstrap.py` |
| Assignment sensitivity (§5.3; all 5,040 route-to-packet bijections per date and design) | `src/cohort_information_capacity_assignment_dispersion.py` |
| Repeated-generation panel | `src/cohort_information_capacity_repeatability_analysis.py` |
| Risk-space effective dimension | `src/cohort_information_capacity_risk_space.py` |
| Harmonized U.S. / Europe / crypto panels (`--market us|europe|crypto`) | `src/cohort_information_capacity_crossmarket_analysis.py` |
| Long-only sensitivity (Table 4, Figure 5) | `src/cohort_information_capacity_audited_longonly_analysis.py` |
| Paper number macros (`paper/tables/numbers.tex`) and figures | `src/icaif_v4_assets.py` |
| Rendered prompt examples in the record | `src/export_information_capacity_prompt_examples.py` |

Cached model responses: `data/cohort_agents_information_capacity_*.json` (pooled U.S. audited curve, architecture,
integrity, repeatability, long-only, and `_us` / `_europe` / `_crypto` harmonized panels). Frozen packets:
`data/information_diversity_integrity*_packets.json` and `data/information_capacity_{us,europe,crypto}_packets.json`.

Smoke check performed before publishing: from a clean checkout of this repository,
`python3 src/cohort_information_capacity_assignment_dispersion.py` regenerated
`data/results/information_capacity_assignment_dispersion_results.json` with values numerically identical to the shipped file.

## Scope and provenance

- Requested routes are OpenRouter endpoints as named in the paper (§4.1); the record also lists the realized routed provider for every archived response. These identify requested endpoints, not immutable model weights.
- The caches contain the full archived responses, request metadata, prompt and packet hashes, and seed metadata (including recovery offsets). Treat them as the frozen experimental record; re-collecting from the APIs will not reproduce the same responses.
- Valuation prices and share counts are split-adjusted to a July 20, 2026 corporate-action cutoff; return, volatility, drawdown, and beta fields use an adjusted-close series. Source URLs, cutoffs, and hashes accompany the frozen packets (`data/information_diversity_integrity_packets.json`, `data/information_diversity_us_price_basis.json`).
- Nothing in this repository is investment advice. See the Disclaimer in the paper.

## Citation

```bibtex
@inproceedings{cha2026division,
  title     = {The Division of Research: Information Access and Independent-Book Capacity in LLM Investment Teams},
  author    = {Cha, Doohwi and Lee, Minjae and Sung, Minsuk and Lee, Juyeong and Seo, Donghwa and Son, Seunghan and Hwang, Yoontae},
  booktitle = {7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM}
}
```
