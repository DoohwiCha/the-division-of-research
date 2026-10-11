# The Division of Research

**Information Access and Independent-Book Capacity in LLM Investment Teams** — ICAIF '26 (Milan, November 14–17, 2026).
Doohwi Cha, Minjae Lee, Minsuk Sung, Juyeong Lee, Donghwa Seo, Seunghan Son, Yoontae Hwang.

An LLM investment team is characterized jointly by its model roster and its information-access matrix. This
repository holds the camera-ready paper, the executed-protocol record, the analysis code, and the frozen caches
that regenerate every number in the paper. The experiment is a bundled shared-versus-split access contrast on
proposed books; it measures geometric breadth, not predictive quality, alpha, or fundable capacity.

## Headline Findings

Pooled U.S. panel: 34 month-ends (July 2023 – April 2026), 14 large-cap stocks, seven requested OpenRouter routes,
32 point-in-time fields, 13,090 archived books. At seven seats, shared access (every seat reads the full packet)
produces `N_eff = 2.05` effective books, versus `3.98` under mixed split access and `3.89` under coherent split
access; the assignment-averaged mixed-minus-shared gain is `+1.93` (95% three-month moving-block interval
`[+1.76, +2.09]`) and is positive on all 34 dates. Split access lowers same-side overlap, proposed external gross,
order concentration, and common-book correlation. Mixed access also uses 68% fewer prompt tokens and 52% less
model-reported request cost. Both splits exceed shared access in harmonized U.S., European, and crypto-perpetual
panels; there coherent exceeds mixed, reversing the pooled ordering. A long-only sensitivity is descriptive only.

## Layout

```text
the-division-of-research/
|-- The_Division_of_Research_ICAIF26.pdf   # camera-ready paper (9 pages)
|-- paper/            # LaTeX source: main.tex, tables/*.tex (generated macros), fig_*.pdf, refs.bib, acmart.cls
|-- reproducibility/  # executed-protocol record (PDF + LaTeX) and machine-readable protocol/prompt/schema files
|-- src/              # measurements, experiment analyses, collectors, table and figure generators
|-- data/             # frozen packets, price/volume inputs, cached model responses
|-- data/results/     # plans and machine-readable outputs used by the paper
`-- requirements.txt
```

`reproducibility/` contains the record referenced in §4.1 of the paper: requested and canonical route identifiers,
observed providers, both frozen 7×32 access matrices, all 32 field definitions and transformations, the full
executed prompt template (`repro_prompt_template.txt`), three fully rendered prompts with packet and prompt hashes
(`repro_prompt_examples.json`), and the exact API `response_format` (`repro_response_format.json`).

## Reproduce The Paper

Python ≥ 3.10. The offline path uses the cached responses under `data/` and needs no API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/cohort_information_capacity_audited_curve_analysis.py      # Effective-Book Curve, Table 2 pooled + 22-date rows
python3 src/cohort_information_capacity_audited_analysis.py            # order diagnostics, audited architecture results
python3 src/cohort_information_capacity_block_bootstrap.py             # moving-block bootstrap intervals
python3 src/cohort_information_capacity_assignment_dispersion.py       # §5.3 assignment sensitivity (5,040 bijections/date/design)
python3 src/cohort_information_capacity_repeatability_analysis.py      # six-date repeated-generation panel
python3 src/cohort_information_capacity_risk_space.py                  # risk-space effective dimension
python3 src/cohort_information_capacity_crossmarket_analysis.py --market us
python3 src/cohort_information_capacity_crossmarket_analysis.py --market europe
python3 src/cohort_information_capacity_crossmarket_analysis.py --market crypto
python3 src/cohort_information_capacity_audited_longonly_analysis.py   # long-only sensitivity (Table 4, Figure 5)
python3 src/export_information_capacity_prompt_examples.py             # rendered prompt examples
python3 src/icaif_v4_assets.py                                         # table fragments, number macros, figures
```

Generators write to `submissions/icaif-2026-v4-information-architecture/` (tables, `numbers.tex`, figures) and
`submissions/icaif-2026-v7-camera-ready-review/tables/` (assignment-sensitivity tables); the copies under `paper/`
are the files the camera-ready PDF was built from.

Verified before publishing (2026-10-11, clean `git clone`, no `OPENROUTER_API_KEY` in the environment): every
command above exits 0; the regenerated `data/results/*_results.json`, `*_summary.csv`,
`information_capacity_crossmarket_combined.json`, and `information_capacity_assignment_dispersion_results.json`
are numerically identical to the shipped files (differences limited to run timestamps and the provenance hashes
that embed them); all seven regenerated table fragments (`numbers.tex`, `protocol.tex`, `primary_results.tex`,
`market_results.tex`, `longonly_results.tex`, `assignment_dispersion.tex`, `assignment_dispersion_numbers.tex`)
are byte-identical to `paper/tables/`; the five regenerated figures carry identical labels and values to
`paper/fig_*.pdf` (sub-pixel canvas differences arise from the local matplotlib/font build). Total wall time is
roughly 20 minutes on a laptop.

To intentionally refresh the external LLM caches, set `OPENROUTER_API_KEY` in the shell and run the relevant
collector (`src/cohort_information_capacity_experiment.py`, `..._crossmarket_experiment.py`, `..._integrity.py`).
Reruns will not reproduce the archived responses: requested routes name OpenRouter endpoints, not immutable
model weights, and serving providers can vary. Never commit an API key or `.env` file.

## Data Provenance

- Prices and market series come from Yahoo Finance; accounting and filings from SEC EDGAR. Observation and
  filing cutoffs exclude dates after each decision date, and forward returns never enter a prompt.
- Valuation prices and share counts are split-adjusted to a July 20, 2026 corporate-action cutoff; return,
  volatility, drawdown, and beta fields use an adjusted-close series. Neither is a historical download-vintage
  archive. Source URLs, cutoffs, and hashes accompany the frozen packets
  (`data/information_diversity_integrity_packets.json`, `data/information_diversity_us_price_basis.json`).
- Model registration uses the `created` timestamp in the July 21, 2026 OpenRouter catalog snapshot
  (`data/results/information_capacity_integrity_model_catalog.json`). Caches store the requested model,
  canonical slug, routed provider, generation ID, prompt and packet hashes, and seed metadata including
  recovery offsets.
- Nothing in this repository is investment advice. See the Disclaimer in the paper.

## License

- Code (`src/`): [MIT License](LICENSE).
- Data, results, figures, the reproducibility record, and the paper PDF (`data/`, `reproducibility/`, `paper/`, `The_Division_of_Research_ICAIF26.pdf`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA). Please cite the paper when reusing them.

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
