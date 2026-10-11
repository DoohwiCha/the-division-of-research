"""End-to-end validation for the ICAIF-v4 submission."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = DATA / "results"
SUBMISSION = ROOT / "submissions" / "icaif-2026-v4-information-architecture"
DESIGNS = ("shared", "mixed", "coherent")
MARKETS = ("us", "europe", "crypto")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(script: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, str(ROOT / "src" / script)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        keys.update(key.strip() for key in group.split(","))
    return keys


def _allowlist(name: str) -> list[str]:
    return [
        line.strip()
        for line in (SUBMISSION / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_local_dependency_closure(entries: list[str]) -> None:
    allowed = set(entries)
    local = {path.stem: path for path in (ROOT / "src").glob("*.py")}
    for relative in entries:
        path = ROOT / relative
        if path.suffix != ".py":
            continue
        dependencies: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                dependencies.add(node.module.split(".", 1)[0])
        missing = sorted(
            str(local[name].relative_to(ROOT))
            for name in dependencies
            if name in local and str(local[name].relative_to(ROOT)) not in allowed
        )
        _require(not missing, f"artifact omits dependencies of {relative}: {missing}")


def validate_canonical_results() -> None:
    for script in (
        "cohort_information_capacity_audited_validate.py",
        "cohort_information_capacity_audited_curve_validate.py",
        "cohort_information_capacity_audited_longonly_validate.py",
        "validate_information_capacity_robustness.py",
    ):
        _run(script)

    primary = _load(RESULTS / "information_capacity_audited_curve_results.json")
    endpoint = _load(RESULTS / "information_capacity_audited_architecture_results.json")
    longonly = _load(RESULTS / "information_capacity_audited_longonly_results.json")
    repeatability = _load(RESULTS / "information_capacity_repeatability_results.json")
    risk_space = _load(RESULTS / "information_capacity_risk_space_results.json")
    block_result = _load(RESULTS / "information_capacity_block_bootstrap_results.json")
    prompt_examples = _load(RESULTS / "information_capacity_prompt_examples.json")
    _require(primary["n_dates"] == endpoint["n_dates"] == longonly["n_dates"] == 34,
             "canonical U.S. panels must contain 34 dates")
    _require(primary["integrity_correction_id"] == endpoint["integrity_correction_id"]
             == longonly["integrity_correction_id"], "correction waves do not match")
    _require(prompt_examples["schema_version"] == "information-capacity-prompt-examples-v1",
             "prompt-example schema changed")
    for label, example in prompt_examples["examples"].items():
        digest = hashlib.sha256(example["rendered_prompt"].encode("utf-8")).hexdigest()
        _require(digest == example["prompt_sha256"],
                 f"{label}: rendered prompt hash mismatch")
    for source in prompt_examples["sources"]["packet_files"]:
        _require(not Path(source["path"]).is_absolute(),
                 f"prompt-example source exposes an absolute path: {source['path']}")
    _require(not Path(prompt_examples["sources"]["assignment_plan"]["path"]).is_absolute(),
             "prompt-example assignment source exposes an absolute path")

    for k in range(2, 8):
        levels = primary["levels"][str(k)]
        _require(levels["shared"]["N_eff"]["mean"] < levels["mixed"]["N_eff"]["mean"],
                 f"K={k}: mixed access no longer adds effective books")
        _require(levels["shared"]["N_eff"]["mean"] < levels["coherent"]["N_eff"]["mean"],
                 f"K={k}: coherent access no longer adds effective books")
        for treatment in ("mixed", "coherent"):
            effect = primary["paired_comparisons"][str(k)][f"{treatment}_minus_shared"]["N_eff"]
            _require(effect["date_bootstrap_95"][0] > 0,
                     f"K={k}: {treatment} capacity interval includes zero")
        benchmark = primary["random_book_benchmarks"][str(k)]
        _require(benchmark["p2_5"] < benchmark["mean"] < benchmark["p97_5"],
                 f"K={k}: malformed random-book reference")

    endpoint_levels = primary["levels"]["7"]
    for design in DESIGNS:
        for metric in ("N_eff", "same_side_overlap", "surviving_gross", "convex_order_concentration"):
            observed = endpoint_levels[design][metric]["mean"]
            expected = endpoint["levels"][design][metric]["mean"]
            _require(np.isclose(observed, expected, atol=1e-12),
                     f"K=7 endpoint mismatch: {design} {metric}")
    _require(endpoint_levels["coherent"]["same_side_overlap"]["mean"]
             < endpoint_levels["mixed"]["same_side_overlap"]["mean"],
             "coherent endpoint no longer lowers overlap beyond mixed access")
    _require(endpoint_levels["coherent"]["convex_order_concentration"]["mean"]
             < endpoint_levels["mixed"]["convex_order_concentration"]["mean"],
             "coherent endpoint no longer lowers order concentration")

    levels = longonly["levels"]
    performance = longonly["operating_performance"]["levels"]
    allocation = longonly["operating_performance"]["allocation_decomposition"]
    for treatment in ("mixed", "coherent"):
        _require(levels[treatment]["N_eff"]["mean"] > levels["shared"]["N_eff"]["mean"],
                 f"long-only {treatment} capacity gain disappeared")
        _require(levels[treatment]["same_side_overlap"]["mean"]
                 < levels["shared"]["same_side_overlap"]["mean"],
                 f"long-only {treatment} overlap does not fall")
        _require(performance[treatment]["annualized_total_volatility"]["mean"]
                 < performance["shared"]["annualized_total_volatility"]["mean"],
                 f"long-only {treatment} volatility does not fall")
        _require(performance[treatment]["maximum_drawdown"]["mean"]
                 > performance["shared"]["maximum_drawdown"]["mean"],
                 f"long-only {treatment} drawdown does not improve")
        _require(performance[treatment]["mean_monthly_rebalance_turnover"]["mean"]
                 < performance["shared"]["mean_monthly_rebalance_turnover"]["mean"],
                 f"long-only {treatment} turnover does not fall")
        _require(performance[treatment]["sharpe_ratio"]["mean"]
                 > performance["shared"]["sharpe_ratio"]["mean"],
                 f"long-only {treatment} Sharpe does not improve")
        _require(performance[treatment]["annualized_total_return"]["mean"]
                 < performance["shared"]["annualized_total_return"]["mean"],
                 f"long-only {treatment} gross-return boundary changed")
        _require(performance[treatment]["annualized_total_return"]["mean"]
                 > performance["equal_weight"]["annualized_total_return"]["mean"],
                 f"long-only {treatment} no longer exceeds equal-weight gross return")
        _require(performance[treatment]["sharpe_ratio"]["mean"]
                 > performance["equal_weight"]["sharpe_ratio"]["mean"],
                 f"long-only {treatment} no longer exceeds equal-weight Sharpe")
        comparison = block_result["long_only"]["paired_comparisons"][
            f"{treatment}_minus_shared"
        ]
        _require(comparison["N_eff"]["block_3"]["moving_block_95"][0] > 0,
                 f"long-only {treatment} structural capacity interval includes zero")
        _require(comparison["same_side_overlap"]["block_3"]["moving_block_95"][1] < 0,
                 f"long-only {treatment} structural overlap interval includes zero")
        allocation_comparison = allocation["comparisons"][
            f"{treatment}_minus_shared"
        ]
        _require(allocation_comparison["team_variance_reduction_fraction"] > 0.20,
                 f"long-only {treatment} team-variance reduction weakened")
        _require(allocation_comparison["covariance_share_of_variance_reduction"] > 0.80,
                 f"long-only {treatment} covariance mechanism weakened")
        _require(
            allocation["levels"][treatment]["mean_pairwise_return_correlation"]["mean"]
            < allocation["levels"]["shared"]["mean_pairwise_return_correlation"]["mean"],
            f"long-only {treatment} seat-return correlation does not fall",
        )

    from icaif_v4_assets import _cross_results, number_macros, numeric_crosswalk

    cross = _cross_results()
    for market in MARKETS:
        validation = _load(RESULTS / f"information_capacity_{market}_data_validation.json")
        result = cross[market]
        _require(validation["passed"] and not validation["issues"],
                 f"{market}: source packet validation failed")
        _require(result["n_dates"] == 12, f"{market}: transfer panel must contain 12 dates")
        rows = {
            row["architecture"]: row
            for row in result["curve"] if row["team_size"] == 7
        }
        _require(set(rows) == set(DESIGNS), f"{market}: incomplete K=7 transfer result")
        _require(rows["shared"]["N_eff"] < rows["mixed"]["N_eff"] < rows["coherent"]["N_eff"],
                 f"{market}: K=7 effective-book ordering changed")
        for metric in ("same_side_overlap", "surviving_gross", "convex_order_concentration"):
            _require(rows["shared"][metric] > rows["mixed"][metric] > rows["coherent"][metric],
                     f"{market}: K=7 {metric} ordering changed")
        diagnostics = result["collection_diagnostics"]
        _require(diagnostics["collection_wave"] == "earlier harmonized transfer wave",
                 f"{market}: transfer collection boundary is missing")
        _require(diagnostics["generation_id_coverage"] == 0.0,
                 f"{market}: unexpected generation-ID coverage")
        stable = result["format_stable_route_sensitivity"]
        _require(stable["team_size"] == 3,
                 f"{market}: format-stable route sensitivity must use K=3")
        _require(stable["vendors"] == ["alibaba", "anthropic", "deepseek"],
                 f"{market}: format-stable route subset changed")
        for treatment in ("mixed", "coherent"):
            effect = stable["contrasts"][f"{treatment}_minus_shared"]["N_eff"]
            _require(effect["ci95"][0] > 0,
                     f"{market}: format-stable {treatment} interval includes zero")

    controls = endpoint["mechanism_controls"]["paired_comparisons"]
    for design in ("mixed", "coherent"):
        broadcast = f"{design}_short_packet_broadcast"
        _require(controls[f"{design}_minus_{broadcast}"]["N_eff"]["date_bootstrap_95"][0] > 0,
                 f"{design}: short-packet broadcast control failed")

    macros = number_macros(
        primary, cross, longonly, endpoint, repeatability, risk_space, block_result
    )
    for design, prefix in (("shared", "Shared"), ("mixed", "Mixed"), ("coherent", "Coherent")):
        expected = primary["levels"]["7"][design]["kappa_squared_correlation"]["mean"]
        _require(
            rf"\newcommand{{\{prefix}Kappa}}{{{expected:.3f}}}" in macros,
            f"{design}: generated kappa must use the mean date-level squared correlation",
        )
    _require(r"\newcommand{\NullKappa}{0.077}" in macros,
             "random-reference kappa must equal 1/(D-1) at D=14")
    risk_null_share = (
        100
        * risk_space["summary"]["shrinkage_0.00"]["levels"]["mixed"]["mean"]
        / risk_space["summary"]["shrinkage_0.00"]["levels"]["random_book_null"]["mean"]
    )
    _require(
        rf"\newcommand{{\RiskMixedNullShare}}{{{risk_null_share:.1f}\%}}" in macros,
        "risk-space random-reference share is stale",
    )
    retry_files = (
        "cohort_agents_information_capacity_audited_architecture.json",
        "cohort_agents_information_capacity_audited_curve.json",
        "cohort_agents_information_capacity_audited_longonly.json",
        "cohort_agents_information_capacity_repeatability.json",
    )
    retried_rows = sum(
        int(row.get("format_retry_count", 0)) > 0
        for name in retry_files
        for row in _load(DATA / name)
    )
    recovery_offsets = sum(
        len(_load(RESULTS / name)["seed_recovery"]["offsets"])
        for name in (
            "information_capacity_audited_curve_plan.json",
            "information_capacity_audited_longonly_plan.json",
        )
    )
    _require(retried_rows == 60, "U.S. collection retry disclosure is stale")
    _require(recovery_offsets == 6, "U.S. recovery-seed disclosure is stale")
    for market, prefix in (("us", "CrossUS"), ("europe", "CrossEU"), ("crypto", "CrossCrypto")):
        rows = {
            row["architecture"]: row
            for row in cross[market]["curve"]
            if int(row["team_size"]) == 3
        }
        expected = rows["mixed"]["N_eff"] - rows["shared"]["N_eff"]
        _require(
            rf"\newcommand{{\{prefix}KThreeMixedGain}}{{{expected:+.2f}}}" in macros,
            f"{market}: generated all-route K=3 comparison is stale",
        )
    crosswalk = numeric_crosswalk(
        primary, cross, longonly, endpoint, repeatability, risk_space, block_result
    )
    _require((SUBMISSION / "tables" / "numbers.tex").read_text(encoding="utf-8") == macros,
             "number macros are stale")
    _require((SUBMISSION / "NUMERIC_CROSSWALK.md").read_text(encoding="utf-8") == crosswalk,
             "numeric crosswalk is stale")


def validate_sources_and_pdfs() -> None:
    bib_text = (SUBMISSION / "refs.bib").read_text(encoding="utf-8")
    bib_keys = _bib_keys(bib_text)
    english = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    korean = (SUBMISSION / "main_ko.tex").read_text(encoding="utf-8")
    _require(not (bib_keys - _citation_keys(english)),
             f"uncited bibliography entries: {sorted(bib_keys - _citation_keys(english))}")

    deprecated = (
        "prespecified held-out sample", "Shapley", "role-label placebo", "ticker-mask placebo",
        "information_capacity_experiment_results", "information_capacity_longonly_results",
    )
    for name, text in (("main.tex", english), ("main_ko.tex", korean)):
        _require(not (_citation_keys(text) - bib_keys), f"{name} cites a missing key")
        _require("TODO" not in text and "TBD" not in text, f"{name} contains unfinished text")
        _require(not any(term in text for term in deprecated), f"{name} contains a legacy result")
        for include in re.findall(r"\\input\{([^}]+)\}", text):
            path = SUBMISSION / include
            _require(path.exists() or path.with_suffix(".tex").exists(), f"missing input: {include}")
        for include in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            path = SUBMISSION / include
            candidates = [path] if path.suffix else [path.with_suffix(".pdf"), path.with_suffix(".png")]
            _require(any(candidate.exists() for candidate in candidates), f"missing graphic: {include}")
        labels = set(re.findall(r"\\label\{([^}]+)\}", text))
        refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", text))
        _require(refs <= labels, f"{name} references missing labels: {sorted(refs - labels)}")

    for name, text in (("English", english), ("Korean", korean)):
        _require(len(re.findall(r"\\begin\{figure\*?\}", text)) == 5,
                f"{name} paper must contain five focused figures")
        _require(len(re.findall(r"\\begin\{table\*?\}", text)) == 4,
                 f"{name} paper must contain four focused tables")
    normalized = re.sub(r"\s+", " ", english)
    for phrase in (
        "Investment quality, compliance, and model validation remain prior gates",
        "All 34 dates form one pooled panel",
        "Every seat receives at least one asset-specific field",
        "first-order effect",
        "second-order improvement",
        "same smaller packet",
        "earlier collection round",
        "temperatures 0.0 and 0.7",
        "point-in-time covariance matrix",
        "three-month moving-block",
        "date means of $\\kappa_K$ and $N_{\\mathrm{eff}}$ do not commute",
        "Across the endpoint, curve, long-only, and repeatability collections",
        "SR 26-2 excludes generative and agentic AI from its formal scope",
        "At $K=7$, coherent domains add little breadth beyond mixed access",
        "Shared-access marginal capacity falls at every step",
        "(\\PostRegistrationStart{} onward)",
        "access separation reduces repeated funding",
        "At seven seats, economic domains retain similar breadth",
    ):
        _require(phrase in normalized, f"missing required manuscript statement: {phrase}")
    normalized_ko = re.sub(r"\s+", " ", korean)
    for phrase in (
        "향후 수익률은 지시문에 포함하지 않고",
        "외부 지식이나 누락 변수를 추론하지 못하게",
        "공통 접근은 하나의 고정된 일곱 모델 경로 팀",
        "끝점, 곡선, 롱온리와 반복 수집 전체에서 60개 응답",
        "날짜별 $\\kappa_K$의 평균과 $N_{\\mathrm{eff}}$의 평균",
        "생성형·에이전트형 AI를 공식 적용범위에서 제외",
        "$K=7$에서 분야별 분할은 혼합 분할과 비슷한 독립성",
        "\\PostRegistrationStart{}부터",
        "배정하는 일을 줄인다",
        "일곱 역할에서 경제적 분야로 정보를 묶으면",
    ):
        _require(phrase in normalized_ko, f"missing required Korean statement: {phrase}")

    forbidden_log = (
        "Overfull \\hbox", "Overfull \\vbox", "undefined references",
        "Undefined control sequence", "Citation `", "Reference `", "Fatal error",
    )
    for stem in ("main", "main_ko"):
        pdf = SUBMISSION / f"{stem}.pdf"
        log = SUBMISSION / f"{stem}.log"
        _require(pdf.exists() and log.exists(), f"{stem} build outputs are missing")
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE).group(1))
        _require(pages > 0, f"{stem}.pdf has no pages")
        if stem == "main":
            _require(pages <= 8, f"English submission has {pages} pages; limit is 8")
        log_text = log.read_text(encoding="utf-8", errors="replace")
        _require(not any(token in log_text for token in forbidden_log),
                 f"{stem}.log contains a layout, citation, or reference error")
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        _require(len(extracted) > 10_000 and "TBD" not in extracted,
                 f"{stem}.pdf text is incomplete")
        for page in range(1, pages + 1):
            page_text = subprocess.check_output(
                ["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"], text=True
            )
            _require(len(page_text.strip()) > 250, f"{stem}.pdf page {page} is nearly blank")


def validate_archives() -> None:
    secret_prefix = "sk-or-" + "v1-"
    local_homes = ("/" + "Users/", "/" + "home/")
    source_entries = _allowlist("SUBMISSION_FILES.txt")
    artifact_entries = _allowlist("ARTIFACT_FILES.txt")
    _validate_local_dependency_closure(artifact_entries)

    source_zip = SUBMISSION / "ICAIF_v4_sources.zip"
    artifact_zip = SUBMISSION / "ICAIF_v4_artifact.zip"
    _require(source_zip.exists() and artifact_zip.exists(), "submission archives are missing")
    with zipfile.ZipFile(source_zip) as archive:
        _require(archive.namelist() == source_entries, "source ZIP differs from its allowlist")
        for relative in source_entries:
            _require(
                archive.read(relative) == (SUBMISSION / relative).read_bytes(),
                f"source ZIP contains a stale copy of {relative}",
            )
    with zipfile.ZipFile(artifact_zip) as archive:
        _require(archive.namelist() == artifact_entries, "artifact ZIP differs from its allowlist")
        for relative in artifact_entries:
            _require(
                archive.read(relative) == (ROOT / relative).read_bytes(),
                f"artifact ZIP contains a stale copy of {relative}",
            )

    _require(not any("_ko" in name or name.endswith("main.pdf") for name in source_entries),
             "source ZIP contains a companion or compiled paper")
    allowed_masked_audit = "data/cohort_agents_information_capacity_integrity_macro_masked.json"
    stale_tokens = ("held_out", "_placebo", "attribution", "longonly_performance", "fig_common_beta")
    stale_entries = [
        name for name in artifact_entries
        if not name.startswith("src/")
        and any(token in name.lower() for token in stale_tokens)
    ]
    stale_entries.extend(
        name for name in artifact_entries
        if "_masked" in name.lower() and name != allowed_masked_audit
    )
    _require(not stale_entries, f"artifact allowlist contains legacy evidence: {stale_entries}")
    _require("src/icaif_v4_validate.py" not in artifact_entries,
             "development-only aggregate validator must not ship in the artifact")
    text_suffixes = {".py", ".tex", ".md", ".txt", ".csv", ".json", ".bib", ".bst", ".cls"}
    identity_tokens = (
        "doo" + "hwi",
        "dom" + "inic",
        "hj" + "yeo",
        "crowd" + "bench",
        "best-" + "paper",
        "mock " + "reviewer",
        "reviewer " + "objection",
        "manuscript " + "reviewer",
        "korean " + "companion",
        "internal " + "review",
        "em" + "nlp",
        "ap" + "jfs",
    )
    for relative in artifact_entries:
        path = ROOT / relative
        _require(path.exists(), f"artifact entry is missing: {relative}")
        _require(".git" not in relative and "partial" not in relative.lower(),
                 f"forbidden artifact path: {relative}")
        if path.suffix.lower() in text_suffixes:
            content = path.read_text(encoding="utf-8", errors="replace")
            _require(secret_prefix not in content, f"credential found in {relative}")
            _require(not any(home in content for home in local_homes),
                     f"local path found in {relative}")
            lowered = content.lower()
            _require(not any(token in lowered for token in identity_tokens),
                     f"identity or internal-workflow token found in {relative}")

    with tempfile.TemporaryDirectory(prefix="icaif-v4-cleanroom-") as directory:
        clean_root = Path(directory)
        with zipfile.ZipFile(artifact_zip) as archive:
            archive.extractall(clean_root)
        env = {**os.environ, "PYTHONPATH": str(clean_root / "src")}
        commands = [[
            "make",
            "-f",
            "submissions/icaif-2026-v4-information-architecture/Makefile.artifact",
            "reproduce",
            f"PYTHON={sys.executable}",
        ]]
        for command in commands:
            subprocess.run(
                command,
                cwd=clean_root,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )


def main() -> None:
    validate_canonical_results()
    print("loop 1/3 - corrected collections, results, and numeric assets: PASS")
    validate_sources_and_pdfs()
    print("loop 2/3 - English/Korean sources, citations, and PDFs: PASS")
    validate_archives()
    print("loop 3/3 - allowlisted source and artifact archives: PASS")
    print("ICAIF-v4 final validation: PASS")


if __name__ == "__main__":
    main()
