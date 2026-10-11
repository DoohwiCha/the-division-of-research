"""Generate the corrected ICAIF-v4 figures, tables, and number macros."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from great_tables import GT
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT = ROOT / "submissions" / "icaif-2026-v4-information-architecture"
TABLES = OUT / "tables"

DESIGNS = ("shared", "mixed", "coherent")
INK = "#273238"
MID = "#6F777B"
LIGHT = "#D9DEE0"
COLORS = {"shared": INK, "mixed": "#4F7185", "coherent": "#557A66"}
MARKERS = {"shared": "o", "mixed": "s", "coherent": "^"}

BLUE = "#55758A"
GREEN = "#567861"
ARCH_COLOR = {"shared": MID, "mixed": BLUE, "coherent": GREEN}
DESIGN_LABELS_EN = {"shared": "Shared", "mixed": "Mixed", "coherent": "Coherent"}
DESIGN_LABELS_KO = {"shared": "공통 접근", "mixed": "혼합 분할", "coherent": "분야별 분할"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _latex_fragment(gt: GT) -> str:
    lines = []
    for line in str(gt.as_latex()).splitlines():
        stripped = line.strip()
        if stripped.startswith(r"\begin{table}") or stripped == r"\end{table}" or not stripped:
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines) + "\n"
    replacements = {
        r"N\_eff": r"$N_{\mathrm{eff}}$",
        "Overlap O": r"Overlap $O$",
        "Order Q": r"Order $Q$",
        "Concentration Q": r"Concentration $Q$",
        "External gross (M)": r"Proposed external gross (\$M)",
        "중첩 O": r"중첩 $O$",
        "주문 Q": r"주문 $Q$",
        "{[}": "[",
        "{]}": "]",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\\fontsize\{[^}]+\}\{[^}]+\}\\selectfont", r"\\footnotesize", text)


def _write_table(path: Path, gt: GT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_latex_fragment(gt), encoding="utf-8")


def _curve_arrays(
    result: dict,
    block_result: dict,
    design: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, low, high = [1.0], [1.0], [1.0]
    for k in range(2, 8):
        center = float(result["levels"][str(k)][design][metric]["mean"])
        interval = block_result["effective_book_curve"]["levels"][str(k)][design]["block_3"]
        left, right = map(float, interval["moving_block_95"])
        mean.append(center)
        low.append(left)
        high.append(right)
    return np.asarray(mean), np.asarray(low), np.asarray(high)


def _null_curve(result: dict) -> np.ndarray:
    values = [1.0]
    for k in range(2, 8):
        neff = result["levels"][str(k)]["shared"]["N_eff"]["mean"]
        ratio = result["levels"][str(k)]["shared"]["N_eff_over_null"]["mean"]
        values.append(float(neff / ratio))
    return np.asarray(values)


def _set_font(ko: bool) -> None:
    plt.rcParams.update({
        "font.family": "AppleGothic" if ko else "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _cross_results() -> dict[str, dict]:
    return {
        market: _load(RESULTS / f"information_capacity_{market}_results.json")
        for market in ("us", "europe", "crypto")
    }


def _k7(result: dict, architecture: str) -> dict:
    return next(
        row for row in result["curve"]
        if row["team_size"] == 7 and row["architecture"] == architecture
    )


def design_figure(
    primary: dict,
    path: Path,
    ko: bool,
    cropped_path: Path | None = None,
) -> None:
    """Show the corrected K=7 access matrices and their portfolio endpoints."""
    _set_font(ko)
    plan = _load(RESULTS / "information_capacity_audited_architecture_plan.json")
    coherent_rows = plan["assignment"]["coherent"]
    mixed_rows = plan["assignment"]["mixed"]
    domains = [row["name"] for row in coherent_rows]
    domain_fields = {row["name"]: row["field_ids"] for row in coherent_rows}
    fields = [field for domain in domains for field in domain_fields[domain]]
    field_index = {field: index for index, field in enumerate(fields)}

    matrices = {"shared": np.ones((7, len(fields)), dtype=int)}
    for architecture, rows in (("mixed", mixed_rows), ("coherent", coherent_rows)):
        matrix = np.zeros((7, len(fields)), dtype=int)
        for row in rows:
            for field in row["field_ids"]:
                matrix[int(row["seat"]) - 1, field_index[field]] = 1
        matrices[architecture] = matrix

    labels = DESIGN_LABELS_KO if ko else DESIGN_LABELS_EN
    titles = {
        key: f"{letter}. {labels[key]}"
        for letter, key in zip("ABC", ("shared", "mixed", "coherent"))
    }
    subtitles = {
        "shared": "모든 역할이 32개 변수를 읽음" if ko else "Every seat reads all 32 fields",
        "mixed": "각 변수는 한 역할에만, 분야는 혼합" if ko else "Each field once; domains mixed",
        "coherent": "각 역할이 경제적 조사 분야를 담당" if ko else "Each seat owns one research domain",
    }
    domain_labels = {
        "market_trend": "추세" if ko else "Trend",
        "market_liquidity": "유동" if ko else "Liq.",
        "valuation": "가치" if ko else "Val.",
        "company_quality": "건전" if ko else "Qual.",
        "corporate_events": "공시" if ko else "Events",
        "systematic_risk_and_rates": "시장·금리" if ko else "Sys./rates",
        "idiosyncratic_and_stress_risk": "고유" if ko else "Stress",
    }

    fig = plt.figure(figsize=(7.0, 2.46), facecolor="white")
    panel_x = (0.052, 0.372, 0.692)
    panel_w, panel_y, panel_h = 0.274, 0.305, 0.405
    fig.text(
        0.5, 0.965,
        (
            "고정: 동일 32개 시점 일치 변수 · 34개 월말 · 7개 모델 경로 · 동일 지시문과 포트폴리오 규칙"
            if ko else
            "FIXED: the same 32 point-in-time fields · 34 month-ends · 7 model routes · one prompt template and portfolio rule"
        ),
        ha="center", va="top", fontsize=7.4, color=INK, weight="bold",
    )
    sizes = [len(domain_fields[domain]) for domain in domains]
    boundaries = np.cumsum([0] + sizes)
    centers = [(left + right - 1) / 2 for left, right in zip(boundaries[:-1], boundaries[1:])]

    for x0, architecture in zip(panel_x, ("shared", "mixed", "coherent")):
        fig.text(x0, 0.855, titles[architecture], ha="left", va="center",
                 fontsize=8.0, weight="bold", color=INK)
        fig.text(x0, 0.792, subtitles[architecture], ha="left", va="center",
                 fontsize=6.25, color=MID)
        ax = fig.add_axes([x0, panel_y, panel_w, panel_h])
        ax.imshow(
            matrices[architecture], origin="upper", aspect="auto", vmin=0, vmax=1,
            cmap=ListedColormap(["#F1F3F3", ARCH_COLOR[architecture]]),
            interpolation="nearest",
        )
        ax.set_yticks(range(7), [f"S{i}" for i in range(1, 8)], fontsize=5.8)
        ax.set_xticks(centers, [domain_labels[domain] for domain in domains], fontsize=5.35)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                       length=0, pad=2)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, len(fields), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.32)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in boundaries[1:-1]:
            ax.axvline(boundary - 0.5, color="#9DA5A8", linewidth=0.70)
        for spine in ax.spines.values():
            spine.set_color("#A8AFB2")
            spine.set_linewidth(0.65)

        value = primary["levels"]["7"][architecture]
        fig.text(
            x0 + panel_w / 2, 0.235,
            rf"$N_{{\mathrm{{eff}}}}$ {value['N_eff']['mean']:.2f}   "
            rf"$O$ {value['same_side_overlap']['mean']:.3f}   "
            rf"$Q$ {value['convex_order_concentration']['mean']:.3f}",
            ha="center", va="center", fontsize=6.85,
            color=ARCH_COLOR[architecture], weight="bold",
        )

    shared = primary["levels"]["7"]["shared"]
    mixed = primary["levels"]["7"]["mixed"]
    coherent = primary["levels"]["7"]["coherent"]
    gain = mixed["N_eff"]["mean"] - shared["N_eff"]["mean"]
    overlap = coherent["same_side_overlap"]["mean"] - mixed["same_side_overlap"]["mean"]
    order = coherent["convex_order_concentration"]["mean"] - mixed["convex_order_concentration"]["mean"]
    fig.text(
        0.5, 0.125,
        (
            f"접근 분리: 유효 북 {gain:+.2f} · 분야 구성: 중첩률 {overlap:+.3f}, 주문 집중도 {order:+.3f}"
            if ko else
            f"Access separation: {gain:+.2f} effective books · Domain organization: overlap {overlap:+.3f}, order concentration {order:+.3f}"
        ),
        ha="center", va="center", fontsize=6.8, color=INK, weight="bold",
    )
    fig.text(
        0.5, 0.045,
        (
            "행은 분석 역할, 열은 변수다. 채운 칸은 해당 역할이 그 변수를 읽는다는 뜻이다."
            if ko else
            "Rows are research seats and columns are fields; a filled cell grants that seat access to the field."
        ),
        ha="center", va="center", fontsize=6.2, color=MID,
    )
    if cropped_path is not None:
        from matplotlib.transforms import Bbox

        cropped_path.parent.mkdir(parents=True, exist_ok=True)
        crop_viewport = Bbox.from_bounds(0.150375, 0.509722, 6.69925, 1.636069)
        fig.savefig(
            cropped_path,
            bbox_inches=crop_viewport,
            pad_inches=0,
            facecolor="white",
            transparent=False,
        )
    _save(fig, path)


def capacity_curve_figure(primary: dict, block_result: dict, path: Path, ko: bool) -> None:
    _set_font(ko)
    fig, ax = plt.subplots(figsize=(3.35, 2.62))
    x = np.arange(1, 8)
    ax.plot(
        x, _null_curve(primary), color=MID, linewidth=1.15, linestyle=(0, (4, 2)),
        label="무작위 북 기준" if ko else "Random-book benchmark",
    )
    labels = DESIGN_LABELS_KO if ko else {
        "shared": "Shared access", "mixed": "Mixed split", "coherent": "Coherent split",
    }
    offsets = {"shared": -0.12, "mixed": 0.18, "coherent": -0.24}
    for design in DESIGNS:
        mean, low, high = _curve_arrays(primary, block_result, design, "N_eff")
        ax.fill_between(x, low, high, color=COLORS[design], alpha=0.12, linewidth=0)
        ax.plot(
            x, mean, color=COLORS[design], marker=MARKERS[design], markersize=4.8,
            linewidth=1.75, label=labels[design],
        )
        ax.annotate(
            f"{mean[-1]:.2f}", xy=(7, mean[-1]), xytext=(7.31, mean[-1] + offsets[design]),
            ha="left", va="center", fontsize=6.8, color=COLORS[design], weight="bold",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.96},
            arrowprops={"arrowstyle": "-", "color": COLORS[design], "linewidth": 0.65},
        )
    ax.set_xlim(0.85, 7.78)
    ax.set_xticks(x)
    ax.set_xlabel("분석 역할 수 K" if ko else "Research seats K", fontsize=8.1)
    ax.set_ylabel(r"$N_{\mathrm{eff}}$" if ko else "Effective books", fontsize=8.1)
    ax.tick_params(axis="both", labelsize=7.1, length=2.5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False, fontsize=6.15, ncol=2, loc="lower left",
        bbox_to_anchor=(-0.01, 1.01), borderaxespad=0,
        columnspacing=0.55, handlelength=1.7, handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.80)
    _save(fig, path)


def capacity_time_figure(primary: dict, path: Path, ko: bool) -> None:
    """Trace the audited K=7 treatment levels over the 34 U.S. month-ends."""
    _set_font(ko)
    rows = sorted(
        (row for row in primary["per_date"] if int(row["team_size"]) == 7),
        key=lambda row: row["date"],
    )
    dates = pd.to_datetime([row["date"] for row in rows])
    labels = DESIGN_LABELS_KO if ko else {
        "shared": "Shared access", "mixed": "Mixed split", "coherent": "Coherent split",
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.40))
    for design in DESIGNS:
        values = np.asarray([float(row[design]["N_eff"]) for row in rows])
        ax.plot(
            dates, values, color=ARCH_COLOR[design], marker=MARKERS[design],
            markersize=2.8, linewidth=1.25, label=labels[design],
        )
    year_ticks = pd.to_datetime(["2023-07-31", "2024-07-31", "2025-07-31", "2026-04-30"])
    ax.set_xticks(year_ticks, ["2023.07", "2024.07", "2025.07", "2026.04"])
    ax.set_ylim(1.05, 4.80)
    ax.set_ylabel(r"$N_{\mathrm{eff}}$" if ko else "Effective books at K=7", fontsize=7.8)
    ax.set_xlabel("미국 월말" if ko else "U.S. month-end", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=6.5, length=2.2)
    ax.tick_params(axis="y", labelsize=6.9, length=2.2)
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False, fontsize=5.65, ncol=3, loc="lower left",
        bbox_to_anchor=(-0.01, 1.01), borderaxespad=0, columnspacing=0.55,
        handlelength=1.45, handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.84)
    _save(fig, path)


def crossmarket_figure(results: dict[str, dict], path: Path, ko: bool) -> None:
    _set_font(ko)
    markets = ("us", "europe", "crypto")
    market_labels = {
        "us": "미국\n(조화 표본)" if ko else "U.S.\n(harmonized)",
        "europe": "유럽" if ko else "Europe",
        "crypto": "암호자산" if ko else "Crypto",
    }
    labels = DESIGN_LABELS_KO if ko else {
        "shared": "Shared access", "mixed": "Mixed split", "coherent": "Coherent split",
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.42))
    for x, market in enumerate(markets):
        values = {design: _k7(results[market], design)["N_eff"] for design in DESIGNS}
        ax.plot([x, x], [min(values.values()), max(values.values())], color=LIGHT,
                linewidth=3.0, solid_capstyle="round", zorder=0)
        for design, marker in zip(DESIGNS, ("o", "s", "^")):
            value = values[design]
            ax.scatter(x, value, marker=marker, s=34, color=ARCH_COLOR[design],
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(f"{value:.2f}", (x, value), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6.35, color=ARCH_COLOR[design])
    ax.set_xticks(range(3), [market_labels[market] for market in markets])
    ax.set_xlim(-0.20, 2.30)
    ax.set_ylim(1.15, 3.85)
    ax.set_ylabel("K=7 유효 북 수" if ko else "Effective books at K=7", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2, length=0)
    ax.tick_params(axis="y", labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color=ARCH_COLOR[design],
                   label=labels[design])
        for design, marker in zip(DESIGNS, ("o", "s", "^"))
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.15, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), columnspacing=0.60, handletextpad=0.22)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.80)
    _save(fig, path)


def allocation_decomposition_figure(longonly: dict, path: Path, ko: bool) -> None:
    """Show how each split reduces long-only team variance from shared access."""
    _set_font(ko)
    allocation = longonly["operating_performance"]["allocation_decomposition"]
    levels = allocation["levels"]
    comparisons = allocation["comparisons"]
    shared_total = float(levels["shared"]["annualized_team_variance"]["mean"])
    shared_correlation = float(
        levels["shared"]["mean_pairwise_return_correlation"]["mean"]
    )
    rows = []
    for design in ("mixed", "coherent"):
        comparison = comparisons[f"{design}_minus_shared"]
        total = 100.0 * float(levels[design]["annualized_team_variance"]["mean"]) / shared_total
        decline = 100.0 - total
        covariance_share = 100.0 * float(
            comparison["covariance_share_of_variance_reduction"]
        )
        covariance_decline = decline * covariance_share / 100.0
        standalone_decline = decline - covariance_decline
        rows.append({
            "design": design,
            "total": total,
            "decline": decline,
            "covariance_share": covariance_share,
            "covariance_decline": covariance_decline,
            "standalone_decline": standalone_decline,
            "boundary": total + standalone_decline,
            "correlation": float(
                levels[design]["mean_pairwise_return_correlation"]["mean"]
            ),
        })

    fig, ax = plt.subplots(figsize=(6.75, 1.90), facecolor="white")
    y_positions = (1.0, 0.0)
    bar_height = 0.42
    standalone_color = "#B7B4AC"
    covariance_color = "#3F6B7C"
    for y, row in zip(y_positions, rows):
        ax.barh(
            y, row["standalone_decline"], left=row["total"], height=bar_height,
            color=standalone_color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.barh(
            y, row["covariance_decline"], left=row["boundary"], height=bar_height,
            color=covariance_color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.scatter(
            row["total"], y, s=29, marker="D", facecolor="white", edgecolor=INK,
            linewidth=0.9, zorder=5,
        )
        ax.text(
            row["total"] - 0.55, y, f"{row['total']:.1f}", ha="right", va="center",
            fontsize=8.0, color=INK, weight="bold",
        )
        ax.text(
            row["total"] - 0.55, y - 0.20, f"-{row['decline']:.1f}%",
            ha="right", va="top", fontsize=6.6, color=INK,
        )
        covariance_mid = row["boundary"] + row["covariance_decline"] / 2.0
        covariance_label = (
            f"감소분의 {row['covariance_share']:.1f}%"
            if ko else f"{row['covariance_share']:.1f}% of decline"
        )
        ax.text(
            covariance_mid, y, covariance_label, ha="center", va="center",
            fontsize=7.1, color="white", weight="bold",
        )
        ax.text(
            101.35, y + 0.05,
            f"{shared_correlation:.3f} → {row['correlation']:.3f}",
            ha="left", va="center", fontsize=7.15, color=INK, weight="bold",
        )

    ax.axvline(100, color=INK, linewidth=1.0, linestyle=(0, (2.0, 2.0)), zorder=2)
    ax.scatter(
        [100, 100], y_positions, s=19, marker="o", facecolor=INK,
        edgecolor=INK, zorder=5,
    )
    ax.text(
        100, 1.48, "공통 접근\n100.0" if ko else "Shared access\n100.0",
        ha="center", va="bottom", fontsize=7.3, color=INK, weight="bold",
        linespacing=1.0,
    )
    ax.text(
        107.0, 1.48, "역할 수익률 상관" if ko else "Seat-return correlation",
        ha="center", va="bottom", fontsize=6.4, color=INK,
    )
    row_labels = (
        ["혼합 분할", "분야별 분할"] if ko else ["Mixed split", "Coherent split"]
    )
    ax.set_yticks(y_positions, row_labels)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8.2)
        tick.set_fontweight("bold")
    ax.set_xlim(68.5, 114.0)
    ax.set_ylim(-0.48, 1.62)
    ax.set_xticks([70, 75, 80, 85, 90, 95, 100])
    ax.set_xlabel(
        "팀 분산 지수 (낮을수록 좋음; 공통 접근 = 100)" if ko else
        "Team-variance index (lower is better; shared access = 100)",
        fontsize=7.4,
    )
    ax.tick_params(axis="x", labelsize=6.8, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x", color=LIGHT, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.legend(
        handles=[
            Patch(
                facecolor=covariance_color, edgecolor="none",
                label="역할 간 공분산" if ko else "Cross-seat covariance",
            ),
            Patch(
                facecolor=standalone_color, edgecolor="none",
                label="개별 역할 분산" if ko else "Standalone-seat variance",
            ),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
        fontsize=6.8, handlelength=1.55, handleheight=0.8, columnspacing=1.6,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.80, bottom=0.26)
    _save(fig, path)


def protocol_table(ko: bool) -> GT:
    if ko:
        rows = [
            ["주표본", "미국 주식 · 34개 월말 · 14개 종목"],
            ["팀", "7개 모델 경로 · 분석 역할 K=1~7"],
            ["고정 정보", "시점 일치 32개 변수 · 모든 조건에서 같은 합집합"],
            ["접근 조건", "공통 접근 · 혼합 분할 · 분야별 분할"],
            ["추론 단위", "날짜 · 날짜 안에서 모든 모델/패킷 배정을 평균"],
        ]
        labels = ("설계 항목", "고정 명세")
    else:
        rows = [
            ["Primary sample", "U.S. equities · 34 month-ends · 14 assets"],
            ["Team", "7 model routes · research seats K=1 to K=7"],
            ["Fixed evidence", "32 point-in-time fields · same union in every condition"],
            ["Access conditions", "shared access · mixed split · coherent split"],
            ["Inference unit", "date · all model/packet assignments averaged within date"],
        ]
        labels = ("Design item", "Frozen specification")
    df = pd.DataFrame(rows, columns=["item", "specification"])
    return (
        GT(df).tab_options(table_font_size="7.6pt", data_row_padding="1.3px")
        .cols_label(**dict(zip(df.columns, labels)))
        .cols_align("left", ["item", "specification"])
    )


def primary_results_table(
    primary: dict,
    endpoint: dict,
    repeatability: dict,
    risk_space: dict,
    ko: bool,
) -> GT:
    levels = primary["levels"]["7"]

    def value(design: str, metric: str) -> float:
        return float(levels[design][metric]["mean"])

    metrics = (
        (
            "유효 독립 북 수 (높을수록 좋음)" if ko else
            "Effective books (higher)",
            lambda d: f"{value(d, 'N_eff'):.2f}",
        ),
        (
            "동일 방향 중첩률 (낮을수록 좋음)" if ko else "Same-side overlap (lower)",
            lambda d: f"{value(d, 'same_side_overlap'):.3f}",
        ),
        (
            "1억 달러 중 외부 주문 (낮을수록 좋음)" if ko else "Proposed external gross from $100M (lower)",
            lambda d: f"${100 * value(d, 'surviving_gross'):.1f}M",
        ),
        (
            "주문 집중도 (낮을수록 좋음)" if ko else "Order concentration (lower)",
            lambda d: f"{value(d, 'convex_order_concentration'):.3f}",
        ),
        (
            "공통 북 상관" if ko else "Common-book correlation",
            lambda d: f"{value(d, 'common_book_beta'):.3f}",
        ),
    )
    rows = [
        [label, *[formatter(design) for design in DESIGNS]]
        for label, formatter in metrics
    ]
    repeat_levels = repeatability["summary"]["temperature_0.7"]["persistent_levels"]
    risk_levels = risk_space["summary"]["shrinkage_0.00"]["levels"]
    risk_null = float(risk_levels["random_book_null"]["mean"])
    repeat_keys = {
        "shared": "repeat_shared_full",
        "mixed": "repeat_mixed_split",
        "coherent": "repeat_coherent_split",
    }
    within_route = endpoint["within_route_split_diagnostic"]
    complement_rows = [
        row for row in primary["per_date"]
        if row["sample"] == "complement" and row["team_size"] == 7
    ]
    complement_levels = {
        design: sum(row[design]["N_eff"] for row in complement_rows) / len(complement_rows)
        for design in DESIGNS
    }
    rows.extend([
        [
            f"보완 {len(complement_rows)}개 날짜: 유효 북 수" if ko else
            f"Complementary {len(complement_rows)} dates: effective books",
            *[f"{complement_levels[design]:.2f}" for design in DESIGNS],
        ],
        [
            "단일 경로 내 분할 검산" if ko else "Within-route split check",
            "N/A",
            f"{within_route['mixed']['levels']['N_eff']['mean']:.2f}",
            f"{within_route['coherent']['levels']['N_eff']['mean']:.2f}",
        ],
        [
            "반복 검산: 유효 북 수 (6개 날짜)" if ko else
            "Repeat check: effective books (6 dates)",
            *[f"{repeat_levels[repeat_keys[design]]['mean']:.2f}" for design in DESIGNS],
        ],
        [
            (
                f"위험공간 검산 (무작위 기준 {risk_null:.2f})"
                if ko else
                f"Risk-space check (random ref. {risk_null:.2f})"
            ),
            *[f"{risk_levels[design]['mean']:.2f}" for design in DESIGNS],
        ],
    ])
    df = pd.DataFrame(rows, columns=["metric", *DESIGNS])
    labels = (
        ("결과 및 검산", "공통 접근", "혼합 분할", "분야별 분할")
        if ko else
        ("Outcome / check", "Shared", "Mixed", "Coherent")
    )
    return (
        GT(df).tab_options(table_font_size="7.45pt", data_row_padding="1.2px")
        .cols_label(**dict(zip(df.columns, labels)))
        .cols_align("left", ["metric"])
        .cols_align("center", ["shared", "mixed", "coherent"])
    )


def market_results_table(results: dict[str, dict], ko: bool) -> GT:
    market_labels = {
        "us": "미국" if ko else "U.S.",
        "europe": "유럽" if ko else "Europe",
        "crypto": "암호자산" if ko else "Crypto",
    }
    rows = []
    for market in ("us", "europe", "crypto"):
        shared = _k7(results[market], "shared")
        coherent = _k7(results[market], "coherent")
        rows.append([
            market_labels[market],
            f"{shared['same_side_overlap']:.3f} → {coherent['same_side_overlap']:.3f}",
            f"{100 * shared['surviving_gross']:.1f} → {100 * coherent['surviving_gross']:.1f}",
            f"{shared['convex_order_concentration']:.3f} → {coherent['convex_order_concentration']:.3f}",
        ])
    labels = (
        ("시장", "중첩률 O", "$100M 중 외부 주문", "집중도 Q")
        if ko else
        ("Market", "Overlap O", "External gross (M)", "Concentration Q")
    )
    df = pd.DataFrame(rows, columns=["market", "overlap", "external", "order"])
    return (
        GT(df).tab_options(table_font_size="7.7pt", data_row_padding="1.35px")
        .cols_label(**dict(zip(df.columns, labels)))
        .cols_align("left", ["market"])
        .cols_align("center", ["overlap", "external", "order"])
    )


def longonly_results_table(longonly: dict, ko: bool) -> GT:
    levels = longonly["levels"]
    performance = longonly["operating_performance"]["levels"]

    def value(design: str, metric: str) -> float:
        return float(levels[design][metric]["mean"])

    def perf(design: str, metric: str) -> float:
        return float(performance[design][metric]["mean"])

    def structural(design: str, metric: str, formatter) -> str:
        if design == "equal_weight":
            reference = {
                "team_effective_holdings": 14.0,
                "team_top3_weight": 3.0 / 14.0,
                "team_max_weight": 1.0 / 14.0,
            }
            return "N/A" if metric not in reference else formatter(reference[metric])
        return formatter(value(design, metric))

    metrics = (
        ("유효 액티브 북 수" if ko else "Effective active books", lambda d: structural(d, "N_eff", lambda x: f"{x:.2f}")),
        ("액티브 동일 방향 중첩률" if ko else "Active same-side overlap", lambda d: structural(d, "same_side_overlap", lambda x: f"{x:.3f}")),
        ("실질 보유종목 수" if ko else "Effective holdings", lambda d: structural(d, "team_effective_holdings", lambda x: f"{x:.1f}")),
        ("상위 3종목 비중" if ko else "Top-three weight", lambda d: structural(d, "team_top3_weight", lambda x: f"{100 * x:.1f}%")),
        ("최대 단일종목 비중" if ko else "Largest stock weight", lambda d: structural(d, "team_max_weight", lambda x: f"{100 * x:.1f}%")),
        ("연환산 수익률" if ko else "Annualized return", lambda d: f"{100 * perf(d, 'annualized_total_return'):.2f}%"),
        ("연환산 변동성" if ko else "Annualized volatility", lambda d: f"{100 * perf(d, 'annualized_total_volatility'):.2f}%"),
        ("최대낙폭 규모" if ko else "Drawdown magnitude", lambda d: f"{100 * abs(perf(d, 'maximum_drawdown')):.2f}%"),
        ("샤프비율" if ko else "Sharpe ratio", lambda d: f"{perf(d, 'sharpe_ratio'):.3f}"),
        ("월평균 리밸런싱 회전율" if ko else "Monthly rebalance turnover", lambda d: f"{100 * perf(d, 'mean_monthly_rebalance_turnover'):.1f}%"),
    )
    columns = ("equal_weight", *DESIGNS)
    df = pd.DataFrame({
        "metric": [label for label, _ in metrics],
        **{
            design: [formatter(design) for _, formatter in metrics]
            for design in columns
        },
    })
    labels = (
        ("운용 결과", "동일비중", "공통", "혼합", "분야별")
        if ko else
        ("Portfolio outcome", "Equal wt.", "Shared", "Mixed", "Coherent")
    )
    return (
        GT(df).tab_options(table_font_size="7.0pt", data_row_padding="1.1px")
        .cols_label(**dict(zip(df.columns, labels)))
        .cols_align("left", ["metric"])
        .cols_align("center", list(columns))
    )


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def number_macros(
    primary: dict,
    cross: dict[str, dict],
    longonly: dict,
    endpoint: dict,
    repeatability: dict,
    risk_space: dict,
    block_result: dict,
) -> str:
    lines = ["% Generated by src/icaif_v4_assets.py; do not edit."]
    for design, prefix in (("shared", "Shared"), ("mixed", "Mixed"), ("coherent", "Coherent")):
        for k, word in ((2, "Two"), (4, "Four"), (7, "Seven")):
            lines.append(_macro(f"{prefix}K{word}", f"{primary['levels'][str(k)][design]['N_eff']['mean']:.2f}"))
        endpoint_level = primary["levels"]["7"][design]
        for metric, suffix, digits in (
            ("same_side_overlap", "Overlap", 3),
            ("surviving_gross", "Gross", 3),
            ("convex_order_concentration", "Order", 3),
            ("common_book_beta", "CommonBeta", 3),
        ):
            lines.append(_macro(
                f"{prefix}{suffix}", f"{endpoint_level[metric]['mean']:.{digits}f}"
            ))

    comparisons = primary["paired_comparisons"]["7"]
    for key, prefix in (
        ("mixed_minus_shared", "MixedShared"),
        ("coherent_minus_shared", "CoherentShared"),
        ("coherent_minus_mixed", "CoherentMixed"),
    ):
        for metric, suffix, digits in (
            ("N_eff", "Neff", 2),
            ("same_side_overlap", "Overlap", 3),
            ("surviving_gross", "Gross", 3),
            ("convex_order_concentration", "Order", 3),
        ):
            item = comparisons[key][metric]
            ci = block_result["position_space"][key][metric]["block_3"]["moving_block_95"]
            lines.extend([
                _macro(f"{prefix}{suffix}", f"{item['mean']:+.{digits}f}"),
                _macro(f"{prefix}{suffix}CI", f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"),
            ])
    shared = primary["levels"]["7"]["shared"]
    mixed = primary["levels"]["7"]["mixed"]
    coherent = primary["levels"]["7"]["coherent"]
    null_k7 = primary["random_book_benchmarks"]["7"]
    analytic_null_k7 = 7.0 / (1.0 + 6.0 / 13.0)
    analytic_null_kappa = 1.0 / 13.0

    lines.extend([
        _macro("NullKSeven", f"{null_k7['mean']:.2f}"),
        _macro("NullKSevenLow", f"{null_k7['p2_5']:.2f}"),
        _macro("NullKSevenHigh", f"{null_k7['p97_5']:.2f}"),
        _macro("AnalyticNullKSeven", f"{analytic_null_k7:.2f}"),
        _macro("SharedKappa", f"{shared['kappa_squared_correlation']['mean']:.3f}"),
        _macro("MixedKappa", f"{mixed['kappa_squared_correlation']['mean']:.3f}"),
        _macro("CoherentKappa", f"{coherent['kappa_squared_correlation']['mean']:.3f}"),
        _macro("NullKappa", f"{analytic_null_kappa:.3f}"),
        _macro("SharedNullShare", f"{100 * shared['N_eff_over_null']['mean']:.1f}\\%"),
        _macro("MixedNullShare", f"{100 * mixed['N_eff_over_null']['mean']:.1f}\\%"),
        _macro("CoherentNullShare", f"{100 * coherent['N_eff_over_null']['mean']:.1f}\\%"),
        _macro("SharedFirstMarginal", f"{primary['marginal_capacity']['shared']['1_to_2']['mean']:.2f}"),
        _macro("SharedLastMarginal", f"{primary['marginal_capacity']['shared']['6_to_7']['mean']:.2f}"),
        _macro("MixedLastMarginal", f"{primary['marginal_capacity']['mixed']['6_to_7']['mean']:.2f}"),
        _macro("CoherentLastMarginal", f"{primary['marginal_capacity']['coherent']['6_to_7']['mean']:.2f}"),
    ])
    k7_rows = [row for row in primary["per_date"] if int(row["team_size"]) == 7]
    for design, prefix in (("shared", "Shared"), ("mixed", "Mixed"), ("coherent", "Coherent")):
        values = [float(row[design]["N_eff"]) for row in k7_rows]
        lines.append(_macro(f"{prefix}NeffRange", f"{min(values):.2f}--{max(values):.2f}"))
    lines.append(_macro(
        "OverlapReductionPercent",
        f"{100 * (shared['same_side_overlap']['mean'] - coherent['same_side_overlap']['mean']) / shared['same_side_overlap']['mean']:.0f}\\%",
    ))
    lines.append(_macro(
        "MixedCapacityMultiple",
        f"{primary['levels']['7']['mixed']['N_eff']['mean'] / shared['N_eff']['mean']:.2f}",
    ))

    mechanism = endpoint["mechanism_controls"]
    audit = endpoint["design_audit"]["legacy_macro_ticker_sensitivity"]
    narrative_changes = [
        abs(
            mechanism["paired_comparisons"][
                f"{design}_without_explicit_ai_narratives_minus_{design}"
            ]["N_eff"]["mean"]
        )
        for design in ("mixed", "coherent")
    ]
    lines.extend([
        _macro(
            "ShortMixedBroadcastNeff",
            f"{mechanism['levels']['mixed_short_packet_broadcast']['N_eff']['mean']:.2f}",
        ),
        _macro(
            "ShortCoherentBroadcastNeff",
            f"{mechanism['levels']['coherent_short_packet_broadcast']['N_eff']['mean']:.2f}",
        ),
        _macro("LegacyMacroAuditPairs", str(audit["successful_matched_pairs"])),
        _macro("LegacyMacroNarratives", str(audit["visible_ticker_explicit_ai_narratives"])),
        _macro("LegacyMaskedNarratives", str(audit["masked_ticker_explicit_ai_narratives"])),
        _macro(
            "FinalExplicitNarratives",
            str(endpoint["collection_diagnostics"]["split_responses_with_explicit_ai_narrative"]),
        ),
        _macro("FinalNarrativeMaxNeffChange", f"{max(narrative_changes):.3f}"),
    ])

    complement = endpoint["sample_sensitivity"]["complement_dates"]
    post_registration = endpoint["sample_sensitivity"][
        "post_latest_route_registration_dates"
    ]
    within_route = endpoint["within_route_split_diagnostic"]
    mixed_tokens = endpoint["paired_comparisons"]["mixed_minus_shared"][
        "team_prompt_tokens"
    ]
    mixed_cost = endpoint["paired_comparisons"]["mixed_minus_shared"][
        "team_reported_cost"
    ]
    lines.extend([
        _macro("ComplementDates", str(complement["n_dates"])),
        _macro(
            "ComplementMixedGain",
            f"{complement['mixed_minus_shared_N_eff']['mean']:+.2f}",
        ),
        _macro(
            "ComplementCoherentGain",
            f"{complement['coherent_minus_shared_N_eff']['mean']:+.2f}",
        ),
        _macro(
            "PostRegistrationDates",
            str(post_registration["n_dates"]),
        ),
        _macro(
            "PostRegistrationStart",
            post_registration["dates"][0],
        ),
        _macro(
            "PostRegistrationMixedGain",
            f"{post_registration['mixed_minus_shared_N_eff']['mean']:+.2f}",
        ),
        _macro(
            "PostRegistrationCoherentGain",
            f"{post_registration['coherent_minus_shared_N_eff']['mean']:+.2f}",
        ),
        _macro(
            "WithinRouteMixedNeff",
            f"{within_route['mixed']['levels']['N_eff']['mean']:.2f}",
        ),
        _macro(
            "WithinRouteCoherentNeff",
            f"{within_route['coherent']['levels']['N_eff']['mean']:.2f}",
        ),
        _macro(
            "MixedPromptTokenReduction",
            f"{100 * (1 - mixed_tokens['treatment_mean'] / mixed_tokens['control_mean']):.0f}\\%",
        ),
        _macro(
            "MixedReportedCostReduction",
            f"{100 * (1 - mixed_cost['treatment_mean'] / mixed_cost['control_mean']):.0f}\\%",
        ),
    ])

    repeat_summary = repeatability["summary"]
    for temperature, suffix in (("temperature_0.0", "TZero"), ("temperature_0.7", "TSeven")):
        summary = repeat_summary[temperature]
        persistent = summary["persistent_levels"]
        contrast = summary["contrasts"]["repeat_mixed_split_minus_shared"]["persistent_repeat_average"]
        lines.extend([
            _macro(f"RepeatShared{suffix}", f"{persistent['repeat_shared_full']['mean']:.2f}"),
            _macro(f"RepeatSmall{suffix}", f"{persistent['repeat_same_small']['mean']:.2f}"),
            _macro(f"RepeatMixed{suffix}", f"{persistent['repeat_mixed_split']['mean']:.2f}"),
            _macro(f"RepeatCoherent{suffix}", f"{persistent['repeat_coherent_split']['mean']:.2f}"),
            _macro(f"RepeatMixedShared{suffix}", f"{contrast['mean']:+.2f}"),
            _macro(
                f"RepeatMixedShared{suffix}CI",
                f"[{contrast['date_bootstrap_95'][0]:+.2f}, {contrast['date_bootstrap_95'][1]:+.2f}]",
            ),
        ])
    repeat_exception = {
        (row["temperature"], row["condition"]): row["N_eff"]
        for row in repeatability["per_persistent_date"]
        if row["date"] == "2024-04-30"
    }
    lines.extend([
        _macro(
            "RepeatSmallException",
            f"{repeat_exception[(0.7, 'repeat_same_small')]:.2f}",
        ),
        _macro(
            "RepeatMixedExceptionRange",
            f"{repeat_exception[(0.0, 'repeat_mixed_split')]:.2f}--"
            f"{repeat_exception[(0.7, 'repeat_mixed_split')]:.2f}",
        ),
    ])

    risk_summary = risk_space["summary"]["shrinkage_0.00"]
    risk_levels = risk_summary["levels"]
    for design, prefix in (("shared", "RiskShared"), ("mixed", "RiskMixed"), ("coherent", "RiskCoherent")):
        lines.append(_macro(f"{prefix}Neff", f"{risk_levels[design]['mean']:.2f}"))
    risk_contrast = risk_summary["paired_comparisons"]["mixed_minus_shared"]
    risk_block = block_result["risk_space"]["paired_comparisons"]["mixed_minus_shared"]["N_eff"]["block_3"]
    lines.extend([
        _macro("RiskNullNeff", f"{risk_levels['random_book_null']['mean']:.2f}"),
        _macro(
            "RiskMixedNullShare",
            f"{100 * risk_levels['mixed']['mean'] / risk_levels['random_book_null']['mean']:.1f}\\%",
        ),
        _macro("RiskMixedSharedNeff", f"{risk_contrast['mean']:+.2f}"),
        _macro(
            "RiskMixedSharedNeffCI",
            f"[{risk_block['moving_block_95'][0]:+.2f}, {risk_block['moving_block_95'][1]:+.2f}]",
        ),
        _macro("RiskMixedPositiveDates", str(risk_contrast["dates_positive"])),
    ])

    for market, market_prefix in (("us", "CrossUS"), ("europe", "CrossEU"), ("crypto", "CrossCrypto")):
        for design, design_prefix in (("shared", "Shared"), ("mixed", "Mixed"), ("coherent", "Coherent")):
            row = _k7(cross[market], design)
            lines.extend([
                _macro(f"{market_prefix}{design_prefix}Neff", f"{row['N_eff']:.2f}"),
                _macro(f"{market_prefix}{design_prefix}Overlap", f"{row['same_side_overlap']:.3f}"),
                _macro(f"{market_prefix}{design_prefix}External", f"{100 * row['surviving_gross']:.1f}"),
                _macro(f"{market_prefix}{design_prefix}Order", f"{row['convex_order_concentration']:.3f}"),
            ])
        stable = cross[market]["format_stable_route_sensitivity"]
        k3_rows = {
            row["architecture"]: row
            for row in cross[market]["curve"]
            if int(row["team_size"]) == 3
        }
        lines.extend([
            _macro(
                f"{market_prefix}StableMixedGain",
                f"{stable['contrasts']['mixed_minus_shared']['N_eff']['mean']:+.2f}",
            ),
            _macro(
                f"{market_prefix}KThreeMixedGain",
                f"{k3_rows['mixed']['N_eff'] - k3_rows['shared']['N_eff']:+.2f}",
            ),
            _macro(
                f"{market_prefix}RetryRate",
                f"{100 * cross[market]['collection_diagnostics']['format_retry_rate']:.0f}\\%",
            ),
        ])

    long_levels = longonly["levels"]
    long_performance = longonly["operating_performance"]["levels"]
    for design, prefix in (("shared", "LongShared"), ("mixed", "LongMixed"), ("coherent", "LongCoherent")):
        level = long_levels[design]
        performance = long_performance[design]
        lines.extend([
            _macro(f"{prefix}Neff", f"{level['N_eff']['mean']:.2f}"),
            _macro(f"{prefix}Overlap", f"{level['same_side_overlap']['mean']:.3f}"),
            _macro(f"{prefix}CommonBeta", f"{level['common_book_beta']['mean']:.3f}"),
            _macro(f"{prefix}Holdings", f"{level['team_effective_holdings']['mean']:.1f}"),
            _macro(f"{prefix}TopThree", f"{100 * level['team_top3_weight']['mean']:.1f}"),
            _macro(f"{prefix}MaxWeight", f"{100 * level['team_max_weight']['mean']:.1f}"),
            _macro(f"{prefix}Return", f"{100 * performance['annualized_total_return']['mean']:.2f}"),
            _macro(f"{prefix}Volatility", f"{100 * performance['annualized_total_volatility']['mean']:.2f}"),
            _macro(f"{prefix}Drawdown", f"{100 * performance['maximum_drawdown']['mean']:.2f}"),
            _macro(f"{prefix}Sharpe", f"{performance['sharpe_ratio']['mean']:.3f}"),
            _macro(f"{prefix}Turnover", f"{100 * performance['mean_monthly_rebalance_turnover']['mean']:.1f}"),
        ])
    equal_weight = long_performance["equal_weight"]
    lines.extend([
        _macro("LongEqualWeightHoldings", "14.0"),
        _macro("LongEqualWeightTopThree", f"{100 * 3 / 14:.1f}"),
        _macro("LongEqualWeightMaxWeight", f"{100 / 14:.1f}"),
        _macro("LongEqualWeightReturn", f"{100 * equal_weight['annualized_total_return']['mean']:.2f}"),
        _macro("LongEqualWeightVolatility", f"{100 * equal_weight['annualized_total_volatility']['mean']:.2f}"),
        _macro("LongEqualWeightDrawdown", f"{100 * equal_weight['maximum_drawdown']['mean']:.2f}"),
        _macro("LongEqualWeightSharpe", f"{equal_weight['sharpe_ratio']['mean']:.3f}"),
        _macro("LongEqualWeightTurnover", f"{100 * equal_weight['mean_monthly_rebalance_turnover']['mean']:.1f}"),
    ])
    shared_perf = long_performance["shared"]
    coherent_perf = long_performance["coherent"]
    allocation = longonly["operating_performance"]["allocation_decomposition"]
    allocation_levels = allocation["levels"]
    allocation_comparisons = allocation["comparisons"]
    lines.extend([
        _macro("LongCoherentReturnGap", f"{100 * (coherent_perf['annualized_total_return']['mean'] - shared_perf['annualized_total_return']['mean']):+.2f}"),
        _macro("LongCoherentVolatilityDrop", f"{100 * (shared_perf['annualized_total_volatility']['mean'] - coherent_perf['annualized_total_volatility']['mean']):.2f}"),
        _macro("LongCoherentDrawdownImprovement", f"{100 * (abs(shared_perf['maximum_drawdown']['mean']) - abs(coherent_perf['maximum_drawdown']['mean'])):.2f}"),
        _macro("LongCoherentTurnoverReduction", f"{100 * (shared_perf['mean_monthly_rebalance_turnover']['mean'] - coherent_perf['mean_monthly_rebalance_turnover']['mean']) / shared_perf['mean_monthly_rebalance_turnover']['mean']:.0f}\\%"),
        _macro("AllocationSharedCorrelation", f"{allocation_levels['shared']['mean_pairwise_return_correlation']['mean']:.3f}"),
        _macro("AllocationMixedCorrelation", f"{allocation_levels['mixed']['mean_pairwise_return_correlation']['mean']:.3f}"),
        _macro("AllocationCoherentCorrelation", f"{allocation_levels['coherent']['mean_pairwise_return_correlation']['mean']:.3f}"),
        _macro("AllocationMixedVarianceReduction", f"{100 * allocation_comparisons['mixed_minus_shared']['team_variance_reduction_fraction']:.1f}\\%"),
        _macro("AllocationCoherentVarianceReduction", f"{100 * allocation_comparisons['coherent_minus_shared']['team_variance_reduction_fraction']:.1f}\\%"),
        _macro("AllocationMixedCovarianceShare", f"{100 * allocation_comparisons['mixed_minus_shared']['covariance_share_of_variance_reduction']:.1f}\\%"),
        _macro("AllocationCoherentCovarianceShare", f"{100 * allocation_comparisons['coherent_minus_shared']['covariance_share_of_variance_reduction']:.1f}\\%"),
    ])
    for design, prefix in (("mixed", "LongMixedShared"), ("coherent", "LongCoherentShared")):
        estimates = block_result["long_only"]["paired_comparisons"][
            f"{design}_minus_shared"
        ]
        for metric, suffix, digits, scale in (
            ("N_eff", "Neff", 2, 1.0),
            ("same_side_overlap", "Overlap", 3, 1.0),
            ("team_forward_return", "MonthlyReturn", 2, 100.0),
        ):
            estimate = estimates[metric]["block_3"]
            lines.extend([
                _macro(f"{prefix}{suffix}", f"{scale * estimate['mean']:+.{digits}f}"),
                _macro(
                    f"{prefix}{suffix}CI",
                    f"[{scale * estimate['moving_block_95'][0]:+.{digits}f}, "
                    f"{scale * estimate['moving_block_95'][1]:+.{digits}f}]",
                ),
            ])
    return "\n".join(lines) + "\n"


def numeric_crosswalk(
    primary: dict,
    cross: dict[str, dict],
    longonly: dict,
    endpoint: dict,
    repeatability: dict,
    risk_space: dict,
    block_result: dict,
) -> str:
    def primary_values(metric: str, digits: int) -> str:
        return " / ".join(
            f"{primary['levels']['7'][design][metric]['mean']:.{digits}f}"
            for design in DESIGNS
        )

    def cross_values(market: str, metric: str, digits: int) -> str:
        return " / ".join(f"{_k7(cross[market], design)[metric]:.{digits}f}" for design in DESIGNS)

    def cross_k3_mixed_gain(market: str) -> float:
        rows = {
            row["architecture"]: row
            for row in cross[market]["curve"]
            if int(row["team_size"]) == 3
        }
        return rows["mixed"]["N_eff"] - rows["shared"]["N_eff"]

    def long_values(container: dict, metric: str, digits: int, scale: float = 1.0, suffix: str = "") -> str:
        return " / ".join(
            f"{scale * container[design][metric]['mean']:.{digits}f}{suffix}"
            for design in DESIGNS
        )

    long_levels = longonly["levels"]
    long_performance = longonly["operating_performance"]["levels"]
    rows = [
        ("Corrected primary K=7 effective books", primary_values("N_eff", 2),
         "`information_capacity_audited_curve_results.json -> levels.7.*.N_eff.mean`"),
        (
            "K=7 random-book reference (analytic / simulated mean / 95% range)",
            f"{7 / (1 + 6 / 13):.2f} / "
            f"{primary['random_book_benchmarks']['7']['mean']:.2f} / "
            f"[{primary['random_book_benchmarks']['7']['p2_5']:.2f}, "
            f"{primary['random_book_benchmarks']['7']['p97_5']:.2f}]",
            "`information_capacity_audited_curve_results.json -> random_book_benchmarks.7`",
        ),
        (
            "K=7 mean squared off-diagonal correlation kappa (shared / mixed / coherent / random reference)",
            " / ".join([
                *[
                    f"{primary['levels']['7'][design]['kappa_squared_correlation']['mean']:.3f}"
                    for design in DESIGNS
                ],
                f"{1 / 13:.3f}",
            ]),
            "`information_capacity_audited_curve_results.json -> "
            "levels.7.*.kappa_squared_correlation.mean`; analytic random reference is 1/(D-1)",
        ),
        ("Corrected primary K=7 overlap", primary_values("same_side_overlap", 3),
         "`information_capacity_audited_curve_results.json -> levels.7.*.same_side_overlap.mean`"),
        ("Corrected primary K=7 order concentration", primary_values("convex_order_concentration", 3),
         "`information_capacity_audited_curve_results.json -> levels.7.*.convex_order_concentration.mean`"),
        ("Corrected primary K=7 common-book correlation", primary_values("common_book_beta", 3),
         "`information_capacity_audited_curve_results.json -> levels.7.*.common_book_beta.mean`"),
        (
            "Short-packet broadcast effective books (mixed / coherent packet sets)",
            " / ".join(
                f"{endpoint['mechanism_controls']['levels'][design]['N_eff']['mean']:.2f}"
                for design in (
                    "mixed_short_packet_broadcast",
                    "coherent_short_packet_broadcast",
                )
            ),
            "`information_capacity_audited_architecture_results.json -> mechanism_controls.levels`",
        ),
        (
            "Legacy macro audit explicit narratives (visible / masked)",
            f"{endpoint['design_audit']['legacy_macro_ticker_sensitivity']['visible_ticker_explicit_ai_narratives']} / "
            f"{endpoint['design_audit']['legacy_macro_ticker_sensitivity']['masked_ticker_explicit_ai_narratives']}",
            "`information_capacity_audited_architecture_results.json -> design_audit`",
        ),
        (
            "Complement-date capacity gain (mixed / coherent minus shared)",
            f"{endpoint['sample_sensitivity']['complement_dates']['mixed_minus_shared_N_eff']['mean']:+.2f} / "
            f"{endpoint['sample_sensitivity']['complement_dates']['coherent_minus_shared_N_eff']['mean']:+.2f}",
            "`information_capacity_audited_architecture_results.json -> sample_sensitivity.complement_dates`",
        ),
        (
            "Post-latest-route-registration gain (mixed / coherent minus shared)",
            f"{endpoint['sample_sensitivity']['post_latest_route_registration_dates']['mixed_minus_shared_N_eff']['mean']:+.2f} / "
            f"{endpoint['sample_sensitivity']['post_latest_route_registration_dates']['coherent_minus_shared_N_eff']['mean']:+.2f}",
            "`information_capacity_audited_architecture_results.json -> sample_sensitivity.post_latest_route_registration_dates`",
        ),
        (
            "Within-route split effective books (mixed / coherent)",
            f"{endpoint['within_route_split_diagnostic']['mixed']['levels']['N_eff']['mean']:.2f} / "
            f"{endpoint['within_route_split_diagnostic']['coherent']['levels']['N_eff']['mean']:.2f}",
            "`information_capacity_audited_architecture_results.json -> within_route_split_diagnostic`",
        ),
        (
            "Mixed-access prompt tokens / reported request cost (shared -> mixed)",
            f"{endpoint['paired_comparisons']['mixed_minus_shared']['team_prompt_tokens']['control_mean']:.0f} -> "
            f"{endpoint['paired_comparisons']['mixed_minus_shared']['team_prompt_tokens']['treatment_mean']:.0f} / "
            f"{endpoint['paired_comparisons']['mixed_minus_shared']['team_reported_cost']['control_mean']:.5f} -> "
            f"{endpoint['paired_comparisons']['mixed_minus_shared']['team_reported_cost']['treatment_mean']:.5f}",
            "`information_capacity_audited_architecture_results.json -> paired_comparisons.mixed_minus_shared`",
        ),
        (
            "Repeat-mean effective books at T=0.7 (shared / mixed / coherent)",
            " / ".join(
                f"{repeatability['summary']['temperature_0.7']['persistent_levels'][key]['mean']:.2f}"
                for key in ("repeat_shared_full", "repeat_mixed_split", "repeat_coherent_split")
            ),
            "`information_capacity_repeatability_results.json -> summary.temperature_0.7.persistent_levels`",
        ),
        (
            "PIT risk-space effective books (shared / mixed / coherent)",
            " / ".join(
                f"{risk_space['summary']['shrinkage_0.00']['levels'][design]['mean']:.2f}"
                for design in DESIGNS
            ),
            "`information_capacity_risk_space_results.json -> summary.shrinkage_0.00.levels`",
        ),
        (
            "Three-month moving-block mixed-minus-shared CI (position / risk)",
            f"[{block_result['position_space']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][0]:+.2f}, "
            f"{block_result['position_space']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][1]:+.2f}] / "
            f"[{block_result['risk_space']['paired_comparisons']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][0]:+.2f}, "
            f"{block_result['risk_space']['paired_comparisons']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][1]:+.2f}]",
            "`information_capacity_block_bootstrap_results.json`",
        ),
        (
            "Three-month moving-block long-only monthly-return gap (mixed / coherent minus shared)",
            f"{100 * block_result['long_only']['paired_comparisons']['mixed_minus_shared']['team_forward_return']['block_3']['mean']:+.2f} pp / "
            f"{100 * block_result['long_only']['paired_comparisons']['coherent_minus_shared']['team_forward_return']['block_3']['mean']:+.2f} pp",
            "`information_capacity_block_bootstrap_results.json -> long_only.paired_comparisons`",
        ),
        (
            "Three-month moving-block long-only structural CI (mixed minus shared: N_eff / overlap)",
            f"[{block_result['long_only']['paired_comparisons']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][0]:+.2f}, "
            f"{block_result['long_only']['paired_comparisons']['mixed_minus_shared']['N_eff']['block_3']['moving_block_95'][1]:+.2f}] / "
            f"[{block_result['long_only']['paired_comparisons']['mixed_minus_shared']['same_side_overlap']['block_3']['moving_block_95'][0]:+.3f}, "
            f"{block_result['long_only']['paired_comparisons']['mixed_minus_shared']['same_side_overlap']['block_3']['moving_block_95'][1]:+.3f}]",
            "`information_capacity_block_bootstrap_results.json -> long_only.paired_comparisons.mixed_minus_shared`",
        ),
    ]
    for market, label in (("us", "Harmonized U.S."), ("europe", "Harmonized Europe"), ("crypto", "Harmonized crypto")):
        rows.extend([
            (f"{label} K=7 effective books", cross_values(market, "N_eff", 2),
             f"`information_capacity_{market}_results.json -> curve[K=7]`"),
            (f"{label} K=7 overlap", cross_values(market, "same_side_overlap", 3),
             f"`information_capacity_{market}_results.json -> curve[K=7]`"),
            (
                f"{label} format-stable / all-route K=3 mixed-minus-shared gain",
                f"{cross[market]['format_stable_route_sensitivity']['contrasts']['mixed_minus_shared']['N_eff']['mean']:+.2f} / "
                f"{cross_k3_mixed_gain(market):+.2f}",
                f"`information_capacity_{market}_results.json -> "
                "format_stable_route_sensitivity / curve[K=3]`",
            ),
        ])
    rows.extend([
        ("Corrected long-only effective active books", long_values(long_levels, "N_eff", 2),
         "`information_capacity_audited_longonly_results.json -> levels.*.N_eff.mean`"),
        ("Corrected long-only active overlap", long_values(long_levels, "same_side_overlap", 3),
         "`information_capacity_audited_longonly_results.json -> levels.*.same_side_overlap.mean`"),
        ("Corrected long-only top-three weight", long_values(long_levels, "team_top3_weight", 1, 100, "%"),
         "`information_capacity_audited_longonly_results.json -> levels.*.team_top3_weight.mean`"),
        ("Corrected long-only annualized gross return", long_values(long_performance, "annualized_total_return", 2, 100, "%"),
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.*`"),
        ("Corrected long-only annualized volatility", long_values(long_performance, "annualized_total_volatility", 2, 100, "%"),
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.*`"),
        ("Corrected long-only maximum drawdown", long_values(long_performance, "maximum_drawdown", 2, 100, "%"),
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.*`"),
        ("Corrected long-only Sharpe", long_values(long_performance, "sharpe_ratio", 3),
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.*`"),
        ("Corrected long-only monthly rebalance turnover", long_values(long_performance, "mean_monthly_rebalance_turnover", 1, 100, "%"),
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.*`"),
        (
            "Long-only mean pairwise seat-return correlation",
            " / ".join(
                f"{longonly['operating_performance']['allocation_decomposition']['levels'][design]['mean_pairwise_return_correlation']['mean']:.3f}"
                for design in DESIGNS
            ),
            "`information_capacity_audited_longonly_results.json -> "
            "operating_performance.allocation_decomposition.levels.*`",
        ),
        (
            "Long-only variance reduction / covariance share (mixed / coherent vs shared)",
            " / ".join(
                f"{100 * longonly['operating_performance']['allocation_decomposition']['comparisons'][f'{design}_minus_shared']['team_variance_reduction_fraction']:.1f}%"
                f" / {100 * longonly['operating_performance']['allocation_decomposition']['comparisons'][f'{design}_minus_shared']['covariance_share_of_variance_reduction']:.1f}%"
                for design in ("mixed", "coherent")
            ),
            "`information_capacity_audited_longonly_results.json -> "
            "operating_performance.allocation_decomposition.comparisons`",
        ),
        ("Equal-weight reference return / volatility / Sharpe / turnover",
         f"{100 * long_performance['equal_weight']['annualized_total_return']['mean']:.2f}% / "
         f"{100 * long_performance['equal_weight']['annualized_total_volatility']['mean']:.2f}% / "
         f"{long_performance['equal_weight']['sharpe_ratio']['mean']:.3f} / "
         f"{100 * long_performance['equal_weight']['mean_monthly_rebalance_turnover']['mean']:.1f}%",
         "`information_capacity_audited_longonly_results.json -> operating_performance.levels.equal_weight`"),
    ])
    lines = [
        "# Numeric Crosswalk", "",
        "Values separated by slashes follow shared access, mixed splitting, and coherent splitting.", "",
        "| Manuscript result | Reported value | Frozen source |",
        "|---|---:|---|",
        *[f"| {label} | {value} | {source} |" for label, value, source in rows],
        "", "All manuscript values are emitted from the corrected result objects.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    primary = _load(RESULTS / "information_capacity_audited_curve_results.json")
    endpoint = _load(RESULTS / "information_capacity_audited_architecture_results.json")
    cross = _cross_results()
    longonly = _load(RESULTS / "information_capacity_audited_longonly_results.json")
    repeatability = _load(RESULTS / "information_capacity_repeatability_results.json")
    risk_space = _load(RESULTS / "information_capacity_risk_space_results.json")
    block_result = _load(RESULTS / "information_capacity_block_bootstrap_results.json")

    combined = {
        "schema_version": "information-capacity-crossmarket-combined-v1",
        "markets": {
            market: {
                "n_dates": result["n_dates"],
                "curve": result["curve"],
                "k7_contrasts": result["k7_contrasts"],
                "collection_diagnostics": result["collection_diagnostics"],
                "format_stable_route_sensitivity": result["format_stable_route_sensitivity"],
            }
            for market, result in cross.items()
        },
    }
    (RESULTS / "information_capacity_crossmarket_combined.json").write_text(
        json.dumps(combined, indent=2, allow_nan=False), encoding="utf-8"
    )

    for ko, suffix in ((False, ""), (True, "_ko")):
        design_figure(
            primary,
            OUT / f"fig_design{suffix}.pdf",
            ko,
            OUT / "fig_design_cropped.pdf" if not ko else None,
        )
        capacity_curve_figure(primary, block_result, OUT / f"fig_capacity_curve{suffix}.pdf", ko)
        capacity_time_figure(primary, OUT / f"fig_capacity_time{suffix}.pdf", ko)
        crossmarket_figure(cross, OUT / f"fig_crossmarket{suffix}.pdf", ko)
        allocation_decomposition_figure(
            longonly, OUT / f"fig_allocation_decomposition{suffix}.pdf", ko
        )
        _write_table(TABLES / f"protocol{suffix}.tex", protocol_table(ko))
        _write_table(
            TABLES / f"primary_results{suffix}.tex",
            primary_results_table(primary, endpoint, repeatability, risk_space, ko),
        )
        _write_table(TABLES / f"market_results{suffix}.tex", market_results_table(cross, ko))
        _write_table(TABLES / f"longonly_results{suffix}.tex", longonly_results_table(longonly, ko))
    (TABLES / "numbers.tex").write_text(
        number_macros(primary, cross, longonly, endpoint, repeatability, risk_space, block_result),
        encoding="utf-8",
    )
    (OUT / "NUMERIC_CROSSWALK.md").write_text(
        numeric_crosswalk(primary, cross, longonly, endpoint, repeatability, risk_space, block_result),
        encoding="utf-8",
    )
    print(f"generated corrected ICAIF-v4 assets in {OUT}")


if __name__ == "__main__":
    main()
