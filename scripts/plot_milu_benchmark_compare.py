#!/usr/bin/env python3
"""Create an interactive comparison dashboard for MILU benchmark JSON files.

The dashboard compares each benchmark run across the base and mHC sections and
summarizes:
- overall accuracy
- per-language accuracy
- latency distributions
- mHC minus base language deltas
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


LANGUAGES = [
    "English",
    "Bengali",
    "Hindi",
    "Tamil",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Odia",
]


@dataclass
class SeriesSummary:
    source_path: Path
    run_label: str
    section: str
    section_label: str
    model_label: str
    overall_accuracy: float
    language_accuracy: Dict[str, float]
    latencies_ms: List[float]
    sample_count: int
    correct_count: int


def _load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_run_label(payload: Dict, path: Path) -> str:
    config = payload.get("config", {})
    fewshot = config.get("num_fewshot")
    if fewshot is not None:
        return f"{fewshot}-shot"
    parent = path.parent.name
    if parent:
        return parent.replace("_", " ")
    return path.stem


def _summarize_section(payload: Dict, path: Path, section: str) -> SeriesSummary:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        raise ValueError(f"Missing section '{section}' in {path}")

    run_label = _infer_run_label(payload, path)
    model_label = section.capitalize()
    section_label = f"{run_label} / {model_label}"

    samples = section_payload.get("samples", []) or []
    latencies_ms = [float(sample.get("latency_ms", 0.0)) for sample in samples if sample.get("latency_ms") is not None]
    correct_count = sum(1 for sample in samples if sample.get("correct"))

    language_accuracy = {
        language: float(section_payload.get("language_accuracy", {}).get(language, 0.0))
        for language in LANGUAGES
    }

    return SeriesSummary(
        source_path=path,
        run_label=run_label,
        section=section,
        section_label=section_label,
        model_label=model_label,
        overall_accuracy=float(section_payload.get("overall_accuracy", 0.0)),
        language_accuracy=language_accuracy,
        latencies_ms=latencies_ms,
        sample_count=len(samples),
        correct_count=correct_count,
    )


def _iter_sections(payload: Dict) -> Iterable[str]:
    for section in ("base", "mhc"):
        if section in payload:
            yield section


def _build_color_map(series: Sequence[SeriesSummary]) -> Dict[str, str]:
    palette = {
        "base": "#3B82F6",
        "mhc": "#F97316",
    }
    return {item.section_label: palette.get(item.section, "#64748B") for item in series}


def _build_figure(series: Sequence[SeriesSummary], payloads: Sequence[Dict]) -> go.Figure:
    if not series:
        raise ValueError("No benchmark sections found")

    all_delta_rows = []
    for payload in payloads:
        source_path = Path(payload.get("__source_path__", "benchmark.json"))
        run_label = _infer_run_label(payload, source_path)
        base = payload.get("base")
        mhc = payload.get("mhc")
        if isinstance(base, dict) and isinstance(mhc, dict):
            base_lang = base.get("language_accuracy", {})
            mhc_lang = mhc.get("language_accuracy", {})
            all_delta_rows.append(
                {
                    "run_label": run_label,
                    **{
                        language: float(mhc_lang.get(language, 0.0)) - float(base_lang.get(language, 0.0))
                        for language in LANGUAGES
                    },
                }
            )

    colors = _build_color_map(series)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "heatmap"}]],
        subplot_titles=(
            "Overall Accuracy by Run",
            "Per-Language Accuracy",
            "Latency Distribution",
            "mHC - Base Language Delta",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    x_positions = [item.section_label for item in series]
    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=[item.overall_accuracy for item in series],
            marker_color=[colors[item.section_label] for item in series],
            text=[f"{item.overall_accuracy:.3f}" for item in series],
            textposition="outside",
            hovertemplate=(
                "Run=%{x}<br>Overall accuracy=%{y:.3f}<br>Correct=%{customdata[0]} / %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=[[item.correct_count, item.sample_count] for item in series],
            name="Overall accuracy",
        ),
        row=1,
        col=1,
    )

    for item in series:
        fig.add_trace(
            go.Bar(
                x=LANGUAGES,
                y=[item.language_accuracy[language] for language in LANGUAGES],
                name=item.section_label,
                marker_color=colors[item.section_label],
                opacity=0.9 if item.section == "mhc" else 0.65,
                hovertemplate=(
                    f"Run={item.section_label}<br>Language=%{{x}}<br>Accuracy=%{{y:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    for item in series:
        fig.add_trace(
            go.Box(
                y=item.latencies_ms,
                name=item.section_label,
                marker_color=colors[item.section_label],
                boxmean=True,
                hovertemplate=(
                    f"Run={item.section_label}<br>Latency=%{{y:.2f}} ms<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if all_delta_rows:
        heatmap_rows = [row["run_label"] for row in all_delta_rows]
        heatmap_values = [[row[language] for language in LANGUAGES] for row in all_delta_rows]
        hover_labels = [
            [f"Run={row['run_label']}<br>Language={language}<br>Delta={row[language]:+.3f}" for language in LANGUAGES]
            for row in all_delta_rows
        ]
        fig.add_trace(
            go.Heatmap(
                z=heatmap_values,
                x=LANGUAGES,
                y=heatmap_rows,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Delta"),
                customdata=hover_labels,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    else:
        fig.add_annotation(
            text="No paired base/mHC deltas were found in the provided files.",
            xref="x4 domain",
            yref="y4 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=2,
            col=2,
        )

    fig.update_layout(
        title=(
            "MILU benchmark comparison: interactive view of accuracy, latency, and mHC gains"
        ),
        template="plotly_white",
        barmode="group",
        boxmode="group",
        legend_title_text="Series",
        height=1200,
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_xaxes(title_text="Run", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", row=1, col=1)

    fig.update_xaxes(title_text="Language", tickangle=-25, row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", row=1, col=2)

    fig.update_xaxes(title_text="Series", row=2, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)

    fig.update_xaxes(title_text="Language", tickangle=-25, row=2, col=2)
    fig.update_yaxes(title_text="Run", row=2, col=2)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interactive MILU benchmark comparison dashboard")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more MILU benchmark JSON files produced by scripts/benchmark_milu.py",
    )
    parser.add_argument(
        "--output",
        default="output/diagnostics/milu_benchmark_compare.html",
        help="Output HTML file for the interactive dashboard",
    )
    args = parser.parse_args()

    payloads: List[Dict] = []
    series: List[SeriesSummary] = []
    for input_path in args.inputs:
        path = Path(input_path)
        payload = _load_payload(path)
        payload["__source_path__"] = str(path)
        payloads.append(payload)
        for section in _iter_sections(payload):
            series.append(_summarize_section(payload, path, section))

    if not series:
        raise SystemExit("No benchmark sections found in the provided files.")

    fig = _build_figure(series, payloads)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"Saved interactive dashboard to: {output_path}")


if __name__ == "__main__":
    main()