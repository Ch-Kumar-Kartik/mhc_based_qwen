#!/usr/bin/env python3
"""Analyze resumed training logs for repeated data windows.

This script parses train logs produced by train.py (pipe format), creates
diagnostic plots, and detects likely replayed ranges after checkpoint resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<msg>.*)$"
)
KV_RE = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_/]*)=(?P<v>[^\s]+)")
RESUME_RE = re.compile(r"checkpoint-(?P<ckpt>\d+).*global_step=(?P<step>\d+)")


@dataclass
class EventRow:
    timestamp: str
    event_type: str
    session_id: int
    message: str
    checkpoint_step: Optional[int] = None
    global_step: Optional[int] = None


@dataclass
class StepRow:
    timestamp: str
    unix_time: float
    session_id: int
    resumed_from_checkpoint: Optional[int]
    step: int
    total_steps: Optional[int]
    loss: Optional[float]
    avg_loss: Optional[float]
    lm_loss: Optional[float]
    avg_lm_loss: Optional[float]
    hc_loss: Optional[float]
    avg_hc_loss: Optional[float]
    lr: Optional[float]
    grad_norm: Optional[float]
    step_time: Optional[float]
    avg_step_time: Optional[float]
    tokens_per_s: Optional[float]
    samples_per_s: Optional[float]
    h_res_entropy: Optional[float]
    h_res_identity_dist: Optional[float]


def parse_timestamp(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_step_total(msg: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"step=(\d+)/(\d+)", msg)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_log(log_path: str) -> Tuple[List[StepRow], List[EventRow]]:
    steps: List[StepRow] = []
    events: List[EventRow] = []

    session_id = -1
    current_resume_ckpt: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue

            ts = m.group("ts")
            msg = m.group("msg")
            unix_time = parse_timestamp(ts)

            if "Starting training run" in msg:
                session_id += 1
                current_resume_ckpt = None
                events.append(
                    EventRow(
                        timestamp=ts,
                        event_type="start",
                        session_id=session_id,
                        message=msg,
                    )
                )
                continue

            if "Resumed optimizer/scheduler state" in msg:
                ckpt_step = None
                global_step = None
                rm = RESUME_RE.search(msg)
                if rm:
                    ckpt_step = int(rm.group("ckpt"))
                    global_step = int(rm.group("step"))
                    current_resume_ckpt = ckpt_step
                events.append(
                    EventRow(
                        timestamp=ts,
                        event_type="resume",
                        session_id=session_id,
                        message=msg,
                        checkpoint_step=ckpt_step,
                        global_step=global_step,
                    )
                )
                continue

            if "Checkpoint saved:" in msg:
                cm = re.search(r"checkpoint-(\d+)", msg)
                events.append(
                    EventRow(
                        timestamp=ts,
                        event_type="checkpoint_save",
                        session_id=session_id,
                        message=msg,
                        checkpoint_step=int(cm.group(1)) if cm else None,
                    )
                )
                continue

            step, total_steps = parse_step_total(msg)
            if step is None:
                continue

            kv = {km.group("k"): km.group("v") for km in KV_RE.finditer(msg)}

            steps.append(
                StepRow(
                    timestamp=ts,
                    unix_time=unix_time,
                    session_id=session_id,
                    resumed_from_checkpoint=current_resume_ckpt,
                    step=step,
                    total_steps=total_steps,
                    loss=to_float(kv.get("loss")),
                    avg_loss=to_float(kv.get("avg_loss")),
                    lm_loss=to_float(kv.get("lm_loss")),
                    avg_lm_loss=to_float(kv.get("avg_lm_loss")),
                    hc_loss=to_float(kv.get("hc_loss")),
                    avg_hc_loss=to_float(kv.get("avg_hc_loss")),
                    lr=to_float(kv.get("lr")),
                    grad_norm=to_float(kv.get("grad_norm")),
                    step_time=to_float(kv.get("step_time", "").replace("s", "")),
                    avg_step_time=to_float(kv.get("avg_step_time", "").replace("s", "")),
                    tokens_per_s=to_float(kv.get("tokens/s")),
                    samples_per_s=to_float(kv.get("samples/s")),
                    h_res_entropy=to_float(kv.get("h_res_entropy")),
                    h_res_identity_dist=to_float(kv.get("h_res_identity_dist")),
                )
            )

    return steps, events


def safe_absdiff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b)


def detect_duplicates(
    steps: List[StepRow],
    tolerance: float,
) -> Tuple[List[Dict], Dict[int, int], List[Dict]]:
    by_step: Dict[int, List[StepRow]] = {}
    for row in steps:
        by_step.setdefault(row.step, []).append(row)

    duplicate_rows: List[Dict] = []
    duplicate_score_by_step: Dict[int, int] = {}

    metrics = ["loss", "avg_loss", "lm_loss", "hc_loss", "lr", "grad_norm"]
    for step, entries in by_step.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]
                b = entries[j]
                if a.session_id == b.session_id:
                    continue

                diffs: List[float] = []
                per_metric: Dict[str, Optional[float]] = {}
                for metric in metrics:
                    d = safe_absdiff(getattr(a, metric), getattr(b, metric))
                    per_metric[metric] = d
                    if d is not None:
                        diffs.append(d)
                if not diffs:
                    continue

                max_abs_diff = max(diffs)
                mean_abs_diff = sum(diffs) / len(diffs)
                is_duplicate = max_abs_diff <= tolerance
                if is_duplicate:
                    duplicate_score_by_step[step] = duplicate_score_by_step.get(step, 0) + 1

                duplicate_rows.append(
                    {
                        "step": step,
                        "session_a": a.session_id,
                        "session_b": b.session_id,
                        "timestamp_a": a.timestamp,
                        "timestamp_b": b.timestamp,
                        "max_abs_diff": max_abs_diff,
                        "mean_abs_diff": mean_abs_diff,
                        "is_duplicate": is_duplicate,
                        **{f"diff_{k}": v for k, v in per_metric.items()},
                    }
                )

    duplicate_rows.sort(key=lambda x: (x["step"], x["session_a"], x["session_b"]))

    duplicate_ranges = build_duplicate_ranges(duplicate_rows)
    return duplicate_rows, duplicate_score_by_step, duplicate_ranges


def infer_step_delta(steps: List[StepRow]) -> int:
    unique_steps = sorted({r.step for r in steps})
    deltas = [b - a for a, b in zip(unique_steps, unique_steps[1:]) if b > a]
    if not deltas:
        return 1
    return max(1, int(min(deltas)))


def build_duplicate_ranges(duplicate_rows: List[Dict]) -> List[Dict]:
    pair_to_steps: Dict[Tuple[int, int], List[int]] = {}
    for row in duplicate_rows:
        if not row["is_duplicate"]:
            continue
        pair = (row["session_a"], row["session_b"])
        pair_to_steps.setdefault(pair, []).append(int(row["step"]))

    ranges: List[Dict] = []
    for (a, b), steps in pair_to_steps.items():
        steps = sorted(set(steps))
        if not steps:
            continue
        deltas = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        stride = min(deltas) if deltas else 1
        start = steps[0]
        prev = steps[0]
        count = 1
        for s in steps[1:]:
            if s - prev <= stride:
                prev = s
                count += 1
            else:
                ranges.append(
                    {
                        "session_a": a,
                        "session_b": b,
                        "start_step": start,
                        "end_step": prev,
                        "num_points": count,
                        "stride": stride,
                    }
                )
                start = s
                prev = s
                count = 1
        ranges.append(
            {
                "session_a": a,
                "session_b": b,
                "start_step": start,
                "end_step": prev,
                "num_points": count,
                "stride": stride,
            }
        )

    ranges.sort(key=lambda x: (x["session_a"], x["session_b"], x["start_step"]))
    return ranges


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def session_start_steps(steps: List[StepRow]) -> Dict[int, int]:
    starts: Dict[int, int] = {}
    for row in sorted(steps, key=lambda r: (r.session_id, r.step, r.unix_time)):
        if row.session_id not in starts:
            starts[row.session_id] = row.step
    return starts


def plot_outputs(
    outdir: str,
    steps: List[StepRow],
    events: List[EventRow],
    duplicate_score: Dict[int, int],
) -> str:
    import matplotlib.pyplot as plt

    steps_sorted = sorted(steps, key=lambda r: (r.step, r.unix_time))
    x = [r.step for r in steps_sorted]

    def series(name: str) -> List[float]:
        vals = []
        for r in steps_sorted:
            v = getattr(r, name)
            vals.append(float("nan") if v is None else float(v))
        return vals

    loss = series("loss")
    avg_loss = series("avg_loss")
    lm_loss = series("lm_loss")
    hc_loss = series("hc_loss")
    lr = series("lr")
    grad_norm = series("grad_norm")
    tokens_per_s = series("tokens_per_s")
    samples_per_s = series("samples_per_s")
    entropy = series("h_res_entropy")
    identity_dist = series("h_res_identity_dist")

    session_first_step = session_start_steps(steps)
    restart_steps = sorted(set(session_first_step.values()))

    resume_steps: List[int] = []
    for ev in events:
        if ev.event_type == "resume" and ev.global_step is not None:
            resume_steps.append(ev.global_step)
    resume_steps = sorted(set(resume_steps))

    dup_x = sorted(duplicate_score.keys())
    dup_y = [duplicate_score[s] for s in dup_x]

    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    ax = axes[0]
    ax.plot(x, loss, label="loss", linewidth=1.2)
    ax.plot(x, avg_loss, label="avg_loss", linewidth=1.2)
    ax.plot(x, lm_loss, label="lm_loss", linewidth=1.0, alpha=0.9)
    ax.set_ylabel("Loss")
    ax.set_title("Training loss metrics with restart/resume markers")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(x, hc_loss, label="hc_loss", linewidth=1.0)
    ax.plot(x, grad_norm, label="grad_norm", linewidth=1.0)
    ax.plot(x, lr, label="lr", linewidth=1.0)
    ax.set_ylabel("Aux / Optimization")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(x, tokens_per_s, label="tokens/s", linewidth=1.0)
    ax.plot(x, samples_per_s, label="samples/s", linewidth=1.0)
    ax.plot(x, entropy, label="h_res_entropy", linewidth=1.0)
    ax.plot(x, identity_dist, label="h_res_identity_dist", linewidth=1.0)
    ax.set_ylabel("Throughput / Stability")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[3]
    if dup_x:
        ax.bar(dup_x, dup_y, width=8, alpha=0.8, label="duplicate pair count")
    ax.set_ylabel("Duplicate score")
    ax.set_xlabel("Global step")
    ax.set_title("Repeated-step signal across sessions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    for axis in axes:
        for rs in restart_steps:
            axis.axvline(rs, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        for rs in resume_steps:
            axis.axvline(rs, color="red", linestyle=":", linewidth=1.0, alpha=0.5)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "resume_repetition_analysis.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def write_summary(
    path: str,
    steps: List[StepRow],
    events: List[EventRow],
    duplicate_rows: List[Dict],
    duplicate_ranges: List[Dict],
    tolerance: float,
) -> Dict:
    sessions = sorted({r.session_id for r in steps})
    unique_steps = sorted({r.step for r in steps})
    duplicate_true = [r for r in duplicate_rows if r["is_duplicate"]]
    duplicate_steps = sorted({int(r["step"]) for r in duplicate_true})

    resume_events = [e for e in events if e.event_type == "resume"]
    resume_ckpts = sorted({e.checkpoint_step for e in resume_events if e.checkpoint_step is not None})

    coverage = 0.0
    if unique_steps:
        coverage = 100.0 * (len(duplicate_steps) / len(unique_steps))

    report = {
        "num_sessions": len(sessions),
        "session_ids": sessions,
        "num_step_rows": len(steps),
        "num_unique_steps": len(unique_steps),
        "duplicate_tolerance": tolerance,
        "num_duplicate_pairs": len(duplicate_true),
        "num_duplicate_steps": len(duplicate_steps),
        "duplicate_step_coverage_percent": round(coverage, 4),
        "resume_checkpoint_steps": resume_ckpts,
        "duplicate_ranges": duplicate_ranges,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def print_console_summary(report: Dict) -> None:
    print("Resume repetition analysis complete")
    print(f"- sessions: {report['num_sessions']} ({report['session_ids']})")
    print(f"- step rows: {report['num_step_rows']}")
    print(f"- unique steps: {report['num_unique_steps']}")
    print(f"- duplicate pairs: {report['num_duplicate_pairs']}")
    print(f"- duplicate steps: {report['num_duplicate_steps']}")
    print(f"- duplicate step coverage: {report['duplicate_step_coverage_percent']:.2f}%")
    print(f"- resume checkpoints: {report['resume_checkpoint_steps']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics and detect repeated resumed windows from train.log"
    )
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument(
        "--outdir",
        default=os.path.join("output", "log_analysis"),
        help="Directory for plots and reports",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Absolute tolerance for step-level duplicate matching",
    )
    args = parser.parse_args()

    steps, events = parse_log(args.log)
    if not steps:
        raise RuntimeError("No step rows parsed from log. Check log format.")

    duplicate_rows, duplicate_score, duplicate_ranges = detect_duplicates(steps, args.tolerance)

    os.makedirs(args.outdir, exist_ok=True)

    parsed_csv = os.path.join(args.outdir, "parsed_steps.csv")
    save_csv(parsed_csv, [asdict(r) for r in steps])

    events_csv = os.path.join(args.outdir, "events.csv")
    save_csv(events_csv, [asdict(e) for e in events])

    duplicates_csv = os.path.join(args.outdir, "duplicate_step_pairs.csv")
    save_csv(duplicates_csv, duplicate_rows)

    ranges_json = os.path.join(args.outdir, "duplicate_ranges.json")
    with open(ranges_json, "w", encoding="utf-8") as f:
        json.dump(duplicate_ranges, f, indent=2)

    plot_path = plot_outputs(args.outdir, steps, events, duplicate_score)

    report_path = os.path.join(args.outdir, "repetition_report.json")
    report = write_summary(report_path, steps, events, duplicate_rows, duplicate_ranges, args.tolerance)
    report["plot_path"] = plot_path
    report["parsed_csv"] = parsed_csv
    report["events_csv"] = events_csv
    report["duplicates_csv"] = duplicates_csv

    print_console_summary(report)
    print(f"- plot: {plot_path}")
    print(f"- report: {report_path}")
    print(f"- parsed rows: {parsed_csv}")
    print(f"- duplicate pairs csv: {duplicates_csv}")


if __name__ == "__main__":
    main()
