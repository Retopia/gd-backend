import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import msgpack


@dataclass(frozen=True)
class ExpectedEvent:
    idx: int
    frame: int
    t: float
    kind: str


@dataclass(frozen=True)
class VisualNote:
    start_t: float
    end_t: float
    is_hold: bool


@dataclass
class ResultRow:
    idx: int
    kind: str
    expected_t: float
    expected_frame: int
    actual_t: Optional[float]
    offset_ms: Optional[float]
    verdict: str


def load_gdr(path: str) -> Dict[str, Any]:
    """Load a .gdr macro file (msgpack format)"""
    raw = Path(path).read_bytes()
    obj = msgpack.unpackb(raw, raw=False)
    if not isinstance(obj, dict) or "inputs" not in obj:
        raise ValueError("Not a valid .gdr (expected msgpack dict with 'inputs')")
    return obj


def infer_fps(gdr: Dict[str, Any]) -> float:
    """Infer FPS from gdr data"""
    duration = float(gdr.get("duration") or 0.0)
    inputs = gdr.get("inputs") or []
    frames = [int(e.get("frame", 0)) for e in inputs if isinstance(e, dict) and "frame" in e]
    max_frame = max(frames) if frames else 0
    if duration <= 0.0 or max_frame <= 0:
        return 240.0
    fps = max_frame / duration
    if fps < 30:
        return 60.0
    if fps > 1000:
        return 240.0
    return float(fps)


def build_expected_events(gdr: Dict[str, Any], fps: float, btn: int, second_player: bool) -> List[ExpectedEvent]:
    """Build list of expected events from gdr data"""
    expected: List[ExpectedEvent] = []
    idx = 0
    for e in gdr["inputs"]:
        if not isinstance(e, dict):
            continue
        if int(e.get("btn", -1)) != btn:
            continue
        if bool(e.get("2p", False)) != second_player:
            continue
        frame = int(e.get("frame", 0))
        down = bool(e.get("down", False))
        t = frame / fps
        expected.append(ExpectedEvent(idx=idx, frame=frame, t=t, kind="down" if down else "up"))
        idx += 1
    expected.sort(key=lambda x: x.frame)
    return [ExpectedEvent(idx=i, frame=ev.frame, t=ev.t, kind=ev.kind) for i, ev in enumerate(expected)]


def build_visual_notes(expected: List[ExpectedEvent], hold_min_frames: int) -> List[VisualNote]:
    """Build visual notes from expected events"""
    notes: List[VisualNote] = []
    i = 0
    while i < len(expected):
        ev = expected[i]
        if ev.kind != "down":
            notes.append(VisualNote(start_t=ev.t, end_t=ev.t, is_hold=False))
            i += 1
            continue
        j = i + 1
        while j < len(expected) and expected[j].kind != "up":
            j += 1
        if j < len(expected):
            up = expected[j]
            is_hold = (up.frame - ev.frame) >= hold_min_frames
            notes.append(VisualNote(start_t=ev.t, end_t=up.t, is_hold=is_hold))
            i = j + 1
        else:
            notes.append(VisualNote(start_t=ev.t, end_t=ev.t, is_hold=False))
            i += 1
    return notes


def compute_compact_stats(results: List[ResultRow], expected_count: int) -> Dict[str, float]:
    """Compute statistics from results"""
    hits = [r for r in results if r.verdict == "hit" and r.offset_ms is not None and not math.isnan(r.expected_t)]
    misses = [r for r in results if r.verdict == "miss" and not math.isnan(r.expected_t)]
    extras = [r for r in results if math.isnan(r.expected_t)]
    hit_offsets = [r.offset_ms for r in hits if r.offset_ms is not None]

    if hit_offsets:
        mean = sum(hit_offsets) / len(hit_offsets)
        mae = sum(abs(x) for x in hit_offsets) / len(hit_offsets)
        worst = max(hit_offsets, key=lambda v: abs(v))
    else:
        mean = 0.0
        mae = 0.0
        worst = 0.0

    # Completion = fraction of expected events that were judged (hit or miss)
    judged = len([r for r in results if not math.isnan(r.expected_t)])
    completion = (judged / expected_count) if expected_count > 0 else 0.0

    return {
        "hits": float(len(hits)),
        "misses": float(len(misses)),
        "extras": float(len(extras)),
        "mean": float(mean),
        "mae": float(mae),
        "worst": float(worst),
        "completion": float(completion),
    }


def export_results_text(results: List[ResultRow], out_path: Optional[Path], meta: Dict[str, str]) -> str:
    """Export results to text file"""
    expected_misses = [r for r in results if r.verdict == "miss" and not math.isnan(r.expected_t)]
    extras = [r for r in results if math.isnan(r.expected_t)]
    hits = [r for r in results if r.verdict == "hit" and r.offset_ms is not None and not math.isnan(r.expected_t)]
    hit_offsets = [r.offset_ms for r in hits if r.offset_ms is not None]

    def fmt_ms(x: Optional[float]) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f} ms"

    if hit_offsets:
        mean = sum(hit_offsets) / len(hit_offsets)
        mae = sum(abs(x) for x in hit_offsets) / len(hit_offsets)
        worst = max(hit_offsets, key=lambda v: abs(v))
    else:
        mean = float("nan")
        mae = float("nan")
        worst = float("nan")

    lines: List[str] = []
    lines.append("GD Rhythm Trainer")
    lines.append("=" * 60)
    for k in ["macro_file", "fps", "window_ms", "expected_events", "inputs_captured", "generated_at", "target_x_frac"]:
        if k in meta:
            label = {
                "macro_file": "Macro file",
                "fps": "Detected FPS",
                "window_ms": "Hit window",
                "expected_events": "Expected events",
                "inputs_captured": "Inputs captured",
                "generated_at": "Generated at",
                "target_x_frac": "Target position",
            }[k]
            suffix = " ms" if k == "window_ms" else ""
            if k == "target_x_frac":
                lines.append(f"{label}: {float(meta[k]) * 100:.1f}% of screen width from left")
            else:
                lines.append(f"{label}: {meta[k]}{suffix}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"Hits: {len(hits)}")
    lines.append(f"Misses: {len(expected_misses)}")
    lines.append(f"Unexpected inputs: {len(extras)}")
    lines.append(f"Mean hit offset: {fmt_ms(mean)}   (negative=early, positive=late)")
    lines.append(f"Mean abs error: {('n/a' if math.isnan(mae) else f'{mae:.2f} ms')}")
    lines.append(f"Worst hit offset: {fmt_ms(worst)}")
    lines.append("")
    lines.append("Misses (detailed)")
    lines.append("-" * 60)
    if not expected_misses:
        lines.append("No misses!")
    else:
        lines.append("Format: \n[KIND] expected_time  frame    your_action           offset\n")
        for r in expected_misses:
            if r.actual_t is None:
                your_action = "NO INPUT"
                off = ""
            else:
                your_action = f"INPUT @ {r.actual_t:.6f}s"
                off = fmt_ms(r.offset_ms)
            lines.append(f"[{r.kind.upper():4}] t={r.expected_t:.6f}s  f={r.expected_frame:7d}  {your_action:20}  {off}")

    if extras:
        lines.append("")
        lines.append("Unexpected inputs")
        lines.append("-" * 60)
        lines.append("These are presses/releases that did not match any expected event (extra or too far off)")
        for r in extras[:250]:
            if r.actual_t is None:
                continue
            lines.append(f"[{r.kind.upper():4}] input_time={r.actual_t:.6f}s")
        if len(extras) > 250:
            lines.append(f"... ({len(extras) - 250} more omitted)")

    lines.append("")
    lines.append("How offsets work")
    lines.append("-" * 60)
    lines.append("Offset = (your input time - expected time)")
    lines.append("  Negative offset -> you were early")
    lines.append("  Positive offset -> you were late")
    lines.append("If an expected event shows NO INPUT, you never pressed/releases within the hit window for that event")
    lines.append("")

    content = "\n".join(lines) + "\n"
    if out_path:
        out_path.write_text(content, encoding="utf-8")
    return content
