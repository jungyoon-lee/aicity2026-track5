#!/usr/bin/env python3
"""Build a WTS all-view Cosmos3 SFT dataset.

This creates one train split from WTS train+val, with overhead_view and
vehicle_view rows. It keeps each view as a separate sample row; paired WTS
views share captions/phase labels through the prompt, not through multi-video
model inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare import base as BASE

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_PHASES = {
    "0": "pre-recognition phase",
    "1": "recognition phase",
    "2": "judgement phase",
    "3": "action phase",
    "4": "avoidance or outcome phase",
}

VIEW_TEXT = {
    "overhead_view": "overhead fixed-camera view",
    "vehicle_view": "vehicle/front view",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wts-root", default=str(PROJECT_ROOT / "datasets/aicity_track5/data/WTS"))
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "data/wts"
        ),
    )
    parser.add_argument("--source-splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--views", nargs="+", default=["overhead_view", "vehicle_view"])
    parser.add_argument("--labels", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--output-split", default="train")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=165)
    parser.add_argument("--context-frames", type=int, default=45)
    parser.add_argument(
        "--context-latent-options",
        default="0,1,5,8,10,12",
        help=(
            "Comma-separated latent context lengths. Each option creates windows "
            "ending immediately before --context-frames so short contexts use the "
            "latest observed frames before the target boundary."
        ),
    )
    parser.add_argument(
        "--context-latent-weights",
        default="0.01,0.04,0.10,0.20,0.25,0.40",
        help=(
            "Comma-separated sampling weights aligned with --context-latent-options. "
            "Weights are written into t2w_windows as window_sampling_weight."
        ),
    )
    parser.add_argument("--min-future-frames", type=int, default=48)
    parser.add_argument("--max-future-frames", type=int, default=120)
    parser.add_argument("--window-stride-frames", type=int, default=4)
    parser.add_argument(
        "--min-loader-frames",
        type=int,
        default=61,
        help="Mirror Cosmos3 SFT metadata loader's minimum kept window length.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--label0-fraction",
        type=float,
        default=0.08,
        help="Approximate final fraction of label0 rows. Set <0 to keep all label0.",
    )
    parser.add_argument("--max-caption-files-per-view-split", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--legacy-v1",
        action="store_true",
        help=(
            "Reproduce the start_frame=0 window and prompt contract used by "
            "C3Super1680. The default remains the context-aligned v2 contract."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def caption_files(wts_root: Path, split: str, view: str) -> list[Path]:
    return sorted((wts_root / "caption" / split).glob(f"**/{view}/*_caption.json"))


def video_names_for_caption(caption: dict[str, Any], video_dir: Path, view: str) -> list[str]:
    if view == "overhead_view":
        names = caption.get("overhead_videos") or []
    elif view == "vehicle_view":
        value = caption.get("vehicle_view")
        names = [value] if value else []
    else:
        names = []
    if not names:
        names = [path.name for path in sorted(video_dir.glob("*.mp4"))]
    return [str(name) for name in names]


def build_prompt(row: dict[str, Any], *, legacy_v1: bool = False) -> str:
    phase = LABEL_PHASES.get(str(row["label"]), "target future phase")
    view_text = VIEW_TEXT.get(row["view"], row["view"])
    pre_recognition_note = ""
    if str(row["label"]) == "0":
        pre_recognition_note = (
            " This phase is before the main pedestrian-vehicle risk interaction "
            "is recognized; use it as scene, approach, and domain context."
        )
    if legacy_v1:
        visual_context_rule = (
            "Visual context rule: Use the observed frames for camera viewpoint, road layout, lighting, "
            "object identity, and current object positions. Keep the future consistent with the specified view.\n\n"
        )
    else:
        visual_context_rule = (
            "Visual context rule: If visual context frames are provided, use them for camera viewpoint, "
            "road layout, lighting, object identity, and current object positions. If no visual context "
            "frames are provided, synthesize a plausible traffic scene from the captions. Keep the future "
            "consistent with the specified view.\n\n"
        )
    return (
        "Task: Generate the future of an AI City WTS traffic video.\n\n"
        f"Source dataset: WTS.\n"
        f"Source split: {row['source_split']}.\n"
        f"View: {view_text}.\n\n"
        f"{visual_context_rule}"
        f"Target phase: {phase}.{pre_recognition_note}\n\n"
        "Use both official descriptions below to determine the future pedestrian-vehicle interaction.\n\n"
        "Pedestrian description:\n"
        f"{row['pedestrian_caption']}\n\n"
        "Vehicle description:\n"
        f"{row['vehicle_caption']}\n\n"
        "Generation rule: The generated future must be consistent with both descriptions. "
        "The pedestrian motion, vehicle motion, risk level, stopping or yielding behavior, "
        "and outcome should reflect the combined pedestrian-vehicle interaction."
    )


def ffmpeg_extract(
    *,
    src_video: Path,
    dst_video: Path,
    clip_start: float,
    clip_seconds: float,
    pad_start_seconds: float,
    fps: int,
    frames: int,
    width: int,
    height: int,
    crf: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if dst_video.exists() and not overwrite:
        return
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    scale_height = int(round(width * 9 / 16))
    crop_y = max(0, (scale_height - height) // 2)
    vf_parts = [
        f"fps={fps}",
        f"scale={width}:{scale_height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={width}:{height}:0:{crop_y}",
        "setsar=1",
    ]
    if pad_start_seconds > 1e-6:
        vf_parts.append(f"tpad=start_mode=clone:start_duration={pad_start_seconds:.6f}")
    vf_parts.extend(
        [
            f"tpad=stop_mode=clone:stop_duration={clip_seconds:.6f}",
            f"trim=start_frame=0:end_frame={frames}",
            f"setpts=N/({fps}*TB)",
        ]
    )
    vf = ",".join(vf_parts)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{clip_start:.6f}",
        "-i",
        str(src_video),
        "-t",
        f"{clip_seconds:.6f}",
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-an",
        str(dst_video),
    ]
    if dry_run:
        print("DRYRUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def pick_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: (row["view"], row["source_split"], row["scenario"], row["basename"]))
    if count == 1:
        return [ordered[len(ordered) // 2]]
    selected = []
    for i in range(count):
        idx = round(i * (len(ordered) - 1) / (count - 1))
        selected.append(ordered[idx])
    seen = set()
    unique = []
    for row in selected:
        if row["basename"] not in seen:
            seen.add(row["basename"])
            unique.append(row)
    for row in ordered:
        if len(unique) >= count:
            break
        if row["basename"] not in seen:
            seen.add(row["basename"])
            unique.append(row)
    return unique


def downsample_label0(tasks: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fraction < 0:
        return tasks, {"policy": "keep_all_label0"}
    non0 = [task for task in tasks if str(task["label"]) != "0"]
    label0 = [task for task in tasks if str(task["label"]) == "0"]
    if not label0:
        return tasks, {"policy": "no_label0_found"}
    keep_total = int(round(len(non0) * fraction / max(1e-6, 1.0 - fraction)))
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in label0:
        by_view[str(row["view"])].append(row)
    non0_by_view = Counter(str(row["view"]) for row in non0)
    keep_rows: list[dict[str, Any]] = []
    for view, rows in sorted(by_view.items()):
        view_share = non0_by_view[view] / max(1, len(non0))
        keep_rows.extend(pick_evenly(rows, int(round(keep_total * view_share))))
    if len(keep_rows) < keep_total:
        remaining = [row for row in label0 if row["basename"] not in {item["basename"] for item in keep_rows}]
        keep_rows.extend(pick_evenly(remaining, keep_total - len(keep_rows)))
    keep_set = {row["basename"] for row in keep_rows[:keep_total]}
    merged = non0 + [row for row in label0 if row["basename"] in keep_set]
    return merged, {
        "policy": "downsample_label0",
        "requested_fraction": fraction,
        "non0_count": len(non0),
        "label0_source_count": len(label0),
        "label0_keep_count": len(keep_set),
        "label0_source_by_view": dict(Counter(row["view"] for row in label0)),
        "label0_keep_by_view": dict(Counter(row["view"] for row in label0 if row["basename"] in keep_set)),
    }


def parse_int_csv(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_float_csv(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    if any(value < 0 for value in values):
        raise ValueError("weights must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [value / total for value in values]


def latent_context_to_pixel_frames(latent_frames: int, temporal_compression_factor: int = 4) -> int:
    if latent_frames < 0:
        raise ValueError(f"latent context must be >= 0, got {latent_frames}")
    if latent_frames == 0:
        return 0
    return (latent_frames - 1) * temporal_compression_factor + 1


def build_context_options(args: argparse.Namespace) -> list[dict[str, int | float]]:
    options: list[dict[str, int | float]] = []
    seen: set[int] = set()
    latent_options = parse_int_csv(args.context_latent_options)
    latent_weights = parse_float_csv(args.context_latent_weights)
    if len(latent_options) != len(latent_weights):
        raise ValueError("--context-latent-options and --context-latent-weights must have the same length")
    for latent_frames, weight in zip(latent_options, latent_weights, strict=True):
        if latent_frames in seen:
            continue
        seen.add(latent_frames)
        pixel_frames = latent_context_to_pixel_frames(latent_frames)
        if pixel_frames > args.context_frames:
            raise ValueError(
                f"latent context {latent_frames} maps to {pixel_frames} pixel frames, "
                f"which exceeds --context-frames={args.context_frames}"
            )
        if pixel_frames > 0 and not BASE.valid_vae_frame_count(pixel_frames):
            raise ValueError(f"context pixel frames must be 1 mod 4, got {pixel_frames}")
        options.append({"latent_frames": latent_frames, "pixel_frames": pixel_frames, "weight": weight})
    return options


def build_window_specs(args: argparse.Namespace) -> list[dict[str, int | float]]:
    if args.min_future_frames <= 0:
        raise ValueError("--min-future-frames must be positive")
    if args.max_future_frames < args.min_future_frames:
        raise ValueError("--max-future-frames must be >= --min-future-frames")
    if args.window_stride_frames <= 0:
        raise ValueError("--window-stride-frames must be positive")

    if args.legacy_v1:
        specs: list[dict[str, int | float]] = []
        for future_frames in range(args.min_future_frames, args.max_future_frames + 1, args.window_stride_frames):
            total_frames = args.context_frames + future_frames
            if total_frames > args.frames or not BASE.valid_vae_frame_count(total_frames):
                continue
            specs.append(
                {
                    "start_frame": 0,
                    "end_frame": total_frames - 1,
                    "total_frames": total_frames,
                    "future_frames_after_context": future_frames,
                }
            )
        if not specs:
            raise ValueError("No valid legacy v1 t2w window specs; check context/future settings.")
        return specs

    specs = []
    for context in build_context_options(args):
        context_latent = int(context["latent_frames"])
        context_pixels = int(context["pixel_frames"])
        context_specs: list[dict[str, int | float]] = []
        first_valid_future: int | None = None
        for future_frames in range(args.min_future_frames, args.max_future_frames + 1):
            total_frames = context_pixels + future_frames
            start_frame = args.context_frames - context_pixels
            end_frame = args.context_frames + future_frames - 1
            if not BASE.valid_vae_frame_count(total_frames):
                continue
            if first_valid_future is None:
                first_valid_future = future_frames
            if (future_frames - first_valid_future) % args.window_stride_frames != 0:
                continue
            if total_frames < args.min_loader_frames:
                continue
            if start_frame < 0 or end_frame >= args.frames:
                continue
            context_specs.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "total_frames": total_frames,
                    "context_latent_frames": context_latent,
                    "context_pixel_frames": context_pixels,
                    "future_frames_after_context": future_frames,
                }
            )
        if not context_specs:
            continue
        per_window_weight = float(context["weight"]) / len(context_specs)
        for spec in context_specs:
            spec["window_sampling_weight"] = per_window_weight
        specs.extend(context_specs)
    if not specs:
        raise ValueError("No valid t2w window specs; check context/future settings.")
    return specs


def build_window_frame_counts(args: argparse.Namespace) -> list[int]:
    return sorted({int(spec["total_frames"]) for spec in build_window_specs(args)})


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wts_root = Path(args.wts_root)
    output_root = Path(args.output_root)
    selected_labels = {str(label) for label in args.labels}
    window_specs = build_window_specs(args)
    window_frame_counts = sorted({int(spec["total_frames"]) for spec in window_specs})
    clip_seconds = args.frames / args.fps
    context_seconds = args.context_frames / args.fps

    tasks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "caption_errors": [],
        "duration_errors": [],
        "source_missing": [],
        "phase_counts_source": defaultdict(Counter),
        "pad_start_rows": 0,
        "pad_start_seconds": Counter(),
    }

    for source_split in args.source_splits:
        for view in args.views:
            files = caption_files(wts_root, source_split, view)
            if args.max_caption_files_per_view_split:
                files = files[: args.max_caption_files_per_view_split]
            for caption_path in files:
                try:
                    caption = read_json(caption_path)
                except Exception as exc:  # noqa: BLE001
                    stats["caption_errors"].append({"path": str(caption_path), "error": str(exc)})
                    continue
                rel_parent = caption_path.relative_to(wts_root / "caption" / source_split).parent
                scenario = rel_parent.parent.as_posix().replace("/", "__")
                video_dir = wts_root / "video" / source_split / rel_parent
                for video_name in video_names_for_caption(caption, video_dir, view):
                    src_video = video_dir / video_name
                    if not src_video.exists():
                        stats["source_missing"].append(str(src_video))
                        continue
                    try:
                        source_duration = BASE.ffprobe_duration(src_video)
                    except Exception as exc:  # noqa: BLE001
                        stats["duration_errors"].append({"path": str(src_video), "error": str(exc)})
                        continue
                    for phase_order_index, phase in enumerate(BASE.sorted_event_phases(caption)):
                        label = BASE.phase_label(phase)
                        if label not in selected_labels:
                            continue
                        phase_start = BASE.as_float(phase.get("start_time"))
                        phase_end = BASE.as_float(phase.get("end_time"), phase_start)
                        clip_start = max(0.0, phase_start - context_seconds)
                        pad_start_seconds = max(0.0, context_seconds - phase_start)
                        target_start_in_clip = phase_start - clip_start + pad_start_seconds
                        target_end_in_clip = phase_end - clip_start + pad_start_seconds
                        basename = (
                            f"wts_{source_split}_{view}_{BASE.safe_token(scenario)}_{BASE.safe_token(Path(video_name).stem)}"
                            f"_label{label}_ps{BASE.safe_float_token(phase_start)}_cs{BASE.safe_float_token(clip_start)}"
                        )
                        dst_video = output_root / args.output_split / "videos" / f"{basename}.mp4"
                        dst_meta = output_root / args.output_split / "metas" / f"{basename}.txt"
                        dst_record = output_root / args.output_split / "records" / f"{basename}.json"
                        row = {
                            "split": args.output_split,
                            "source_dataset": "WTS",
                            "source_split": source_split,
                            "scenario": scenario,
                            "view": view,
                            "source_video": str(src_video),
                            "caption_json": str(caption_path),
                            "video": str(dst_video),
                            "meta": str(dst_meta),
                            "record": str(dst_record),
                            "video_name": video_name,
                            "basename": basename,
                            "target_label": label,
                            "label": label,
                            "target_phase": LABEL_PHASES.get(label, "target future phase"),
                            "phase_order_index": phase_order_index,
                            "phase_start": round(phase_start, 3),
                            "phase_end": round(phase_end, 3),
                            "clip_start": round(clip_start, 3),
                            "clip_duration": clip_seconds,
                            "pad_start_seconds": round(pad_start_seconds, 6),
                            "synthetic_preroll_frames": int(round(pad_start_seconds * args.fps)),
                            "target_phase_start_in_clip": round(target_start_in_clip, 3),
                            "target_phase_end_in_clip": round(target_end_in_clip, 3),
                            "context_frames": args.context_frames,
                            "context_seconds": context_seconds,
                            "window_specs": window_specs,
                            "window_frame_counts": window_frame_counts,
                            "window_count": len(window_specs),
                            "min_future_frames": args.min_future_frames,
                            "max_future_frames": args.max_future_frames,
                            "window_stride_frames": args.window_stride_frames,
                            "fps": args.fps,
                            "frames": args.frames,
                            "width": args.width,
                            "height": args.height,
                            "source_duration": round(source_duration, 3),
                            "pedestrian_caption": BASE.normalize_text(phase.get("caption_pedestrian")),
                            "vehicle_caption": BASE.normalize_text(phase.get("caption_vehicle")),
                            "label_quality": "wts_gt",
                            "phase_timing_quality": "wts_gt",
                            "context_policy": (
                                "target_aligned_at_context_with_optional_start_padding"
                                if args.legacy_v1
                                else "target_boundary_aligned_per_window_context"
                            ),
                            "ok": False,
                        }
                        row["prompt"] = build_prompt(row, legacy_v1=args.legacy_v1)
                        tasks.append(row)
                        stats["phase_counts_source"][f"{source_split}/{view}"][label] += 1
                        if pad_start_seconds > 1e-6:
                            stats["pad_start_rows"] += 1
                            stats["pad_start_seconds"][f"{pad_start_seconds:.3f}"] += 1
                        if args.max_clips and len(tasks) >= args.max_clips:
                            filtered, label0_stats = downsample_label0(tasks, args.label0_fraction)
                            stats["label0_sampling"] = label0_stats
                            return filtered, stats

    filtered, label0_stats = downsample_label0(tasks, args.label0_fraction)
    stats["label0_sampling"] = label0_stats
    return filtered, stats


def process_task(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    result = dict(task)
    result["ok"] = False
    try:
        dst_video = Path(task["video"])
        ffmpeg_extract(
            src_video=Path(task["source_video"]),
            dst_video=dst_video,
            clip_start=float(task["clip_start"]),
            clip_seconds=float(task["clip_duration"]),
            pad_start_seconds=float(task["pad_start_seconds"]),
            fps=int(task["fps"]),
            frames=int(task["frames"]),
            width=int(task["width"]),
            height=int(task["height"]),
            crf=int(args.crf),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        if args.dry_run:
            result["ok"] = True
            return result
        probe = BASE.ffprobe_clip(dst_video)
        result["output_probe"] = probe
        result["output_width"] = int(probe.get("width") or 0)
        result["output_height"] = int(probe.get("height") or 0)
        result["output_frames"] = int(probe.get("nb_read_frames") or 0)
        result["output_avg_frame_rate"] = probe.get("avg_frame_rate")
        result["output_fps"] = BASE.rate_to_float(probe.get("avg_frame_rate"))
        result["ok"] = (
            result["output_width"] == int(task["width"])
            and result["output_height"] == int(task["height"])
            and result["output_frames"] == int(task["frames"])
            and abs(result["output_fps"] - float(task["fps"])) < 0.01
        )
        BASE.write_text(Path(task["meta"]), task["prompt"], overwrite=args.overwrite)
        BASE.write_json(Path(task["record"]), result, overwrite=args.overwrite)
        return result
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if not args.dry_run:
            BASE.write_json(Path(task["record"]), result, overwrite=True)
        return result


def cosmos3_jsonl_row(row: dict[str, Any], split_root: Path) -> dict[str, Any]:
    video_path = Path(row["video"])
    try:
        vision_path = video_path.relative_to(split_root).as_posix()
    except ValueError:
        vision_path = str(video_path)
    windows = []
    for spec in row.get("window_specs") or []:
        window = {
            "start_frame": int(spec["start_frame"]),
            "end_frame": int(spec["end_frame"]),
            "temporal_interval": 1,
            "caption": row["prompt"],
            "window_num_frames": int(spec["total_frames"]),
            "future_frames_after_context": int(spec["future_frames_after_context"]),
        }
        if "context_latent_frames" in spec:
            window.update(
                {
                    "condition_latent_frames": int(spec["context_latent_frames"]),
                    "context_pixel_frames": int(spec["context_pixel_frames"]),
                    "window_sampling_weight": float(spec["window_sampling_weight"]),
                    "target_boundary_frame": int(row["context_frames"]),
                }
            )
        windows.append(window)
    return {
        "uuid": row["basename"],
        "duration": row["frames"] / row["fps"],
        "width": row["width"],
        "height": row["height"],
        "vision_path": vision_path,
        "t2w_windows": windows,
    }


def counter_to_plain(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {k: counter_to_plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: counter_to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [counter_to_plain(v) for v in value]
    return value


def write_outputs(args: argparse.Namespace, output_root: Path, results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    window_mode = "legacy-v1 start-at-zero" if args.legacy_v1 else "context-aligned"
    split = args.output_split
    split_root = output_root / split
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_root / "manifest_all.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sft_path = split_root / "video_dataset_file.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_path.open("w", encoding="utf-8") as f:
        for row in results:
            if row.get("ok"):
                f.write(json.dumps(cosmos3_jsonl_row(row, split_root), ensure_ascii=False) + "\n")

    ok_rows = [row for row in results if row.get("ok")]
    window_lengths = Counter()
    for row in ok_rows:
        specs = row.get("window_specs") or []
        window_lengths.update(int(spec["total_frames"]) for spec in specs)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": "Cosmos3 Nano WTS train+val all-view SFT dataset with view-aware Qwen-style captions",
        "source_dataset": "WTS",
        "source_splits": args.source_splits,
        "output_split": args.output_split,
        "views": args.views,
        "labels": args.labels,
        "important": [
            "WTS train and val are merged into the train split by design.",
            "overhead_view and vehicle_view are separate sample rows, not simultaneous model inputs.",
            "label0 is downsampled and should be treated as auxiliary domain/context regularization.",
            "WTS_TRACK5_TEST is not used.",
        ],
        "clip_contract": {
            "window_mode": window_mode,
            "fps": args.fps,
            "frames": args.frames,
            "duration": args.frames / args.fps,
            "width": args.width,
            "height": args.height,
            "context_frames": args.context_frames,
            "context_seconds": args.context_frames / args.fps,
            "context_latent_options": parse_int_csv(args.context_latent_options),
            "context_latent_weights": parse_float_csv(args.context_latent_weights),
            "target_boundary_frame": args.context_frames,
            "random_window_frame_counts": build_window_frame_counts(args),
        },
        "counts": {
            "total": len(results),
            "ok": len(ok_rows),
            "failed": len(results) - len(ok_rows),
            "by_source_split": dict(Counter(row["source_split"] for row in results)),
            "by_view": dict(Counter(row["view"] for row in results)),
            "by_label": dict(Counter(str(row["label"]) for row in results)),
            "ok_by_view": dict(Counter(row["view"] for row in ok_rows)),
            "ok_by_label": dict(Counter(str(row["label"]) for row in ok_rows)),
            "ok_by_source_split_view": dict(Counter(f"{row['source_split']}/{row['view']}" for row in ok_rows)),
            "t2w_windows_per_ok_clip": len(build_window_specs(args)),
            "total_t2w_windows": sum(window_lengths.values()),
            "t2w_window_lengths": dict(sorted(window_lengths.items())),
        },
        "build_stats": counter_to_plain(stats),
        "args": vars(args),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = "\n".join(
        [
            "# Cosmos3 WTS All-View Train+Val",
            "",
            "WTS train+val merged into one Cosmos3 SFT train split.",
            "Views are separate rows: overhead_view and vehicle_view.",
            "Prompts are Qwen-friendly and include Source split, View, Target phase, Pedestrian description, and Vehicle description.",
            "Label0 is auxiliary context/domain data and is downsampled.",
            (
                "Each t2w window starts at clip frame zero, matching the submitted legacy-v1 manifest."
                if args.legacy_v1
                else "Each t2w window is aligned so its visual context ends at the target boundary."
            ),
            "",
            "Official SFT JSONL:",
            "",
            "```text",
            "train/video_dataset_file.jsonl",
            "```",
            "",
            f"Clip contract: {args.frames} frames at {args.fps} fps, {args.width}x{args.height}.",
            f"{window_mode.capitalize()} t2w window lengths: {build_window_frame_counts(args)}.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.frames <= args.context_frames:
        raise ValueError("--frames must be greater than --context-frames")
    if not BASE.valid_vae_frame_count(args.frames):
        raise ValueError("--frames should be 1 mod 4 for Wan2.2 temporal compression")
    if not BASE.valid_vae_frame_count(args.context_frames):
        raise ValueError("--context-frames should be 1 mod 4 for Wan2.2 temporal compression")
    output_root = Path(args.output_root)
    window_mode = "Legacy-v1 start-at-zero" if args.legacy_v1 else "Context-aligned"
    print(f"{window_mode} t2w window lengths: {build_window_frame_counts(args)}", flush=True)
    tasks, stats = build_tasks(args)
    print(f"Built {len(tasks)} extraction tasks", flush=True)
    if not tasks:
        write_outputs(args, output_root, [], stats)
        return 1
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_task, args, task) for task in tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = fut.result()
            results.append(row)
            if idx % 50 == 0 or idx == len(futures):
                ok = sum(1 for item in results if item.get("ok"))
                print(f"processed {idx}/{len(futures)} ok={ok}", flush=True)
    results.sort(key=lambda row: (row["source_split"], row["view"], row["label"], row["basename"]))
    write_outputs(args, output_root, results, stats)
    ok_count = sum(1 for row in results if row.get("ok"))
    print(f"Done. ok={ok_count} failed={len(results)-ok_count} output={output_root}", flush=True)
    return 0 if ok_count == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
