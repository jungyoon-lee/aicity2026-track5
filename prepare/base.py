#!/usr/bin/env python3
"""Build WTS overhead-only Cosmos3 SFT clips and JSONL files.

The script reads the original WTS overhead videos/captions and writes a
separate processed dataset. It keeps the official Cosmos3 training code
untouched by producing the JSONL format expected by the official SFT loader.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import subprocess
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


LABEL_MEANINGS = {
    "1": "recognition onset / the pedestrian becomes relevant to the scene.",
    "2": "judgement / risk assessment and conflict prediction phase.",
    "3": "action / pedestrian or vehicle response is actively unfolding.",
    "4": "avoidance or outcome / braking, stopping, collision, passing, or resolution.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wts-root",
        default=str(PROJECT_ROOT / "datasets/aicity_track5/data/WTS"),
        help="WTS root containing video/{train,val} and caption/{train,val}.",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "data/wts_overhead"
        ),
        help="Output root for the processed Cosmos3 SFT dataset.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--labels", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=165)
    parser.add_argument(
        "--min-future-frames",
        type=int,
        default=48,
        help="Minimum future frames after context for random-length t2w windows.",
    )
    parser.add_argument(
        "--max-future-frames",
        type=int,
        default=120,
        help="Maximum future frames after context for random-length t2w windows.",
    )
    parser.add_argument(
        "--window-stride-frames",
        type=int,
        default=4,
        help="Stride between candidate t2w window lengths. Keep this at 4 for Wan2.2 temporal compression.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=45,
        help="Pixel frames before the target phase start. 45 matches 12 latent frames with temporal compression 4.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-caption-files-per-split", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def rate_to_float(value: Any, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if "/" in text:
        num, den = text.split("/", 1)
        den_value = as_float(den, 0.0)
        if den_value:
            return as_float(num, default) / den_value
        return default
    return as_float(text, default)


def safe_token(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def safe_float_token(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def valid_vae_frame_count(frames: int, temporal_compression_factor: int = 4) -> bool:
    return frames >= 1 and (frames - 1) % temporal_compression_factor == 0


def build_window_frame_counts(args: argparse.Namespace) -> list[int]:
    min_frames = args.context_frames + args.min_future_frames
    max_frames = min(args.frames, args.context_frames + args.max_future_frames)
    if args.min_future_frames <= 0:
        raise ValueError("--min-future-frames must be positive")
    if args.max_future_frames < args.min_future_frames:
        raise ValueError("--max-future-frames must be >= --min-future-frames")
    if args.window_stride_frames <= 0:
        raise ValueError("--window-stride-frames must be positive")
    if min_frames > max_frames:
        raise ValueError(
            "window min is greater than max: "
            f"context={args.context_frames}, min_future={args.min_future_frames}, "
            f"max_future={args.max_future_frames}, frames={args.frames}"
        )

    counts = [
        frames
        for frames in range(min_frames, max_frames + 1, args.window_stride_frames)
        if valid_vae_frame_count(frames)
    ]
    if not counts:
        raise ValueError(
            "No valid window frame counts. Use values where total_frames = context + future is 1 mod 4."
        )
    if counts[-1] != max_frames and valid_vae_frame_count(max_frames):
        counts.append(max_frames)
    return counts


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return as_float(proc.stdout.strip(), 0.0)


def ffprobe_clip(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def sorted_event_phases(caption: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(caption.get("event_phase") or [], key=lambda phase: as_float(phase.get("start_time")))


def phase_label(phase: dict[str, Any]) -> str | None:
    labels = phase.get("labels") or []
    if not labels:
        return None
    return str(labels[0]).strip()


def build_prompt(
    *,
    split: str,
    scenario: str,
    video_name: str,
    label: str,
    phase_order_index: int,
    phase_start: float,
    phase_end: float,
    clip_start: float,
    target_start_in_clip: float,
    pedestrian_caption: str,
    vehicle_caption: str,
) -> str:
    meaning = LABEL_MEANINGS.get(label, "target future phase")
    return (
        "Generate an AI City WTS overhead-view future traffic video for the target phase. "
        "If visual context frames are provided, continue from them with consistent fixed-camera "
        "geometry, object locations, appearance, and motion. "
        "If no visual context is provided, synthesize a plausible fixed overhead-view traffic "
        "scene matching the target phase and captions. "
        "SOURCE_DATASET=WTS. "
        f"SOURCE_SPLIT={split}. "
        "VIEW=overhead_view. "
        f"SCENARIO={scenario}. "
        f"VIDEO={video_name}. "
        f"TARGET_LABEL={label}. "
        f"PHASE_ORDER_INDEX={phase_order_index}. "
        f"PHASE_MEANING={meaning} "
        f"PHASE_START_SOURCE={phase_start:.3f}s. "
        f"PHASE_END_SOURCE={phase_end:.3f}s. "
        f"CLIP_START_SOURCE={clip_start:.3f}s. "
        f"TARGET_PHASE_START_IN_CLIP={target_start_in_clip:.3f}s. "
        "Use TARGET_LABEL and both captions to control pedestrian motion, vehicle motion, "
        "risk level, and outcome. "
        f"PEDESTRIAN_CAPTION={pedestrian_caption} "
        f"VEHICLE_CAPTION={vehicle_caption}"
    )


def ffmpeg_extract(
    *,
    src_video: Path,
    dst_video: Path,
    clip_start: float,
    clip_seconds: float,
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
    vf = (
        f"fps={fps},"
        f"scale={width}:{scale_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}:0:{crop_y},"
        "setsar=1,"
        "tpad=stop_mode=clone:stop_duration=2,"
        f"trim=start_frame=0:end_frame={frames},"
        f"setpts=N/({fps}*TB)"
    )
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


def find_caption_files(wts_root: Path, split: str) -> list[Path]:
    paths = sorted((wts_root / "caption" / split).glob("**/overhead_view/*_caption.json"))
    return paths


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wts_root = Path(args.wts_root)
    output_root = Path(args.output_root)
    window_frame_counts = build_window_frame_counts(args)
    clip_seconds = args.frames / args.fps
    context_seconds = args.context_frames / args.fps
    selected_labels = {str(label) for label in args.labels}

    tasks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "caption_errors": [],
        "duration_errors": [],
        "source_missing": [],
        "too_short_preroll": [],
        "phase_counts_source": defaultdict(Counter),
    }

    for split in args.splits:
        caption_files = find_caption_files(wts_root, split)
        if args.max_caption_files_per_split:
            caption_files = caption_files[: args.max_caption_files_per_split]

        for caption_path in caption_files:
            try:
                caption = read_json(caption_path)
            except Exception as exc:  # noqa: BLE001
                stats["caption_errors"].append({"path": str(caption_path), "error": str(exc)})
                continue

            rel_parent = caption_path.relative_to(wts_root / "caption" / split).parent
            video_dir = wts_root / "video" / split / rel_parent
            scenario = rel_parent.parent.as_posix().replace("/", "__")
            overhead_videos = caption.get("overhead_videos") or []
            if not overhead_videos:
                overhead_videos = [path.name for path in sorted(video_dir.glob("*.mp4"))]

            for video_name in overhead_videos:
                src_video = video_dir / video_name
                if not src_video.exists():
                    stats["source_missing"].append(str(src_video))
                    continue
                try:
                    source_duration = ffprobe_duration(src_video)
                except Exception as exc:  # noqa: BLE001
                    stats["duration_errors"].append({"path": str(src_video), "error": str(exc)})
                    continue

                for phase_order_index, phase in enumerate(sorted_event_phases(caption)):
                    label = phase_label(phase)
                    if label not in selected_labels:
                        continue

                    phase_start = as_float(phase.get("start_time"))
                    phase_end = as_float(phase.get("end_time"), phase_start)
                    if phase_start < context_seconds:
                        stats["too_short_preroll"].append(
                            {
                                "caption": str(caption_path),
                                "video": str(src_video),
                                "label": label,
                                "phase_start": phase_start,
                                "required_context_seconds": context_seconds,
                            }
                        )
                        continue

                    clip_start = phase_start - context_seconds
                    target_start_in_clip = phase_start - clip_start
                    target_end_in_clip = phase_end - clip_start
                    basename = (
                        f"wts_{split}_{safe_token(scenario)}_{safe_token(Path(video_name).stem)}"
                        f"_label{label}_ps{safe_float_token(phase_start)}_cs{safe_float_token(clip_start)}"
                    )
                    dst_video = output_root / split / "videos" / f"{basename}.mp4"
                    dst_meta = output_root / split / "metas" / f"{basename}.txt"
                    dst_record = output_root / split / "records" / f"{basename}.json"

                    pedestrian_caption = normalize_text(phase.get("caption_pedestrian"))
                    vehicle_caption = normalize_text(phase.get("caption_vehicle"))
                    prompt = build_prompt(
                        split=split,
                        scenario=scenario,
                        video_name=video_name,
                        label=label,
                        phase_order_index=phase_order_index,
                        phase_start=phase_start,
                        phase_end=phase_end,
                        clip_start=clip_start,
                        target_start_in_clip=target_start_in_clip,
                        pedestrian_caption=pedestrian_caption,
                        vehicle_caption=vehicle_caption,
                    )

                    task = {
                        "split": split,
                        "source_dataset": "WTS",
                        "scenario": scenario,
                        "view": "overhead_view",
                        "source_video": str(src_video),
                        "caption_json": str(caption_path),
                        "video": str(dst_video),
                        "meta": str(dst_meta),
                        "record": str(dst_record),
                        "video_name": video_name,
                        "basename": basename,
                        "target_label": label,
                        "label": label,
                        "phase_order_index": phase_order_index,
                        "phase_start": round(phase_start, 3),
                        "phase_end": round(phase_end, 3),
                        "clip_start": round(clip_start, 3),
                        "clip_duration": clip_seconds,
                        "target_phase_start_in_clip": round(target_start_in_clip, 3),
                        "target_phase_end_in_clip": round(target_end_in_clip, 3),
                        "context_frames": args.context_frames,
                        "context_seconds": context_seconds,
                        "window_frame_counts": window_frame_counts,
                        "window_count": len(window_frame_counts),
                        "min_future_frames": args.min_future_frames,
                        "max_future_frames": args.max_future_frames,
                        "window_stride_frames": args.window_stride_frames,
                        "fps": args.fps,
                        "frames": args.frames,
                        "width": args.width,
                        "height": args.height,
                        "source_duration": round(source_duration, 3),
                        "pedestrian_caption": pedestrian_caption,
                        "vehicle_caption": vehicle_caption,
                        "prompt": prompt,
                        "label_quality": "wts_gt",
                        "phase_timing_quality": "wts_gt",
                        "context_policy": "fixed_pre_target_context",
                        "ok": False,
                    }
                    tasks.append(task)
                    stats["phase_counts_source"][split][label] += 1
                    if args.max_clips and len(tasks) >= args.max_clips:
                        return tasks, stats

    return tasks, stats


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
            fps=int(task["fps"]),
            frames=int(task["frames"]),
            width=int(task["width"]),
            height=int(task["height"]),
            crf=int(args.crf),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            probe = ffprobe_clip(dst_video)
            result["output_probe"] = probe
            result["output_width"] = int(probe.get("width") or 0)
            result["output_height"] = int(probe.get("height") or 0)
            result["output_frames"] = int(probe.get("nb_read_frames") or 0)
            result["output_avg_frame_rate"] = probe.get("avg_frame_rate")
            result["output_fps"] = rate_to_float(probe.get("avg_frame_rate"))
            result["ok"] = (
                result["output_width"] == int(task["width"])
                and result["output_height"] == int(task["height"])
                and result["output_frames"] == int(task["frames"])
                and abs(result["output_fps"] - float(task["fps"])) < 0.01
            )
            write_text(Path(task["meta"]), task["prompt"], overwrite=args.overwrite)
            write_json(Path(task["record"]), result, overwrite=args.overwrite)
        else:
            result["ok"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if not args.dry_run:
            write_json(Path(task["record"]), result, overwrite=True)
        return result


def cosmos3_jsonl_row(row: dict[str, Any], split_root: Path) -> dict[str, Any]:
    video_path = Path(row["video"])
    try:
        vision_path = video_path.relative_to(split_root).as_posix()
    except ValueError:
        vision_path = str(video_path)
    windows = []
    for frames in row["window_frame_counts"]:
        windows.append(
            {
                "start_frame": 0,
                "end_frame": int(frames) - 1,
                "temporal_interval": 1,
                "caption": row["prompt"],
                "window_num_frames": int(frames),
                "future_frames_after_context": int(frames) - int(row["context_frames"]),
            }
        )
    return {
        "uuid": row["basename"],
        "duration": row["frames"] / row["fps"],
        "width": row["width"],
        "height": row["height"],
        "vision_path": vision_path,
        "t2w_windows": windows,
    }


def write_manifests(output_root: Path, results: list[dict[str, Any]]) -> None:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_split[row["split"]].append(row)

    for split, rows in by_split.items():
        manifest_path = manifest_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        sft_path = output_root / split / "video_dataset_file.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        with sft_path.open("w", encoding="utf-8") as f:
            split_root = output_root / split
            for row in rows:
                if row.get("ok"):
                    f.write(json.dumps(cosmos3_jsonl_row(row, split_root), ensure_ascii=False) + "\n")

    with (output_root / "manifest_all.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def write_summary(
    output_root: Path,
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    counts_by_split = Counter(row["split"] for row in results)
    ok_by_split = Counter(row["split"] for row in results if row.get("ok"))
    label_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    ok_label_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        label_by_split[row["split"]][row["label"]] += 1
        if row.get("ok"):
            ok_label_by_split[row["split"]][row["label"]] += 1
    ok_rows = [row for row in results if row.get("ok")]
    window_lengths = Counter()
    for row in ok_rows:
        window_lengths.update(int(frames) for frames in row.get("window_frame_counts", []))

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": "Cosmos3 Nano WTS overhead-only Track5-shaped SFT dataset with random-length t2w windows",
        "source_dataset": "WTS",
        "view": "overhead_view",
        "labels": args.labels,
        "output_root": str(output_root),
        "original_dataset_write_policy": "read_only_sources_write_only_to_output_root",
        "clip_contract": {
            "fps": args.fps,
            "frames": args.frames,
            "duration": args.frames / args.fps,
            "width": args.width,
            "height": args.height,
            "context_frames": args.context_frames,
            "context_seconds": args.context_frames / args.fps,
            "min_future_frames_after_context": args.min_future_frames,
            "max_future_frames_after_context": args.max_future_frames,
            "window_stride_frames": args.window_stride_frames,
            "random_window_frame_counts": build_window_frame_counts(args),
            "max_encoded_clip_future_frames_after_context": args.frames - args.context_frames,
        },
        "cosmos3_contract": {
            "sft_jsonl": "<split>/video_dataset_file.jsonl",
            "window_sampling": "official SFT loader randomly chooses one t2w_window per sample access",
            "num_video_frames": -1,
            "resolution_key": "720" if args.width == 1280 and args.height == 720 else "custom",
            "aspect_ratio": "16,9" if args.width * 9 == args.height * 16 else "computed_by_loader",
            "recommended_conditioning_config": {
                "0": 0.01,
                "1": 0.04,
                "5": 0.10,
                "8": 0.20,
                "10": 0.25,
                "12": 0.40,
            },
            "conservative_conditioning_config": {
                "0": 0.02,
                "1": 0.05,
                "5": 0.08,
                "8": 0.20,
                "10": 0.25,
                "12": 0.40,
            },
            "smoke_conditioning_config": {"12": 1.0},
        },
        "counts": {
            "total": len(results),
            "ok": sum(1 for row in results if row.get("ok")),
            "failed": sum(1 for row in results if not row.get("ok")),
            "by_split": dict(counts_by_split),
            "ok_by_split": dict(ok_by_split),
            "label_by_split": counter_to_plain(label_by_split),
            "ok_label_by_split": counter_to_plain(ok_label_by_split),
            "t2w_windows_per_ok_clip": len(build_window_frame_counts(args)),
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
            "# Cosmos3 WTS Overhead Track5 Random-Length",
            "",
            "Generated from WTS overhead_view train/val videos and captions.",
            "Original WTS files are read-only; this directory contains only derived clips and JSONL manifests.",
            "Each encoded MP4 uses the maximum clip length, while each JSONL row contains multiple t2w_windows.",
            "The official Cosmos3 SFT loader randomly samples one t2w_window per sample access.",
            "",
            "Official Cosmos3 SFT JSONL:",
            "",
            "```text",
            "train/video_dataset_file.jsonl",
            "val/video_dataset_file.jsonl",
            "```",
            "",
            f"Clip contract: {args.frames} frames at {args.fps} fps, {args.width}x{args.height}.",
            f"Pre-target context: {args.context_frames} frames.",
            f"Random t2w window lengths: {build_window_frame_counts(args)}.",
            "",
            "Recommended first smoke: V2V only with conditioning_config={12:1.0}.",
            "Recommended mixed WTS-overhead run: conditioning_config={0:0.01,1:0.04,5:0.10,8:0.20,10:0.25,12:0.40}.",
            "Conservative mixed option: conditioning_config={0:0.02,1:0.05,5:0.08,8:0.20,10:0.25,12:0.40}.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.frames <= args.context_frames:
        raise ValueError("--frames must be greater than --context-frames")
    if not valid_vae_frame_count(args.frames):
        raise ValueError("--frames should be 1 mod 4 for Wan2.2 temporal compression")
    if not valid_vae_frame_count(args.context_frames):
        raise ValueError("--context-frames should be 1 mod 4 for Wan2.2 temporal compression")
    output_root = Path(args.output_root)
    print(f"Random t2w window lengths: {build_window_frame_counts(args)}")
    tasks, stats = build_tasks(args)
    print(f"Built {len(tasks)} extraction tasks")
    if not tasks:
        write_summary(output_root, [], stats, args)
        return 1

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_task, args, task) for task in tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = fut.result()
            results.append(row)
            if idx % 50 == 0 or idx == len(futures):
                ok = sum(1 for item in results if item.get("ok"))
                print(f"processed {idx}/{len(futures)} ok={ok}")

    results.sort(key=lambda row: (row["split"], row["basename"]))
    write_manifests(output_root, results)
    write_summary(output_root, results, stats, args)
    ok_count = sum(1 for row in results if row.get("ok"))
    print(f"Done. ok={ok_count} failed={len(results)-ok_count} output={output_root}")
    return 0 if ok_count == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
