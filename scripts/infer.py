#!/usr/bin/env python3
"""Run Cosmos3 inference on AI City Track 5 test cases and write submission PNGs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(
    os.environ.get("AICITY_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


LABEL_PHASES = {
    "1": "recognition phase",
    "2": "judgement phase",
    "3": "action phase",
    "4": "avoidance or outcome phase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/aicity_track5/data/WTS_TRACK5_TEST",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, default=PROJECT_ROOT / "cosmos/packages/cosmos3")
    parser.add_argument("--case-indices", default="", help="Comma-separated case indices after sorted case order.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=0, help="0 means all selected cases.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-condition-latent-frames", type=int, default=12)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--aspect-ratio", default="16,9")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--input-crf", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Write submission_prediction.zip after all cases finish.")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    log("RUN " + " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def inference_env(cosmos_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "7")
    env["AICITY_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{cosmos_root}:{env.get('PYTHONPATH', '')}"
    return env


def sanitize(text: str, limit: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:limit] or "sample"


def pixel_frames_from_latent(latent_frames: int) -> int:
    return (latent_frames - 1) * 4 + 1


def latent_frames_for_input(input_frames: int, max_latent_frames: int) -> int:
    latent = 1 + max(0, (input_frames - 1) // 4)
    return max(1, min(latent, max_latent_frames))


def valid_total_frames_at_least(frames: int) -> int:
    while (frames - 1) % 4 != 0:
        frames += 1
    return frames


def numeric_pngs(input_dir: Path) -> list[Path]:
    paths = []
    for path in input_dir.glob("*.png"):
        if path.stem.isdigit():
            paths.append(path)
    return sorted(paths, key=lambda path: int(path.stem))


def phase_caption(caption: dict[str, Any]) -> dict[str, Any]:
    phases = caption.get("event_phase") or []
    if not phases:
        raise ValueError("caption.json has no event_phase")
    return phases[0]


def case_view(case_id: str) -> str:
    if case_id.startswith("video"):
        return "BDD front/dashcam-style vehicle-view"
    return "WTS overhead fixed-camera view"


def phase_label(phase: dict[str, Any]) -> str:
    labels = phase.get("labels") or []
    return str(labels[0]) if labels else ""


def build_prompt(case_id: str, caption: dict[str, Any], input_frames: int, future_frames: int, condition_frames: int) -> str:
    phase = phase_caption(caption)
    label = phase_label(phase)
    phase_name = LABEL_PHASES.get(label, "target future phase")
    view = case_view(case_id)
    pedestrian_caption = str(phase.get("caption_pedestrian") or "").strip()
    vehicle_caption = str(phase.get("caption_vehicle") or "").strip()
    return (
        f"Task: Generate the future of an AI City Track 5 {view} traffic video.\n\n"
        "Visual context rule: Use the observed frames for camera viewpoint, road layout, lighting, "
        "object identity, and current object positions. Keep the future consistent with that visual context.\n\n"
        f"Target phase: {phase_name}.\n\n"
        "Use both official descriptions below to determine the future pedestrian-vehicle interaction.\n\n"
        "Pedestrian description:\n"
        f"{pedestrian_caption}\n\n"
        "Vehicle description:\n"
        f"{vehicle_caption}\n\n"
        "Generation rule: The generated future must be consistent with both descriptions. "
        "The pedestrian motion, vehicle motion, risk level, stopping or yielding behavior, "
        "and outcome should reflect the combined pedestrian-vehicle interaction.\n\n"
        f"For this inference run, visual conditioning uses the most recent {condition_frames} frames "
        f"from the {input_frames}-frame input history. Generate {future_frames} future frames after that context."
    )


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_dirs = sorted(path for path in args.test_root.iterdir() if path.is_dir())
    if args.case_indices:
        indices = [int(item.strip()) for item in args.case_indices.split(",") if item.strip()]
    else:
        end = len(all_dirs) if args.num_cases <= 0 else min(len(all_dirs), args.start_index + args.num_cases)
        indices = list(range(args.start_index, end))

    cases = []
    for order, index in enumerate(indices):
        if index < 0 or index >= len(all_dirs):
            raise IndexError(f"case index {index} out of range for {len(all_dirs)} cases")
        case_dir = all_dirs[index]
        caption_path = case_dir / "caption.json"
        input_dir = case_dir / "input"
        caption = json.loads(caption_path.read_text())
        images = numeric_pngs(input_dir)
        if not images:
            raise FileNotFoundError(f"no PNG input frames under {input_dir}")
        expected = list(range(len(images)))
        actual = [int(path.stem) for path in images]
        if actual != expected:
            raise ValueError(f"input frames for {case_dir.name} are not contiguous 0..K-1: {actual[:5]}...{actual[-5:]}")
        future_frames = int(caption["frame length"])
        condition_latent_frames = latent_frames_for_input(len(images), args.max_condition_latent_frames)
        condition_pixel_frames = pixel_frames_from_latent(condition_latent_frames)
        total_frames = valid_total_frames_at_least(condition_pixel_frames + future_frames)
        phase = phase_caption(caption)
        label = phase_label(phase)
        cases.append(
            {
                "order": order,
                "case_index": index,
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "input_dir": str(input_dir),
                "caption": caption,
                "label": label,
                "view": case_view(case_dir.name),
                "input_frames": len(images),
                "future_frames": future_frames,
                "condition_latent_frames": condition_latent_frames,
                "condition_pixel_frames": condition_pixel_frames,
                "generated_total_frames": total_frames,
            }
        )
    return cases


def ffprobe_stream(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return {}
    streams = json.loads(proc.stdout or "{}").get("streams") or []
    return streams[0] if streams else {}


def make_input_mp4(args: argparse.Namespace, case: dict[str, Any], media_dir: Path) -> Path:
    case_id = str(case["case_id"])
    out_path = media_dir / f"{sanitize(case_id)}_input.mp4"
    if out_path.exists() and not args.overwrite:
        return out_path
    input_pattern = Path(case["input_dir"]) / "%d.png"
    scale_height = int(round(args.width * 9 / 16))
    crop_y = max(0, (scale_height - args.height) // 2)
    vf = (
        f"scale={args.width}:{scale_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={args.width}:{args.height}:0:{crop_y},setsar=1"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-start_number",
            "0",
            "-i",
            str(input_pattern),
            "-vf",
            vf,
            "-frames:v",
            str(case["input_frames"]),
            "-r",
            str(args.fps),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(args.input_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return out_path


def write_sample_json(args: argparse.Namespace, case: dict[str, Any], media_dir: Path, samples_dir: Path) -> dict[str, Any]:
    input_mp4 = make_input_mp4(args, case, media_dir)
    sample_name = sanitize(
        f"case{case['order']:03d}_idx{case['case_index']}_{case['case_id']}_L{case['label']}_K{case['input_frames']}_N{case['future_frames']}"
    )
    sample_json = samples_dir / f"{sample_name}.json"
    prompt = build_prompt(
        str(case["case_id"]),
        case["caption"],
        int(case["input_frames"]),
        int(case["future_frames"]),
        int(case["condition_pixel_frames"]),
    )
    sample = {
        "name": sample_name,
        "prompt": prompt,
        "vision_path": str(input_mp4),
        "condition_frame_indexes_vision": list(range(int(case["condition_latent_frames"]))),
        "condition_video_keep": "last",
        "model_mode": "video2video",
        "resolution": args.resolution,
        "aspect_ratio": args.aspect_ratio,
        "fps": args.fps,
        "num_frames": int(case["generated_total_frames"]),
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "video_save_quality": 8,
        "prompt_upsampling": False,
        "negative_prompt": "",
    }
    sample_json.write_text(json.dumps(sample, indent=2))
    return {**case, "input_mp4": str(input_mp4), "sample_name": sample_name, "sample_json": str(sample_json)}


def find_generated_video(inference_dir: Path, sample_name: str) -> Path:
    outputs_json = inference_dir / sample_name / "sample_outputs.json"
    if outputs_json.exists():
        data = json.loads(outputs_json.read_text())
        for output in data.get("outputs", []):
            for file_name in output.get("files", []):
                path = Path(file_name)
                if path.suffix.lower() == ".mp4" and path.exists():
                    return path
    candidates = sorted(inference_dir.glob(f"{sample_name}/**/*.mp4"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"generated mp4 not found under {inference_dir / sample_name}")


def prediction_complete(pred_dir: Path, expected_frames: int) -> bool:
    frames = sorted((path for path in pred_dir.glob("*.png") if path.stem.isdigit()), key=lambda path: int(path.stem))
    if len(frames) != expected_frames:
        return False
    return [int(path.stem) for path in frames] == list(range(expected_frames))


def extract_prediction_pngs(args: argparse.Namespace, case: dict[str, Any], gen_video: Path, pred_root: Path) -> dict[str, Any]:
    case_id = str(case["case_id"])
    pred_dir = pred_root / case_id
    if pred_dir.exists() and (args.overwrite or not args.skip_existing):
        for path in pred_dir.glob("*.png"):
            path.unlink()
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and prediction_complete(pred_dir, int(case["future_frames"])):
        log(f"skip existing prediction: {case_id}")
    else:
        output_pattern = pred_dir / "%d.png"
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(gen_video),
                "-vf",
                (
                    f"fps={args.fps},"
                    f"trim=start_frame={case['condition_pixel_frames']}:"
                    f"end_frame={int(case['condition_pixel_frames']) + int(case['future_frames'])},"
                    "setpts=PTS-STARTPTS,"
                    f"scale={args.width}:{args.height},setsar=1"
                ),
                "-frames:v",
                str(case["future_frames"]),
                "-r",
                str(args.fps),
                "-start_number",
                "0",
                str(output_pattern),
            ]
        )
    complete = prediction_complete(pred_dir, int(case["future_frames"]))
    first = pred_dir / "0.png"
    stream = ffprobe_stream(first) if first.exists() else {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    ok = complete and width == args.width and height == args.height
    return {
        "case_id": case_id,
        "prediction_dir": str(pred_dir),
        "expected_frames": int(case["future_frames"]),
        "actual_frames": len(list(pred_dir.glob("*.png"))),
        "first_frame": str(first),
        "width": width,
        "height": height,
        "ok": ok,
    }


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_inference_batch(args: argparse.Namespace, batch_infos: list[dict[str, Any]], batch_dir: Path) -> Path:
    inference_dir = batch_dir / "inference"
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.inference",
        "-i",
        *[str(info["sample_json"]) for info in batch_infos],
        "-o",
        str(inference_dir),
        "--checkpoint-path",
        str(args.checkpoint),
        "--config-file",
        str(args.config_file),
        "--no-guardrails",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
        "--benchmark",
    ]
    if args.adapter_path is not None:
        command.extend(["--adapter-path", str(args.adapter_path)])
    run(
        command,
        cwd=args.cosmos_root,
        env=inference_env(args.cosmos_root),
        log_path=inference_dir / "console.log",
    )
    return inference_dir


def write_zip(output_dir: Path, pred_root: Path) -> Path:
    zip_path = output_dir / "submission_prediction.zip"
    tmp_path = output_dir / "submission_prediction.tmp.zip"
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(pred_root.rglob("*.png")):
            zf.write(path, path.relative_to(output_dir))
    tmp_path.replace(zip_path)
    return zip_path


def main() -> int:
    args = parse_args()
    args.test_root = args.test_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cosmos_root = args.cosmos_root.expanduser().resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.expanduser().resolve()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.test_root.is_dir():
        raise NotADirectoryError(args.test_root)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.config_file.is_file():
        raise FileNotFoundError(args.config_file)
    if not args.cosmos_root.is_dir():
        raise NotADirectoryError(args.cosmos_root)
    if args.adapter_path is not None and not args.adapter_path.is_file():
        raise FileNotFoundError(args.adapter_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = args.output_dir / "media"
    samples_dir = args.output_dir / "samples"
    batches_dir = args.output_dir / "batches"
    pred_root = args.output_dir / "prediction"
    for path in (media_dir, samples_dir, batches_dir, pred_root):
        path.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args)
    manifest_path = args.output_dir / "test_inference_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "test_root": str(args.test_root),
                "checkpoint": str(args.checkpoint),
                "config_file": str(args.config_file),
                "adapter_path": (
                    str(args.adapter_path) if args.adapter_path is not None else None
                ),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_cases": len(cases),
                "cases": [
                    {
                        key: value
                        for key, value in case.items()
                        if key not in {"caption"}
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
    )
    log(f"loaded {len(cases)} test cases")

    sample_infos = [write_sample_json(args, case, media_dir, samples_dir) for case in cases]
    results: list[dict[str, Any]] = []
    selected_batches = chunks(sample_infos, args.batch_size)
    for batch_index, batch_infos in enumerate(selected_batches):
        pending = [
            info
            for info in batch_infos
            if not (args.skip_existing and prediction_complete(pred_root / str(info["case_id"]), int(info["future_frames"])))
        ]
        if not pending:
            log(f"batch {batch_index:03d}: all predictions already exist")
            for info in batch_infos:
                results.append(extract_prediction_pngs(args, info, Path(info.get("generated_video", "")), pred_root))
            continue
        batch_dir = batches_dir / f"batch_{batch_index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"batch {batch_index + 1}/{len(selected_batches)}: "
            + ", ".join(str(info["case_id"]) for info in pending)
        )
        inference_dir = run_inference_batch(args, pending, batch_dir)
        for info in pending:
            gen_video = find_generated_video(inference_dir, str(info["sample_name"]))
            info["generated_video"] = str(gen_video)
            result = extract_prediction_pngs(args, info, gen_video, pred_root)
            result.update(
                {
                    "case_index": info["case_index"],
                    "label": info["label"],
                    "view": info["view"],
                    "input_frames": info["input_frames"],
                    "condition_pixel_frames": info["condition_pixel_frames"],
                    "generated_total_frames": info["generated_total_frames"],
                    "generated_video": str(gen_video),
                }
            )
            results.append(result)
            status = "ok" if result["ok"] else "BAD"
            log(f"{status}: {info['case_id']} frames={result['actual_frames']}/{result['expected_frames']} size={result['width']}x{result['height']}")
        progress_path = args.output_dir / "progress.json"
        progress_path.write_text(json.dumps({"finished": len(results), "total": len(cases), "results": results}, indent=2))

    summary = {
        "checkpoint": str(args.checkpoint),
        "config_file": str(args.config_file),
        "adapter_path": (
            str(args.adapter_path) if args.adapter_path is not None else None
        ),
        "test_root": str(args.test_root),
        "output_dir": str(args.output_dir),
        "prediction_root": str(pred_root),
        "num_cases": len(cases),
        "ok_cases": sum(1 for result in results if result.get("ok")),
        "bad_cases": [result for result in results if not result.get("ok")],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.zip:
        summary["zip_path"] = str(write_zip(args.output_dir, pred_root))
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "infer.done").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not summary["bad_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
