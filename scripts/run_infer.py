#!/usr/bin/env python3
"""Run Cosmos3 Track5 testset inference as 8 single-GPU shards and merge output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(
    os.environ.get("AICITY_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


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
    parser.add_argument(
        "--infer-script", type=Path, default=PROJECT_ROOT / "scripts/infer.py"
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size. The submitted run used 1.",
    )
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def case_dirs(test_root: Path) -> list[Path]:
    return sorted(path for path in test_root.iterdir() if path.is_dir())


def split_indices(num_cases: int, num_shards: int) -> list[list[int]]:
    shards = [[] for _ in range(num_shards)]
    for index in range(num_cases):
        shards[index % num_shards].append(index)
    return shards


def run_shards(
    args: argparse.Namespace, shards: list[list[int]], gpus: list[str]
) -> list[dict[str, Any]]:
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    procs: list[dict[str, Any]] = []
    for shard_index, indices in enumerate(shards):
        if not indices:
            continue
        gpu = gpus[shard_index]
        shard_dir = shard_root / f"gpu{gpu}_shard{shard_index:02d}"
        if shard_dir.exists() and args.overwrite:
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "worker.log"
        cmd = [
            sys.executable,
            str(args.infer_script),
            "--test-root",
            str(args.test_root),
            "--checkpoint",
            str(args.checkpoint),
            "--config-file",
            str(args.config_file),
            "--output-dir",
            str(shard_dir),
            "--case-indices",
            ",".join(str(index) for index in indices),
            "--batch-size",
            str(args.batch_size),
            "--num-steps",
            str(args.num_steps),
            "--guidance",
            str(args.guidance),
            "--seed",
            str(args.seed),
            "--overwrite",
        ]
        if args.adapter_path is not None:
            cmd.extend(["--adapter-path", str(args.adapter_path)])
        env = os.environ.copy()
        env["AICITY_ROOT"] = str(PROJECT_ROOT)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log(f"launch shard {shard_index} gpu={gpu} cases={indices}")
        log("RUN " + " ".join(cmd))
        with log_path.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append(
            {
                "proc": proc,
                "gpu": gpu,
                "shard_index": shard_index,
                "indices": indices,
                "dir": shard_dir,
                "log": log_path,
            }
        )
    return procs


def wait_shards(procs: list[dict[str, Any]], poll_seconds: float) -> None:
    unfinished = set(range(len(procs)))
    while unfinished:
        for idx in list(unfinished):
            proc = procs[idx]["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            procs[idx]["returncode"] = rc
            unfinished.remove(idx)
            log(
                f"shard {procs[idx]['shard_index']} gpu={procs[idx]['gpu']} exited rc={rc}"
            )
        if unfinished:
            status = ", ".join(
                f"s{procs[idx]['shard_index']}:pid{procs[idx]['proc'].pid}"
                for idx in sorted(unfinished)
            )
            log(f"waiting shards: {status}")
            time.sleep(poll_seconds)
    failed = [info for info in procs if info.get("returncode") != 0]
    if failed:
        for info in failed:
            log(
                f"FAILED shard {info['shard_index']} gpu={info['gpu']} log={info['log']}"
            )
        raise RuntimeError(f"{len(failed)} shard(s) failed")


def merge_predictions(
    args: argparse.Namespace, procs: list[dict[str, Any]], cases: list[Path]
) -> dict[str, Any]:
    pred_root = args.output_dir / "prediction"
    if pred_root.exists() and args.overwrite:
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    shard_summaries = []
    copied = []
    for info in procs:
        summary_path = info["dir"] / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        if summary.get("bad_cases"):
            raise RuntimeError(
                f"bad cases in shard {info['shard_index']}: {summary['bad_cases']}"
            )
        shard_pred_root = info["dir"] / "prediction"
        for case_dir in sorted(
            path for path in shard_pred_root.iterdir() if path.is_dir()
        ):
            dst = pred_root / case_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(case_dir, dst)
            copied.append(case_dir.name)

    expected_case_ids = [path.name for path in cases]
    missing = sorted(set(expected_case_ids) - set(copied))
    extra = sorted(set(copied) - set(expected_case_ids))
    bad_counts = []
    for case in cases:
        caption = json.loads((case / "caption.json").read_text(encoding="utf-8"))
        expected_frames = int(caption["frame length"])
        frames = sorted(
            (pred_root / case.name).glob("*.png"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
        )
        numeric_names = [path.name for path in frames if path.stem.isdigit()]
        expected_names = [f"{idx}.png" for idx in range(expected_frames)]
        if numeric_names != expected_names:
            bad_counts.append(
                {
                    "case_id": case.name,
                    "expected": expected_frames,
                    "actual": len(numeric_names),
                }
            )
    if missing or extra or bad_counts:
        raise RuntimeError(
            f"merged prediction validation failed missing={missing} extra={extra} bad_counts={bad_counts[:5]}"
        )

    return {
        "num_cases": len(cases),
        "copied_cases": len(copied),
        "prediction_root": str(pred_root),
        "shard_summaries": shard_summaries,
    }


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
    args.infer_script = args.infer_script.expanduser().resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.expanduser().resolve()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    if not args.test_root.is_dir():
        raise NotADirectoryError(args.test_root)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.config_file.exists():
        raise FileNotFoundError(args.config_file)
    if not args.infer_script.is_file():
        raise FileNotFoundError(args.infer_script)
    if args.adapter_path is not None and not args.adapter_path.is_file():
        raise FileNotFoundError(args.adapter_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = case_dirs(args.test_root)
    if not cases:
        raise ValueError(f"No test cases found under {args.test_root}")
    shards = split_indices(len(cases), len(gpus))
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_root": str(args.test_root),
        "checkpoint": str(args.checkpoint),
        "config_file": str(args.config_file),
        "adapter_path": (
            str(args.adapter_path) if args.adapter_path is not None else None
        ),
        "output_dir": str(args.output_dir),
        "gpus": gpus,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "num_cases": len(cases),
        "shards": [
            {"gpu": gpus[idx], "indices": indices} for idx, indices in enumerate(shards)
        ],
    }
    (args.output_dir / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    procs = run_shards(args, shards, gpus)
    wait_shards(procs, args.poll_seconds)
    merged = merge_predictions(args, procs, cases)
    zip_path = write_zip(args.output_dir, args.output_dir / "prediction")
    summary = {
        **manifest,
        **merged,
        "zip_path": str(zip_path),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ok": True,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "infer.done").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
