#!/usr/bin/env python3
"""Launch the submitted Cosmos3-Super LoRA SFT recipe for AI City Track 5.

This wrapper loads the official Cosmos3 SFT TOML/config, applies AI City
dataset-specific overrides that are awkward to express via Hydra dotlist, then
calls the official cosmos_framework.scripts.train.launch().  The canonical
entry point is scripts/train.sh, which supplies both WTS and BDD manifests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.scripts import train as official_train
from cosmos_framework.utils.serialization import to_yaml
from cosmos_framework.utils.lazy_config import LazyConfig
from cosmos_framework.utils import log


PROJECT_ROOT = Path(os.environ.get("AICITY_ROOT", Path(__file__).resolve().parents[1]))
COSMOS_ROOT = PROJECT_ROOT / "cosmos/packages/cosmos3"


def parse_conditioning_config(text: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid conditioning item {item!r}; expected K:WEIGHT")
        key, value = item.split(":", 1)
        result[int(key.strip())] = float(value.strip())
    if not result:
        raise ValueError("conditioning config cannot be empty")
    return result


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_path_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one path")
    return values


def train_jsonl_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.train_jsonl_paths:
        paths.extend(parse_path_list(args.train_jsonl_paths))
    paths.extend(args.train_jsonl_path or [])
    if not paths:
        paths.append(f"{args.dataset_path}/train/video_dataset_file.jsonl")
    return paths


def model_runtime_config(config):
    """Return the OmniMoTModel runtime config across saved/live config shapes."""
    model = config.model
    if hasattr(model, "config"):
        return model.config
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-toml",
        default=str(COSMOS_ROOT / "examples/toml/sft_config/vision_sft_super.toml"),
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data/wts"),
    )
    parser.add_argument(
        "--train-jsonl-paths",
        default="",
        help="Optional comma-separated train JSONL paths. Overrides the default dataset-path train JSONL.",
    )
    parser.add_argument(
        "--train-jsonl-path",
        action="append",
        help="Optional train JSONL path. Can be repeated and may be combined with --train-jsonl-paths.",
    )
    parser.add_argument(
        "--base-checkpoint-path",
        default=str(COSMOS_ROOT / "examples/checkpoints/Cosmos3-Super"),
    )
    parser.add_argument(
        "--vlm-pretrained-backbone-path",
        default="",
        help=(
            "Optional local/HF/S3 VLM safetensors directory to overlay onto the "
            "understanding/reasoner pathway via the official LoadPretrained callback. "
            "When used with --base-checkpoint-path, the generator pathway remains "
            "from the base checkpoint and the understanding pathway is reloaded."
        ),
    )
    parser.add_argument(
        "--vlm-pretrained-checkpoint-format",
        default="",
        choices=["", "qwen3", "nemotron_3_dense_vl", "nemotron_3_llm"],
        help="Optional checkpoint format hint for --vlm-pretrained-backbone-path.",
    )
    parser.add_argument(
        "--wan-vae-path",
        default=str(COSMOS_ROOT / "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"),
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--project", default="aicity_track5")
    parser.add_argument("--group", default="track5")
    parser.add_argument("--name", default="c3super1680")
    parser.add_argument("--resume-checkpoint-path", default="")
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help="Resume model, optimizer, scheduler, trainer, and dataloader state from --resume-checkpoint-path.",
    )
    parser.add_argument("--no-strict-resume", action="store_true")
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument(
        "--conditioning-config", default="0:0.01,1:0.04,5:0.10,8:0.20,10:0.25,12:0.40"
    )
    parser.add_argument("--conditioning-fps", type=float, default=-1.0)
    parser.add_argument("--cfg-dropout-rate", type=float, default=0.05)
    parser.add_argument(
        "--tokenizer-chunk-duration",
        type=int,
        default=165,
        help="Pixel-frame chunk duration for the Wan2.2 VAE interface. Match the processed clip length.",
    )
    parser.add_argument(
        "--frame-selection-mode", default="first", choices=["first", "center", "random"]
    )
    parser.add_argument(
        "--temporal-interval-mode",
        default="force_one",
        choices=["force_one", "max_30fps", "entire_chunk"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=1680)
    parser.add_argument("--save-iter", type=int, default=100)
    parser.add_argument(
        "--scheduler-cycle-lengths",
        default="1000000",
        help="Optional comma-separated LR scheduler cycle lengths. Needed when max_iter exceeds the official 1000-step default cycle.",
    )
    parser.add_argument(
        "--scheduler-warm-up-steps",
        default="50",
        help="Optional comma-separated LR scheduler warmup steps matching --scheduler-cycle-lengths.",
    )
    parser.add_argument(
        "--eval-save-iter",
        type=int,
        default=20,
        help="Optional model-only checkpoint interval for eval/inference. Full resume checkpoints still use --save-iter.",
    )
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attach-vscode-debugger", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def ensure_inputs(args: argparse.Namespace) -> None:
    required = [
        Path(args.sft_toml),
        Path(args.base_checkpoint_path),
        Path(args.wan_vae_path),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in train_jsonl_paths(args):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.resume_checkpoint_path and not Path(args.resume_checkpoint_path).exists():
        raise FileNotFoundError(args.resume_checkpoint_path)
    if args.vlm_pretrained_backbone_path:
        path = Path(args.vlm_pretrained_backbone_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not any(path.glob("*.safetensors")):
            raise FileNotFoundError(f"No *.safetensors files found under {path}")


def main() -> int:
    args = parse_args()
    ensure_inputs(args)

    os.environ["DATASET_PATH"] = args.dataset_path
    os.environ["BASE_CHECKPOINT_PATH"] = args.base_checkpoint_path
    os.environ["WAN_VAE_PATH"] = args.wan_vae_path
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = args.output_root
    os.environ["AICITY_EVAL_SAVE_ITER"] = str(args.eval_save_iter)

    if args.deterministic:
        official_train._setup_deterministic_env_and_backends()

    extra_overrides = list(args.opts)
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=extra_overrides)

    config.job.project = args.project
    config.job.group = args.group
    config.job.name = args.name
    config.job.wandb_mode = args.wandb_mode
    config.trainer.max_iter = args.max_iter
    config.checkpoint.save_iter = args.save_iter
    if args.scheduler_cycle_lengths:
        config.scheduler.cycle_lengths = parse_int_list(args.scheduler_cycle_lengths)
    if args.scheduler_warm_up_steps:
        config.scheduler.warm_up_steps = parse_int_list(args.scheduler_warm_up_steps)
    if args.resume_checkpoint_path:
        config.checkpoint.load_path = args.resume_checkpoint_path
        config.checkpoint.load_training_state = args.resume_training_state
        config.checkpoint.strict_resume = not args.no_strict_resume
        if args.resume_training_state:
            config.checkpoint.keys_to_skip_loading = []
    config.optimizer.lr = args.lr
    model_config = model_runtime_config(config)
    model_config.tokenizer.chunk_duration = args.tokenizer_chunk_duration
    model_config.tokenizer.vae_path = args.wan_vae_path
    model_config.resolution = args.resolution
    if args.vlm_pretrained_backbone_path:
        pretrained = model_config.vlm_config.pretrained_weights
        pretrained.enabled = True
        pretrained.backbone_path = args.vlm_pretrained_backbone_path
        pretrained.credentials_path = ""
        pretrained.checkpoint_format = args.vlm_pretrained_checkpoint_format or None

    dataset = config.dataloader_train.dataloader.datasets.video.dataset
    dataset.resolution = args.resolution
    dataset.num_video_frames = args.num_video_frames
    dataset.conditioning_config = parse_conditioning_config(args.conditioning_config)
    dataset.conditioning_fps = args.conditioning_fps
    dataset.cfg_dropout_rate = args.cfg_dropout_rate
    dataset.frame_selection_mode = args.frame_selection_mode
    dataset.temporal_interval_mode = args.temporal_interval_mode
    dataset.jsonl_paths = train_jsonl_paths(args)

    rank_loader = config.dataloader_train.dataloader
    rank_loader.num_workers = args.num_workers
    if args.num_workers == 0:
        rank_loader.persistent_workers = False
        rank_loader.prefetch_factor = None

    official_args = argparse.Namespace(
        deterministic=args.deterministic,
        attach_vscode_debugger=args.attach_vscode_debugger,
        config=args.sft_toml,
        sft_toml=args.sft_toml,
        opts=extra_overrides,
    )

    if args.dryrun:
        log.info("Config:\n" + config.pretty_print(use_color=True))
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
        return 0

    official_train.launch(config, official_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
