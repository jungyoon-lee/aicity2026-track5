#!/usr/bin/env python3
"""Export only LoRA tensors from a Cosmos DCP model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Iteration directory or its model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output .safetensors path.")
    parser.add_argument("--config-output", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        default="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
    )
    return parser.parse_args()


def model_dir(path: Path) -> Path:
    candidate = path / "model"
    return candidate if candidate.is_dir() else path


def main() -> int:
    args = parse_args()
    checkpoint = model_dir(args.checkpoint.resolve())
    reader = FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata()

    tensors: dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if "lora_" not in key:
            continue
        properties = getattr(value, "properties", None)
        dtype = getattr(properties, "dtype", None)
        size = tuple(int(item) for item in getattr(value, "size", ()))
        if dtype is None or not size:
            raise TypeError(f"Unsupported DCP metadata for {key}: {value!r}")
        tensors[key] = torch.empty(size, dtype=dtype, device="cpu")

    if not tensors:
        raise RuntimeError(f"No LoRA tensors found in {checkpoint}")
    dcp.load(state_dict=tensors, storage_reader=reader)
    tensors = {key: value.contiguous() for key, value in sorted(tensors.items())}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "format": "cosmos3-native-lora",
            "source_checkpoint": (
                checkpoint.parent.name if checkpoint.name == "model" else checkpoint.name
            ),
            "rank": str(args.rank),
            "alpha": str(args.alpha),
            "target_modules": args.target_modules,
        },
    )

    config_path = args.config_output or args.output.with_name("adapter_config.json")
    payload = {
        "format": "cosmos3-native-lora",
        "base_model": "nvidia/Cosmos3-Super",
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        "tensor_count": len(tensors),
        "parameter_count": sum(value.numel() for value in tensors.values()),
        "dtype_counts": {},
    }
    for value in tensors.values():
        name = str(value.dtype).removeprefix("torch.")
        payload["dtype_counts"][name] = payload["dtype_counts"].get(name, 0) + 1
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
