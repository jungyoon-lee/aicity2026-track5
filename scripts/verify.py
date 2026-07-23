#!/usr/bin/env python3
"""Validate an AI City Track 5 prediction directory or ZIP."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prediction-root", type=Path)
    group.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def expected_cases(test_root: Path) -> dict[str, int]:
    result = {}
    for case_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        caption = json.loads((case_dir / "caption.json").read_text(encoding="utf-8"))
        result[case_dir.name] = int(caption["frame length"])
    return result


def verify_directory(args: argparse.Namespace, expected: dict[str, int]) -> list[str]:
    errors = []
    actual_cases = {
        path.name for path in args.prediction_root.iterdir() if path.is_dir()
    }
    if actual_cases != set(expected):
        errors.append(
            f"case set mismatch missing={sorted(set(expected) - actual_cases)} extra={sorted(actual_cases - set(expected))}"
        )
    for case_id, count in expected.items():
        case_dir = args.prediction_root / case_id
        paths = list(case_dir.glob("*.png"))
        invalid_names = sorted(path.name for path in paths if not path.stem.isdigit())
        if invalid_names:
            errors.append(f"{case_id}: non-numeric PNG names: {invalid_names[:5]}")
        names = sorted(
            (path.name for path in paths if path.stem.isdigit()),
            key=lambda name: int(Path(name).stem),
        )
        wanted = [f"{index}.png" for index in range(count)]
        if names != wanted:
            errors.append(
                f"{case_id}: expected {count} contiguous PNGs, got {len(names)}"
            )
            continue
        for name in wanted:
            with Image.open(case_dir / name) as image:
                if image.size != (args.width, args.height):
                    errors.append(f"{case_id}/{name}: resolution {image.size}")
    return errors


def verify_zip(args: argparse.Namespace, expected: dict[str, int]) -> list[str]:
    errors = []
    with zipfile.ZipFile(args.zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [PurePosixPath(info.filename) for info in infos]
        if any(path.is_absolute() or ".." in path.parts for path in names):
            errors.append("ZIP contains an unsafe path")
        pngs = [path for path in names if path.suffix.lower() == ".png"]
        extras = [str(path) for path in names if path.suffix.lower() != ".png"]
        if extras:
            errors.append(f"ZIP contains non-PNG files: {extras[:5]}")
        actual: dict[str, list[int]] = {}
        info_by_name = {PurePosixPath(info.filename): info for info in infos}
        for path in pngs:
            if (
                len(path.parts) != 3
                or path.parts[0] != "prediction"
                or not path.stem.isdigit()
            ):
                errors.append(f"invalid ZIP path: {path}")
                continue
            actual.setdefault(path.parts[1], []).append(int(path.stem))
        if set(actual) != set(expected):
            errors.append(
                f"case set mismatch missing={sorted(set(expected) - set(actual))} extra={sorted(set(actual) - set(expected))}"
            )
        for case_id, count in expected.items():
            indices = sorted(actual.get(case_id, []))
            if indices != list(range(count)):
                errors.append(
                    f"{case_id}: expected {count} contiguous PNGs, got {len(indices)}"
                )
                continue
            for index in range(count):
                path = PurePosixPath("prediction", case_id, f"{index}.png")
                with (
                    archive.open(info_by_name[path]) as stream,
                    Image.open(io.BytesIO(stream.read())) as image,
                ):
                    if image.size != (args.width, args.height):
                        errors.append(f"{path}: resolution {image.size}")
    return errors


def main() -> int:
    args = parse_args()
    expected = expected_cases(args.test_root)
    errors = (
        verify_zip(args, expected)
        if args.zip_path
        else verify_directory(args, expected)
    )
    result = {
        "ok": not errors,
        "cases": len(expected),
        "expected_pngs": sum(expected.values()),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
