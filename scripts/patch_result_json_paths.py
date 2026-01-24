#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class FileStats:
    result_json: Path
    total_images: int = 0
    updated: int = 0
    unchanged: int = 0
    missing: int = 0
    missing_basenames: list[str] | None = None


def _iter_result_json_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "result.json":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("result.json"))


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_write_json(path: Path, obj: Any, *, backup: bool) -> None:
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_bytes(path.read_bytes())

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _build_global_image_index(search_dirs: Iterable[Path]) -> dict[str, Path]:
    """Map basename -> absolute file path.

    If duplicates exist, keep the first one found.
    """
    index: dict[str, Path] = {}
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            name = p.name
            if name not in index:
                index[name] = p.resolve()
    return index


def _guess_search_dirs(repo_root: Path, data_poly_root: Path) -> list[Path]:
    # Prefer per-project images folders first (handled separately),
    # then fall back to common dataset image locations.
    candidates = [
        repo_root / "data" / "images",
        repo_root / "moss-detection" / "images",
        repo_root / "CrackForest-dataset" / "image",
        data_poly_root,
    ]
    return candidates


def _normalize_path_str(path: Path, *, mode: str, relative_to: Path | None) -> str:
    resolved = path.resolve()
    if mode == "absolute":
        return str(resolved)
    if mode == "posix":
        return resolved.as_posix()
    if mode == "relative":
        if relative_to is None:
            raise ValueError("relative_to is required when mode=relative")
        return os.path.relpath(str(resolved), str(relative_to.resolve()))
    raise ValueError(f"Unknown mode: {mode}")


def _extract_basename_candidates(value: str) -> list[str]:
    """Return possible basenames for an image path.

    Handles both POSIX and Windows-style separators. Also supports a common
    renaming pattern where "<hash>-<digits>.png" becomes "<digits>.png".
    """

    if "\\" in value and "/" not in value:
        name = PureWindowsPath(value).name
    else:
        name = Path(value).name

    candidates = [name]

    # Fallback: 7aea399c-0198.png -> 0198.png
    if "-" in name:
        tail = name.split("-")[-1]
        tail_path = Path(tail)
        if tail_path.suffix.lower() in IMAGE_EXTS and tail_path.stem.isdigit():
            if tail not in candidates:
                candidates.append(tail)

    return candidates


def patch_one_result_json(
    result_json: Path,
    *,
    mode: str,
    relative_to: Path | None,
    global_index: dict[str, Path],
    prefer_project_images: bool,
) -> tuple[Any, FileStats]:
    obj = _safe_load_json(result_json)

    stats = FileStats(result_json=result_json, missing_basenames=[])

    if not isinstance(obj, dict) or "images" not in obj or not isinstance(obj["images"], list):
        raise ValueError(f"Unsupported result.json format: {result_json}")

    project_images_dir = result_json.parent / "images"

    for item in obj["images"]:
        if not isinstance(item, dict):
            continue
        if "file_name" not in item:
            continue

        old_value = item.get("file_name")
        if not isinstance(old_value, str) or not old_value:
            continue

        stats.total_images += 1
        basename_candidates = _extract_basename_candidates(old_value)

        candidate: Path | None = None
        if prefer_project_images and project_images_dir.exists():
            for base in basename_candidates:
                p = project_images_dir / base
                if p.exists():
                    candidate = p
                    break

        if candidate is None:
            for base in basename_candidates:
                candidate = global_index.get(base)
                if candidate is not None:
                    break

        if candidate is None:
            stats.missing += 1
            stats.missing_basenames.append(basename_candidates[0])
            continue

        new_value = _normalize_path_str(candidate, mode=mode, relative_to=relative_to)
        if new_value == old_value:
            stats.unchanged += 1
            continue

        item["file_name"] = new_value
        stats.updated += 1

    return obj, stats


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Patch data_poly/**/result.json: replace each images[*].file_name with the current, existing image path. "
            "Matching is done by image basename (e.g. a18b6919-0001.png)."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(repo_root / "data_poly"),
        help="Path to data_poly (or a specific result.json).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to result.json (default is dry-run).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak backup before writing.",
    )
    parser.add_argument(
        "--mode",
        choices=["absolute", "posix", "relative"],
        default="absolute",
        help="How to store the new path string.",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        default=str(repo_root),
        help="Base path for --mode relative.",
    )
    parser.add_argument(
        "--no-prefer-project-images",
        action="store_true",
        help="Do not prioritize <project>/images/<basename> when patching.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    data_poly_root = root if root.is_dir() else root.parent

    result_files = _iter_result_json_files(root)
    if not result_files:
        print(f"No result.json found under: {root}")
        return 2

    search_dirs = _guess_search_dirs(repo_root, data_poly_root)
    global_index = _build_global_image_index(search_dirs)

    total_updated = 0
    total_missing = 0

    for result_json in result_files:
        patched_obj, stats = patch_one_result_json(
            result_json,
            mode=args.mode,
            relative_to=Path(args.relative_to) if args.mode == "relative" else None,
            global_index=global_index,
            prefer_project_images=not args.no_prefer_project_images,
        )

        total_updated += stats.updated
        total_missing += stats.missing

        status = "WRITE" if args.write else "DRY"
        print(
            f"[{status}] {result_json} :: total={stats.total_images} updated={stats.updated} "
            f"unchanged={stats.unchanged} missing={stats.missing}"
        )
        if stats.missing and stats.missing_basenames:
            sample = ", ".join(stats.missing_basenames[:5])
            more = "" if len(stats.missing_basenames) <= 5 else f" ...(+{len(stats.missing_basenames) - 5})"
            print(f"  missing examples: {sample}{more}")

        if args.write:
            _safe_write_json(result_json, patched_obj, backup=args.backup)

    print(f"Done. updated={total_updated}, missing={total_missing}, files={len(result_files)}")
    if not args.write:
        print("Tip: re-run with --write (and optionally --backup) to apply changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
