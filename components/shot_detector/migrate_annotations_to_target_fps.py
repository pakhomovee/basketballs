"""
Remap shot annotation frame indices from native/source video frames to the
logical timeline used by :class:`~video_reader.VideoReader` (``main.target_fps``).

Run after switching the annotation tool to VideoReader if older JSON was produced
at native FPS (e.g. 60) with one index per source frame.

Usage (from ``components/``)::

    python3 -m shot_detector.migrate_annotations_to_target_fps \\
        --dataset-dir shot_detector/dataset \\
        --annotations shot_detector/dataset/annotations.json

Use ``--dry-run`` to print changes without writing. Optional ``--config`` for YAML.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config import AppConfig, load_app_config, load_default_config
from shot_detector.annotation_tool import _annotation_from_record
from video_reader import VideoReader

FRAME_FIELDS = (
    "start_frame",
    "finish_frame",
    "shot_start_frame",
    "shot_end_frame",
    "make_start_frame",
    "make_end_frame",
)


def _remap_record(rec: dict, vr: VideoReader) -> dict:
    out = dict(rec)
    for key in FRAME_FIELDS:
        v = out.get(key)
        if v is None:
            continue
        out[key] = vr.nearest_logical_for_physical(int(v))
    out["fps"] = float(vr.fps)
    out["total_frames"] = int(vr.total_frames)
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def migrate_annotations(
    dataset_dir: Path,
    annotations_path: Path,
    *,
    cfg: AppConfig | None = None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Returns
    -------
    (updated_count, skipped_count, error_count)
    """
    if cfg is None:
        cfg = load_default_config()

    dataset_dir = dataset_dir.resolve()
    annotations_path = annotations_path.resolve()

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[dict] = list(payload.get("annotations", []))

    updated = 0
    skipped = 0
    errors = 0
    new_list: list[dict] = []

    target_fps = cfg.main.target_fps
    print(f"Target FPS (VideoReader / config): {target_fps}")

    for rec in records:
        clip_rel = rec.get("clip_path", "")
        clip_path = dataset_dir / clip_rel if clip_rel else None
        if not clip_rel or clip_path is None or not clip_path.is_file():
            print(f"[SKIP] missing clip: {clip_rel!r}")
            new_list.append(rec)
            skipped += 1
            continue

        try:
            with VideoReader(str(clip_path), target_fps=target_fps) as vr:
                new_rec = _remap_record(rec, vr)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            new_list.append(rec)
            skipped += 1
            continue
        except Exception as e:
            print(f"[ERROR] {clip_rel}: {e}")
            new_list.append(rec)
            errors += 1
            continue

        try:
            ann = _annotation_from_record(new_rec)
            ann.validate()
        except (TypeError, ValueError) as e:
            print(f"[WARN] {clip_rel}: validation after remap: {e}")
            errors += 1

        if any(rec.get(k) != new_rec.get(k) for k in FRAME_FIELDS if rec.get(k) is not None):
            for k in FRAME_FIELDS:
                if rec.get(k) is not None and rec.get(k) != new_rec.get(k):
                    print(f"  {clip_rel} {k}: {rec[k]} -> {new_rec[k]}")
        if rec.get("total_frames") != new_rec.get("total_frames") or rec.get("fps") != new_rec.get("fps"):
            print(
                f"  {clip_rel} meta: fps {rec.get('fps')} -> {new_rec.get('fps')}, "
                f"total_frames {rec.get('total_frames')} -> {new_rec.get('total_frames')}"
            )

        new_list.append(new_rec)
        updated += 1

    out_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "annotations": sorted(new_list, key=lambda r: r["clip_path"]),
    }

    if dry_run:
        print("[DRY-RUN] No file written.")
        return updated, skipped, errors

    backup = annotations_path.with_suffix(".json.bak")
    shutil.copy2(annotations_path, backup)
    print(f"Backup: {backup}")

    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {annotations_path}")

    return updated, skipped, errors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remap annotation frames to VideoReader logical indices")
    default_ds = Path(__file__).resolve().parent / "dataset"
    p.add_argument("--dataset-dir", type=Path, default=default_ds, help="Dataset root (contains data/* clips)")
    p.add_argument(
        "--annotations",
        type=Path,
        default=default_ds / "annotations.json",
        help="annotations.json path",
    )
    p.add_argument("--config", type=str, default=None, help="Optional AppConfig YAML")
    p.add_argument("--dry-run", action="store_true", help="Print diffs only, do not write")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_default_config() if args.config is None else load_app_config(args.config)
    u, s, e = migrate_annotations(
        dataset_dir=args.dataset_dir,
        annotations_path=args.annotations,
        cfg=cfg,
        dry_run=args.dry_run,
    )
    print(f"Done. remapped={u} skipped={s} warnings_or_validation_errors={e}")
