from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
WINDOW_NAME = "Shot Annotation Tool"


@dataclass
class ClipAnnotation:
    clip_path: str
    status: str = "ok"  # "ok" | "invalid"
    start_frame: int | None = None
    finish_frame: int | None = None
    shot_start_frame: int | None = None
    shot_end_frame: int | None = None
    make_start_frame: int | None = None
    make_end_frame: int | None = None
    fps: float | None = None
    total_frames: int | None = None
    updated_at: str | None = None

    def validate(self) -> None:
        if self.status == "invalid":
            return
        _check_order(self.start_frame, self.finish_frame, "clip")
        _check_segment_pair(self.shot_start_frame, self.shot_end_frame, "shot")
        _check_segment_pair(self.make_start_frame, self.make_end_frame, "make")
        _check_inside(self.shot_start_frame, self.start_frame, self.finish_frame, "shot_start_frame")
        _check_inside(self.shot_end_frame, self.start_frame, self.finish_frame, "shot_end_frame")
        _check_inside(self.make_start_frame, self.start_frame, self.finish_frame, "make_start_frame")
        _check_inside(self.make_end_frame, self.start_frame, self.finish_frame, "make_end_frame")

        if self.make_start_frame is not None and self.make_end_frame is not None:
            if self.shot_start_frame is None or self.shot_end_frame is None:
                raise ValueError("make segment is present but shot segment is missing")
            if not (self.shot_start_frame <= self.make_start_frame and self.make_end_frame <= self.shot_end_frame):
                raise ValueError(
                    f"make segment [{self.make_start_frame}, {self.make_end_frame}] "
                    f"is not inside shot segment [{self.shot_start_frame}, {self.shot_end_frame}]"
                )


def _check_order(start: int | None, end: int | None, name: str) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError(f"{name}_start_frame ({start}) > {name}_end_frame ({end})")


def _check_segment_pair(start: int | None, end: int | None, name: str) -> None:
    """
    Segment validity rules:
    - either both are None
    - or both are present and start < end
    """
    if start is None and end is None:
        return
    if start is None or end is None:
        raise ValueError(f"{name} segment must have both start and end (got {start}->{end})")
    if start >= end:
        raise ValueError(f"{name} segment start must be < end (got {start}->{end})")


def _check_inside(value: int | None, start: int | None, finish: int | None, name: str) -> None:
    if value is None or start is None or finish is None:
        return
    if value < start or value > finish:
        raise ValueError(f"{name} ({value}) is outside clip range [{start}, {finish}]")


def _discover_clips(clips_root: Path) -> list[Path]:
    clips = [p for p in clips_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(clips)


def _load_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("annotations", [])
    return {rec["clip_path"]: rec for rec in records}


def _save_annotations(path: Path, records: dict[str, dict]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "annotations": sorted(records.values(), key=lambda r: r["clip_path"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _draw_help_overlay(
    frame,
    clip_name: str,
    frame_idx: int,
    total_frames: int,
    ann: ClipAnnotation,
    playing: bool,
) -> None:
    lines = [
        f"Clip: {clip_name}",
        f"Frame: {frame_idx + 1}/{max(total_frames, 1)}",
        f"Mode: {'PLAY' if playing else 'PAUSE'}",
        f"Status: {ann.status}",
        f"Clip range: {ann.start_frame} -> {ann.finish_frame}",
        f"Shot: {ann.shot_start_frame} -> {ann.shot_end_frame}",
        f"Make: {ann.make_start_frame} -> {ann.make_end_frame}",
        "Keys:",
        "space play/pause | a/d -/+1 | j/l -/+10 | u/o -/+50",
        "z clip_start | x clip_end",
        "1 shot_start | 2 shot_end | 3 make_start | 4 make_end",
        "i toggle invalid | r reset marks | s save+next | n skip | b back | q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def _jump(cap: cv2.VideoCapture, target: int, total_frames: int) -> int:
    target = max(0, min(total_frames - 1, target)) if total_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    return target


def _annotate_single_clip(clip_path: Path, ann: ClipAnnotation) -> tuple[str, ClipAnnotation]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open clip: {clip_path}")
        return "skip", ann

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ann.fps = fps
    ann.total_frames = total_frames
    if ann.start_frame is None:
        ann.start_frame = 0
    if ann.finish_frame is None:
        ann.finish_frame = max(total_frames - 1, 0)
    ann.start_frame = max(0, min(ann.start_frame, max(total_frames - 1, 0)))
    ann.finish_frame = max(0, min(ann.finish_frame, max(total_frames - 1, 0)))
    if ann.start_frame > ann.finish_frame:
        ann.start_frame = 0
        ann.finish_frame = max(total_frames - 1, 0)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = ann.start_frame
    playing = False
    last_frame = None

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                playing = False
                if last_frame is None:
                    break
                frame = last_frame.copy()
            else:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                last_frame = frame.copy()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()

        shown = frame.copy()
        _draw_help_overlay(shown, clip_path.name, frame_idx, total_frames, ann, playing)
        cv2.imshow(WINDOW_NAME, shown)

        wait_ms = max(1, int(1000.0 / max(fps, 1.0))) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 255:
            continue

        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            playing = False
            frame_idx = _jump(cap, frame_idx - 1, total_frames)
        elif key == ord("d"):
            playing = False
            frame_idx = _jump(cap, frame_idx + 1, total_frames)
        elif key == ord("j"):
            playing = False
            frame_idx = _jump(cap, frame_idx - 10, total_frames)
        elif key == ord("l"):
            playing = False
            frame_idx = _jump(cap, frame_idx + 10, total_frames)
        elif key == ord("u"):
            playing = False
            frame_idx = _jump(cap, frame_idx - 50, total_frames)
        elif key == ord("o"):
            playing = False
            frame_idx = _jump(cap, frame_idx + 50, total_frames)
        elif key == ord("z"):
            ann.start_frame = frame_idx
            if ann.finish_frame is not None and ann.start_frame > ann.finish_frame:
                ann.finish_frame = ann.start_frame
        elif key == ord("x"):
            ann.finish_frame = frame_idx
            if ann.start_frame is not None and ann.start_frame > ann.finish_frame:
                ann.start_frame = ann.finish_frame
        elif key == ord("1"):
            ann.shot_start_frame = frame_idx
        elif key == ord("2"):
            ann.shot_end_frame = frame_idx
        elif key == ord("3"):
            ann.make_start_frame = frame_idx
        elif key == ord("4"):
            ann.make_end_frame = frame_idx
        elif key == ord("i"):
            ann.status = "invalid" if ann.status != "invalid" else "ok"
        elif key == ord("r"):
            ann.start_frame = 0
            ann.finish_frame = max(total_frames - 1, 0)
            ann.shot_start_frame = None
            ann.shot_end_frame = None
            ann.make_start_frame = None
            ann.make_end_frame = None
            ann.status = "ok"
        elif key == ord("n"):
            cap.release()
            return "skip", ann
        elif key == ord("b"):
            cap.release()
            return "back", ann
        elif key == ord("s"):
            try:
                ann.validate()
            except ValueError as e:
                print(f"[ERROR] Invalid annotation: {e}")
                continue
            ann.updated_at = datetime.now(timezone.utc).isoformat()
            cap.release()
            return "save", ann
        elif key == ord("q"):
            cap.release()
            return "quit", ann

    cap.release()
    return "skip", ann


def _annotation_from_record(record: dict) -> ClipAnnotation:
    """Create ClipAnnotation from a JSON record (ignores unknown keys)."""
    allowed = {
        "clip_path",
        "status",
        "start_frame",
        "finish_frame",
        "shot_start_frame",
        "shot_end_frame",
        "make_start_frame",
        "make_end_frame",
        "fps",
        "total_frames",
        "updated_at",
    }
    payload = {k: record.get(k) for k in allowed if k in record}
    return ClipAnnotation(**payload)  # type: ignore[arg-type]


def run_annotation_tool(
    dataset_dir: Path,
    annotations_path: Path,
    *,
    seed: int | None = None,
) -> None:
    clips_root = dataset_dir / "data"
    clips = _discover_clips(clips_root)
    if not clips:
        raise RuntimeError(f"No video clips found in {clips_root}")

    saved = _load_annotations(annotations_path)
    migrated = False
    for rec in saved.values():
        if "start_frame" in rec and "finish_frame" in rec:
            continue
        total = int(rec.get("total_frames") or 0)
        rec["start_frame"] = 0
        rec["finish_frame"] = max(total - 1, 0)
        migrated = True
    if migrated:
        _save_annotations(annotations_path, saved)
        print(f"Migrated existing annotations with clip range defaults in {annotations_path}")
    rel_clips = [str(p.relative_to(dataset_dir)) for p in clips]
    unannotated = [cp for cp in rel_clips if cp not in saved]
    rng = random.Random(seed)
    rng.shuffle(unannotated)

    print(f"Found clips: {len(clips)}")
    print(f"Already annotated: {len(saved)}")
    non_invalid = sum(1 for rec in saved.values() if rec.get("status") != "invalid")
    with_shot = sum(
        1
        for rec in saved.values()
        if rec.get("status") != "invalid"
        and rec.get("shot_start_frame") is not None
        and rec.get("shot_end_frame") is not None
    )
    with_make = sum(
        1
        for rec in saved.values()
        if rec.get("status") != "invalid"
        and rec.get("make_start_frame") is not None
        and rec.get("make_end_frame") is not None
    )
    print(f"Stats (saved): non-invalid={non_invalid}, with_shot={with_shot}, with_make={with_make}")
    print(f"Remaining: {len(unannotated)}")
    if seed is not None:
        print(f"Random seed: {seed}")

    i = 0
    while 0 <= i < len(unannotated):
        rel_path = unannotated[i]
        clip_path = dataset_dir / rel_path
        ann = ClipAnnotation(clip_path=rel_path)

        print(f"\nAnnotating ({i + 1}/{len(unannotated)}): {rel_path}")
        action, ann = _annotate_single_clip(clip_path, ann)

        if action == "save":
            saved[rel_path] = asdict(ann)
            _save_annotations(annotations_path, saved)
            print(f"[SAVED] {rel_path}")
            i += 1
        elif action == "back":
            i = max(0, i - 1)
            # If we moved back to a clip that was already saved, open it with existing marks.
            prev_path = unannotated[i]
            if prev_path in saved:
                prev_ann = _annotation_from_record(saved[prev_path])
                print(f"\nBack to ({i + 1}/{len(unannotated)}): {prev_path}")
                action2, edited = _annotate_single_clip(dataset_dir / prev_path, prev_ann)
                if action2 == "save":
                    saved[prev_path] = asdict(edited)
                    _save_annotations(annotations_path, saved)
                    print(f"[SAVED] {prev_path}")
                elif action2 == "quit":
                    break
            # Continue the loop at the same i (either edited or not).
        elif action == "quit":
            break
        else:
            print(f"[SKIP] {rel_path}")
            i += 1

    cv2.destroyAllWindows()
    print("Annotation session finished.")
    print(f"Saved annotations file: {annotations_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shot detector clip annotation tool")
    default_dataset_dir = Path(__file__).resolve().parent / "dataset"
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory containing dataset.csv and data/* clips",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=default_dataset_dir / "annotations.json",
        help="Output JSON file with annotations",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random clip order")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_annotation_tool(dataset_dir=args.dataset_dir, annotations_path=args.annotations, seed=args.seed)

