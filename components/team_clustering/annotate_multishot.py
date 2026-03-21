from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

_repo_root = Path(__file__).resolve().parents[2]
_components_root = _repo_root / "components"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_components_root) not in sys.path:
    sys.path.insert(0, str(_components_root))

from cache import load_detections_cache, save_detections_cache
from common.classes.player import PlayersDetections
from detector.detector import Detector, get_video_players_detections
from team_clustering.embedding import extract_player_embeddings
from team_clustering.shared import DEFAULT_SEG_MODEL, REPO_ROOT, resolve_repo_path
from team_clustering.team_clustering import TeamClustering
from tracking import FlowTracker

TEAM_A = 0
TEAM_B = 1
TEAM_SIZE = 5
TOTAL_PLAYERS = 10
FRAMES_PER_SAMPLE = 5
WINDOW_NAME = "Multishot Annotator"
PANEL_WIDTH = 440
FONT = cv2.FONT_HERSHEY_SIMPLEX

TEAM_COLORS = {
    TEAM_A: (80, 190, 255),
    TEAM_B: (255, 170, 70),
    None: (180, 180, 180),
}
SELECTED_COLOR = (80, 255, 120)

DEFAULT_MULTISHOT_ROOT = REPO_ROOT / "dataset" / "team_clustering_benchmark" / "multishot"
DEFAULT_DETECTOR_MODEL = REPO_ROOT / "models" / "best-4.pt"


@dataclass(frozen=True)
class FrameDetection:
    source_track_id: int
    bbox: tuple[int, int, int, int]
    initial_team_id: int | None


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    image_name: str
    image_path: Path
    detections: list[FrameDetection]


@dataclass(frozen=True)
class SampleBatch:
    sample_id: str
    sampled_frames: list[SampledFrame]


def canonical_team(canonical_id: int | None) -> int | None:
    if canonical_id is None or not 1 <= canonical_id <= TOTAL_PLAYERS:
        return None
    return TEAM_A if canonical_id <= TEAM_SIZE else TEAM_B


def team_slot(canonical_id: int) -> int:
    return ((canonical_id - 1) % TEAM_SIZE) + 1


def first_free_canonical_id(source_to_canonical: dict[int, int | None], team_id: int) -> int | None:
    start = 1 if team_id == TEAM_A else TEAM_SIZE + 1
    stop = TEAM_SIZE if team_id == TEAM_A else TOTAL_PLAYERS
    used = {canonical_id for canonical_id in source_to_canonical.values() if canonical_id is not None}
    for canonical_id in range(start, stop + 1):
        if canonical_id not in used:
            return canonical_id
    return None


def swap_canonical_ids(source_to_canonical: dict[int, int | None], left: int, right: int) -> None:
    if left == right:
        return
    for source_track_id, canonical_id in list(source_to_canonical.items()):
        if canonical_id == left:
            source_to_canonical[source_track_id] = right
        elif canonical_id == right:
            source_to_canonical[source_track_id] = left


def move_canonical_to_team(source_to_canonical: dict[int, int | None], canonical_id: int, target_team: int) -> int:
    target_canonical = team_slot(canonical_id) + (0 if target_team == TEAM_A else TEAM_SIZE)
    swap_canonical_ids(source_to_canonical, canonical_id, target_canonical)
    return target_canonical


def initial_team_by_track(sampled_frames: list[SampledFrame]) -> dict[int, int | None]:
    counts: dict[int, dict[int | None, int]] = {}
    for sampled_frame in sampled_frames:
        for detection in sampled_frame.detections:
            counts.setdefault(detection.source_track_id, {})
            counts[detection.source_track_id][detection.initial_team_id] = (
                counts[detection.source_track_id].get(detection.initial_team_id, 0) + 1
            )

    majority: dict[int, int | None] = {}
    for source_track_id, team_counts in counts.items():
        majority[source_track_id] = max(team_counts.items(), key=lambda item: item[1])[0]
    return majority


def build_initial_mapping(sampled_frames: list[SampledFrame]) -> dict[int, int | None]:
    source_track_ids = sorted(
        {
            detection.source_track_id
            for sampled_frame in sampled_frames
            for detection in sampled_frame.detections
            if detection.source_track_id > 0
        }
    )
    team_by_track = initial_team_by_track(sampled_frames)
    source_to_canonical = {source_track_id: None for source_track_id in source_track_ids}

    for team_id in (TEAM_A, TEAM_B):
        team_tracks = [source_track_id for source_track_id in source_track_ids if team_by_track.get(source_track_id) == team_id]
        start = 1 if team_id == TEAM_A else TEAM_SIZE + 1
        for offset, source_track_id in enumerate(team_tracks[:TEAM_SIZE]):
            source_to_canonical[source_track_id] = start + offset

    free_slots = [canonical_id for canonical_id in range(1, TOTAL_PLAYERS + 1) if canonical_id not in source_to_canonical.values()]
    for source_track_id in source_track_ids:
        if source_to_canonical[source_track_id] is None and free_slots:
            source_to_canonical[source_track_id] = free_slots.pop(0)
    return source_to_canonical


def bbox_to_yolo(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def write_yolo_annotations(sampled_frames: list[SampledFrame], source_to_canonical: dict[int, int | None], labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for label_path in labels_dir.glob("*.txt"):
        label_path.unlink()

    for sampled_frame in sampled_frames:
        image = cv2.imread(str(sampled_frame.image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read sampled frame: {sampled_frame.image_path}")

        lines = []
        for detection in sampled_frame.detections:
            canonical_id = source_to_canonical.get(detection.source_track_id)
            if canonical_id is None:
                continue
            cx, cy, bw, bh = bbox_to_yolo(detection.bbox, image.shape)
            lines.append(f"{canonical_id - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (labels_dir / sampled_frame.image_name.replace(".jpg", ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))


def manifest_path(multishot_root: Path, sample_id: str) -> Path:
    return multishot_root / "manifests" / f"{sample_id}.json"


def save_manifest(
    multishot_root: Path,
    sample_id: str,
    video_path: Path,
    frame_stride: int,
    sampled_frames: list[SampledFrame],
    source_to_canonical: dict[int, int | None],
    initial_mapping: dict[int, int | None],
) -> None:
    path = manifest_path(multishot_root, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "frame_stride": frame_stride,
        "sample_id": sample_id,
        "source_to_canonical": source_to_canonical,
        "initial_mapping": initial_mapping,
        "frames": [
            {
                "frame_index": sampled_frame.frame_index,
                "image_name": sampled_frame.image_name,
                "detections": [
                    {
                        "source_track_id": detection.source_track_id,
                        "bbox": list(detection.bbox),
                        "initial_team_id": detection.initial_team_id,
                    }
                    for detection in sampled_frame.detections
                ],
            }
            for sampled_frame in sampled_frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_manifest(multishot_root: Path, sample_id: str) -> tuple[Path, int, list[SampledFrame], dict[int, int | None], dict[int, int | None]]:
    path = manifest_path(multishot_root, sample_id)
    payload = json.loads(path.read_text())
    images_dir = multishot_root / "images" / sample_id
    sampled_frames = [
        SampledFrame(
            frame_index=frame_data["frame_index"],
            image_name=frame_data["image_name"],
            image_path=images_dir / frame_data["image_name"],
            detections=[
                FrameDetection(
                    source_track_id=detection["source_track_id"],
                    bbox=tuple(detection["bbox"]),
                    initial_team_id=detection["initial_team_id"],
                )
                for detection in frame_data["detections"]
            ],
        )
        for frame_data in payload["frames"]
    ]
    source_to_canonical = {int(key): value for key, value in payload["source_to_canonical"].items()}
    initial_mapping = {int(key): value for key, value in payload["initial_mapping"].items()}
    return Path(payload["video_path"]), int(payload["frame_stride"]), sampled_frames, source_to_canonical, initial_mapping


def next_sample_id(multishot_root: Path) -> str:
    existing = []
    for root_name in ("images", "labels"):
        root = multishot_root / root_name
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.isdigit():
                existing.append(int(child.name))
    return str(max(existing, default=0) + 1)


def chunk_list(items: list[SampledFrame], chunk_size: int, padding_seed: int = 0) -> list[list[SampledFrame]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]
    if not chunks:
        return []

    last_chunk = chunks[-1]
    if len(last_chunk) == chunk_size:
        return chunks

    previous_frames = [frame for chunk in chunks[:-1] for frame in chunk]
    if not previous_frames:
        raise ValueError(
            f"Need at least {chunk_size} sampled frames to create a multishot sample; got {len(items)}"
        )

    missing = chunk_size - len(last_chunk)
    if len(previous_frames) < missing:
        raise ValueError(
            f"Need at least {missing} previous sampled frames to pad the final batch; got {len(previous_frames)}"
        )

    rng = random.Random(padding_seed)
    padding_frames = rng.sample(previous_frames, missing)
    chunks[-1] = [*last_chunk, *padding_frames]
    return chunks


def _video_frame_width(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return width if width > 0 else None


def bootstrap_players_detections(
    video_path: Path,
    seg_model: Path,
    detector_model: Path | None,
    use_cache: bool,
) -> PlayersDetections:
    detections = load_detections_cache(str(video_path), str(seg_model)) if use_cache else None
    if detections is None:
        detector = Detector(model_path=detector_model or DEFAULT_DETECTOR_MODEL)
        video_detections = detector.detect_video(str(video_path))
        detections = get_video_players_detections(video_detections)
        extract_player_embeddings(str(video_path), detections, seg_model=str(seg_model))
        if use_cache:
            save_detections_cache(str(video_path), detections, str(seg_model))

    tracker = FlowTracker(num_tracks=10, frame_width=_video_frame_width(str(video_path)))
    tracker.track(detections)

    team_clustering = TeamClustering()
    team_clustering.run(detections)
    return detections


def extract_sampled_frames(
    video_path: Path,
    detections: PlayersDetections,
    frame_stride: int,
) -> list[SampledFrame]:
    sampled_frames: list[SampledFrame] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_stem = video_path.stem.replace(" ", "_")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, frame_stride) == 0:
            image_name = f"{video_stem}-f{frame_idx:06d}.jpg"
            frame_detections = [
                FrameDetection(
                    source_track_id=player.player_id,
                    bbox=tuple(map(int, player.bbox)),
                    initial_team_id=player.team_id,
                )
                for player in detections.get(frame_idx, [])
                if player.player_id > 0 and len(player.bbox) >= 4
            ]
            sampled_frames.append(
                SampledFrame(
                    frame_index=frame_idx,
                    image_name=image_name,
                    image_path=video_path.parent / image_name,
                    detections=frame_detections,
                )
            )
        frame_idx += 1

    cap.release()
    return sampled_frames


def materialize_sample_batches(
    video_path: Path,
    multishot_root: Path,
    starting_sample_id: str,
    sampled_frames: list[SampledFrame],
    overwrite: bool,
    frames_per_sample: int = FRAMES_PER_SAMPLE,
    padding_seed: int = 0,
) -> list[SampleBatch]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_images: dict[int, object] = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if any(sampled_frame.frame_index == frame_index for sampled_frame in sampled_frames):
                frame_images[frame_index] = frame.copy()
    finally:
        cap.release()

    sample_batches: list[SampleBatch] = []
    base_sample_id = int(starting_sample_id)
    for batch_offset, frame_chunk in enumerate(chunk_list(sampled_frames, frames_per_sample, padding_seed=padding_seed)):
        sample_id = str(base_sample_id + batch_offset)
        images_dir = multishot_root / "images" / sample_id
        labels_dir = multishot_root / "labels" / sample_id
        if overwrite and images_dir.exists():
            shutil.rmtree(images_dir)
        if overwrite and labels_dir.exists():
            shutil.rmtree(labels_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        batch_frames: list[SampledFrame] = []
        for sampled_frame in frame_chunk:
            image_path = images_dir / sampled_frame.image_name
            frame = frame_images.get(sampled_frame.frame_index)
            if frame is None:
                raise RuntimeError(f"Missing decoded frame for sampled frame {sampled_frame.frame_index}")
            cv2.imwrite(str(image_path), frame)
            batch_frames.append(
                SampledFrame(
                    frame_index=sampled_frame.frame_index,
                    image_name=sampled_frame.image_name,
                    image_path=image_path,
                    detections=sampled_frame.detections,
                )
            )
        sample_batches.append(SampleBatch(sample_id=sample_id, sampled_frames=batch_frames))

    return sample_batches


def hit_test(detections: list[FrameDetection], x: int, y: int) -> int | None:
    best_source = None
    best_area = None
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        area = (x2 - x1) * (y2 - y1)
        if best_area is None or area < best_area:
            best_area = area
            best_source = detection.source_track_id
    return best_source


def remove_detection_by_source(sampled_frame: SampledFrame, source_track_id: int) -> bool:
    remaining_detections = [
        detection for detection in sampled_frame.detections if detection.source_track_id != source_track_id
    ]
    if len(remaining_detections) == len(sampled_frame.detections):
        return False
    sampled_frame.detections[:] = remaining_detections
    return True


def draw_text(canvas: cv2.typing.MatLike, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(canvas, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


class MultishotAnnotator:
    def __init__(
        self,
        multishot_root: Path,
        sample_id: str,
        video_path: Path,
        frame_stride: int,
        sampled_frames: list[SampledFrame],
        source_to_canonical: dict[int, int | None],
        initial_mapping: dict[int, int | None],
    ):
        self.multishot_root = multishot_root
        self.sample_id = sample_id
        self.video_path = video_path
        self.frame_stride = frame_stride
        self.sampled_frames = sampled_frames
        self.source_to_canonical = dict(source_to_canonical)
        self.initial_mapping = dict(initial_mapping)
        self.current_index = 0
        self.selected_source_track_id: int | None = None
        self.status = "Click a bbox to select a source track."
        self.show_help = True

    @property
    def labels_dir(self) -> Path:
        return self.multishot_root / "labels" / self.sample_id

    @property
    def images_dir(self) -> Path:
        return self.multishot_root / "images" / self.sample_id

    def current_frame(self) -> SampledFrame:
        return self.sampled_frames[self.current_index]

    def save(self) -> None:
        write_yolo_annotations(self.sampled_frames, self.source_to_canonical, self.labels_dir)
        save_manifest(
            self.multishot_root,
            self.sample_id,
            self.video_path,
            self.frame_stride,
            self.sampled_frames,
            self.source_to_canonical,
            self.initial_mapping,
        )
        self.status = f"Saved annotations for sample {self.sample_id}."

    def select_source(self, source_track_id: int | None) -> None:
        self.selected_source_track_id = source_track_id
        if source_track_id is None:
            self.status = "No detection selected."
            return
        canonical_id = self.source_to_canonical.get(source_track_id)
        self.status = f"Selected source track {source_track_id}, canonical player {canonical_id}."

    def assign_selected(self, canonical_id: int | None) -> None:
        if self.selected_source_track_id is None:
            self.status = "Select a detection before reassigning it."
            return
        self.source_to_canonical[self.selected_source_track_id] = canonical_id
        self.status = f"Mapped source track {self.selected_source_track_id} to canonical player {canonical_id}."

    def move_selected_to_team(self, target_team: int) -> None:
        if self.selected_source_track_id is None:
            self.status = "Select a detection before changing its team."
            return
        canonical_id = self.source_to_canonical.get(self.selected_source_track_id)
        if canonical_id is None:
            free_canonical = first_free_canonical_id(self.source_to_canonical, target_team)
            if free_canonical is None:
                self.status = f"Team {'A' if target_team == TEAM_A else 'B'} has no free canonical slots."
                return
            self.assign_selected(free_canonical)
            return

        new_canonical = move_canonical_to_team(self.source_to_canonical, canonical_id, target_team)
        self.status = (
            f"Moved canonical player {canonical_id} to team {'A' if target_team == TEAM_A else 'B'} as {new_canonical}."
        )

    def reset_selected(self) -> None:
        if self.selected_source_track_id is None:
            self.status = "Select a detection before resetting it."
            return
        self.source_to_canonical[self.selected_source_track_id] = self.initial_mapping.get(self.selected_source_track_id)
        self.status = f"Reset source track {self.selected_source_track_id}."

    def reset_all(self) -> None:
        self.source_to_canonical = dict(self.initial_mapping)
        self.status = "Reset all mappings to the bootstrap state."

    def remove_selected_from_current_frame(self) -> None:
        if self.selected_source_track_id is None:
            self.status = "Select a detection before removing it."
            return

        source_track_id = self.selected_source_track_id
        removed = remove_detection_by_source(self.current_frame(), source_track_id)
        if not removed:
            self.status = f"Source track {source_track_id} is not present in the current frame."
            return

        self.selected_source_track_id = None
        self.status = f"Removed source track {source_track_id} from the current frame only."

    def render(self) -> cv2.typing.MatLike:
        sampled_frame = self.current_frame()
        frame = cv2.imread(str(sampled_frame.image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read sampled frame: {sampled_frame.image_path}")

        canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=(22, 22, 22))

        for detection in sampled_frame.detections:
            canonical_id = self.source_to_canonical.get(detection.source_track_id)
            team_id = canonical_team(canonical_id)
            color = TEAM_COLORS[team_id]
            thickness = 2
            if detection.source_track_id == self.selected_source_track_id:
                color = SELECTED_COLOR
                thickness = 3

            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"S{detection.source_track_id} -> P{canonical_id if canonical_id is not None else '-'}"
            draw_text(canvas, label, (x1, max(18, y1 - 6)), color)

        panel_x = frame.shape[1] + 16
        draw_text(canvas, f"Sample {self.sample_id}", (panel_x, 28), (245, 245, 245), 0.7)
        draw_text(canvas, f"Video: {self.video_path.name}", (panel_x, 56), (200, 200, 200), 0.5)
        draw_text(
            canvas,
            f"Frame {self.current_index + 1}/{len(self.sampled_frames)}  (video frame {sampled_frame.frame_index})",
            (panel_x, 82),
            (200, 200, 200),
            0.5,
        )

        selected_canonical = (
            self.source_to_canonical.get(self.selected_source_track_id) if self.selected_source_track_id is not None else None
        )
        draw_text(
            canvas,
            f"Selected: S{self.selected_source_track_id} -> P{selected_canonical}" if self.selected_source_track_id else "Selected: none",
            (panel_x, 112),
            (245, 245, 245),
            0.55,
        )
        draw_text(canvas, self.status[:58], (panel_x, 138), (160, 220, 255), 0.5)

        mapping_y = 172
        draw_text(canvas, "Current Mapping", (panel_x, mapping_y), (245, 245, 245), 0.65)
        mapping_y += 24
        for source_track_id in sorted(self.source_to_canonical):
            canonical_id = self.source_to_canonical[source_track_id]
            team_id = canonical_team(canonical_id)
            team_name = "A" if team_id == TEAM_A else "B" if team_id == TEAM_B else "-"
            color = SELECTED_COLOR if source_track_id == self.selected_source_track_id else TEAM_COLORS[team_id]
            draw_text(canvas, f"S{source_track_id:>2} -> P{str(canonical_id):>2}  T:{team_name}", (panel_x, mapping_y), color, 0.52)
            mapping_y += 20
            if mapping_y > frame.shape[0] - 170:
                break

        if self.show_help:
            help_lines = [
                "Controls",
                "a / d : prev / next frame",
                "click : select detection",
                "right click / f : remove detection from this frame",
                "1..9,0 : map selected to player 1..10",
                "q / e : move selected player to team A / B",
                "u : unassign selected from export",
                "r / R : reset selected / all",
                "s : save labels + manifest",
                "h : toggle this help",
                "Esc : save and exit",
            ]
            help_y = frame.shape[0] - 170
            for idx, line in enumerate(help_lines):
                color = (245, 245, 245) if idx == 0 else (200, 200, 200)
                scale = 0.62 if idx == 0 else 0.5
                draw_text(canvas, line, (panel_x, help_y + idx * 20), color, scale)

        return canvas

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        sampled_frame = self.current_frame()
        if x >= cv2.imread(str(sampled_frame.image_path)).shape[1]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.select_source(hit_test(sampled_frame.detections, x, y))
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            source_track_id = hit_test(sampled_frame.detections, x, y)
            if source_track_id is None:
                self.status = "Right click on a detection to remove it from the current frame."
                return
            self.selected_source_track_id = source_track_id
            self.remove_selected_from_current_frame()

    def handle_key(self, key: int) -> bool:
        if key in (27, ord("x")):
            self.save()
            return False
        if key == ord("a"):
            self.current_index = max(0, self.current_index - 1)
            return True
        if key == ord("d"):
            self.current_index = min(len(self.sampled_frames) - 1, self.current_index + 1)
            return True
        if key == ord("h"):
            self.show_help = not self.show_help
            return True
        if key == ord("s"):
            self.save()
            return True
        if key in (8, 127, ord("f")):
            self.remove_selected_from_current_frame()
            return True
        if key == ord("u"):
            self.assign_selected(None)
            return True
        if key == ord("r"):
            self.reset_selected()
            return True
        if key == ord("R"):
            self.reset_all()
            return True
        if key == ord("q"):
            self.move_selected_to_team(TEAM_A)
            return True
        if key == ord("e"):
            self.move_selected_to_team(TEAM_B)
            return True
        if ord("1") <= key <= ord("9"):
            self.assign_selected(key - ord("0"))
            return True
        if key == ord("0"):
            self.assign_selected(10)
            return True
        return True

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(0) & 0xFF
            if not self.handle_key(key):
                break
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive annotator for multishot team-clustering samples")
    parser.add_argument("video_path", help="Path to the source video")
    parser.add_argument("--sample-id", default=None, help="Numeric sample id under multishot/images and multishot/labels")
    parser.add_argument("--multishot-root", default=str(DEFAULT_MULTISHOT_ROOT), help="Root of the multishot dataset")
    parser.add_argument("--frame-stride", type=int, default=10, help="Take every Nth frame from the video")
    parser.add_argument("--frames-per-sample", type=int, default=FRAMES_PER_SAMPLE, help="Number of sampled frames per multishot sample")
    parser.add_argument("--padding-seed", type=int, default=0, help="Random seed used to pad the final incomplete sample")
    parser.add_argument("--seg-model", default=str(DEFAULT_SEG_MODEL), help="Segmentation model checkpoint")
    parser.add_argument("--detector-model", default=None, help="Detector checkpoint override")
    parser.add_argument("--no-cache", action="store_true", help="Disable detection and embedding cache reuse")
    parser.add_argument("--resume", action="store_true", help="Resume from the saved manifest for this sample id")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sampled images for this sample id")
    args = parser.parse_args()

    video_path = resolve_repo_path(args.video_path)
    multishot_root = resolve_repo_path(args.multishot_root)
    seg_model = resolve_repo_path(args.seg_model)
    detector_model = resolve_repo_path(args.detector_model) if args.detector_model else None
    sample_id = args.sample_id or next_sample_id(multishot_root)

    images_dir = multishot_root / "images" / sample_id
    labels_dir = multishot_root / "labels" / sample_id
    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        video_path, frame_stride, sampled_frames, source_to_canonical, initial_mapping = load_manifest(multishot_root, sample_id)
        annotator = MultishotAnnotator(
            multishot_root=multishot_root,
            sample_id=sample_id,
            video_path=video_path,
            frame_stride=frame_stride,
            sampled_frames=sampled_frames,
            source_to_canonical=source_to_canonical,
            initial_mapping=initial_mapping,
        )
        annotator.run()
        return
    else:
        detections = bootstrap_players_detections(
            video_path=video_path,
            seg_model=seg_model,
            detector_model=detector_model,
            use_cache=not args.no_cache,
        )
        sampled_frames = extract_sampled_frames(
            video_path=video_path,
            detections=detections,
            frame_stride=args.frame_stride,
        )
        sample_batches = materialize_sample_batches(
            video_path=video_path,
            multishot_root=multishot_root,
            starting_sample_id=sample_id,
            sampled_frames=sampled_frames,
            overwrite=args.overwrite,
            frames_per_sample=args.frames_per_sample,
            padding_seed=args.padding_seed,
        )

        for sample_batch in sample_batches:
            source_to_canonical = build_initial_mapping(sample_batch.sampled_frames)
            initial_mapping = dict(source_to_canonical)
            annotator = MultishotAnnotator(
                multishot_root=multishot_root,
                sample_id=sample_batch.sample_id,
                video_path=video_path,
                frame_stride=args.frame_stride,
                sampled_frames=sample_batch.sampled_frames,
                source_to_canonical=source_to_canonical,
                initial_mapping=initial_mapping,
            )
            annotator.run()


if __name__ == "__main__":
    main()