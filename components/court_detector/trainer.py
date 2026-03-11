from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.data.utils import img2label_paths, exif_size
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import colorstr
from PIL import Image
import numpy as np
from tqdm import tqdm


class CourtDetectionDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix

    def get_labels(self) -> list[dict]:
        """
        Custom get_labels that reads the pose-format label files
        (one bbox + keypoints per line, last line = dataset tag)
        and balances datasets so each has roughly equal representation.
        """
        print("Starting loading images labels...")
        self.label_files = img2label_paths(self.im_files)
        assert len(self.label_files) == len(self.im_files)

        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        assert nkpt > 0 and ndim in (2, 3), "kpt_shape must be set in data.yaml"
        expected_cols = 5 + nkpt * ndim

        labels = []
        datasets_to_labels = {}

        for im_file, lb_file in tqdm(
            zip(self.im_files, self.label_files), total=len(self.label_files)
        ):
            im = Image.open(im_file)
            im.verify()
            shape = exif_size(im)

            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                assert len(lb[-1]) == 1, f"Last line must be dataset tag, got: {lb[-1]}"
                dataset_id = lb[-1][0]
                lb = lb[:-1]
                if not lb:
                    continue
                lb = np.array(lb, dtype=np.float32)

            assert lb.shape[1] == expected_cols, (
                f"Label {lb_file}: expected {expected_cols} cols, got {lb.shape[1]}"
            )

            kpts = lb[:, 5:].reshape(lb.shape[0], nkpt, ndim)

            if dataset_id not in datasets_to_labels:
                datasets_to_labels[dataset_id] = []

            labels.append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:5],
                    "segments": [],
                    "keypoints": kpts,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

            datasets_to_labels[dataset_id].append(len(labels) - 1)

        # Balance datasets: upsample smaller datasets to match the largest
        max_size = max((len(v) for v in datasets_to_labels.values()), default=0)
        self.images_sequence = []
        for dataset_id in datasets_to_labels:
            cur_labels = datasets_to_labels[dataset_id]
            num_added = 0
            while num_added < max_size:
                np.random.shuffle(cur_labels)
                for label_id in cur_labels:
                    assert 0 <= label_id < len(labels)
                    self.images_sequence.append(label_id)
                    num_added += 1
                    if num_added >= max_size:
                        break
        np.random.shuffle(self.images_sequence)
        print("Labels loaded!")
        return labels

    def __len__(self):
        if self.train_mode:
            return len(self.images_sequence)
        return super().__len__()

    def __getitem__(self, index):
        if self.train_mode:
            index = self.images_sequence[index]
            img_and_label = self.get_image_and_label(index)
            return self.transforms(img_and_label)
        return super().__getitem__(index)


def build_court_detection_dataset(
    cfg,
    img_path,
    batch,
    data,
    mode,
    rect,
    stride,
):
    return CourtDetectionDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


class CourtDetectionTrainer(PoseTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_court_detection_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
        )
