
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import numpy as np
import torch
from executorch.extension.pybindings import portable_lib
from executorch.runtime import Runtime


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def find_pte(model_path):
    """--model can be a folder (e.g. yolo11n_320/) or a .pte file. Return the .pte path."""
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path
    ptes = sorted(model_path.glob("*.pte"))
    if len(ptes) != 1:
        raise FileNotFoundError(f"Expected exactly one .pte in {model_path}, found {len(ptes)}")
    return ptes[0]


def read_metadata(model_dir):
    """Read the input size and class names from Ultralytics' metadata.yaml.

    We only need two fields from a simple file, so we read it line by line
    instead of adding a YAML library to the Pi's dependencies.
    Returns (imgsz or None, list of class names).
    """
    path = Path(model_dir) / "metadata.yaml"
    if not path.is_file():
        return None, list(COCO80_NAMES)

    imgsz, names, section = [], {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith("-"):
            section = line.split(":", 1)[0]          # a top-level key, e.g. "imgsz"
        elif section == "imgsz" and line.startswith("- "):
            imgsz.append(int(line[2:]))              # "- 640"
        elif section == "names" and ":" in line:
            idx, name = line.split(":", 1)           # "  0: person"
            names[int(idx)] = name.strip().strip("'\"")

    names_list = [names[i] for i in range(len(names))] if names else list(COCO80_NAMES)
    return (imgsz[0] if imgsz else None), names_list


def model_description(model_dir):
    """The model's own description from metadata.yaml, e.g.
    'Ultralytics YOLO11n model trained on coco.yaml' (folder names in paths are dropped)."""
    path = Path(model_dir) / "metadata.yaml"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("description:"):
                words = line.split(":", 1)[1].split()
                return " ".join(word.rsplit("/", 1)[-1] for word in words)
    return "unknown"


def load_model(pte_path, threads):
    """Load the .pte with the ExecuTorch runtime and return its "forward" method."""
    # XNNPACK runs on ExecuTorch's shared thread pool, so we size it before loading.
    # (The leading underscore means ExecuTorch marks it experimental; it is the
    # only thread control in executorch 1.5.1.)
    portable_lib._unsafe_reset_threadpool(threads)
    torch.set_num_threads(threads)
    program = Runtime.get().load_program(Path(pte_path))
    return program.load_method("forward")


def model_input_size(method):
    """The .pte was exported for one fixed input shape (1, 3, S, S). Return S."""
    return method.metadata.input_tensor_meta(0).sizes()[-1]


def read_image(path):
    """Read an image as BGR uint8. Fails loudly instead of returning None."""
    # np.fromfile + imdecode also works for non-ASCII paths on Windows (cv2.imread does not).
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path, img):
    """Save an image, creating the folder if needed (imencode + tofile, same reason as above)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


class VideoOut:
    """Write annotated frames to an .mp4 (--save-video). Plays in VLC / the Windows player."""

    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.writer = None
        self.size = None

    def write(self, img):
        if self.writer is None:
            h, w = img.shape[:2]
            self.size = (w - w % 2, h - h % 2)    # video encoders want even width and height
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                raise ValueError(f"Could not create video: {self.path}")
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)      # one video = one frame size (image folders can mix sizes)
        self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


class Frame(NamedTuple):
    index: int                # TRUE position in the source (video frame number or image number)
    time_s: Optional[float]   # seconds into the video; None for images
    name: str                 # used for the annotated file name
    image: np.ndarray         # BGR uint8


class FrameSource:
    """The frames to process, from a video file, a folder of images, or one image.

    --stride N keeps every Nth frame, --max-frames N stops after N processed frames.
    Each Frame keeps its true index and timestamp, so skipping never shifts the record.
    """

    def __init__(self, source, stride=1, max_frames=None):
        self.source = Path(source)
        self.stride = stride
        self.max_frames = max_frames
        self.fps = None
        self.frames_read = 0      # frames looked at so far: processed + skipped by stride

        # Check the source now, so bad input fails before the (slow) model load.
        if not self.source.exists():
            raise FileNotFoundError(f"Source not found: {self.source}")
        if self.source.is_dir():
            self.kind = "folder"
            self.paths = sorted(p for p in self.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.paths:
                raise FileNotFoundError(f"No images ({', '.join(sorted(IMAGE_EXTS))}) in {self.source}")
        elif self.source.suffix.lower() in VIDEO_EXTS:
            self.kind = "video"
            cap = cv2.VideoCapture(str(self.source))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.source}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            cap.release()
        else:
            self.kind = "image"
            self.paths = [self.source]

    def __iter__(self):
        return self._video() if self.kind == "video" else self._images()

    def _video(self):
        cap = cv2.VideoCapture(str(self.source))
        index = processed = 0
        try:
            while self.max_frames is None or processed < self.max_frames:
                if index % self.stride:           # skipped by stride: grab() moves on without decoding to BGR
                    if not cap.grab():
                        break
                else:
                    ok, image = cap.read()
                    if not ok:
                        break
                    processed += 1
                    time_s = index / self.fps if self.fps else None
                    yield Frame(index, time_s, f"frame_{index:06d}", image)
                index += 1
                self.frames_read = index
        finally:
            cap.release()

    def _images(self):
        processed = 0
        for index, path in enumerate(self.paths):
            if self.max_frames is not None and processed >= self.max_frames:
                break
            self.frames_read = index + 1
            if index % self.stride:
                continue
            processed += 1
            yield Frame(index, None, path.stem, read_image(path))
