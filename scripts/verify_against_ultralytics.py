"""Stage 5: check that our pipeline gives the same per-class counts as Ultralytics.

Both sides run the SAME .pte file on the SAME frames with the same settings
(conf 0.25, NMS IoU 0.7), for every exported size (320 / 416 / 640).
Ultralytics is used only in this script - never in run_poc.py or countsense/.

Why the .pte and not yolo11n.pt: Ultralytics feeds a .pt model a smaller,
unpadded input (e.g. 480x640 for bus.jpg), so its scores differ slightly from a
fixed-size export. Using the same .pte means any difference is in OUR
preprocessing or decoding.

usage (from the project folder):
    .venv/Scripts/python.exe scripts/verify_against_ultralytics.py
    .venv/Scripts/python.exe scripts/verify_against_ultralytics.py --source data/clip.mp4 --preset traffic
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                                        # to import run_poc and countsense
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # keep Ultralytics settings on this drive

from countsense.counting import PRESETS, preset_class_ids  # noqa: E402
from countsense.decode import IOU_THRES  # noqa: E402
from countsense.io import FrameSource  # noqa: E402
from run_poc import find_models, prepare_model, process_frame  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Compare our per-class counts with Ultralytics on the same .pte files")
    p.add_argument("--source", type=Path, default=ROOT / "data", help="image folder, one image, or a video")
    p.add_argument("--model", type=Path, default=ROOT / "yolo11n_640",
                   help="any model folder; every model folder next to it is checked")
    p.add_argument("--preset", choices=["all", *PRESETS], default="all",
                   help="which classes to compare (default: all 80)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--max-frames", type=int, default=20, help="how many frames to use from a video")
    return p.parse_args()


def ultralytics_model(pte_path):
    """Ultralytics only recognises an ExecuTorch model by a folder name ending in
    '_executorch_model', so give it a renamed copy of our model folder (fresh each run)."""
    folder = ROOT / ".tmp" / "verify_models" / f"{pte_path.parent.name}_executorch_model"
    shutil.copytree(pte_path.parent, folder, dirs_exist_ok=True)
    return YOLO(str(folder), task="detect")


def describe(counts, names):
    """Counter of class ids -> 'bus 1, person 4'."""
    return ", ".join(f"{names[c]} {n}" for c, n in sorted(counts.items())) or "(nothing)"


def main():
    args = parse_args()
    frames = list(FrameSource(args.source, max_frames=args.max_frames))
    class_ids = None if args.preset == "all" else preset_class_ids(args.preset)
    models = find_models(args.model)
    if not models:
        raise SystemExit(f"No .pte models found next to {args.model}")

    print(f"Ours vs Ultralytics on the same .pte: {len(frames)} frame(s) from {args.source}, "
          f"classes: {args.preset}, conf {args.conf}, IoU {IOU_THRES}\n")
    name_w = max(len(f.name) for f in frames) + 2              # column width for the frame names
    checked = matched = 0
    for pte in models:
        pte_path, method, imgsz, names = prepare_model(pte, None, threads=4)
        reference = ultralytics_model(pte_path)
        model_matched = 0
        for frame in frames:
            _, _, cls, _ = process_frame(method, frame.image, imgsz, args.conf, class_ids)
            ours = Counter(int(c) for c in cls)
            # Same BGR frame as we used; Ultralytics does its own letterbox + decode + NMS.
            result = reference.predict(frame.image, imgsz=imgsz, conf=args.conf, iou=IOU_THRES,
                                       classes=class_ids, verbose=False)[0]
            theirs = Counter(int(c) for c in result.boxes.cls.tolist())

            if ours == theirs:
                model_matched += 1
                print(f"{pte_path.parent.name:<12} {frame.name:<{name_w}} OK    {describe(ours, names)}")
            else:
                print(f"{pte_path.parent.name:<12} {frame.name:<{name_w}} DIFF  ours: {describe(ours, names)}\n"
                      f"{'':<{name_w + 19}}ultralytics: {describe(theirs, names)}")
        print(f"{pte_path.parent.name:<12} -> {model_matched}/{len(frames)} frames match\n")
        checked += len(frames)
        matched += model_matched

    verdict = "PASS" if matched == checked else "FAIL"
    print(f"{verdict}: {matched}/{checked} frame-model pairs have identical per-class counts.")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
