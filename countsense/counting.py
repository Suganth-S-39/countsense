"""Categories, presets and per-frame counting (COCO class ids). Counts are per frame only."""
import numpy as np

from countsense.decode import IOU_THRES, class_aware_nms

CATEGORIES = {
    "pedestrian": [0],           # person
    "cyclist": [1],              # bicycle
    "vehicle": [2, 3, 5, 7],     # car, motorcycle, bus, truck
}

PRESETS = {
    "traffic": ["vehicle", "cyclist", "pedestrian"],
    "pedestrian": ["pedestrian"],
    "public-space": ["pedestrian", "cyclist"],
}

# Reverse lookup: class id -> category name, e.g. 5 -> "vehicle".
CLASS_TO_CATEGORY = {cid: cat for cat, ids in CATEGORIES.items() for cid in ids}

DEFAULT_DENSITY = (8, 16)        # LOW < 8, MODERATE 8-15, HIGH >= 16
CONFIDENT_SCORE = 0.5            # a detection at or above this score is "confident"; below it, only "possible"


def preset_class_ids(preset):
    """All COCO class ids a preset cares about, e.g. traffic -> [2, 3, 5, 7, 1, 0]."""
    return [cid for cat in PRESETS[preset] for cid in CATEGORIES[cat]]


def one_box_per_object(boxes, scores, class_ids):
    """Drop the second box of an object that was detected as two classes.

    A van is often detected as "car" AND "truck", a coach as "bus" AND "truck". The NMS in
    decode.py works per class, so both boxes survive and the vehicle would count twice.
    Here NMS runs once more per CATEGORY: of two boxes of the same category that overlap
    by more than IOU_THRES, only the better-scored one (and its class name) is kept.
    A rider and their motorcycle are different categories, so both stay."""
    category_index = np.array([list(CATEGORIES).index(CLASS_TO_CATEGORY[int(c)]) for c in class_ids], dtype=int)
    keep = class_aware_nms(boxes, scores, category_index, IOU_THRES)
    return boxes[keep], scores[keep], class_ids[keep]


def count_frame(class_ids, categories):
    """Count one frame's detections per category, e.g. {"vehicle": 3, "cyclist": 0, "pedestrian": 5}."""
    counts = {cat: 0 for cat in categories}
    for cid in class_ids:
        counts[CLASS_TO_CATEGORY[int(cid)]] += 1
    return counts


def density_level(total, thresholds=DEFAULT_DENSITY):
    """LOW if total < low, HIGH if total >= high, MODERATE in between."""
    low, high = thresholds
    if total < low:
        return "LOW"
    if total >= high:
        return "HIGH"
    return "MODERATE"


def range_text(confident, possible):
    """A count as 'confident-possible', e.g. '6-8'; just '7' when every detection is confident."""
    return str(possible) if confident == possible else f"{confident}-{possible}"


def count_stats(records, categories):
    """Average and maximum count per frame, for each category and for the total.
    (Per frame only: the same car in 30 frames is 30 sightings, not 30 cars.)
    "avg_confident" counts only detections scoring CONFIDENT_SCORE or more."""
    stats = {}
    for key in [*categories, "total"]:
        if key == "total":
            values = [r["total"] for r in records]
            confident = [r["total_confident"] for r in records]
        else:
            values = [r["counts"][key] for r in records]
            confident = [r["counts_confident"][key] for r in records]
        stats[key] = {"avg": round(sum(values) / len(values), 2), "max": max(values),
                      "avg_confident": round(sum(confident) / len(confident), 2)}
    return stats


def density_distribution(records):
    """How many processed frames were LOW / MODERATE / HIGH."""
    dist = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    for r in records:
        dist[r["density"]] += 1
    return dist
