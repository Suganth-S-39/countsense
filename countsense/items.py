"""Optional item notices (--watch-items): say when a knife, baseball bat or scissors is
detected next to a person, so that a human can look at that frame.

This is plain object detection with the same model. It does not know what the person
is doing or intends, it misses small or distant items, and it also fires on harmless
ones (a cook's knife, a cricket bat). So every notice says "possible" and "unverified".
"""
import numpy as np

from countsense.counting import CATEGORIES, CONFIDENT_SCORE

WATCH_NAMES = ("knife", "baseball bat", "scissors")
REACH = 0.5     # "next to" = the item's box, grown by 50% of its longer side, touches a person's box


def watch_class_ids(names):
    """Class ids of the watched items in this model's class list (COCO: 43, 34, 76)."""
    return [names.index(name) for name in WATCH_NAMES if name in names]


def check_items(boxes, scores, class_ids, watch_ids, names):
    """Split one frame's detections into item notices and everything else.

    Returns (notices, boxes, scores, class_ids), where the last three no longer contain
    the watched items, so these never end up in the vehicle / pedestrian counts.
    A notice needs a confident item (score >= 0.5) next to a detected person.
    """
    is_item = np.isin(class_ids, watch_ids)
    people = boxes[np.isin(class_ids, CATEGORIES["pedestrian"])]

    notices = []
    for (x1, y1, x2, y2), score, cid in zip(boxes[is_item], scores[is_item], class_ids[is_item]):
        if score < CONFIDENT_SCORE:
            continue
        grow = REACH * max(x2 - x1, y2 - y1)
        touches = ((people[:, 0] <= x2 + grow) & (people[:, 2] >= x1 - grow)
                   & (people[:, 1] <= y2 + grow) & (people[:, 3] >= y1 - grow))
        if touches.any():
            notices.append({"item": names[cid], "conf": round(float(score), 4),
                            "box_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)]})
    return notices, boxes[~is_item], scores[~is_item], class_ids[~is_item]
