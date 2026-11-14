"""Turn the raw model output (1, 4 + num_classes, N) into final boxes, in pure NumPy."""
import numpy as np

# NMS overlap threshold. Ultralytics' default is 0.7, but at 0.7 one person often kept two or
# three boxes (a tight one inside a loose one), which inflates a count. 0.5 removes those.
IOU_THRES = 0.5


def decode(output, scale, pad, orig_shape, conf_thres=0.25, class_ids=None, iou_thres=IOU_THRES):
    """Raw model output -> final detections in original-image pixels.

    Returns:
        boxes     - (K, 4) float32, x1 y1 x2 y2 in original-image pixels
        scores    - (K,) confidence of the best class
        class_ids - (K,) COCO class id
    Sorted by score, highest first.
    """
    preds = output[0].T                          # (N, 4 + num_classes): one row per candidate
    class_scores = preds[:, 4:]
    cls = class_scores.argmax(axis=1)            # best class for each candidate
    scores = class_scores.max(axis=1)            # ...and its score

    # 1. Cheap filters first (confidence, then the preset's classes), so NMS
    #    only sees a few dozen boxes instead of all N candidates.
    keep = scores > conf_thres
    if class_ids is not None:
        keep &= np.isin(cls, class_ids)
    preds, scores, cls = preds[keep], scores[keep], cls[keep]

    # 2. Centre/size -> corners.
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 3. Undo the letterbox: remove the padding, then undo the resize.
    pad_x, pad_y = pad
    boxes -= [pad_x, pad_y, pad_x, pad_y]
    boxes /= scale

    # 4. Clip to the original image.
    img_h, img_w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)

    # 5. Remove duplicate boxes of the same object.
    keep = class_aware_nms(boxes, scores, cls, iou_thres)
    return boxes[keep], scores[keep], cls[keep]


def class_aware_nms(boxes, scores, class_ids, iou_thres):
    """Run NMS separately per class, so e.g. a person box never removes a bicycle box."""
    keep = []
    for c in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == c)
        keep.extend(idx[nms(boxes[idx], scores[idx], iou_thres)])
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(-scores[keep])]       # highest score first


def nms(boxes, scores, iou_thres):
    """Greedy NMS: keep the best box, drop the boxes that overlap it by more
    than iou_thres, repeat with what is left. Returns indices into boxes."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        best, rest = order[0], order[1:]
        keep.append(best)
        # Intersection-over-union of the best box with every remaining box.
        inter_w = np.clip(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0, None)
        inter_h = np.clip(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0, None)
        inter = inter_w * inter_h
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)
