"""Draw detections (box + label + confidence) and the header bar on a copy of the image."""
import cv2
import numpy as np

from countsense.counting import CATEGORIES, CLASS_TO_CATEGORY, CONFIDENT_SCORE, range_text

# --blur-people: the head area is the top of a person's box. It is as tall as 70% of the
# box's width (a head is a little narrower than the shoulders), and at least a quarter of the box.
HEAD_WIDTH_RATIO = 0.7
HEAD_MIN_FRACTION = 0.25
BLUR_BLOCKS = 5                   # the head area is reduced to 5 flat blocks across: no face detail is left

CATEGORY_COLORS = {
    "vehicle": (255, 160, 0),     # blue
    "cyclist": (0, 200, 255),     # orange
    "pedestrian": (60, 220, 60),  # green
}
OTHER_COLOR = (200, 200, 200)     # grey, for classes outside every category
NOTICE_COLOR = (40, 40, 230)      # red, for --watch-items notices
LINE_COLOR = (0, 255, 255)        # yellow, for the --count-line line
DENSITY_COLORS = {"LOW": (60, 200, 60), "MODERATE": (0, 170, 255), "HIGH": (70, 70, 255)}  # green / orange / red
HEADER_BG = (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def blur_people(img, boxes, class_ids):
    """Privacy (--blur-people): return a copy of the image with the head area of every
    detected person pixelated. There is no face detector: the head is taken to be the
    top of the person's box, so a person the model misses is not blurred."""
    out = img.copy()
    for x1, y1, x2, y2 in boxes[np.isin(class_ids, CATEGORIES["pedestrian"])].astype(int):
        w, h = x2 - x1, y2 - y1
        head_h = min(h, int(max(HEAD_MIN_FRACTION * h, HEAD_WIDTH_RATIO * w)))
        if w < 1 or head_h < 1:
            continue
        head = out[y1:y1 + head_h, x1:x2]
        # Shrink to a few blocks, then enlarge again: each block becomes one flat colour.
        blocks = cv2.resize(head, (BLUR_BLOCKS, max(1, round(BLUR_BLOCKS * head_h / w))),
                            interpolation=cv2.INTER_AREA)
        out[y1:y1 + head_h, x1:x2] = cv2.resize(blocks, (w, head_h), interpolation=cv2.INTER_NEAREST)
    return out


def draw_detections(img, boxes, scores, class_ids, names, track_ids=None):
    """Confident detections get a normal box. "Possible" ones (score below
    CONFIDENT_SCORE) get a thin box and a '?' after the score.
    With --count-line, track_ids puts each object's track number in front of its label."""
    out = img.copy()
    # Scale line width and text with the image, so labels stay readable on 4K and on 640px.
    short_side = min(out.shape[:2])
    thickness = max(2, round(short_side / 400))
    font_scale = max(0.5, short_side / 1200)

    for i, ((x1, y1, x2, y2), score, cid) in enumerate(zip(boxes.astype(int), scores, class_ids)):
        confident = score >= CONFIDENT_SCORE
        color = CATEGORY_COLORS.get(CLASS_TO_CATEGORY.get(int(cid)), OTHER_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness if confident else 1)

        label = f"{names[cid]} {score:.2f}" + ("" if confident else "?")
        if track_ids is not None:
            label = f"#{track_ids[i]} {label}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
        # Keep the label inside the image: below the top edge, left of the right edge.
        x_text = max(0, min(x1, out.shape[1] - tw - 2))
        y_text = max(y1, th + baseline + 2)
        cv2.rectangle(out, (x_text, y_text - th - baseline - 2), (x_text + tw + 2, y_text), color, -1)
        cv2.putText(out, label, (x_text + 1, y_text - baseline - 1), FONT, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_notices(img, notices):
    """Mark each item notice (--watch-items) with a red box labelled "possible <item>"."""
    font_scale = max(0.5, min(img.shape[:2]) / 1200)
    for n in notices:
        x1, y1, x2, y2 = (int(v) for v in n["box_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), NOTICE_COLOR, 3)
        label = f"possible {n['item']} {n['conf']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
        x_text = max(0, min(x1, img.shape[1] - tw - 2))
        y_text = max(y1, th + baseline + 2)
        cv2.rectangle(img, (x_text, y_text - th - baseline - 2), (x_text + tw + 2, y_text), NOTICE_COLOR, -1)
        cv2.putText(img, label, (x_text + 1, y_text - baseline - 1), FONT, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_count_line(img, endpoints, text):
    """Draw the --count-line line across the image, with the running tally next to its start."""
    (x1, y1), (x2, y2) = endpoints
    cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR, max(2, round(min(img.shape[:2]) / 300)))
    font_scale = max(0.5, min(img.shape[:2]) / 1000)
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    # Just above a horizontal line's left end, or right of a vertical line's top end; always inside the image.
    x_text = max(0, min(x1 + 6, img.shape[1] - tw - 4))
    y_text = max(y1 - 6, th + baseline + 4)
    cv2.rectangle(img, (x_text, y_text - th - baseline - 2), (x_text + tw + 4, y_text), LINE_COLOR, -1)
    cv2.putText(img, text, (x_text + 2, y_text - baseline - 1), FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def add_header(img, counts, counts_confident, density, density_confident, infer_ms):
    """Return the image with a bar ABOVE it (so no detection is covered) showing
    per-category counts, the total, the density level and the inference time.
    Counts are shown as confident-possible, e.g. "vehicle: 6-8"."""
    width = img.shape[1]
    parts = [f"{cat}: {range_text(counts_confident[cat], n)}" for cat, n in counts.items()]
    parts.append(f"total: {range_text(sum(counts_confident.values()), sum(counts.values()))}")
    counts_text = "   ".join(parts)
    # Borderline frame: the level depends on the low-confidence detections, so show both.
    badge = density if density == density_confident else f"{density_confident}-{density}"
    time_text = f"inference {infer_ms:.0f} ms"
    margin = 12

    # Font size grows with the image, but the whole line must always fit.
    text_widths = sum(cv2.getTextSize(s, FONT, 1.0, 1)[0][0] for s in (counts_text, badge, time_text))
    font_scale = max(0.35, min(width / 1300, (width - 6 * margin) / text_widths))
    thickness = max(1, round(font_scale * 1.5))
    (_, text_h), baseline = cv2.getTextSize("Ag", FONT, font_scale, thickness)
    bar_h = text_h + baseline + 2 * margin
    bar = np.full((bar_h, width, 3), HEADER_BG, dtype=np.uint8)
    y = margin + text_h

    # Left: counts and total.
    cv2.putText(bar, counts_text, (margin, y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Middle: density level on a coloured badge.
    x = margin + cv2.getTextSize(counts_text, FONT, font_scale, thickness)[0][0] + 2 * margin
    badge_w = cv2.getTextSize(badge, FONT, font_scale, thickness)[0][0]
    cv2.rectangle(bar, (x - margin // 2, margin // 2), (x + badge_w + margin // 2, bar_h - margin // 2),
                  DENSITY_COLORS[density], -1)
    cv2.putText(bar, badge, (x, y), FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    # Right: inference time.
    time_w = cv2.getTextSize(time_text, FONT, font_scale, thickness)[0][0]
    cv2.putText(bar, time_text, (width - margin - time_w, y), FONT, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

    return np.vstack([bar, img])
