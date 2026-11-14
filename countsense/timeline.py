"""Timeline (--timeline): a chart of the counts over the run, plus the busiest moment
and the busy periods. Drawn with OpenCV, so the Pi needs no plotting library."""
import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIDTH, HEIGHT = 1000, 440
LEFT, RIGHT, TOP, BOTTOM = 56, 140, 84, 52         # margins around the plot area (right: the level labels)
BUSY_GAP_S = 1.0       # a dip shorter than this does not end a busy period
# Colours are BGR. Text is always dark grey; only the lines carry a series colour.
SURFACE, INK, MUTED, GRID = (251, 252, 252), (40, 40, 40), (110, 110, 110), (228, 230, 230)
# One fixed colour per category, chosen so the three stay apart for colour-blind readers too.
SERIES_COLORS = {"vehicle": (214, 120, 42), "cyclist": (52, 104, 235), "pedestrian": (122, 175, 27)}
LEVELS = ("LOW", "MODERATE", "HIGH")


def timeline_stats(records):
    """The busiest moment, and the busy periods: stretches of processed frames at the
    highest density level the run reached (none if it stayed LOW throughout)."""
    peak = max(records, key=lambda r: r["total"])
    top = max((r["density"] for r in records), key=LEVELS.index)
    periods, start = [], None
    for i, r in enumerate([*records, None]):           # the None closes a period that runs to the end
        busy = r is not None and r["density"] == top and top != "LOW"
        if busy and start is None:
            start = i
        elif not busy and start is not None:
            first, last = records[start], records[i - 1]
            period = {"level": top, "from": first["name"], "to": last["name"],
                      "from_s": first["time_s"], "to_s": last["time_s"], "frames": i - start,
                      "start": start, "max_total": max(x["total"] for x in records[start:i])}
            start = None
            # In a video, a short dip does not end a period: one flickering detection
            # near the threshold would otherwise cut it into many one-frame pieces.
            before = periods[-1] if periods else None
            if before and period["from_s"] is not None and period["from_s"] - before["to_s"] < BUSY_GAP_S:
                before.update(to=period["to"], to_s=period["to_s"], frames=before["frames"] + period["frames"],
                              max_total=max(before["max_total"], period["max_total"]))
            else:
                periods.append(period)
            # "frames" of the "span" processed frames in the period were at that level (the rest are dips).
            periods[-1]["span"] = i - periods[-1]["start"]
    for period in periods:
        del period["start"]
    return {"peak": {"name": peak["name"], "time_s": peak["time_s"], "total": peak["total"]},
            "top_level": top, "busy_periods": periods}


def nice_step(span, max_ticks, steps):
    """The smallest step from `steps` that covers `span` with at most max_ticks ticks."""
    return next((s for s in steps if span / s <= max_ticks), steps[-1])


def put_text(img, text, x, y, color=INK, scale=0.5, align="left"):
    """Text whose left edge, centre or right edge is at x."""
    width = cv2.getTextSize(text, FONT, scale, 1)[0][0]
    x -= {"left": 0, "center": width // 2, "right": width}[align]
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, 1, cv2.LINE_AA)


def draw_timeline(records, categories, thresholds):
    """Line chart of the per-frame counts (needs at least two processed frames).
    x = seconds into the video, or the image number for a folder; y = objects in that frame."""
    timed = records[0]["time_s"] is not None
    xs = [r["time_s"] if timed else r["frame_index"] + 1 for r in records]
    totals = [r["total"] for r in records]
    x_min, x_max = xs[0], xs[-1]
    y_step = nice_step(max(totals) + 1, 6, (1, 2, 5, 10, 20, 50, 100))
    y_top = y_step * (max(totals) // y_step + 1)       # the first tick above the highest count
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def px(x):
        return round(LEFT + (x - x_min) / (x_max - x_min) * plot_w)

    def py(y):
        return round(TOP + plot_h - y / y_top * plot_h)

    img = np.full((HEIGHT, WIDTH, 3), SURFACE, dtype=np.uint8)

    # Title, and a legend when there is more than one line.
    one_line = len(categories) == 1
    put_text(img, f"{categories[0]} count per frame" if one_line else "Counts per frame", LEFT, 30, scale=0.7)
    put_text(img, f"{len(records)} processed frames", WIDTH - 20, 30, MUTED, 0.45, "right")
    x = LEFT
    for name, color in ([] if one_line else [*((c, SERIES_COLORS[c]) for c in categories), ("total", INK)]):
        cv2.rectangle(img, (x, 51), (x + 18, 54), color, -1)
        put_text(img, name, x + 24, 58, scale=0.45)
        x += 24 + cv2.getTextSize(name, FONT, 0.45, 1)[0][0] + 22

    # Horizontal grid lines with the count on the left.
    for y in range(0, y_top + 1, y_step):
        cv2.line(img, (LEFT, py(y)), (WIDTH - RIGHT, py(y)), GRID, 1)
        put_text(img, str(y), LEFT - 8, py(y) + 5, MUTED, 0.45, "right")
    # Where the density level changes, named in the right margin (a dict, so equal thresholds give one line).
    low, high = thresholds
    for value, level in {low: "MODERATE", high: "HIGH"}.items():
        if 0 < value <= y_top:
            cv2.line(img, (LEFT, py(value)), (WIDTH - RIGHT, py(value)), MUTED, 1)
            put_text(img, f"{level} from {value}", WIDTH - RIGHT + 8, py(value) + 4, MUTED, 0.42)

    # Time (or image number) along the bottom.
    x_step = nice_step(x_max - x_min, 8, (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300) if timed
                       else (1, 2, 5, 10, 20, 50, 100))
    tick = math.ceil(x_min / x_step) * x_step
    while tick <= x_max:
        cv2.line(img, (px(tick), TOP + plot_h), (px(tick), TOP + plot_h + 5), MUTED, 1)
        put_text(img, f"{tick:g}", px(tick), TOP + plot_h + 21, MUTED, 0.45, "center")
        tick += x_step
    put_text(img, "time (s)" if timed else "image number", LEFT + plot_w // 2, HEIGHT - 10, MUTED, 0.45, "center")

    # The lines: the total underneath, then one per category.
    lines = [] if one_line else [(totals, INK)]
    lines += [([r["counts"][cat] for r in records], SERIES_COLORS[cat]) for cat in categories]
    for values, color in lines:
        points = np.array([[px(x), py(y)] for x, y in zip(xs, values)], dtype=np.int32)
        cv2.polylines(img, [points], False, color, 2, cv2.LINE_AA)

    # Mark the busiest moment (a dot with a light ring, so it stays visible on the line).
    peak = int(np.argmax(totals))
    dot = (px(xs[peak]), py(totals[peak]))
    cv2.circle(img, dot, 7, SURFACE, -1, cv2.LINE_AA)
    cv2.circle(img, dot, 5, INK, -1, cv2.LINE_AA)
    where = f"{xs[peak]:.1f} s" if timed else f"image {xs[peak]}"
    right_side = dot[0] < WIDTH - RIGHT - 200           # room for the label on the right of the dot?
    put_text(img, f"busiest: {totals[peak]} at {where}", dot[0] + (12 if right_side else -12), dot[1] - 9,
             scale=0.45, align="left" if right_side else "right")
    return img
