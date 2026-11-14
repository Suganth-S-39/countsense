"""Passing counts (--count-line): count each object once when it crosses a line, with its direction.

A small tracker links the boxes of one processed frame to the boxes of the next, so a
car seen in 30 frames is one car. It uses box positions only: no appearance model and
no identities. It only makes sense for a FIXED camera.
"""
import re

import numpy as np

from countsense.counting import CLASS_TO_CATEGORY, CONFIDENT_SCORE

MATCH_REACH = 1.0      # a box continues a track if its centre is within one box size of where the track is expected
MAX_GAP_S = 0.5        # a track that is not detected for longer than this is forgotten
MIN_TRAVEL = 0.5       # a crossing counts once the object has moved this many box sizes in that direction
SAME_OBJECT = 0.3      # two tracks closer than this many box sizes are the same object
DIRECTIONS = {"h": ("down", "up"), "v": ("right", "left")}    # (towards larger y or x, towards smaller)


def parse_line(text):
    """'h50' -> ('h', 0.5): a horizontal line at 50% of the height. 'v30' -> ('v', 0.3):
    a vertical line at 30% of the width. Returns None if the text is not in that form."""
    if not re.fullmatch(r"[hv][1-9]\d?", text):
        return None
    return text[0], int(text[1:]) / 100


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


class LineCounter:
    def __init__(self, axis, fraction, frame_shape, categories, processed_fps):
        h, w = frame_shape[:2]
        self.axis = axis                                           # "h" (horizontal line) or "v" (vertical)
        self.position = fraction * (h if axis == "h" else w)       # the line's y (h) or x (v), in pixels
        p = int(self.position)
        self.endpoints = ((0, p), (w - 1, p)) if axis == "h" else ((p, 0), (p, h - 1))
        self.forward, self.backward = DIRECTIONS[axis]
        # MAX_GAP_S as a number of processed frames (it depends on the video's fps and on --stride).
        self.max_misses = max(1, round(MAX_GAP_S * processed_fps)) if processed_fps else 5
        self.tracks = []
        self.next_id = 1
        self.passed = {cat: {self.forward: 0, self.backward: 0} for cat in categories}

    def across(self, box):
        """Where the box's centre is, measured across the line: its y for a horizontal line, its x for a vertical one."""
        return centre(box)[1 if self.axis == "h" else 0]

    def side(self, box):
        """+1 if the box's centre is past the line (below a horizontal line, right of a vertical one), else -1."""
        return 1 if self.across(box) >= self.position else -1

    def update(self, boxes, scores, class_ids):
        """Link this frame's boxes to the tracks and count line crossings.
        Returns (the track number of every box, the crossings counted in this frame)."""
        boxes = boxes.astype(float)
        categories = [CLASS_TO_CATEGORY[int(c)] for c in class_ids]
        ids = np.zeros(len(boxes), dtype=int)                      # 0 = not linked to a track yet
        crossings = []

        # 1. Match boxes to tracks. A track is expected where its last box would be by now, moving
        #    at its last speed. The distance from there to each box's centre is measured in box
        #    sizes (so a bus may move further than a person). Nearest pair first; each used once.
        matched = set()
        if self.tracks and len(boxes):
            expected = np.array([centre(t["box"]) + t["speed"] * (t["misses"] + 1) for t in self.tracks])
            size = np.array([(t["box"][2] - t["box"][0] + t["box"][3] - t["box"][1]) / 2 for t in self.tracks])
            centres = (boxes[:, :2] + boxes[:, 2:]) / 2
            distance = np.linalg.norm(expected[:, None] - centres[None, :], axis=2) / (size[:, None] + 1e-9)
            for ti, track in enumerate(self.tracks):
                distance[ti, [c != track["category"] for c in categories]] = np.inf
            while distance.min() <= MATCH_REACH:
                ti, bi = np.unravel_index(distance.argmin(), distance.shape)
                track, box = self.tracks[ti], boxes[bi]
                step = (centre(box) - centre(track["box"])) / (track["misses"] + 1)
                track["speed"] = 0.5 * track["speed"] + 0.5 * step      # averaged, to calm box jitter
                track["box"], track["misses"] = box, 0
                track["best_score"] = max(track["best_score"], float(scores[bi]))
                ids[bi] = track["id"]
                matched.add(ti)
                # 2. Crossing: the centre is now on the other side of the line. The latest crossing
                #    is the one remembered, so a wobble on the line is corrected by the real crossing.
                side = self.side(box)
                if side != track["side"] and not track["counted"]:
                    track["crossed"] = self.forward if side > 0 else self.backward
                track["side"] = side
                #    It is counted once, as soon as two things hold (now or in a later frame):
                #    (a) the model was sure about the object at least once, which keeps flickering
                #        low-score boxes out;
                #    (b) the object has really moved in that direction since it was first seen, by
                #        MIN_TRAVEL box sizes, which keeps out a stopped vehicle whose box wobbles
                #        over the line and a box that only changes shape at the picture's edge.
                if track["crossed"] and not track["counted"] and track["best_score"] >= CONFIDENT_SCORE:
                    moved = (self.across(box) - track["start"]) * (1 if track["crossed"] == self.forward else -1)
                    if moved >= MIN_TRAVEL * size[ti]:
                        track["counted"] = True
                        self.passed[track["category"]][track["crossed"]] += 1
                        crossings.append({"track_id": track["id"], "category": track["category"],
                                          "direction": track["crossed"]})
                distance[ti, :] = np.inf
                distance[:, bi] = np.inf

        # 3. Tracks without a box this frame: remember the miss, forget them after too many.
        #    A track expected right on top of a track that did get a box is a second track of the
        #    same object (it had two boxes for a moment). It is dropped at once and hands over its
        #    "counted" mark, or that object could be counted twice.
        kept = []
        for ti, track in enumerate(self.tracks):
            if ti not in matched:
                track["misses"] += 1
                twin = next((self.tracks[mi] for mi in matched if self.tracks[mi]["category"] == track["category"]
                             and np.linalg.norm(centre(self.tracks[mi]["box"]) - expected[ti]) <= SAME_OBJECT * size[ti]),
                            None)
                if twin:
                    twin["counted"] = twin["counted"] or track["counted"]
                if twin or track["misses"] > self.max_misses:
                    continue
            kept.append(track)
        self.tracks = kept

        # 4. Boxes without a track start a new one.
        for bi in range(len(boxes)):
            if ids[bi] == 0:
                self.tracks.append({"id": self.next_id, "category": categories[bi], "box": boxes[bi],
                                    "speed": np.zeros(2), "misses": 0, "side": self.side(boxes[bi]),
                                    "start": self.across(boxes[bi]), "best_score": float(scores[bi]),
                                    "crossed": None, "counted": False})
                ids[bi] = self.next_id
                self.next_id += 1
        return ids, crossings

    def total(self):
        return sum(n for directions in self.passed.values() for n in directions.values())

    def tally_text(self):
        """e.g. 'passed: 7 (down 4, up 3)', for the label next to the line."""
        fwd = sum(d[self.forward] for d in self.passed.values())
        back = sum(d[self.backward] for d in self.passed.values())
        return f"passed: {fwd + back} ({self.forward} {fwd}, {self.backward} {back})"

    def stats(self, line_text, duration_s):
        """Totals for summary.json and REPORT.md. duration_s = length of the video that was read."""
        return {
            "line": line_text,
            "orientation": "horizontal" if self.axis == "h" else "vertical",
            "position_px": int(self.position),
            "directions": [self.forward, self.backward],
            "passed": self.passed,
            "total": self.total(),
            "duration_s": round(duration_s, 2) if duration_s else None,
            "per_minute": round(60 * self.total() / duration_s, 1) if duration_s else None,
        }
