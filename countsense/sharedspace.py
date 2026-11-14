"""Shared-space map: where pedestrians and vehicles used the same ground during a clip.

No tracking and no identities. Each processed frame only marks which ground cells
were occupied by a pedestrian and which by a vehicle; the map adds those up.
It shows presence, not risk, and it only makes sense for a FIXED camera.
"""
import cv2
import numpy as np

from countsense.counting import CATEGORIES

CELL = 8                     # one ground cell = 8 x 8 pixels
FOOT_FRACTION = 0.2          # the bottom 20% of a box is where the object touches the ground
RIDER_CLASSES = (1, 3)       # bicycle, motorcycle: a person with their feet in one is a rider

PEDESTRIAN_COLOR = (60, 220, 60)    # BGR green  (same colours as the detection boxes)
VEHICLE_COLOR = (255, 160, 0)       # BGR blue
SHARED_COLOR = (60, 60, 255)        # BGR red
FONT = cv2.FONT_HERSHEY_SIMPLEX


class SharedSpace:
    def __init__(self, frame_shape):
        h, w = frame_shape[:2]
        grid = (h // CELL + 1, w // CELL + 1)
        self.ped = np.zeros(grid, dtype=np.float32)   # in how many frames a pedestrian stood on each cell
        self.veh = np.zeros(grid, dtype=np.float32)   # ... and a vehicle
        self.frames = 0
        self.background = None

    def add(self, image, boxes, class_ids):
        """Mark the ground cells that this frame's detections stand on."""
        if self.background is None:
            self.background = image.copy()
        self.frames += 1
        ped_now = np.zeros(self.ped.shape, dtype=bool)
        veh_now = np.zeros(self.veh.shape, dtype=bool)
        rides = boxes[np.isin(class_ids, RIDER_CLASSES)]      # bicycle and motorcycle boxes

        for (x1, y1, x2, y2), cid in zip(boxes, class_ids):
            foot_top = y2 - FOOT_FRACTION * (y2 - y1)
            rows = slice(int(foot_top) // CELL, int(y2) // CELL + 1)
            cols = slice(int(x1) // CELL, int(x2) // CELL + 1)
            if cid in CATEGORIES["vehicle"]:
                veh_now[rows, cols] = True
            elif cid in CATEGORIES["pedestrian"]:
                # Someone whose feet are inside a bicycle / motorcycle box is riding it.
                # Riders are left out, or every two-wheeler's path would look "shared".
                feet_x = (x1 + x2) / 2
                is_rider = np.any((rides[:, 0] <= feet_x) & (feet_x <= rides[:, 2])
                                  & (rides[:, 1] <= y2) & (y2 <= rides[:, 3] + 0.1 * (rides[:, 3] - rides[:, 1])))
                if not is_rider:
                    ped_now[rows, cols] = True

        self.ped += ped_now       # each frame counts once per cell, however many boxes overlap there
        self.veh += veh_now

    def stats(self):
        """How much of the picture each group used, and how much of the pedestrians' ground vehicles also used."""
        ped_cells, veh_cells = self.ped > 0, self.veh > 0
        shared = ped_cells & veh_cells
        return {
            "frames": self.frames,
            "pedestrian_area_pct": round(100 * float(ped_cells.mean()), 1),
            "vehicle_area_pct": round(100 * float(veh_cells.mean()), 1),
            "shared_area_pct": round(100 * float(shared.mean()), 1),
            "shared_pct_of_pedestrian_space": round(100 * float(shared.sum()) / max(int(ped_cells.sum()), 1), 1),
        }

    def render(self):
        """The scene in grey, pedestrians' ground in green, vehicles' in blue, shared ground in red."""
        h, w = self.background.shape[:2]
        grey = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.8

        def to_image(grid):
            """Grid of cells -> picture-sized layer with soft edges."""
            big = cv2.resize(grid, (grid.shape[1] * CELL, grid.shape[0] * CELL), interpolation=cv2.INTER_NEAREST)
            return cv2.GaussianBlur(big[:h, :w], (0, 0), CELL * 0.6)

        def tint(image, alpha, color):
            """Blend a colour over the image; alpha (0..1 per pixel) is how strongly."""
            alpha = alpha[:, :, None]
            return image * (1 - alpha) + np.array(color, dtype=np.float32) * alpha

        # See-through colour wherever the ground was used at all, stronger where it was used in more frames.
        for grid, color in ((self.ped, PEDESTRIAN_COLOR), (self.veh, VEHICLE_COLOR)):
            used = to_image((grid > 0).astype(np.float32))
            often = to_image(grid / max(grid.max(), 1))
            out = tint(out, used * (0.3 + 0.3 * often), color)
        # Shared ground on top, so it stands out.
        shared = to_image(((self.ped > 0) & (self.veh > 0)).astype(np.float32))
        out = tint(out, 0.8 * shared, SHARED_COLOR)
        return np.vstack([self._legend(w), out.astype(np.uint8)])

    def _legend(self, width):
        s = self.stats()
        items = [("pedestrians", PEDESTRIAN_COLOR), ("vehicles", VEHICLE_COLOR), ("shared space", SHARED_COLOR)]
        note = f"{s['shared_pct_of_pedestrian_space']:.0f}% of pedestrian ground shared | {s['frames']} frames"
        margin = 12
        text_w = sum(cv2.getTextSize(t, FONT, 1.0, 1)[0][0] for t, _ in items) + cv2.getTextSize(note, FONT, 1.0, 1)[0][0]
        font_scale = max(0.35, min(width / 1300, (width - 14 * margin) / text_w))   # always fits the width
        thickness = max(1, round(font_scale * 1.5))
        (_, text_h), baseline = cv2.getTextSize("Ag", FONT, font_scale, thickness)
        bar = np.full((text_h + baseline + 2 * margin, width, 3), (40, 40, 40), dtype=np.uint8)
        x, y = margin, margin + text_h
        for text, color in items:
            cv2.rectangle(bar, (x, y - text_h), (x + text_h, y), color, -1)          # colour swatch
            x += text_h + margin // 2
            cv2.putText(bar, text, (x, y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] + 2 * margin
        note_w = cv2.getTextSize(note, FONT, font_scale, thickness)[0][0]
        cv2.putText(bar, note, (width - margin - note_w, y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return bar
