"""Turn a BGR image into the model's input: letterbox to S x S."""
import cv2
import numpy as np

PAD_VALUE = 114  # grey padding, the same value Ultralytics uses


def letterbox(img, size):
    """Resize keeping the aspect ratio (long side = size), then pad to size x size.

    Returns:
        blob  - float32 array (1, 3, size, size), RGB, values 0..1
        scale - resize factor: model pixels = original pixels * scale
        pad   - (pad_x, pad_y): grey pixels added on the left and on the top
    We keep scale and pad so boxes can be mapped back to the original image.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = canvas[:, :, ::-1]                                  # BGR -> RGB
    chw = rgb.transpose(2, 0, 1)                              # HWC -> CHW
    blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    return blob[None], scale, (pad_x, pad_y)                  # add batch dim
