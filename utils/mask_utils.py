# utils/mask_utils.py

import cv2
import numpy as np
import os

def apply_mask_and_save(img, mask_img, save_dir, original_filename="image"):
    if mask_img is None or img is None:
        return img, None

    if mask_img.ndim == 3 and mask_img.shape[2] == 4:
        alpha = mask_img[:, :, 3]
        mask = alpha
    elif mask_img.ndim == 3:
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_img

    mask = mask.astype(np.uint8)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # 黒塗りではなく、周囲の色で埋める
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        ys, xs = np.where(mask_bool)
        sx = max(xs.min() - 1, 0)
        sy = max(ys.min() - 1, 0)
        fill_color = img[sy, sx].tolist()
        img[mask_bool] = fill_color

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.splitext(original_filename)[0] + "_masked.jpg"
    saved_path = os.path.join(save_dir, filename)
    cv2.imwrite(saved_path, img)

    return img, saved_path

