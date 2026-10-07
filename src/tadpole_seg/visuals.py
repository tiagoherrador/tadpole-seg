import numpy as np
import cv2

def overlay(img_rgb, mask_bin, alpha=0.4):
    if mask_bin.shape[:2] != img_rgb.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(img_rgb)
    color[:, :, 0] = mask_bin  # red
    return cv2.addWeighted(img_rgb, 1 - alpha, color, alpha, 0)
