"""Colour constancy and colour ablations.

Central to Phase 3: the gap between dermoscopy and smartphone photography
is, in large part, an illumination and colour-calibration gap. These
functions are the instruments that test that claim.
"""
import cv2
import numpy as np


def shades_of_gray(img: np.ndarray, power: int = 6, gamma: float | None = None) -> np.ndarray:
    """Finlayson & Trezzi (2004) shades-of-gray colour constancy.

    Standard preprocessing in the dermoscopy literature; power=6 is the
    usual choice. Input and output are uint8 RGB.
    """
    out = img.astype(np.float32)
    if gamma is not None:
        out = 255.0 * np.power(out / 255.0, gamma)
    flat = np.power(out, power)
    illum = np.power(np.mean(flat.reshape(-1, 3), axis=0), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) / np.sqrt(3) + 1e-8)
    out = out / (illum[None, None, :] + 1e-8)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_grayscale_3ch(img: np.ndarray) -> np.ndarray:
    """Ablation: strip all colour, keep the tensor shape a 3-channel
    ImageNet-pretrained backbone expects."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


COLOR_MODES = ("raw", "shades_of_gray", "grayscale")


def apply_color_mode(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return img
    if mode == "shades_of_gray":
        return shades_of_gray(img)
    if mode == "grayscale":
        return to_grayscale_3ch(img)
    raise ValueError(f"unknown color_mode {mode!r}; options: {COLOR_MODES}")
