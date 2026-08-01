"""Classic hand-crafted features -- kept as an honest baseline.

This is the v1 pipeline with its bugs fixed. It is retained rather than deleted
because "the CNN beats the classical approach" is only a meaningful claim if the
classical approach was implemented correctly.

Bugs repaired from the original:

1. ``area_ratio`` was ``countNonZero`` on the k-means *reconstruction*. Both
   cluster colours are non-zero, so it returned exactly 1.0 for every image --
   a constant feature carrying zero information. It now measures the true
   lesion fraction from the cluster labels.

2. GLCM ran on the segmented image, which has only two grey levels. A 256-level
   co-occurrence matrix over two levels is 99.99% empty and the texture
   statistics were meaningless. GLCM now runs on the *original* greyscale,
   quantised to 32 levels and averaged over four angles.

3. Colour statistics were taken from the same two-colour reconstruction,
   discarding the actual colour distribution. They are now computed on the
   original pixels, separately for the whole leaf and the lesion region.

4. k=2 was applied to the full frame. On PlantVillage the dominant split is
   leaf-versus-studio-backdrop, not lesion-versus-healthy. The background is now
   masked out first, so clustering happens *within* the leaf.

5. Cluster identity was arbitrary per image. The lesion cluster is now chosen
   deterministically as the one with lower greenness.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

IMG_SIZE = (256, 256)
GLCM_LEVELS = 32
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _leaf_mask(rgb: np.ndarray, saturation_threshold: int = 40) -> np.ndarray:
    """Foreground leaf via HSV saturation, cleaned up and reduced to one blob."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] > saturation_threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.ones(rgb.shape[:2], dtype=bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = labels == largest

    # Guard against a degenerate mask swallowing the image.
    return leaf if leaf.sum() >= 0.02 * leaf.size else np.ones(rgb.shape[:2], dtype=bool)


def _clean_lesion_mask(lesion: np.ndarray, leaf_area: int, min_frac: float = 0.001) -> np.ndarray:
    """Remove speckle so only spatially coherent lesions survive.

    Morphological opening kills isolated pixels; the area filter then discards
    blobs smaller than ``min_frac`` of the leaf. Without this, a healthy leaf's
    sensor noise reports a large fake lesion fraction.
    """
    mask = lesion.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(lesion)

    min_area = max(int(min_frac * leaf_area), 12)
    keep = np.zeros(n_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _greenness(pixels: np.ndarray) -> np.ndarray:
    """G - (R+B)/2. Higher means healthier tissue; lesions score lower."""
    return pixels[:, 1] - 0.5 * (pixels[:, 0] + pixels[:, 2])


def _stats(values: np.ndarray) -> list[float]:
    """Mean, std, and the 25th/75th percentiles of a 1-D array."""
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def feature_names() -> list[str]:
    names: list[str] = []
    for region in ("leaf", "lesion"):
        for space, channels in (("rgb", "RGB"), ("hsv", "HSV")):
            for ch in channels:
                for stat in ("mean", "std", "p25", "p75"):
                    names.append(f"{region}_{space}_{ch}_{stat}")
    names += ["lesion_ratio", "lesion_blobs", "largest_blob_ratio", "lesion_compactness"]
    names += ["greenness_gap", "intensity_gap"]
    names += [f"glcm_{p}" for p in GLCM_PROPS]
    return names


def extract_features(image_path: str) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # --- 1. Isolate the leaf from the backdrop ---------------------------
    leaf = _leaf_mask(rgb)
    leaf_rgb = rgb[leaf].astype(np.float32)
    leaf_hsv = hsv[leaf].astype(np.float32)

    # --- 2. Cluster within the leaf only ---------------------------------
    lesion = np.zeros_like(leaf)
    greenness_gap = 0.0
    intensity_gap = 0.0

    if leaf_rgb.shape[0] >= 2:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # KMEANS_PP_CENTERS is reproducible where the original RANDOM_CENTERS was not.
        _, labels, centers = cv2.kmeans(
            leaf_rgb, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

        green = _greenness(centers)
        lesion_cluster = int(np.argmin(green))  # deterministic: least green = lesion
        greenness_gap = float(abs(green[0] - green[1]))
        intensity_gap = float(abs(centers[0].mean() - centers[1].mean()))

        # If the two centres are nearly identical there is no real lesion; k=2
        # merely split noise. Report no lesion rather than inventing one.
        if np.linalg.norm(centers[0] - centers[1]) > 18.0:
            flat = np.zeros(leaf.sum(), dtype=bool)
            flat[labels == lesion_cluster] = True
            lesion[leaf] = flat

            # Centre distance alone is not enough: on a healthy leaf, sensor
            # noise still separates into two clusters that are far apart in RGB
            # but spatially incoherent -- thousands of single-pixel specks.
            # Real lesions are contiguous, so open the mask and drop tiny blobs.
            lesion = _clean_lesion_mask(lesion, leaf_area=int(leaf.sum()))

    # --- 3. Colour statistics on ORIGINAL pixels -------------------------
    features: list[float] = []
    for mask_pixels_rgb, mask_pixels_hsv in (
        (leaf_rgb, leaf_hsv),
        (rgb[lesion].astype(np.float32), hsv[lesion].astype(np.float32)),
    ):
        for channel in range(3):
            values = mask_pixels_rgb[:, channel] if mask_pixels_rgb.size else np.array([])
            features += _stats(values)
        for channel in range(3):
            values = mask_pixels_hsv[:, channel] if mask_pixels_hsv.size else np.array([])
            features += _stats(values)

    # --- 4. Shape descriptors from the real lesion mask ------------------
    leaf_area = max(int(leaf.sum()), 1)
    lesion_area = int(lesion.sum())
    lesion_ratio = lesion_area / leaf_area

    n_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(
        lesion.astype(np.uint8), connectivity=8
    )
    n_blobs = max(n_blobs - 1, 0)

    if n_blobs > 0:
        areas = blob_stats[1:, cv2.CC_STAT_AREA]
        largest_ratio = float(areas.max() / leaf_area)
        biggest = 1 + int(np.argmax(areas))
        blob = (blob_labels == biggest).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        area = float(areas.max())
        compactness = float((perimeter**2) / (4 * np.pi * area)) if area > 0 else 0.0
    else:
        largest_ratio = 0.0
        compactness = 0.0

    features += [lesion_ratio, float(n_blobs), largest_ratio, compactness]
    features += [greenness_gap, intensity_gap]

    # --- 5. GLCM on the ORIGINAL greyscale, 32 levels, 4 angles ----------
    quantised = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quantised = np.clip(quantised, 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(
        quantised,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    # Average over distances and angles for rotation invariance.
    features += [float(graycoprops(glcm, prop).mean()) for prop in GLCM_PROPS]

    return np.asarray(features, dtype=np.float32)
