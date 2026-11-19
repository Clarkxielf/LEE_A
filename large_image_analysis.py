"""PSIM analysis with contrast-based reporting and legacy field compatibility.

Fingerprint channel order and all measurement/model calculations are unchanged.
Channel 0 is reported as Dark-contrast area fraction, not a crystallographic
phase identification. Bright-contrast area fraction is its binary complement.
Legacy beta/alpha identifiers, checkpoint paths and calibration signatures are
retained where needed for compatibility; they do not establish phase identity.
"""


import os
import json
import warnings
import math
import random
import argparse
import csv
import hashlib

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    from gpu_selector import configure_automatic_gpu_selection
except ModuleNotFoundError as error:
    if error.name != "gpu_selector":
        raise
    configure_automatic_gpu_selection = None

if configure_automatic_gpu_selection is not None:
    configure_automatic_gpu_selection()

warnings.filterwarnings("ignore", message="The torchvision.datapoints")
warnings.filterwarnings("ignore", message="The torchvision.transforms.v2")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.special import expit
from skimage.filters import frangi
from skimage.segmentation import watershed
from skimage.morphology import medial_axis, skeletonize, remove_small_holes, remove_small_objects
from tqdm import tqdm
import cv2


# ========================== Configuration ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bump this whenever inference behavior changes.  It is included in the
# cached-result version so an old PSIM_results.npz cannot silently survive a
# code update and make a corrected run look unchanged.
PROGRAM_VERSION = "psim-v14-pair-boundary-separated-area-filter"
# Patch inference is unchanged; v12 model predictions remain reusable.
INFERENCE_VERSION = "psim-v12-source-crops-votes-boundaries"
TERMINOLOGY_VERSION = "contrast-area-v1"
FINGERPRINT_FIELDS = ["dark_contrast_area_fraction", "mean_lath_width_measurement_px"]


HF_LOCAL_FILES_ONLY = True
HF_CACHE_DIR = None

# Keep the original path as a fallback, but allow the inference environment to
# point at its own checkpoint/data tree without editing the program.
PROJECT_ROOT = os.environ.get("PSIM_PROJECT_ROOT", r"/home/user/data8t/LEE/LEE_A")
PRETRAINED_MODEL = os.path.join(PROJECT_ROOT, "models", "stable-diffusion-v1-5")
CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "Diffusion", "Deep-TiDiffusion-Outputs-FourClass-v3"
)
FP_STATS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "example_data_train_val_test",
    "morphology_fields_tc4_2d_four_class_v3",
    "fingerprints_2d_beta_lath_four_class_v3",
    "fp_stats_2d_beta_lath_four_class_v3.npy",
)
LARGE_IMAGE_PATH = os.environ.get(
    "PSIM_INPUT", os.path.join(PROJECT_ROOT, "data", "example_large_image")
)
OUTPUT_DIR = os.environ.get("PSIM_OUTPUT_DIR", os.path.join(CHECKPOINT_DIR, "psim_results_1"))
DEFAULT_PAIR_CALIBRATOR_PATH = os.environ.get(
    "PSIM_PAIR_CALIBRATOR",
    os.path.join(CHECKPOINT_DIR, "pair_calibrator.json"),
)

# A supplied training image is 1024 px and train/test resize it to 512 px.
# This source crop size assumes equal source microns/pixel. If scales differ:
# source_crop = training_source_size * training_mpp / large_image_mpp.
MODEL_INPUT_SIZE = 512
PATCH_SIZE = 1024
STRIDE = 256
SIGMA = PATCH_SIZE / 8
AREA_MODE = "patch-vote"
AUTO_PAIR_CALIBRATION = False
# The learned four-class head is already calibrated on patch fingerprints.
# Applying a second prototype-similarity weight during large-image fusion can
# create a global class bias, so keep spatial aggregation neutral by default.
LAMBDA_FINGERPRINT = 0.0
BATCH_SIZE = 4
NUM_TIMESTEPS = 3
FP_EPS = 0.001
# Evaluation should be repeatable.  The diffusion noise is part of the
# feature extractor, so leaving the RNG uncontrolled makes class overlays
# change between two runs of the same image.
INFERENCE_SEED = 100
MORPHOLOGY_MEASUREMENT_SIZE = 512
MICRONS_PER_SOURCE_PIXEL = 1.0 / 7.32
LATH_MIN_LENGTH_RATIO = 0.05
LATH_MAX_WIDTH_RATIO = 0.15

# Quantification settings for the calibrated reference workflow.
BETA_BLUR_KERNEL = 3
BETA_MIN_COMPONENT_AREA_PX = 8
BETA_IS_DARKER = True
BETA_CLOSE_KERNEL = 3
BETA_SHARPEN_AMOUNT = 1.25
BETA_ADAPTIVE_BLOCK_SIZE = 41
BETA_ADAPTIVE_C = 3
# This is a *global class-presence* rule.  It is deliberately not reused as a
# connected-component cutoff: Widmanstatten structures often form thin,
# elongated regions whose individual connected components are much smaller
# than 3% even though the class is real.
MIN_CLASS_PRESENCE_RATIO =  0.03 #0.03
# Optional local-island cleanup.  Zero means disabled; use the pixel cutoff
# below when a dataset-specific cleanup is justified.
MIN_CLASS_REGION_AREA_RATIO = 0.0
MIN_CLASS_REGION_AREA_PX = 0
LEGACY_QUANTITATIVE_OUTPUTS = (
    "PSIM_mean_lath_width_map.png",
    "PSIM_beta_phase_fraction_map.png",
    "PSIM_dark_contrast_area_fraction_map.png",
    "PSIM_roi_fingerprint_radar.png",
)

MAX_IMAGES = None
SKIP_EXISTING = True
# Reuse compatible model outputs when only hard-area reporting settings change.
REUSE_CACHED_PATCH_RESULTS = True
PANEL_DPI = 220
ROI_TOPK_SAMPLES = 20

FONT_BASE = 10.5
FONT_TITLE = 12.0
FONT_AXIS = 11.0
FONT_TICK = 10.0
FONT_LEGEND = 10.5
FONT_ANNOT = 10.0
PIL_LABEL_FONT_SIZE = 28
PIL_SMALL_FONT_SIZE = 20

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_AXIS,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "legend.frameon": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# A class can be locally present even when it occupies less than 5% globally.
ROI_PRESENT_THRESHOLD = 0.01

# Boundary refinement is applied only in a narrow band around the raw class
# transitions; pixels far from a transition keep their model prediction.
BOUNDARY_REFINEMENT_ENABLED = True
CLASS_POSTPROCESS_ENABLED = True
# Apply the configured 3% reporting cutoffs by default, independently of boundaries.
CLASS_AREA_FILTER_ENABLED = True
BOUNDARY_BAND_RADIUS_PX = 8
BOUNDARY_SMOOTH_KERNEL_PX = 3
BOUNDARY_MAX_VOTE_MARGIN = 0.35
# Image-guided refinement is restricted to the pair that is visually hard to
# separate.  Basket weave and Discontinuous keep the model vote unchanged.
BOUNDARY_REFINEMENT_CLASS_PAIRS = ((2, 3),)
# Conservative spatial estimates, not a learned probability calibration.
# All radii are SOURCE-image pixels. Turn off for a raw Gaussian comparison.
WIDMAN_GUIDED_ENABLED = True
WIDMAN_GUIDED_RADIUS_PX = 32
WIDMAN_GUIDED_BAND_PX = 64
WIDMAN_GUIDED_EPS = 0.0025
WIDMAN_GUIDED_MAX_DELTA = 0.10
WIDMAN_GUIDED_MIN_MASS = 0.75

CLASS_NAMES = [
    "Basket_weave",
    "Discontinuous",
    "Widmanstatten",
    "Alpha_coarsened_Widmanstatten",
]
ROI_NAMES = CLASS_NAMES + ["Transition_boundary"]

COLOR_MAP = {
    "Basket_weave": [225, 76, 70],
    "Discontinuous": [61, 102, 214],
    "Widmanstatten": [62, 173, 108],
    "Alpha_coarsened_Widmanstatten": [155, 89, 182],
}

ROI_COLOR_MAP = {
    "Basket_weave": (225, 76, 70),
    "Discontinuous": (61, 102, 214),
    "Widmanstatten": (62, 173, 108),
    "Alpha_coarsened_Widmanstatten": (155, 89, 182),
    "Transition_boundary": (209, 154, 0),
}

FINGERPRINT_LABELS = [
    "Dark-contrast area fraction",
    "Mean α-lath width",
]

PROMPT_DICT = {
    0: "basket weave microstructure",
    1: "discontinuous microstructure",
    2: "Widmanstatten microstructure",
    3: "alpha-coarsened Widmanstatten microstructure",
}

FP_DIM = len(FINGERPRINT_LABELS)
CONNECTIVITY_DIM = 3
CONNECTIVITY_NAMES = [
    "connectivity_length",
    "fragmentation_degree",
    "orientation_consistency",
]
CONNECTIVITY_THRESHOLD = 0.5
# Apply the auxiliary score only within the Basket/Discontinuous pair.  This
# gives fragmented tissue a controlled push toward Discontinuous without
# changing the probability mass assigned to the other two classes.
# A low connectivity score is evidence for Discontinuous.  The auxiliary
# head is only used inside the Basket/Discontinuous pair, so this cannot
# promote a third class.  A small gain avoids letting a locally aligned,
# broken patch be classified as Basket_weave solely from texture direction.
CONNECTIVITY_LOGIT_GAIN = 0.0

# Width is used as an evidence term for the Basket_weave/Discontinuous pair.
# The threshold and scale are learned from class_mean_fingerprints.npy when it
# is available, so they stay in the normalized fingerprint space used by the
# checkpoint. These values are only a fallback for older checkpoints.
WIDTH_AWARE_ENABLED = False
WIDTH_LOGIT_GAIN = 0.20
WIDTH_FALLBACK_MIDPOINT_Z = 0.0
WIDTH_FALLBACK_HALF_GAP_Z = 1.0
WIDTH_EVIDENCE_CLIP = 3.0
# When the measured width is clearly on the Discontinuous side of the
# calibrated midpoint, do not allow an over-confident Basket_weave logit to
# override that independent morphology evidence.
# Kept for backwards-compatible diagnostics; hard pairwise forcing is disabled
# because morphology must not overturn a learned class decision.
WIDTH_HARD_GATE_Z = 2.00
WIDTH_HARD_MARGIN = 0.05
# Width is only allowed to resolve an existing Basket/Discontinuous ambiguity.
# It must never promote either member above an originally dominant third class.
WIDTH_PAIR_MIN_MASS = 0.55
WIDTH_PAIR_MAX_TOP_GAP = 0.12
# Auxiliary morphology is a tie-breaker only.  Do not change a patch when the
# learned classifier already has a clear Basket/Discontinuous preference.
PAIRWISE_MAX_LOGIT_GAP = 0.25
STRUCTURE_AWARE_ENABLED = False
# These terms are deliberately weaker than the learned classifier.  They are
# only a tie-breaker for the Basket/Discontinuous pair, where the visual cue
# is fine, open laths rather than a closed basket network.
STRUCTURE_LOGIT_GAIN = 0.20
STRUCTURE_PAIR_MIN_MASS = 0.55
STRUCTURE_PAIR_MAX_TOP_GAP = 0.16
WIDTH_CALIBRATION_VERSION = INFERENCE_VERSION + "-baseline"
PAIR_FEATURE_VERSION = "unclosed-mask-topology-v2"
PAIR_FEATURE_NAMES = [
    "lath_width_p25_px", "lath_width_median_px", "lath_width_p75_px",
    "endpoint_per_skeleton_pixel", "short_fragment_length_fraction",
    "component_per_skeleton_pixel", "enclosed_area_fraction",
    "enclosed_regions_per_10000_pixels", "line_area_fraction",
]
PAIR_MIN_MASS = 0.50
CALIBRATION_PATCH_STRIDE = None
MAX_CALIBRATION_PATCHES_PER_IMAGE = 64

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
CURRENT_RESULT_KEYS = {
    "class_vote_field", "raw_class_id_map", "spatial_config_json",
    "program_version",
    "prob_field",
    "dark_contrast_area_fraction_field",
    "mean_lath_width_um_field",
    "connectivity_score_field",
    "width_calibration_version",
    "patch_raw_probs", "patch_raw_logits", "patch_pair_features",
    "patch_feature_valid", "patch_calibration_applied", "patch_structure_scores",
}

RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def sample_multi_timesteps(batch_size, num_train_timesteps, K, device, generator=None):
    t = torch.zeros(batch_size, K, device=device, dtype=torch.long)
    low_bound = max(1, num_train_timesteps // 10)
    high_bound = num_train_timesteps * 8 // 10
    mid_start = num_train_timesteps // 5
    t[:, 0] = torch.randint(0, low_bound, (batch_size,), device=device, generator=generator)
    if K > 1:
        t[:, -1] = torch.randint(high_bound, num_train_timesteps, (batch_size,), device=device, generator=generator)
    for k in range(1, K - 1):
        t[:, k] = torch.randint(mid_start, high_bound, (batch_size,), device=device, generator=generator)
    return t


def set_inference_seed(seed=INFERENCE_SEED):
    """Make diffusion-noise sampling reproducible for evaluation overlays."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def split_repeated_features_by_timestep(features, batch_size, num_timesteps):
    expected = batch_size * num_timesteps
    if features.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} repeated features, received {features.shape[0]}."
        )
    features_bk = features.reshape(batch_size, num_timesteps, *features.shape[1:])
    return list(features_bk.unbind(dim=1))


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = F.relu(x) + self.eps
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def _robust_z01(x, lower=1, upper=99, eps=1e-6):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [lower, upper])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)


def _to_gray_u8(img_array):
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _illumination_correct_gray(gray_u8):
    k = max(15, int(round(min(gray_u8.shape) * 0.16)))
    if k % 2 == 0:
        k += 1
    background = cv2.GaussianBlur(gray_u8, (k, k), 0)
    corrected = gray_u8.astype(np.float32) - background.astype(np.float32) + float(background.mean())
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _clahe_gray(gray_u8):
    corrected = _illumination_correct_gray(gray_u8)
    blurred = cv2.GaussianBlur(corrected, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)


def _iterative_threshold(gray_u8, initial=127.0, tolerance=1.0, max_iter=32):
    threshold = float(initial)
    for _ in range(max_iter):
        lower = gray_u8[gray_u8 <= threshold]
        upper = gray_u8[gray_u8 > threshold]
        if lower.size == 0 or upper.size == 0:
            break
        updated = 0.5 * (float(lower.mean()) + float(upper.mean()))
        if abs(updated - threshold) <= tolerance:
            threshold = updated
            break
        threshold = updated
    return float(threshold)


def _close_mask_for_quantification(mask):
    """Close only a measurement copy; never use it for topology features."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BETA_CLOSE_KERNEL, BETA_CLOSE_KERNEL)
    )
    closed = cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    ).astype(bool)
    return _remove_small_components(closed, BETA_MIN_COMPONENT_AREA_PX)


def _reference_beta_masks(img_array):
    """Return (measurement_mask, topology_mask, enhanced, threshold).

    The measurement mask preserves the historical contrast-area measurement.
    The topology mask deliberately retains narrow breaks for lath continuity,
    endpoint, fragmentation and pair-calibration features.
    """
    gray = _to_gray_u8(img_array)
    blurred = cv2.GaussianBlur(gray, (BETA_BLUR_KERNEL, BETA_BLUR_KERNEL), 0)
    sharpened = cv2.addWeighted(
        gray, BETA_SHARPEN_AMOUNT, blurred, 1.0 - BETA_SHARPEN_AMOUNT, 0.0
    )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sharpened)
    threshold = _iterative_threshold(enhanced, initial=127.0, tolerance=1.0)
    global_mask = enhanced <= threshold if BETA_IS_DARKER else enhanced > threshold
    block_size = max(3, int(BETA_ADAPTIVE_BLOCK_SIZE))
    if block_size % 2 == 0:
        block_size += 1
    adaptive_type = cv2.THRESH_BINARY_INV if BETA_IS_DARKER else cv2.THRESH_BINARY
    local_mask = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        adaptive_type, block_size, BETA_ADAPTIVE_C,
    ).astype(bool)
    blackhat = cv2.morphologyEx(
        enhanced, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    line_support = blackhat >= np.percentile(blackhat, 45.0)
    candidate_mask = global_mask | (local_mask & line_support)
    # Keep the train-time order for the legacy fingerprint: close, then clean.
    measurement_mask = _close_mask_for_quantification(candidate_mask)
    topology_mask = _remove_small_components(candidate_mask, BETA_MIN_COMPONENT_AREA_PX)
    beta_fraction = float(measurement_mask.mean())
    if beta_fraction < 0.02 or beta_fraction > 0.80:
        otsu_threshold, fallback = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        candidate_mask = fallback == 0 if BETA_IS_DARKER else fallback > 0
        measurement_mask = _close_mask_for_quantification(candidate_mask)
        topology_mask = _remove_small_components(candidate_mask, BETA_MIN_COMPONENT_AREA_PX)
        threshold = float(otsu_threshold)
    return measurement_mask.astype(bool), topology_mask.astype(bool), enhanced, threshold


def _reference_beta_mask(img_array):
    """Historical closed dark-contrast mask; legacy function name retained."""
    measurement_mask, _, enhanced, threshold = _reference_beta_masks(img_array)
    return measurement_mask, enhanced, threshold

def _remove_small_components(mask, min_area):
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool)
    # Indexing every component in a Python loop dominates runtime on textured
    # 512x512 patches.  Build the keep mask once and remap labels in C-backed
    # NumPy code instead.
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels].astype(bool)


def _deterministic_medial_axis(mask):
    try:
        return medial_axis(mask.astype(bool), return_distance=True, rng=0)
    except TypeError:
        return medial_axis(mask.astype(bool), return_distance=True, random_state=0)


def remove_legacy_quantitative_outputs(output_dir):
    for filename in LEGACY_QUANTITATIVE_OUTPUTS:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)

def _estimate_beta_line_mask(enhanced, alpha_mask):
    dark_response = _robust_z01(255 - enhanced)
    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    blackhat = _robust_z01(blackhat)
    phase_edge = cv2.morphologyEx(
        alpha_mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(np.float32)
    phase_edge = cv2.dilate(
        phase_edge.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(np.float32)

    beta_score = 0.55 * dark_response + 0.30 * blackhat + 0.15 * phase_edge
    beta_mask = beta_score > np.percentile(beta_score, 82)
    beta_mask &= (dark_response > np.percentile(dark_response, 55)) | (phase_edge > 0)
    beta_mask = cv2.morphologyEx(
        beta_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    ).astype(bool)
    beta_mask = _remove_small_components(beta_mask, max(4, int(0.0003 * beta_mask.size)))
    return beta_mask.astype(bool)


def _segment_alpha_beta(gray_u8, alpha_is_darker=False, min_area_ratio=0.0015):
    del min_area_ratio
    beta, enhanced, _ = _reference_beta_mask(gray_u8)
    alpha = beta if alpha_is_darker else ~beta
    if alpha_is_darker:
        beta = ~alpha
    return alpha.astype(bool), beta.astype(bool), enhanced


def _degree_map_from_skeleton(skel_bool):
    kernel = np.ones((3, 3), dtype=np.uint8)
    conv = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    degree = np.zeros_like(conv, dtype=np.uint8)
    degree[skel_bool] = conv[skel_bool] - 1
    return degree


def _extract_lath_tracks(alpha_mask, min_len_px=8, min_width_px=1.0, max_width_px=80.0):
    if alpha_mask.sum() == 0:
        return np.zeros_like(alpha_mask, dtype=np.int32), np.zeros_like(alpha_mask, dtype=np.float32), []

    skel, dist = _deterministic_medial_axis(alpha_mask)
    dist = dist.astype(np.float32)
    if not skel.any():
        return np.zeros_like(alpha_mask, dtype=np.int32), dist, []

    degree = _degree_map_from_skeleton(skel)
    skel_line = skel.copy()
    skel_line[degree >= 3] = False
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_line.astype(np.uint8), connectivity=8)

    # Do not run ``labels == label_idx`` / ``np.where`` for every component.
    # Fine basket textures can contain thousands of skeleton fragments, and
    # each such operation scans the full image.  Accumulate width and spatial
    # moments for all labels in one vectorized pass instead.
    length_px = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    active = labels > 0
    active_labels = labels[active].astype(np.intp, copy=False)
    if active_labels.size == 0:
        return np.zeros_like(labels, dtype=np.int32), dist, []

    active_widths = (dist[active] * 2.0).astype(np.float64, copy=False)
    valid_width = np.isfinite(active_widths) & (active_widths > 0)
    width_sum = np.bincount(
        active_labels[valid_width],
        weights=active_widths[valid_width],
        minlength=n_labels,
    )
    width_count = np.bincount(active_labels[valid_width], minlength=n_labels)
    mean_width = width_sum / np.maximum(width_count, 1)

    ys, xs = np.nonzero(active)
    xs = xs.astype(np.float64, copy=False)
    ys = ys.astype(np.float64, copy=False)
    count = np.bincount(active_labels, minlength=n_labels).astype(np.float64)
    sum_x = np.bincount(active_labels, weights=xs, minlength=n_labels)
    sum_y = np.bincount(active_labels, weights=ys, minlength=n_labels)
    sum_xx = np.bincount(active_labels, weights=xs * xs, minlength=n_labels)
    sum_yy = np.bincount(active_labels, weights=ys * ys, minlength=n_labels)
    sum_xy = np.bincount(active_labels, weights=xs * ys, minlength=n_labels)

    candidate_ids = np.flatnonzero(
        (np.arange(n_labels) > 0)
        & (length_px >= int(min_len_px))
        & (width_count > 0)
        & (mean_width >= float(min_width_px))
        & (mean_width <= float(max_width_px))
    )
    remap = np.zeros(n_labels, dtype=np.int32)
    remap[candidate_ids] = np.arange(1, len(candidate_ids) + 1, dtype=np.int32)
    out_labels = remap[labels]

    tracks = []
    for new_id, label_idx in enumerate(candidate_ids, start=1):
        n = count[label_idx]
        if n >= 2:
            mean_x = sum_x[label_idx] / n
            mean_y = sum_y[label_idx] / n
            var_x = max(sum_xx[label_idx] / n - mean_x * mean_x, 0.0)
            var_y = max(sum_yy[label_idx] / n - mean_y * mean_y, 0.0)
            cov_xy = sum_xy[label_idx] / n - mean_x * mean_y
            trace = var_x + var_y
            if trace > 1e-8:
                spread = np.hypot(var_x - var_y, 2.0 * cov_xy)
                angle = float((0.5 * np.arctan2(2.0 * cov_xy, var_x - var_y)) % np.pi)
                linearity = float((trace + spread) / (2.0 * trace))
            else:
                angle = 0.0
                linearity = 0.0
        else:
            angle = 0.0
            linearity = 0.0
        tracks.append({
            "label": new_id,
            "length_px": int(length_px[label_idx]),
            "mean_width_px": float(mean_width[label_idx]),
            "angle": angle,
            "linearity": linearity,
        })
    return out_labels, dist, tracks


def _build_quantification_masks(img_array, mask_size=(128, 128), return_topology=False):
    gray = _to_gray_u8(img_array)
    measurement_side = max(MORPHOLOGY_MEASUREMENT_SIZE, int(max(mask_size)))
    gray = cv2.resize(gray, (measurement_side, measurement_side), interpolation=cv2.INTER_AREA)
    beta_mask, beta_topology, enhanced, _ = _reference_beta_masks(gray)
    alpha_mask = ~beta_mask
    alpha_topology = ~beta_topology
    if return_topology:
        return gray, enhanced, alpha_mask, beta_mask, alpha_topology, beta_topology
    return gray, enhanced, alpha_mask, beta_mask


def _compose_morphology_field(img_array, mask_size=(128, 128), return_analysis=False):
    gray, enhanced, alpha_mask, beta_mask, alpha_topology, beta_topology = (
        _build_quantification_masks(img_array, mask_size, return_topology=True)
    )
    enhanced_small = cv2.resize(enhanced, mask_size, interpolation=cv2.INTER_AREA)
    vesselness = frangi(
        enhanced_small, sigmas=(1, 2, 3, 4, 5), beta=0.5, gamma=15, black_ridges=False
    )
    vesselness = _normalize01(np.clip(vesselness, 0, 1))
    phase_edge = cv2.morphologyEx(
        alpha_topology.astype(np.uint8), cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(np.float32)
    phase_edge = cv2.resize(phase_edge, mask_size, interpolation=cv2.INTER_AREA)
    lath_labels, alpha_dist, tracks = _extract_lath_tracks(
        alpha_topology,
        min_len_px=max(8, int(round(LATH_MIN_LENGTH_RATIO * min(alpha_topology.shape)))),
        min_width_px=1.0,
        max_width_px=max(8.0, LATH_MAX_WIDTH_RATIO * min(alpha_topology.shape)),
    )
    # Do not dilate: dilation turns nearby fragments into visually continuous laths.
    lath_skeleton = cv2.resize(
        (lath_labels > 0).astype(np.float32), mask_size, interpolation=cv2.INTER_AREA
    )
    beta_line = cv2.resize(beta_topology.astype(np.float32), mask_size, interpolation=cv2.INTER_AREA)
    line_field = _normalize01(
        0.45 * vesselness + 0.25 * phase_edge + 0.20 * lath_skeleton + 0.10 * beta_line
    )
    max_width_px = max(8.0, LATH_MAX_WIDTH_RATIO * min(alpha_topology.shape))
    local_width_px = np.where(alpha_topology, alpha_dist * 2.0, 0.0).astype(np.float32)
    width_field = np.log1p(np.clip(local_width_px, 0.0, max_width_px)) / np.log1p(max_width_px)
    width_field = cv2.resize(width_field, mask_size, interpolation=cv2.INTER_AREA)
    field = np.stack([line_field, width_field], axis=0).astype(np.float32)
    if return_analysis:
        # Keep the historical closed-mask fingerprint for checkpoint compatibility.
        fp_labels, fp_dist, fp_tracks = _extract_lath_tracks(
            alpha_mask,
            min_len_px=max(8, int(round(LATH_MIN_LENGTH_RATIO * min(alpha_mask.shape)))),
            min_width_px=1.0,
            max_width_px=max(8.0, LATH_MAX_WIDTH_RATIO * min(alpha_mask.shape)),
        )
        fp_raw = fingerprint_from_masks(alpha_mask, beta_mask, fp_dist, fp_tracks)
        # Pair topology features use the unclosed mask and original measurement scale.
        features, valid = measure_pair_features(enhanced, alpha_topology, alpha_dist)
        return field, fp_raw, features, valid
    return field

def generate_morphology_fields_from_array(img_array, mask_size=(128, 128)):
    return _compose_morphology_field(img_array, mask_size=mask_size)


def compute_extended_fingerprint_np(field, img_array=None):
    if img_array is not None:
        _, _, alpha_mask, beta_mask = _build_quantification_masks(
            img_array,
            mask_size=(field.shape[-1], field.shape[-2]),
        )
    else:
        raise ValueError("Original image is required: a downsampled field is not a phase mask.")

    measurement_side = min(alpha_mask.shape)
    _, alpha_dist, tracks = _extract_lath_tracks(
        alpha_mask,
        min_len_px=max(8, int(round(LATH_MIN_LENGTH_RATIO * measurement_side))),
        min_width_px=1.0,
        max_width_px=max(8.0, LATH_MAX_WIDTH_RATIO * measurement_side),
    )

    return fingerprint_from_masks(alpha_mask, beta_mask, alpha_dist, tracks)


def fingerprint_from_masks(alpha_mask, beta_mask, alpha_dist, tracks):
    """Preserve the original 512-pixel measurement definition, not field pixels."""
    if tracks:
        mean_lath_width = float(np.average(
            [track["mean_width_px"] for track in tracks],
            weights=[max(track["length_px"], 1) for track in tracks],
        ))
    else:
        fallback_skeleton, _ = _deterministic_medial_axis(alpha_mask)
        valid_width = alpha_dist[fallback_skeleton] * 2.0
        mean_lath_width = float(valid_width.mean()) if valid_width.size > 0 else 0.0

    beta_phase_fraction = float(beta_mask.mean())
    return np.array([beta_phase_fraction, mean_lath_width], dtype=np.float32)


def measure_pair_features(enhanced, alpha_mask, alpha_dist):
    """Measure topology without closing gaps; lengths are measurement pixels.

    These are segmentation-dependent descriptors, not ground-truth topology.
    Training and inference must use the same scale and extraction settings.
    """
    gray = np.asarray(enhanced, dtype=np.uint8)
    alpha_skeleton = skeletonize(alpha_mask.astype(bool))
    widths = 2.0 * alpha_dist[alpha_skeleton]
    widths = widths[np.isfinite(widths) & (widths > 0)]
    width_quantiles = np.percentile(widths, [25, 50, 75]) if widths.size else np.zeros(3)

    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lines = gray <= threshold if BETA_IS_DARKER else gray > threshold
    lines = _remove_small_components(lines, BETA_MIN_COMPONENT_AREA_PX)
    skeleton = skeletonize(lines)
    degree = _degree_map_from_skeleton(skeleton)
    interior = np.ones(gray.shape, dtype=bool)
    interior[:2, :] = interior[-2:, :] = False
    interior[:, :2] = interior[:, -2:] = False
    length = max(int((skeleton & interior).sum()), 1)
    endpoint_rate = float(((degree == 1) & skeleton & interior).sum()) / length

    # Exclude edge-truncated fragments from the short-fragment statistic.
    segments = skeleton & (degree < 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(segments.astype(np.uint8), connectivity=8)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.ones(n, dtype=bool)
    keep[edge_ids] = False
    keep[0] = False
    lengths = stats[keep, cv2.CC_STAT_AREA]
    short_limit = max(8, int(round(LATH_MIN_LENGTH_RATIO * min(gray.shape))))
    short_fraction = float(lengths[lengths < short_limit].sum()) / max(int(lengths.sum()), 1)

    # Dual connectivity: 8-connected lines, 4-connected complement regions.
    n_holes, hole_ids, hole_stats, _ = cv2.connectedComponentsWithStats(
        (~lines).astype(np.uint8), connectivity=4
    )
    border_ids = np.unique(np.concatenate([hole_ids[0], hole_ids[-1], hole_ids[:, 0], hole_ids[:, -1]]))
    enclosed = np.ones(n_holes, dtype=bool)
    enclosed[border_ids] = False
    enclosed[0] = False
    enclosed &= hole_stats[:, cv2.CC_STAT_AREA] >= BETA_MIN_COMPONENT_AREA_PX
    enclosed_area = float(hole_stats[enclosed, cv2.CC_STAT_AREA].sum()) / gray.size
    line_fraction = float(lines.mean())
    features = np.array([
        *width_quantiles, endpoint_rate, short_fraction,
        float(len(lengths)) / length, enclosed_area,
        float(enclosed.sum()) * 10000.0 / gray.size, line_fraction,
    ], dtype=np.float32)
    valid = bool(widths.size >= 16 and length >= 16 and 0.01 < line_fraction < 0.95
                 and float(gray.std()) > 5.0 and np.isfinite(features).all())
    return features, valid


def compute_open_fine_structure_score(field):
    """Estimate the visual cue that separates fine, open tissue from baskets.

    The score is intentionally a weak auxiliary signal.  It combines narrow
    lath evidence with gaps that a small morphological closing would fill and
    local line fragmentation.  Higher values mean finer and less closed.
    """
    field = np.asarray(field, dtype=np.float32)
    line = np.clip(field[0], 0.0, 1.0)
    width = np.clip(field[1], 0.0, 1.0)
    threshold = float(np.percentile(line, 72.0))
    line_mask = (line >= threshold).astype(np.uint8)
    if not np.any(line_mask):
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    gap_ratio = float(np.logical_and(closed > 0, line_mask == 0).mean())
    closed_area = float((closed > 0).mean())
    gap_score = np.clip(gap_ratio / max(closed_area, 1e-6) * 4.0, 0.0, 1.0)

    fine_score = np.clip((0.62 - float(np.median(width))) / 0.42, 0.0, 1.0)
    local_mean = uniform_filter(line, size=7, mode="nearest")
    fragmentation = np.clip(
        float(np.mean(np.abs(line - local_mean))) / 0.16,
        0.0,
        1.0,
    )
    # Fine texture alone is common in Basket_weave.  Require the open-gap
    # signal to dominate before emitting a positive discontinuity cue.
    open_score = np.clip(gap_score * (0.55 + 0.45 * fragmentation), 0.0, 1.0)
    return float(np.clip(0.20 * fine_score + 0.60 * open_score + 0.20 * fragmentation, 0.0, 1.0))


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=8.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear.")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def _set_nested_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def inject_lora_into_unet(unet, rank=8, alpha=8.0, dropout=0.0):
    target_suffixes = ("to_q", "to_k", "to_v", "to_out.0")
    replaced = []
    for name, module in list(unet.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes):
            _set_nested_module(unet, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(name)
    return replaced


class TokenizedACE(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4, n_classes=3, fp_dim=2):
        super().__init__()
        self.fp_token_proj = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(fp_dim)])
        self.cls_embed = nn.Embedding(n_classes, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, fp_dim + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, fingerprint, class_ids):
        fp_tokens = torch.stack(
            [proj(fingerprint[:, i:i + 1]) for i, proj in enumerate(self.fp_token_proj)],
            dim=1,
        )
        cls_tok = self.cls_embed(class_ids).unsqueeze(1)
        tokens = torch.cat([fp_tokens, cls_tok], dim=1) + self.pos_embed
        encoded = self.transformer(tokens)
        return self.out_proj(encoded.mean(dim=1))


class MidBlockTransformer(nn.Module):
    def __init__(self, channels, num_heads=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        return x.transpose(1, 2).reshape(B, C, H, W) + residual


class PhysicalQueryAttention(nn.Module):
    def __init__(self, feat_channels=1280, embed_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Conv2d(feat_channels, embed_dim, 1)
        self.v_proj = nn.Conv2d(feat_channels, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, feat_channels, 1)
        self.last_attn_weights = None

    def forward(self, features, physics_emb):
        B, C, H, W = features.shape
        Q = self.q_proj(physics_emb).view(B, self.num_heads, 1, self.head_dim)
        K = self.k_proj(features).view(B, self.num_heads, self.head_dim, H * W)
        V = self.v_proj(features).view(B, self.num_heads, self.head_dim, H * W)
        attn = torch.matmul(Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        self.last_attn_weights = attn.detach()
        context = torch.matmul(attn, V.transpose(-2, -1)).transpose(1, 2).reshape(
            B, self.num_heads * self.head_dim, 1, 1
        )
        return features * self.out_proj(context).sigmoid()


class UNetWithMidFeature(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        captured = {}

        def hook(module, input, output):
            captured["mid"] = output

        handle = self.unet.mid_block.register_forward_hook(hook)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        handle.remove()
        return noise_pred, captured["mid"]


class UnifiedHeadNoSeg_v2(nn.Module):
    def __init__(self, in_channels=1280, n_classes=4, hidden_dim=256):
        super().__init__()
        self.cls_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.cls_branch = nn.Sequential(
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            GeM(),
            nn.Flatten(),
            nn.Linear(hidden_dim, n_classes),
        )

        self.fp_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.fp_branch = nn.Sequential(
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, FP_DIM),
        )

    def forward(self, features):
        cls_feat = self.cls_proj(features)
        cls_logits = self.cls_branch(cls_feat)
        fp_feat = self.fp_proj(features)
        fingerprint = self.fp_branch(fp_feat)
        return cls_logits, fingerprint, cls_feat, fp_feat


class ConnectivityHead(nn.Module):
    """Predict the three structure descriptors from aggregated UNet features."""

    def __init__(self, in_channels=1280, hidden_dim=128, out_dim=CONNECTIVITY_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(16, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cls_features):
        return torch.sigmoid(self.net(cls_features))


def connectivity_score_from_descriptors(descriptors):
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim == 1:
        descriptors = descriptors[None, :]
    if descriptors.shape[-1] != CONNECTIVITY_DIM:
        raise ValueError(
            f"Expected {CONNECTIVITY_DIM} connectivity values, got {descriptors.shape[-1]}"
        )
    score = (
        0.55 * descriptors[:, 0]
        - 0.35 * descriptors[:, 1]
        + 0.10 * descriptors[:, 2]
        + 0.35
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def derive_width_calibration(class_means=None):
    """Return (midpoint, half_gap, basket_direction) in normalized width space."""
    if class_means is not None:
        means = np.asarray(class_means, dtype=np.float32)
        if means.ndim == 2 and means.shape[0] >= 2 and means.shape[1] >= 2:
            basket_mean = float(means[0, 1])
            discontinuous_mean = float(means[1, 1])
            if np.isfinite(basket_mean) and np.isfinite(discontinuous_mean):
                gap = abs(basket_mean - discontinuous_mean)
                if gap > 1e-4:
                    return (
                        0.5 * (basket_mean + discontinuous_mean),
                        max(0.5 * gap, 0.25),
                        1.0 if basket_mean > discontinuous_mean else -1.0,
                    )
    return (
        float(WIDTH_FALLBACK_MIDPOINT_Z),
        float(max(WIDTH_FALLBACK_HALF_GAP_Z, 0.25)),
        1.0,
    )


def adjust_pairwise_class_probabilities(
    probabilities,
    connectivity_descriptors=None,
    normalized_width=None,
    open_fine_structure=None,
    width_calibration=None,
):
    """Calibrate only the Basket_weave/Discontinuous probability pair.

    Width and connectivity are evidence terms; all other class probabilities
    retain their relative logits. ``normalized_width`` is column 1 of the
    normalized two-dimensional fingerprint produced for each patch.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "probabilities must have shape (n_patches, n_classes) with at least two classes"
        )
    probabilities = np.clip(probabilities, 1e-6, 1.0)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)
    logits = np.log(probabilities)
    delta = np.zeros(probabilities.shape[0], dtype=np.float32)

    # Only calibrate a patch when Basket/Discontinuous already contains the
    # classifier's top class. A narrow Widmanstatten lath is therefore not
    # re-labelled merely because it resembles Discontinuous in width.
    raw_top = np.argmax(probabilities, axis=1)
    pair_mass = probabilities[:, 0] + probabilities[:, 1]
    pair_top = np.maximum(probabilities[:, 0], probabilities[:, 1])
    other_top = (
        np.max(probabilities[:, 2:], axis=1)
        if probabilities.shape[1] > 2
        else np.zeros(probabilities.shape[0], dtype=np.float32)
    )
    pair_logit_gap = np.abs(logits[:, 1] - logits[:, 0])
    pair_applicable = (
        np.isin(raw_top, (0, 1))
        & (pair_mass >= WIDTH_PAIR_MIN_MASS)
        & ((pair_top - other_top) >= -WIDTH_PAIR_MAX_TOP_GAP)
        & (pair_logit_gap <= PAIRWISE_MAX_LOGIT_GAP)
    )

    scores = np.zeros(probabilities.shape[0], dtype=np.float32)
    width_evidence = None
    if connectivity_descriptors is not None:
        scores = connectivity_score_from_descriptors(connectivity_descriptors)
        delta[pair_applicable] += (
            (scores[pair_applicable] - CONNECTIVITY_THRESHOLD) * CONNECTIVITY_LOGIT_GAIN
        )

    if WIDTH_AWARE_ENABLED and normalized_width is not None:
        width = np.asarray(normalized_width, dtype=np.float32).reshape(-1)
        if width.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "normalized_width length does not match probability batch: "
                f"{width.shape[0]} versus {probabilities.shape[0]}"
            )
        midpoint, half_gap, basket_direction = (
            width_calibration
            if width_calibration is not None
            else derive_width_calibration()
        )
        width_evidence = basket_direction * (width - midpoint) / max(half_gap, 0.25)
        width_evidence = np.clip(width_evidence, -WIDTH_EVIDENCE_CLIP, WIDTH_EVIDENCE_CLIP)
        delta[pair_applicable] += (
            WIDTH_LOGIT_GAIN * np.tanh(width_evidence[pair_applicable])
        )

    if STRUCTURE_AWARE_ENABLED and open_fine_structure is not None:
        structure = np.asarray(open_fine_structure, dtype=np.float32).reshape(-1)
        if structure.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "open_fine_structure length does not match probability batch: "
                f"{structure.shape[0]} versus {probabilities.shape[0]}"
            )
        structure_pair = (
            np.isin(raw_top, (0, 1))
            & (pair_mass >= STRUCTURE_PAIR_MIN_MASS)
            & ((pair_top - other_top) >= -STRUCTURE_PAIR_MAX_TOP_GAP)
            & (pair_logit_gap <= PAIRWISE_MAX_LOGIT_GAP)
        )
        # Require the independent width cue to agree with the open/fine cue.
        # This blocks a global Basket_weave -> Discontinuous flip caused by
        # fine texture alone.
        if width_evidence is not None:
            structure_pair &= width_evidence <= 0.0
        # Positive structure evidence means fine/open and therefore favours
        # Discontinuous.  Do not penalize Basket below the neutral midpoint:
        # this protects continuous basket-weave regions from a false global
        # flip when the auxiliary score is noisy.
        structure_delta = np.maximum(structure[structure_pair] - 0.85, 0.0)
        delta[structure_pair] -= STRUCTURE_LOGIT_GAIN * structure_delta

    # Reallocate probability only inside the pair and preserve its total mass.
    # This prevents the calibration itself from stealing probability from
    # Widmanstatten or Alpha-coarsened Widmanstatten.
    adjusted = probabilities.copy()
    if np.any(pair_applicable):
        pair_logits = logits[pair_applicable, :2].copy()
        pair_logits[:, 0] += delta[pair_applicable]
        pair_logits[:, 1] -= delta[pair_applicable]
        pair_logits -= pair_logits.max(axis=1, keepdims=True)
        pair_conditional = np.exp(pair_logits)
        pair_conditional /= np.maximum(pair_conditional.sum(axis=1, keepdims=True), 1e-8)
        adjusted[pair_applicable, :2] = (
            pair_mass[pair_applicable, None] * pair_conditional
        )
    adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-8)
    return adjusted.astype(np.float32), scores


class TimestepAwareChannelGate(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, feat_dim))

    def forward(self, features, timesteps):
        t_norm = timesteps.float().unsqueeze(1) / 1000.0
        gate = torch.sigmoid(self.t_embed(t_norm)).unsqueeze(-1).unsqueeze(-1)
        return features * gate


class MultiTimestepAggregator(nn.Module):
    def __init__(self, feat_dim=1280, fusion_dim=256, max_timesteps=5):
        super().__init__()
        self.channel_gate = TimestepAwareChannelGate(feat_dim)
        self.t_weights = nn.Parameter(torch.ones(max_timesteps) / max_timesteps)
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim, fusion_dim, 3, padding=1),
            nn.GroupNorm(16, fusion_dim),
            nn.SiLU(),
            nn.Conv2d(fusion_dim, feat_dim, 3, padding=1),
            nn.GroupNorm(32, feat_dim),
        )

    def forward(self, features_list, timesteps_matrix):
        K = len(features_list)
        weights = F.softmax(self.t_weights[:K], dim=0)
        aggregated = 0
        for k in range(K):
            gated = self.channel_gate(features_list[k], timesteps_matrix[:, k])
            aggregated = aggregated + weights[k] * gated
        return F.silu(self.fusion(aggregated) + features_list[0])


def hf_kwargs():
    kwargs = {"local_files_only": HF_LOCAL_FILES_ONLY}
    if HF_CACHE_DIR is not None:
        kwargs["cache_dir"] = HF_CACHE_DIR
    return kwargs


def ensure_file_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found:\n{path}\n"
            "Please check whether CHECKPOINT_DIR / FP_STATS_PATH is pointing to the correct project folder."
        )
    return path


def safe_name(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_input_images(path):
    if os.path.isfile(path):
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    if os.path.isdir(path):
        image_paths = []
        for root, _, files in os.walk(path):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, name))
        image_paths.sort()
        if not image_paths:
            raise FileNotFoundError(f"No image file found in directory tree: {path}")
        return image_paths

    raise FileNotFoundError(f"Path does not exist: {path}")


# Legacy export names are explicit aliases, never extra model inputs.
CONTRAST_SUMMARY_ALIASES = {
    "global_dark_contrast_area_fraction": "global_beta_phase_fraction",
    "global_bright_contrast_area_fraction": "global_alpha_phase_fraction",
    "dark_contrast_quantification_method": "beta_quantification_method",
    "dark_contrast_preprocessing": "beta_preprocessing",
    "dark_contrast_min_component_area_px": "beta_min_component_area_px",
    "dark_contrast_adaptive_block_size": "beta_adaptive_block_size",
    "dark_contrast_adaptive_c": "beta_adaptive_c",
    "dark_contrast_threshold_enhanced_gray": "beta_threshold_enhanced_gray",
}


def normalize_contrast_summary(summary):
    """Accept legacy names, expose canonical names, and reject conflicting aliases.

    Global fractions retain the historical reconstructed-field mean. They are
    not replaced with the direct whole-image mask fraction used in the plot.
    """
    result = dict(summary)
    for current, legacy in CONTRAST_SUMMARY_ALIASES.items():
        if current in result and legacy in result:
            left, right = result[current], result[legacy]
            equal = (left == right) if isinstance(left, str) or isinstance(right, str) else np.allclose(
                left, right, rtol=1e-6, atol=1e-8, equal_nan=True
            )
            if not equal:
                raise ValueError(f"Conflicting contrast aliases: {current} / {legacy}")
        if current not in result and legacy in result:
            result[current] = result[legacy]
        if current in result:
            result[legacy] = result[current]
    return result


def contrast_result_keys(result):
    """Validate either NPZ spelling without changing any stored arrays."""
    keys = set(result.files)
    current = "dark_contrast_area_fraction_field"
    legacy = "beta_phase_fraction_field"
    if current in keys and legacy in keys and not np.array_equal(
        result[current], result[legacy], equal_nan=True
    ):
        raise ValueError("Conflicting dark-contrast and legacy beta field arrays")
    if current in keys or legacy in keys:
        keys.add(current)
    return keys


def contrast_terminology_metadata():
    return {
        "version": TERMINOLOGY_VERSION,
        "fingerprint_fields": FINGERPRINT_FIELDS,
        "fingerprint_labels": FINGERPRINT_LABELS,
        "fraction_unit": "0_to_1",
        "bright_contrast_definition": "1 - dark_contrast_area_fraction",
        "phase_identification": "Intensity contrast only; no crystallographic phase identification",
        "summary_fraction_definition": "mean of overlapping-patch reconstructed fraction field",
        "segmentation_plot_fraction_definition": "direct whole-source binary-mask mean",
        "legacy_summary_aliases": CONTRAST_SUMMARY_ALIASES,
        "legacy_npz_aliases": {"beta_phase_fraction_field": "dark_contrast_area_fraction_field"},
        "unchanged_calibration_signature": "Legacy beta_* config keys retained for cache identity",
    }


def result_has_current_fields(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as result:
            if not CURRENT_RESULT_KEYS.issubset(contrast_result_keys(result)):
                return False
            if "terminology_version" not in result.files or str(result["terminology_version"].item()) != TERMINOLOGY_VERSION:
                return False  # Re-render compatible patch cache; do not change inference version.
            cached_version = str(
                np.asarray(result["width_calibration_version"]).reshape(-1)[0]
            )
            if cached_version != WIDTH_CALIBRATION_VERSION:
                return False
            cached_program = str(np.asarray(result["program_version"]).reshape(-1)[0])
            if cached_program != PROGRAM_VERSION:
                return False
            if str(result["spatial_config_json"].item()) != spatial_config_json():
                return False
            prob_field = np.asarray(result["prob_field"])
            return prob_field.ndim == 3 and prob_field.shape[-1] == len(CLASS_NAMES)
    except (OSError, ValueError, KeyError):
        return False


def _load_fingerprint_statistics_only():
    fp_stats_file = ensure_file_exists(FP_STATS_PATH, "Fingerprint stats file")
    fp_mean, fp_std = np.load(fp_stats_file)
    fp_mean = np.asarray(fp_mean, dtype=np.float32).reshape(-1)
    fp_std = np.asarray(fp_std, dtype=np.float32).reshape(-1)
    if fp_mean.shape != (FP_DIM,) or fp_std.shape != (FP_DIM,):
        raise ValueError(
            f"Fingerprint statistics must both have shape {(FP_DIM,)}; "
            f"received mean={fp_mean.shape}, std={fp_std.shape}."
        )
    return fp_mean, fp_std


def refresh_cached_reports(input_path, output_root):
    """Rebuild plots/JSON from compatible NPZ arrays without neural inference."""
    class_means_path = os.path.join(CHECKPOINT_DIR, "test_results", "class_mean_fingerprints.npy")
    class_means = np.asarray(np.load(class_means_path), dtype=np.float32) if os.path.isfile(class_means_path) else None
    if class_means is not None and class_means.shape != (len(CLASS_NAMES), FP_DIM):
        class_means = None

    module = PhysicsAwareSpatialInferenceModule.__new__(PhysicsAwareSpatialInferenceModule)
    module.patch_size = PATCH_SIZE
    module.stride = STRIDE
    module.sigma = SIGMA
    module.local_window = module._make_local_gaussian_window()
    module.fp_mean = module.fp_std = None
    module.class_means = class_means
    module.connectivity_head = None
    module.pair_calibrator = None
    module.pair_calibrator_source = None

    image_paths = resolve_input_images(input_path)
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    if not image_paths:
        raise FileNotFoundError(f"No input images found under: {input_path}")

    refreshed = 0
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(output_root, safe_name(stem), "PSIM_results_1.npz")
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f"Cached result not found for {image_path}: {result_path}")
        with np.load(result_path, allow_pickle=False) as cached:
            if "fp_mean" in cached.files and "fp_std" in cached.files:
                module.fp_mean = np.asarray(cached["fp_mean"], dtype=np.float32).reshape(-1)
                module.fp_std = np.asarray(cached["fp_std"], dtype=np.float32).reshape(-1)
            else:
                module.fp_mean, module.fp_std = _load_fingerprint_statistics_only()
            if module.fp_mean.shape != (FP_DIM,) or module.fp_std.shape != (FP_DIM,):
                raise ValueError("Cached fingerprint statistics must be two-dimensional.")
        summary_path = os.path.join(os.path.dirname(result_path), "PSIM_summary.json")
        module.cached_connectivity_head_loaded = False
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as stream:
                previous_summary = json.load(stream)
            module.cached_connectivity_head_loaded = bool(
                previous_summary.get("connectivity_analysis", {}).get("auxiliary_head_loaded", False)
            )
        print(f"\n=== Refreshing cached reports only: {image_path} ===")
        module.rerender_cached_result(image_path, result_path)
        refreshed += 1
    print(f"Refreshed {refreshed} cached image result(s); neural inference was not run.")


def normalize_map(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min + eps)


def basket_discontinuous_conditional(prob_field, min_pair_mass=WIDTH_PAIR_MIN_MASS):
    """Return P(Discontinuous | Basket_weave or Discontinuous).

    The four-class probability is useful for global classification, but it is
    a poor visual diagnostic for this specific confusion pair: probability
    assigned to either Widmanstatten class suppresses both pair members.  We
    expose the conditional probability while retaining the original field.
    Areas with almost no pair evidence fall back to the absolute
    Discontinuous probability instead of displaying an unstable ratio.
    """
    field = np.asarray(prob_field, dtype=np.float32)
    if field.ndim != 3 or field.shape[-1] < 2:
        raise ValueError("prob_field must have shape (height, width, n_classes)")
    pair_mass = np.clip(field[..., 0] + field[..., 1], 0.0, 1.0)
    conditional = field[..., 1] / np.maximum(pair_mass, 1e-6)
    return np.where(pair_mass >= float(min_pair_mass), conditional, field[..., 1]).astype(np.float32)


def spatial_config_json():
    """Invalidate cached area outputs when crop, vote or boundary rules change."""
    return json.dumps({
        "source_patch_size": PATCH_SIZE, "model_input_size": MODEL_INPUT_SIZE,
        "stride": STRIDE, "sigma": SIGMA, "area_mode": AREA_MODE,
        "postprocess": CLASS_POSTPROCESS_ENABLED,
        "area_filter": CLASS_AREA_FILTER_ENABLED,
        "area_filter_policy": "global-presence-local-components-separate-v3",
        "component_ratio": MIN_CLASS_REGION_AREA_RATIO,
        "presence_ratio": MIN_CLASS_PRESENCE_RATIO,
        "component_px": MIN_CLASS_REGION_AREA_PX,
        "boundary_enabled": BOUNDARY_REFINEMENT_ENABLED,
        "boundary_radius": BOUNDARY_BAND_RADIUS_PX,
        "boundary_smoothing": BOUNDARY_SMOOTH_KERNEL_PX,
        "boundary_margin": BOUNDARY_MAX_VOTE_MARGIN,
        "boundary_pairs": BOUNDARY_REFINEMENT_CLASS_PAIRS,
        "widman_guided_enabled": WIDMAN_GUIDED_ENABLED,
        "widman_guided_radius": WIDMAN_GUIDED_RADIUS_PX,
        "widman_guided_band": WIDMAN_GUIDED_BAND_PX,
        "widman_guided_eps": WIDMAN_GUIDED_EPS,
        "widman_guided_max_delta": WIDMAN_GUIDED_MAX_DELTA,
        "widman_guided_min_mass": WIDMAN_GUIDED_MIN_MASS,
        "microns_per_source_pixel": MICRONS_PER_SOURCE_PIXEL,
    }, sort_keys=True)


def cached_spatial_fields_are_compatible(cached_config):
    """Postprocessing changes can reuse the same probability and vote fields."""
    previous = json.loads(cached_config)
    current = json.loads(spatial_config_json())
    reporting_keys = {
        "postprocess", "area_filter", "area_filter_policy", "component_ratio",
        "presence_ratio", "component_px", "boundary_enabled", "boundary_radius",
        "boundary_smoothing", "boundary_margin", "boundary_pairs",
        "widman_guided_enabled", "widman_guided_radius", "widman_guided_band",
        "widman_guided_eps", "widman_guided_max_delta", "widman_guided_min_mass",
    }
    previous = {key: value for key, value in previous.items() if key not in reporting_keys}
    current = {key: value for key, value in current.items() if key not in reporting_keys}
    return previous == current


def class_transition_band(class_id_map, radius, allowed_pairs=None):
    labels = np.asarray(class_id_map)
    edge = np.zeros(labels.shape, dtype=bool)
    def allowed(a, b):
        if allowed_pairs is None:
            return a != b
        changes = np.zeros(a.shape, dtype=bool)
        for first, second in allowed_pairs:
            changes |= ((a == first) & (b == second)) | ((a == second) & (b == first))
        return changes
    changes = allowed(labels[1:], labels[:-1])
    edge[1:] |= changes
    edge[:-1] |= changes
    changes = allowed(labels[:, 1:], labels[:, :-1])
    edge[:, 1:] |= changes
    edge[:, :-1] |= changes
    if radius <= 0 or not edge.any():
        return np.zeros(labels.shape, dtype=bool)
    return distance_transform_edt(~edge) <= radius


def refine_widmanstatten_fields(image_np, prob_field, class_vote_field):
    """Guide only ambiguous W/coarsened-W transitions; preserve other classes.

    Uses grayscale guided filtering with a tapered, bounded update. Probability
    and vote fields retain their respective meanings. No new class, width
    threshold, or global probability sharpening is introduced.
    """
    stats = {"enabled": bool(WIDMAN_GUIDED_ENABLED), "changed_pixel_count": 0,
             "changed_hard_pixel_count": 0, "max_absolute_probability_change": 0.0,
             "max_conditional_change": float(WIDMAN_GUIDED_MAX_DELTA),
             "definition": "bounded image-guided spatial estimate; not calibrated pixel probability"}
    if not WIDMAN_GUIDED_ENABLED:
        return prob_field, class_vote_field, stats
    if (prob_field.ndim != 3 or prob_field.shape[-1] != len(CLASS_NAMES)
            or class_vote_field.shape != prob_field.shape):
        raise ValueError("Guided refinement requires matching four-class probability/vote fields.")
    if image_np.shape[:2] != prob_field.shape[:2]:
        raise ValueError("Image and probability field sizes differ.")
    w, c = CLASS_NAMES.index("Widmanstatten"), CLASS_NAMES.index("Alpha_coarsened_Widmanstatten")
    probability_winner = prob_field.argmax(axis=-1)
    vote_winner = class_vote_field.argmax(axis=-1)
    score_map = probability_winner if AREA_MODE == "probability" else vote_winner
    band = class_transition_band(score_map, WIDMAN_GUIDED_BAND_PX, ((w, c),))
    eligible = band & np.isin(probability_winner, [w, c]) & np.isin(vote_winner, [w, c])
    mass = prob_field[..., w] + prob_field[..., c]
    eligible &= mass >= WIDMAN_GUIDED_MIN_MASS
    if not np.any(eligible):
        return prob_field, class_vote_field, stats
    guide = np.asarray(image_np, dtype=np.uint8)
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32) / 255.0
    # Suppress individual dark lath edges before using the image as guidance.
    guide = cv2.GaussianBlur(guide, (5, 5), 0)
    size = 2 * int(WIDMAN_GUIDED_RADIUS_PX) + 1
    mean_i = uniform_filter(guide, size=size, mode="reflect")
    variance = np.maximum(uniform_filter(guide * guide, size=size, mode="reflect") - mean_i * mean_i, 0)
    eligible &= variance > 1e-6
    if not np.any(eligible):
        return prob_field, class_vote_field, stats
    outputs = []
    changed = np.zeros(band.shape, dtype=bool)
    for field in (prob_field, class_vote_field):
        pair_mass = field[..., w] + field[..., c]
        conditional = field[..., c] / np.maximum(pair_mass, 1e-8)
        mean_p = uniform_filter(conditional, size=size, mode="reflect")
        cov = uniform_filter(guide * conditional, size=size, mode="reflect") - mean_i * mean_p
        a = cov / (variance + WIDMAN_GUIDED_EPS)
        b = mean_p - a * mean_i
        filtered = uniform_filter(a, size=size, mode="reflect") * guide + uniform_filter(b, size=size, mode="reflect")
        # Protect high-confidence interiors and taper rather than cut sharply
        # at a confidence threshold. Only the two pair channels may change.
        taper = np.clip((0.8 - np.abs(2 * conditional - 1)) / 0.8, 0, 1)
        delta = np.clip(filtered - conditional, -WIDMAN_GUIDED_MAX_DELTA, WIDMAN_GUIDED_MAX_DELTA) * taper
        updated_c = pair_mass * np.clip(conditional + delta, 0, 1)
        updated_w = pair_mass - updated_c
        other_max = np.maximum(field[..., 0], field[..., 1])
        active = eligible & (taper > 0) & (np.maximum(updated_w, updated_c) > other_max)
        output = field.copy()
        output[..., w][active] = updated_w[active]
        output[..., c][active] = updated_c[active]
        changed |= np.any(output != field, axis=-1)
        outputs.append(output)
    stats["changed_pixel_count"] = int(changed.sum())
    stats["changed_area_fraction"] = float(changed.mean())
    stats["max_absolute_probability_change"] = float(np.max(np.abs(outputs[0][..., c] - prob_field[..., c])))
    stats["changed_hard_pixel_count"] = int(np.count_nonzero(
        (outputs[0] if AREA_MODE == "probability" else outputs[1]).argmax(axis=-1) != score_map))
    return outputs[0], outputs[1], stats


def refine_class_boundaries(image_np, class_id_map, band_radius=None, class_scores=None):
    """Snap raw patch-class boundaries toward strong tissue edges.

    The classifier supplies the regional labels. Watershed is used only within
    a narrow class-transition band, so it cannot rewrite whole regions.
    """
    raw_map = np.asarray(class_id_map, dtype=np.int32)
    band_radius = BOUNDARY_BAND_RADIUS_PX if band_radius is None else band_radius
    if raw_map.ndim != 2 or raw_map.size == 0:
        return raw_map.copy()
    if not BOUNDARY_REFINEMENT_ENABLED or band_radius <= 0:
        return raw_map.copy()

    image = np.asarray(image_np)
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    if gray.shape != raw_map.shape:
        gray = cv2.resize(gray, (raw_map.shape[1], raw_map.shape[0]), interpolation=cv2.INTER_AREA)

    kernel_size = max(3, int(BOUNDARY_SMOOTH_KERNEL_PX))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize_map(cv2.magnitude(gx, gy))

    boundary = class_transition_band(raw_map, band_radius, BOUNDARY_REFINEMENT_CLASS_PAIRS)
    pair_ids = {value for pair in BOUNDARY_REFINEMENT_CLASS_PAIRS for value in pair}
    boundary &= np.isin(raw_map, list(pair_ids))
    if class_scores is not None:
        ordered = np.sort(class_scores, axis=-1)
        margin = ordered[..., -1] - ordered[..., -2]
        boundary &= margin <= BOUNDARY_MAX_VOTE_MARGIN
    if not boundary.any() or float(gradient.max()) == 0:
        return raw_map.copy()

    markers = np.zeros(raw_map.shape, dtype=np.int32)
    markers[~boundary] = raw_map[~boundary] + 1
    # Preserve EVERY thin/isolated component without an interior seed, even
    # when another component of the same class already has seeds elsewhere.
    for class_id in np.unique(raw_map):
        n, components = cv2.connectedComponents((raw_map == class_id).astype(np.uint8), connectivity=8)
        seeded = np.unique(components[~boundary])
        protect = np.ones(n, dtype=bool)
        protect[seeded] = False
        protect[0] = False
        protected = protect[components]
        markers[protected] = class_id + 1
        boundary[protected] = False

    markers = watershed(gradient, markers=markers, connectivity=1)
    refined = raw_map.copy()
    valid = np.zeros(raw_map.shape, dtype=bool)
    for first, second in BOUNDARY_REFINEMENT_CLASS_PAIRS:
        valid |= (((raw_map == first) & (markers == second + 1))
                  | ((raw_map == second) & (markers == first + 1)))
    valid &= boundary
    if class_scores is not None:
        candidate = np.maximum(markers - 1, 0)
        supported = np.take_along_axis(class_scores, candidate[..., None], axis=-1)[..., 0]
        valid &= supported > 0
    refined[valid] = markers[valid] - 1
    return refined


def safe_savefig(fig, path, dpi=220):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)


def save_rgb_array(path, arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def blend_overlay(base_rgb, overlay_rgb, alpha=0.4):
    base = base_rgb.astype(np.float32)
    overlay = overlay_rgb.astype(np.float32)
    out = base * (1.0 - alpha) + overlay * alpha
    return np.clip(out, 0, 255).astype(np.uint8)



def get_pil_font(size):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_label_with_background(draw, xy, text, font, fill, bg_fill=(255, 255, 255), pad=6):
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    except Exception:
        w = len(text) * max(8, font.size // 2 if hasattr(font, "size") else 8)
        h = max(16, font.size if hasattr(font, "size") else 16)
        rect = [x - pad, y - pad, x + w + pad, y + h + pad]
    draw.rectangle(rect, fill=bg_fill)
    draw.text((x, y), text, fill=fill, font=font)

def draw_rectangles_on_image(image_np, roi_records, color_map, line_width=6):
    img = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    label_font = get_pil_font(PIL_LABEL_FONT_SIZE)

    for rec in roi_records:
        if not rec.get("present", True):
            continue
        if rec["x"] is None or rec["y"] is None:
            continue

        x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]
        color = color_map[rec["name"]]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
        label_y = max(y + line_width + 8, y + 12)
        draw_label_with_background(
            draw,
            (x + line_width + 10, label_y),
            rec["name"],
            label_font,
            fill=color,
            bg_fill=(255, 255, 255),
            pad=6,
        )

    return np.array(img)


def compute_global_prototype_weights(fingerprints, class_means, lambda_fp=2.0):
    if class_means is None:
        return np.ones(len(fingerprints), dtype=np.float32)

    sim = fingerprints @ class_means.T
    norm_fp = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    norm_c = np.linalg.norm(class_means, axis=1, keepdims=True).T
    cos_sim = sim / (norm_fp * norm_c + 1e-8)
    max_sim = np.max(cos_sim, axis=1)
    return np.exp(-lambda_fp * (1 - max_sim)).astype(np.float32)


def compute_patch_fingerprints_batch(patch_images, fp_mean, fp_std):
    fingerprints = []
    feature_rows = []
    valid_rows = []
    for img in tqdm(patch_images, desc="Computing patch fingerprints", leave=False):
        img_np = np.array(img.convert("RGB"))
        _, fp_raw, features, valid = _compose_morphology_field(
            img_np, mask_size=(128, 128), return_analysis=True
        )
        fp_norm = (fp_raw - fp_mean) / (fp_std + FP_EPS)
        fingerprints.append(fp_norm)
        feature_rows.append(features)
        valid_rows.append(valid)
    return (
        np.array(fingerprints, dtype=np.float32),
        np.array(feature_rows, dtype=np.float32),
        np.array(valid_rows, dtype=bool),
    )


@torch.no_grad()
def encode_class_prompts(tokenizer, text_encoder, class_ids):
    prompts = [PROMPT_DICT.get(int(class_id), f"class {int(class_id)} microstructure")
               for class_id in class_ids.detach().cpu().tolist()]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(DEVICE)
    return text_encoder(input_ids)[0]


@torch.no_grad()
def build_train_condition(tokenizer, text_encoder, ace, fingerprint, class_ids):
    """Reproduce the training condition: text prompt plus one ACE token."""
    text_hidden = encode_class_prompts(tokenizer, text_encoder, class_ids)
    fp_token = ace(fingerprint, class_ids)
    fp_token = F.normalize(fp_token, dim=-1)
    encoder_hidden_states = text_hidden.clone()
    encoder_hidden_states[:, 0, :] = fp_token
    return encoder_hidden_states, fp_token


@torch.no_grad()
def build_class_prompt_cache(tokenizer, text_encoder):
    class_ids = torch.arange(len(CLASS_NAMES), device=DEVICE, dtype=torch.long)
    return encode_class_prompts(tokenizer, text_encoder, class_ids).detach()


def load_models():
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    global NUM_TIMESTEPS
    print("Loading models for PSIM...")

    unet_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "unet_final.pth"), "UNet checkpoint")
    ace_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "ace_final.pth"), "ACE checkpoint")
    mta_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "mta_final.pth"), "MTA checkpoint")
    dit_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "dit_final.pth"), "DiT checkpoint")
    pqa_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "pqa_final.pth"), "PQA checkpoint")
    unified_ckpt = ensure_file_exists(os.path.join(CHECKPOINT_DIR, "unified_head_final.pth"), "Unified head checkpoint")
    connectivity_ckpt = os.path.join(CHECKPOINT_DIR, "connectivity_head_final.pth")
    fp_stats_file = ensure_file_exists(FP_STATS_PATH, "Fingerprint stats file")
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="tokenizer",
        **hf_kwargs(),
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="text_encoder",
        **hf_kwargs(),
    ).to(DEVICE)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    class_prompt_cache = build_class_prompt_cache(tokenizer, text_encoder)

    noise_scheduler = DDPMScheduler.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="scheduler",
        **hf_kwargs(),
    )
    total_train_timesteps = noise_scheduler.config.num_train_timesteps

    vae = AutoencoderKL.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="vae",
        **hf_kwargs(),
    ).to(DEVICE)
    vae.eval()
    vae.requires_grad_(False)

    unet_orig = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL,
        subfolder="unet",
        **hf_kwargs(),
    )
    inject_lora_into_unet(unet_orig, rank=8, alpha=8.0, dropout=0.0)
    unet_wrapper = UNetWithMidFeature(unet_orig).to(DEVICE)
    unet_wrapper.load_state_dict(torch.load(unet_ckpt, map_location=DEVICE), strict=False)
    unet_wrapper.eval()

    ace = TokenizedACE(embed_dim=768, n_classes=len(CLASS_NAMES), fp_dim=2).to(DEVICE)
    ace.load_state_dict(torch.load(ace_ckpt, map_location=DEVICE))
    ace.eval()

    mta_state = torch.load(mta_ckpt, map_location="cpu")
    checkpoint_t_weights = mta_state.get("t_weights")
    if checkpoint_t_weights is not None:
        checkpoint_num_timesteps = int(checkpoint_t_weights.numel())
        if checkpoint_num_timesteps < 1:
            raise ValueError("MTA checkpoint contains an empty t_weights parameter.")
        if checkpoint_num_timesteps != NUM_TIMESTEPS:
            print(
                f"Adjusting NUM_TIMESTEPS from {NUM_TIMESTEPS} to "
                f"{checkpoint_num_timesteps} to match mta_final.pth."
            )
        NUM_TIMESTEPS = checkpoint_num_timesteps

    mta = MultiTimestepAggregator(
        feat_dim=1280,
        fusion_dim=256,
        max_timesteps=NUM_TIMESTEPS,
    ).to(DEVICE)
    mta.load_state_dict(mta_state)
    mta.eval()

    dit = MidBlockTransformer(channels=1280).to(DEVICE)
    dit.load_state_dict(torch.load(dit_ckpt, map_location=DEVICE))
    dit.eval()

    pqa = PhysicalQueryAttention(feat_channels=1280).to(DEVICE)
    pqa.load_state_dict(torch.load(pqa_ckpt, map_location=DEVICE))
    pqa.eval()

    unified_head = UnifiedHeadNoSeg_v2(
        in_channels=1280,
        n_classes=len(CLASS_NAMES),
        hidden_dim=256,
    ).to(DEVICE)
    unified_head.load_state_dict(torch.load(unified_ckpt, map_location=DEVICE))
    unified_head.eval()

    connectivity_head = None
    if os.path.exists(connectivity_ckpt):
        connectivity_state = torch.load(connectivity_ckpt, map_location="cpu")
        expected_shapes = {
            "net.0.weight": (128, 1280, 1, 1),
            "net.7.weight": (CONNECTIVITY_DIM, 128),
            "net.7.bias": (CONNECTIVITY_DIM,),
        }
        shape_errors = []
        for key, expected_shape in expected_shapes.items():
            actual = connectivity_state.get(key)
            actual_shape = tuple(actual.shape) if actual is not None else None
            if actual_shape != expected_shape:
                shape_errors.append(
                    f"{key}: checkpoint={actual_shape}, expected={expected_shape}"
                )

        if shape_errors:
            print(
                "Connectivity head checkpoint is incompatible with the current "
                "large-image model; correction disabled. "
                + "; ".join(shape_errors)
            )
        else:
            candidate_head = ConnectivityHead(
                in_channels=1280,
                hidden_dim=128,
                out_dim=CONNECTIVITY_DIM,
            ).to(DEVICE)
            candidate_head.load_state_dict(connectivity_state)
            candidate_head.eval()
            connectivity_head = candidate_head
            print("Loaded three-descriptor structure head for diagnostics only.")
    else:
        print("Connectivity auxiliary head not found; pairwise connectivity correction disabled.")

    fp_mean, fp_std = np.load(fp_stats_file)
    fp_mean = np.asarray(fp_mean, dtype=np.float32).reshape(-1)
    fp_std = np.asarray(fp_std, dtype=np.float32).reshape(-1)
    if fp_mean.shape != (FP_DIM,) or fp_std.shape != (FP_DIM,):
        raise ValueError(
            "Fingerprint statistics are incompatible with the four-class 2-D model: "
            f"mean={fp_mean.shape}, std={fp_std.shape}, expected {(FP_DIM,)}."
        )

    class_means_path = os.path.join(CHECKPOINT_DIR, "test_results", "class_mean_fingerprints.npy")
    if os.path.exists(class_means_path):
        class_means = np.asarray(np.load(class_means_path), dtype=np.float32)
        if class_means.shape != (len(CLASS_NAMES), FP_DIM):
            print(
                "Warning: class_mean_fingerprints.npy has shape "
                f"{class_means.shape}; expected {(len(CLASS_NAMES), FP_DIM)}. "
                "Prototype weighting disabled."
            )
            class_means = None
        else:
            print("Loaded four-class class_mean_fingerprints.npy")
    else:
        class_means = None
        print("Warning: class_mean_fingerprints.npy not found. Prototype weighting disabled.")

    width_midpoint_z, width_half_gap_z, width_basket_direction = derive_width_calibration(class_means)
    print("Legacy width/connectivity corrections disabled; learned pair calibration is opt-in.")

    print("All models loaded.")
    return (
        class_prompt_cache,
        vae,
        unet_wrapper,
        ace,
        mta,
        dit,
        pqa,
        unified_head,
        connectivity_head,
        noise_scheduler,
        total_train_timesteps,
        fp_mean,
        fp_std,
        class_means,
    )


def preprocess_patches(patches):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.CenterCrop(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return torch.stack([transform(p.convert("RGB")) for p in patches], dim=0).to(DEVICE)


@torch.no_grad()
def infer_patches_batch(
    class_prompt_cache,
    vae,
    unet_wrapper,
    ace,
    mta,
    dit,
    pqa,
    unified_head,
    connectivity_head,
    batch_tensors,
    batch_fingerprints,
    noise_scheduler,
    total_timesteps,
    class_means=None,
    open_fine_structure=None,
):
    batch_size = batch_tensors.shape[0]
    num_timesteps = NUM_TIMESTEPS
    latents = vae.encode(batch_tensors).latent_dist.mode() * vae.config.scaling_factor
    fp_tensor = torch.from_numpy(batch_fingerprints).to(DEVICE, dtype=torch.float32)
    timestep_rows, noise_rows = [], []
    # Key the stochastic feature extractor by crop content, not by processing
    # order. The same crop then gets the same noise in calibration and inference.
    for index in range(batch_size):
        crop_bytes = batch_tensors[index].detach().cpu().numpy().tobytes()
        crop_seed = int.from_bytes(hashlib.sha256(crop_bytes).digest()[:8], "little")
        generator = torch.Generator(device=DEVICE).manual_seed((crop_seed ^ INFERENCE_SEED) % (2**63 - 1))
        timestep_rows.append(sample_multi_timesteps(1, total_timesteps, num_timesteps, DEVICE, generator))
        noise_rows.append(torch.randn((num_timesteps, *latents.shape[1:]),
                                      dtype=latents.dtype, device=latents.device, generator=generator))
    timesteps_matrix = torch.cat(timestep_rows, dim=0)

    # These tensors do not depend on the candidate class.  Reuse them across
    # the three counterfactual conditions instead of rebuilding them three times.
    latents_exp = latents.repeat_interleave(num_timesteps, dim=0)
    t_flat = timesteps_matrix.reshape(-1)
    noise = torch.cat(noise_rows, dim=0)
    noisy_latent = noise_scheduler.add_noise(latents_exp, noise, t_flat)

    zero_condition = torch.zeros(
        batch_size,
        class_prompt_cache.shape[1],
        class_prompt_cache.shape[2],
        device=DEVICE,
        dtype=class_prompt_cache.dtype,
    )
    zero_condition_expanded = zero_condition.repeat_interleave(num_timesteps, dim=0)
    _, unconditioned_mid_batch = unet_wrapper(noisy_latent, t_flat, zero_condition_expanded)
    unconditioned_mid_list = split_repeated_features_by_timestep(
        unconditioned_mid_batch,
        batch_size,
        num_timesteps,
    )
    class_features = (
        mta(unconditioned_mid_list, timesteps_matrix)
        if len(unconditioned_mid_list) > 1
        else unconditioned_mid_list[0]
    )
    class_features = dit(class_features.float().clamp(-50, 50))
    class_logits, _, _, _ = unified_head(class_features)
    probs = F.softmax(class_logits, dim=1)
    connectivity_descriptors = (
        connectivity_head(class_features).detach().cpu().numpy()
        if connectivity_head is not None
        else np.zeros((batch_size, CONNECTIVITY_DIM), dtype=np.float32)
    )
    # Classification stays raw here. The validated pair model is applied by
    # inference_all_patches; no legacy heuristic can silently change logits.
    selected_condition = probs.argmax(dim=1)

    fp_token = ace(fp_tensor, selected_condition)
    fp_token = F.normalize(fp_token, dim=-1)
    condition = class_prompt_cache[selected_condition].clone()
    condition[:, 0, :] = fp_token
    condition_expanded = condition.repeat_interleave(num_timesteps, dim=0)
    _, conditioned_mid_batch = unet_wrapper(noisy_latent, t_flat, condition_expanded)
    conditioned_mid_list = split_repeated_features_by_timestep(
        conditioned_mid_batch,
        batch_size,
        num_timesteps,
    )
    fingerprint_features = (
        mta(conditioned_mid_list, timesteps_matrix)
        if len(conditioned_mid_list) > 1
        else conditioned_mid_list[0]
    )
    fingerprint_features = dit(fingerprint_features.float().clamp(-50, 50))
    fingerprint_features = pqa(fingerprint_features, fp_token)
    _, selected_fp, _, _ = unified_head(fingerprint_features)

    return (probs.cpu().numpy(), selected_fp.cpu().numpy(), connectivity_descriptors,
            class_logits.cpu().numpy())


class PhysicsAwareSpatialInferenceModule:
    def __init__(
        self,
        class_prompt_cache,
        vae,
        unet_wrapper,
        ace,
        mta,
        dit,
        pqa,
        unified_head,
        connectivity_head,
        noise_scheduler,
        total_timesteps,
        fp_mean,
        fp_std,
        class_means=None,
        patch_size=None,
        stride=None,
        sigma=None,
        lambda_fp=2.0,
    ):
        self.class_prompt_cache = class_prompt_cache
        self.vae = vae
        self.unet_wrapper = unet_wrapper
        self.ace = ace
        self.mta = mta
        self.dit = dit
        self.pqa = pqa
        self.unified_head = unified_head
        self.connectivity_head = connectivity_head
        self.noise_scheduler = noise_scheduler
        self.total_timesteps = total_timesteps
        self.fp_mean = fp_mean
        self.fp_std = fp_std
        self.class_means = class_means
        self.patch_size = PATCH_SIZE if patch_size is None else int(patch_size)
        self.stride = STRIDE if stride is None else int(stride)
        if not 0 < self.stride <= self.patch_size:
            raise ValueError("Require 0 < stride <= source patch size for full coverage.")
        self.sigma = sigma if sigma is not None else self.patch_size / 8.0
        self.lambda_fp = lambda_fp
        self.local_window = self._make_local_gaussian_window()

    def _make_positions_1d(self, length):
        if length < self.patch_size:
            raise ValueError(f"Image side {length} is smaller than patch_size={self.patch_size}.")
        positions = list(range(0, length - self.patch_size + 1, self.stride))
        last = length - self.patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _make_local_gaussian_window(self):
        yy, xx = np.meshgrid(
            np.arange(self.patch_size),
            np.arange(self.patch_size),
            indexing="ij",
        )
        center = (self.patch_size - 1) / 2.0
        dist2 = (xx - center) ** 2 + (yy - center) ** 2
        return np.exp(-dist2 / (2 * self.sigma ** 2)).astype(np.float32)

    def extract_patches_with_positions(self, image):
        w, h = image.size
        xs = self._make_positions_1d(w)
        ys = self._make_positions_1d(h)

        patches = []
        positions = []
        centers = []
        for y in ys:
            for x in xs:
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append(patch)
                positions.append((x, y))
                centers.append((x + self.patch_size // 2, y + self.patch_size // 2))
        return patches, positions, centers

    @torch.no_grad()
    def inference_all_patches(self, patches):
        real_fps, pair_features, feature_valid = compute_patch_fingerprints_batch(
            patches, self.fp_mean, self.fp_std
        )
        all_probs = []
        all_fingerprints = []
        all_connectivity = []
        all_logits = []

        for i in tqdm(range(0, len(patches), BATCH_SIZE), desc="Inference batches"):
            batch_patches = patches[i:i + BATCH_SIZE]
            batch_tensors = preprocess_patches(batch_patches)
            batch_fps = real_fps[i:i + BATCH_SIZE]
            probs, fps_out, connectivity_out, raw_logits = infer_patches_batch(
                self.class_prompt_cache,
                self.vae,
                self.unet_wrapper,
                self.ace,
                self.mta,
                self.dit,
                self.pqa,
                self.unified_head,
                self.connectivity_head,
                batch_tensors,
                batch_fps,
                self.noise_scheduler,
                self.total_timesteps,
                class_means=self.class_means,
            )
            all_probs.append(probs)
            all_fingerprints.append(fps_out)
            all_connectivity.append(connectivity_out)
            all_logits.append(raw_logits)

        raw_probs = np.vstack(all_probs)
        logits = np.vstack(all_logits)
        calibrated, structure_scores, applied = apply_pair_calibrator(
            raw_probs, logits, pair_features, feature_valid,
            getattr(self, "pair_calibrator", None),
        )
        self.patch_diagnostics = {
            "patch_raw_probs": raw_probs, "patch_raw_logits": logits,
            "patch_pair_features": pair_features,
            "patch_feature_valid": feature_valid,
            "patch_calibration_applied": applied,
        }

        return (
            calibrated,
            np.vstack(all_fingerprints),
            real_fps,
            np.vstack(all_connectivity),
            structure_scores,
        )

    def reconstruct_fields(self, probs, fingerprints, positions, image_size):
        height, width = image_size
        num_classes = len(CLASS_NAMES)

        prob_field = np.zeros((height, width, num_classes), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        for i, (x, y) in enumerate(positions):
            x2 = x + self.patch_size
            y2 = y + self.patch_size

            w_patch = self.local_window
            prob_field[y:y2, x:x2, :] += w_patch[..., None] * probs[i][None, None, :]
            weight_sum[y:y2, x:x2] += w_patch

        if np.any(weight_sum <= 0):
            raise ValueError("Patch positions leave uncovered pixels in the source image.")
        # The edge Gaussian may be below 1e-8. Normalize by actual coverage,
        # so all votes remain normalized even at the image corners.
        prob_field /= weight_sum[..., None]

        prob_field = np.nan_to_num(prob_field, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = np.sum(prob_field, axis=-1, keepdims=True)
        prob_sum = np.maximum(prob_sum, 1e-8)
        prob_field /= prob_sum

        return prob_field

    def reconstruct_class_votes(self, patch_probs, positions, image_size):
        """One class vote per crop; normalize overlaps so pixels count once."""
        labels = np.argmax(patch_probs, axis=1)
        one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[labels]
        return self.reconstruct_fields(one_hot, None, positions, image_size)

    def initial_class_map(self, prob_field, class_vote_field=None):
        if AREA_MODE == "probability":
            return np.argmax(prob_field, axis=-1), prob_field
        if class_vote_field is None or class_vote_field.shape != prob_field.shape:
            raise ValueError("patch-vote area mode requires a matching class_vote_field.")
        winners = np.isclose(class_vote_field, class_vote_field.max(axis=-1, keepdims=True),
                             rtol=0, atol=1e-7)
        # Only exact/numerical ties use the original soft probabilities.
        labels = np.argmax(np.where(winners, prob_field, -np.inf), axis=-1)
        return labels, class_vote_field

    def build_class_id_map(self, prob_field, image_np=None, class_vote_field=None):
        """Create one consistent hard map for statistics and visual outputs."""
        raw_map, decision_scores = self.initial_class_map(prob_field, class_vote_field)
        if CLASS_POSTPROCESS_ENABLED and image_np is not None:
            raw_map = refine_class_boundaries(image_np, raw_map, class_scores=decision_scores)
        if not CLASS_AREA_FILTER_ENABLED:
            present = np.bincount(raw_map.ravel(), minlength=len(CLASS_NAMES)) > 0
            return raw_map, present, 0, {"enabled": False}, {"enabled": False}
        filtered_map, min_region_area_px, region_stats = self._filter_small_class_regions(
            raw_map,
            prob_field,
        )
        qualified_map, retained_mask, presence_stats = self._apply_class_presence_filter(
            filtered_map,
            prob_field,
        )
        return (
            qualified_map,
            retained_mask,
            min_region_area_px,
            region_stats,
            presence_stats,
        )

    def build_overlay_rgb(self, class_id_map):
        rgb_map = np.zeros((class_id_map.shape[0], class_id_map.shape[1], 3), dtype=np.uint8)
        for i, name in enumerate(CLASS_NAMES):
            rgb_map[class_id_map == i] = COLOR_MAP[name]
        return rgb_map

    def _filter_small_class_regions(self, class_id_map, prob_field):
        """Merge isolated class regions below the configured area threshold."""
        filtered_map = np.asarray(class_id_map, dtype=np.int32).copy()
        height, width = filtered_map.shape
        min_area = max(
            int(MIN_CLASS_REGION_AREA_PX),
            int(math.ceil(MIN_CLASS_REGION_AREA_RATIO * height * width)),
        )
        if min_area <= 1:
            return filtered_map, min_area, {"enabled": False}
        region_stats = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = (filtered_map == class_id).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
            removed = 0
            retained = 0
            for label_id in range(1, n_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                component = labels == label_id
                if area >= min_area:
                    retained += 1
                    continue

                dilated = cv2.dilate(
                    component.astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                    iterations=1,
                ).astype(bool)
                ring = dilated & ~component
                neighbor_ids = filtered_map[ring]
                neighbor_ids = neighbor_ids[neighbor_ids != class_id]

                if neighbor_ids.size > 0:
                    candidate_ids, candidate_counts = np.unique(
                        neighbor_ids,
                        return_counts=True,
                    )
                    replacement = int(candidate_ids[np.argmax(candidate_counts)])
                else:
                    component_prob = np.asarray(prob_field[component], dtype=np.float32)
                    mean_prob = component_prob.mean(axis=0)
                    mean_prob[class_id] = -np.inf
                    replacement = int(np.argmax(mean_prob))

                filtered_map[component] = replacement
                removed += 1

            region_stats[class_name] = {
                "raw_region_count": int(max(0, n_labels - 1)),
                "retained_region_count": int(retained),
                "removed_region_count": int(removed),
            }

        return filtered_map, min_area, region_stats

    def _apply_class_presence_filter(self, class_id_map, prob_field):
        """Remove globally negligible classes from the qualified hard map."""
        qualified_map = np.asarray(class_id_map, dtype=np.int32).copy()
        prob_field = np.asarray(prob_field, dtype=np.float32)
        pixel_count = max(1, qualified_map.size)
        counts = np.bincount(
            qualified_map.reshape(-1),
            minlength=len(CLASS_NAMES),
        ).astype(np.float64)
        area_ratio = counts / pixel_count
        retained_ids = np.flatnonzero((counts > 0) & (area_ratio >= MIN_CLASS_PRESENCE_RATIO))

        # Always retain the dominant class so the rule cannot produce an
        # empty candidate set on a nearly uniform image.
        if retained_ids.size == 0:
            retained_ids = np.array([int(np.argmax(counts))], dtype=np.int64)

        retained_mask = np.zeros(len(CLASS_NAMES), dtype=bool)
        retained_mask[retained_ids] = True
        removed_counts = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            if retained_mask[class_id]:
                continue
            class_pixels = qualified_map == class_id
            removed_counts[class_name] = int(class_pixels.sum())
            if not np.any(class_pixels):
                continue
            candidate_prob = prob_field[class_pixels][:, retained_ids]
            qualified_map[class_pixels] = retained_ids[np.argmax(candidate_prob, axis=1)]

        return qualified_map, retained_mask, {
            "enabled": True,
            "minimum_ratio": float(MIN_CLASS_PRESENCE_RATIO),
            "pre_filter_area_fraction": {
                CLASS_NAMES[i]: float(area_ratio[i])
                for i in range(len(CLASS_NAMES))
            },
            "retained_classes": [CLASS_NAMES[i] for i in retained_ids],
            "removed_pixel_count": removed_counts,
        }

    def validate_probability_field(self, prob_field, image_np=None, class_vote_field=None):
        pixel_sum = prob_field.sum(axis=-1)
        # Large mosaics accumulate millions of values; float32 summation can
        # noticeably bias the displayed mean even for a constant field.
        soft_ratio = prob_field.mean(axis=(0, 1), dtype=np.float64)

        raw_class_id_map, decision_scores = self.initial_class_map(prob_field, class_vote_field)
        raw_hard_counts = np.bincount(
            raw_class_id_map.reshape(-1),
            minlength=len(CLASS_NAMES),
        ).astype(np.float64)
        raw_hard_ratio = raw_hard_counts / max(1, raw_class_id_map.size)
        pre_presence_map = (
            refine_class_boundaries(image_np, raw_class_id_map, class_scores=decision_scores)
            if image_np is not None and CLASS_POSTPROCESS_ENABLED else raw_class_id_map
        )
        pre_presence_counts = np.bincount(
            pre_presence_map.reshape(-1),
            minlength=len(CLASS_NAMES),
        ).astype(np.float64)
        pre_presence_ratio = pre_presence_counts / max(1, pre_presence_map.size)
        (
            class_id_map,
            retained_mask,
            min_region_area_px,
            region_stats,
            presence_stats,
        ) = self.build_class_id_map(
            prob_field,
            image_np=image_np,
            class_vote_field=class_vote_field,
        )
        hard_counts = np.bincount(class_id_map.reshape(-1), minlength=len(CLASS_NAMES)).astype(np.float64)
        hard_ratio = hard_counts / max(1, class_id_map.size)

        present_classes = {
            CLASS_NAMES[i]: bool(retained_mask[i])
            for i in range(len(CLASS_NAMES))
        }
        width_midpoint_z, width_half_gap_z, width_basket_direction = derive_width_calibration(
            self.class_means
        )

        diagnostics = {
            "spatial_inference": json.loads(spatial_config_json()),
            "widmanstatten_spatial_refinement": getattr(self, "spatial_refinement_diagnostics", {}),
            "hard_area_definition": "Each source pixel counted once after crop voting and optional boundary refinement",
            "boundary_changed_pixel_count": int(np.count_nonzero(pre_presence_map != raw_class_id_map)),
            "boundary_changed_area_fraction": float(np.mean(pre_presence_map != raw_class_id_map)),
            "area_filter_enabled": bool(CLASS_AREA_FILTER_ENABLED),
            "soft_argmax_area_fraction": {
                name: float(np.mean(prob_field.argmax(axis=-1) == i))
                for i, name in enumerate(CLASS_NAMES)
            },
            "pixel_prob_sum_min": float(np.min(pixel_sum)),
            "pixel_prob_sum_max": float(np.max(pixel_sum)),
            "pixel_prob_sum_mean": float(np.mean(pixel_sum)),
            "soft_ratio_sum": float(np.sum(soft_ratio)),
            "raw_hard_ratio_sum": float(np.sum(raw_hard_ratio)),
            "hard_ratio_sum": float(np.sum(hard_ratio)),
            "raw_hard_global_phase_fraction": {
                CLASS_NAMES[i]: float(raw_hard_ratio[i])
                for i in range(len(CLASS_NAMES))
            },
            "filtered_hard_global_phase_fraction": {
                CLASS_NAMES[i]: float(hard_ratio[i])
                for i in range(len(CLASS_NAMES))
            },
            "pre_presence_filter_hard_global_phase_fraction": {
                CLASS_NAMES[i]: float(pre_presence_ratio[i])
                for i in range(len(CLASS_NAMES))
            },
            "post_boundary_hard_global_phase_fraction": {
                CLASS_NAMES[i]: float(pre_presence_ratio[i])
                for i in range(len(CLASS_NAMES))
            },
            "post_component_filter_hard_global_phase_fraction": presence_stats.get(
                "pre_filter_area_fraction", {
                    CLASS_NAMES[i]: float(pre_presence_ratio[i])
                    for i in range(len(CLASS_NAMES))
                },
            ),
            "present_classes": present_classes,
            "roi_present_threshold": float(ROI_PRESENT_THRESHOLD),
            "minimum_class_presence_ratio": float(MIN_CLASS_PRESENCE_RATIO),
            "boundary_refinement_enabled": bool(BOUNDARY_REFINEMENT_ENABLED and CLASS_POSTPROCESS_ENABLED),
            "boundary_refinement_band_radius_px": int(BOUNDARY_BAND_RADIUS_PX),
            "boundary_refinement_smooth_kernel_px": int(BOUNDARY_SMOOTH_KERNEL_PX),
            "boundary_refinement_class_pairs": [
                [CLASS_NAMES[a], CLASS_NAMES[b]] for a, b in BOUNDARY_REFINEMENT_CLASS_PAIRS
            ],
            "width_aware_enabled": bool(WIDTH_AWARE_ENABLED),
            "width_logit_gain": float(WIDTH_LOGIT_GAIN),
            "width_hard_gate_z": float(WIDTH_HARD_GATE_Z),
            "width_hard_margin": float(WIDTH_HARD_MARGIN),
            "width_pair_min_mass": float(WIDTH_PAIR_MIN_MASS),
            "width_pair_max_top_gap": float(WIDTH_PAIR_MAX_TOP_GAP),
            "width_calibration_version": WIDTH_CALIBRATION_VERSION,
            "width_calibration_midpoint_z": float(width_midpoint_z),
            "width_calibration_half_gap_z": float(width_half_gap_z),
            "width_calibration_basket_direction": float(width_basket_direction),
            "minimum_class_region_area_px": int(min_region_area_px),
            "minimum_class_region_area_ratio": float(MIN_CLASS_REGION_AREA_RATIO),
            "minimum_class_region_area_um2": float(
                min_region_area_px * MICRONS_PER_SOURCE_PIXEL ** 2
            ),
            "class_region_filter": region_stats,
            "class_presence_filter": presence_stats,
        }
        return soft_ratio, hard_ratio, diagnostics

    def _select_best_patch_index(self, patch_probs, target_name):
        target_id = CLASS_NAMES.index(target_name)
        scores = patch_probs[:, target_id].copy()
        pred_ids = np.argmax(patch_probs, axis=1)
        valid = pred_ids == target_id

        if np.any(valid):
            valid_indices = np.where(valid)[0]
            chosen = valid_indices[np.argmax(scores[valid_indices])]
            return int(chosen)

        return int(np.argmax(scores))

    def _select_transition_patch_index(self, patch_probs, patch_real_fps):
        entropy = -np.sum(patch_probs * np.log(patch_probs + 1e-8), axis=1)

        if patch_real_fps.shape[0] > 2:
            grad_like = np.zeros(len(patch_real_fps), dtype=np.float32)
            grad_like[1:-1] = np.linalg.norm(
                patch_real_fps[2:] - patch_real_fps[:-2],
                axis=1,
            )
            score = entropy + 0.3 * normalize_map(grad_like)
        else:
            score = entropy

        return int(np.argmax(score))

    def _make_empty_roi_record(self, name):
        blank = np.full((self.patch_size, self.patch_size, 3), 255, dtype=np.uint8)
        return {
            "name": name,
            "present": False,
            "patch_index": -1,
            "x": None,
            "y": None,
            "w": int(self.patch_size),
            "h": int(self.patch_size),
            "fingerprint": np.zeros(FP_DIM, dtype=np.float32),
            "prob": np.zeros(len(CLASS_NAMES), dtype=np.float32),
            "crop": blank,
        }

    def _build_roi_records(self, image_np, positions, patch_probs, patch_real_fps, hard_ratio):
        records = []

        present_class_flags = {
            CLASS_NAMES[i]: bool(hard_ratio[i] > ROI_PRESENT_THRESHOLD)
            for i in range(len(CLASS_NAMES))
        }

        for name in CLASS_NAMES:
            if not present_class_flags[name]:
                records.append(self._make_empty_roi_record(name))
                continue

            idx = self._select_best_patch_index(patch_probs, name)
            x, y = positions[idx]
            records.append({
                "name": name,
                "present": True,
                "patch_index": idx,
                "x": int(x),
                "y": int(y),
                "w": int(self.patch_size),
                "h": int(self.patch_size),
                "fingerprint": patch_real_fps[idx].astype(np.float32),
                "prob": patch_probs[idx].astype(np.float32),
                "crop": image_np[y:y + self.patch_size, x:x + self.patch_size].copy(),
            })

        num_present_classes = sum(present_class_flags.values())
        if num_present_classes < 2:
            records.append(self._make_empty_roi_record("Transition_boundary"))
        else:
            idx_transition = self._select_transition_patch_index(patch_probs, patch_real_fps)
            x, y = positions[idx_transition]
            records.append({
                "name": "Transition_boundary",
                "present": True,
                "patch_index": idx_transition,
                "x": int(x),
                "y": int(y),
                "w": int(self.patch_size),
                "h": int(self.patch_size),
                "fingerprint": patch_real_fps[idx_transition].astype(np.float32),
                "prob": patch_probs[idx_transition].astype(np.float32),
                "crop": image_np[y:y + self.patch_size, x:x + self.patch_size].copy(),
            })

        return records

    def _build_roi_distribution_records(self, patch_probs, patch_real_fps, hard_ratio):
        records = {}
        pred_ids = np.argmax(patch_probs, axis=1)

        present_class_flags = {
            CLASS_NAMES[i]: bool(hard_ratio[i] > ROI_PRESENT_THRESHOLD)
            for i in range(len(CLASS_NAMES))
        }

        for name in CLASS_NAMES:
            cid = CLASS_NAMES.index(name)
            if not present_class_flags[name]:
                records[name] = np.empty((0, FP_DIM), dtype=np.float32)
                continue

            mask = pred_ids == cid
            selected = patch_real_fps[mask]
            if selected.shape[0] == 0:
                topk = np.argsort(-patch_probs[:, cid])[: min(ROI_TOPK_SAMPLES, len(patch_probs))]
                selected = patch_real_fps[topk]
            elif selected.shape[0] > ROI_TOPK_SAMPLES:
                conf = patch_probs[mask, cid]
                order = np.argsort(-conf)[:ROI_TOPK_SAMPLES]
                selected = selected[order]
            records[name] = selected

        if sum(present_class_flags.values()) < 2:
            records["Transition_boundary"] = np.empty((0, FP_DIM), dtype=np.float32)
        else:
            entropy = -np.sum(patch_probs * np.log(patch_probs + 1e-8), axis=1)
            transition_idx = np.argsort(-entropy)[: min(ROI_TOPK_SAMPLES, len(patch_probs))]
            records["Transition_boundary"] = patch_real_fps[transition_idx]

        return records

    def rerender_cached_result(self, image_path, result_path):
        image_path = os.path.abspath(image_path)
        image_np = np.array(Image.open(image_path).convert("RGB"))
        with np.load(result_path, allow_pickle=False) as cached:
            required_keys = CURRENT_RESULT_KEYS | {"prob_field", "patch_probs", "patch_real_fps", "positions", "class_names"}
            if self.connectivity_head is not None:
                required_keys.add("patch_connectivity")
            missing_keys = required_keys.difference(contrast_result_keys(cached))
            if missing_keys:
                raise KeyError(
                    "Cached PSIM result is missing required arrays: "
                    + ", ".join(sorted(missing_keys))
                )
            cached_version = (
                str(np.asarray(cached["width_calibration_version"]).reshape(-1)[0])
                if "width_calibration_version" in cached.files
                else ""
            )
            if cached_version != WIDTH_CALIBRATION_VERSION:
                raise ValueError(
                    "Cached probability calibration version is "
                    f"{cached_version or 'missing'}, expected {WIDTH_CALIBRATION_VERSION}. "
                    "Full inference is required."
                )
            if not cached_spatial_fields_are_compatible(str(cached["spatial_config_json"].item())):
                raise ValueError("Cached crop/vote/physical-scale settings differ; full inference is required.")
            if list(cached["class_names"].astype(str)) != CLASS_NAMES:
                raise ValueError("Cached class order differs from this program.")
            prob_field = np.asarray(cached["prob_field"], dtype=np.float32)
            class_vote_field = np.asarray(cached["class_vote_field"], dtype=np.float32)
            patch_probs = np.asarray(cached["patch_probs"], dtype=np.float32)
            patch_real_fps = np.asarray(cached["patch_real_fps"], dtype=np.float32)
            patch_structure_scores = np.asarray(cached["patch_structure_scores"], dtype=np.float32)
            self.patch_diagnostics = {
                key: np.array(cached[key]) for key in (
                    "patch_raw_probs", "patch_raw_logits", "patch_pair_features",
                    "patch_feature_valid", "patch_calibration_applied",
                )
            }
            positions_array = np.asarray(cached["positions"], dtype=np.int64)
            if "centers" in cached.files:
                centers_array = np.asarray(cached["centers"], dtype=np.int64)
            else:
                centers_array = positions_array + self.patch_size // 2
            if "patch_connectivity" in cached.files:
                patch_connectivity = np.asarray(cached["patch_connectivity"], dtype=np.float32)
            else:
                patch_connectivity = None

        if prob_field.shape[:2] != image_np.shape[:2]:
            raise ValueError(
                "Cached probability-field size does not match the source image: "
                f"{prob_field.shape[:2]} versus {image_np.shape[:2]}."
            )

        positions = [tuple(map(int, row)) for row in positions_array]
        centers = [tuple(map(int, row)) for row in centers_array]
        # Always start from saved PATCH predictions, never from a field that
        # was already refined in a previous report refresh (no cumulative drift).
        prob_field = self.reconstruct_fields(patch_probs, None, positions, image_np.shape[:2])
        class_vote_field = self.reconstruct_class_votes(patch_probs, positions, image_np.shape[:2])
        prob_field, class_vote_field, self.spatial_refinement_diagnostics = refine_widmanstatten_fields(
            image_np, prob_field, class_vote_field,
        )
        soft_ratio, hard_ratio, diagnostics = self.validate_probability_field(
            prob_field,
            image_np=image_np,
            class_vote_field=class_vote_field,
        )
        output_dir = os.path.dirname(result_path)
        self.save_results(
            image_np=image_np,
            prob_field=prob_field,
            patch_probs=patch_probs,
            patch_real_fps=patch_real_fps,
            positions=positions,
            centers=centers,
            output_dir=output_dir,
            soft_ratio=soft_ratio,
            hard_ratio=hard_ratio,
            diagnostics=diagnostics,
            image_path=image_path,
            patch_connectivity=patch_connectivity,
            patch_structure_scores=patch_structure_scores,
            class_vote_field=class_vote_field,
        )
        print(f"Re-rendered cached four-class PSIM result: {output_dir}")

    def run(self, image_path):
        set_inference_seed()
        image_path = os.path.abspath(image_path)
        img = Image.open(image_path).convert("RGB")
        image_np = np.array(img)
        width, height = img.size

        print(f"Using image: {image_path}")
        print(f"Image size: {width} x {height}")

        patches, positions, centers = self.extract_patches_with_positions(img)
        print(f"Number of patches: {len(patches)}")

        (
            patch_probs,
            patch_fingerprints,
            patch_real_fps,
            patch_connectivity,
            patch_structure_scores,
        ) = self.inference_all_patches(patches)
        prob_field = self.reconstruct_fields(
            patch_probs,
            patch_fingerprints,
            positions,
            (height, width),
        )
        class_vote_field = self.reconstruct_class_votes(patch_probs, positions, (height, width))
        prob_field, class_vote_field, self.spatial_refinement_diagnostics = refine_widmanstatten_fields(
            image_np, prob_field, class_vote_field,
        )

        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(OUTPUT_DIR, safe_name(image_stem))
        os.makedirs(image_output_dir, exist_ok=True)

        soft_ratio, hard_ratio, diagnostics = self.validate_probability_field(
            prob_field,
            image_np=image_np,
            class_vote_field=class_vote_field,
        )
        pairwise_discontinuous = basket_discontinuous_conditional(prob_field)
        diagnostics.update({
            "pairwise_discontinuous_conditional_mean": float(pairwise_discontinuous.mean()),
            "pairwise_discontinuous_conditional_p25": float(np.percentile(pairwise_discontinuous, 25)),
            "pairwise_discontinuous_conditional_p75": float(np.percentile(pairwise_discontinuous, 75)),
            "pairwise_evidence_mean": float((prob_field[..., 0] + prob_field[..., 1]).mean()),
        })

        self.save_results(
            image_np=image_np,
            prob_field=prob_field,
            patch_probs=patch_probs,
            patch_real_fps=patch_real_fps,
            positions=positions,
            centers=centers,
            output_dir=image_output_dir,
            soft_ratio=soft_ratio,
            hard_ratio=hard_ratio,
            diagnostics=diagnostics,
            image_path=image_path,
            patch_connectivity=patch_connectivity,
            patch_structure_scores=patch_structure_scores,
            class_vote_field=class_vote_field,
        )

        print("\n===== Global Phase Fraction (PSIM) =====")
        print("[Soft ratio: mean class probability, NOT measured area]")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name}: {soft_ratio[i] * 100:.2f}%")

        print(f"[Hard area estimate: {AREA_MODE}, one count per source pixel]")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name}: {hard_ratio[i] * 100:.2f}%")

        print(f"Soft ratio sum check: {diagnostics['soft_ratio_sum'] * 100:.2f}%")
        print(f"Hard ratio sum check: {diagnostics['hard_ratio_sum'] * 100:.2f}%")
        print(
            "Per-pixel probability sum min/max: "
            f"{diagnostics['pixel_prob_sum_min']:.6f} / {diagnostics['pixel_prob_sum_max']:.6f}"
        )
        print(f"Present classes by hard ratio: {diagnostics['present_classes']}")
        print(
            "Basket/Discontinuous conditional P(Discontinuous) mean/p25/p75: "
            f"{diagnostics['pairwise_discontinuous_conditional_mean']:.3f} / "
            f"{diagnostics['pairwise_discontinuous_conditional_p25']:.3f} / "
            f"{diagnostics['pairwise_discontinuous_conditional_p75']:.3f}"
        )
        summary_path = os.path.join(image_output_dir, "PSIM_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            quantitative_summary = normalize_contrast_summary(summary_payload.get("quantitative_morphology", {}))
            if quantitative_summary:
                print("[Quantitative morphology]")
                print(
                    "Global Dark-contrast area fraction: "
                    f"{quantitative_summary.get('global_dark_contrast_area_fraction', 0.0) * 100.0:.2f}%"
                )
                print(
                    "Global mean α-lath width: "
                    f"{quantitative_summary.get('global_mean_lath_width_um', 0.0):.3f} um"
                )
                print(
                    "Spatial calibration: "
                    f"{quantitative_summary.get('microns_per_source_pixel', MICRONS_PER_SOURCE_PIXEL):.6f} um/pixel"
                )

        return prob_field

    def save_results(
        self,
        image_np,
        prob_field,
        patch_probs,
        patch_real_fps,
        positions,
        centers,
        output_dir,
        soft_ratio,
        hard_ratio,
        diagnostics,
        image_path,
        patch_connectivity=None,
        patch_structure_scores=None,
        class_vote_field=None,
    ):
        if patch_connectivity is None:
            patch_connectivity = np.zeros(
                (len(positions), CONNECTIVITY_DIM), dtype=np.float32
            )
        patch_connectivity = np.asarray(patch_connectivity, dtype=np.float32)
        if patch_connectivity.shape != (len(positions), CONNECTIVITY_DIM):
            raise ValueError(
                "Patch connectivity descriptor shape does not match patch positions: "
                f"{patch_connectivity.shape} versus {(len(positions), CONNECTIVITY_DIM)}"
            )
        patch_connectivity_score = connectivity_score_from_descriptors(patch_connectivity)
        if patch_structure_scores is None:
            patch_structure_scores = np.zeros(len(positions), dtype=np.float32)
        patch_structure_scores = np.asarray(patch_structure_scores, dtype=np.float32)
        if patch_structure_scores.shape != (len(positions),):
            raise ValueError(
                "Patch structure-score shape does not match patch positions: "
                f"{patch_structure_scores.shape} versus {(len(positions),)}"
            )
        pairwise_discontinuous = basket_discontinuous_conditional(prob_field)
        pair_mass = np.asarray(prob_field[..., 0] + prob_field[..., 1], dtype=np.float32)
        connectivity_score_field = self._reconstruct_scalar_field(
            patch_connectivity_score,
            positions,
            image_np.shape[:2],
        )
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "connectivity_auxiliary_head_loaded": bool(self.connectivity_head is not None or getattr(self, "cached_connectivity_head_loaded", False)),
            "connectivity_score_threshold": float(CONNECTIVITY_THRESHOLD),
            "connectivity_logit_gain": float(CONNECTIVITY_LOGIT_GAIN),
            "connectivity_score_mean": float(patch_connectivity_score.mean()),
            "connectivity_score_p25": float(np.percentile(patch_connectivity_score, 25)),
            "connectivity_score_p75": float(np.percentile(patch_connectivity_score, 75)),
            "pairwise_discontinuous_conditional_mean": float(pairwise_discontinuous.mean()),
            "pairwise_discontinuous_conditional_p25": float(np.percentile(pairwise_discontinuous, 25)),
            "pairwise_discontinuous_conditional_p75": float(np.percentile(pairwise_discontinuous, 75)),
            "pairwise_evidence_mean": float(pair_mass.mean()),
            "pair_model_loaded": getattr(self, "pair_calibrator", None) is not None,
            "pair_model_source": getattr(self, "pair_calibrator_source", None),
            "pair_model_score_mean": (float(patch_structure_scores.mean())
                                      if getattr(self, "pair_calibrator", None) is not None else None),
            "pair_calibration_applied_count": int(self.patch_diagnostics["patch_calibration_applied"].sum()),
            "patch_prediction_changed_count": int(np.count_nonzero(
                patch_probs.argmax(axis=1) != self.patch_diagnostics["patch_raw_probs"].argmax(axis=1))),
            "patch_raw_argmax_fraction": {
                name: float(np.mean(self.patch_diagnostics["patch_raw_probs"].argmax(axis=1) == i))
                for i, name in enumerate(CLASS_NAMES)
            },
            "crop_class_count": {
                name: int(np.sum(patch_probs.argmax(axis=1) == i))
                for i, name in enumerate(CLASS_NAMES)
            },
            "crop_class_count_fraction": {
                name: float(np.mean(patch_probs.argmax(axis=1) == i))
                for i, name in enumerate(CLASS_NAMES)
            },
            "pair_measurement": pair_measurement_config(),
            "soft_ratio_definition": "Mean class probability, not measured area fraction",
            "probability_field_definition": (
                "Gaussian patch probabilities with bounded W/coarsened-W image guidance"
                if WIDMAN_GUIDED_ENABLED else "Gaussian patch probabilities"
            ),
            "unmodified_model_evidence": "patch_raw_probs, patch_raw_logits and patch_probs in NPZ",
        })
        remove_legacy_quantitative_outputs(output_dir)
        class_id_map, _, _, _, _ = self.build_class_id_map(
            prob_field,
            image_np=image_np,
            class_vote_field=class_vote_field,
        )
        # All exported area values must describe the exact saved hard map.
        hard_ratio = np.bincount(class_id_map.ravel(), minlength=len(CLASS_NAMES)).astype(np.float64)
        hard_ratio /= class_id_map.size
        if CLASS_AREA_FILTER_ENABLED and np.any((hard_ratio > 0) & (hard_ratio < MIN_CLASS_PRESENCE_RATIO)):
            raise ValueError("Filtered hard map still contains a class below the minimum presence ratio.")
        soft_ratio = prob_field.mean(axis=(0, 1), dtype=np.float64)
        diagnostics["filtered_hard_global_phase_fraction"] = {
            name: float(hard_ratio[i]) for i, name in enumerate(CLASS_NAMES)
        }
        diagnostics["present_classes"] = {
            name: bool(hard_ratio[i] > 0) for i, name in enumerate(CLASS_NAMES)
        }
        raw_class_id_map, _ = self.initial_class_map(prob_field, class_vote_field)
        overlay_rgb = self.build_overlay_rgb(class_id_map)
        (
            beta_fraction_field,
            lath_width_um_field,
            patch_raw_fingerprints,
            quantitative_summary,
        ) = self._reconstruct_quantitative_fingerprint_fields(
            image_np=image_np,
            patch_real_fps=patch_real_fps,
            positions=positions,
            image_size=image_np.shape[:2],
        )
        class_area_summary = {
            name: float(hard_ratio[index])
            for index, name in enumerate(CLASS_NAMES)
        }

        roi_records = self._build_roi_records(
            image_np=image_np,
            positions=positions,
            patch_probs=patch_probs,
            patch_real_fps=patch_real_fps,
            hard_ratio=hard_ratio,
        )
        roi_distribution = self._build_roi_distribution_records(
            patch_probs=patch_probs,
            patch_real_fps=patch_real_fps,
            hard_ratio=hard_ratio,
        )

        self._save_class_overlay(image_np, overlay_rgb, output_dir)
        self._save_per_class_overlays(image_np, prob_field, output_dir, hard_ratio=hard_ratio)
        self._save_roi_overview(image_np, overlay_rgb, roi_records, output_dir)
        self._save_roi_crops(roi_records, output_dir)
        self._save_roi_grouped_bar(roi_records, output_dir)
        self._save_roi_boxplot(roi_distribution, output_dir)
        self._save_global_ratio_plot(soft_ratio, hard_ratio, output_dir)
        self._save_beta_phase_fraction_statistics(image_np, output_dir)
        self._save_lath_width_statistics(
            patch_raw_fingerprints=patch_raw_fingerprints,
            output_dir=output_dir,
        )
        if self.connectivity_head is not None or getattr(self, "cached_connectivity_head_loaded", False):
            self._save_connectivity_score_map(
                image_np=image_np,
                connectivity_score_field=connectivity_score_field,
                output_dir=output_dir,
            )

        np.savez_compressed(
            os.path.join(output_dir, "PSIM_results_1.npz"),
            prob_field=prob_field,
            class_id_map=class_id_map,
            raw_class_id_map=raw_class_id_map,
            class_vote_field=class_vote_field,
            spatial_config_json=np.array(spatial_config_json()),
            dark_contrast_area_fraction_field=beta_fraction_field,
            bright_contrast_area_fraction_field=1.0 - beta_fraction_field,
            beta_phase_fraction_field=beta_fraction_field,  # Legacy alias, identical values.
            terminology_version=np.array(TERMINOLOGY_VERSION),
            fingerprint_fields=np.array(FINGERPRINT_FIELDS),
            fingerprint_labels=np.array(FINGERPRINT_LABELS),
            mean_lath_width_um_field=lath_width_um_field,
            connectivity_score_field=connectivity_score_field,
            pairwise_discontinuous_conditional=pairwise_discontinuous,
            soft_ratio=soft_ratio,
            hard_ratio=hard_ratio,
            patch_probs=patch_probs,
            patch_real_fps=patch_real_fps,
            patch_raw_fingerprints=patch_raw_fingerprints,
            patch_connectivity=patch_connectivity,
            patch_connectivity_score=patch_connectivity_score,
            patch_structure_scores=patch_structure_scores,
            connectivity_names=np.array(CONNECTIVITY_NAMES),
            positions=np.array(positions),
            centers=np.array(centers),
            class_names=np.array(CLASS_NAMES),
            fp_mean=np.asarray(self.fp_mean),
            fp_std=np.asarray(self.fp_std),
            program_version=np.array(PROGRAM_VERSION),
            width_calibration_version=np.array(WIDTH_CALIBRATION_VERSION),
            pair_feature_names=np.array(PAIR_FEATURE_NAMES),
            **self.patch_diagnostics,
        )
        # A crop audit makes it possible to inspect exact native crops with
        # high Basket scores, instead of inferring their contents from a map.
        with open(os.path.join(output_dir, "PSIM_patch_predictions.csv"), "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "source_size", "model_input_size", "raw_class", "used_class"]
                            + ["raw_" + n for n in CLASS_NAMES] + ["used_" + n for n in CLASS_NAMES])
            for i, (x, y) in enumerate(positions):
                raw_probs = self.patch_diagnostics["patch_raw_probs"][i]
                writer.writerow([x, y, self.patch_size, MODEL_INPUT_SIZE,
                                 CLASS_NAMES[int(raw_probs.argmax())], CLASS_NAMES[int(patch_probs[i].argmax())]]
                                + raw_probs.tolist() + patch_probs[i].tolist())

        roi_summary = []
        for rec in roi_records:
            roi_summary.append({
                "name": rec["name"],
                "present": bool(rec["present"]),
                "patch_index": int(rec["patch_index"]),
                "x": None if rec["x"] is None else int(rec["x"]),
                "y": None if rec["y"] is None else int(rec["y"]),
                "w": int(rec["w"]),
                "h": int(rec["h"]),
                "fingerprint": rec["fingerprint"].tolist(),
                "prob": rec["prob"].tolist(),
            })

        summary = {
            "spatial_inference": json.loads(spatial_config_json()),
            "program_version": PROGRAM_VERSION,
            "program_file": os.path.abspath(__file__),
            "image_path": image_path,
            "soft_global_phase_fraction": {
                CLASS_NAMES[i]: float(soft_ratio[i]) for i in range(len(CLASS_NAMES))
            },
            "hard_global_phase_fraction": {
                CLASS_NAMES[i]: float(hard_ratio[i]) for i in range(len(CLASS_NAMES))
            },
            "crop_class_count_fraction": diagnostics["crop_class_count_fraction"],
            "diagnostics": diagnostics,
            "fingerprint_labels": FINGERPRINT_LABELS,
            "fingerprint_fields": FINGERPRINT_FIELDS,
            "terminology": contrast_terminology_metadata(),
            "roi_records": roi_summary,
            "class_area_fraction": class_area_summary,
            "quantitative_morphology": quantitative_summary,
            "connectivity_analysis": {
                "descriptor_names": CONNECTIVITY_NAMES,
                "auxiliary_head_loaded": bool(self.connectivity_head is not None or getattr(self, "cached_connectivity_head_loaded", False)),
                "score_threshold": float(CONNECTIVITY_THRESHOLD),
                "logit_gain": float(CONNECTIVITY_LOGIT_GAIN),
                "patch_score_mean": float(patch_connectivity_score.mean()),
                "patch_score_p25": float(np.percentile(patch_connectivity_score, 25)),
                "patch_score_p75": float(np.percentile(patch_connectivity_score, 75)),
            },
        }

        with open(os.path.join(output_dir, "PSIM_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "PSIM_class_names.json"), "w", encoding="utf-8") as f:
            json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

        print(f"All PSIM results saved to: {output_dir}")

    def _save_connectivity_score_map(self, image_np, connectivity_score_field, output_dir):
        display_image = image_np
        max_side = 1800
        scale = min(1.0, max_side / max(display_image.shape[:2]))
        if scale < 1.0:
            display_image = cv2.resize(
                display_image,
                (int(display_image.shape[1] * scale), int(display_image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        score_display = cv2.resize(
            connectivity_score_field,
            (display_image.shape[1], display_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=PANEL_DPI)
        ax.imshow(display_image, interpolation="nearest")
        im = ax.imshow(
            score_display,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.62,
            interpolation="nearest",
        )
        ax.contour(
            score_display >= CONNECTIVITY_THRESHOLD,
            levels=[0.5],
            colors=["white"],
            linewidths=0.75,
            alpha=0.9,
        )
        ax.set_title(
            "Connectivity score: Basket_weave versus Discontinuous",
            fontsize=FONT_TITLE,
            pad=10,
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
        cbar.set_label("Connectivity score", fontsize=FONT_AXIS, labelpad=8)
        cbar.ax.tick_params(labelsize=FONT_TICK, width=0.8)
        safe_savefig(
            fig,
            os.path.join(output_dir, "PSIM_connectivity_score_map.png"),
            dpi=PANEL_DPI,
        )

    def _save_class_overlay(self, image_np, overlay_rgb, output_dir):
        blended = blend_overlay(image_np, overlay_rgb, alpha=0.38)
        save_rgb_array(os.path.join(output_dir, "PSIM_organization_overlay.png"), blended)

    def _save_per_class_overlays(self, image_np, prob_field, output_dir, hard_ratio=None):
        """Save one probability map per retained class; remove stale class plots."""
        for filename in os.listdir(output_dir):
            if filename.startswith("PSIM_prob_overlay_") and filename.endswith(".png"):
                os.remove(os.path.join(output_dir, filename))
        display_image = np.asarray(image_np)
        max_side = 2200
        scale = min(1.0, max_side / max(display_image.shape[:2]))
        if scale < 1.0:
            display_image = cv2.resize(
                display_image,
                (int(round(display_image.shape[1] * scale)), int(round(display_image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        for c, name in enumerate(CLASS_NAMES):
            if hard_ratio is not None and hard_ratio[c] <= 0:
                continue
            probability = np.nan_to_num(
                np.asarray(prob_field[:, :, c], dtype=np.float32),
                nan=0.0, posinf=1.0, neginf=0.0,
            )
            probability = np.clip(probability, 0.0, 1.0)
            probability = cv2.resize(
                probability, (display_image.shape[1], display_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            display_name = str(name).replace("_", " ")
            fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=PANEL_DPI)
            ax.imshow(display_image, interpolation="nearest")
            im = ax.imshow(
                probability, cmap="magma", norm=Normalize(vmin=0.0, vmax=1.0, clip=True),
                alpha=0.58, interpolation="nearest",
            )
            ax.set_title(
                f"PSIM {display_name} Overlay\n"
                + ("(four-class spatial probability estimate)"
                   if WIDMAN_GUIDED_ENABLED and c in (2, 3) else "(absolute four-class probability)"),
                fontsize=FONT_TITLE, pad=10,
            )
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
            cbar.set_label(f"P({display_name})", fontsize=FONT_AXIS, labelpad=8)
            cbar.ax.tick_params(labelsize=FONT_TICK, width=0.8)
            safe_savefig(
                fig, os.path.join(output_dir, f"PSIM_prob_overlay_{safe_name(name)}.png"),
                dpi=PANEL_DPI,
            )

    def _save_roi_overview(self, image_np, overlay_rgb, roi_records, output_dir):
        blended = blend_overlay(image_np, overlay_rgb, alpha=0.28)
        valid_records = [rec for rec in roi_records if rec.get("present", True)]
        if len(valid_records) > 0:
            annotated = draw_rectangles_on_image(blended, valid_records, ROI_COLOR_MAP, line_width=6)
        else:
            annotated = blended
        save_rgb_array(os.path.join(output_dir, "PSIM_roi_overview.png"), annotated)

    def _save_roi_crops(self, roi_records, output_dir):
        crops = []
        target_h = 256
        title_font = get_pil_font(PIL_SMALL_FONT_SIZE)
        absent_font = get_pil_font(max(14, PIL_SMALL_FONT_SIZE - 4))

        def format_title(name):
            """Keep long ROI names inside their own tile instead of overlapping."""
            normalized = str(name).replace("_", " ").strip()
            key = normalized.casefold()
            if key.startswith("alpha coarsened"):
                return "Alpha-coarsened\nWidmanstatten"
            if key == "transition boundary":
                return "Transition\nboundary"
            return normalized

        for rec in roi_records:
            if rec["present"]:
                crop = Image.fromarray(rec["crop"].astype(np.uint8))
            else:
                crop = Image.new("RGB", (self.patch_size, self.patch_size), (255, 255, 255))

            w, h = crop.size
            new_w = int(w * (target_h / h))
            crop = crop.resize((new_w, target_h), RESAMPLE_BILINEAR)
            crops.append((rec["name"], crop, rec["present"]))

        if not crops:
            return

        title_blocks = []
        for name, crop, present in crops:
            title = format_title(name)
            title_box = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                (0, 0), title, font=title_font, spacing=2
            )
            absent_box = (0, 0, 0, 0)
            if not present:
                absent_box = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                    (0, 0), "Absent in hard\nphase fraction", font=absent_font, spacing=2
                )
            title_blocks.append((title, title_box, absent_box))

        header_h = max(
            68,
            max(
                (box[3] - box[1]) + (absent[3] - absent[1]) + 14
                for _, box, absent in title_blocks
            ),
        )
        gap = 24
        total_w = sum(img.size[0] for _, img, _ in crops) + gap * (len(crops) - 1)
        canvas = Image.new("RGB", (total_w, target_h + header_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        x = 0
        for (name, crop, present), (title, title_box, absent_box) in zip(crops, title_blocks):
            canvas.paste(crop, (x, header_h))
            title_w = title_box[2] - title_box[0]
            title_x = x + max(0, (crop.size[0] - title_w) // 2)
            draw.multiline_text(
                (title_x, 7), title, fill=(0, 0, 0), font=title_font,
                spacing=2, align="center"
            )
            if not present:
                absent_text = "Absent in hard\nphase fraction"
                absent_w = absent_box[2] - absent_box[0]
                absent_x = x + max(0, (crop.size[0] - absent_w) // 2)
                title_h = title_box[3] - title_box[1]
                draw.multiline_text(
                    (absent_x, 10 + title_h), absent_text, fill=(120, 120, 120),
                    font=absent_font, spacing=2, align="center"
                )
            x += crop.size[0] + gap

        canvas.save(os.path.join(output_dir, "PSIM_roi_crops.png"))

    def _get_present_roi_records(self, roi_records):
        return [rec for rec in roi_records if rec.get("present", True)]

    def _save_roi_grouped_bar(self, roi_records, output_dir):
        present_records = self._get_present_roi_records(roi_records)

        fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=PANEL_DPI)

        if len(present_records) == 0:
            ax.text(
                0.5,
                0.5,
                "No ROI present",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FONT_AXIS,
            )
            ax.axis("off")
            safe_savefig(fig, os.path.join(output_dir, "PSIM_roi_fingerprint_grouped_bar.png"), dpi=PANEL_DPI)
            return

        data = np.stack([rec["fingerprint"] for rec in present_records], axis=0)
        dim_means = data.mean(axis=0, keepdims=True)
        dim_stds = data.std(axis=0, keepdims=True) + 1e-8
        data_norm = (data - dim_means) / dim_stds

        x = np.arange(len(FINGERPRINT_LABELS))
        width = min(0.18, 0.75 / max(1, len(present_records)))

        for i, rec in enumerate(present_records):
            offset = (i - (len(present_records) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                data_norm[i],
                width=width,
                label=rec["name"],
                color=np.array(ROI_COLOR_MAP[rec["name"]]) / 255.0,
                alpha=0.85,
                linewidth=0,
            )

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(FINGERPRINT_LABELS, rotation=18, ha="right", fontsize=FONT_TICK)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_ylabel("Z-scored fingerprint value", fontsize=FONT_AXIS, labelpad=8)
        ax.set_title("ROI Fingerprint Grouped Comparison", fontsize=FONT_TITLE, pad=10)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.2)
        safe_savefig(fig, os.path.join(output_dir, "PSIM_roi_fingerprint_grouped_bar.png"), dpi=PANEL_DPI)

    def _save_roi_boxplot(self, roi_distribution, output_dir):
        fig, axes = plt.subplots(1, FP_DIM, figsize=(12.5, 5.8), dpi=PANEL_DPI)
        axes = np.atleast_1d(axes).flatten()

        group_order = [
            "Basket_weave",
            "Discontinuous",
            "Widmanstatten",
            "Alpha_coarsened_Widmanstatten",
            "Transition_boundary",
        ]

        for d, ax in enumerate(axes):
            values = []
            labels = []
            colors = []
            positions = []

            for pos, name in enumerate(group_order, start=1):
                arr = roi_distribution.get(name, np.empty((0, FP_DIM), dtype=np.float32))
                if arr.shape[0] == 0:
                    ax.text(
                        pos,
                        0.50,
                        "Absent",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="center",
                        rotation=90,
                        color=(0.45, 0.45, 0.45),
                        fontsize=FONT_TICK,
                    )
                    continue

                values.append(arr[:, d])
                labels.append(name)
                colors.append(np.array(ROI_COLOR_MAP[name]) / 255.0)
                positions.append(pos)

            if len(values) > 0:
                bp = ax.boxplot(
                    values,
                    positions=positions,
                    patch_artist=True,
                    labels=labels,
                    showfliers=False,
                    widths=0.55,
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                    patch.set_linewidth(0.9)
                for median in bp["medians"]:
                    median.set_linewidth(1.1)
                    median.set_color("black")
                for key in ["whiskers", "caps"]:
                    for artist in bp[key]:
                        artist.set_linewidth(0.9)

            ax.set_title(FINGERPRINT_LABELS[d], fontsize=FONT_TITLE, pad=8)
            ax.set_xlim(0.5, len(group_order) + 0.5)
            ax.set_xticks(np.arange(1, len(group_order) + 1))
            ax.set_xticklabels(group_order, rotation=18, ha="right", fontsize=FONT_TICK)
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("ROI Fingerprint Distribution Boxplots", fontsize=FONT_TITLE + 1, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94], pad=1.2, w_pad=2.0, h_pad=1.2)
        safe_savefig(fig, os.path.join(output_dir, "PSIM_roi_fingerprint_boxplot.png"), dpi=PANEL_DPI)

    def _save_global_ratio_plot(self, soft_ratio, hard_ratio, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=PANEL_DPI)
        colors = [np.array(COLOR_MAP[name]) / 255.0 for name in CLASS_NAMES]

        for ax, ratio, title in zip(
            axes,
            [soft_ratio, hard_ratio],
            ["Mean Class Probability",
             ("Filtered Area" if CLASS_AREA_FILTER_ENABLED else "Estimated Area")
             + (" (crop votes)" if AREA_MODE == "patch-vote" else " (soft argmax)")],
        ):
            bars = ax.bar(CLASS_NAMES, ratio * 100.0, color=colors, width=0.58, linewidth=0)
            ax.set_ylabel("Mean probability (%)" if title.startswith("Mean Class Probability") else "Area fraction (%)",
                          fontsize=FONT_AXIS, labelpad=8)
            ax.set_title(title, fontsize=FONT_TITLE, pad=10)
            ax.set_ylim(0, max(100.0, float(np.max(ratio * 100.0)) * 1.18))
            ax.tick_params(axis="x", labelsize=FONT_TICK, rotation=12)
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, value in zip(bars, ratio):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.5,
                    f"{value * 100:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_ANNOT,
                )

        fig.tight_layout(w_pad=2.5)
        safe_savefig(fig, os.path.join(output_dir, "PSIM_global_phase_fraction.png"), dpi=PANEL_DPI)

    def _reconstruct_scalar_field(self, scores, positions, image_size, weights=None):
        height, width = image_size
        field = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        if weights is None:
            weights = np.ones(len(scores), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)

        for i, (x, y) in enumerate(positions):
            x2 = x + self.patch_size
            y2 = y + self.patch_size
            w_patch = self.local_window * float(weights[i])
            field[y:y2, x:x2] += w_patch * float(scores[i])
            weight_sum[y:y2, x:x2] += w_patch

        np.divide(field, weight_sum, out=field, where=weight_sum > 0)
        return np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    def _reconstruct_quantitative_fingerprint_fields(
        self,
        image_np,
        patch_real_fps,
        positions,
        image_size,
    ):
        patch_real_fps = np.asarray(patch_real_fps, dtype=np.float32)
        raw_fingerprints = (
            patch_real_fps * (self.fp_std[None, :] + FP_EPS)
            + self.fp_mean[None, :]
        )
        beta_mask, _, beta_threshold = _reference_beta_mask(image_np)
        patch_beta_fraction = np.asarray([
            beta_mask[y:y + self.patch_size, x:x + self.patch_size].mean()
            for x, y in positions
        ], dtype=np.float32)
        patch_lath_width = np.clip(raw_fingerprints[:, 1], 0.0, None)
        measurement_px_to_source_px = (
            float(self.patch_size) / float(MORPHOLOGY_MEASUREMENT_SIZE)
        )
        patch_lath_width_um = (
            patch_lath_width
            * measurement_px_to_source_px
            * MICRONS_PER_SOURCE_PIXEL
        )

        beta_fraction_field = self._reconstruct_scalar_field(
            patch_beta_fraction,
            positions,
            image_size,
        )
        lath_width_um_field = self._reconstruct_scalar_field(
            patch_lath_width_um,
            positions,
            image_size,
        )
        beta_fraction_field = np.clip(beta_fraction_field, 0.0, 1.0).astype(np.float32)
        lath_width_um_field = np.clip(lath_width_um_field, 0.0, None).astype(np.float32)

        summary = {
            "global_dark_contrast_area_fraction": float(beta_fraction_field.mean()),
            "global_bright_contrast_area_fraction": float(1.0 - beta_fraction_field.mean()),
            "global_mean_lath_width_um": float(lath_width_um_field.mean()),
            "lath_width_median_um": float(np.median(patch_lath_width_um)),
            "lath_width_p25_um": float(np.percentile(patch_lath_width_um, 25)),
            "lath_width_p75_um": float(np.percentile(patch_lath_width_um, 75)),
            "lath_width_sample_count": int(patch_lath_width_um.size),
            "dark_contrast_quantification_method": "global_plus_local_adaptive_threshold",
            "dark_contrast_preprocessing": "GaussianBlur_3x3_unsharp_CLAHE_blackhat_support",
            "dark_contrast_min_component_area_px": int(BETA_MIN_COMPONENT_AREA_PX),
            "dark_contrast_adaptive_block_size": int(BETA_ADAPTIVE_BLOCK_SIZE),
            "dark_contrast_adaptive_c": int(BETA_ADAPTIVE_C),
            "dark_contrast_threshold_enhanced_gray": float(beta_threshold),
            "microns_per_source_pixel": float(MICRONS_PER_SOURCE_PIXEL),
            "measurement_px_to_source_px": float(measurement_px_to_source_px),
            "lath_width_reference_patch_px": int(MORPHOLOGY_MEASUREMENT_SIZE),
            "lath_width_quantification_method": "medial_axis_distance_transform",
            "spatial_reconstruction": "overlapping_patch_fingerprint_field",
        }
        return beta_fraction_field, lath_width_um_field, raw_fingerprints, normalize_contrast_summary(summary)

    def _save_quantitative_fingerprint_maps(
        self,
        image_np,
        beta_fraction_field,
        lath_width_um_field,
        quantitative_summary,
        output_dir,
    ):
        quantitative_summary = normalize_contrast_summary(quantitative_summary)
        beta_vmax = float(np.clip(
            self.fp_mean[0] + 3.0 * self.fp_std[0],
            0.05,
            1.0,
        ))
        plot_specs = [
            (
                beta_fraction_field,
                "Blues",
                0.0,
                beta_vmax,
                "Dark-contrast area fraction",
                (
                    "Dark-contrast area fraction map\n"
                    f"Global Dark-contrast area fraction: {quantitative_summary['global_dark_contrast_area_fraction'] * 100.0:.2f}%"
                ),
                "PSIM_dark_contrast_area_fraction_map.png",
            ),
        ]

        for field, cmap, vmin, vmax, colorbar_label, title, filename in plot_specs:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=PANEL_DPI)
            ax.imshow(image_np, interpolation="nearest")
            im = ax.imshow(
                field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.68,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=FONT_TITLE, pad=10)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
            cbar.set_label(colorbar_label, fontsize=FONT_AXIS, labelpad=8)
            cbar.ax.tick_params(labelsize=FONT_TICK, width=0.8)
            safe_savefig(fig, os.path.join(output_dir, filename), dpi=PANEL_DPI)

    def _save_beta_phase_fraction_statistics(self, image_np, output_dir):
        beta_mask, _, threshold = _reference_beta_mask(image_np)
        beta_fraction = float(beta_mask.mean())
        alpha_fraction = 1.0 - beta_fraction

        display_image = image_np
        max_side = 1800
        scale = min(1.0, max_side / max(display_image.shape[:2]))
        if scale < 1.0:
            display_image = cv2.resize(
                display_image,
                (int(display_image.shape[1] * scale), int(display_image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        display_mask = cv2.resize(
            beta_mask.astype(np.uint8),
            (display_image.shape[1], display_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        overlay = display_image.astype(np.float32).copy()
        beta_color = np.array([45, 115, 220], dtype=np.float32)
        alpha_color = np.array([220, 82, 78], dtype=np.float32)
        overlay[display_mask] = 0.58 * overlay[display_mask] + 0.42 * beta_color
        overlay[~display_mask] = 0.78 * overlay[~display_mask] + 0.22 * alpha_color
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        Image.fromarray((beta_mask.astype(np.uint8) * 255)).save(
            os.path.join(output_dir, "PSIM_dark_contrast_mask.png")
        )

        fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), dpi=PANEL_DPI)
        axes[0].imshow(overlay)
        axes[0].set_title("Dark-/Bright-contrast segmentation", fontsize=FONT_TITLE, pad=9)
        axes[0].text(
            0.02,
            0.03,
            f"Dark-contrast area fraction (blue): {beta_fraction * 100.0:.2f}%\n"
            f"Bright-contrast area fraction (red): {alpha_fraction * 100.0:.2f}%",
            transform=axes[0].transAxes,
            color="white",
            fontsize=FONT_ANNOT,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.72, "pad": 5, "edgecolor": "none"},
        )
        axes[0].axis("off")

        labels = ["Dark-contrast\narea fraction", "Bright-contrast\narea fraction"]
        values = np.array([beta_fraction, alpha_fraction]) * 100.0
        bars = axes[1].bar(
            labels,
            values,
            color=[beta_color / 255.0, alpha_color / 255.0],
            width=0.56,
            linewidth=0,
        )
        axes[1].set_ylabel("Area fraction (%)", fontsize=FONT_AXIS)
        axes[1].set_ylim(0, max(100.0, float(values.max()) * 1.18))
        axes[1].set_title("Dark-/Bright-contrast area fractions", fontsize=FONT_TITLE, pad=9)
        axes[1].text(
            0.5,
            0.96,
            f"Iterative threshold after CLAHE: {threshold:.1f}",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=FONT_TICK,
            color="#555555",
        )
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
            )
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        fig.tight_layout(w_pad=2.6)
        safe_savefig(
            fig,
            os.path.join(output_dir, "PSIM_contrast_area_fraction_statistics.png"),
            dpi=PANEL_DPI,
        )

        for legacy_name in ("PSIM_beta_phase_mask.png", "PSIM_beta_phase_fraction_statistics.png"):
            legacy_path = os.path.join(output_dir, legacy_name)
            if os.path.isfile(legacy_path):
                os.remove(legacy_path)

    def _save_lath_width_statistics(self, patch_raw_fingerprints, output_dir):
        patch_raw_fingerprints = np.asarray(patch_raw_fingerprints, dtype=np.float32)
        # Morphology extraction resamples every native crop to
        # MORPHOLOGY_MEASUREMENT_SIZE before measuring width. Convert those
        # measurement pixels back to source pixels for physical units.
        width_um = (
            patch_raw_fingerprints[:, 1]
            * (float(self.patch_size) / float(MORPHOLOGY_MEASUREMENT_SIZE))
            * MICRONS_PER_SOURCE_PIXEL
        )
        width_um = width_um[np.isfinite(width_um) & (width_um > 0)]
        if width_um.size == 0:
            return

        mean_value = float(width_um.mean())
        median_value = float(np.median(width_um))
        p25, p75 = np.percentile(width_um, [25, 75])

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=PANEL_DPI)
        axes[0].hist(
            width_um,
            bins=min(24, max(8, int(np.sqrt(width_um.size) * 2))),
            color="#2A9D8F",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[0].axvline(mean_value, color="#D1495B", linewidth=1.6, label=f"Mean {mean_value:.3f}")
        axes[0].axvline(median_value, color="#264653", linewidth=1.6, linestyle="--", label=f"Median {median_value:.3f}")
        axes[0].set_xlabel("Mean α-lath width (um)", fontsize=FONT_AXIS)
        axes[0].set_ylabel("Patch count", fontsize=FONT_AXIS)
        axes[0].set_title("α-lath width distribution", fontsize=FONT_TITLE, pad=9)
        axes[0].legend(fontsize=FONT_TICK, frameon=False)

        axes[1].boxplot(
            width_um,
            vert=False,
            patch_artist=True,
            showfliers=False,
            widths=0.48,
            boxprops={"facecolor": "#2A9D8F", "alpha": 0.55, "linewidth": 0.9},
            medianprops={"color": "#D1495B", "linewidth": 1.6},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        axes[1].set_xlabel("Mean α-lath width (um)", fontsize=FONT_AXIS)
        axes[1].set_yticks([1])
        axes[1].set_yticklabels(["All patches"])
        axes[1].set_title("α-lath width summary", fontsize=FONT_TITLE, pad=9)
        axes[1].text(
            0.02,
            0.08,
            f"n = {width_um.size}\nIQR = {p25:.3f}-{p75:.3f} um",
            transform=axes[1].transAxes,
            fontsize=FONT_TICK,
            color="#444444",
        )
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("Patch-level α-lath width statistics", fontsize=FONT_TITLE + 1, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.4)
        safe_savefig(
            fig,
            os.path.join(output_dir, "PSIM_lath_width_statistics.png"),
            dpi=PANEL_DPI,
        )
        np.save(os.path.join(output_dir, "PSIM_lath_width_samples_um.npy"), width_um.astype(np.float32))

def pair_measurement_config():
    return {
        "feature_version": PAIR_FEATURE_VERSION,
        "inference_seed": INFERENCE_SEED,
        "num_timesteps": NUM_TIMESTEPS,
        "noise_mode": "crop-sha256-v1",
        "patch_size": PATCH_SIZE,
        "model_input_size": MODEL_INPUT_SIZE,
        "measurement_size": MORPHOLOGY_MEASUREMENT_SIZE,
        "beta_is_darker": BETA_IS_DARKER,
        "beta_min_component_area": BETA_MIN_COMPONENT_AREA_PX,
        "beta_blur_kernel": BETA_BLUR_KERNEL,
        "beta_close_kernel": BETA_CLOSE_KERNEL,
        "beta_closing_scope": "phase_fraction_and_legacy_fingerprint_only",
        "topology_mask_closing": False,
        "topology_skeleton_dilation_px": 0,
        "beta_sharpen_amount": BETA_SHARPEN_AMOUNT,
        "beta_adaptive_block_size": BETA_ADAPTIVE_BLOCK_SIZE,
        "beta_adaptive_c": BETA_ADAPTIVE_C,
        "lath_min_length_ratio": LATH_MIN_LENGTH_RATIO,
        "lath_max_width_ratio": LATH_MAX_WIDTH_RATIO,
    }


def checkpoint_signature():
    signature = {}
    paths = [os.path.join(CHECKPOINT_DIR, name) for name in (
        "unet_final.pth", "mta_final.pth", "dit_final.pth", "unified_head_final.pth")]
    paths.append(FP_STATS_PATH)
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        signature[os.path.basename(path)] = digest.hexdigest()
    return signature


def pair_design_matrix(logits, features):
    logits = np.asarray(logits, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("Expected four-class logits.")
    if features.shape != (len(logits), len(PAIR_FEATURE_NAMES)):
        raise ValueError("Unexpected pair-feature shape.")
    return np.column_stack([logits[:, 1] - logits[:, 0], features])


def apply_pair_calibrator(probs, logits, features, valid, model):
    probs = np.asarray(probs, dtype=np.float32)
    result = probs.copy()
    conditional = np.full(len(probs), 0.5, dtype=np.float32)
    applied = np.zeros(len(probs), dtype=bool)
    if model is None or float(model["strength"]) == 0.0:
        return result, conditional, applied
    x = pair_design_matrix(logits, features)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    # Refuse extrapolation far outside the calibration feature support.
    in_support = np.all((x >= np.asarray(model["support_low"]))
                        & (x <= np.asarray(model["support_high"])), axis=1)
    mass = probs[:, 0] + probs[:, 1]
    applied = (np.asarray(valid, dtype=bool) & np.isfinite(z).all(axis=1)
               & in_support & np.isin(probs.argmax(axis=1), [0, 1])
               & (mass >= PAIR_MIN_MASS))
    learned_log_odds = z @ np.asarray(model["coef"]) + float(model["intercept"])
    strength = float(model["strength"])
    log_odds = (1.0 - strength) * x[:, 0] + strength * learned_log_odds
    conditional[:] = expit(learned_log_odds)
    q = expit(log_odds[applied])
    result[applied, 1] = mass[applied] * q
    result[applied, 0] = mass[applied] - result[applied, 1]
    # Other classes remain exactly unchanged, including on unselected rows.
    return result, conditional, applied


def load_pair_calibrator(path):
    with open(path, encoding="utf-8") as stream:
        model = json.load(stream)
    if model.get("measurement") != pair_measurement_config():
        raise ValueError("Calibrator measurement settings differ; recalibration is required.")
    if model.get("feature_names") != ["raw_log_odds_D_over_B"] + PAIR_FEATURE_NAMES:
        raise ValueError("Calibrator feature order differs.")
    for key in ("mean", "scale", "coef", "support_low", "support_high"):
        values = np.asarray(model[key], dtype=float)
        if values.shape != (len(PAIR_FEATURE_NAMES) + 1,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid calibrator {key}.")
    if (np.any(np.asarray(model["scale"]) <= 0)
            or not np.isfinite(float(model["intercept"]))
            or not 0 <= float(model["strength"]) <= 1):
        raise ValueError("Invalid calibrator scale, intercept or strength.")
    return model


def pair_metrics(probabilities, labels, sample_weight=None):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != (len(labels),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, non-negative and match labels.")
    predictions = probabilities.argmax(axis=1)
    recalls = []
    for class_id in (0, 1):
        selected = labels == class_id
        denominator = weights[selected].sum()
        recalls.append(float(np.sum(weights[selected] * (predictions[selected] == class_id))
                             / max(denominator, 1e-12)))
    matrix = np.zeros((2, 4), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1))
    nll = float(np.sum(weights * losses) / max(weights.sum(), 1e-12))
    return {"recall_B": recalls[0], "recall_D": recalls[1],
            "balanced_accuracy": float(np.mean(recalls)), "nll": nll,
            "confusion_rows_B_D_columns_four_classes": matrix.tolist()}


def source_balancing_weights(groups):
    """Give each independent source group equal total influence."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups must be a non-empty one-dimensional array.")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights * (len(weights) / weights.sum())


def extract_calibration_patches(
    image,
    patch_size=None,
    stride=CALIBRATION_PATCH_STRIDE,
    max_patches=MAX_CALIBRATION_PATCHES_PER_IMAGE,
):
    """Extract deterministic, weakly correlated crops from one labelled source image."""
    patch_size = PATCH_SIZE if patch_size is None else int(patch_size)
    stride = patch_size if stride is None else int(stride)
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    width, height = image.size
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Calibration source {width}x{height} is smaller than "
            f"the required {patch_size}x{patch_size} field of view."
        )

    def positions_1d(length):
        positions = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    candidates = [(x, y) for y in positions_1d(height) for x in positions_1d(width)]
    if max_patches is not None and len(candidates) > max_patches:
        selected = np.linspace(0, len(candidates) - 1, max_patches, dtype=int)
        candidates = [candidates[index] for index in np.unique(selected)]
    patches = [
        image.crop((x, y, x + patch_size, y + patch_size))
        for x, y in candidates
    ]
    return patches, candidates


def train_pair_calibrator(psim, manifest_path, output_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if os.path.exists(output_path):
        raise FileExistsError(f"Use a new calibration output filename: {output_path}")
    with open(manifest_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "group", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires path,label,group,split columns.")
        rows = list(reader)
    base = os.path.dirname(os.path.abspath(manifest_path))
    groups = {}
    group_labels = {}
    hashes = {}
    images, labels, splits = [], [], []
    patch_sources, patch_groups, patch_positions = [], [], []
    source_count = 0
    for row in rows:
        label = row["label"].strip()
        split = row["split"].strip()
        group = row["group"].strip()
        if label not in CLASS_NAMES[:2] or split not in {"train", "val", "test"} or not group:
            raise ValueError(f"Invalid manifest row: {row}")
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group appears in multiple splits: {group}")
        if group in group_labels and group_labels[group] != label:
            raise ValueError(f"Source group has conflicting labels: {group}")
        groups[group] = split
        group_labels[group] = label
        path = os.path.join(base, row["path"])
        with Image.open(path) as source:
            image = source.convert("RGB")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate calibration image: {path}, previously {hashes[digest]}")
        hashes[digest] = path
        source_patches, source_positions = extract_calibration_patches(image)
        source_count += 1
        images.extend(source_patches)
        labels.extend([CLASS_NAMES.index(label)] * len(source_patches))
        splits.extend([split] * len(source_patches))
        patch_sources.extend([path] * len(source_patches))
        patch_groups.extend([group] * len(source_patches))
        patch_positions.extend(source_positions)
    labels = np.asarray(labels, dtype=int)
    splits = np.asarray(splits)
    patch_groups = np.asarray(patch_groups)
    sample_weights = source_balancing_weights(patch_groups)
    for split in ("train", "val"):
        for label in (0, 1):
            if np.sum((splits == split) & (labels == label)) < 4:
                raise ValueError(
                    f"Need at least four {CLASS_NAMES[label]} patches in {split}; "
                    "more independent source images are recommended."
                )
            class_groups = {
                patch_groups[index]
                for index in np.flatnonzero((splits == split) & (labels == label))
            }
            if len(class_groups) < 2:
                raise ValueError(
                    f"Need at least two independent source groups for "
                    f"{CLASS_NAMES[label]} in {split}."
                )
    if np.any(splits == "test") and set(labels[splits == "test"]) != {0, 1}:
        raise ValueError("Test split must contain both classes.")
    psim.pair_calibrator = None
    probabilities, _, _, _, _ = psim.inference_all_patches(images)
    diagnostics = psim.patch_diagnostics
    x = pair_design_matrix(diagnostics["patch_raw_logits"], diagnostics["patch_pair_features"])
    valid = diagnostics["patch_feature_valid"]
    train = (splits == "train") & valid
    if min(np.sum(train & (labels == c)) for c in (0, 1)) < 4:
        raise ValueError("Not enough valid segmented training samples for both classes.")
    scaler = StandardScaler().fit(x[train], sample_weight=sample_weights[train])
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=INFERENCE_SEED,
    )
    classifier.fit(
        scaler.transform(x[train]),
        labels[train],
        sample_weight=sample_weights[train],
    )
    low = x[train].min(axis=0) - 0.5 * scaler.scale_
    high = x[train].max(axis=0) + 0.5 * scaler.scale_
    model = {
        "checkpoint_signature": checkpoint_signature(),
        "measurement": pair_measurement_config(),
        "feature_names": ["raw_log_odds_D_over_B"] + PAIR_FEATURE_NAMES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "support_low": low.tolist(), "support_high": high.tolist(), "strength": 0.0,
    }
    val = splits == "val"
    baseline = pair_metrics(probabilities[val], labels[val], sample_weights[val])
    best = baseline
    selected_strength = 0.0
    candidates = []
    for strength in (0.25, 0.5, 0.75, 1.0):
        model["strength"] = strength
        adjusted, _, applied = apply_pair_calibrator(
            probabilities, diagnostics["patch_raw_logits"],
            diagnostics["patch_pair_features"], valid, model,
        )
        metrics = pair_metrics(adjusted[val], labels[val], sample_weights[val])
        candidates.append({"strength": strength, "applied_val_count": int(applied[val].sum()), **metrics})
        # A candidate cannot trade away either class's measured validation recall.
        if (metrics["recall_B"] >= baseline["recall_B"]
                and metrics["recall_D"] >= baseline["recall_D"]
                and metrics["nll"] < baseline["nll"]
                and (metrics["balanced_accuracy"], -metrics["nll"])
                > (best["balanced_accuracy"], -best["nll"])):
            best = metrics
            selected_strength = strength
    model["strength"] = selected_strength
    report = {"validation_used_for_strength_selection": True,
              "validation_baseline": baseline, "validation_selected": best,
              "candidates": candidates, "selected_strength": selected_strength,
              "valid_count": int(valid.sum()), "patch_count": len(images),
              "source_image_count": source_count,
              "group_splits": groups,
              "group_labels": group_labels,
              "warning": "Validation selected the strength; only a separate test split estimates generalization."}
    if np.any(splits == "test"):
        adjusted, _, _ = apply_pair_calibrator(
            probabilities, diagnostics["patch_raw_logits"], diagnostics["patch_pair_features"], valid, model)
        test = splits == "test"
        report["test_baseline"] = pair_metrics(
            probabilities[test], labels[test], sample_weights[test]
        )
        report["test_calibrated"] = pair_metrics(
            adjusted[test], labels[test], sample_weights[test]
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
    with open(output_path + ".report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    np.savez_compressed(
        output_path + ".samples.npz",
        labels=labels,
        splits=splits,
        paths=np.asarray(patch_sources),
        groups=patch_groups,
        sample_weights=sample_weights,
        positions=np.asarray(patch_positions, dtype=np.int32),
        **diagnostics,
    )
    print(json.dumps(report, indent=2))
    print(f"Calibrator saved: {output_path}; strength={selected_strength:.2f}")
    if selected_strength == 0:
        print("No validated improvement: correction remains disabled. Inspect segmentation and labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four-class PSIM with validated pairwise morphology correction")
    parser.add_argument("--input", default=LARGE_IMAGE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE,
                        help="Source pixels per crop; 1024 for supplied 1024px training crops at the same physical scale.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Source-pixel spacing between crops; default is one quarter of patch size.")
    parser.add_argument("--area-mode", choices=("patch-vote", "probability"), default=AREA_MODE,
                        help="Estimate area from crop class votes or from averaged probabilities.")
    parser.add_argument("--boundary-radius", type=int, default=BOUNDARY_BAND_RADIUS_PX,
                        help="Maximum boundary refinement band in source pixels; 0 disables it.")
    parser.add_argument("--no-widman-guidance", action="store_true",
                        help="Disable W/coarsened-W spatial guidance for a raw Gaussian comparison.")
    parser.add_argument("--min-class-presence-ratio", type=float, default=MIN_CLASS_PRESENCE_RATIO,
                        help="Minimum GLOBAL hard-class fraction (default 0.03).")
    parser.add_argument("--min-component-area-ratio", type=float, default=MIN_CLASS_REGION_AREA_RATIO,
                        help="Optional per-component cleanup fraction (default 0, disabled).")
    parser.add_argument("--min-component-area-px", type=int, default=MIN_CLASS_REGION_AREA_PX,
                        help="Optional per-component cleanup size in source pixels (default 0).")
    area_filter_options = parser.add_mutually_exclusive_group()
    area_filter_options.add_argument("--enable-area-filter", dest="enable_area_filter", action="store_true",
                                     help="Apply component/global hard-area cutoffs (enabled by default).")
    area_filter_options.add_argument("--disable-area-filter", dest="enable_area_filter", action="store_false",
                                     help="Explicitly disable component/global hard-area cutoffs.")
    parser.set_defaults(enable_area_filter=CLASS_AREA_FILTER_ENABLED)
    parser.add_argument("--refresh-cached-only", action="store_true",
                        help="Update reports from existing NPZ results without loading or running neural models; fail if incompatible.")
    parser.add_argument(
        "--calibrator",
        help=(
            "Optional validated JSON pair calibrator. Omit to use the small-image classifier directly."
        ),
    )
    parser.add_argument(
        "--no-pair-calibration",
        action="store_true",
        help="Disable automatic pair-calibrator discovery and run the raw four-class baseline.",
    )
    parser.add_argument("--fit-calibrator", metavar="MANIFEST_CSV")
    parser.add_argument("--calibrator-output", default=DEFAULT_PAIR_CALIBRATOR_PATH)
    parser.add_argument("--force", action="store_true", help="Recompute existing image results")
    args = parser.parse_args()
    if args.patch_size <= 0 or args.boundary_radius < 0:
        parser.error("patch-size must be positive and boundary-radius nonnegative.")
    PATCH_SIZE = args.patch_size
    STRIDE = max(1, PATCH_SIZE // 4) if args.stride is None else args.stride
    if not 0 < STRIDE <= PATCH_SIZE:
        parser.error("Require 0 < stride <= patch-size to cover every source pixel.")
    SIGMA = PATCH_SIZE / 8
    AREA_MODE = args.area_mode
    BOUNDARY_BAND_RADIUS_PX = args.boundary_radius
    WIDMAN_GUIDED_ENABLED = not args.no_widman_guidance
    MIN_CLASS_PRESENCE_RATIO = args.min_class_presence_ratio
    MIN_CLASS_REGION_AREA_RATIO = args.min_component_area_ratio
    MIN_CLASS_REGION_AREA_PX = args.min_component_area_px
    CLASS_AREA_FILTER_ENABLED = args.enable_area_filter
    if not 0 <= MIN_CLASS_PRESENCE_RATIO <= 1 or not 0 <= MIN_CLASS_REGION_AREA_RATIO <= 1:
        parser.error("Class-area ratios must lie in [0, 1].")
    if MIN_CLASS_REGION_AREA_PX < 0:
        parser.error("min-component-area-px must be nonnegative.")
    print(f"PSIM program: {os.path.abspath(__file__)}")
    print(f"PSIM program version: {PROGRAM_VERSION}")
    print(f"Native crop {PATCH_SIZE}px -> model {MODEL_INPUT_SIZE}px; stride={STRIDE}; area={AREA_MODE}")
    print(f"Hard-area filtering: {'ON' if CLASS_AREA_FILTER_ENABLED else 'OFF'}; "
          f"component minimum={MIN_CLASS_REGION_AREA_RATIO:.2%}; "
          f"class presence minimum={MIN_CLASS_PRESENCE_RATIO:.2%}")
    print(f"W/coarsened-W image guidance: {WIDMAN_GUIDED_ENABLED}; "
          f"band={WIDMAN_GUIDED_BAND_PX}px; max conditional change={WIDMAN_GUIDED_MAX_DELTA}")
    LARGE_IMAGE_PATH = args.input
    OUTPUT_DIR = args.output_dir
    SKIP_EXISTING = not args.force
    if args.calibrator and args.no_pair_calibration:
        parser.error("Choose --calibrator or --no-pair-calibration, not both.")
    if args.calibrator and args.fit_calibrator:
        parser.error("Choose inference or calibration training, not both.")
    if args.refresh_cached_only:
        if args.force or args.fit_calibrator or args.calibrator:
            parser.error("--refresh-cached-only cannot be combined with --force, --fit-calibrator or --calibrator.")
        refresh_cached_reports(LARGE_IMAGE_PATH, OUTPUT_DIR)
        raise SystemExit(0)
    active_calibrator = None
    active_calibrator_path = None
    if args.fit_calibrator:
        try:
            import sklearn
        except ImportError:
            parser.error("Calibration training requires scikit-learn: pip install scikit-learn")
    set_inference_seed()
    (
        class_prompt_cache,
        vae,
        unet_wrapper,
        ace,
        mta,
        dit,
        pqa,
        unified_head,
        connectivity_head,
        noise_scheduler,
        total_timesteps,
        fp_mean,
        fp_std,
        class_means,
    ) = load_models()

    if args.calibrator:
        active_calibrator_path = os.path.abspath(args.calibrator)
    elif (
        AUTO_PAIR_CALIBRATION
        and
        not args.fit_calibrator
        and not args.no_pair_calibration
        and os.path.isfile(DEFAULT_PAIR_CALIBRATOR_PATH)
    ):
        active_calibrator_path = os.path.abspath(DEFAULT_PAIR_CALIBRATOR_PATH)
        print(f"Auto-discovered pair calibrator: {active_calibrator_path}")

    if active_calibrator_path:
        active_calibrator = load_pair_calibrator(active_calibrator_path)
        if active_calibrator.get("checkpoint_signature") != checkpoint_signature():
            raise ValueError("Calibrator was fitted with different checkpoint weights or fingerprint statistics.")
        digest = hashlib.sha256(json.dumps(active_calibrator, sort_keys=True).encode()).hexdigest()[:16]
        WIDTH_CALIBRATION_VERSION = INFERENCE_VERSION + "-" + digest
        print(f"Loaded pair calibrator: {active_calibrator_path}")
        print(f"Pair calibrator strength: {float(active_calibrator['strength']):.3f}")

    psim = PhysicsAwareSpatialInferenceModule(
        class_prompt_cache,
        vae,
        unet_wrapper,
        ace,
        mta,
        dit,
        pqa,
        unified_head,
        connectivity_head,
        noise_scheduler,
        total_timesteps,
        fp_mean,
        fp_std,
        class_means,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        sigma=SIGMA,
        lambda_fp=LAMBDA_FINGERPRINT,
    )

    psim.pair_calibrator = active_calibrator
    psim.pair_calibrator_source = active_calibrator_path
    if args.fit_calibrator:
        train_pair_calibrator(psim, args.fit_calibrator, args.calibrator_output)
        raise SystemExit(0)
    print("Pair correction: " + (
        f"validated model, strength={active_calibrator['strength']}, path={active_calibrator_path}"
        if active_calibrator
        else "disabled (raw four-class baseline; Basket/Discontinuous bias is not corrected)"
    ))

    image_paths = resolve_input_images(LARGE_IMAGE_PATH)
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(f"Found {len(image_paths)} image(s) to analyze.")

    for image_path in image_paths:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(OUTPUT_DIR, safe_name(image_stem))
        result_path = os.path.join(image_output_dir, "PSIM_results_1.npz")

        if SKIP_EXISTING and os.path.exists(result_path):
            if result_has_current_fields(result_path):
                remove_legacy_quantitative_outputs(image_output_dir)
                print(f"\n=== Skipping current result for: {image_path} ===")
                continue
            if REUSE_CACHED_PATCH_RESULTS:
                print(f"\n=== Re-rendering cached patch result for: {image_path} ===")
                try:
                    psim.rerender_cached_result(image_path, result_path)
                    continue
                except (OSError, ValueError, KeyError) as error:
                    print(
                        "Warning: cached-result re-rendering failed; "
                        f"running full inference instead. Reason: {error}"
                    )

        print(f"\n=== Running PSIM for: {image_path} ===")
        psim.run(image_path)