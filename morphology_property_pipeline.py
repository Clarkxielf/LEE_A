# -*- coding: utf-8 -*-
"""Measured morphology-property comparisons; no property-model fitting.

Default: reuse valid PSIM caches, then quantify pending images with up to
two GPU-isolated workers. Set CUDA_VISIBLE_DEVICES before launch to choose
physical GPUs; each worker sees only cuda:0. --psim-gpu-count 1 retains the
original in-process device route. --skip-psim never starts GPU workers.

The default SPECIMEN-level morphology prior triggers when the equal-image
mean ORIGINAL hard-map W+ACW area fraction is strictly above 60%. Then
BW/DC pixels in EVERY image of that specimen are reassigned by local
saved PSIM evidence; original physical measurements
and archives are preserved. Use --widmanstatten-prior off to disable.
Exact score ties use the nearest original W/ACW label; equal distances
use fixed coordinate order, recorded explicitly as a boundary convention.
Completed v2/v3 classification maps are reusable after hash checks ONLY
for specimens satisfying the new rule; trigger decisions are never cached.
Derived maps/fractions are prior-constrained, not independent verification.
"""

import argparse

import importlib

import json

import os

import re

import sys
import subprocess
import time
import uuid

from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib import font_manager

import numpy as np

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent

_CONTRACT_ENV = os.environ.get("LEE_A_PROJECT_ROOT") or os.environ.get("MORPHOLOGY_PROJECT_ROOT")

_IMPORT_DIRS = []

if _CONTRACT_ENV:
    _IMPORT_DIRS.append(Path(_CONTRACT_ENV).expanduser())

_IMPORT_DIRS.extend([_SCRIPT_DIR, *_SCRIPT_DIR.parents[:5]])

for _import_dir in _IMPORT_DIRS:
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

try:
    _contract = importlib.import_module("structure_performance_contract")
    CLASS_NAMES = _contract.CLASS_NAMES
    CONTRACT_VERSION = _contract.CONTRACT_VERSION
    FINGERPRINT_LABELS = _contract.FINGERPRINT_LABELS
    parse_mean_std = _contract.parse_mean_std
except (ImportError, AttributeError):
    # This is identical to structure_performance_contract.py in the four-class
    # 2-D release and keeps the server-side Diffusion folder self-contained.
    CLASS_NAMES = (
        "Basket_weave",
        "Discontinuous",
        "Widmanstatten",
        "Alpha_coarsened_Widmanstatten",
    )
    CONTRACT_VERSION = "four-class-2d-directional-v1"
    FINGERPRINT_LABELS = ("dark_contrast_area_fraction", "mean_lath_width_um")

    def parse_mean_std(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return np.nan, np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value), np.nan
        text = str(value).strip().replace("＋", "+").replace("－", "-").replace("−", "-")
        if not text:
            return np.nan, np.nan
        text = text.replace("±", "+/-")
        values = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if not values:
            return np.nan, np.nan
        mean = float(values[0])
        std = abs(float(values[1])) if "+/-" in text and len(values) > 1 else np.nan
        return mean, std

DARK_CONTRAST_FIELD = "dark_contrast_area_fraction"

BRIGHT_CONTRAST_FIELD = "bright_contrast_area_fraction"

CONTRAST_TERMINOLOGY = {
    "version": "contrast-area-naming-v1",
    "dark_field": DARK_CONTRAST_FIELD,
    "dark_label": "Dark-contrast area fraction",
    "bright_field": BRIGHT_CONTRAST_FIELD,
    "bright_label": "Bright-contrast area fraction",
    "width_field": "mean_lath_width_um",
    "width_label": "Mean α-lath width (µm)",
    "legacy_input_aliases": {"beta_phase_fraction": DARK_CONTRAST_FIELD},
    "fraction_unit": "fraction_0_to_1",
    "fingerprint_order": [DARK_CONTRAST_FIELD, "mean_lath_width_um"],
    "bright_export_rule": "1 - dark on the same original binary contrast domain; display/export only, never a third predictor",
    "interpretation": "Optical contrast fractions do not identify crystallographic alpha/beta phase fractions.",
    "measurement_change": False,
    "segmentation_titles": {"dark": "Dark-contrast area fraction", "bright": "Bright-contrast area fraction"},
    "external_scope": "This script does not draw segmentation maps or sample diffusion images; external PSIM/generator labels must be updated in their owning programs.",
}

def canonical_contrast_name(name):
    """Translate descriptor IDs only, not filesystem paths or arbitrary prose."""
    if isinstance(name, str) and re.fullmatch(
            r"(?:global_|fp_)?beta_phase_fraction(?:_[A-Za-z0-9_]+)?", name):
        return name.replace("beta_phase_fraction", DARK_CONTRAST_FIELD)
    return name

def normalize_contrast_frame(frame):
    """Read legacy/new columns without changing values; reject conflicting aliases.

    Missing canonical cells may be filled from the matching legacy column.
    Equal finite values tolerate only serialization roundoff (rtol=1e-10).
    """
    out = pd.DataFrame(frame).copy()
    if out.columns.duplicated().any():
        raise ValueError("Duplicate columns in morphology table.")
    for old in list(out.columns):
        new = canonical_contrast_name(old)
        if new == old:
            continue
        if new not in out:
            out = out.rename(columns={old: new})
            continue
        a, b = out[new], out[old]
        both = a.notna() & b.notna()
        av = pd.to_numeric(a[both], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b[both], errors="coerce").to_numpy(float)
        equal = np.isclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=False)
        equal |= (a[both].astype(str).to_numpy() == b[both].astype(str).to_numpy())
        if not equal.all():
            raise ValueError(f"Conflicting legacy/new morphology columns: {old} / {new}")
        out[new] = a.where(a.notna(), b)
        out = out.drop(columns=old)
    return out

def contrast_export_table(frame):
    """Add the complementary bright fraction to physical descriptor exports only.

    This derived column is not added to training feature matrices. No pixel
    threshold, segmentation, width measurement, scaling or model is changed.
    Unknown or out-of-range dark values yield a missing bright value.
    """
    out = normalize_contrast_frame(frame)
    if DARK_CONTRAST_FIELD not in out:
        return out
    dark = pd.to_numeric(out[DARK_CONTRAST_FIELD], errors="coerce")
    bright = (1.0 - dark).where(dark.between(0, 1) & np.isfinite(dark))
    if BRIGHT_CONTRAST_FIELD in out:
        existing = pd.to_numeric(out[BRIGHT_CONTRAST_FIELD], errors="coerce")
        both = existing.notna() & bright.notna()
        if not np.allclose(existing[both], bright[both], rtol=1e-10, atol=1e-12):
            raise ValueError("Bright/dark contrast fractions do not share the original binary domain.")
    out[BRIGHT_CONTRAST_FIELD] = bright
    return out

def write_contrast_terminology(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "contrast_terminology.json").write_text(
        json.dumps(CONTRAST_TERMINOLOGY, ensure_ascii=False, indent=2), encoding="utf-8")

FINGERPRINT_LABELS = tuple(canonical_contrast_name(name) for name in FINGERPRINT_LABELS)

def resolve_project_root():
    """Use the LEE_A server project unless an explicit environment override is supplied."""
    env_root = os.environ.get("LEE_A_PROJECT_ROOT") or os.environ.get("MORPHOLOGY_PROJECT_ROOT")
    return Path(env_root).expanduser() if env_root else Path("/home/user/data8t/LEE/LEE_A")


PROJECT_ROOT = resolve_project_root()

DIFFUSION_DIR = PROJECT_ROOT / "Diffusion"

sys.path.insert(0, str(DIFFUSION_DIR))

FOUR_CLASS_OUTPUT_DIR = DIFFUSION_DIR / "Deep-TiDiffusion-Outputs-FourClass-v3"

DESCRIPTOR_LABELS = {
    "dark_contrast_area_fraction": "Dark-contrast area fraction",
    "mean_lath_width_um": "Mean α-lath width (µm)",
    "hard_Basket_weave": "Basket-weave fraction",
    "hard_Discontinuous": "Discontinuous fraction",
    "hard_Widmanstatten": "Widmanstatten fraction",
    "hard_Alpha_coarsened_Widmanstatten": "Coarsened Widmanstatten fraction",
}


CONTRACT_VERSION = "specimen-cohort-three-endpoint-v2"
FIVE_POINT_FONT_SIZE = AXIS_FONT_SIZE = 8.0
TITLE_FONT_SIZE = 8.5
TICK_FONT_SIZE = LEGEND_FONT_SIZE = 7.2
COLORBAR_FONT_SIZE = 7.0
FIGURE_DPI = 300
MIN_CLASS_REGION_AREA_RATIO = MIN_CLASS_PRESENCE_RATIO = HARD_CLASS_MIN_RATIO = 0.03


def configure_plot_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = ("Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")
    selected_font = next((name for name in preferred_fonts if name in available_fonts), "DejaVu Sans")
    if selected_font != "Arial":
        print(f"Plot font: Arial is unavailable; using {selected_font}.")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font, "DejaVu Sans"],
            "font.size": FIVE_POINT_FONT_SIZE,
            "axes.labelsize": AXIS_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
            "figure.constrained_layout.use": False,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 5.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3,
        }
    )

def suppress_psim_visualization_flags(lia):
    false_flags = [
        "SAVE_VISUALIZATION", "SAVE_VISUALIZATIONS", "SAVE_FIGURE", "SAVE_FIGURES",
        "SAVE_PLOTS", "MAKE_FIGURES", "MAKE_PLOTS", "PLOT_RESULTS", "SHOW_RESULTS",
        "SHOW_FIGURES", "ENABLE_VISUALIZATION", "ENABLE_VISUALIZATIONS", "VISUALIZE",
        "SAVE_OVERLAY", "SAVE_OVERLAYS", "SAVE_HEATMAP", "SAVE_HEATMAPS",
        "SAVE_PATCH_VISUALIZATION", "SAVE_FIELD_VISUALIZATION", "SAVE_ANALYSIS_FIGURES",
    ]
    for flag in false_flags:
        if hasattr(lia, flag):
            setattr(lia, flag, False)
    if hasattr(lia, "plt"):
        lia.plt.show = lambda *args, **kwargs: None

    def is_psim_png(path):
        try:
            name = os.path.basename(os.fspath(path)).lower()
        except TypeError:
            return False
        return name.startswith("psim_") and name.endswith(".png")

    if hasattr(lia, "safe_savefig"):
        original_safe_savefig = lia.safe_savefig

        def safe_savefig_no_psim_png(fig, path, *args, **kwargs):
            if is_psim_png(path):
                try:
                    lia.plt.close(fig)
                except Exception:
                    plt.close(fig)
                return None
            return original_safe_savefig(fig, path, *args, **kwargs)

        lia.safe_savefig = safe_savefig_no_psim_png

    if hasattr(lia, "save_rgb_array"):
        original_save_rgb_array = lia.save_rgb_array

        def save_rgb_array_no_psim_png(path, arr, *args, **kwargs):
            if is_psim_png(path):
                return None
            return original_save_rgb_array(path, arr, *args, **kwargs)

        lia.save_rgb_array = save_rgb_array_no_psim_png

    if hasattr(lia, "Image"):
        original_image_save = lia.Image.Image.save

        def image_save_no_psim_png(self, fp, *args, **kwargs):
            if is_psim_png(fp):
                return None
            return original_image_save(self, fp, *args, **kwargs)

        lia.Image.Image.save = image_save_no_psim_png

def normalize_no_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))

def find_existing_psim_archive(result_dir):
    """Return an existing PSIM archive, including numbered legacy names.

    Some PSIM versions avoid overwriting an old result by writing
    ``PSIM_results_1.npz`` (and higher suffixes).  Treat those files as the
    same completed record; requiring the unsuffixed name caused needless
    re-inference on every rerun.
    """
    result_dir = Path(result_dir)
    exact = result_dir / "PSIM_results.npz"
    if exact.exists():
        return exact
    numbered = sorted(
        result_dir.glob("PSIM_results_*.npz"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return numbered[0] if numbered else None

def parse_mean_std_cell(value):
    # Parse the mean BEFORE looking for a standard deviation. A cell such as
    # 'nan +/- 0.4' must never turn into a measured mean of 0.4.
    if value is None or pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value), np.nan) if np.isfinite(value) else (np.nan, np.nan)
    text = str(value).strip().replace("−", "-").replace("－", "-").replace("＋", "+").replace("±", "+/-")
    parts = text.split("+/-", 1)
    number = r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    match = re.match(number, parts[0])
    if match is None:
        return np.nan, np.nan
    mean = float(match.group(1))
    std_match = re.match(number, parts[1]) if len(parts) > 1 else None
    std = float(std_match.group(1)) if std_match else np.nan
    if np.isfinite(std) and std < 0:
        raise ValueError(f"Experimental SD cannot be negative: {value}")
    return mean if np.isfinite(mean) else np.nan, std if np.isfinite(std) else np.nan

def sanitize_hard_class_name(name):
    text = str(name).strip()
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def normalize_fraction_dict(frac_dict):
    clean = {}
    for key, value in frac_dict.items():
        value = pd.to_numeric(value, errors="coerce")
        if pd.isna(value):
            continue
        clean[sanitize_hard_class_name(key)] = float(value)

    if not clean:
        return clean

    total = float(np.asarray(list(clean.values()), dtype=float).sum())
    if total > 1.5:
        clean = {k: v / 100.0 for k, v in clean.items()}
    return clean

def enforce_four_class_fractions(frac_dict, source):
    """Return four-class hard fractions after applying the 3% presence rule.

    PSIM's region and global presence filters use the same threshold. Applying
    it again here is intentional: summaries produced by older PSIM versions
    can contain a tiny hard-class remainder even though that class should not
    be reported as an independent tissue category.
    """
    clean = normalize_fraction_dict(frac_dict)
    expected = {sanitize_hard_class_name(name) for name in CLASS_NAMES}
    present = set(clean)
    unexpected = present - expected
    if unexpected:
        raise ValueError(
            f"{source} has unsupported class fractions {sorted(unexpected)}, but this pipeline requires "
            f"{list(CLASS_NAMES)}. Re-run four-class large-image analysis."
        )
    if not clean:
        return {}
    values = np.asarray([clean.get(sanitize_hard_class_name(name), 0.0) for name in CLASS_NAMES], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{source} contains no positive four-class fractions.")
    fractions = values / total
    # Remove classes below the declared minimum and renormalize the retained
    # composition. The argmax class is retained as a guard against an all-zero
    # vector caused by numerical rounding or an unusually sparse legacy file.
    retained = fractions >= HARD_CLASS_MIN_RATIO
    if not np.any(retained):
        retained[int(np.argmax(fractions))] = True
    filtered = np.where(retained, fractions, 0.0)
    filtered_sum = float(filtered.sum())
    if filtered_sum <= 0.0:
        filtered[int(np.argmax(fractions))] = 1.0
        filtered_sum = 1.0
    filtered /= filtered_sum
    return {
        sanitize_hard_class_name(name): float(value)
        for name, value in zip(CLASS_NAMES, filtered)
    }

def extract_hard_ratio_from_summary(summary):
    """Read old and v15 PSIM class-fraction summary keys."""
    candidate_keys = [
        # Prefer the post-presence-filter report vector.  The same summary may
        # also contain raw/pre-filter vectors for audit purposes.
        "output_hard_global_phase_fraction",
        "class_area_fraction",
        "hard_ratio", "hard_ratios", "hard_fraction", "hard_fractions",
        "hard_phase_ratio", "hard_phase_fraction", "hard_area_fraction",
        "hard_argmax_ratio", "argmax_ratio", "argmax_fraction",
        "hard_global_phase_fraction", "hard_global_phase_fractions",
    ]
    parents = [summary]
    for key in ("diagnostics", "phase_fraction", "phase_fractions",
                "global_phase_fraction", "global_phase_fractions",
                "phase_statistics", "global_statistics"):
        value = summary.get(key)
        if isinstance(value, dict):
            parents.append(value)
    for parent in parents:
        for key in candidate_keys:
            value = parent.get(key)
            if isinstance(value, dict):
                frac = enforce_four_class_fractions(value, f"PSIM summary key '{key}'")
                if frac:
                    return frac
    return {}

def default_class_names(n_classes):
    if n_classes == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [f"class_{i}" for i in range(n_classes)]

def extract_hard_ratio_from_npz(npz_path, summary=None):
    data = np.load(npz_path, allow_pickle=True)
    summary = summary or {}

    class_names = None
    if "class_names" in data:
        class_names = [str(x) for x in data["class_names"].tolist()]
    elif "class_names" in summary and isinstance(summary["class_names"], list):
        class_names = [str(x) for x in summary["class_names"]]

    if "hard_ratio" in data:
        values = np.asarray(data["hard_ratio"], dtype=float).reshape(-1)
        if values.size != len(CLASS_NAMES):
            raise ValueError(
                f"{npz_path} has {values.size} hard classes; expected {len(CLASS_NAMES)}. "
                "Re-run four-class large-image analysis."
            )
        if class_names is None:
            class_names = default_class_names(len(values))

        frac = {}
        for i, value in enumerate(values):
            cls_name = class_names[i] if i < len(class_names) else f"class_{i}"
            frac[sanitize_hard_class_name(cls_name)] = float(value)
        return enforce_four_class_fractions(frac, f"{npz_path}:hard_ratio")

    # Do not derive area fractions from prob_field here.  That array is a
    # softmax probability field and its runner-up probabilities are non-zero
    # even when a class has no hard classified area.  This pipeline requires
    # the hard summary/area vector produced by PSIM.
    raise ValueError(
        f"{npz_path} has no hard_ratio array. Available keys: {list(data.files)}. "
        "Re-run PSIM with hard four-class reporting enabled."
    )

def finite_summary_value(payload, key):
    try:
        value = float(payload.get(key, np.nan))
    except (AttributeError, TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan

def extract_quantitative_morphology(summary, npz_path):
    """Read original dark contrast and α-lath width, accepting legacy PSIM keys.

    Summary values take precedence over NPZ as before; the original nanmean
    fallback, units, descriptor ordering and numeric values are unchanged.
    Only aliases at the SAME precedence level are checked for conflicts.
    """
    quantitative = summary.get("quantitative_morphology", {})
    if not isinstance(quantitative, dict):
        quantitative = {}
    def alias_value(mapping, canonical, legacy):
        a = finite_summary_value(mapping, canonical)
        b = finite_summary_value(mapping, legacy)
        if np.isfinite(a) and np.isfinite(b) and not np.isclose(a, b, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Conflicting PSIM aliases: {canonical} / {legacy}")
        return a if np.isfinite(a) else b

    dark = alias_value(quantitative, "global_dark_contrast_area_fraction", "global_beta_phase_fraction")
    lath = finite_summary_value(quantitative, "global_mean_lath_width_um")
    if np.isfinite(dark) and np.isfinite(lath):
        return dark, lath
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if not np.isfinite(dark):
                keys = [key for key in ("dark_contrast_area_fraction_field", "beta_phase_fraction_field") if key in data]
                if len(keys) == 2:
                    a, b = [np.asarray(data[key], dtype=float) for key in keys]
                    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
                        raise RuntimeError("Conflicting PSIM NPZ dark-contrast aliases.")
                if keys:
                    dark = float(np.nanmean(np.asarray(data[keys[0]], dtype=float)))
            if not np.isfinite(lath) and "mean_lath_width_um_field" in data:
                lath = float(np.nanmean(np.asarray(data["mean_lath_width_um_field"], dtype=float)))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not read quantitative morphology fallback from {npz_path}: {exc}")
    return dark if np.isfinite(dark) else np.nan, lath if np.isfinite(lath) else np.nan

def annotate_association_ids(ax, x, y, ids):
    """Use rendered text extents and leader lines for unambiguous specimen IDs."""
    from matplotlib.transforms import Bbox
    renderer = ax.figure.canvas.get_renderer()
    font = font_manager.FontProperties(size=6.5)
    points_to_pixels = ax.figure.dpi / 72.0
    points = [ax.transData.transform((a, b)) for a, b in zip(x, y)]
    occupied = [Bbox.from_bounds(a - 5, b - 5, 10, 10) for a, b in points]
    candidates = [(0, 9), (0, -14), (15, 6), (-15, 6), (15, -14), (-15, -14),
                  (0, 22), (0, -27), (28, 0), (-28, 0)]
    for a, b, label, anchor in zip(x, y, ids, points):
        width, height, descent = renderer.get_text_width_height_descent(str(label), font, ismath=False)
        best, best_score = None, float("inf")
        for dx, dy in candidates:
            left = anchor[0] + dx * points_to_pixels - width / 2
            bottom = anchor[1] + dy * points_to_pixels - (height + descent if dy < 0 else 0)
            box = Bbox.from_bounds(left - 2, bottom - 2, width + 4, height + descent + 4)
            score = sum(box.overlaps(old) for old in occupied)
            if not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1):
                score += 100
            if score < best_score:
                best, best_score = ((dx, dy), box), score
            if score == 0:
                break
        offset, box = best
        ax.annotate(str(label), (a, b), xytext=offset, textcoords="offset points",
                    fontsize=6.5, ha="center", va="bottom" if offset[1] >= 0 else "top", zorder=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .3, "alpha": .9},
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .5,
                                "shrinkA": 2, "shrinkB": 4})
        occupied.append(box)

# Specimen-level cohort analysis. Appended to the unchanged PSIM readers.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import shlex
from zipfile import BadZipFile

ORIGINAL_IDS = (903, 733, 380, 186, 171, 169)
VALIDATION_IDS = (961, 375)
TARGETS = ('Elongation', 'Tensile_strength', 'Fatigue_strength')
TARGET_LABELS = {
    'Elongation': 'Elongation (%)',
    'Tensile_strength': 'Tensile strength (MPa)',
    'Fatigue_strength': 'Fatigue strength (MPa)',
}
HARD_COLS = [f'hard_{name}' for name in CLASS_NAMES]
FEATURES = [DARK_CONTRAST_FIELD, 'mean_lath_width_um', *HARD_COLS]
CLASS_LABELS = ['Basket-weave', 'Discontinuous', 'Widmanst\u00e4tten',
                '\u03b1-coarsened Widmanst\u00e4tten']
CLASS_COLORS = ['#0072B2', '#CC79A7', '#009E73', '#E69F00']
CLASS_HATCHES = ['...', '///', '', 'xxx']
COHORT_LABELS = {'original': 'Original experiments', 'validation': 'Validation experiments'}
COHORT_MARKERS = {'original': 'o', 'validation': 'D'}
COHORT_COLORS = {'original': '#333333', 'validation': '#9C4728'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def parse_args():
    p = argparse.ArgumentParser(description='Specimen-level morphology/property comparisons: original and validation cohorts, no direction split.')
    p.add_argument('--large-image-dir', type=Path, default=PROJECT_ROOT / 'data' / 'ALL-DATA')
    p.add_argument('--property-xlsx', type=Path, default=None,
                   help='Sectioned XLSX or canonical CSV; defaults to Mechanical_Property.xlsx under the image root.')
    p.add_argument('--checkpoint-dir', type=Path,
                   default=PROJECT_ROOT / 'Diffusion' / 'Deep-TiDiffusion-Ablation' / 'full' / 'seed_42',
                   help='Directory containing all final model weights from the same training run.')
    p.add_argument('--fp-stats-path', type=Path,
                   help='Training fingerprint statistics (.npy or .npz with mean/std). Default: run-local statistics, then legacy PSIM path.')
    p.add_argument('--psim-output-dir', type=Path, default=None,
                   help='PSIM output/cache directory. Default: psim_results. With --skip-psim, compatible sibling caches may be discovered.')
    p.add_argument('--analysis-output-dir', type=Path, default=None)
    p.add_argument('--widmanstatten-prior', choices=('specimen-sum-over-60','off'), default='specimen-sum-over-60',
                   help='When original specimen mean W+ACW > 60%%, reassign BW/DC in ALL images of that specimen. Default: specimen-sum-over-60; off keeps original PSIM fractions.')
    p.add_argument('--image-records-csv', type=Path,
                   help='Explicit image-level PSIM table with NO, image_name and six morphology descriptors.')
    p.add_argument('--properties-only', action='store_true', help='Plot measured properties without morphology; does not create morphology estimates.')
    p.add_argument('--force-psim', action='store_true', help='Recompute images with the external PSIM module; output is isolated by specimen.')
    p.add_argument('--skip-psim', action='store_true', help='Only read existing results; default quantifies missing or incomplete image results.')
    p.add_argument('--psim-module', default='large_image_analysis')
    p.add_argument('--psim-gpu-count', type=int, choices=(1, 2), default=PSIM_GPU_COUNT,
                   help='Default 2: isolated workers for pending images; 1: original in-process PSIM route')
    p.add_argument('--psim-min-free-gib', type=float, default=PSIM_MIN_FREE_GIB,
                   help='Free GPU memory required at startup; not a guarantee for full inference')
    p.add_argument('--max-images', type=int, help='Maximum pending images to quantify; incomplete datasets are not plotted.')
    p.add_argument('--save-psim-visuals', action='store_true')
    p.add_argument('--validation-ids', type=int, nargs='+', default=list(VALIDATION_IDS))
    p.add_argument('--axis-label', choices=['specimen', 'powder-hip', 'powder-full'], default='powder-hip')
    p.add_argument('--sort-by', choices=['property', 'specimen', 'hip'], default='property')
    p.add_argument('--pooled-correlations', action='store_true', default=True, help='Compatibility flag: combined specimen-level descriptive correlations are now always exported.')
    p.add_argument('--error-kind', choices=['reported', 'sd', 'sem', 'ci'], default='reported',
                   help='Meaning of the reported +/- values. Default does not assume SD.')
    p.add_argument('--dpi', type=int, default=300)
    # Old cache flags are accepted for command-line compatibility. Tables are
    # always rebuilt from current sources to avoid mixing directional caches.
    p.add_argument('--use-cache', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--no-cache', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--rebuild-cache', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not np.isfinite(args.psim_min_free_gib) or args.psim_min_free_gib < 0:
        p.error('--psim-min-free-gib must be finite and nonnegative.')
    if args.force_psim and args.skip_psim:
        p.error('--force-psim and --skip-psim cannot be combined.')
    if args.force_psim and args.properties_only:
        p.error('--properties-only cannot run PSIM.')
    if args.force_psim and args.image_records_csv:
        p.error('--image-records-csv cannot be combined with --force-psim.')
    if args.max_images is not None and args.max_images < 1:
        p.error('--max-images must be positive.')
    args.property_xlsx = args.property_xlsx or args.large_image_dir / 'Mechanical_Property.xlsx'
    args.psim_output_explicit = args.psim_output_dir is not None
    args.psim_output_dir = args.psim_output_dir or args.checkpoint_dir / 'psim_results'
    args.analysis_output_dir = args.analysis_output_dir or args.checkpoint_dir / (
        'specimen_cohort_associations_widmanstatten_sum60' if args.widmanstatten_prior=='specimen-sum-over-60'
        else 'specimen_cohort_associations')
    return args


def strict_no(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    return int(float(s)) if re.fullmatch(r'\d+(?:\.0+)?', s) else None


def cohort_name(value):
    s = str(value).lower().strip()
    if 'validation' in s or '\u9a8c\u8bc1' in s:
        return 'validation'
    if 'original' in s or '\u539f\u59cb' in s:
        return 'original'
    return None


def header_key(value):
    s = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', str(value).lower())
    if s in ('no', 'n0', 'specimen', 'specimenid', 'sampleid', '\u7f16\u53f7', '\u8bd5\u6837\u7f16\u53f7'):
        return 'NO'
    if 'equipment' in s or '\u8bbe\u5907' in s:
        return 'Equipment_ID'
    if 'powder' in s or '\u7c89\u672b' in s:
        return 'Powder_ID'
    if 'temperature' in s or '\u6e29\u5ea6' in s:
        return 'HIP_temperature_C'
    if 'pressure' in s or '\u538b\u529b' in s:
        return 'HIP_pressure_MPa'
    if 'holdingtime' in s or '\u4fdd\u6e29' in s:
        return 'HIP_holding_time_h'
    if 'tensile' in s or '\u6297\u62c9' in s:
        return 'Tensile_strength'
    if 'elongation' in s or '\u5ef6\u4f38' in s or '\u4f38\u957f' in s:
        return 'Elongation'
    if 'fatigue' in s or '\u75b2\u52b3' in s:
        return 'Fatigue_strength'
    if 'cohort' in s or 'dataset' in s or s == '\u5206\u7ec4':
        return 'cohort'
    return None


def read_mechanical_property(path, validation_ids=VALIDATION_IDS):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path)
        required = {'NO', *TARGETS}
        if not required.issubset(frame):
            raise ValueError(f'Canonical CSV requires {sorted(required)}.')
    else:
        sheets = pd.read_excel(path, header=None, sheet_name=None)
        rows = []
        for sheet, raw in sheets.items():
            active_cohort = cohort_name(sheet)
            column_map = None
            for row_index, cells in raw.iterrows():
                values = cells.tolist()
                first = next((str(x).strip() for x in values if pd.notna(x) and str(x).strip()), '')
                if strict_no(values[0]) is None:
                    section = cohort_name(first)
                    if section:
                        active_cohort = section
                    candidate = {header_key(v): j for j, v in enumerate(values) if header_key(v)}
                    if {'NO', *TARGETS}.issubset(candidate):
                        for j, v in enumerate(values):
                            if header_key(v) in TARGETS and re.search(r'\b(?:xy|z)\b', str(v), re.I):
                                raise ValueError('Directional workbook detected. Supply the new specimen-level table.')
                        if len([v for v in values if header_key(v) in TARGETS]) != len(TARGETS):
                            raise ValueError('Duplicate property headers; no implicit direction averaging is permitted.')
                        column_map = candidate
                    continue
                if column_map is None:
                    raise ValueError(f'No recognized property header before numeric row {row_index + 1}, sheet {sheet}.')
                record = {k: values[j] for k, j in column_map.items()}
                record['NO'] = strict_no(record['NO'])
                record['cohort'] = cohort_name(record.get('cohort')) or active_cohort
                record['source_sheet'] = sheet
                record['source_row'] = row_index + 1
                rows.append(record)
        frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f'No specimen rows parsed from {path}.')
    frame['NO'] = frame.NO.map(strict_no)
    if frame.NO.isna().any() or frame.NO.duplicated().any():
        raise ValueError('Specimen IDs must be numeric and unique across the workbook.')
    frame['NO'] = frame.NO.astype(int)
    if 'cohort' not in frame:
        frame['cohort'] = None
    assigned = []
    for _, row in frame.iterrows():
        group = cohort_name(row.get('cohort'))
        expected = 'validation' if row.NO in validation_ids else 'original'
        if group and group != expected:
            raise ValueError(f'Cohort conflict for specimen {row.NO}: table={group}, --validation-ids implies {expected}.')
        assigned.append(group or expected)
    frame['cohort'] = assigned
    for target in TARGETS:
        pairs = [parse_mean_std_cell(x) for x in frame[target]]
        frame[target] = [pair[0] for pair in pairs]
        error_col = target + '_std'
        if error_col in frame:
            supplied = pd.to_numeric(frame[error_col], errors='coerce')
            parsed = np.array([pair[1] for pair in pairs])
            both = supplied.notna() & np.isfinite(parsed)
            if not np.allclose(supplied[both], parsed[both]):
                raise ValueError(f'Conflicting error values for {target}.')
            frame[error_col] = supplied.where(supplied.notna(), parsed)
        else:
            frame[error_col] = [pair[1] for pair in pairs]
        if (frame[error_col].dropna() < 0).any():
            raise ValueError(f'Negative reported error for {target}.')
    for key in ['Equipment_ID', 'Powder_ID']:
        if key not in frame:
            frame[key] = ''
        frame[key] = frame[key].fillna('').astype(str).str.strip()
    for key in ['HIP_temperature_C', 'HIP_pressure_MPa', 'HIP_holding_time_h']:
        frame[key] = pd.to_numeric(frame[key], errors='coerce') if key in frame else np.nan
    lots = sorted(set(frame.Powder_ID) - {''})
    lot_codes = {lot: f'P{i + 1}' for i, lot in enumerate(lots)}
    frame['powder_code'] = frame.Powder_ID.map(lot_codes).fillna('NA')
    return frame.reset_index(drop=True)


def portable_parts(value):
    return [part for part in str(value).replace('\\', '/').split('/') if part]


def resolve_specimen(summary, summary_path, root, valid_ids):
    """Resolve a specimen without treating numeric image folders as specimens.

    This pipeline writes <root>/<specimen>/<image>/PSIM_summary.json.
    Only the first relative directory denotes the specimen in that layout.
    Legacy <root>/<specimen>__<image>/ outputs are also recognized. Explicit
    metadata and structured output paths must agree. Source-path fallback is
    used only when those stronger identifiers are missing; it never uses the
    image filename and never guesses between multiple specimen ancestors.
    """
    valid = set(map(int, valid_ids))
    evidence = {}
    for key in ('NO', 'specimen_id', 'sample_id'):
        value = summary.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        no = strict_no(value)
        if no not in valid:
            raise ValueError(f'Invalid specimen metadata {key}={value!r} in {summary_path}; '
                             'the ID must exist in the property table.')
        evidence[f'metadata.{key}'] = no

    # Normalize separators so saved Linux/Windows paths remain readable on
    # either platform. Relative output folders, not absolute numeric ancestors,
    # define the pipeline's specimen/image layout.
    output_parts = portable_parts(summary_path)
    root_parts = portable_parts(root)
    relative = (output_parts[len(root_parts):]
                if root_parts and output_parts[:len(root_parts)] == root_parts else [])
    if relative:
        first = relative[0]
        no = strict_no(first) if len(relative) >= 3 else None
        prefix = re.match(r'^(\d+)__.+$', first) if len(relative) >= 2 else None
        if no is not None:
            if no not in valid:
                raise ValueError(f'Output specimen folder {first!r} is not in the property table: {summary_path}')
            evidence['output.specimen_folder'] = no
        elif prefix:
            no = int(prefix[1])
            if no not in valid:
                raise ValueError(f'Output specimen prefix {no} is not in the property table: {summary_path}')
            evidence['output.specimen_prefix'] = no

    ids = set(evidence.values())
    if len(ids) > 1:
        raise ValueError(f'Conflicting specimen mapping in {summary_path}: {evidence}. '
                         'Correct the specimen metadata/output directory; image indices are not specimen IDs.')
    source = summary.get('image_path') or summary.get('source_image') or ''
    source_ids = set()
    for part in portable_parts(source)[:-1]:
        no = strict_no(part)
        if no in valid:
            source_ids.add(no)
    if ids:
        no = next(iter(ids))
        if source_ids and no not in source_ids:
            raise ValueError(f'Specimen/source-path conflict in {summary_path}: '
                             f'{evidence}; source specimen candidates={sorted(source_ids)}.')
        return no, source
    if len(source_ids) == 1:
        return next(iter(source_ids)), source
    raise ValueError(f'Ambiguous/missing specimen mapping in {summary_path}: '
                     f'source candidates={sorted(source_ids)}. Use explicit NO/specimen_id metadata '
                     'or <psim-output-root>/<specimen>/<image>/PSIM_summary.json. '
                     'A numeric image directory alone is not a specimen identifier.')


def configure_psim_model_paths(lia, args):
    """Override legacy module globals before any model or statistics loading."""
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    lia.CHECKPOINT_DIR = str(checkpoint)
    if hasattr(lia, 'DEFAULT_PAIR_CALIBRATOR_PATH'):
        lia.DEFAULT_PAIR_CALIBRATOR_PATH = os.environ.get(
            'PSIM_PAIR_CALIBRATOR', str(checkpoint / 'pair_calibrator.json'))
    required = ('unet_final.pth', 'ace_final.pth', 'mta_final.pth',
                'dit_final.pth', 'pqa_final.pth', 'unified_head_final.pth')
    missing = [str(checkpoint / name) for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise PSIMInputError('Missing model weights from the selected training run:\n' + '\n'.join(missing)
                             + '\nUse --checkpoint-dir to select the directory containing the complete compatible model.')
    explicit = getattr(args, 'fp_stats_path', None)
    candidates = [Path(explicit)] if explicit else [
        checkpoint / 'fingerprint_stats_2d_four_class.npz',
        checkpoint / 'fp_stats_2d_beta_lath_four_class_v3.npy',
    ]
    if not explicit and getattr(lia, 'FP_STATS_PATH', None):
        candidates.append(Path(lia.FP_STATS_PATH))
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise PSIMInputError('Training fingerprint statistics not found. Checked:\n'
                             + '\n'.join(map(str, candidates))
                             + '\nSpecify --fp-stats-path using the statistics used by this training run.')
    if source.suffix.lower() == '.npz':
        with np.load(source, allow_pickle=False) as data:
            mean, std = np.asarray(data['mean']), np.asarray(data['std'])
    else:
        mean, std = np.load(source, allow_pickle=False)
    mean, std = np.asarray(mean).reshape(-1), np.asarray(std).reshape(-1)
    if (mean.shape != (2,) or std.shape != (2,) or not np.isfinite([mean, std]).all()
            or (std < 0).any()):
        raise PSIMInputError(f'Invalid two-descriptor fingerprint statistics: {source}')
    if source.suffix.lower() == '.npz':
        # Legacy PSIM expects a two-row NPY. Preserve the training values exactly.
        adapter_root = Path(getattr(args, '_psim_metadata_dir', args.psim_output_dir / '_model_metadata'))
        adapter = adapter_root / 'fp_stats_2d.npy'
        adapter.parent.mkdir(parents=True, exist_ok=True)
        np.save(adapter, np.stack([mean, std]), allow_pickle=False)
        lia.FP_STATS_PATH = str(adapter)
    else:
        lia.FP_STATS_PATH = str(source)
    print(f'PSIM fingerprint statistics source: {source}')
    if source.parent != checkpoint:
        print('Using legacy statistics; ensure these are the statistics used by the selected training run.')


# Parallelism is across complete images. Each worker has one isolated CUDA
# device and its own PSIM module globals/model; the parent only aggregates.
PSIM_GPU_COUNT = 2
PSIM_MIN_FREE_GIB = 12.0


def _atomic_psim_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _probe_psim_gpu(token, minimum_free_gib):
    """Fresh interpreter: test one visible GPU without initializing parent CUDA."""
    code = '''import json, torch
try:
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device')
    torch.cuda.set_device(0)
    x = torch.ones((8, 8), device='cuda:0')
    if (x @ x).sum().item() != 512:
        raise RuntimeError('CUDA kernel probe failed')
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    print('PSIM_GPU_PROBE=' + json.dumps(dict(ok=True, free_gib=free/2**30,
        total_gib=total/2**30, name=torch.cuda.get_device_name(0))))
except Exception as exc:
    print('PSIM_GPU_PROBE=' + json.dumps(dict(ok=False, error=type(exc).__name__+': '+str(exc))))
    raise
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = token
    try:
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, timeout=90, check=False)
        records = [json.loads(line.split('=', 1)[1]) for line in result.stdout.splitlines()
                   if line.startswith('PSIM_GPU_PROBE=')]
        report = records[-1] if records else {'ok': False, 'error': result.stderr[-1200:]}
        if result.returncode or report.get('ok') is not True:
            report['ok'] = False
        if report.get('ok') and float(report['free_gib']) < minimum_free_gib:
            report.update(ok=False, error=f'Less than {minimum_free_gib:g} GiB free after CUDA initialization')
        return report
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def _select_psim_gpus(count, minimum_free_gib):
    """Honor launch visibility; use UUIDs so each child sees its GPU as cuda:0."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() in ('', '-1'):
        raise PSIMInputError('CUDA is explicitly hidden. Use --psim-gpu-count 1 for the original CPU/device route, '
                             'or launch with working CUDA_VISIBLE_DEVICES.')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PSIMInputError(f'Cannot inspect GPUs: {exc}. Use --psim-gpu-count 1 for the original device route.') from exc
    if result.returncode:
        raise PSIMInputError('nvidia-smi failed: ' + result.stderr.strip())
    inventory = []
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(',')]
        if len(parts) != 4 or not parts[1].startswith('GPU-'):
            continue
        try:
            index, free, total = int(parts[0]), float(parts[2])/1024, float(parts[3])/1024
        except ValueError:
            continue
        if not np.isfinite([free, total]).all() or free < 0 or total <= 0:
            continue
        inventory.append(dict(physical_index=index, token=parts[1], free_gib=free))
    if visible is None:
        candidates = inventory
    else:
        candidates = []
        for token in (part.strip() for part in visible.split(',')):
            matches = [row for row in inventory if token == str(row['physical_index']) or
                       (token.startswith('GPU-') and row['token'].startswith(token))]
            if token.startswith('MIG-'):
                # MIG UUIDs are already explicitly isolated; the runtime probe
                # determines actual available memory in that instance.
                candidates.append(dict(physical_index='MIG', token=token, free_gib=None))
            elif len(matches) == 1:
                candidates.append(matches[0])
            else:
                raise PSIMInputError(f'Cannot uniquely map CUDA_VISIBLE_DEVICES token {token!r}; '
                                     'use complete GPU UUIDs or nvidia-smi physical indices.')
    candidates = list({row['token']: row for row in candidates}.values())
    candidates.sort(key=lambda row: row['free_gib'] if row['free_gib'] is not None else float('inf'), reverse=True)
    selected = []
    for candidate in candidates:
        if candidate['free_gib'] is not None and candidate['free_gib'] < minimum_free_gib:
            continue
        print(f"[PSIM GPU probe] physical={candidate['physical_index']}; UUID={candidate['token']}", flush=True)
        report = _probe_psim_gpu(candidate['token'], minimum_free_gib)
        if not report.get('ok'):
            print(f"[PSIM GPU skipped] {candidate['token']}: {report.get('error')}", flush=True)
            continue
        selected.append({**candidate, **report})
        if len(selected) == count:
            break
    if not selected:
        raise PSIMInputError('No allowed GPU passed the CUDA/memory probe. Existing results are preserved. '
                             'Select working GPUs or use --psim-gpu-count 1 for the original device route.')
    if len(selected) < count:
        print(f'[PSIM GPUs] Only {len(selected)} usable device; pending jobs will run in one isolated worker.', flush=True)
    return selected


def _partition_psim_jobs(jobs, count):
    """Assign complete images once, balancing file-byte estimates of workload."""
    groups, loads = [[] for _ in range(count)], [0] * count
    weighted = [(max(1, path.stat().st_size), position, (int(no), path))
                for position, (no, path) in enumerate(jobs)]
    for size, position, job in sorted(weighted, key=lambda item: (-item[0], item[1])):
        slot = min(range(count), key=lambda i: (loads[i], i))
        groups[slot].append((position, job))
        loads[slot] += size
    return [[job for _, job in sorted(group)] for group in groups]


def _run_psim_job_batch(args, selected, valid_ids, all_jobs, progress=None):
    """Original PSIM inference, with process-local model globals and atomic metadata."""
    os.environ.setdefault('PSIM_PROJECT_ROOT', str(PROJECT_ROOT))
    pinned = getattr(args, '_worker_gpu_token', None)
    if pinned:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != pinned:
            raise PSIMInputError('Worker CUDA visibility changed before PSIM import.')
        import torch
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise PSIMInputError('An isolated PSIM worker must see exactly one GPU.')
        torch.cuda.set_device(0)
    try:
        lia = importlib.import_module(args.psim_module)
    except ImportError as exc:
        raise PSIMInputError(f'Automatic image quantification requires PSIM module {args.psim_module!r} '
                             f'and its dependencies: {exc}') from exc
    if pinned:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != pinned:
            raise PSIMInputError('The PSIM module changed worker CUDA_VISIBLE_DEVICES. '
                                 'Its GPU selector must preserve the launcher-provided visibility.')
        lia.DEVICE = torch.device('cuda:0')
        if hasattr(lia, 'device') and isinstance(lia.device, (str, torch.device)):
            lia.device = torch.device('cuda:0')
    print(f'PSIM module: {getattr(lia, "__file__", args.psim_module)}', flush=True)
    if hasattr(args, 'checkpoint_dir'):
        configure_psim_model_paths(lia, args)
    if hasattr(lia, 'CHECKPOINT_DIR'):
        print(f'PSIM checkpoints: {lia.CHECKPOINT_DIR}', flush=True)
    if not args.save_psim_visuals:
        suppress_psim_visualization_flags(lia)
    output_keys = []
    for no, path in all_jobs:
        safe = lia.safe_name(path.stem)
        if not safe or Path(safe).name != safe or '/' in safe or '\\' in safe or safe in ('.', '..'):
            raise PSIMInputError(f'Unsafe output name returned by PSIM for {path.name}: {safe!r}')
        output_keys.append((int(no), os.path.normcase(safe)))
    if len(set(output_keys)) != len(output_keys):
        raise ValueError('Image names collide after PSIM safe_name normalization; rename them before inference.')
    models = lia.load_models()
    engine = lia.PhysicsAwareSpatialInferenceModule(
        *models, patch_size=lia.PATCH_SIZE, stride=lia.STRIDE,
        sigma=lia.SIGMA, lambda_fp=lia.LAMBDA_FINGERPRINT)
    if pinned:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != pinned:
            raise PSIMInputError('PSIM changed the worker device mapping during model loading.')
        print(f'[PSIM worker] UUID={pinned}; logical cuda:0; images={len(selected)}', flush=True)
    rows = []
    if progress:
        progress(rows, None)
    for index, (no, path) in enumerate(selected, 1):
        specimen_output = args.psim_output_dir / str(no)
        specimen_output.mkdir(parents=True, exist_ok=True)
        lia.OUTPUT_DIR = str(specimen_output)
        destination = specimen_output / lia.safe_name(path.stem)
        marker = destination / '_pipeline_incomplete.json'
        print(f'PSIM {index}/{len(selected)}: specimen={no}, image={path.name}', flush=True)
        source_stat = path.stat()
        _atomic_psim_json(marker, dict(NO=int(no), image_path=str(path), pid=os.getpid(), status='running'))
        engine.run(str(path))
        if (path.stat().st_size, path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError(f'Source image changed during inference: {path}; rerun before plotting.')
        summary_path = destination / 'PSIM_summary.json'
        if not summary_path.exists():
            raise RuntimeError(f'External PSIM did not produce the expected isolated output: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        summary.update(NO=int(no), specimen_id=int(no), image_name=path.name, image_path=str(path))
        summary['source_file_signature'] = dict(size=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns)
        _atomic_psim_json(summary_path, summary)
        row = read_image_record(summary_path, args.psim_output_dir, valid_ids, allow_in_progress=True)
        validate_image_records(pd.DataFrame([row]), valid_ids)
        marker.unlink()
        rows.append(row)
        if progress:
            progress(rows, row)
    return rows


def _psim_worker_entry(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding='utf-8'))
    args = argparse.Namespace(**request['args'])
    for key in ('large_image_dir', 'psim_output_dir', 'checkpoint_dir', 'fp_stats_path'):
        if getattr(args, key, None) is not None:
            setattr(args, key, Path(getattr(args, key)))
    args._worker_gpu_token = request['gpu']['token']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args._worker_gpu_token:
        raise PSIMInputError('Worker must be launched by the parent with its assigned GPU UUID.')
    args._psim_metadata_dir = request_path.parent / f"worker_{request['worker_index']}_metadata"
    selected = [(int(no), Path(path)) for no, path in request['jobs']]
    all_jobs = [(int(no), Path(path)) for no, path in request['all_jobs']]
    status_path = request_path.with_name(request_path.stem + '_status.json')
    status = dict(worker_index=request['worker_index'], gpu=request['gpu'], pid=os.getpid(),
                  state='loading_models', total=len(selected), completed=0, rows=[])
    def save_progress(rows, row):
        status.update(state='running', rows=rows, completed=len(rows),
                      last_image=row['image_key'] if row else None)
        _atomic_psim_json(status_path, status)
    _atomic_psim_json(status_path, status)
    try:
        rows = _run_psim_job_batch(args, selected, request['valid_ids'], all_jobs, progress=save_progress)
        status.update(state='complete', rows=rows, completed=len(rows))
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        _atomic_psim_json(status_path, status)
        raise
    _atomic_psim_json(status_path, status)


def _stop_psim_workers(children):
    """Stop only child processes created by this run, preserving completed outputs."""
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def _run_psim_parallel(args, selected, valid_ids, all_jobs):
    gpus = _select_psim_gpus(min(int(args.psim_gpu_count), len(selected)), args.psim_min_free_gib)
    groups = _partition_psim_jobs(selected, len(gpus))
    run_dir = args.psim_output_dir / '_parallel_psim' / ('run_' + uuid.uuid4().hex)
    run_dir.mkdir(parents=True, exist_ok=False)
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    children, streams, requests = [], [], []
    status_paths, logs, returncodes = [], [], []
    try:
        for index, (gpu, jobs) in enumerate(zip(gpus, groups)):
            request_path = run_dir / f'worker_{index}.json'
            request = dict(worker_index=index, gpu=gpu, args=serialized_args,
                           valid_ids=list(map(int, valid_ids)),
                           jobs=[[no, str(path)] for no, path in jobs],
                           all_jobs=[[int(no), str(path)] for no, path in all_jobs])
            _atomic_psim_json(request_path, request)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu['token']
            env['PSIM_PROJECT_ROOT'] = env.get('PSIM_PROJECT_ROOT', str(PROJECT_ROOT))
            env['PYTHONUNBUFFERED'] = '1'
            # Preserve explicit user thread limits; otherwise bound CPU contention
            # from the two morphology/model processes.
            threads = str(max(1, min(8, (os.cpu_count() or 2) // len(gpus))))
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env.setdefault(key, threads)
            log_path = run_dir / f'worker_{index}.log'
            stream = log_path.open('w', encoding='utf-8')
            streams.append(stream)
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--_psim-worker', str(request_path.resolve())],
                                     env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
            children.append(child)
            requests.append(request)
            status_paths.append(request_path.with_name(request_path.stem + '_status.json'))
            logs.append(log_path)
            print(f"[PSIM parallel] worker {index}: physical GPU {gpu['physical_index']} -> cuda:0; "
                  f"{len(jobs)} images; log={log_path}", flush=True)
        last_progress, last_heartbeat = None, time.monotonic()
        while True:
            returncodes = [child.poll() for child in children]
            progress = []
            for index, path in enumerate(status_paths):
                try:
                    status = json.loads(path.read_text(encoding='utf-8'))
                    progress.append(f"worker{index}={status.get('completed', 0)}/{status.get('total', '?')} ({status.get('state')})")
                except (OSError, ValueError):
                    progress.append(f'worker{index}=starting')
            progress = '; '.join(progress)
            now = time.monotonic()
            if progress != last_progress or now - last_heartbeat >= 30:
                print('[PSIM parallel] ' + progress, flush=True)
                last_progress, last_heartbeat = progress, now
            if all(code is not None for code in returncodes):
                break
            time.sleep(1)
    except BaseException:
        _stop_psim_workers(children)
        print(f'[PSIM parallel] Interrupted; completed images retained. Worker logs: {run_dir}', flush=True)
        raise
    finally:
        for stream in streams:
            stream.close()
    failures, collected = [], {}
    for index, (code, status_path, request) in enumerate(zip(returncodes, status_paths, requests)):
        try:
            status = json.loads(status_path.read_text(encoding='utf-8'))
            if code != 0 or status.get('state') != 'complete':
                raise ValueError(status.get('error', f'worker exit={code}; incomplete status'))
            expected = {f'{no}/{Path(path).stem}': Path(path) for no, path in request['jobs']}
            returned = status.get('rows', [])
            if len(returned) != len(expected) or {row['image_key'] for row in returned} != set(expected):
                raise ValueError('Worker result list does not match its assigned images')
            # Re-read disk records using the unchanged numerical readers before
            # parent-level specimen aggregation. Do not trust a worker counter alone.
            for row in returned:
                actual = read_image_record(Path(row['summary_path']), args.psim_output_dir, valid_ids)
                if actual['image_key'] != row['image_key'] or not cached_source_matches(actual, expected[row['image_key']]):
                    raise ValueError('Worker output/source changed before collection')
                if actual['image_key'] in collected:
                    raise ValueError('The same image was returned by more than one worker')
                collected[actual['image_key']] = actual
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f'worker {index}: {exc}; log={logs[index]}')
    if failures:
        raise PSIMInputError('Parallel PSIM did not complete. Saved complete images remain reusable; '
                             'rerun without --force-psim. No partial-dataset plots were generated.\n' + '\n'.join(failures))
    print(f'[PSIM parallel] Completed {len(collected)} pending images; worker logs: {run_dir}', flush=True)
    return [collected[f'{int(no)}/{path.stem}'] for no, path in selected]



def run_psim(args, properties):
    """Quantify missing images by default; isolate repeated names by specimen."""
    if (getattr(args, 'skip_psim', False) or getattr(args, 'image_records_csv', None)
            or getattr(args, 'properties_only', False)):
        return
    jobs = []
    for no in sorted(properties.NO):
        folder = args.large_image_dir / str(no)
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing specimen image folder: {folder}')
        images = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                        key=lambda p: (int(p.stem) if p.stem.isdigit() else float('inf'), str(p)))
        if not images:
            raise FileNotFoundError(f'No images in {folder}')
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f'Duplicate image stems inside {folder}; rename these images before PSIM.')
        jobs.extend((int(no), path) for path in images)
    unmatched = sorted(p.name for p in args.large_image_dir.iterdir()
                       if p.is_dir() and strict_no(p.name) not in set(properties.NO))
    if unmatched:
        print(f'Folders not matched to Excel specimen IDs (excluded): {unmatched}')
    # Inspect saved records without importing the GPU module. An interrupted
    # run can resume, and a complete cache does not require model loading.
    cached = {}
    for summary_path in sorted(args.psim_output_dir.rglob('PSIM_summary.json')):
        try:
            row = read_image_record(summary_path, args.psim_output_dir, properties.NO)
            validate_image_records(pd.DataFrame([row]), properties.NO)
        except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile):
            continue
        cached.setdefault(row['image_key'], []).append(row)
    rows, pending = [], []
    for no, path in jobs:
        candidates = cached.get(f'{no}/{path.stem}', [])
        if len(candidates) > 1:
            raise PSIMInputError(f'Multiple cached outputs for {no}/{path.name}; use a separate --psim-output-dir to avoid ambiguous reuse.')
        row = candidates[0] if candidates else None
        if not args.force_psim and row and cached_source_matches(row, path):
            rows.append(row)
        else:
            pending.append((no, path))
    print(f'Excel-matched images: {len(jobs)}; reused: {len(rows)}; pending: {len(pending)}.')
    if not pending:
        return validate_image_records(pd.DataFrame(rows), properties.NO)
    selected = pending[:args.max_images] if args.max_images else pending
    if getattr(args, 'psim_gpu_count', PSIM_GPU_COUNT) == 2:
        rows.extend(_run_psim_parallel(args, selected, properties.NO, jobs))
    else:
        rows.extend(_run_psim_job_batch(args, selected, properties.NO, jobs))
    if len(selected) < len(pending):
        raise PSIMInputError(f'Quantified {len(selected)} images; {len(pending) - len(selected)} remain. '
                             'Results were saved. Rerun without --force-psim to resume; no partial-dataset plots were generated.')
    return validate_image_records(pd.DataFrame(rows), properties.NO)


def cached_source_matches(row, path):
    summary = json.loads(Path(row['summary_path']).read_text(encoding='utf-8-sig'))
    source = row['source_image']
    if source and portable_parts(source) != portable_parts(path):
        return False
    signature = summary.get('source_file_signature')
    stat = path.stat()
    if signature is not None:
        return signature == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    # Legacy records have no signature: only reuse when the source is no newer
    # than both saved outputs. --force-psim also refreshes unchanged images.
    return stat.st_mtime_ns <= min(Path(row[key]).stat().st_mtime_ns
                                  for key in ('summary_path', 'npz_path'))


def read_image_record(path, root, valid_ids, *, allow_in_progress=False):
    if not allow_in_progress and (path.parent / '_pipeline_incomplete.json').exists():
        raise ValueError(f'Interrupted/uncommitted PSIM image result: {path}; recompute this image before reuse.')
    summary = json.loads(path.read_text(encoding='utf-8-sig'))
    no, source = resolve_specimen(summary, path, root, valid_ids)
    archive = find_existing_psim_archive(path.parent)
    if archive is None:
        raise FileNotFoundError(f'Missing PSIM archive next to {path}')
    # Check that the archive is readable even when summary fields are sufficient.
    with np.load(archive, allow_pickle=False) as data:
        if not data.files:
            raise ValueError(f'Empty PSIM archive: {archive}')
    fractions = extract_hard_ratio_from_summary(summary) or extract_hard_ratio_from_npz(archive, summary)
    dark, width = extract_quantitative_morphology(summary, archive)
    image_name = portable_parts(source)[-1] if source else summary.get('image_name', path.parent.name)
    row = dict(NO=no, image_name=image_name, image_key=f'{no}/{Path(image_name).stem}',
               summary_path=str(path), npz_path=str(archive), source_image=source,
               dark_contrast_area_fraction=dark, mean_lath_width_um=width)
    row.update({f'hard_{name}': value for name, value in fractions.items()})
    return row


def collect_image_records(psim_output_dir, valid_ids):
    root = Path(psim_output_dir)
    paths = sorted(root.rglob('PSIM_summary.json'))
    if not paths:
        raise FileNotFoundError(f'No PSIM_summary.json under {root}; supply --image-records-csv or run --force-psim.')
    rows = [read_image_record(path, root, valid_ids) for path in paths]
    return validate_image_records(pd.DataFrame(rows), valid_ids)


def validate_image_records(frame, valid_ids):
    images = normalize_contrast_frame(frame)
    required = {'NO', 'image_name', *FEATURES}
    if not required.issubset(images):
        raise ValueError(f'Image table lacks {sorted(required - set(images))}')
    images['NO'] = images.NO.map(strict_no)
    if images.NO.isna().any() or not images.NO.isin(valid_ids).all():
        raise ValueError('Image NO must match a specimen in the property table; image indices are not specimen IDs.')
    images['NO'] = images.NO.astype(int)
    if images.image_name.isna().any() or images.image_name.astype(str).str.strip().eq('').any():
        raise ValueError('Every image needs an image_name.')
    images['image_key'] = [f'{no}/{Path(portable_parts(name)[-1]).stem}' for no, name in zip(images.NO, images.image_name)]
    for col in FEATURES:
        images[col] = pd.to_numeric(images[col], errors='raise')
    if not np.isfinite(images[FEATURES].to_numpy(float)).all():
        raise ValueError('Incomplete/nonfinite morphology measurements; do not silently drop fields or images.')
    if not images[DARK_CONTRAST_FIELD].between(0,1).all() or (images.mean_lath_width_um <= 0).any():
        raise ValueError('Dark area must be 0..1 and width must be positive micrometres.')
    for key, group in images.groupby('image_key'):
        if not np.allclose(group[FEATURES], group[FEATURES].iloc[0], rtol=1e-10, atol=1e-12):
            raise ValueError(f'Conflicting duplicate image: {key}')
    images = images.drop_duplicates('image_key').copy()
    fractions = images[HARD_COLS].to_numpy(float)
    if (fractions < 0).any() or not np.allclose(fractions.sum(axis=1),1,atol=1e-5):
        raise ValueError('Hard class fractions must be nonnegative and sum to one.')
    # Retain the supplied pipeline's existing per-image 3% presence policy.
    filtered = [enforce_four_class_fractions(dict(zip(CLASS_NAMES, row)), 'image table') for row in fractions]
    images[HARD_COLS] = [[row[name] for name in CLASS_NAMES] for row in filtered]
    return images.reset_index(drop=True)


# User-specified morphology prior, applied after raw PSIM measurement.
# This is not evidence that the excluded tissues are physically absent.
WIDMANSTATTEN_PRIOR_REVISION = 'specimen-w-plus-acw-over-60-percent-v4'
WIDMANSTATTEN_PRIOR_COMPATIBLE_REVISIONS = (
    'both-w-acw-over-10-percent-spatial-ties-v3',
    'both-w-acw-over-10-percent-pixel-prior-v2',
)
WIDMANSTATTEN_PRIOR_MIN_FRACTION = 0.60
WIDMANSTATTEN_PRIOR_COMPARE_ATOL = 1e-12
WIDMANSTATTEN_PRIOR_RULE = (
    'Specimen NO: equal-image mean of ORIGINAL saved hard-map W+ACW fractions >60%; '
    'then reassign BW/DC pixels in ALL images of that specimen using local PSIM evidence. '
    'Exactly 60% does not trigger (roundoff tolerance 1e-12); no per-class minimum.'
)
WIDMANSTATTEN_PRIOR_LEGACY_RULE = (
    'W > 10% AND ACW > 10% in one saved hard map (strictly greater, not combined); '
    'reassign only BW/DC pixels to W/ACW using local PSIM evidence; no rank condition.'
)
WIDMANSTATTEN_PRIOR_TIE_POLICY = (
    'Exact vote/probability ties use the nearest ORIGINAL W/ACW hard-map pixel '
    'by Euclidean pixel distance. Equal nearest distances use the smallest '
    '(row, column) coordinate. This is a deterministic boundary convention, '
    'not additional classifier evidence. If a whole image has no original W/ACW '
    'seed, only its non-tied local-evidence assignments may supply fallback seeds; '
    'resolved tie pixels are never propagated as seeds.'
)


def _prior_digest(path):
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _prior_trigger(fractions):
    fractions=np.asarray(fractions,dtype=float)
    if (fractions.shape!=(4,) or not np.isfinite(fractions).all()
            or (fractions<0).any() or (fractions>1).any()
            or not np.isclose(fractions.sum(),1.,rtol=0,atol=1e-5)):
        raise ValueError('Invalid raw four-class vector for morphology prior.')
    combined=float(fractions[2]+fractions[3])
    return bool(combined>WIDMANSTATTEN_PRIOR_MIN_FRACTION
                and not np.isclose(combined,WIDMANSTATTEN_PRIOR_MIN_FRACTION,
                                   rtol=0,atol=WIDMANSTATTEN_PRIOR_COMPARE_ATOL))


def _prior_map_fractions(labels):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.size==0 or not np.isin(labels,[0,1,2,3]).all():
        raise PSIMInputError('Prior requires a complete four-class hard map; unknown pixels are not silently relabelled.')
    return np.bincount(labels.astype(np.int64).ravel(),minlength=4)/labels.size


def _prior_resolve_spatial_ties(labels, flat_indices, fallback_labels=None):
    """Resolve sparse exact ties using fixed original labels, without propagation.

    The expanding search checks a distance bound before stopping: a candidate
    outside a radius-r square is at least r+1 pixels away. Strict inequality
    also ensures all equally near pixels are inside the searched square.
    Thus boundaries and search order do not change the nearest-seed result.
    No SciPy/GPU dependency or global target fraction is needed.
    """
    height,width=labels.shape
    seed_source='original_hard_labels'
    if not np.isin(labels,[2,3]).any():
        if fallback_labels is None or fallback_labels.shape!=labels.shape or not np.isin(fallback_labels,[2,3]).any():
            raise PSIMInputError('No original or non-tied local-evidence W/ACW seed exists; cannot invent an assignment.')
        labels=fallback_labels
        seed_source='non_tied_local_evidence_assignments; no original allowed seeds'
    chosen=np.empty(len(flat_indices),dtype=np.uint8)
    equal_class_ties=0
    maximum_distance=0.
    samples=[]
    for index,flat in enumerate(flat_indices):
        row,column=divmod(int(flat),width)
        radius=1
        while True:
            top,bottom=max(0,row-radius),min(height,row+radius+1)
            left,right=max(0,column-radius),min(width,column+radius+1)
            window=labels[top:bottom,left:right]
            ys,xs=np.nonzero((window==2)|(window==3))
            if len(ys):
                ys=ys.astype(np.int64)+top;xs=xs.astype(np.int64)+left
                distances=(ys-row)**2+(xs-column)**2
                minimum=int(distances.min())
                full_window=top==0 and bottom==height and left==0 and right==width
                if minimum<(radius+1)**2 or full_window:
                    nearest=np.flatnonzero(distances==minimum)
                    # np.nonzero is row-major: first nearest has the smallest
                    # (row, column), independent of the class numeric index.
                    first=int(nearest[0])
                    source_row,source_column=int(ys[first]),int(xs[first])
                    label=int(labels[source_row,source_column])
                    different_classes=bool(np.unique(labels[ys[nearest],xs[nearest]]).size>1)
                    equal_class_ties+=int(different_classes)
                    distance=float(np.sqrt(minimum))
                    maximum_distance=max(maximum_distance,distance)
                    chosen[index]=label
                    if len(samples)<20:
                        samples.append(dict(row=row,column=column,source_row=source_row,
                            source_column=source_column,assigned_class=CLASS_NAMES[label],
                            distance_pixels=distance,equidistant_classes=different_classes))
                    break
            if top==0 and bottom==height and left==0 and right==width:
                raise PSIMInputError('No original W/ACW seed exists for spatial tie resolution.')
            radius*=2
    return chosen,dict(spatial_tie_pixel_count=int(len(flat_indices)),
        spatial_tie_area_fraction=float(len(flat_indices)/labels.size),
        spatial_equidistant_class_tie_pixel_count=equal_class_ties,
        spatial_tie_max_distance_pixels=maximum_distance,
        spatial_tie_locations_sample=samples,spatial_tie_policy=WIDMANSTATTEN_PRIOR_TIE_POLICY,
        spatial_tie_seed_source=seed_source)


def _prior_reassign(labels, probability, votes=None):
    """Apply an already-authorized specimen constraint; no per-image gate."""
    labels=np.asarray(labels)
    _prior_map_fractions(labels)
    changed=(labels==0)|(labels==1)
    result=labels.astype(np.uint8,copy=True)
    if not changed.any():
        return result,dict(changed_pixel_count=0,changed_area_fraction=0.,vote_tie_pixel_count=0,
                           probability_decision_pixel_count=0,evidence='no_reassignment')
    if probability is None or probability.shape!=labels.shape+(4,):
        raise PSIMInputError('Prior requires full-resolution prob_field; summary-only deletion/renormalization is forbidden.')
    pair=np.asarray(probability[changed,2:4],dtype=float)
    if not np.isfinite(pair).all() or (pair<0).any() or (pair>1+1e-5).any():
        raise PSIMInputError('Invalid local PSIM probabilities.')
    if (pair.sum(axis=1)<=0).any():
        raise PSIMInputError('Some excluded pixels have zero evidence for both allowed classes; cannot assign reliably.')
    if votes is not None:
        if votes.shape!=probability.shape:raise PSIMInputError('class_vote_field dimensions mismatch.')
        scores=np.asarray(votes[changed,2:4],dtype=float)
        if not np.isfinite(scores).all() or (scores<0).any():raise PSIMInputError('Invalid class-vote evidence.')
        tied=np.isclose(scores[:,0],scores[:,1],rtol=0,atol=1e-7)
        decisions=np.argmax(scores,axis=1)
        decisions[tied]=np.argmax(pair[tied],axis=1)
        evidence='class_vote_field; probability breaks vote ties'
    else:
        tied=np.ones(len(pair),dtype=bool)
        decisions=np.argmax(pair,axis=1)
        evidence='prob_field; probability mode or class_vote_field unavailable'
    unresolved=tied & (pair[:,0]==pair[:,1])
    spatial_audit={}
    if unresolved.any():
        changed_indices=np.flatnonzero(changed)
        tie_indices=changed_indices[unresolved]
        fallback=None
        if not np.isin(labels,[2,3]).any():
            fallback=labels.astype(np.uint8,copy=True)
            fallback.ravel()[changed_indices[~unresolved]]=(decisions[~unresolved]+2).astype(np.uint8)
        chosen,spatial_audit=_prior_resolve_spatial_ties(labels,tie_indices,fallback)
        decisions[unresolved]=chosen.astype(np.int64)-2
        evidence+='; exact unresolved ties use audited spatial boundary convention'
    result[changed]=(decisions+2).astype(np.uint8)
    assert np.isin(result,[2,3]).all()
    return result,dict(changed_pixel_count=int(changed.sum()),changed_area_fraction=float(changed.mean()),
                       vote_tie_pixel_count=int(tied.sum()) if votes is not None else 0,
                       probability_comparison_pixel_count=int(tied.sum()),
                       probability_decision_pixel_count=int((tied & ~unresolved).sum()),
                       evidence=evidence,**spatial_audit)


def _prior_cache_directory(out,archive,source_hash,revision):
    import hashlib
    legacy=revision in WIDMANSTATTEN_PRIOR_COMPATIBLE_REVISIONS
    identity=dict(source=str(archive.resolve()),sha256=source_hash,revision=revision,
                  threshold=.10 if legacy else WIDMANSTATTEN_PRIOR_MIN_FRACTION,
                  rule=WIDMANSTATTEN_PRIOR_LEGACY_RULE if legacy else WIDMANSTATTEN_PRIOR_RULE)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    return out/'class_prior_maps'/key


def _prior_original_maps(images):
    """Read saved hard labels, not model inference or previously constrained CSV ratios."""
    if images.empty or not images.index.is_unique:
        raise PSIMInputError('Specimen constraint requires nonempty, uniquely indexed image records.')
    stale=[col for col in images.columns if col.startswith(('class_prior_','raw_hard_','original_map_hard_'))
           or col=='constrained_class_archive']
    baseline=images.drop(columns=stale).copy()
    baseline=baseline.astype({col:float for col in HARD_COLS})
    records=[];signatures={}
    for ordinal,(index,row) in enumerate(images.iterrows(),1):
        archive=Path(str(row.get('npz_path',''))).expanduser()
        if not archive.is_file():
            raise PSIMInputError(f'{row.NO}/{row.image_name}: original PSIM archive required to evaluate '
                'the specimen 60% rule; provide class_id_map and local probability results, not only a ratio CSV.')
        before=archive.stat()
        with np.load(archive,allow_pickle=False) as data:
            if 'class_names' not in data or list(data['class_names'].astype(str))!=list(CLASS_NAMES):
                raise PSIMInputError(f'Class order not verified: {archive}')
            if 'class_id_map' not in data:
                raise PSIMInputError(f'{archive} lacks class_id_map for the specimen 60% rule.')
            raw=data['class_id_map']
            fractions=_prior_map_fractions(raw)
            pixel_count=int(raw.size)
        after=archive.stat()
        signature=(after.st_size,after.st_mtime_ns)
        if signature!=(before.st_size,before.st_mtime_ns):
            raise PSIMInputError(f'Source PSIM archive changed while reading the specimen baseline: {archive}')
        signatures[index]=signature
        records.append(dict(index=index,NO=int(row.NO),pixel_count=pixel_count,
                            **dict(zip(HARD_COLS,fractions))))
        # Retain the supplied pipeline's 3% display policy for non-triggered
        # specimens. The trigger itself uses unrenormalized saved-map areas.
        displayed=enforce_four_class_fractions(dict(zip(CLASS_NAMES,fractions)),'original saved hard map')
        for col,name in zip(HARD_COLS,CLASS_NAMES):baseline.at[index,col]=displayed[name]
        if ordinal%40==0 or ordinal==len(images):
            print(f'[Specimen prior baseline] Saved hard maps {ordinal}/{len(images)}; no model inference.',flush=True)
    return baseline,pd.DataFrame(records).set_index('index'),signatures


def apply_widmanstatten_prior(images,args):
    mode=getattr(args,'widmanstatten_prior','specimen-sum-over-60')
    if mode=='off':return images
    if mode!='specimen-sum-over-60':raise ValueError(f'Unknown morphology prior mode: {mode}')
    out=Path(args.analysis_output_dir)
    out.mkdir(parents=True,exist_ok=True)
    print('[Class prior] Specimen-level original mean (W + ACW) > 60%; '
          'apply to ALL images of a triggered specimen. Exactly 60% does not trigger.',flush=True)
    images,original_maps,source_signatures=_prior_original_maps(images)
    specimen_means=original_maps.groupby('NO',sort=True)[HARD_COLS].mean()
    specimen_reports={}
    for no,mean in specimen_means.iterrows():
        values=mean.to_numpy(float)
        trigger=_prior_trigger(values)
        specimen_reports[int(no)]=dict(NO=int(no),image_count=int((original_maps.NO==no).sum()),
            original_mean_map_fractions=values.tolist(),original_mean_combined_fraction=float(values[2:].sum()),
            triggered=trigger,weighting='equal images; matches specimen aggregation',
            trigger_source='original class_id_map before this or previous priors')
        print(f'[Specimen prior] NO={int(no)}: original W+ACW={100*values[2:].sum():.6f}%; '
              f'{"TRIGGERED: process every image" if trigger else "not triggered"}',flush=True)
    result=images.copy()
    for col in HARD_COLS:result['raw_'+col]=result[col]
    for col in HARD_COLS:result['original_map_'+col]=original_maps[col]
    result['class_prior_scope']='specimen'
    result['class_prior_specimen_raw_combined_fraction']=result.NO.map(
        {no:report['original_mean_combined_fraction'] for no,report in specimen_reports.items()})
    result['class_prior_specimen_triggered']=result.NO.map(
        {no:report['triggered'] for no,report in specimen_reports.items()})
    result['class_prior_status']='not_triggered'
    result['class_prior_changed_area_fraction']=0.
    result['class_prior_tie_pixel_count']=0
    result['class_prior_tie_area_fraction']=0.
    result['constrained_class_archive']=''
    reports=[]
    for index,row in images.iterrows():
        fractions=row[HARD_COLS].to_numpy(dtype=float)
        specimen=specimen_reports[int(row.NO)]
        record=dict(NO=int(row.NO),image_name=str(row.image_name),raw_reported_fractions=fractions.tolist(),
                    original_map_fractions=original_maps.loc[index,HARD_COLS].to_numpy(float).tolist(),
                    specimen_raw_combined_fraction=specimen['original_mean_combined_fraction'],
                    specimen_triggered=specimen['triggered'],status='not_triggered')
        if not specimen['triggered']:
            reports.append(record);continue
        archive=Path(str(row.get('npz_path',''))).expanduser()
        if not archive.is_file():
            raise PSIMInputError(f'{row.image_name}: original PSIM archive required for pixel-level reassignment. '
                                 'Use saved PSIM results with prob_field and class_id_map, not only a CSV.')
        source_hash=_prior_digest(archive)
        stat=archive.stat()
        if (stat.st_size,stat.st_mtime_ns)!=source_signatures[index]:
            raise PSIMInputError(f'Source changed after specimen trigger calculation: {archive}; rerun from originals.')
        dest=_prior_cache_directory(out,archive,source_hash,WIDMANSTATTEN_PRIOR_REVISION)
        map_path=dest/'constrained_class_map.npz';audit_path=dest/'constraint_audit.json'
        reused=False
        # Old completed maps have the same pixel-assignment rule. Reuse is
        # permitted ONLY AFTER this run's specimen trigger has been computed.
        cache_roots=list(dict.fromkeys([out,out.parent/'specimen_cohort_associations_widmanstatten_prior']))
        for root,revision in ((root,rev) for root in cache_roots
                              for rev in (WIDMANSTATTEN_PRIOR_REVISION,*WIDMANSTATTEN_PRIOR_COMPATIBLE_REVISIONS)):
            cached_dir=_prior_cache_directory(root,archive,source_hash,revision)
            cached_map=cached_dir/'constrained_class_map.npz'
            cached_audit=cached_dir/'constraint_audit.json'
            if not cached_map.is_file() or not cached_audit.is_file():continue
            try:
                candidate=json.loads(cached_audit.read_text(encoding='utf-8'))
                expected_rule=WIDMANSTATTEN_PRIOR_LEGACY_RULE if revision in WIDMANSTATTEN_PRIOR_COMPATIBLE_REVISIONS else WIDMANSTATTEN_PRIOR_RULE
                valid=(candidate['source_sha256']==source_hash and candidate['map_sha256']==_prior_digest(cached_map)
                       and candidate['revision']==revision and candidate['rule']==expected_rule)
                if revision.endswith('v2'):
                    valid=valid and candidate.get('spatial_tie_pixel_count',0)==0
                if valid:
                    audit=candidate;map_path=cached_map;audit_path=cached_audit;reused=True
                    break
            except (OSError,ValueError,KeyError,TypeError):pass
        if not reused:
            with np.load(archive,allow_pickle=False) as data:
                if 'class_names' not in data or list(data['class_names'].astype(str))!=list(CLASS_NAMES):
                    raise PSIMInputError(f'Class order not verified: {archive}')
                if 'class_id_map' not in data:
                    raise PSIMInputError(f'{archive} lacks class_id_map; cannot enforce a pixel-level prior.')
                raw=data['class_id_map']
                raw_counts=_prior_map_fractions(raw)
                if not np.array_equal(raw_counts,original_maps.loc[index,HARD_COLS].to_numpy(float)):
                    raise PSIMInputError(f'{row.NO}/{row.image_name}: hard map differs from specimen-trigger baseline.')
                probability=data['prob_field'] if 'prob_field' in data else None
                votes=data['class_vote_field'] if 'class_vote_field' in data else None
                if 'spatial_config_json' in data:
                    config=json.loads(str(np.asarray(data['spatial_config_json']).item()))
                    if config.get('area_mode')=='probability':votes=None
                try:
                    hard,evidence=_prior_reassign(raw,probability,votes)
                except PSIMInputError as exc:
                    raise PSIMInputError(f'{row.NO}/{row.image_name}: {exc}') from exc
                proportions=np.bincount(hard.ravel(),minlength=4)/hard.size
            if _prior_digest(archive)!=source_hash:raise PSIMInputError('Source PSIM archive changed during processing.')
            dest.mkdir(parents=True,exist_ok=True)
            temp=map_path.with_suffix('.partial')
            with temp.open('wb') as stream:
                np.savez_compressed(stream,class_id_map=hard,original_class_id_map=raw,
                    hard_ratio=proportions,original_hard_ratio=raw_counts,class_names=np.array(CLASS_NAMES))
            os.replace(temp,map_path)
            audit=dict(revision=WIDMANSTATTEN_PRIOR_REVISION,source_archive=str(archive.resolve()),
                source_sha256=source_hash,map_sha256=_prior_digest(map_path),
                original_map_fractions=raw_counts.tolist(),constrained_map_fractions=proportions.tolist(),
                rule=WIDMANSTATTEN_PRIOR_RULE,threshold_fraction=WIDMANSTATTEN_PRIOR_MIN_FRACTION,
                trigger_basis='current run specimen audit; pixel mapping reusable only for triggered specimens',
                interpretation='User-provided morphology prior, not independently validated absence of excluded tissues.',
                physical_measurements='Unchanged; retained in original PSIM archive.',**evidence)
            _atomic_psim_json(audit_path,audit)
            del raw,hard,probability,votes
        fractions=np.asarray(audit['constrained_map_fractions'],dtype=float)
        if (fractions.shape!=(4,) or not np.isfinite(fractions).all() or (fractions<0).any()
                or np.any(fractions[:2]!=0) or not np.isclose(fractions.sum(),1)):
            raise PSIMInputError('Invalid constrained class fractions.')
        # Do not apply a second presence threshold to the constrained map.
        # Reported areas must agree exactly with this saved final hard map.
        for col,value in zip(HARD_COLS,fractions):result.at[index,col]=value
        status='applied' if audit['changed_area_fraction']>0 else 'already_allowed_pair'
        result.at[index,'class_prior_status']=status
        result.at[index,'class_prior_changed_area_fraction']=audit['changed_area_fraction']
        tie_count=int(audit.get('spatial_tie_pixel_count',0))
        tie_area=float(audit.get('spatial_tie_area_fraction',0.))
        result.at[index,'class_prior_tie_pixel_count']=tie_count
        result.at[index,'class_prior_tie_area_fraction']=tie_area
        result.at[index,'constrained_class_archive']=str(map_path.resolve())
        record.update(status=status,fractions=fractions.tolist(),audit=str(audit_path.resolve()),
                      changed_area_fraction=audit['changed_area_fraction'],reused=reused,
                      cache_revision=audit['revision'],spatial_tie_pixel_count=tie_count,
                      spatial_tie_area_fraction=tie_area)
        reports.append(record)
        if tie_count:
            print(f'[Class prior tie] {row.NO}/{row.image_name}: resolved {tie_count} pixel(s) '
                  f'({100*tie_area:.8f}% of image) by audited spatial boundary convention; '
                  f'equidistant class ties={audit.get("spatial_equidistant_class_tie_pixel_count",0)}.',flush=True)
        print(f'[Class prior] {row.NO}/{row.image_name}: W={100*fractions[2]:.2f}%, ACW={100*fractions[3]:.2f}%; '
              f'reassigned={100*audit["changed_area_fraction"]:.2f}% (prior-constrained); '
              f'cache={"reused" if reused else "new"}',flush=True)
    # Enforce the requested outcome on every triggered specimen, before plotting.
    # This also guards against accidentally loading an unrelated cache artifact.
    for no,report in specimen_reports.items():
        selected=result.loc[result.NO.eq(no),HARD_COLS]
        if report['triggered'] and (selected[HARD_COLS[:2]].to_numpy(float)!=0).any():
            raise PSIMInputError(f'Specimen {no}: excluded class remains after specimen constraint; no final plot written.')
        report['final_mean_fractions']=selected.mean().to_numpy(float).tolist()
    images.to_csv(out/'image_level_psim_records_unconstrained.csv',index=False,encoding='utf-8-sig')
    specimen_rows=[]
    for report in specimen_reports.values():
        item={k:report[k] for k in ('NO','image_count','original_mean_combined_fraction','triggered','weighting','trigger_source')}
        item.update(dict(zip(['original_mean_'+col for col in HARD_COLS],report['original_mean_map_fractions'])))
        item.update(dict(zip(['final_mean_'+col for col in HARD_COLS],report['final_mean_fractions'])))
        specimen_rows.append(item)
    pd.DataFrame(specimen_rows).to_csv(out/'specimen_class_prior_audit.csv',index=False,encoding='utf-8-sig')
    _atomic_psim_json(out/'class_prior_audit.json',dict(revision=WIDMANSTATTEN_PRIOR_REVISION,
        mode=mode,rule=WIDMANSTATTEN_PRIOR_RULE,threshold_fraction=WIDMANSTATTEN_PRIOR_MIN_FRACTION,
        spatial_tie_policy=WIDMANSTATTEN_PRIOR_TIE_POLICY,
        specimens=list(specimen_reports.values()),images=reports,
        comparison_roundoff_tolerance=WIDMANSTATTEN_PRIOR_COMPARE_ATOL,
        scope='per-specimen; every image of a triggered specimen excludes BW/DC; non-triggered specimens retain original reporting'))
    return result


def aggregate_specimens(images, properties):
    missing = sorted(set(properties.NO) - set(images.NO))
    if missing:
        raise ValueError(f'Missing PSIM data for specimens {missing}. Complete all requested specimens before plotting.')
    grouped = images.groupby('NO', sort=True)
    summary = grouped[FEATURES].mean()
    summary['image_count'] = grouped.size()
    for col in FEATURES[:2]:
        summary[col + '_image_sd'] = grouped[col].std(ddof=1)
    merged = summary.reset_index().merge(properties, on='NO', validate='one_to_one')
    return merged


class PSIMInputError(RuntimeError):
    """Actionable missing/ambiguous PSIM inputs without a Python traceback."""


def psim_candidate_directories(args):
    roots = [args.psim_output_dir]
    for parent in (args.psim_output_dir.parent, DIFFUSION_DIR):
        if parent.is_dir():
            roots.extend(sorted((p for p in parent.iterdir() if p.is_dir()
                                 and p.name.lower().startswith('psim')), key=lambda p: p.name))
    return list(dict.fromkeys(roots))


def psim_input_help(args):
    base = [sys.executable, str(Path(__file__).resolve()),
            '--large-image-dir', str(args.large_image_dir),
            '--property-xlsx', str(args.property_xlsx),
            '--analysis-output-dir', str(args.analysis_output_dir)]
    inference = base + ['--psim-output-dir', str(args.psim_output_dir),
                        '--psim-module', args.psim_module,
                        '--validation-ids', *map(str, args.validation_ids), '--force-psim']
    if getattr(args, 'checkpoint_dir', None):
        inference += ['--checkpoint-dir', str(args.checkpoint_dir)]
    if getattr(args, 'fp_stats_path', None):
        inference += ['--fp-stats-path', str(args.fp_stats_path)]
    return ('\nIf results already exist elsewhere, pass --psim-output-dir /actual/result/directory.\n'
            'If these micrographs have not been analyzed, run PSIM first (requires model weights and its GPU environment):\n'
            + shlex.join(inference) + '\n'
            'To plot only the measured mechanical properties without morphology:\n'
            + shlex.join(base + ['--properties-only']) + '\n'
            'Creating an empty psim_results folder will not generate morphology measurements.')


def load_psim_records(args, properties):
    if args.image_records_csv:
        return validate_image_records(pd.read_csv(args.image_records_csv), properties.NO)
    requested = args.psim_output_dir
    has_requested = requested.is_dir() and next(requested.rglob('PSIM_summary.json'), None) is not None
    if has_requested:
        print(f'Reuse PSIM results: {requested}')
        return collect_image_records(requested, properties.NO)
    if args.psim_output_explicit or args.force_psim:
        raise PSIMInputError(f'No PSIM_summary.json in the specified result directory: {requested}.'
                             + psim_input_help(args))
    candidates = psim_candidate_directories(args)
    usable, rejected = [], []
    for path in candidates:
        if not path.is_dir() or next(path.rglob('PSIM_summary.json'), None) is None:
            continue
        try:
            records = collect_image_records(path, properties.NO)
            missing = sorted(set(properties.NO) - set(records.NO))
            if missing:
                raise ValueError(f'Missing specimens: {missing}')
            # A partially populated folder must not win discovery over a
            # complete folder solely because its name sorts first.
            mismatch = [r for r in image_coverage_rows(args, properties, records)
                        if r['missing_images'] or r['extra_images']]
            if mismatch:
                raise ValueError('Image coverage differs from current raw folders: '
                                 + ', '.join(str(r['NO']) for r in mismatch))
            usable.append((path, records))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            rejected.append(f'{path}: {exc}')
    if len(usable) == 1:
        args.psim_output_dir, records = usable[0]
        print(f'No results in default directory {requested}.')
        print(f'Automatically selected compatible PSIM results: {args.psim_output_dir}')
        args.analysis_output_dir.mkdir(parents=True, exist_ok=True)
        (args.analysis_output_dir/'psim_discovery.json').write_text(json.dumps({
            'requested': str(requested), 'selected': str(args.psim_output_dir),
            'checked': [str(p) for p in candidates], 'rejected': rejected,
            'matched_specimens': sorted(map(int, records.NO.unique())),
        }, indent=2), encoding='utf-8')
        return records
    if len(usable) > 1:
        raise PSIMInputError('Multiple complete PSIM directories found. Select one with --psim-output-dir; '
                             'results will not be mixed or chosen by modification time:\n'
                             + '\n'.join(str(path) for path, _ in usable))
    details = '\n'.join(str(p) for p in candidates)
    invalid = '\nRejected records:\n' + '\n'.join(rejected) if rejected else ''
    raise PSIMInputError('No compatible PSIM morphology results were found. Checked:\n'
                         + details + invalid + psim_input_help(args))


def image_coverage_rows(args, properties, images):
    """Check current raw folders when available, without guessing missing records."""
    rows=[]
    for no in properties.NO:
        folder=args.large_image_dir/str(no)
        matched=images[images.NO.eq(no)]
        expected={f'{no}/{p.stem}' for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES} if folder.is_dir() else None
        actual=set(matched.image_key)
        missing=sorted(expected-actual) if expected is not None else []
        extra=sorted(actual-expected) if expected is not None else []
        rows.append(dict(NO=int(no),raw_folder_available=expected is not None,
                         raw_image_count=len(expected) if expected is not None else None,
                         psim_image_count=len(actual),missing_images=missing,extra_images=extra))
    return rows


def audit_image_coverage(args, properties, images):
    rows=image_coverage_rows(args,properties,images)
    args.analysis_output_dir.mkdir(parents=True,exist_ok=True)
    (args.analysis_output_dir/'image_coverage_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    mismatches=[r for r in rows if r['missing_images'] or r['extra_images']]
    if mismatches:
        raise ValueError('Raw-image/PSIM coverage mismatch. Inspect image_coverage_audit.json before aggregating a partial or stale dataset.')


def ordered(frame, target, args):
    valid = frame[frame[target].notna()]
    key = {'property':target, 'specimen':'NO', 'hip':'HIP_temperature_C'}[args.sort_by]
    return valid.sort_values([key, 'NO'] if key!='NO' else ['NO']).reset_index(drop=True)


def tick_label(row, args):
    if args.axis_label=='specimen':
        return str(int(row.NO))
    temp = f'{row.HIP_temperature_C:g}' if pd.notna(row.HIP_temperature_C) else 'HIP NA'
    lot = row.Powder_ID if args.axis_label=='powder-full' else row.powder_code
    return f'{int(row.NO)}\n{lot}\n{temp}'


def cohort_handles(frame):
    return [Line2D([],[],color=COHORT_COLORS[g],marker=COHORT_MARKERS[g],linestyle='none',
                   markerfacecolor=COHORT_COLORS[g] if g=='original' else 'white',markersize=4,
                   label=f'{COHORT_LABELS[g]} (n = {int(frame.cohort.eq(g).sum())})')
            for g in COHORT_LABELS if frame.cohort.eq(g).any()]


def class_handles():
    return [Patch(facecolor=c,edgecolor='white',hatch=h,label=l)
            for c,h,l in zip(CLASS_COLORS,CLASS_HATCHES,CLASS_LABELS)]


def draw_property(ax, data, target, args, limits=None):
    x = np.arange(len(data))
    mixed = data.cohort.nunique()>1
    for i,row in data.iterrows():
        group=row.cohort; error=row[target+'_std']
        if mixed and group=='validation':
            ax.axvspan(i-.46,i+.46,color='#F3EEE9',zorder=0)
        ax.errorbar(i,row[target],yerr=error if np.isfinite(error) else None,
                    fmt=COHORT_MARKERS[group],color=COHORT_COLORS[group],
                    mfc=COHORT_COLORS[group] if group=='original' else 'white',
                    mec=COHORT_COLORS[group],ms=4,capsize=2,elinewidth=.8,zorder=3)
    ax.set_xticks(x,[tick_label(row,args) for _,row in data.iterrows()])
    ax.set_xlim(-.65,max(len(data)-.35,.65))
    ax.set_ylabel(TARGET_LABELS[target])
    if limits:ax.set_ylim(limits)
    ax.tick_params(axis='x',labelsize=6.7,pad=4)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.spines[['top','right']].set_visible(False)


def save_figure(fig, path, args):
    fig.canvas.draw()
    for ext in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+ext),dpi=args.dpi,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def y_limits(frame):
    result={}
    for target in TARGETS:
        values=frame[target].to_numpy(float)
        errors=frame[target+'_std'].fillna(0).to_numpy(float)
        valid=np.isfinite(values)
        if not valid.any():continue
        low=float(np.min((values-errors)[valid])); high=float(np.max((values+errors)[valid]))
        pad=max((high-low)*.13, .5 if target=='Elongation' else 2)
        result[target]=(low-pad,high+pad)
    return result


def plot_comparison(frame, targets, out, stem, title, args, limits, morphology=True):
    n=len(targets)
    fig=plt.figure(figsize=(7.2 if n>1 else 4.1,4.3 if morphology else 3.2))
    gs=fig.add_gridspec(2 if morphology else 1,n,left=.12 if n==1 else .09,right=.985,
                        bottom=.22 if morphology else .28,top=.78 if morphology else .77,
                        hspace=.08,wspace=.46,height_ratios=[1,1] if morphology else [1])
    for col,target in enumerate(targets):
        data=ordered(frame,target,args)
        if morphology:
            top=fig.add_subplot(gs[0,col]); bottom=fig.add_subplot(gs[1,col],sharex=top)
            base=np.zeros(len(data))
            for feature,color,hatch in zip(HARD_COLS,CLASS_COLORS,CLASS_HATCHES):
                vals=data[feature].to_numpy(float)*100
                top.bar(np.arange(len(data)),vals,bottom=base,color=color,edgecolor='white',linewidth=.4,hatch=hatch,width=.75)
                base+=vals
            top.set_ylim(0,100);top.set_yticks([0,50,100]);top.set_ylabel('Class fraction (%)')
            top.tick_params(axis='x',bottom=False,labelbottom=False)
            top.spines[['top','right']].set_visible(False)
            top.text(-.14,.98,chr(97+col),transform=top.transAxes,fontweight='bold',fontsize=9)
        else:
            bottom=fig.add_subplot(gs[0,col])
        draw_property(bottom,data,target,args,limits.get(target))
        data.to_csv(out/f'{stem}_{target}_source.csv',index=False,encoding='utf-8-sig')
    fig.suptitle(title,y=.985,fontsize=10)
    if morphology:
        fig.legend(handles=class_handles(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,fontsize=6.5)
    fig.legend(handles=cohort_handles(frame),loc='upper center',bbox_to_anchor=(.5,.835 if morphology else .91),ncol=2,fontsize=7)
    xlabel='Specimen ID' if args.axis_label=='specimen' else 'Specimen ID / powder lot / HIP temperature (\u00b0C)'
    fig.text(.5,.06,xlabel,ha='center',fontsize=7)
    if args.axis_label=='powder-hip':
        codes=frame[['powder_code','Powder_ID']].drop_duplicates().sort_values('powder_code')
        entries=[f'{r.powder_code}: {r.Powder_ID}' for _,r in codes.iterrows()]
        lines=[';  '.join(entries)] if n>1 else [';  '.join(entries[j:j+2]) for j in range(0,len(entries),2)]
        fig.text(.5,.002,'\n'.join(lines),ha='center',va='bottom',fontsize=5.8)
    save_figure(fig,out/stem,args)


def correlations(frame):
    rows=[]
    for target in TARGETS:
        for feature in FEATURES:
            paired=frame[[feature,target]].replace([np.inf,-np.inf],np.nan).dropna()
            status='descriptive_only';rho=np.nan
            if len(paired)<3:status='insufficient_specimens'
            elif paired[feature].nunique()<2 or paired[target].nunique()<2:status='constant_values'
            else:rho=float(paired[feature].rank().corr(paired[target].rank()))
            rows.append(dict(target=target,feature=feature,n_specimens=len(paired),spearman_rho=rho,status=status))
    return pd.DataFrame(rows)


def feature_label(feature):
    if feature==DARK_CONTRAST_FIELD:return 'Dark-contrast area fraction (%)'
    if feature=='mean_lath_width_um':return 'Mean \u03b1-lath width (\u00b5m)'
    return CLASS_LABELS[HARD_COLS.index(feature)]+' fraction (%)'


def plot_correlations(corr, out, args, title):
    matrix=corr.pivot(index='feature',columns='target',values='spearman_rho').reindex(index=FEATURES,columns=TARGETS)
    counts=corr.pivot(index='feature',columns='target',values='n_specimens').reindex(index=FEATURES,columns=TARGETS)
    fig,ax=plt.subplots(figsize=(6.2,3.7),layout='constrained')
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#EEEEEE')
    im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set_yticks(range(6),[feature_label(f).replace(' (%)','').replace(' (\u00b5m)','') for f in FEATURES])
    ax.set_xticks(range(3),['Elongation','Tensile strength','Fatigue strength'])
    for y in range(6):
        for x in range(3):
            value=matrix.iloc[y,x];label=f'{value:.2f}' if np.isfinite(value) else 'NA'
            ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(value) and abs(value)>.65 else '#222222')
    ax.set_title(title);fig.colorbar(im,ax=ax,label='Spearman \u03c1',shrink=.85)
    save_figure(fig,out/'rank_association',args)
    matrix.to_csv(out/'spearman_matrix.csv',encoding='utf-8-sig')
    counts.to_csv(out/'spearman_sample_counts.csv',encoding='utf-8-sig')


def plot_associations(frame,target,out,args,title):
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.1))
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.84,wspace=.47,hspace=.58)
    for feature,ax in zip(FEATURES,axes.flat):
        for _,row in frame.iterrows():
            if not np.isfinite(row[target]):continue
            scale=1 if feature=='mean_lath_width_um' else 100
            group=row.cohort;error=row[target+'_std']
            ax.errorbar(row[feature]*scale,row[target],yerr=error if np.isfinite(error) else None,
                        fmt=COHORT_MARKERS[group],color=COHORT_COLORS[group],
                        mfc=COHORT_COLORS[group] if group=='original' else 'white',ms=4,capsize=2,elinewidth=.6)
        label=feature_label(feature).replace(' area fraction',' area\nfraction').replace(' fraction','\nfraction')
        ax.set_xlabel(label,fontsize=7);ax.set_ylabel(TARGET_LABELS[target],fontsize=7)
        ax.tick_params(labelsize=6.5);ax.margins(x=.20,y=.22)
        ax.spines[['top','right']].set_visible(False)
    fig.canvas.draw()
    for feature,ax in zip(FEATURES,axes.flat):
        valid=frame[frame[target].notna()]
        scale=1 if feature=='mean_lath_width_um' else 100
        annotate_association_ids(ax,valid[feature].to_numpy()*scale,valid[target].to_numpy(),valid.NO.to_numpy())
    fig.suptitle(title,fontsize=10,y=.99)
    fig.legend(handles=cohort_handles(frame),loc='upper center',bbox_to_anchor=(.5,.945),ncol=2)
    save_figure(fig,out/f'association_{target}',args)


def run_analysis(args, properties, images=None):
    out=args.analysis_output_dir;out.mkdir(parents=True,exist_ok=True)
    properties.to_csv(out/'mechanical_property_parsed.csv',index=False,encoding='utf-8-sig')
    properties[['NO','cohort','Powder_ID','powder_code','HIP_temperature_C','HIP_pressure_MPa','HIP_holding_time_h','Equipment_ID']].to_csv(out/'specimen_metadata.csv',index=False,encoding='utf-8-sig')
    morphology=images is not None
    frame=aggregate_specimens(images,properties) if morphology else properties
    if morphology:
        contrast_export_table(images).to_csv(out/'image_level_psim_records.csv',index=False,encoding='utf-8-sig')
        contrast_export_table(frame).to_csv(out/'specimen_morphology_catalog.csv',index=False,encoding='utf-8-sig')
    limits=y_limits(frame)
    for cohort in ('original','validation','combined'):
        selected=frame if cohort=='combined' else frame[frame.cohort.eq(cohort)]
        if selected.empty:continue
        dest=out/cohort;dest.mkdir(exist_ok=True)
        title=COHORT_LABELS.get(cohort,'Original and validation experiments')
        if morphology and getattr(args,'widmanstatten_prior','off')!='off':
            title=cohort.capitalize()+' experiments (prior-constrained)'
        prefix='morphology' if morphology else 'measured_properties'
        plot_comparison(selected,TARGETS[:2],dest,prefix+'_tensile',title,args,limits,morphology)
        plot_comparison(selected,TARGETS[2:],dest,prefix+'_fatigue',title,args,limits,morphology)
        if morphology:
            for target in TARGETS:
                plot_associations(selected,target,dest,args,title)
            corr=correlations(selected)
            if cohort in ('original', 'combined'):
                corr.to_csv(dest/'descriptive_spearman.csv',index=False,encoding='utf-8-sig')
                correlation_title = ('Combined experiments (original + validation)\nDescriptive Spearman correlations'
                                     if cohort=='combined' else title+': descriptive correlations')
                if cohort=='combined' and getattr(args,'widmanstatten_prior','off')!='off':
                    correlation_title+='; prior-constrained'
                plot_correlations(corr,dest,args,correlation_title)
            if cohort=='validation':
                (dest/'validation_note.txt').write_text('Validation experiments are displayed separately. No correlation or predictive-validation metric is estimated from the two validation specimens.\n',encoding='utf-8')
    caption=(
        'Specimen-level comparisons of measured mechanical properties and, where available, PSIM morphology-class area fractions. '
        'Original experiments: filled circles; validation experiments: open diamonds. Shaded columns in the combined figure denote validation specimens. '
        'Specimens are ordered independently for each endpoint using the requested sorting rule. '
        'Powder codes identify powder lots, not measured chemical composition. Equipment is retained in source tables but omitted from plot coordinates. '
        f'Error bars: {args.error_kind} experimental +/- values from the input table; missing error values are not imputed. '
        'Morphology is averaged over unique images within each specimen using equal image weights; comparable image areas and representative sampling are assumed. '
        'Image counts are not independent mechanical replicates. The original per-image 3% hard-class presence filter is retained; no threshold is applied to the specimen average. '
        'Combined Spearman correlations pool original and validation specimens and recompute ranks over all pairwise-complete specimens; subgroup coefficients are not averaged. '
        'Original-only correlations are retained as a separate descriptive comparison. Pairwise specimen counts are exported in spearman_sample_counts.csv. '
        'Pooled associations may reflect cohort differences and do not establish predictive accuracy or causality.\n')
    if morphology and getattr(args,'widmanstatten_prior','off')!='off':
        caption+='User-provided specimen-level morphology prior: compute the equal-image mean W+ACW fraction from ORIGINAL saved hard maps, before any class prior or additional reporting threshold. When the sum is strictly greater than 60% (1e-12 roundoff tolerance), reassign BW/DC pixels in EVERY image of that specimen by local PSIM evidence. No individual-class minimum is required. Exact score ties use the nearest original W/ACW hard label; if an image has no original allowed seeds, its non-tied local-evidence assignments supply fallback seeds. Equidistant classes use fixed (row, column) order. Such boundary assignments and their area are audited and do not add classifier evidence. This is not independent proof of tissue absence. Non-triggered specimens retain the original 3% reporting policy; no additional threshold is applied after reassignment. Raw outputs and physical measurements are preserved.\n'
    (out/'figure_captions.txt').write_text(caption,encoding='utf-8')
    manifest=dict(schema=CONTRACT_VERSION,mode='properties_only' if not morphology else 'measured_association_only',
                  original_ids=properties.loc[properties.cohort.eq('original'),'NO'].tolist(),
                  validation_ids=properties.loc[properties.cohort.eq('validation'),'NO'].tolist(),
                  endpoints=list(TARGETS),directions_used=False,equipment_in_axes=False,
                  morphology_class_prior=getattr(args,'widmanstatten_prior','off') if morphology else 'off',
                  morphology_prior_rule=WIDMANSTATTEN_PRIOR_RULE if morphology and getattr(args,'widmanstatten_prior','off')!='off' else None,
                  morphology_prior_revision=WIDMANSTATTEN_PRIOR_REVISION,
                  morphology_prior_scope='specimen' if morphology and getattr(args,'widmanstatten_prior','off')!='off' else None,
                  primary_correlation_cohort='combined',pooled_correlations=bool(morphology),
                  correlation_cohorts=['original','combined'] if morphology else [],
                  correlation_method='pairwise_complete_specimen_level_spearman',
                  error_bar_kind=args.error_kind,property_model_fitted=False,
                  independent_unit='specimen',image_count=0 if images is None else len(images),
                  source_property_file=str(args.property_xlsx),axis_label=args.axis_label,
                  psim_output_directory=str(getattr(args,'psim_output_dir','')),
                  actual_figures_created=sorted(str(p.relative_to(out)) for p in out.rglob('*.png')))
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    write_contrast_terminology(out)
    print(f'Completed: {out}; original n={sum(frame.cohort.eq("original"))}, validation n={sum(frame.cohort.eq("validation"))}.')
    return frame


def main():
    args=parse_args();configure_plot_style()
    properties=read_mechanical_property(args.property_xlsx,args.validation_ids)
    if args.properties_only:
        run_analysis(args,properties)
        return
    images=run_psim(args,properties)
    if images is None:
        images=load_psim_records(args,properties)
    audit_image_coverage(args,properties,images)
    images=apply_widmanstatten_prior(images,args)
    run_analysis(args,properties,images)


if __name__=='__main__':
    try:
        if len(sys.argv) == 3 and sys.argv[1] == '--_psim-worker':
            _psim_worker_entry(sys.argv[2])
        else:
            main()
    except PSIMInputError as exc:
        print(f'\nPSIM input required:\n{exc}', file=sys.stderr)
        sys.exit(2)
