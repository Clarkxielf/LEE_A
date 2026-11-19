

import argparse
import os, math, logging, copy, json, unicodedata
import hashlib
import inspect
import subprocess
import sys
from functools import wraps

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Some PyTorch builds route eager bmm backward through a native Triton
# override, whose first call compiles a driver shim requiring Python.h.
# Use the standard ATen kernels by default on this deployment. Set this
# BEFORE gpu_selector too: helper modules may import torch themselves.
# Subprocess ablation runs inherit the same setting. An explicit external
# TORCH_DISABLE_NATIVE_JIT=0 opts back in once the build environment is ready.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
from gpu_selector import configure_automatic_gpu_selection

configure_automatic_gpu_selection()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import ssl
import certifi

_original_create_default_context = ssl.create_default_context

def _create_default_context_with_certifi(*args, **kwargs):
    if not any(kwargs.get(k) for k in ("cafile", "capath", "cadata")):
        kwargs["cafile"] = certifi.where()
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = _create_default_context_with_certifi
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt, uniform_filter
from skimage.morphology import (
    medial_axis,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)
from skimage.filters import frangi
import cv2
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import WeightedRandomSampler
import random
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logger = get_logger(__name__, log_level="INFO")

ABLATION_MODES = ("full", "no_mtfa", "no_fp")
ABLATION_DISPLAY_NAMES = {
    "full": "Full model",
    "no_mtfa": "w/o MTFA",
    "no_fp": "w/o FP pathway",
}

FP_DIM = 2
CONNECTIVITY_DIM = 3
CONNECTIVITY_NAMES = [
    "connectivity_length",
    "fragmentation_degree",
    "orientation_consistency",
]
FINGERPRINT_NAMES = [
    "dark_contrast_area_fraction",
    "mean_lath_width",
]
FINGERPRINT_LABELS = [
    "Dark-contrast area fraction",
    "Mean lath width",
]
SEQUENTIAL_CMAP = "Blues"

# Compact paper palette derived from the supplied confusion-matrix and latent-
# manifold references.  Colour is secondary to position, labels and markers.
_CLASS_COLORS = ("#1E88E5", "#00A86B", "#FF0052", "#8A2BE2")
_FEATURE_COLORS = ("#356FAE", "#D58B55")
VISUALIZATION_SCHEMA_VERSION = "four-class-evaluation-v2"


def _publication_rcparams():
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }


def _save_visualization_manifest(output_dir, class_names):
    """Persist every style-independent setting needed to redraw figures."""
    rcparams = _publication_rcparams()
    manifest = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "class_names": [str(name) for name in class_names],
        "fingerprint_names": list(FINGERPRINT_NAMES),
        "class_colors": list(_CLASS_COLORS),
        "feature_colors": list(_FEATURE_COLORS),
        "sequential_cmap": SEQUENTIAL_CMAP,
        "publication_rcparams": rcparams,
        "figure_export": {
            "png_dpi": 600,
            "svg_enabled": True,
            "pdf_enabled": False,
            "bbox_inches": "tight",
            "facecolor": "white",
        },
        "random_seeds": {
            "fingerprint_jitter": 42,
            "global_fingerprint_jitter": 17,
            "tsne": 42,
        },
        "tsne": {
            "n_components": 2,
            "perplexity_rule": "min(30, n_samples - 1)",
            "init": "random",
            "learning_rate": "auto",
        },
        "figure_specs": {
            "confusion_matrix": {"figsize_inches": [5.7, 4.85], "normalized": True, "value_decimals": 2},
            "per_class_performance": {"figsize_inches": [5.2, 3.35], "value_decimals": 2},
            "per_class_accuracy": {"figsize_inches": [5.6, 3.55], "ylim": [0.0, 1.05]},
            "fingerprint_boxplot": {"width_per_feature_inches": 3.5, "point_alpha": 0.24},
            "global_fingerprint": {"figsize_inches": [4.55, 3.25], "violin_alpha": 0.40},
            "latent_manifold": {"figsize_inches": [5.7, 4.35], "point_size": 10, "point_alpha": 0.88},
            "morphology_field_examples": {"row_height_inches": 1.48, "columns": 3},
            "diffusion_trajectory": {"row_count": 3, "max_steps": 5},
        },
        "data_files": {
            "confusion_matrix": "classification_metrics.npz",
            "labels": "labels.npy",
            "predictions": "predictions.npy",
            "fingerprints": "fingerprints.npy",
            "fingerprint_tsne_coordinates": "fingerprint_tsne_coordinates.npy",
            "ace_embeddings": "ace_embeddings.npy",
            "ace_tsne_coordinates": "latent_manifold_coordinates.npy",
            "morphology_examples": "morphology_examples.npz",
            "diffusion_trajectory": "diffusion_morphology_trajectory.npz",
        },
    }
    with open(os.path.join(output_dir, "visualization_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)


def _save_morphology_examples_data(morphology_examples, output_dir):
    """Save source images and morphology fields used by the example panel."""
    if not morphology_examples:
        return
    arrays = {}
    class_ids = []
    for class_id, value in sorted(morphology_examples.items()):
        if value is None or len(value) != 2:
            continue
        image, field = value
        arrays[f"image_{int(class_id)}"] = np.asarray(image, dtype=np.uint8)
        arrays[f"field_{int(class_id)}"] = np.asarray(field, dtype=np.float32)
        class_ids.append(int(class_id))
    if arrays:
        arrays["class_ids"] = np.asarray(class_ids, dtype=np.int64)
        fields = [arrays[f"field_{class_id}"] for class_id in class_ids]
        channel_limits = []
        for channel in (0, 1):
            pooled = np.concatenate([field[channel].ravel() for field in fields])
            low, high = np.percentile(pooled, [1, 99])
            channel_limits.append((float(low), float(max(high, low + 1e-8))))
        arrays["channel_limits"] = np.asarray(channel_limits, dtype=np.float32)
        np.savez_compressed(os.path.join(output_dir, "morphology_examples.npz"), **arrays)



def _save_publication_figure(fig, save_path):
    """Save a high-resolution PNG and an editable SVG; never create PDF."""
    save_path = os.fspath(save_path)
    root, _ = os.path.splitext(save_path)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(root + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _wrap_class_label(name):
    text = str(name).replace("_", " ")
    if text.lower().startswith("alpha coarsened"):
        return "Alpha-coarsened\nWidmanstatten"
    return text


def _save_per_class_performance(cm, class_names, output_dir):
    """Persist precision, recall, F1 and support in comparison-ready formats."""
    cm = np.asarray(cm, dtype=np.int64)
    labels = np.arange(len(class_names), dtype=np.int64)
    predicted_support = cm.sum(axis=0)
    true_support = cm.sum(axis=1)
    precision = np.divide(np.diag(cm), predicted_support, out=np.zeros(len(labels), dtype=float), where=predicted_support > 0)
    recall = np.divide(np.diag(cm), true_support, out=np.zeros(len(labels), dtype=float), where=true_support > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall,
                   out=np.zeros(len(labels), dtype=float), where=(precision + recall) > 0)
    support = true_support.astype(np.int64)
    total = max(int(cm.sum()), 1)
    accuracy = float(np.trace(cm) / total)
    metrics_frame = pd.DataFrame({
        "class_id": labels,
        "class_name": [str(name) for name in class_names],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    metrics_frame.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
    summary = {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "total_support": int(support.sum()),
        "per_class": {
            str(class_names[index]): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }
    with open(os.path.join(output_dir, "per_class_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(output_dir, "classification_metrics.npz"),
        class_names=np.asarray(class_names), precision=precision, recall=recall,
        f1=f1, support=support, confusion_matrix=cm, accuracy=np.asarray(accuracy),
    )

    matrix = np.vstack([precision, recall, f1])
    plt.rcParams.update(_publication_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.35), dpi=300)
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if value >= 0.55 else "#1A1A1A")
    ax.set_xticks(labels, [_wrap_class_label(name) for name in class_names])
    ax.set_yticks(np.arange(3), ["Precision", "Recall", "F1"])
    ax.tick_params(axis="both", length=0, pad=5)
    ax.set_xlabel("Class", labelpad=8)
    ax.set_ylabel("Metric", labelpad=8)
    ax.set_title("Per-class performance", pad=10, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Score", labelpad=6)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    fig.tight_layout()
    _save_publication_figure(fig, os.path.join(output_dir, "per_class_performance.png"))


def plot_diffusion_trajectory(trajectory, save_path, max_steps=5):
    """Save intermediate morphology diagnostics across diffusion timesteps.

    This panel is a model-inspection figure rather than a quality score: the
    input micrograph is shown alongside the learned morphology-response and
    local-lath-width fields at early, middle, and late denoising states.  Shared
    colour limits make changes across timesteps comparable.
    """
    steps = list(trajectory["step"])
    if not steps:
        return
    indices = np.linspace(0, len(steps) - 1, min(max_steps, len(steps)), dtype=int)
    ncols = len(indices)
    plt.rcParams.update(_publication_rcparams())
    fig, axes = plt.subplots(
        3, ncols, figsize=(2.15 * ncols, 6.2), squeeze=False,
        gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    )
    line_values = np.concatenate([np.asarray(trajectory["line"][i]).ravel() for i in indices])
    width_values = np.concatenate([np.asarray(trajectory["width"][i]).ravel() for i in indices])
    line_limits = np.percentile(line_values, [2, 98])
    width_limits = np.percentile(width_values, [2, 98])
    for column, index in enumerate(indices):
        panels = [
            (trajectory["image"][index], "Input micrograph", None),
            (trajectory["line"][index], "Morphology response", SEQUENTIAL_CMAP),
            (trajectory["width"][index], "Local lath width", SEQUENTIAL_CMAP),
        ]
        for row, (panel, title, cmap) in enumerate(panels):
            ax = axes[row, column]
            kwargs = {"cmap": cmap} if cmap else {}
            if row == 1:
                kwargs.update(vmin=line_limits[0], vmax=line_limits[1])
            elif row == 2:
                kwargs.update(vmin=width_limits[0], vmax=width_limits[1])
            ax.imshow(panel, **kwargs)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t = {steps[index]}", pad=4, fontweight="bold")
            if column == 0:
                ax.text(-0.10, 0.5, title, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=7.5)
    fig.text(0.5, 0.995, "Diffusion trajectory: morphology diagnostics",
             ha="center", va="top", fontsize=9, fontweight="bold")
    _save_publication_figure(fig, save_path)


@torch.no_grad()
def collect_diffusion_trajectory(
    image_np,
    target_field,
    latent,
    cond_emb,
    noise_scheduler,
    unet_wrapper,
    morphology_head,
    mid_features_list,
    timesteps_matrix,
    device,
):
    """Collect the diffusion trajectory used by the retained trajectory figure."""
    if isinstance(target_field, torch.Tensor):
        target_field_np = target_field.detach().float().cpu().numpy()
    else:
        target_field_np = np.asarray(target_field, dtype=np.float32)
    target_shape = target_field_np.shape[-2:]
    viz_features = list(mid_features_list)
    viz_timesteps = [int(timesteps_matrix[0, k].item()) for k in range(len(viz_features))]
    if len(viz_features) < 3:
        viz_features = []
        viz_timesteps = []
        for timestep_value in (100, 500, 900):
            timestep = torch.tensor([timestep_value], device=device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(latent, torch.randn_like(latent), timestep)
            _, feature = unet_wrapper(noisy, timestep, cond_emb)
            viz_features.append(feature.detach())
            viz_timesteps.append(timestep_value)

    line_maps, width_maps = [], []
    for feature in viz_features[:3]:
        predicted = morphology_head(feature.detach())
        predicted = F.interpolate(predicted, size=target_shape, mode="bilinear", align_corners=False)
        predicted = predicted[0].float().cpu().numpy()
        line_maps.append(predicted[0])
        width_maps.append(predicted[1])
    count = min(3, len(line_maps))
    return {
        "step": viz_timesteps[:count],
        "image": [image_np] * count,
        "line": line_maps[:count],
        "width": width_maps[:count],
    }


def _display_class_names(class_names):
    return [str(name).replace("_", " ") for name in class_names]


def _save_confusion_matrix_figure(cm, class_names, save_path):
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = cm.astype(np.float64) / np.maximum(row_sum, 1.0)
    labels = [_wrap_class_label(name) for name in class_names]

    plt.rcParams.update(_publication_rcparams())
    fig, ax = plt.subplots(figsize=(5.7, 4.85), dpi=300)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9.0,
                color="white" if value >= 0.55 else "#1A1A1A",
            )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=0, fontsize=7.2)
    ax.set_yticks(np.arange(len(labels)), labels, rotation=0, fontsize=7.2)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xlabel("Predicted label", labelpad=7, fontsize=8.5)
    ax.set_ylabel("True label", labelpad=7, fontsize=8.5)
    ax.set_aspect("equal")
    fig.text(0.28, 0.962, "Test confusion matrix", ha="left", va="top",
             fontsize=10.0, fontweight="bold")
    fig.text(0.88, 0.962, f"Accuracy {np.trace(cm) / max(cm.sum(), 1):.3f}",
             ha="right", va="top", fontsize=7.8, color="#4B4B4B")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Fraction", labelpad=5, fontsize=8.0)
    colorbar.ax.tick_params(length=2.5, labelsize=7.0)
    fig.subplots_adjust(left=0.28, right=0.88, bottom=0.20, top=0.89)
    _save_publication_figure(fig, save_path)


def _save_embedding_plot(embedding, labels, class_names, save_path, title):
    """Reference-style latent manifold with an unobstructed shared legend."""
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    plt.rcParams.update(_publication_rcparams())
    fig, ax = plt.subplots(figsize=(5.7, 4.35), dpi=300)
    for class_id, name in enumerate(class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.88,
            color=_CLASS_COLORS[class_id % len(_CLASS_COLORS)],
            label=str(name).replace("_", " "),
            edgecolors="none",
            rasterized=True,
        )
    ax.set_xlabel("Dimension 1", labelpad=4)
    ax.set_ylabel("Dimension 2", labelpad=4)
    ax.set_title(title, pad=7, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        markerscale=1.1,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.2,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.91)
    _save_publication_figure(fig, save_path)


def _save_fingerprint_tsne(
    fingerprints,
    labels,
    class_names,
    save_path,
    coordinates_path=None,
):
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if fingerprints.ndim != 2 or fingerprints.shape[0] < 3:
        return None
    perplexity = min(30, fingerprints.shape[0] - 1)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    ).fit_transform(fingerprints)

    if coordinates_path is not None:
        np.save(coordinates_path, np.asarray(embedding, dtype=np.float32))

    _save_embedding_plot(
        embedding,
        labels,
        class_names,
        save_path,
        "Fingerprint manifold",
    )
    return embedding


def _save_diffusion_trajectory_data(trajectory, output_dir):
    """Persist the arrays behind the diffusion-trajectory figure."""
    if not trajectory or not trajectory.get("step"):
        return
    try:
        np.savez_compressed(
            os.path.join(output_dir, "diffusion_morphology_trajectory.npz"),
            step=np.asarray(trajectory["step"], dtype=np.int64),
            image=np.asarray(trajectory["image"], dtype=np.uint8),
            line=np.asarray(trajectory["line"], dtype=np.float32),
            width=np.asarray(trajectory["width"], dtype=np.float32),
        )
    except (TypeError, ValueError):
        # Keep figure generation robust if a future trajectory contains
        # variable-size arrays; the PNG remains available in that case.
        np.savez_compressed(
            os.path.join(output_dir, "diffusion_morphology_trajectory.npz"),
            step=np.asarray(trajectory["step"], dtype=np.int64),
            image=np.asarray(trajectory["image"], dtype=object),
            line=np.asarray(trajectory["line"], dtype=object),
            width=np.asarray(trajectory["width"], dtype=object),
        )


def _save_fingerprint_boxplot(fingerprints, labels, class_names, save_path):
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if fingerprints.ndim != 2 or fingerprints.shape[0] == 0:
        return

    display_names = _display_class_names(class_names)
    plt.rcParams.update(_publication_rcparams())
    fig, axes = plt.subplots(
        1,
        fingerprints.shape[1],
        figsize=(3.5 * fingerprints.shape[1], 3.15),
        squeeze=False,
        dpi=300,
    )
    rng = np.random.default_rng(42)
    for dimension, ax in enumerate(axes[0]):
        positions = []
        tick_labels = []
        for class_id, name in enumerate(display_names):
            selected = fingerprints[labels == class_id, dimension]
            if selected.size == 0:
                continue
            position = class_id + 1
            color = _CLASS_COLORS[class_id % len(_CLASS_COLORS)]
            plotted = selected
            if plotted.size > 240:
                plotted = rng.choice(plotted, size=240, replace=False)
            jitter = rng.normal(0.0, 0.055, size=plotted.size)
            ax.scatter(
                position + jitter,
                plotted,
                s=7,
                color=color,
                alpha=0.24,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            q1, median, q3 = np.percentile(selected, [25, 50, 75])
            ax.plot(
                [position, position],
                [q1, q3],
                color=color,
                linewidth=2.8,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                [position],
                [median],
                s=28,
                marker="D",
                color="white",
                edgecolor=color,
                linewidth=1.25,
                zorder=4,
            )
            positions.append(position)
            tick_labels.append(name)
        if positions:
            ax.set_xticks(positions, [_wrap_class_label(label) for label in tick_labels], fontsize=6.8)
        title = FINGERPRINT_LABELS[dimension]
        ax.set_title(title.capitalize(), fontweight="bold", pad=6)
        ax.set_ylabel("Standardized value", labelpad=4)
        ax.axhline(0.0, color="#9A9A9A", linewidth=0.75, linestyle="--", zorder=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#B8B8B8")
        ax.spines["bottom"].set_color("#B8B8B8")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.88, wspace=0.22)
    _save_publication_figure(fig, save_path)


def _save_global_fingerprint(fingerprints, save_path):
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    if fingerprints.ndim != 2 or fingerprints.shape[0] == 0:
        return
    plt.rcParams.update(_publication_rcparams())
    fig, ax = plt.subplots(figsize=(4.55, 3.25), dpi=300)
    positions = np.arange(1, fingerprints.shape[1] + 1)
    values = [fingerprints[:, index] for index in range(fingerprints.shape[1])]
    violin = ax.violinplot(
        values,
        positions=positions,
        widths=0.58,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.28,
    )
    for index, body in enumerate(violin["bodies"]):
        color = _FEATURE_COLORS[index % len(_FEATURE_COLORS)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.8)
        body.set_alpha(0.40)
    rng = np.random.default_rng(17)
    for index, selected in enumerate(values):
        position = positions[index]
        plotted = selected
        if plotted.size > 350:
            plotted = rng.choice(plotted, size=350, replace=False)
        jitter = rng.normal(0.0, 0.045, size=plotted.size)
        color = _FEATURE_COLORS[index % len(_FEATURE_COLORS)]
        ax.scatter(
            position + jitter,
            plotted,
            s=5,
            color="#6D7880",
            alpha=0.15,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        q1, median, q3 = np.percentile(selected, [25, 50, 75])
        ax.plot([position, position], [q1, q3], color=color, linewidth=3.0,
                solid_capstyle="round", zorder=3)
        ax.scatter([position], [median], s=24, color="white", edgecolor=color,
                   linewidth=1.0, zorder=4)
    ax.set_xticks(positions, FINGERPRINT_LABELS, fontsize=7.4)
    ax.set_ylabel("Standardized value", labelpad=4)
    ax.set_title("Global fingerprint distribution", pad=7, fontweight="bold")
    ax.axhline(0.0, color="#999999", linewidth=0.75, linestyle="--", zorder=0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.89)
    _save_publication_figure(fig, save_path)


def _save_per_class_accuracy(cm, class_names, save_path):
    accuracy = np.diag(cm).astype(np.float64) / np.maximum(cm.sum(axis=1), 1.0)
    plt.rcParams.update(_publication_rcparams())
    fig, ax = plt.subplots(figsize=(5.6, 3.55), dpi=300)
    bars = ax.bar(
        np.arange(len(class_names)),
        accuracy,
        color=_CLASS_COLORS[: len(class_names)],
        width=0.58,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Recall / class accuracy")
    ax.set_title("Per-class classification accuracy", pad=10, fontweight="bold")
    ax.set_xticks(
        np.arange(len(class_names)),
        [_wrap_class_label(name) for name in class_names],
        rotation=0,
    )
    for bar, value in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.03, value + 0.025),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    fig.tight_layout(pad=1.1)
    _save_publication_figure(fig, save_path)


def _save_morphology_field_examples(morphology_examples, class_names, save_path):
    if not morphology_examples:
        return
    rows = len(class_names)
    available_fields = [
        np.asarray(morphology_examples[class_id][1], dtype=np.float32)
        for class_id in range(rows)
        if class_id in morphology_examples
    ]
    if not available_fields:
        return
    channel_limits = []
    for channel in (0, 1):
        pooled = np.concatenate([field[channel].ravel() for field in available_fields])
        low, high = np.percentile(pooled, [1, 99])
        channel_limits.append((float(low), float(max(high, low + 1e-8))))

    plt.rcParams.update(_publication_rcparams())
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(6.65, 1.48 * rows),
        squeeze=False,
        dpi=300,
    )
    field_image = None
    for class_id, name in enumerate(class_names):
        row_axes = axes[class_id]
        if class_id not in morphology_examples:
            for axis in row_axes:
                axis.axis("off")
            continue
        image, field = morphology_examples[class_id]
        image = np.asarray(image)
        field = np.asarray(field, dtype=np.float32)
        row_axes[0].imshow(image)
        row_axes[0].text(
            -0.06,
            0.5,
            _wrap_class_label(name),
            transform=row_axes[0].transAxes,
            ha="right",
            va="center",
            fontsize=7.5,
        )
        for channel in (0, 1):
            values = field[channel]
            low, high = channel_limits[channel]
            normalized = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
            field_image = row_axes[channel + 1].imshow(
                normalized,
                cmap=SEQUENTIAL_CMAP,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
        for axis in row_axes:
            axis.axis("off")
    column_titles = ("Input micrograph", "Morphology response", "Lath-width field")
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=8.5, fontweight="bold", pad=5)
    fig.suptitle("Core morphology fields by class", y=0.978, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.89, bottom=0.035, top=0.91, wspace=0.045, hspace=0.14)
    if field_image is not None:
        color_axis = fig.add_axes([0.915, 0.14, 0.012, 0.66])
        colorbar = fig.colorbar(field_image, cax=color_axis)
        colorbar.set_label("Normalized value", fontsize=7.5, labelpad=5)
        colorbar.ax.tick_params(labelsize=6.8, length=2.5)
        colorbar.outline.set_linewidth(0.6)
    _save_publication_figure(fig, save_path)


def _save_evaluation_visualizations(
    output_dir,
    all_labels,
    all_preds,
    fingerprints,
    class_names,
    ace_embeddings=None,
    morphology_examples=None,
):
    os.makedirs(output_dir, exist_ok=True)
    # Keep a style-independent bundle next to the rendered figures.  The
    # arrays below make it possible to redraw every statistical panel without
    # loading the model or repeating validation.
    all_labels = np.asarray(all_labels, dtype=np.int64)
    all_preds = np.asarray(all_preds, dtype=np.int64)
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)
    np.save(os.path.join(output_dir, "predictions.npy"), all_preds)
    np.save(os.path.join(output_dir, "fingerprints.npy"), fingerprints)
    np.save(os.path.join(output_dir, "fingerprint_names.npy"), np.asarray(FINGERPRINT_NAMES))
    with open(os.path.join(output_dir, "class_names.json"), "w", encoding="utf-8") as file:
        json.dump([str(name) for name in class_names], file, ensure_ascii=False, indent=2)
    _save_morphology_examples_data(morphology_examples or {}, output_dir)
    _save_visualization_manifest(output_dir, class_names)
    labels = np.arange(len(class_names), dtype=np.int64)
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    _save_per_class_performance(cm, class_names, output_dir)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=labels,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(class_names[index]): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(output_dir, "confusion_matrix.csv")
    )
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame({
        "class_id": labels,
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    _save_confusion_matrix_figure(
        cm, class_names, os.path.join(output_dir, "confusion_matrix.png")
    )
    _save_per_class_accuracy(
        cm, class_names, os.path.join(output_dir, "per_class_accuracy.png")
    )
    _save_fingerprint_boxplot(
        fingerprints,
        all_labels,
        class_names,
        os.path.join(output_dir, "fingerprint_boxplot.png"),
    )
    _save_global_fingerprint(
        fingerprints, os.path.join(output_dir, "global_fingerprint.png")
    )
    _save_fingerprint_tsne(
        fingerprints,
        all_labels,
        class_names,
        os.path.join(output_dir, "fingerprint_tsne.png"),
        coordinates_path=os.path.join(output_dir, "fingerprint_tsne_coordinates.npy"),
    )
    if ace_embeddings is not None:
        ace_embeddings = np.asarray(ace_embeddings, dtype=np.float32)
        if ace_embeddings.ndim == 2 and ace_embeddings.shape[0] >= 3:
            np.save(os.path.join(output_dir, "ace_embeddings.npy"), ace_embeddings)
            perplexity = min(30, ace_embeddings.shape[0] - 1)
            embedding = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="random",
                learning_rate="auto",
            ).fit_transform(ace_embeddings)
            np.save(
                os.path.join(output_dir, "latent_manifold_coordinates.npy"),
                np.asarray(embedding, dtype=np.float32),
            )
            _save_embedding_plot(
                embedding,
                all_labels,
                class_names,
                os.path.join(output_dir, "latent_manifold_with_physics.png"),
                "Latent space manifold (ACE embeddings)",
            )
        elif ace_embeddings.ndim == 2:
            np.save(os.path.join(output_dir, "ace_embeddings.npy"), ace_embeddings)
    _save_morphology_field_examples(
        morphology_examples or {},
        class_names,
        os.path.join(output_dir, "morphology_field_core_examples.png"),
    )
    return cm, metrics

MORPHOLOGY_MEASUREMENT_SIZE = 512
LATH_MIN_LENGTH_RATIO = 0.05
LATH_MAX_WIDTH_RATIO = 0.15
BETA_BLUR_KERNEL = 3
BETA_MIN_COMPONENT_AREA_PX = 8
BETA_IS_DARKER = True
BETA_CLOSE_KERNEL = 3
BETA_SHARPEN_AMOUNT = 1.25
BETA_ADAPTIVE_BLOCK_SIZE = 41
BETA_ADAPTIVE_C = 3


PROMPT_DICT = {
    0: "basket weave microstructure",
    1: "discontinuous microstructure",
    2: "Widmanstatten microstructure",
    3: "alpha-coarsened Widmanstatten microstructure",
}

CANONICAL_CLASS_NAMES = [
    "Basket_weave",
    "Discontinuous",
    "Widmanstatten",
    "Alpha_coarsened_Widmanstatten",
]
CLASS_NAME_ALIASES = {
    "Lamellar": "Widmanstatten",
    "Widmanstatten": "Widmanstatten",
    "Alpha_coarsened_Widmanstatten": "Alpha_coarsened_Widmanstatten",
    "AlphaCoarsenedWidmanstatten": "Alpha_coarsened_Widmanstatten",
}


def _ascii_fold(value):
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")


def canonical_class_id_from_path(path_like, fallback=None):
    """Map validation/test folder names to the fixed four-class label space."""
    path_text = _ascii_fold(path_like).replace("\\", "/")
    folder_name = path_text.rstrip("/").split("/")[-2] if "/" in path_text else ""

    canonical_name = CLASS_NAME_ALIASES.get(folder_name, folder_name)
    if canonical_name in CANONICAL_CLASS_NAMES:
        return CANONICAL_CLASS_NAMES.index(canonical_name)

    if folder_name.isdigit():
        numeric_label = int(folder_name)
        if 0 <= numeric_label < len(CANONICAL_CLASS_NAMES):
            return numeric_label

    if fallback is not None and 0 <= int(fallback) < len(CANONICAL_CLASS_NAMES):
        return int(fallback)

    raise ValueError(f"Cannot map validation/test folder to canonical class: {folder_name!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Four-class FIM-Diffusion training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--validation_img_dir", type=str, default=None)
    parser.add_argument("--test_img_dir", type=str, default=None)
    parser.add_argument("--test_prompt_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=2026)
    parser.add_argument(
        "--ablation",
        choices=[*ABLATION_MODES, "all", "summary"],
        default="full",
        help="Run one ablation, all three sequentially, or only rebuild the summary.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Defaults to the parent directory of this script's Diffusion folder.",
    )
    parser.add_argument(
        "--ablation-root",
        type=str,
        default=None,
        help="Root directory for all ablation outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds used by --ablation all/summary; defaults to --seed.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="When running all modes, keep runs that already contain test metrics.",
    )
    parser.add_argument(
        "--skip-pipeline-export",
        action="store_true",
        help="Skip optional Stable Diffusion pipeline packaging after metrics are written.",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_ablation_seeds(value, fallback):
    if value is None or not str(value).strip():
        return [int(fallback)]
    seeds = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def resolve_project_root(args):
    script_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.abspath(
        os.path.expanduser(
            args.project_root or os.environ.get("LEE_PROJECT_ROOT", script_project_root)
        )
    )


def resolve_ablation_root(args, project_root):
    requested = args.ablation_root or args.output_dir
    if requested:
        return os.path.abspath(os.path.expanduser(requested))
    return os.path.join(project_root, "Diffusion", "Deep-TiDiffusion-Ablation")


def apply_ablation_config(args):
    """Apply one controlled ablation while keeping all other settings fixed."""
    if args.ablation == "full":
        return args

    if args.ablation == "no_mtfa":
        # Keep K=3 so the timestep coverage and DCL objective remain unchanged.
        args.use_mtfa = False
        args.mtfa_num_timesteps = 3
        return args

    if args.ablation == "no_fp":
        # Remove both fingerprint conditioning and all fingerprint-head supervision.
        args.use_ace = False
        args.use_pqa = False
        args.stage2_lambda_fingerprint = 0.0
        args.stage2_lambda_separation = 0.0
        args.stage2_lambda_fp_decorrelation = 0.0
        args.stage2_lambda_fp_head_orthogonal = 0.0
        args.lambda_fp_cls = 0.0
        return args

    raise ValueError(f"Unsupported single-run ablation mode: {args.ablation}")


def summarize_ablation_results(ablation_root, seeds):
    """Collect test metrics and create the paper-style ablation table."""
    run_rows = []
    missing = []
    for mode in ABLATION_MODES:
        for seed in seeds:
            metrics_path = os.path.join(
                ablation_root,
                mode,
                f"seed_{seed}",
                "test_results",
                "per_class_metrics.json",
            )
            if not os.path.isfile(metrics_path):
                missing.append(metrics_path)
                continue
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            run_rows.append({
                "mode": mode,
                "method": ABLATION_DISPLAY_NAMES[mode],
                "seed": int(seed),
                "accuracy": float(metrics["accuracy"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
            })

    if missing:
        rendered = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Cannot build the complete ablation table; missing metrics:\n  - " + rendered
        )

    runs = pd.DataFrame(run_rows)
    summary_rows = []
    for mode in ABLATION_MODES:
        selected = runs[runs["mode"] == mode]
        row = {
            "mode": mode,
            "method": ABLATION_DISPLAY_NAMES[mode],
            "n_seeds": int(len(selected)),
        }
        for metric in ("accuracy", "macro_recall", "macro_f1"):
            values = selected[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    os.makedirs(ablation_root, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    runs.to_csv(os.path.join(ablation_root, "ablation_classification_runs.csv"), index=False)
    summary.to_csv(os.path.join(ablation_root, "ablation_classification_metrics.csv"), index=False)
    with open(os.path.join(ablation_root, "ablation_classification_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"seeds": list(map(int, seeds)), "runs": run_rows, "summary": summary_rows},
            file,
            ensure_ascii=False,
            indent=2,
        )

    def format_metric(row, metric):
        mean = float(row[f"{metric}_mean"])
        std = float(row[f"{metric}_std"])
        return f"{mean:.3f}" if int(row["n_seeds"]) == 1 else f"{mean:.3f} ± {std:.3f}"

    cell_text = [
        [
            row["method"],
            format_metric(row, "accuracy"),
            format_metric(row, "macro_recall"),
            format_metric(row, "macro_f1"),
        ]
        for row in summary_rows
    ]
    plt.rcParams.update(_publication_rcparams())
    fig, axis = plt.subplots(figsize=(7.4, 2.35), dpi=300)
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=["Method", "Accuracy", "Macro Recall", "Macro F1"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#6F6F6F")
        cell.set_linewidth(1.0 if row_index == 0 else 0.45)
        if row_index == 0:
            cell.set_text_props(weight="bold")
    fig.tight_layout(pad=0.3)
    _save_publication_figure(
        fig,
        os.path.join(ablation_root, "ablation_classification_metrics.png"),
    )
    print(summary.to_string(index=False))
    print(f"[Ablation] Summary saved under: {ablation_root}")
    return summary


def load_validation_data(args, accelerator, weight_dtype, img_dir=None, prompt_dir=None):
    del accelerator, weight_dtype, prompt_dir
    root = img_dir or args.validation_img_dir
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"Evaluation image directory not found: {root}")

    image_paths = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for current_root, directories, files in os.walk(root):
        directories.sort()
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() not in valid_extensions:
                continue
            path = os.path.join(current_root, filename)
            try:
                label = canonical_class_id_from_path(path)
            except ValueError:
                continue
            image_paths.append(path)
            labels.append(label)

    if not image_paths:
        raise ValueError(f"No class-organized evaluation images found under: {root}")
    unique_labels = sorted(set(labels))
    unique_names = [CANONICAL_CLASS_NAMES[label] for label in unique_labels]
    return image_paths, labels, labels, unique_names, unique_labels


def set_acta_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "image.cmap": "gray",
        "axes.grid": False,
    })


set_acta_style()


def sample_multi_timesteps(batch_size, num_train_timesteps, K, device):
    t = torch.zeros(batch_size, K, device=device, dtype=torch.long)
    low_bound = max(1, num_train_timesteps // 10)
    high_bound = num_train_timesteps * 8 // 10
    mid_start = num_train_timesteps // 5
    t[:, 0] = torch.randint(0, low_bound, (batch_size,), device=device)
    if K > 1:
        t[:, -1] = torch.randint(high_bound, num_train_timesteps, (batch_size,), device=device)
    for k in range(1, K - 1):
        t[:, k] = torch.randint(mid_start, high_bound, (batch_size,), device=device)
    return t


def split_repeated_features_by_timestep(features, batch_size, num_timesteps):
    expected = batch_size * num_timesteps
    if features.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} repeated features, received {features.shape[0]}."
        )
    features_bk = features.reshape(batch_size, num_timesteps, *features.shape[1:])
    return list(features_bk.unbind(dim=1))


def evaluation_timesteps(num_train_timesteps, K, device):
    # Match the training strata, with fixed representatives for every evaluation.
    fractions = [0.05] if K == 1 else [0.05] + np.linspace(0.2, 0.8, K)[1:-1].tolist() + [0.9]
    values = [min(num_train_timesteps - 1, int(t * num_train_timesteps)) for t in fractions]
    return torch.tensor([values], dtype=torch.long, device=device)


def evaluation_noise(latents, image_path, image_root, seed):
    # A sample's noise is independent of traversal order and diagnostic plotting.
    key = os.path.relpath(image_path, image_root).replace("\\", "/")
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2 ** 63))
    noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=latents.device, dtype=latents.dtype)


def reproducible_evaluation(function):
    """Isolate evaluation RNGs and restore all module modes, even on failure."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*positional, **keywords):
        bound = signature.bind(*positional, **keywords)
        args = bound.arguments["args"]
        seed = int(getattr(args, "eval_seed", 2026))
        modes = {
            child: child.training
            for value in bound.arguments.values() if isinstance(value, nn.Module)
            for child in value.modules()
        }
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                for module in modes:
                    module.training = False
                return function(*positional, **keywords)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
            for module, was_training in modes.items():
                module.training = was_training

    return wrapped


def load_best_checkpoint_for_evaluation(path, accelerator, modules, class_names):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Best validation checkpoint not found: {path}")
    # This checkpoint was created locally by this training run and includes NumPy stats.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if list(checkpoint.get("class_names", [])) != list(class_names):
        raise ValueError("Best checkpoint class order does not match this run.")
    checkpoint_names = [
        "dark_contrast_area_fraction" if name == "beta_phase_fraction" else name
        for name in checkpoint.get("fingerprint_names", [])
    ]
    if checkpoint_names != list(FINGERPRINT_NAMES):
        raise ValueError("Best checkpoint must use the same two-dimensional fingerprint.")
    checkpoint["fingerprint_names"] = checkpoint_names
    for name, module in modules.items():
        state = checkpoint.get(name)
        if module is not None and state is None:
            raise ValueError(f"Best checkpoint is missing active module: {name}")
        if module is None and state is not None:
            raise ValueError(f"Best checkpoint contains an inactive module: {name}")
    for name, module in modules.items():
        if module is not None:
            accelerator.unwrap_model(module).load_state_dict(checkpoint[name], strict=True)
    return checkpoint


def save_evaluation_protocol(output_dir, args, timesteps, split, step=None):
    protocol = {
        "split": split,
        "evaluation_seed": int(args.eval_seed),
        "timesteps": timesteps.detach().cpu().tolist(),
        "noise_policy": "sha256(eval_seed:relative_image_path), CPU torch.Generator",
        "deterministic_algorithms": True,
        "class_decision": "unconditioned four-class logits argmax; no postprocessing",
        "checkpoint": getattr(args, "selected_checkpoint", None) if split == "test" else None,
        "checkpoint_step": getattr(args, "selected_checkpoint_step", None) if split == "test" else step,
        "selection_metric": "validation_accuracy",
    }
    with open(os.path.join(output_dir, "evaluation_protocol.json"), "w", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2)


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


def _remove_small_components(mask, min_area):
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_idx] = 1
    return cleaned.astype(bool)


def _deterministic_medial_axis(mask):
    try:
        return medial_axis(mask.astype(bool), return_distance=True, rng=0)
    except TypeError:
        return medial_axis(mask.astype(bool), return_distance=True, random_state=0)


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


def _reference_beta_mask(img_array):
    gray = _to_gray_u8(img_array)
    blurred = cv2.GaussianBlur(gray, (BETA_BLUR_KERNEL, BETA_BLUR_KERNEL), 0)
    sharpened = cv2.addWeighted(
        gray,
        BETA_SHARPEN_AMOUNT,
        blurred,
        1.0 - BETA_SHARPEN_AMOUNT,
        0.0,
    )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sharpened)
    threshold = _iterative_threshold(enhanced)
    global_mask = enhanced <= threshold if BETA_IS_DARKER else enhanced > threshold

    block_size = max(3, int(BETA_ADAPTIVE_BLOCK_SIZE))
    if block_size % 2 == 0:
        block_size += 1
    adaptive_type = cv2.THRESH_BINARY_INV if BETA_IS_DARKER else cv2.THRESH_BINARY
    local_mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        adaptive_type,
        block_size,
        BETA_ADAPTIVE_C,
    ).astype(bool)
    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    line_support = blackhat >= np.percentile(blackhat, 45.0)
    beta_mask = global_mask | (local_mask & line_support)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BETA_CLOSE_KERNEL, BETA_CLOSE_KERNEL),
    )
    beta_mask = cv2.morphologyEx(
        beta_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    ).astype(bool)
    beta_mask = _remove_small_components(beta_mask, BETA_MIN_COMPONENT_AREA_PX)

    beta_fraction = float(beta_mask.mean())
    if beta_fraction < 0.02 or beta_fraction > 0.80:
        otsu_threshold, fallback = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        beta_mask = fallback == 0 if BETA_IS_DARKER else fallback > 0
        beta_mask = cv2.morphologyEx(
            beta_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
        ).astype(bool)
        beta_mask = _remove_small_components(beta_mask, BETA_MIN_COMPONENT_AREA_PX)
        threshold = float(otsu_threshold)

    return beta_mask.astype(bool), enhanced, threshold


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

    out_labels = np.zeros_like(labels, dtype=np.int32)
    tracks = []
    new_id = 0
    for label_idx in range(1, n_labels):
        seg = labels == label_idx
        length_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        if length_px < min_len_px:
            continue
        widths_px = dist[seg] * 2.0
        widths_px = widths_px[(widths_px > 0) & np.isfinite(widths_px)]
        mean_width_px = float(widths_px.mean()) if widths_px.size else 0.0
        if not (min_width_px <= mean_width_px <= max_width_px):
            continue

        ys, xs = np.where(seg)
        if xs.size >= 2:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            pts -= pts.mean(axis=0, keepdims=True)
            cov = np.cov(pts, rowvar=False)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            eig_vals = np.maximum(eig_vals, 1e-6)
            vx, vy = eig_vecs[:, np.argmax(eig_vals)]
            angle = float((np.arctan2(vy, vx) + np.pi) % np.pi)
            linearity = float(eig_vals.max() / eig_vals.sum())
        else:
            angle = 0.0
            linearity = 0.0

        new_id += 1
        out_labels[seg] = new_id
        tracks.append({
            "label": new_id,
            "length_px": length_px,
            "mean_width_px": mean_width_px,
            "angle": angle,
            "linearity": linearity,
        })
    return out_labels, dist, tracks


def _build_quantification_masks(img_array, mask_size=(128, 128)):
    gray = _to_gray_u8(img_array)
    measurement_side = max(MORPHOLOGY_MEASUREMENT_SIZE, int(max(mask_size)))
    gray = cv2.resize(gray, (measurement_side, measurement_side), interpolation=cv2.INTER_AREA)
    alpha_mask, beta_mask, enhanced = _segment_alpha_beta(gray)
    return gray, enhanced, alpha_mask, beta_mask


def _compose_morphology_field(img_array, mask_size=(128, 128)):
    gray, enhanced, alpha_mask, beta_mask = _build_quantification_masks(img_array, mask_size)
    enhanced_small = cv2.resize(enhanced, mask_size, interpolation=cv2.INTER_AREA)
    vesselness = frangi(enhanced_small, sigmas=(1, 2, 3, 4, 5), beta=0.5, gamma=15, black_ridges=False)
    vesselness = _normalize01(np.clip(vesselness, 0, 1))

    phase_edge = cv2.morphologyEx(
        alpha_mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(np.float32)
    phase_edge = cv2.resize(phase_edge, mask_size, interpolation=cv2.INTER_AREA)
    measurement_side = min(alpha_mask.shape)
    min_len_px = max(8, int(round(LATH_MIN_LENGTH_RATIO * measurement_side)))
    max_width_px = max(8.0, LATH_MAX_WIDTH_RATIO * measurement_side)
    lath_labels, alpha_dist, _ = _extract_lath_tracks(
        alpha_mask,
        min_len_px=min_len_px,
        min_width_px=1.0,
        max_width_px=max_width_px,
    )
    lath_skeleton = (lath_labels > 0).astype(np.uint8)
    lath_skeleton = cv2.dilate(
        lath_skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(np.float32)
    lath_skeleton = cv2.resize(lath_skeleton, mask_size, interpolation=cv2.INTER_AREA)

    beta_line = cv2.resize(beta_mask.astype(np.float32), mask_size, interpolation=cv2.INTER_AREA)
    line_field = _normalize01(0.45 * vesselness + 0.25 * phase_edge + 0.20 * lath_skeleton + 0.10 * beta_line)
    local_width_px = np.where(alpha_mask, alpha_dist * 2.0, 0.0).astype(np.float32)
    width_field = np.log1p(np.clip(local_width_px, 0.0, max_width_px)) / np.log1p(max_width_px)

    width_field = cv2.resize(width_field, mask_size, interpolation=cv2.INTER_AREA)
    return np.stack([line_field, width_field], axis=0).astype(np.float32)


def generate_morphology_fields(img_path, mask_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(img_path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _compose_morphology_field(img, mask_size=mask_size)


def generate_morphology_fields_from_array(img_array, mask_size=(128, 128)):
    return _compose_morphology_field(img_array, mask_size=mask_size)


def _safe_percentile_mask(x, percentile, fallback_threshold=0.5):
    thr = np.percentile(x, percentile)
    if not np.isfinite(thr):
        thr = fallback_threshold
    return x > thr


def _component_aspect_ratio(component_mask):
    ys, xs = np.where(component_mask)
    if len(xs) < 5:
        return np.inf
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    major = max(w, h)
    minor = max(min(w, h), 1e-6)
    return major / minor


def compute_morphology_fingerprint_np(field, img_array=None):
    """Measure dark-contrast area and lath width, without inferring phase fraction."""
    if img_array is not None:
        gray, _, alpha_mask, beta_mask = _build_quantification_masks(
            img_array,
            mask_size=(field.shape[-1], field.shape[-2]),
        )
    else:
        line_field = _normalize01(field[0])
        alpha_mask = _normalize01(field[1]) > 1e-6
        beta_mask = _safe_percentile_mask(line_field, 82)

    measurement_side = min(alpha_mask.shape)
    lath_labels, alpha_dist, tracks = _extract_lath_tracks(
        alpha_mask,
        min_len_px=max(8, int(round(LATH_MIN_LENGTH_RATIO * measurement_side))),
        min_width_px=1.0,
        max_width_px=max(8.0, LATH_MAX_WIDTH_RATIO * measurement_side),
    )

    if tracks:
        mean_lath_width = float(np.average(
            [track["mean_width_px"] for track in tracks],
            weights=[max(track["length_px"], 1) for track in tracks],
        ))
    else:
        fallback_skeleton, _ = _deterministic_medial_axis(alpha_mask)
        valid_width = alpha_dist[fallback_skeleton] * 2.0
        mean_lath_width = float(valid_width.mean()) if valid_width.size > 0 else 0.0

    dark_contrast_area_fraction = float(beta_mask.mean())

    return np.array([
        dark_contrast_area_fraction,
        mean_lath_width,
    ], dtype=np.float32)


def compute_extended_fingerprint_np(field, img_array=None):
    return compute_morphology_fingerprint_np(field, img_array=img_array)


def compute_connectivity_descriptor_np(field):
    """Compute length, fragmentation, and orientation targets from the line field."""
    line_field = np.asarray(field[0], dtype=np.float32)
    line_field = _normalize01(np.nan_to_num(line_field, nan=0.0))
    threshold = float(np.percentile(line_field, 72.0))
    mask = line_field >= threshold
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    min_area = max(3, int(0.001 * mask.size))
    mask = remove_small_objects(mask, min_size=min_area, connectivity=2)
    if not mask.any():
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)

    skeleton = skeletonize(mask)
    if not skeleton.any():
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)

    skeleton_length = float(skeleton.sum())

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    lengths = [
        float(component_stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
        if component_stats[index, cv2.CC_STAT_AREA] > 0
    ]
    if lengths:
        lengths = np.asarray(lengths, dtype=np.float32)
        total_length = float(lengths.sum())
        reference_length = max(8.0, 0.12 * min(mask.shape))
        length_weights = np.clip(lengths / reference_length, 0.0, 1.0)
        connectivity_length = float(
            np.sum(lengths * length_weights) / max(total_length, 1.0)
        )
        long_length_threshold = max(6.0, 0.04 * min(mask.shape))
        short_fraction = float(
            np.sum(lengths[lengths < long_length_threshold])
            / max(total_length, 1.0)
        )
        largest_fraction = float(lengths.max() / max(total_length, 1.0))
        fragmentation_degree = float(
            np.clip(0.5 * short_fraction + 0.5 * (1.0 - largest_fraction), 0.0, 1.0)
        )
    else:
        connectivity_length = 0.0
        fragmentation_degree = 1.0

    _, _, tracks = _extract_lath_tracks(
        mask,
        min_len_px=5,
        min_width_px=0.1,
        max_width_px=float(max(mask.shape)),
    )
    if tracks:
        track_lengths = np.asarray(
            [max(float(track["length_px"]), 1.0) for track in tracks], dtype=np.float32
        )
        angles = np.asarray([float(track["angle"]) for track in tracks], dtype=np.float32)
        orientation_coherence = float(
            np.abs(np.sum(track_lengths * np.exp(2j * angles)) / max(track_lengths.sum(), 1e-6))
        )
    else:
        orientation_coherence = 0.0

    return np.asarray(
        [
            np.clip(connectivity_length, 0.0, 1.0),
            np.clip(fragmentation_degree, 0.0, 1.0),
            np.clip(orientation_coherence, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


class TokenizedACE(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4, n_classes=4, fp_dim=FP_DIM):
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


def get_lora_parameters(module):
    return [p for n, p in module.named_parameters() if "lora_" in n and p.requires_grad]


def get_lora_state_dict(module):
    return {n: p.detach().cpu() for n, p in module.named_parameters() if "lora_" in n}


def merge_lora_into_unet(unet):
    for name, module in list(unet.named_modules()):
        if isinstance(module, LoRALinear):
            merged = nn.Linear(
                module.base_layer.in_features,
                module.base_layer.out_features,
                bias=module.base_layer.bias is not None,
            ).to(module.base_layer.weight.device, dtype=module.base_layer.weight.dtype)
            delta = module.lora_up.weight @ module.lora_down.weight
            merged.weight.data.copy_(module.base_layer.weight.data + module.scale * delta.to(module.base_layer.weight.dtype))
            if module.base_layer.bias is not None:
                merged.bias.data.copy_(module.base_layer.bias.data)
            _set_nested_module(unet, name, merged)
    return unet


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


class MorphologyHead(nn.Module):
    def __init__(self, in_channels=1280, out_channels=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.SiLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, out_channels, 1),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        return self.head(features)


class UnifiedHeadNoSeg_v2(nn.Module):
    def __init__(self, in_channels=1280, n_classes=4, hidden_dim=256, fp_dim=FP_DIM):
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
            nn.Linear(256, fp_dim),
        )

    def forward(self, features):
        cls_feat = self.cls_proj(features)
        cls_logits = self.cls_branch(cls_feat)
        fp_feat = self.fp_proj(features)
        fingerprint = self.fp_branch(fp_feat)
        return cls_logits, fingerprint, cls_feat, fp_feat


class ConnectivityHead(nn.Module):
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

    def forward(self, features):
        return torch.sigmoid(self.net(features))


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


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return F.normalize(self.net(features), dim=-1)


class DiffusionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, div_threshold=0.6, lambda_div=0.1):
        super().__init__()
        self.temperature = temperature
        self.div_threshold = div_threshold
        self.lambda_div = lambda_div

    def forward(self, z_all_timesteps, labels):
        B, K, D = z_all_timesteps.shape
        z = F.normalize(z_all_timesteps, dim=-1)
        z_flat = z.reshape(B * K, D)

        img_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, K).reshape(-1)
        same_image = img_idx.unsqueeze(0) == img_idx.unsqueeze(1)

        sim = torch.matmul(z_flat, z_flat.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim)

        pos_mask = same_image.float()
        pos_mask.fill_diagonal_(0)
        denom = exp_sim.sum(dim=1) - exp_sim.diag() + 1e-8
        pos = (exp_sim * pos_mask).sum(dim=1)
        loss_align = -torch.log(pos / denom + 1e-8).mean()

        sim_kk = torch.matmul(z, z.transpose(1, 2))
        mask = 1 - torch.eye(K, device=z.device).unsqueeze(0)
        div = F.relu(sim_kk - self.div_threshold) * mask
        loss_div = div.sum() / (mask.sum() * B + 1e-8)

        return loss_align + self.lambda_div * loss_div


def extract_unconditioned_class_features(
    unet_wrapper,
    noisy_latents,
    timesteps_flat,
    timesteps_matrix,
    encoder_hidden_states,
    mta_module=None,
    dit=None,
):
    # Training needs gradients through UNet LoRA, MTA and DiT; evaluation owns no_grad.
    zero_hidden_states = torch.zeros_like(encoder_hidden_states)
    _, mid_features_uncond = unet_wrapper(noisy_latents, timesteps_flat, zero_hidden_states)
    B = timesteps_matrix.shape[0]
    K = timesteps_matrix.shape[1]
    mid_features_list = split_repeated_features_by_timestep(mid_features_uncond, B, K)
    if mta_module is not None and len(mid_features_list) > 1:
        cls_features = mta_module(mid_features_list, timesteps_matrix)
    else:
        # The no-MTFA control keeps the same K timesteps and replaces the
        # learned fusion with a parameter-free uniform mean. This isolates the
        # contribution of learned timestep fusion from timestep coverage.
        cls_features = torch.stack(mid_features_list, dim=0).mean(dim=0)
    cls_features = cls_features.float().clamp(-50, 50)
    if dit:
        cls_features = dit(cls_features)
    return cls_features


def get_class_prompts(class_ids):
    prompts = []
    for cls in class_ids.detach().cpu().tolist():
        prompts.append(PROMPT_DICT.get(int(cls), f"class {int(cls)} microstructure"))
    return prompts


@torch.no_grad()
def encode_class_prompts(tokenizer, text_encoder, class_ids, device):
    prompts = get_class_prompts(class_ids)
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    return text_encoder(input_ids)[0]


def build_scheme_a_condition(
    tokenizer,
    text_encoder,
    ace,
    fingerprint,
    class_ids,
    device,
    ace_scale=1.0,
    use_zero_fingerprint=False,
):
    text_hidden = encode_class_prompts(tokenizer, text_encoder, class_ids, device)
    if ace is None:
        return text_hidden, torch.zeros(text_hidden.shape[0], text_hidden.shape[-1], device=device)

    fp_for_token = torch.zeros_like(fingerprint) if use_zero_fingerprint else fingerprint
    fp_token = ace(fp_for_token, class_ids)
    fp_token = F.normalize(fp_token, dim=-1) * ace_scale
    encoder_hidden_states = text_hidden.clone()
    encoder_hidden_states[:, 0, :] = fp_token
    return encoder_hidden_states, fp_token


class MorphologyFieldLoss(nn.Module):
    def __init__(self, lambda_grad=0.2):
        super().__init__()
        self.lambda_grad = lambda_grad
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient_consistency(self, pred, target):
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

    def forward(self, pred_field, target_field):
        loss_mse = F.mse_loss(pred_field, target_field)
        loss_grad = self.gradient_consistency(pred_field[:, 1:2], target_field[:, 1:2])
        return loss_mse + self.lambda_grad * loss_grad


def hybrid_fingerprint_loss(pred, target, beta=0.1):
    l1 = F.smooth_l1_loss(pred, target, beta=beta)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.9 * l1 + 0.1 * cos


def pairwise_class_margin_loss(logits, labels, class_a=0, class_b=1, margin=0.2):
    """Keep the two easily confused classes separated in logit space."""
    pair_mask = (labels == class_a) | (labels == class_b)
    if pair_mask.sum() < 1:
        return logits.new_tensor(0.0)
    pair_logits = logits[pair_mask]
    pair_labels = labels[pair_mask]
    signed_target = torch.where(
        pair_labels == class_a,
        torch.ones_like(pair_labels, dtype=pair_logits.dtype),
        -torch.ones_like(pair_labels, dtype=pair_logits.dtype),
    )
    signed_margin = signed_target * (pair_logits[:, class_a] - pair_logits[:, class_b])
    return F.relu(margin - signed_margin).mean()


def _off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("Expected square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def fingerprint_decorrelation_loss(fp, min_batch=4, eps=1e-6):
    if fp.shape[0] < min_batch:
        return fp.new_tensor(0.0)
    z = fp - fp.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, unbiased=False, keepdim=True) + eps)
    corr = torch.matmul(z.T, z) / fp.shape[0]
    return _off_diagonal(corr).pow(2).mean()


def fingerprint_head_orthogonality_loss(unified_head):
    last_linear = None
    for m in unified_head.fp_branch.modules():
        if isinstance(m, nn.Linear) and m.out_features == FP_DIM:
            last_linear = m
    if last_linear is None:
        return next(unified_head.parameters()).new_tensor(0.0)
    w = F.normalize(last_linear.weight, dim=1)
    gram = torch.matmul(w, w.T)
    return _off_diagonal(gram).pow(2).mean()



@reproducible_evaluation
def run_validation(vae, unet_wrapper, unified_head, morphology_head, args, accelerator,
                   weight_dtype, epoch, noise_scheduler, class_names=None,
                   mta_module=None, ace=None, dit=None, pqa=None,
                   tokenizer=None, text_encoder=None, fp_mean=None, fp_std=None,
                   connectivity_head=None):
    logger.info(f"Step {epoch}: Running validation...")
    vae.eval()
    unet_wrapper.eval()
    unified_head.eval()
    morphology_head.eval()
    for module in (mta_module, ace, dit, pqa, connectivity_head):
        if module is not None:
            module.eval()

    try:
        image_paths, prompts_all, _, _, _ = load_validation_data(
            args, accelerator, weight_dtype
        )
    except Exception as exc:
        raise RuntimeError(f"Validation data error: {exc}") from exc

    records = []
    true_fingerprints_raw = []
    ace_embeddings = []
    morphology_examples = {}
    trajectory_record = None
    eval_transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    K = max(1, args.mtfa_num_timesteps)
    val_timesteps = evaluation_timesteps(
        noise_scheduler.config.num_train_timesteps, K, accelerator.device
    )

    for img_path, loader_label in zip(image_paths, prompts_all):
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = eval_transform(image).unsqueeze(0).to(accelerator.device)
            latent = vae.encode(img_tensor).latent_dist.mode() * vae.config.scaling_factor
            gt_label = canonical_class_id_from_path(img_path, fallback=loader_label)
            img_np = np.asarray(image)
            field = generate_morphology_fields_from_array(img_np)
            if gt_label not in morphology_examples:
                morphology_examples[gt_label] = (img_np, field)
            true_fp = compute_extended_fingerprint_np(field, img_np)
            fp_tensor = torch.from_numpy(true_fp).float().unsqueeze(0).to(accelerator.device)
            if fp_mean is not None and fp_std is not None:
                fp_tensor = (fp_tensor - torch.as_tensor(fp_mean, device=accelerator.device)) / (
                    torch.as_tensor(fp_std, device=accelerator.device) + 1e-8
                )

            class_tensor = torch.tensor([gt_label], device=accelerator.device, dtype=torch.long)
            cond_emb, ace_cond = build_scheme_a_condition(
                tokenizer, text_encoder, ace, fp_tensor, class_tensor,
                accelerator.device, ace_scale=args.ace_scale,
                use_zero_fingerprint=False,
            )
            latents_expanded = latent.repeat_interleave(K, dim=0)
            t_flat = val_timesteps.reshape(-1)
            noise = evaluation_noise(latents_expanded, img_path, args.validation_img_dir, args.eval_seed)
            noisy_latent = noise_scheduler.add_noise(latents_expanded, noise, t_flat)
            cond_expanded = cond_emb.repeat_interleave(K, dim=0)
            _, mid_features = unet_wrapper(noisy_latent, t_flat, cond_expanded)
            mid_features_list = split_repeated_features_by_timestep(mid_features, 1, K)
            features = (
                mta_module(mid_features_list, val_timesteps)
                if mta_module is not None
                else torch.stack(mid_features_list, dim=0).mean(dim=0)
            )
            if dit is not None:
                features = dit(features)
            if pqa is not None:
                features = pqa(features, ace_cond)
            if trajectory_record is None:
                trajectory_record = collect_diffusion_trajectory(
                    img_np,
                    torch.from_numpy(field).float().to(accelerator.device),
                    latent,
                    cond_emb,
                    noise_scheduler,
                    unet_wrapper,
                    morphology_head,
                    mid_features_list,
                    val_timesteps,
                    accelerator.device,
                )
            _, fp_pred, _, _ = unified_head(features)

            cls_features = extract_unconditioned_class_features(
                unet_wrapper, noisy_latent, t_flat, val_timesteps, cond_expanded,
                mta_module=mta_module, dit=dit,
            )
            cls_logits, _, _, _ = unified_head(cls_features)
            connectivity_pred = None
            if connectivity_head is not None:
                connectivity_pred = connectivity_head(cls_features.detach()).cpu().numpy().flatten()
            records.append({
                "path": img_path,
                "true_label": gt_label,
                "pred_label": int(cls_logits.argmax(dim=1).item()),
                "fingerprint": fp_pred.cpu().numpy().flatten().tolist(),
                "confidence": float(F.softmax(cls_logits, dim=1).max().item()),
                "connectivity_target": compute_connectivity_descriptor_np(field).tolist(),
                "connectivity_pred": connectivity_pred.tolist() if connectivity_pred is not None else None,
            })
            if ace_cond is not None:
                ace_embeddings.append(
                    ace_cond.detach().float().cpu().numpy().reshape(-1)
                )
            true_fingerprints_raw.append(true_fp)
        except Exception as exc:
            logger.warning(f"Validation error on {img_path}: {exc}")

    output_dir = os.path.join(args.output_dir, "val_results", f"step_{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    if not records or len(records) != len(image_paths):
        raise RuntimeError(
            f"Incomplete validation: {len(records)}/{len(image_paths)} images. "
            "This evaluation cannot select a best checkpoint; inspect the per-image errors."
        )
    save_evaluation_protocol(output_dir, args, val_timesteps, "validation", step=epoch)

    all_labels = np.asarray([r["true_label"] for r in records], dtype=np.int64)
    all_preds = np.asarray([r["pred_label"] for r in records], dtype=np.int64)
    fingerprints = np.asarray([r["fingerprint"] for r in records], dtype=np.float32)
    true_raw = np.asarray(true_fingerprints_raw, dtype=np.float32)
    if fp_mean is not None and fp_std is not None:
        pred_fingerprints_raw = (
            fingerprints * (np.asarray(fp_std, dtype=np.float32)[None, :] + 1e-8)
            + np.asarray(fp_mean, dtype=np.float32)[None, :]
        )
    else:
        pred_fingerprints_raw = fingerprints.copy()
    logger.info(
        "Step %s fingerprint spread: target_std=%s, predicted_std=%s",
        epoch,
        np.round(true_raw.std(axis=0), 6),
        np.round(pred_fingerprints_raw.std(axis=0), 6),
    )
    _save_evaluation_visualizations(
        output_dir=output_dir,
        all_labels=all_labels,
        all_preds=all_preds,
        fingerprints=fingerprints,
        class_names=class_names,
        ace_embeddings=ace_embeddings,
        morphology_examples=morphology_examples,
    )
    if trajectory_record is not None:
        _save_diffusion_trajectory_data(trajectory_record, output_dir)
        plot_diffusion_trajectory(
            trajectory_record,
            os.path.join(output_dir, "diffusion_morphology_trajectory.png"),
        )
    np.save(os.path.join(output_dir, "fingerprints.npy"), fingerprints)
    np.save(os.path.join(output_dir, "true_fingerprints_raw.npy"), true_raw)
    np.save(os.path.join(output_dir, "pred_fingerprints_raw.npy"), pred_fingerprints_raw)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    if connectivity_head is not None:
        conn_targets = np.asarray([r["connectivity_target"] for r in records], dtype=np.float32)
        conn_preds = np.asarray([r["connectivity_pred"] for r in records], dtype=np.float32)
        np.save(os.path.join(output_dir, "connectivity_targets.npy"), conn_targets)
        np.save(os.path.join(output_dir, "connectivity_predictions.npy"), conn_preds)
        logger.info(f"Step {epoch}: connectivity MAE={np.abs(conn_preds - conn_targets).mean():.4f}")

    return float((all_labels == all_preds).mean())


@reproducible_evaluation
def run_test(vae, unet_wrapper, unified_head, morphology_head, args, accelerator, weight_dtype,
             class_names=None, mta_module=None, ace=None, dit=None, pqa=None,
             projection_head=None, noise_scheduler=None, fp_mean=None, fp_std=None,
             tokenizer=None, text_encoder=None, connectivity_head=None):
    logger.info("Running final four-class test evaluation...")
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            local_files_only=True,
        )

    vae.eval()
    unet_wrapper.eval()
    unified_head.eval()
    morphology_head.eval()
    for module in (mta_module, ace, dit, pqa, projection_head, connectivity_head):
        if module is not None:
            module.eval()

    try:
        image_paths, prompts_all, _, _, _ = load_validation_data(
            args, accelerator, weight_dtype,
            img_dir=args.test_img_dir,
            prompt_dir=args.test_prompt_dir if args.test_prompt_dir else args.test_img_dir,
        )
    except Exception as exc:
        raise RuntimeError(f"Test data error: {exc}") from exc

    records = []
    true_fingerprints_raw = []
    true_fingerprints_norm = []
    ace_embeddings = []
    morphology_examples = {}
    trajectory_record = None
    n_classes = len(class_names)
    eval_transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    K = max(1, args.mtfa_num_timesteps)
    timesteps = evaluation_timesteps(
        noise_scheduler.config.num_train_timesteps, K, accelerator.device
    )

    with torch.no_grad():
        for img_path, loader_label in zip(image_paths, prompts_all):
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = eval_transform(image).unsqueeze(0).to(accelerator.device)
                latent = vae.encode(img_tensor).latent_dist.mode() * vae.config.scaling_factor
                gt_label = canonical_class_id_from_path(img_path, fallback=loader_label)
                img_np = np.asarray(image)
                field = generate_morphology_fields_from_array(img_np)
                if gt_label not in morphology_examples:
                    morphology_examples[gt_label] = (img_np, field)
                true_fp = compute_extended_fingerprint_np(field, img_np)

                fp_tensor = torch.from_numpy(true_fp).float().unsqueeze(0).to(accelerator.device)
                if fp_mean is not None and fp_std is not None:
                    fp_tensor = (fp_tensor - torch.as_tensor(fp_mean, device=accelerator.device)) / (
                        torch.as_tensor(fp_std, device=accelerator.device) + 1e-8
                    )
                class_tensor = torch.tensor([gt_label], device=accelerator.device, dtype=torch.long)
                cond_emb, ace_cond = build_scheme_a_condition(
                    tokenizer, text_encoder, ace, fp_tensor, class_tensor,
                    accelerator.device, ace_scale=args.ace_scale,
                    use_zero_fingerprint=False,
                )
                latents_expanded = latent.repeat_interleave(K, dim=0)
                t_flat = timesteps.reshape(-1)
                noise = evaluation_noise(latents_expanded, img_path, args.test_img_dir, args.eval_seed)
                noisy_latent = noise_scheduler.add_noise(latents_expanded, noise, t_flat)
                cond_expanded = cond_emb.repeat_interleave(K, dim=0)
                _, mid_features = unet_wrapper(noisy_latent, t_flat, cond_expanded)
                mid_features_list = split_repeated_features_by_timestep(mid_features, 1, K)
                features = (
                    mta_module(mid_features_list, timesteps)
                    if mta_module is not None
                    else torch.stack(mid_features_list, dim=0).mean(dim=0)
                )
                if dit is not None:
                    features = dit(features)
                if pqa is not None:
                    features = pqa(features, ace_cond)
                if trajectory_record is None:
                    trajectory_record = collect_diffusion_trajectory(
                        img_np,
                        torch.from_numpy(field).float().to(accelerator.device),
                        latent,
                        cond_emb,
                        noise_scheduler,
                        unet_wrapper,
                        morphology_head,
                        mid_features_list,
                        timesteps,
                        accelerator.device,
                    )
                _, fp_pred, _, _ = unified_head(features)

                cls_features = extract_unconditioned_class_features(
                    unet_wrapper, noisy_latent, t_flat, timesteps, cond_expanded,
                    mta_module=mta_module, dit=dit,
                )
                cls_logits, _, _, _ = unified_head(cls_features)
                connectivity_pred = None
                if connectivity_head is not None:
                    connectivity_pred = connectivity_head(cls_features.detach()).cpu().numpy().flatten()
                records.append({
                    "path": img_path,
                    "true_label": gt_label,
                    "pred_label": int(cls_logits.argmax(dim=1).item()),
                    "fingerprint": fp_pred.cpu().numpy().flatten().tolist(),
                    "confidence": float(F.softmax(cls_logits, dim=1).max().item()),
                    "connectivity_target": compute_connectivity_descriptor_np(field).tolist(),
                    "connectivity_pred": connectivity_pred.tolist() if connectivity_pred is not None else None,
                })
                if ace_cond is not None:
                    ace_embeddings.append(
                        ace_cond.detach().float().cpu().numpy().reshape(-1)
                    )
                true_fingerprints_raw.append(true_fp)
                true_fingerprints_norm.append(fp_tensor.cpu().numpy().flatten())
            except Exception as exc:
                logger.warning(f"Test error on {img_path}: {exc}")

    test_output_dir = os.path.join(args.output_dir, "test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    if not records or len(records) != len(image_paths):
        raise RuntimeError(
            f"Incomplete test: {len(records)}/{len(image_paths)} images. "
            "Refusing to report metrics for a partial test set; inspect the per-image errors."
        )
    save_evaluation_protocol(test_output_dir, args, timesteps, "test")

    all_labels = np.asarray([r["true_label"] for r in records], dtype=np.int64)
    all_preds = np.asarray([r["pred_label"] for r in records], dtype=np.int64)
    fingerprints = np.asarray([r["fingerprint"] for r in records], dtype=np.float32)
    true_raw = np.asarray(true_fingerprints_raw, dtype=np.float32)
    true_norm = np.asarray(true_fingerprints_norm, dtype=np.float32)
    if fp_mean is not None and fp_std is not None:
        pred_fingerprints_raw = (
            fingerprints * (np.asarray(fp_std, dtype=np.float32)[None, :] + 1e-8)
            + np.asarray(fp_mean, dtype=np.float32)[None, :]
        )
    else:
        pred_fingerprints_raw = fingerprints.copy()
    logger.info(
        "Test fingerprint spread: target_std=%s, predicted_std=%s",
        np.round(true_raw.std(axis=0), 6),
        np.round(pred_fingerprints_raw.std(axis=0), 6),
    )
    _save_evaluation_visualizations(
        output_dir=test_output_dir,
        all_labels=all_labels,
        all_preds=all_preds,
        fingerprints=fingerprints,
        class_names=class_names,
        ace_embeddings=ace_embeddings,
        morphology_examples=morphology_examples,
    )
    np.save(os.path.join(test_output_dir, "fingerprints.npy"), fingerprints)
    np.save(os.path.join(test_output_dir, "pred_fingerprints_raw.npy"), pred_fingerprints_raw)
    np.save(os.path.join(test_output_dir, "labels.npy"), all_labels)
    np.save(os.path.join(test_output_dir, "true_fingerprints_raw.npy"), true_raw)
    np.save(os.path.join(test_output_dir, "true_fingerprints.npy"), true_norm)
    np.save(os.path.join(test_output_dir, "fingerprint_names.npy"), np.asarray(FINGERPRINT_NAMES))
    with open(os.path.join(test_output_dir, "fingerprint_names.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(FINGERPRINT_NAMES))
    with open(os.path.join(test_output_dir, "class_names.json"), "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    class_mean_fingerprints = []
    for class_id in range(n_classes):
        mask = all_labels == class_id
        class_mean_fingerprints.append(
            true_norm[mask].mean(axis=0) if mask.any() else np.zeros(FP_DIM, dtype=np.float32)
        )
    np.save(
        os.path.join(test_output_dir, "class_mean_fingerprints.npy"),
        np.stack(class_mean_fingerprints).astype(np.float32),
    )
    pd.DataFrame(records).to_csv(
        os.path.join(test_output_dir, "predictions_epoch0.csv"), index=False
    )
    if connectivity_head is not None:
        conn_targets = np.asarray([r["connectivity_target"] for r in records], dtype=np.float32)
        conn_preds = np.asarray([r["connectivity_pred"] for r in records], dtype=np.float32)
        np.save(os.path.join(test_output_dir, "connectivity_targets.npy"), conn_targets)
        np.save(os.path.join(test_output_dir, "connectivity_predictions.npy"), conn_preds)
        logger.info(f"Test connectivity MAE={np.abs(conn_preds - conn_targets).mean():.4f}")

    if trajectory_record is not None:
        _save_diffusion_trajectory_data(trajectory_record, test_output_dir)
        plot_diffusion_trajectory(
            trajectory_record,
            os.path.join(test_output_dir, "diffusion_morphology_trajectory.png"),
        )

    logger.info("Four-class test evaluation completed.")


def run_single_experiment(args):
    if args.ablation not in ABLATION_MODES:
        raise ValueError(f"Expected one of {ABLATION_MODES}, got {args.ablation!r}.")

    project_root = resolve_project_root(args)
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or os.path.join(
        project_root, "models", "stable-diffusion-v1-5"
    )
    dataset_root = os.path.join(project_root, "data", "example_data_train_val_test")
    args.dataset_name = args.dataset_name or os.path.join(dataset_root, "train")
    ablation_root = resolve_ablation_root(args, project_root)
    args.output_dir = os.path.join(
        ablation_root,
        args.ablation,
        f"seed_{int(args.seed)}",
    )
    args.validation_img_dir = args.validation_img_dir or os.path.join(dataset_root, "val")
    args.test_img_dir = args.test_img_dir or os.path.join(dataset_root, "test")
    args.run_test = True
    args.auto_generate_masks = True
    args.morphology_field_dir = os.path.join(
        dataset_root, "morphology_fields_tc4_2d_four_class_v3"
    )

    FORCE_REGENERATE = False
    FP_EPS = 0.001

    args.resolution = 512
    args.train_batch_size = 2
    args.gradient_accumulation_steps = 4
    args.learning_rate = 1e-5
    args.max_train_steps = args.max_train_steps or 1600
    args.stage1_steps = 800
    args.stage2_steps = args.max_train_steps - args.stage1_steps
    args.validation_steps = 200
    args.mixed_precision = "no"
    args.caption_column = "label"
    args.mtfa_num_timesteps = 3
    args.dataloader_num_workers = 0

    args.use_mtfa = True
    args.use_dit = True
    args.use_pqa = True
    args.use_ace = True
    args.use_dcl = True
    args.use_unet_lora = True
    args.lora_rank = 8
    args.lora_alpha = 8.0
    args.lora_dropout = 0.0
    args.dcl_temperature = 0.2
    args.dcl_output_dim = 128
    args.dcl_warmup_steps = 50

    args.stage1_guidance_dropout = 0.1
    args.stage2_guidance_dropout = 0.2
    args.stage1_lambda_diff = 1.0
    args.stage1_lambda_cls = 0.7
    args.stage1_lambda_field = 0.0
    args.stage1_lambda_fingerprint = 0.0
    args.stage1_lambda_contrast = 0.0
    args.stage1_lambda_separation = 0.0
    args.stage1_lambda_fp_decorrelation = 0.0
    args.stage1_lambda_fp_head_orthogonal = 0.0
    args.stage2_lambda_diff = 0.2
    args.stage2_lambda_cls = 1.2
    args.stage2_lambda_field = 0.05
    args.stage2_lambda_fingerprint = 0.4
    args.stage2_lambda_contrast = 0.05
    args.stage2_lambda_separation = 0.05
    args.stage2_lambda_fp_decorrelation = 0.05
    args.stage2_lambda_fp_head_orthogonal = 0.05
    args.lambda_fp_cls = 0.0
    args.use_connectivity_aux = True
    args.lambda_connectivity = 0.15
    args.connectivity_loss_beta = 0.1
    # Basket_weave and Discontinuous share local texture statistics.  Keep an
    # explicit margin between their logits and train the shared feature map to
    # represent the measured continuity descriptors.
    args.lambda_connectivity_pair = 0.25
    args.connectivity_pair_margin = 0.35
    args.lr_scheduler = "cosine"
    args.lr_warmup_steps = 50
    args.max_grad_norm = 0.5
    args.save_at_end = True
    args.save_sd_pipeline = not args.skip_pipeline_export
    args.tracker_project_name = f"fingerprint-diffusion-{args.ablation}"
    args.ace_scale = 1.0
    args = apply_ablation_config(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )

    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    args.runtime_torch_version = str(torch.__version__)
    args.runtime_disable_native_jit = os.environ.get("TORCH_DISABLE_NATIVE_JIT", "0")
    logger.info(
        "Runtime: torch=%s, TORCH_DISABLE_NATIVE_JIT=%s (1 disables native DSL JIT overrides)",
        args.runtime_torch_version,
        args.runtime_disable_native_jit,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Single-GPU training requires a CUDA-enabled PyTorch installation."
        )
    visible_gpu_count = torch.cuda.device_count()
    logger.info(
        f"Single-GPU mode: visible_gpus={visible_gpu_count}, "
        f"processes={accelerator.num_processes}, device={accelerator.device}"
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "ablation_config.json"), "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2, default=str)
        logger.info(
            "Ablation mode=%s, seed=%s, output=%s, use_mtfa=%s, use_ace=%s, use_pqa=%s",
            args.ablation,
            args.seed,
            args.output_dir,
            args.use_mtfa,
            args.use_ace,
            args.use_pqa,
        )

    dataset = load_dataset("imagefolder", data_dir=args.dataset_name)
    source_class_names = dataset["train"].features[args.caption_column].names
    class_names = CANONICAL_CLASS_NAMES
    n_classes_train = len(class_names)
    source_to_canonical = {}
    for source_idx, source_name in enumerate(source_class_names):
        source_name_ascii = _ascii_fold(source_name)
        canonical_name = CLASS_NAME_ALIASES.get(source_name_ascii, source_name_ascii)
        if canonical_name in class_names:
            source_to_canonical[source_idx] = class_names.index(canonical_name)
        elif source_name_ascii.isdigit() and 0 <= int(source_name_ascii) < n_classes_train:
            source_to_canonical[source_idx] = int(source_name_ascii)
        else:
            source_to_canonical[source_idx] = -1
    source_class_names_canonical = [
        CLASS_NAME_ALIASES.get(_ascii_fold(name), _ascii_fold(name))
        for name in source_class_names
    ]
    has_named_classes = all(name in source_class_names_canonical for name in class_names)
    has_numbered_classes = all(str(i) in source_class_names for i in range(n_classes_train))
    if not has_named_classes and not has_numbered_classes:
        raise ValueError(
            "Training set must contain either named class folders "
            f"{class_names} or numbered folders {[str(i) for i in range(n_classes_train)]}. "
            f"Found: {source_class_names}"
        )
    logger.info(f"Source class folders: {source_class_names}")
    logger.info(f"Canonical four-class mapping: {source_to_canonical}")
    source_labels = np.asarray(dataset["train"][args.caption_column], dtype=np.int64)
    class_counts = np.bincount(
        [source_to_canonical.get(int(label), -1) for label in source_labels if source_to_canonical.get(int(label), -1) >= 0],
        minlength=n_classes_train,
    )
    logger.info(f"Four-class training counts: {dict(zip(class_names, class_counts.tolist()))}")
    if np.any(class_counts == 0):
        missing = [class_names[i] for i, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Every class needs at least one training sample. Missing: {missing}")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    weight_dtype = torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    unet_orig = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", local_files_only=True)

    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)
    text_encoder.eval()
    vae.requires_grad_(False)
    vae.to(accelerator.device)
    vae.eval()

    for p in unet_orig.parameters():
        p.requires_grad = False

    lora_target_names = []
    if args.use_unet_lora:
        lora_target_names = inject_lora_into_unet(
            unet_orig,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        logger.info(f"Injected LoRA into {len(lora_target_names)} UNet attention linear layers.")

    if hasattr(unet_orig, "enable_gradient_checkpointing"):
        unet_orig.enable_gradient_checkpointing()
        logger.info("Enabled UNet gradient checkpointing.")

    unet_wrapper = UNetWithMidFeature(unet_orig).to(accelerator.device)
    unet_wrapper.train()

    morphology_head = MorphologyHead().to(accelerator.device)
    unified_head = UnifiedHeadNoSeg_v2(n_classes=n_classes_train, fp_dim=FP_DIM).to(accelerator.device)
    connectivity_head = (
        ConnectivityHead(in_channels=1280).to(accelerator.device)
        if args.use_connectivity_aux else None
    )
    ace = TokenizedACE(embed_dim=768, n_classes=n_classes_train, fp_dim=FP_DIM).to(accelerator.device) if args.use_ace else None
    dit = MidBlockTransformer(channels=1280).to(accelerator.device) if args.use_dit else None
    pqa = PhysicalQueryAttention(feat_channels=1280).to(accelerator.device) if args.use_pqa else None

    K = max(1, args.mtfa_num_timesteps)
    mta_module = MultiTimestepAggregator(feat_dim=1280, fusion_dim=256, max_timesteps=K).to(accelerator.device) if args.use_mtfa else None
    projection_head = ContrastiveProjectionHead(input_dim=1280, hidden_dim=512, output_dim=args.dcl_output_dim).to(accelerator.device) if args.use_dcl else None
    contrastive_loss_fn = DiffusionContrastiveLoss(temperature=args.dcl_temperature, div_threshold=0.6, lambda_div=0.1) if args.use_dcl else None
    fp_cls_head = nn.Linear(FP_DIM, n_classes_train).to(accelerator.device) if args.lambda_fp_cls > 0 else None

    trainable_params = list(morphology_head.parameters()) + list(unified_head.parameters())
    if connectivity_head: trainable_params += list(connectivity_head.parameters())
    if ace: trainable_params += list(ace.parameters())
    if dit: trainable_params += list(dit.parameters())
    if pqa: trainable_params += list(pqa.parameters())
    if mta_module: trainable_params += list(mta_module.parameters())
    if projection_head: trainable_params += list(projection_head.parameters())
    if fp_cls_head: trainable_params += list(fp_cls_head.parameters())
    if args.use_unet_lora:
        lora_params = get_lora_parameters(unet_wrapper)
        trainable_params += lora_params
        logger.info(f"Trainable UNet LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    field_loss_fn = MorphologyFieldLoss(lambda_grad=0.2).to(accelerator.device)

    field_dir = args.morphology_field_dir
    finger_dir = os.path.join(
        field_dir, "fingerprints_2d_beta_lath_four_class_v3"
    )
    os.makedirs(field_dir, exist_ok=True)
    os.makedirs(finger_dir, exist_ok=True)
    fp_stats_path = os.path.join(
        finger_dir, "fp_stats_2d_beta_lath_four_class_v3.npy"
    )

    def cache_filename(path_like):
        path_text = os.fspath(path_like)
        class_folder = _ascii_fold(os.path.basename(os.path.dirname(path_text)))
        image_stem = _ascii_fold(os.path.splitext(os.path.basename(path_text))[0])
        return f"{class_folder}__{image_stem}.npy"

    if accelerator.is_main_process and (FORCE_REGENERATE or not os.path.exists(fp_stats_path)):
        logger.info("Computing global fingerprint statistics...")
        all_fp = []
        for ex in tqdm(dataset["train"], desc="Gathering stats"):
            if source_to_canonical.get(int(ex[args.caption_column]), -1) < 0:
                continue
            img_path = ex["image"].filename
            field = generate_morphology_fields(img_path, mask_size=(128, 128))
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            all_fp.append(compute_extended_fingerprint_np(field, img_rgb))
        all_fp = np.stack(all_fp)
        fp_mean = all_fp.mean(axis=0)
        fp_std = all_fp.std(axis=0)
        np.save(fp_stats_path, (fp_mean, fp_std))
        logger.info(f"Fingerprint mean: {fp_mean}, std: {fp_std}")

    accelerator.wait_for_everyone()
    fp_mean, fp_std = np.load(fp_stats_path)

    if args.auto_generate_masks and accelerator.is_main_process:
        logger.info("Generating morphology fields and normalized fingerprints...")
        for ex in tqdm(dataset["train"], desc="Morphology Fields"):
            if source_to_canonical.get(int(ex[args.caption_column]), -1) < 0:
                continue
            img_path = ex["image"].filename
            base = cache_filename(img_path)
            np_path = os.path.join(field_dir, base)
            fp_path = os.path.join(finger_dir, base.replace(".npy", "_fingerprint.npy"))

            if not FORCE_REGENERATE and os.path.exists(np_path) and os.path.exists(fp_path):
                continue

            field = generate_morphology_fields(img_path, mask_size=(128, 128))
            np.save(np_path, field)
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            fp_raw = compute_extended_fingerprint_np(field, img_rgb)
            fp_norm = (fp_raw - fp_mean) / (fp_std + FP_EPS)
            np.save(fp_path, fp_norm)

    accelerator.wait_for_everyone()

    def load_morphology_field(path_obj):
        if isinstance(path_obj, str):
            fname = os.path.basename(path_obj)
        elif hasattr(path_obj, "filename"):
            fname = os.path.basename(path_obj.filename)
        else:
            return torch.zeros(2, 128, 128)

        source_path = path_obj if isinstance(path_obj, str) else path_obj.filename
        name = cache_filename(source_path)
        np_path = os.path.join(field_dir, name)
        if os.path.exists(np_path):
            return torch.from_numpy(np.load(np_path)).float()
        return torch.zeros(2, 128, 128)

    def load_extended_fingerprint(path_obj):
        if isinstance(path_obj, str):
            fname = os.path.basename(path_obj)
        elif hasattr(path_obj, "filename"):
            fname = os.path.basename(path_obj.filename)
        else:
            return torch.zeros(FP_DIM)

        source_path = path_obj if isinstance(path_obj, str) else path_obj.filename
        name = cache_filename(source_path)
        fp_path = os.path.join(finger_dir, name.replace(".npy", "_fingerprint.npy"))
        if os.path.exists(fp_path):
            return torch.from_numpy(np.load(fp_path)).float()
        return torch.zeros(FP_DIM)

    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = [train_transforms(img) for img in images]
        class_ids = []
        fields = []
        fingerprints = []
        connectivity_descriptors = []

        for i in range(len(images)):
            source_label = int(examples[args.caption_column][i])
            canonical_id = source_to_canonical.get(source_label, -1)
            is_valid_cls = canonical_id >= 0
            class_ids.append(canonical_id if is_valid_cls else 0)
            fields.append(load_morphology_field(examples["image"][i]))
            fingerprints.append(load_extended_fingerprint(examples["image"][i]))
            field_np = (
                fields[-1].detach().float().cpu().numpy()
                if isinstance(fields[-1], torch.Tensor)
                else np.asarray(fields[-1], dtype=np.float32)
            )
            connectivity_descriptors.append(
                torch.from_numpy(compute_connectivity_descriptor_np(field_np))
            )

        examples["class_id"] = torch.tensor(class_ids, dtype=torch.long)
        examples["morphology_field"] = fields
        examples["fingerprint"] = torch.stack(fingerprints)
        examples["connectivity_descriptor"] = torch.stack(connectivity_descriptors)
        return examples

    with accelerator.main_process_first():
        train_dataset_raw = dataset["train"].filter(
            lambda ex: source_to_canonical.get(int(ex[args.caption_column]), -1) >= 0
        )
        train_dataset = train_dataset_raw.with_transform(preprocess_train)
    logger.info(
        f"Main training samples after four-class filtering: {len(train_dataset)} / {len(dataset['train'])}"
    )

    filtered_source_labels = np.asarray(train_dataset_raw[args.caption_column], dtype=np.int64)
    filtered_class_ids = np.asarray(
        [source_to_canonical[int(label)] for label in filtered_source_labels],
        dtype=np.int64,
    )
    sample_weights = 1.0 / np.maximum(class_counts[filtered_class_ids], 1)
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]).float(),
            "class_id": torch.stack([ex["class_id"] for ex in examples]),
            "morphology_field": torch.stack([ex["morphology_field"] for ex in examples]),
            "fingerprint": torch.stack([ex["fingerprint"] for ex in examples]),
            "connectivity_descriptor": torch.stack(
                [ex["connectivity_descriptor"] for ex in examples]
            ),
        }

    logger.info("Four-class classification training is enabled.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = max(
        args.num_train_epochs,
        math.ceil(args.max_train_steps / max(num_update_steps_per_epoch, 1)),
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    models = [unet_wrapper, morphology_head, unified_head]
    if connectivity_head: models.append(connectivity_head)
    if ace: models.append(ace)
    if dit: models.append(dit)
    if pqa: models.append(pqa)
    if mta_module: models.append(mta_module)
    if projection_head: models.append(projection_head)
    if fp_cls_head: models.append(fp_cls_head)

    prepared = accelerator.prepare(*models, optimizer, train_loader, lr_scheduler)
    n_models = len(models)

    unet_wrapper, morphology_head, unified_head = prepared[0], prepared[1], prepared[2]
    idx = 3
    if connectivity_head:
        connectivity_head = prepared[idx]
        idx += 1
    if ace: ace = prepared[idx]; idx += 1
    if dit: dit = prepared[idx]; idx += 1
    if pqa: pqa = prepared[idx]; idx += 1
    if mta_module: mta_module = prepared[idx]; idx += 1
    if projection_head: projection_head = prepared[idx]; idx += 1
    if fp_cls_head: fp_cls_head = prepared[idx]; idx += 1

    optimizer = prepared[n_models]
    train_loader = prepared[n_models + 1]
    lr_scheduler = prepared[n_models + 2]

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    logger.info("***** Running training *****")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    total_train_steps = noise_scheduler.config.num_train_timesteps
    stage1_checkpoint_saved = False
    best_val_accuracy = -1.0

    for epoch in range(args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet_wrapper):
                is_stage1 = global_step < args.stage1_steps
                stage_name = "stage1_unet_class" if is_stage1 else "stage2_morphology_fingerprint"
                lambda_diff = args.stage1_lambda_diff if is_stage1 else args.stage2_lambda_diff
                lambda_cls = args.stage1_lambda_cls if is_stage1 else args.stage2_lambda_cls
                lambda_field = args.stage1_lambda_field if is_stage1 else args.stage2_lambda_field
                lambda_fingerprint = args.stage1_lambda_fingerprint if is_stage1 else args.stage2_lambda_fingerprint
                lambda_contrast = args.stage1_lambda_contrast if is_stage1 else args.stage2_lambda_contrast
                lambda_separation = args.stage1_lambda_separation if is_stage1 else args.stage2_lambda_separation
                lambda_fp_decorrelation = (
                    args.stage1_lambda_fp_decorrelation if is_stage1 else args.stage2_lambda_fp_decorrelation
                )
                lambda_fp_head_orthogonal = (
                    args.stage1_lambda_fp_head_orthogonal if is_stage1 else args.stage2_lambda_fp_head_orthogonal
                )
                guidance_dropout = args.stage1_guidance_dropout if is_stage1 else args.stage2_guidance_dropout

                inputs_x = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                class_ids = batch["class_id"].to(accelerator.device)
                target_field = batch["morphology_field"].to(accelerator.device)
                target_fp = batch["fingerprint"].to(accelerator.device)
                target_connectivity = batch["connectivity_descriptor"].to(accelerator.device)
                B = inputs_x.shape[0]

                latents = vae.encode(inputs_x).latent_dist.sample() * vae.config.scaling_factor

                encoder_hidden_states, ace_cond = build_scheme_a_condition(
                    tokenizer,
                    text_encoder,
                    ace,
                    target_fp,
                    class_ids,
                    accelerator.device,
                    ace_scale=args.ace_scale,
                    use_zero_fingerprint=is_stage1,
                )

                if random.random() < guidance_dropout:
                    encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

                K = max(1, args.mtfa_num_timesteps)
                t_matrix = sample_multi_timesteps(B, total_train_steps, K, accelerator.device)

                latents_expanded = latents.repeat_interleave(K, dim=0)
                t_flat = t_matrix.reshape(-1)
                noise = torch.randn_like(latents_expanded)
                noisy_batch = noise_scheduler.add_noise(latents_expanded, noise, t_flat)
                cond_expanded = encoder_hidden_states.repeat_interleave(K, dim=0)

                # Both feature passes retain their graphs until the combined backward.
                with accelerator.autocast():
                    cls_features = extract_unconditioned_class_features(
                        unet_wrapper,
                        noisy_batch,
                        t_flat,
                        t_matrix,
                        cond_expanded,
                        mta_module=mta_module if args.use_mtfa else None,
                        dit=dit,
                    )
                cls_logits, _, _, _ = unified_head(cls_features)
                loss_cls = F.cross_entropy(
                    cls_logits,
                    class_ids,
                    label_smoothing=0.03,
                )
                loss_connectivity = torch.tensor(0.0, device=accelerator.device)
                if connectivity_head is not None:
                    # Keep this branch connected to cls_features.  Detaching it
                    # trains only the small auxiliary head and provides no
                    # continuity supervision to the actual classifier.
                    connectivity_pred = connectivity_head(cls_features)
                    loss_connectivity = F.smooth_l1_loss(
                        connectivity_pred,
                        target_connectivity,
                        beta=args.connectivity_loss_beta,
                    )
                loss_connectivity_pair = torch.tensor(0.0, device=accelerator.device)
                if args.lambda_connectivity_pair > 0:
                    loss_connectivity_pair = pairwise_class_margin_loss(
                        cls_logits,
                        class_ids,
                        class_a=0,
                        class_b=1,
                        margin=args.connectivity_pair_margin,
                    )

                with accelerator.autocast():
                    noise_pred_batch, mid_features_batch = unet_wrapper(
                        noisy_batch, t_flat, cond_expanded
                    )
                loss_diff_raw = F.mse_loss(noise_pred_batch, noise)
                loss = lambda_diff * loss_diff_raw

                mid_features_list = split_repeated_features_by_timestep(mid_features_batch, B, K)

                loss_contrast = torch.tensor(0.0, device=accelerator.device)
                if (not is_stage1) and args.use_dcl and projection_head is not None and len(mid_features_list) > 1:
                    z_list = [projection_head(f.float()) for f in mid_features_list]
                    z_stacked = torch.stack(z_list, dim=1)
                    loss_contrast_raw = contrastive_loss_fn(z_stacked, class_ids)
                    contrast_warmup = min(1.0, global_step / max(args.dcl_warmup_steps, 1))
                    loss_contrast = lambda_contrast * contrast_warmup * loss_contrast_raw
                    loss += loss_contrast

                if args.use_mtfa and mta_module is not None and len(mid_features_list) > 1:
                    features = mta_module(mid_features_list, t_matrix)
                else:
                    features = torch.stack(mid_features_list, dim=0).mean(dim=0)

                features = features.float().clamp(-50, 50)

                if dit:
                    features = dit(features)

                if pqa:
                    features = pqa(features, ace_cond)

                pred_field = morphology_head(features.detach())
                pred_field_up = F.interpolate(pred_field, size=target_field.shape[2:], mode="bilinear")
                loss_field = field_loss_fn(pred_field_up, target_field) + 0.01 * torch.mean(pred_field_up ** 2)
                loss += lambda_field * loss_field

                _, fingerprint_pred, _, fp_feat = unified_head(features)

                loss_fingerprint_base = hybrid_fingerprint_loss(fingerprint_pred, target_fp, beta=0.1)
                loss_fingerprint_reg = 0.001 * torch.mean(fingerprint_pred ** 2)

                target_std = target_fp.std(dim=0, unbiased=False).detach()
                pred_std = fingerprint_pred.std(dim=0, unbiased=False)
                loss_fp_dist = F.mse_loss(pred_std, target_std)
                loss_fp_var = F.relu(0.5 * target_std - pred_std).mean()

                loss_fingerprint = (
                    loss_fingerprint_base
                    + 2.0 * loss_fp_dist
                    + loss_fp_var
                    + loss_fingerprint_reg
                )

                loss_fp_decor = fingerprint_decorrelation_loss(fingerprint_pred)
                loss_fp_head_orth = fingerprint_head_orthogonality_loss(unified_head)

                loss_fp_cls = torch.tensor(0.0, device=accelerator.device)
                if fp_cls_head is not None:
                    fp_cls_logits = fp_cls_head(fingerprint_pred)
                    loss_fp_cls = F.cross_entropy(fp_cls_logits, class_ids)

                loss_separation = torch.tensor(0.0, device=accelerator.device)
                if fingerprint_pred.shape[0] > 1 and len(class_ids.unique()) > 1:
                    centers = []
                    for class_id in class_ids.unique():
                        centers.append(fingerprint_pred[class_ids == class_id].mean(dim=0))
                    centers = torch.stack(centers, dim=0)
                    sim = F.cosine_similarity(centers.unsqueeze(1), centers.unsqueeze(0), dim=-1)
                    mask_sep = ~torch.eye(sim.size(0), device=sim.device).bool()
                    loss_separation = (sim[mask_sep] + 1).mean()

                fp_weight = min(1.0, global_step / 50)

                loss += lambda_cls * loss_cls
                if connectivity_head is not None:
                    loss += args.lambda_connectivity * loss_connectivity
                if args.lambda_connectivity_pair > 0:
                    loss += args.lambda_connectivity_pair * loss_connectivity_pair
                loss += fp_weight * lambda_fingerprint * loss_fingerprint
                loss += lambda_separation * loss_separation
                loss += args.lambda_fp_cls * loss_fp_cls
                loss += lambda_fp_decorrelation * loss_fp_decor
                loss += lambda_fp_head_orthogonal * loss_fp_head_orth

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss, skip.")
                    continue

                train_loss += loss.detach().float().item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        list(morphology_head.parameters())
                        + list(unified_head.parameters())
                    )
                    if connectivity_head:
                        params_to_clip += list(connectivity_head.parameters())
                    if ace: params_to_clip += list(ace.parameters())
                    if dit: params_to_clip += list(dit.parameters())
                    if pqa: params_to_clip += list(pqa.parameters())
                    if mta_module: params_to_clip += list(mta_module.parameters())
                    if projection_head: params_to_clip += list(projection_head.parameters())
                    if fp_cls_head: params_to_clip += list(fp_cls_head.parameters())
                    if args.use_unet_lora: params_to_clip += get_lora_parameters(unet_wrapper)

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    if accelerator.is_main_process and global_step % 50 == 0:
                        cls_acc = (cls_logits.argmax(dim=1) == class_ids).float().mean().item()
                        pred_fp_std_log = fingerprint_pred.std(dim=0).detach().cpu().numpy()
                        logger.info(
                            f"Step {global_step} [{stage_name}]: Loss={loss.item():.4f}, "
                            f"ClsAcc={cls_acc:.3f}, "
                            f"FP std={np.round(pred_fp_std_log, 3)}, "
                            f"Diff={loss_diff_raw.item():.3f}, "
                            f"Field={loss_field.item():.3f}, "
                            f"Fp={loss_fingerprint.item():.3f}, "
                            f"Conn={loss_connectivity.item():.3f}, "
                            f"ConnPair={loss_connectivity_pair.item():.3f}, "
                            f"Decor={loss_fp_decor.item():.3f}, "
                            f"HeadOrth={loss_fp_head_orth.item():.3f}, "
                            f"Cls={loss_cls.item():.3f}, "
                            f"Sep={loss_separation.item():.3f}, "
                            f"FpCls={loss_fp_cls.item():.3f}"
                        )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and (not stage1_checkpoint_saved) and global_step >= args.stage1_steps:
                    stage1_dir = os.path.join(args.output_dir, "stage1_class_conditioned_unet")
                    os.makedirs(stage1_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(unet_wrapper).state_dict(),
                        os.path.join(stage1_dir, "unet_stage1.pth"),
                    )
                    if args.use_unet_lora:
                        torch.save(
                            get_lora_state_dict(accelerator.unwrap_model(unet_wrapper)),
                            os.path.join(stage1_dir, "unet_lora_stage1.pth"),
                        )
                    logger.info(f"Saved stage-1 class-conditioned UNet checkpoint to {stage1_dir}")
                    stage1_checkpoint_saved = True

                if accelerator.is_main_process and (
                    global_step % args.validation_steps == 0 or global_step >= args.max_train_steps
                ):
                    torch.save(accelerator.unwrap_model(unet_wrapper).state_dict(), os.path.join(args.output_dir, f"unet_step{global_step}.pth"))
                    if ace:
                        torch.save(accelerator.unwrap_model(ace).state_dict(), os.path.join(args.output_dir, f"ace_step{global_step}.pth"))
                    if mta_module:
                        torch.save(accelerator.unwrap_model(mta_module).state_dict(), os.path.join(args.output_dir, f"mta_step{global_step}.pth"))
                    if projection_head:
                        torch.save(accelerator.unwrap_model(projection_head).state_dict(), os.path.join(args.output_dir, f"proj_step{global_step}.pth"))
                    if args.use_unet_lora:
                        torch.save(
                            get_lora_state_dict(accelerator.unwrap_model(unet_wrapper)),
                            os.path.join(args.output_dir, f"unet_lora_step{global_step}.pth"),
                        )
                    torch.save(
                        accelerator.unwrap_model(unified_head).state_dict(),
                        os.path.join(args.output_dir, f"unified_head_step{global_step}.pth"),
                    )
                    torch.save(
                        accelerator.unwrap_model(morphology_head).state_dict(),
                        os.path.join(args.output_dir, f"morphology_head_step{global_step}.pth"),
                    )
                    if connectivity_head:
                        torch.save(
                            accelerator.unwrap_model(connectivity_head).state_dict(),
                            os.path.join(args.output_dir, f"connectivity_head_step{global_step}.pth"),
                        )
                    if dit:
                        torch.save(
                            accelerator.unwrap_model(dit).state_dict(),
                            os.path.join(args.output_dir, f"dit_step{global_step}.pth"),
                        )
                    if pqa:
                        torch.save(
                            accelerator.unwrap_model(pqa).state_dict(),
                            os.path.join(args.output_dir, f"pqa_step{global_step}.pth"),
                        )
                    if fp_cls_head:
                        torch.save(
                            accelerator.unwrap_model(fp_cls_head).state_dict(),
                            os.path.join(args.output_dir, f"fp_cls_head_step{global_step}.pth"),
                        )

                    try:
                        val_accuracy = run_validation(
                            vae,
                            unet_wrapper,
                            unified_head,
                            morphology_head,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            noise_scheduler,
                            class_names,
                            mta_module=mta_module,
                            ace=ace,
                            dit=dit,
                            pqa=pqa,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            fp_mean=fp_mean,
                            fp_std=fp_std,
                            connectivity_head=connectivity_head,
                        )
                        logger.info(f"Step {global_step}: validation accuracy={val_accuracy:.4f}")
                        if math.isfinite(val_accuracy) and val_accuracy > best_val_accuracy:
                            best_checkpoint = {
                                "unet": accelerator.unwrap_model(unet_wrapper).state_dict(),
                                "unified_head": accelerator.unwrap_model(unified_head).state_dict(),
                                "morphology_head": accelerator.unwrap_model(morphology_head).state_dict(),
                                "connectivity_head": (
                                    accelerator.unwrap_model(connectivity_head).state_dict()
                                    if connectivity_head else None
                                ),
                                "ace": accelerator.unwrap_model(ace).state_dict() if ace else None,
                                "dit": accelerator.unwrap_model(dit).state_dict() if dit else None,
                                "pqa": accelerator.unwrap_model(pqa).state_dict() if pqa else None,
                                "mta": accelerator.unwrap_model(mta_module).state_dict() if mta_module else None,
                                "projection_head": accelerator.unwrap_model(projection_head).state_dict() if projection_head else None,
                                "fp_cls_head": accelerator.unwrap_model(fp_cls_head).state_dict() if fp_cls_head else None,
                                "class_names": class_names,
                                "fingerprint_names": FINGERPRINT_NAMES,
                                "fp_mean": fp_mean,
                                "fp_std": fp_std,
                                "step": global_step,
                                "validation_accuracy": val_accuracy,
                                "evaluation_seed": args.eval_seed,
                                "selection_metric": "validation_accuracy",
                            }
                            checkpoint_path = os.path.join(args.output_dir, "best_four_class_model.pt")
                            torch.save(best_checkpoint, checkpoint_path + ".tmp")
                            os.replace(checkpoint_path + ".tmp", checkpoint_path)
                            best_val_accuracy = val_accuracy
                            logger.info(f"Saved best validation model: step={global_step}, accuracy={val_accuracy:.4f}")
                            del best_checkpoint
                    except Exception as e:
                        logger.warning(f"Validation failed: {e}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if best_val_accuracy < 0:
            raise RuntimeError(
                "No complete validation succeeded in this run. Refusing to use a stale "
                "checkpoint or test the last training weights. Inspect validation errors."
            )
        best_path = os.path.join(args.output_dir, "best_four_class_model.pt")
        selected = load_best_checkpoint_for_evaluation(
            best_path, accelerator,
            {"unet": unet_wrapper, "unified_head": unified_head,
             "morphology_head": morphology_head, "connectivity_head": connectivity_head,
             "ace": ace, "dit": dit, "pqa": pqa, "mta": mta_module,
             "projection_head": projection_head, "fp_cls_head": fp_cls_head},
            class_names,
        )
        fp_mean, fp_std = selected["fp_mean"], selected["fp_std"]
        args.selected_checkpoint = os.path.abspath(best_path)
        args.selected_checkpoint_step = int(selected["step"])
        args.selected_validation_accuracy = float(selected["validation_accuracy"])
        logger.info(
            f"Loaded best validation model for export and test: {best_path}; "
            f"step={args.selected_checkpoint_step}, validation_accuracy={args.selected_validation_accuracy:.4f}"
        )
        del selected
        with open(os.path.join(args.output_dir, "selected_model.json"), "w", encoding="utf-8") as file:
            json.dump({"checkpoint": args.selected_checkpoint,
                       "step": args.selected_checkpoint_step,
                       "validation_accuracy": args.selected_validation_accuracy,
                       "selection_metric": "validation_accuracy",
                       "evaluation_seed": args.eval_seed,
                       "final_files_contain": "best validation weights"}, file, indent=2)
        with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as file:
            json.dump(class_names, file, indent=2)
        np.savez_compressed(
            os.path.join(args.output_dir, "fingerprint_stats_2d_four_class.npz"),
            mean=np.asarray(fp_mean, dtype=np.float32),
            std=np.asarray(fp_std, dtype=np.float32),
            fingerprint_names=np.asarray(FINGERPRINT_NAMES),
            class_names=np.asarray(class_names),
        )
        torch.save(
            accelerator.unwrap_model(unet_wrapper).state_dict(),
            os.path.join(args.output_dir, "unet_final.pth"),
        )
        torch.save(
            accelerator.unwrap_model(unified_head).state_dict(),
            os.path.join(args.output_dir, "unified_head_final.pth"),
        )
        torch.save(
            accelerator.unwrap_model(morphology_head).state_dict(),
            os.path.join(args.output_dir, "morphology_head_final.pth"),
        )
        if connectivity_head:
            torch.save(
                accelerator.unwrap_model(connectivity_head).state_dict(),
                os.path.join(args.output_dir, "connectivity_head_final.pth"),
            )

        if ace:
            torch.save(
                accelerator.unwrap_model(ace).state_dict(),
                os.path.join(args.output_dir, "ace_final.pth"),
            )
        if dit:
            torch.save(
                accelerator.unwrap_model(dit).state_dict(),
                os.path.join(args.output_dir, "dit_final.pth"),
            )
        if pqa:
            torch.save(
                accelerator.unwrap_model(pqa).state_dict(),
                os.path.join(args.output_dir, "pqa_final.pth"),
            )
        if mta_module:
            torch.save(
                accelerator.unwrap_model(mta_module).state_dict(),
                os.path.join(args.output_dir, "mta_final.pth"),
            )
        if projection_head:
            torch.save(
                accelerator.unwrap_model(projection_head).state_dict(),
                os.path.join(args.output_dir, "proj_final.pth"),
            )
        if fp_cls_head:
            torch.save(
                accelerator.unwrap_model(fp_cls_head).state_dict(),
                os.path.join(args.output_dir, "fp_cls_head_final.pth"),
            )
        if args.use_unet_lora:
            torch.save(
                get_lora_state_dict(accelerator.unwrap_model(unet_wrapper)),
                os.path.join(args.output_dir, "unet_lora_final.pth"),
            )

        # Save the large aggregate archive only after all independently usable files exist.
        full_checkpoint = {
            "unet": accelerator.unwrap_model(unet_wrapper).state_dict(),
            "ace": accelerator.unwrap_model(ace).state_dict() if ace else None,
            "dit": accelerator.unwrap_model(dit).state_dict() if dit else None,
            "pqa": accelerator.unwrap_model(pqa).state_dict() if pqa else None,
            "mta": accelerator.unwrap_model(mta_module).state_dict() if mta_module else None,
            "projection_head": accelerator.unwrap_model(projection_head).state_dict() if projection_head else None,
            "unified_head": accelerator.unwrap_model(unified_head).state_dict(),
            "morphology_head": accelerator.unwrap_model(morphology_head).state_dict(),
            "connectivity_head": (
                accelerator.unwrap_model(connectivity_head).state_dict()
                if connectivity_head else None
            ),
            "fp_cls_head": accelerator.unwrap_model(fp_cls_head).state_dict() if fp_cls_head else None,
            "unet_lora": get_lora_state_dict(accelerator.unwrap_model(unet_wrapper)) if args.use_unet_lora else None,
            "lora_target_names": lora_target_names,
            "fp_mean": fp_mean,
            "fp_std": fp_std,
            "fp_dim": FP_DIM,
            "fingerprint_names": FINGERPRINT_NAMES,
            "class_names": class_names,
            "training_strategy": "two_stage_four_class_alpha_coarsened",
            "connectivity_names": CONNECTIVITY_NAMES,
            "connectivity_head_input": "unconditioned_mid_features_with_training_gradients",
            "connectivity_head_independent": False,
            "selected_checkpoint_step": args.selected_checkpoint_step,
            "selected_validation_accuracy": args.selected_validation_accuracy,
            "connectivity_auxiliary_loss": {
                "lambda_connectivity": args.lambda_connectivity,
                "lambda_connectivity_pair": args.lambda_connectivity_pair,
                "pair": ["Basket_weave", "Discontinuous"],
                "margin": args.connectivity_pair_margin,
            },
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "args": vars(args),
        }
        try:
            torch.save(full_checkpoint, os.path.join(args.output_dir, "full_model.pt"))
        except Exception as e:
            logger.exception(f"Aggregate full_model.pt save failed; individual checkpoints remain usable: {e}")

    if accelerator.is_main_process and args.run_test and args.test_img_dir is not None:
        run_test(
            vae,
            unet_wrapper,
            unified_head,
            morphology_head,
            args,
            accelerator,
            weight_dtype,
            class_names,
            mta_module=mta_module,
            ace=ace,
            dit=dit,
            pqa=pqa,
            projection_head=projection_head,
            noise_scheduler=noise_scheduler,
            fp_mean=fp_mean,
            fp_std=fp_std,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            connectivity_head=connectivity_head,
        )

        # Final report generation is intentionally disabled; run_test already writes
        # the core evaluation figures under test_results.

    accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.save_at_end and args.save_sd_pipeline:
        # Pipeline export is a packaging step. It must not invalidate a completed
        # training/test run when the local base model lacks an optional processor
        # file such as preprocessor_config.json.
        pipeline_dir = os.path.join(args.output_dir, "two_stage_sd_pipeline")
        try:
            from diffusers import StableDiffusionPipeline

            export_unet = copy.deepcopy(accelerator.unwrap_model(unet_wrapper).unet).to("cpu")
            if args.use_unet_lora:
                export_unet = merge_lora_into_unet(export_unet)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=export_unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                local_files_only=True,
            )
            pipeline.save_pretrained(pipeline_dir)
            logger.info("Saved Stable Diffusion pipeline to %s", pipeline_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning(
                "Pipeline export skipped; training and test metrics are already complete. "
                "Base model may lack optional processor files: %s",
                exc,
            )

    accelerator.end_training()


def run_ablation_suite(args):
    """Run all variants in isolated processes, then aggregate their test metrics."""
    project_root = resolve_project_root(args)
    ablation_root = resolve_ablation_root(args, project_root)
    seeds = parse_ablation_seeds(args.seeds, args.seed)
    script_path = os.path.abspath(__file__)
    failures = []

    for seed in seeds:
        for mode in ABLATION_MODES:
            metrics_path = os.path.join(
                ablation_root,
                mode,
                f"seed_{seed}",
                "test_results",
                "per_class_metrics.json",
            )
            if args.skip_completed and os.path.isfile(metrics_path):
                print(f"[Ablation] Skip completed run: mode={mode}, seed={seed}")
                continue

            command = [
                sys.executable,
                script_path,
                "--ablation",
                mode,
                "--project-root",
                project_root,
                "--ablation-root",
                ablation_root,
                "--seed",
                str(seed),
                "--eval_seed",
                str(args.eval_seed),
                "--num_train_epochs",
                str(args.num_train_epochs),
            ]
            if args.max_train_steps is not None:
                command.extend(["--max_train_steps", str(args.max_train_steps)])
            optional_paths = {
                "--pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "--dataset_name": args.dataset_name,
                "--validation_img_dir": args.validation_img_dir,
                "--test_img_dir": args.test_img_dir,
                "--test_prompt_dir": args.test_prompt_dir,
            }
            for flag, value in optional_paths.items():
                if value:
                    command.extend([flag, os.path.abspath(os.path.expanduser(value))])
            if args.skip_pipeline_export:
                command.append("--skip-pipeline-export")

            print(f"[Ablation] Starting mode={mode}, seed={seed}")
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append({"mode": mode, "seed": int(seed), "returncode": exc.returncode})
                logger.error(
                    "Ablation run failed; continuing with remaining variants: mode=%s, seed=%s, returncode=%s",
                    mode,
                    seed,
                    exc.returncode,
                )

    if failures:
        details = "; ".join(
            f"mode={item['mode']}, seed={item['seed']}, returncode={item['returncode']}"
            for item in failures
        )
        raise RuntimeError(
            "One or more ablation runs failed after the remaining runs were attempted: "
            + details
        )

    return summarize_ablation_results(ablation_root, seeds)


def main():
    args = parse_args()
    project_root = resolve_project_root(args)
    ablation_root = resolve_ablation_root(args, project_root)
    seeds = parse_ablation_seeds(args.seeds, args.seed)

    if args.ablation == "all":
        run_ablation_suite(args)
        return
    if args.ablation == "summary":
        summarize_ablation_results(ablation_root, seeds)
        return

    if args.skip_completed:
        completed_metrics = os.path.join(
            ablation_root,
            args.ablation,
            f"seed_{int(args.seed)}",
            "test_results",
            "per_class_metrics.json",
        )
        if os.path.isfile(completed_metrics):
            print(
                f"[Ablation] Skip completed run: mode={args.ablation}, "
                f"seed={args.seed}; existing metrics={completed_metrics}"
            )
            return
    run_single_experiment(args)


if __name__ == "__main__":
    main()
