"""Evaluation image loading utilities for the four-class training pipeline."""

import os
import unicodedata
from typing import List, Optional, Tuple


CANONICAL_CLASS_NAMES = [
    "Basket_weave",
    "Discontinuous",
    "Widmanstatten",
    "Alpha_coarsened_Widmanstatten",
]

CLASS_NAME_ALIASES = {
    "Basket_weave": "Basket_weave",
    "BasketWeave": "Basket_weave",
    "Discontinuous": "Discontinuous",
    "Widmanstatten": "Widmanstatten",
    "Lamellar": "Widmanstatten",
    "Alpha_coarsened_Widmanstatten": "Alpha_coarsened_Widmanstatten",
    "AlphaCoarsenedWidmanstatten": "Alpha_coarsened_Widmanstatten",
}

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def _ascii_fold(value: str) -> str:
    """Normalize a path component for stable folder-name matching."""
    return (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def canonical_class_id_from_path(
    path_like: str, fallback: Optional[int] = None
) -> int:
    """Resolve the immediate parent folder to the canonical class ID."""
    path_text = _ascii_fold(path_like).replace("\\", "/").rstrip("/")
    path_parts = [part for part in path_text.split("/") if part]
    folder_name = path_parts[-2] if len(path_parts) >= 2 else ""

    canonical_name = CLASS_NAME_ALIASES.get(folder_name, folder_name)
    if canonical_name in CANONICAL_CLASS_NAMES:
        return CANONICAL_CLASS_NAMES.index(canonical_name)

    if folder_name.isdigit():
        numeric_label = int(folder_name)
        if 0 <= numeric_label < len(CANONICAL_CLASS_NAMES):
            return numeric_label

    if fallback is not None:
        fallback_label = int(fallback)
        if 0 <= fallback_label < len(CANONICAL_CLASS_NAMES):
            return fallback_label

    raise ValueError(
        f"Cannot map evaluation folder to a canonical class: {folder_name!r}"
    )


def load_validation_data(
    args,
    accelerator=None,
    weight_dtype=None,
    use_cache: bool = True,
    img_dir: Optional[str] = None,
    prompt_dir: Optional[str] = None,
) -> Tuple[List[str], List[int], List[int], List[str], List[int]]:
    """Load class-organized validation or test images.

    The return order matches the current training evaluator:
    ``image_paths, labels, labels, unique_names, unique_labels``.
    The duplicated label list is retained because existing callers use the
    third value as the prompt/class-condition list. Text embeddings are not
    created here; the current training program builds its conditioning through
    its own fingerprint and prompt-conditioning path.
    """
    del accelerator, weight_dtype, use_cache, prompt_dir

    root = img_dir or getattr(args, "validation_img_dir", None)
    if not root:
        root = getattr(args, "test_img_dir", None)
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"Evaluation image directory not found: {root}")

    image_paths: List[str] = []
    labels: List[int] = []

    for current_root, _, files in os.walk(root):
        for filename in sorted(files):
            extension = os.path.splitext(filename)[1].lower()
            if extension not in VALID_IMAGE_EXTENSIONS:
                continue

            path = os.path.join(current_root, filename)
            try:
                label = canonical_class_id_from_path(path)
            except ValueError:
                continue

            image_paths.append(path)
            labels.append(label)

    if not image_paths:
        raise ValueError(
            f"No class-organized evaluation images found under: {root}"
        )

    unique_labels = sorted(set(labels))
    unique_names = [CANONICAL_CLASS_NAMES[label] for label in unique_labels]
    return image_paths, labels, labels, unique_names, unique_labels


__all__ = [
    "CANONICAL_CLASS_NAMES",
    "CLASS_NAME_ALIASES",
    "VALID_IMAGE_EXTENSIONS",
    "canonical_class_id_from_path",
    "load_validation_data",
]
