# train_large_field.py
# -*- coding: utf-8 -*-
"""Native 1536-pixel, full-UNet micrograph fine tuning with cached PSIM.

Set LargeFieldConfig.initial_diffusers_model_dir to an existing high-quality
conditional pipeline (best/ or epoch_N/ema/) to inherit its complete weights.
Otherwise the original micrograph LoRA is verified and merged into the base.
v4 checkpoints require restore_generation_weights(), not a LoRA-only loader.

Contrast terminology revision: Dark-contrast area fraction replaces the legacy
beta_phase_fraction label. Bright-contrast area fraction is its complement.
The first channel's masks/numbers and α lath-width measurement are unchanged.
Legacy PSIM arrays, recipe keys and v4 checkpoint wire schemas remain supported.
"""

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import hashlib
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from contextlib import contextmanager, nullcontext, ExitStack

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from skimage.filters import frangi
from skimage.morphology import medial_axis, skeletonize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


logger = get_logger(__name__, log_level="INFO")
# Accelerate's logger cannot be used until Accelerator initializes its global
# state in main(). Use the standard logger for import-time fallback messages.
bootstrap_logger = logging.getLogger(f"{__name__}.bootstrap")

# The contract is shared by the training, analysis, and inverse-design programs.
# When this script is launched from ``LEE_A/Diffusion``, the contract may live in
# either that directory or the project root.  Resolve it from the script location
# instead of depending on the caller's working directory or PYTHONPATH.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT_ENV = os.environ.get("LEE_A_PROJECT_ROOT") or os.environ.get("MORPHOLOGY_PROJECT_ROOT")
_CONTRACT_SEARCH_DIRS = []
if _PROJECT_ROOT_ENV:
    _CONTRACT_SEARCH_DIRS.append(Path(_PROJECT_ROOT_ENV).expanduser())
_CONTRACT_SEARCH_DIRS.extend([_SCRIPT_DIR, *_SCRIPT_DIR.parents[:5]])
for _candidate_dir in _CONTRACT_SEARCH_DIRS:
    if (_candidate_dir / "structure_performance_contract.py").is_file():
        if str(_candidate_dir) not in sys.path:
            sys.path.insert(0, str(_candidate_dir))
        break

try:
    _contract = importlib.import_module("structure_performance_contract")
    CLASS_NAMES = _contract.CLASS_NAMES
    CONTRACT_VERSION = _contract.CONTRACT_VERSION
    FINGERPRINT_LABELS = _contract.FINGERPRINT_LABELS
    CONNECTIVITY_LABELS = _contract.CONNECTIVITY_LABELS
    STRUCTURE_FEATURES = _contract.STRUCTURE_FEATURES
except (ImportError, AttributeError):
    # Keep this entry point runnable when only the files shown in the Diffusion
    # folder are copied to the server.  These values intentionally match the
    # versioned shared module; adding that module later will transparently take
    # precedence over this compatibility fallback.
    bootstrap_logger.info(
        "Using the built-in four-class 2-D contract; no extra shared module is required."
    )
    CONTRACT_VERSION = "four-class-2d-directional-v1"
    CLASS_NAMES = (
        "Basket_weave",
        "Discontinuous",
        "Widmanstatten",
        "Alpha_coarsened_Widmanstatten",
    )
    FINGERPRINT_LABELS = ("beta_phase_fraction", "mean_lath_width_um")
    CONNECTIVITY_LABELS = (
        "connectivity_length",
        "fragmentation_degree",
        "orientation_consistency",
    )
    STRUCTURE_FEATURES = (
        *FINGERPRINT_LABELS,
        *CONNECTIVITY_LABELS,
        *(f"{name}_fraction" for name in CLASS_NAMES),
    )


# Canonical reporting names; the first descriptor remains the SAME dark mask.
# Normalize both old and new shared contracts without changing channel order.
DARK_CONTRAST_FIELD = "dark_contrast_area_fraction"
BRIGHT_CONTRAST_FIELD = "bright_contrast_area_fraction"
LEGACY_DARK_CONTRAST_FIELD = "beta_phase_fraction"
LEGACY_FP_NAMES = [LEGACY_DARK_CONTRAST_FIELD, "mean_lath_width_um"]
FINGERPRINT_LABELS = tuple(
    DARK_CONTRAST_FIELD if name == LEGACY_DARK_CONTRAST_FIELD else name
    for name in FINGERPRINT_LABELS
)
STRUCTURE_FEATURES = tuple(
    DARK_CONTRAST_FIELD if name == LEGACY_DARK_CONTRAST_FIELD else name
    for name in STRUCTURE_FEATURES
)
FINGERPRINT_DISPLAY_LABELS = ("Dark-contrast area fraction", "α lath width (µm)")
CONTRAST_TERMINOLOGY = {
    "dark_field": DARK_CONTRAST_FIELD,
    "dark_label": "Dark-contrast area fraction",
    "bright_field": BRIGHT_CONTRAST_FIELD,
    "bright_label": "Bright-contrast area fraction",
    "dark_segmentation_title": "Dark-contrast segmentation",
    "bright_segmentation_title": "Bright-contrast segmentation",
    "width_field": "mean_lath_width_um",
    "width_label": "α lath width (µm)",
    "legacy_aliases": {LEGACY_DARK_CONTRAST_FIELD: DARK_CONTRAST_FIELD,
        "beta_phase_fraction_field": "dark_contrast_area_fraction_field"},
    "bright_definition": "1 - dark_contrast_area_fraction",
    "interpretation": "Image-contrast area descriptors; no crystallographic phase identification is implied.",
    "numeric_contract": "Unchanged two channels: dark-contrast area fraction, then α lath width in µm.",
    "checkpoint_compatibility": "v4 fp_names and condition_schema retain legacy wire names; use fingerprint_names and canonical_condition_schema for reporting.",
}


def read_dark_contrast_alias(payload, canonical=DARK_CONTRAST_FIELD,
                             legacy=LEGACY_DARK_CONTRAST_FIELD):
    """Read old/new keys without recalculation; reject ambiguous dual values."""
    if canonical in payload:
        value = payload[canonical]
        if legacy in payload and not np.array_equal(
                np.asarray(value), np.asarray(payload[legacy]), equal_nan=True):
            raise ValueError(f"Conflicting contrast aliases: {canonical} and {legacy}")
        return value
    if legacy in payload:
        return payload[legacy]
    raise KeyError(f"Missing {canonical} (legacy alias: {legacy})")


def named_contrast_fingerprint(values):
    """Reporting only; never appended to a model tensor or used for training."""
    dark, width = map(float, values)
    return {DARK_CONTRAST_FIELD: dark, BRIGHT_CONTRAST_FIELD: 1.0 - dark,
            "mean_lath_width_um": width}


FP_DIM = len(FINGERPRINT_LABELS)
CONNECTIVITY_DIM = len(CONNECTIVITY_LABELS)
PROMPT_DICT = {
    0: "basket weave microstructure",
    1: "discontinuous microstructure",
    2: "Widmanstatten microstructure",
    3: "alpha-coarsened Widmanstatten microstructure",
}
FP_NAMES = list(FINGERPRINT_LABELS)
CONNECTIVITY_NAMES = list(CONNECTIVITY_LABELS)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
FP_EPS = 1e-3
CONDITION_SCHEMA_VERSION = "large-field-psim-hard-physical-full-unet-v4"
CHECKPOINT_FORMAT = "large-field-full-unet-v4"
HARD_CLASS_MIN_RATIO = 0.03
if (tuple(CLASS_NAMES) != ("Basket_weave", "Discontinuous", "Widmanstatten", "Alpha_coarsened_Widmanstatten")
        or tuple(FP_NAMES) != (DARK_CONTRAST_FIELD, "mean_lath_width_um")):
    raise ValueError("This program requires the ordered four-class, two-dimensional fingerprint contract.")

# Channel positions are part of the exported inverse-design contract.  Keeping
# them explicit avoids a silent channel-order mismatch when a sampler loads a
# trained spatial conditioner.
CLASS_CHANNEL_SLICE = slice(0, len(CLASS_NAMES))
FP_CHANNEL_SLICE = slice(CLASS_CHANNEL_SLICE.stop, CLASS_CHANNEL_SLICE.stop + FP_DIM)
CONNECTIVITY_CHANNEL_SLICE = slice(FP_CHANNEL_SLICE.stop, FP_CHANNEL_SLICE.stop + CONNECTIVITY_DIM)
UNCERTAINTY_CHANNEL = CONNECTIVITY_CHANNEL_SLICE.stop
X_COORD_CHANNEL = UNCERTAINTY_CHANNEL + 1
Y_COORD_CHANNEL = X_COORD_CHANNEL + 1
SPATIAL_CONDITION_CHANNELS = Y_COORD_CHANNEL + 1
GLOBAL_CONDITION_DIM = FP_DIM + len(CLASS_NAMES) + CONNECTIVITY_DIM + 1


def _default_project_root():
    """Resolve the shared LEE_A root for Linux, Windows, and relocated copies."""
    configured = os.environ.get("LEE_A_PROJECT_ROOT") or os.environ.get("MORPHOLOGY_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser()

    linux_root = Path("/home/user/data8t/LEE/LEE_A")
    if linux_root.exists():
        return linux_root

    # A normal checkout has this script under <project>/Diffusion/.
    if _SCRIPT_DIR.name.lower() == "diffusion":
        return _SCRIPT_DIR.parent
    return _SCRIPT_DIR


_DEFAULT_PROJECT_ROOT = _default_project_root()
_DEFAULT_DIFFUSION_DIR = _DEFAULT_PROJECT_ROOT / "Diffusion"
_DEFAULT_FOUR_CLASS_OUTPUT_DIR = _DEFAULT_DIFFUSION_DIR / "Deep-TiDiffusion-Outputs-FourClass-v3"
_DEFAULT_FP_STATS_PATH = _DEFAULT_FOUR_CLASS_OUTPUT_DIR / "fingerprint_stats_2d_four_class.npz"


@dataclass
class LargeFieldConfig:
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    local_files_only: bool = True
    allow_hf_download: bool = False
    hf_home: str = ""
    local_model_search_dirs: str = ";".join(
        str(path)
        for path in (
            _DEFAULT_PROJECT_ROOT / "models" / "stable-diffusion-v1-5",
            _DEFAULT_PROJECT_ROOT / "stable-diffusion-v1-5",
            _DEFAULT_DIFFUSION_DIR / "stable-diffusion-v1-5",
            _DEFAULT_DIFFUSION_DIR / "models" / "stable-diffusion-v1-5",
            _DEFAULT_FOUR_CLASS_OUTPUT_DIR / "stable-diffusion-v1-5",
        )
    )
    large_image_dir: str = str(_DEFAULT_PROJECT_ROOT / "data" / "ALL-DATA")
    # Retained for checkpoint/CLI compatibility; this generator is not directional.
    image_direction: str = "all"
    # ALL-DATA is a separate training dataset, not the property-study images.
    psim_output_dir: str = str(_DEFAULT_FOUR_CLASS_OUTPUT_DIR / "psim_training_ALL_DATA")
    prepare_missing_psim: bool = False
    prepare_psim_only: bool = False
    psim_script: str = str(_SCRIPT_DIR / "large_image_analysis.py")
    psim_checkpoint_dir: str = ""
    psim_fp_stats_path: str = ""
    psim_pretrained_model_name_or_path: str = ""
    validate_psim_only: bool = False
    microns_per_source_pixel: float = 1.0 / 7.32
    old_output_dir: str = str(_DEFAULT_FOUR_CLASS_OUTPUT_DIR)
    # Start from the existing micrograph LoRA. Set "" only for a deliberate fresh run.
    initial_lora_checkpoint: str = str(_DEFAULT_FOUR_CLASS_OUTPUT_DIR / "full_model.pt")
    # Optional: the complete high-quality model saved by conditional_train_and_generate15.py.
    # Set this to its best/ or epoch_N/ema/ directory to inherit its full UNet,
    # VAE and CLIP. It takes precedence over initial_lora_checkpoint.
    initial_diffusers_model_dir: str = ""
    # Set True when inheriting the reference conditional15 model. Its four
    # folders are a different class contract: PIAN+QIU must NOT be silently
    # equated with Alpha_coarsened_Widmanstatten. Supply a justified mapping.
    initial_diffusers_class_contract_confirmed: bool = False
    allow_unverified_initial_lora: bool = False
    resume_training_state: str = ""
    # High-resolution profile: this is a native-pixel crop from the original
    # 5000+ x 3000+ images, not an upscaled 768 px image. Keep the old model in
    # its original directory and write the new checkpoint separately.
    output_dir: str = str(_DEFAULT_FOUR_CLASS_OUTPUT_DIR / "large_field_full_unet_1536_v4")
    class_means_path: str = ""
    fingerprint_stats_path: str = ""
    resolution: int = 1536
    # Matches MORPHOLOGY_MEASUREMENT_SIZE in four-class large-image PSIM.
    field_size: int = 512
    # 8 x 8 keeps the local condition cell about 192 source pixels wide,
    # matching the 4 x 4 grid used by the former 768 px model.
    patch_grid: int = 8
    latent_downsample: int = 8
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 0
    num_train_epochs: int = 30
    max_train_steps: int = 4000
    learning_rate: float = 1e-5
    condition_learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    enable_gradient_checkpointing: bool = True
    enable_vae_slicing: bool = True
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05
    condition_dropout: float = 0.20
    class_prompts_json: str = json.dumps(list(PROMPT_DICT.values()))
    # Learn appearance first, then smoothly introduce zero-initialized residuals.
    condition_warmup_steps: int = 500
    condition_ramp_steps: int = 500
    global_condition_scale: float = 1.0
    spatial_condition_scale: float = 0.35
    spatial_bias_l2_weight: float = 1e-4
    # Temperature is applied to squared distances in normalised fingerprint
    # space.  Values below one retain class information without converting an
    # ambiguous Discontinuous/Basket_weave patch into a false one-hot label.
    class_probability_temperature: float = 0.75
    cache_stats_samples: int = 160
    save_steps: int = 500
    validation_steps: int = 500
    validation_sampling_steps: int = 500
    validation_num_samples: int = 2
    validation_scheduler: str = "ddpm"
    validation_guidance: float = 7.5
    validation_compare_no_cfg: bool = False
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_device: str = "cpu"
    # Full UNet + EMA + Adam states are large; keep latest by default.
    keep_step_snapshots: bool = False
    tracker_project_name: str = "large-field-microstructure-diffusion"


def parse_args():
    parser = argparse.ArgumentParser(description="Train large-field conditional microstructure diffusion.")
    defaults = LargeFieldConfig()
    for key, value in asdict(defaults).items():
        arg_type = type(value)
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--no-{key}", dest=key, action="store_false")
            else:
                parser.add_argument(f"--{key}", dest=key, action="store_true")
            parser.set_defaults(**{key: value})
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    args = parser.parse_args()
    if args.image_direction != "all":
        bootstrap_logger.warning("--image_direction is retired for large-field generation training; using all images without direction filtering.")
    args.image_direction = "all"
    if args.validation_steps < 0 or args.validation_sampling_steps < 1 or args.validation_num_samples < 1:
        parser.error("validation_steps must be >=0; validation_sampling_steps and validation_num_samples must be positive.")
    if args.save_steps < 1 or args.num_train_epochs < 1 or args.max_train_steps < 0:
        parser.error("save_steps and num_train_epochs must be positive; max_train_steps must be >=0.")
    if args.prepare_psim_only:
        args.prepare_missing_psim = True
    if args.resolution < 64 or args.resolution % 64 or args.latent_downsample != 8:
        parser.error("resolution must be a positive multiple of 64; SD1.5 latent_downsample must be 8.")
    if args.patch_grid < 1 or args.train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("patch_grid, batch size and accumulation steps must be positive.")
    if args.learning_rate <= 0 or args.condition_learning_rate <= 0:
        parser.error("learning rates must be positive.")
    if args.validation_sampling_steps > 1000:
        parser.error("validation_sampling_steps must not exceed the default 1000 training timesteps.")
    if args.global_condition_scale < 0 or args.spatial_condition_scale < 0:
        parser.error("condition scales must be non-negative.")
    if not 0 <= args.condition_dropout <= 1 or not 0 <= args.ema_decay < 1:
        parser.error("condition_dropout must be in [0,1]; ema_decay must be in [0,1).")
    if args.condition_warmup_steps < 0 or args.condition_ramp_steps < 0:
        parser.error("condition warmup/ramp steps must be >=0.")
    if args.validation_scheduler not in ("ddpm", "ddim") or args.validation_guidance < 1:
        parser.error("validation_scheduler must be ddpm/ddim; validation_guidance must be >=1.")
    if args.ema_device not in ("cpu", "cuda") or args.mixed_precision not in ("no", "fp16", "bf16"):
        parser.error("ema_device must be cpu/cuda; mixed_precision must be no/fp16/bf16.")
    try:
        prompts = json.loads(args.class_prompts_json)
        if not isinstance(prompts, list) or len(prompts) != len(CLASS_NAMES) or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("four non-empty prompts required")
    except (ValueError, TypeError) as error:
        parser.error(f"class_prompts_json: {error}")
    return LargeFieldConfig(**vars(args))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_image_files(root):
    paths = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_local_diffusers_model_dir(path):
    if not path or not os.path.isdir(path):
        return False
    required = ["vae", "unet", "tokenizer", "text_encoder", "scheduler"]
    return all(os.path.isdir(os.path.join(path, name)) for name in required)


def split_search_dirs(text):
    if not text:
        return []
    return [p.strip().strip('"') for p in str(text).split(";") if p.strip()]


def normalized_model_path(path):
    """Compare local model paths including /home/... symlinks to /data8t/... ."""
    return os.path.normcase(os.path.realpath(os.path.expandvars(os.path.expanduser(path))))


def initial_lora_local_base(cfg):
    """Prefer the original micrograph base, without overlaying it on a resume."""
    if cfg.resume_training_state or not cfg.initial_lora_checkpoint:
        return ""
    source = os.path.expandvars(os.path.expanduser(cfg.initial_lora_checkpoint))
    checkpoint = safe_torch_load(source)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source} is not a checkpoint dictionary.")
    old_cfg = checkpoint.get("args") or checkpoint.get("config") or {}
    if not isinstance(old_cfg, dict):
        old_cfg = vars(old_cfg)
    recorded = str(old_cfg.get("resolved_pretrained_model_name_or_path") or
                   old_cfg.get("pretrained_model_name_or_path") or "")
    expanded = os.path.expandvars(os.path.expanduser(recorded))
    # Repository IDs (e.g. runwayml/stable-diffusion-v1-5) remain cache IDs.
    if not expanded or not (os.path.isabs(expanded) or os.path.isdir(expanded) or
                            expanded.startswith(".")):
        return ""
    if is_local_diffusers_model_dir(expanded):
        bootstrap_logger.info(
            f"Prioritizing original micrograph base recorded in {source}: {expanded}"
        )
        return expanded
    message = (
        f"Original micrograph base directory {expanded!r} recorded in {source} is missing "
        "or incomplete (required subdirectories: vae, unet, tokenizer, text_encoder, scheduler). "
        "Restore that original model directory. To explicitly try an unverified replacement, "
        "set --allow_unverified_initial_lora and select --pretrained_model_name_or_path."
    )
    if not cfg.allow_unverified_initial_lora:
        raise FileNotFoundError(message)
    bootstrap_logger.warning(message)
    return ""


def candidate_pretrained_paths(cfg):
    paths = []
    env_path = os.environ.get("SD15_LOCAL_DIR", "")
    if env_path:
        paths.append(env_path)
    paths.append(cfg.pretrained_model_name_or_path)
    paths.extend(split_search_dirs(cfg.local_model_search_dirs))
    paths.extend([
        os.path.join(cfg.old_output_dir, "stable-diffusion-v1-5"),
        os.path.join(cfg.old_output_dir, "sd-v1-5"),
        os.path.join(os.path.dirname(cfg.old_output_dir), "stable-diffusion-v1-5"),
        os.path.join(os.path.dirname(cfg.old_output_dir), "models", "stable-diffusion-v1-5"),
    ])
    seen = set()
    unique = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path)) if os.path.isabs(path) else path
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def resolve_pretrained_model_name_or_path(cfg):
    if cfg.resume_training_state:
        state = safe_torch_load(cfg.resume_training_state)
        if state.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise ValueError("This full-UNet trainer cannot exactly resume legacy LoRA optimizer states. Start a new run.")
        recorded = state["config"]["pretrained_model_name_or_path"]
        if os.path.isabs(recorded) and not is_local_diffusers_model_dir(recorded):
            raise FileNotFoundError(f"Resume base is missing or incomplete: {recorded}")
        return recorded
    if cfg.initial_diffusers_model_dir:
        if not cfg.initial_diffusers_class_contract_confirmed:
            raise ValueError(
                "Confirm the inherited pipeline's class-to-prompt mapping first. "
                "conditional15 uses Discont/LAN/PIAN+QIU/WEI, which is not automatically "
                "the current four-class contract. Set class_prompts_json in current "
                "CLASS_NAMES order, then initial_diffusers_class_contract_confirmed=True."
            )
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(cfg.initial_diffusers_model_dir)))
        if not is_local_diffusers_model_dir(path):
            raise FileNotFoundError(f"Complete initial Diffusers model directory required: {path}")
        bootstrap_logger.info(f"Using complete trained micrograph pipeline: {path}")
        return path
    original_base = initial_lora_local_base(cfg)
    if original_base:
        return original_base
    for path in candidate_pretrained_paths(cfg):
        if is_local_diffusers_model_dir(path):
            bootstrap_logger.info(f"Using local Stable Diffusion model directory: {path}")
            return path

    model_id = cfg.pretrained_model_name_or_path
    if cfg.local_files_only and not cfg.allow_hf_download:
        bootstrap_logger.warning(
            "No complete local Stable Diffusion directory was found in the configured search paths. "
            "Will try Hugging Face cache only with local_files_only=True."
        )
        return model_id

    bootstrap_logger.warning(
        "No complete local Stable Diffusion directory was found; Hugging Face download is allowed. "
        "If the network is unstable, pass --local_files_only or set SD15_LOCAL_DIR to a local model path."
    )
    return model_id


def configure_offline_loading(cfg):
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cfg.hf_home, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cfg.hf_home, "transformers")

    if cfg.local_files_only and not cfg.allow_hf_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


def load_diffusers_component(component_cls, model_path, subfolder, cfg, label):
    local_only = bool(cfg.local_files_only and not cfg.allow_hf_download)
    try:
        return component_cls.from_pretrained(
            model_path,
            subfolder=subfolder,
            local_files_only=local_only,
        )
    except Exception as exc:
        search_lines = "\n".join(f"  - {p}" for p in candidate_pretrained_paths(cfg))
        raise RuntimeError(
            f"\nFailed to load {label} from: {model_path}\n"
            f"subfolder: {subfolder}\n"
            f"local_files_only: {local_only}\n\n"
            "This is usually caused by an incomplete local Stable Diffusion v1-5 cache or a network timeout.\n"
            "Recommended fixes:\n"
            "1. Put a complete diffusers-format SD v1-5 directory on disk. It must contain:\n"
            "   vae/, unet/, tokenizer/, text_encoder/, scheduler/\n"
            "2. Run with:\n"
            "   --pretrained_model_name_or_path \"D:\\path\\to\\stable-diffusion-v1-5\"\n"
            "3. Or set an environment variable before running:\n"
            "   set SD15_LOCAL_DIR=D:\\path\\to\\stable-diffusion-v1-5\n"
            "4. If you really want to download from Hugging Face, run with:\n"
            "   --allow_hf_download\n\n"
            f"Configured search paths:\n{search_lines}\n\n"
            f"Original error:\n{repr(exc)}"
        ) from exc


def generate_morphology_fields_from_array(img_array, mask_size=(128, 128)):
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    gray = cv2.resize(gray, mask_size).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    vesselness = frangi(enhanced, sigmas=(1, 2, 3, 4, 5), beta=0.5, gamma=15, black_ridges=False)
    vesselness = np.clip(vesselness, 0, 1).astype(np.float32)
    # The raw feature definition deliberately mirrors the large-image PSIM
    # contract: Dark-contrast area fraction and medial-axis α lath width.
    _, beta_mask = cv2.threshold(enhanced, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    beta_mask = cv2.morphologyEx(beta_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)
    thr = np.percentile(vesselness, 85)
    binary = vesselness > thr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    skeleton, distance = medial_axis(binary, return_distance=True)
    lath_width = (2.0 * distance * skeleton).astype(np.float32)
    return np.stack([beta_mask, lath_width], axis=0).astype(np.float32)


def compute_extended_fingerprint_np(field):
    beta_mask = field[0].astype(np.float32)
    lath_width = field[1].astype(np.float32)
    widths = lath_width[lath_width > 0]
    mean_width = float(widths.mean()) if widths.size else 0.0
    return np.array([float(beta_mask.mean()), mean_width], dtype=np.float32)


def normalize_fp(fp, mean, std):
    return (np.asarray(fp, dtype=np.float32) - mean) / (std + FP_EPS)


def load_or_compute_fp_stats(cfg, image_paths):
    out_path = os.path.join(cfg.output_dir, "large_field_condition_stats.npz")
    if cfg.fingerprint_stats_path:
        stats_path = cfg.fingerprint_stats_path
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Configured --fingerprint_stats_path was not found: {stats_path}")
        loaded = np.load(stats_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if {"fp_mean", "fp_std"}.issubset(loaded.files):
                    fp_mean, fp_std = loaded["fp_mean"], loaded["fp_std"]
                elif {"mean", "std"}.issubset(loaded.files):
                    # train.py/train_merged.py use these historical key names.
                    fp_mean, fp_std = loaded["mean"], loaded["std"]
                else:
                    raise ValueError(
                        f"Fingerprint stats archive {stats_path} has keys {loaded.files}; "
                        "expected fp_mean/fp_std or mean/std."
                    )
            finally:
                loaded.close()
        else:
            loaded_array = np.asarray(loaded)
            if loaded_array.ndim == 0 or loaded_array.shape[0] != 2:
                raise ValueError(
                    f"Fingerprint stats array {stats_path} must contain [mean, std]; "
                    f"received shape {loaded_array.shape}."
                )
            fp_mean, fp_std = loaded_array[0], loaded_array[1]
        fp_mean = np.asarray(fp_mean, dtype=np.float32).reshape(-1)
        fp_std = np.asarray(fp_std, dtype=np.float32).reshape(-1)
        if fp_mean.shape != (FP_DIM,) or fp_std.shape != (FP_DIM,):
            raise ValueError(
                f"Fingerprint stats are {fp_mean.shape}/{fp_std.shape}; expected ({FP_DIM},)."
            )
        ensure_dir(cfg.output_dir)
        np.savez_compressed(out_path, fp_mean=fp_mean, fp_std=fp_std)
        return fp_mean, np.maximum(fp_std, 1e-4)
    if os.path.exists(out_path):
        data = np.load(out_path)
        fp_mean = data["fp_mean"].astype(np.float32)
        fp_std = data["fp_std"].astype(np.float32)
        if fp_mean.shape == (FP_DIM,) and fp_std.shape == (FP_DIM,):
            return fp_mean, fp_std
        raise ValueError(
            f"Existing condition statistics have shapes {fp_mean.shape}/{fp_std.shape}; "
            f"expected ({FP_DIM},). Use a new output directory for the four-class 2-D model."
        )

    logger.warning(
        "No --fingerprint_stats_path supplied. Computing local 2-D statistics; "
        "use the same fp_stats file as large-image PSIM for strict compatibility."
    )
    rng = np.random.default_rng(cfg.seed)
    samples = []
    chosen = rng.choice(image_paths, size=min(cfg.cache_stats_samples, len(image_paths)), replace=False)
    for path in tqdm(chosen, desc="Computing fingerprint stats"):
        img = Image.open(path).convert("RGB")
        crop = random_square_crop(img, cfg.resolution, rng)
        field = generate_morphology_fields_from_array(np.asarray(crop), (cfg.field_size, cfg.field_size))
        samples.append(compute_extended_fingerprint_np(field))
    fps = np.stack(samples, axis=0)
    fp_mean = fps.mean(axis=0).astype(np.float32)
    fp_std = np.maximum(fps.std(axis=0), 1e-4).astype(np.float32)
    ensure_dir(cfg.output_dir)
    np.savez_compressed(out_path, fp_mean=fp_mean, fp_std=fp_std)
    return fp_mean, fp_std


def load_class_means(cfg, fp_mean, fp_std):
    path = cfg.class_means_path or os.path.join(cfg.old_output_dir, "test_results", "class_mean_fingerprints.npy")
    if os.path.exists(path):
        arr = np.load(path).astype(np.float32)
        if arr.shape == (len(CLASS_NAMES), FP_DIM):
            return arr
    raise FileNotFoundError(
        "Four-class 2-D class means are required. Pass --class_means_path to "
        "class_mean_fingerprints.npy with shape (4, 2)."
    )


def random_square_crop(img, crop_size, rng):
    w, h = img.size
    side = min(w, h)
    if side < crop_size:
        scale = crop_size / max(side, 1)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BICUBIC)
        w, h = img.size
        side = min(w, h)
    x0 = int(rng.integers(0, max(w - side + 1, 1)))
    y0 = int(rng.integers(0, max(h - side + 1, 1)))
    img = img.crop((x0, y0, x0 + side, y0 + side))
    if side != crop_size:
        img = img.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    return img


def soft_class_probabilities(fp_norm, class_means, temperature=0.75):
    """Return calibrated class probabilities instead of a fingerprint argmin.

    The four class centres are useful weak supervision for unlabelled large
    fields, but dark-contrast area fraction and α lath width alone cannot separate all local
    Basket_weave and Discontinuous patches.  The entropy is therefore retained
    as a condition feature rather than discarded after an argmin operation.
    """
    if class_means is None or class_means.shape != (len(CLASS_NAMES), FP_DIM):
        return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32)
    temperature = max(float(temperature), 1e-4)
    delta = class_means.astype(np.float32) - np.asarray(fp_norm, dtype=np.float32)[None, :]
    logits = -np.sum(delta * delta, axis=1) / temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return (weights / max(float(weights.sum()), 1e-8)).astype(np.float32)


def compute_connectivity_descriptor_np(field):
    """Measure the inverse-design continuity descriptors directly from laths.

    The output names and ranges match ``structure_performance_contract``.  No
    morphology closing is applied here, because closing short gaps would erase
    the fragmentation cue that distinguishes Discontinuous tissue from a truly
    connected Basket_weave region.
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] < 2:
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)

    # ``lath_width`` is non-zero only on the medial axis, hence it retains
    # genuine gaps while being insensitive to illumination in the source image.
    support = np.nan_to_num(field[1], nan=0.0, posinf=0.0, neginf=0.0) > 0.0
    if support.size == 0 or not support.any():
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)
    skeleton = skeletonize(support)
    if not skeleton.any():
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    lengths = np.asarray(
        [stats[index, cv2.CC_STAT_AREA] for index in range(1, n_labels)
         if stats[index, cv2.CC_STAT_AREA] > 0],
        dtype=np.float32,
    )
    if lengths.size == 0:
        return np.zeros(CONNECTIVITY_DIM, dtype=np.float32)

    total_length = float(lengths.sum())
    reference_length = max(8.0, 0.12 * min(support.shape))
    connectivity_length = float(
        np.sum(lengths * np.clip(lengths / reference_length, 0.0, 1.0))
        / max(total_length, 1.0)
    )
    short_length = max(6.0, 0.04 * min(support.shape))
    short_fraction = float(np.sum(lengths[lengths < short_length]) / max(total_length, 1.0))
    largest_fraction = float(lengths.max() / max(total_length, 1.0))
    fragmentation_degree = float(
        np.clip(0.5 * short_fraction + 0.5 * (1.0 - largest_fraction), 0.0, 1.0)
    )

    # Estimate line orientation per connected component.  The doubled angle
    # gives axial rather than directional orientation (theta == theta + pi).
    component_weights = []
    component_angles = []
    for label_id in range(1, n_labels):
        yy, xx = np.nonzero(labels == label_id)
        if yy.size < 3:
            continue
        coords = np.column_stack((xx.astype(np.float32), yy.astype(np.float32)))
        covariance = np.cov(coords, rowvar=False)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            continue
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        component_angles.append(float(np.arctan2(direction[1], direction[0])))
        component_weights.append(float(yy.size))
    if component_weights:
        weights = np.asarray(component_weights, dtype=np.float32)
        angles = np.asarray(component_angles, dtype=np.float32)
        orientation_consistency = float(np.abs(np.sum(weights * np.exp(2j * angles)) / max(weights.sum(), 1e-6)))
    else:
        orientation_consistency = 0.0

    return np.asarray(
        [
            np.clip(connectivity_length, 0.0, 1.0),
            np.clip(fragmentation_degree, 0.0, 1.0),
            np.clip(orientation_consistency, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def class_entropy(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-8, 1.0)
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(CLASS_NAMES)))


def condition_schema():
    """Serializable channel layout consumed by the later inverse sampler."""
    return {
        "version": CONDITION_SCHEMA_VERSION,
        "inverse_recipe_feature_order": list(STRUCTURE_FEATURES),
        "global_feature_order": [
            *FP_NAMES,
            *(f"{name}_fraction" for name in CLASS_NAMES),
            *CONNECTIVITY_NAMES,
            "class_assignment_entropy",
        ],
        "spatial_channel_order": [
            *(f"{name}_probability" for name in CLASS_NAMES),
            *FP_NAMES,
            *CONNECTIVITY_NAMES,
            "class_assignment_entropy",
            "x_coordinate",
            "y_coordinate",
        ],
        "class_fraction_type": "area_fraction_of_final_PSIM_hard_map_retained_classes",
        "hard_class_min_ratio_per_source_image": HARD_CLASS_MIN_RATIO,
        "rare_class_policy": "exclude_below_threshold_pixels_from_class_denominator_only; keep_physical_fields_and_pixels; no_crop_level_threshold",
        "fingerprint_units": ["fraction", "um"],
        "fingerprint_names": list(FP_NAMES),
        "fingerprint_display_labels": list(FINGERPRINT_DISPLAY_LABELS),
        "contrast_terminology": dict(CONTRAST_TERMINOLOGY),
        "fingerprint_source": "cached_PSIM_quantitative_fields",
        "mechanical_property_conditioning": "none",
        "direction_conditioning": "none",
        "image_transform": "native_pixel_crop_and_synchronized_flips_no_resize",
        "connectivity_usage": "reserved_zero_channels_not_measured",
        "connectivity_range": [0.0, 1.0],
        "normalised_fingerprint": True,
        "global_token_injection": "additive_residual_to_last_text_token_no_normalization",
        "condition_schedule": "zero_during_appearance_warmup_then_linear_ramp",
        "checkpoint_format": CHECKPOINT_FORMAT,
    }


def legacy_condition_schema():
    """Original v4 wire schema for existing strict inverse-design loaders.

    Reporting uses condition_schema(); weights/channel order are identical.
    """
    schema = condition_schema()
    for key in ("fingerprint_names", "fingerprint_display_labels", "contrast_terminology"):
        schema.pop(key)
    for key in ("inverse_recipe_feature_order", "global_feature_order", "spatial_channel_order"):
        schema[key] = [LEGACY_DARK_CONTRAST_FIELD if name == DARK_CONTRAST_FIELD else name
                       for name in schema[key]]
    return schema


def make_spatial_condition(crop_img, cfg, fp_mean, fp_std, class_means):
    raise RuntimeError("Pixel-width/centroid conditioning was retired. Use make_cached_spatial_condition with existing physical PSIM fields.")


def make_inverse_spatial_condition(features, cfg, fp_mean, fp_std, class_probability_map=None):
    """Build the physical-v3 conditioning tensor from an inverse structure recipe.

    ``features`` accepts the keys written by ``inverse_design_directional.py``:
    the two physical fingerprint names and ``<class>_fraction`` for all four
    classes. Connectivity channels are reserved zeros. A sampler may provide a
    ``(grid, grid, 4)`` area-fraction map to prescribe spatial composition; when
    omitted, the global recipe is broadcast uniformly.  This utility is kept
    here so the sampler and trainer cannot accidentally disagree on channel
    order or normalisation.
    """
    def read_feature(name, default=None):
        if name == DARK_CONTRAST_FIELD:
            return float(read_dark_contrast_alias(features))
        if name in features:
            return float(features[name])
        if default is not None:
            return float(default)
        raise KeyError(f"Inverse recipe is missing required feature: {name}")

    fp_raw = np.asarray([read_feature(name) for name in FP_NAMES], dtype=np.float32)
    if not np.isfinite(fp_raw).all() or not 0 <= fp_raw[0] <= 1 or fp_raw[1] <= 0:
        raise ValueError("Recipe needs Dark-contrast area fraction 0..1 and positive α lath width in micrometres.")
    fp_norm = normalize_fp(fp_raw, fp_mean, fp_std).astype(np.float32)
    connectivity = np.zeros(CONNECTIVITY_DIM, dtype=np.float32)
    fractions = np.asarray([read_feature(f"{name}_fraction") for name in CLASS_NAMES], dtype=np.float32)
    if not np.isfinite(fractions).all() or (fractions < 0).any() or not np.isclose(fractions.sum(), 1, atol=1e-5):
        raise ValueError("Four hard area fractions must be finite, nonnegative and sum to one.")
    fractions /= fractions.sum()

    g = int(cfg.patch_grid)
    latent_h = cfg.resolution // cfg.latent_downsample
    latent_w = cfg.resolution // cfg.latent_downsample
    cond = np.zeros((SPATIAL_CONDITION_CHANNELS, latent_h, latent_w), dtype=np.float32)
    if class_probability_map is None:
        class_probability_map = np.broadcast_to(fractions, (g, g, len(CLASS_NAMES))).copy()
    else:
        class_probability_map = np.asarray(class_probability_map, dtype=np.float32)
        expected_shape = (g, g, len(CLASS_NAMES))
        if class_probability_map.shape != expected_shape:
            raise ValueError(
                f"class_probability_map must have shape {expected_shape}, got {class_probability_map.shape}"
            )
        if (not np.isfinite(class_probability_map).all() or (class_probability_map < 0).any()
                or not np.allclose(class_probability_map.sum(-1), 1, atol=1e-5)
                or not np.allclose(class_probability_map.mean((0, 1)), fractions, atol=1e-5)):
            raise ValueError("Spatial area fractions must sum to one per cell and match the global recipe.")
        totals = class_probability_map.sum(axis=-1, keepdims=True)
        class_probability_map = np.divide(
            class_probability_map,
            np.maximum(totals, 1e-8),
            out=np.broadcast_to(fractions, class_probability_map.shape).copy(),
            where=totals > 1e-8,
        )

    lat_cell_h = latent_h // g
    lat_cell_w = latent_w // g
    for iy in range(g):
        for ix in range(g):
            ly0, lx0 = iy * lat_cell_h, ix * lat_cell_w
            ly1 = latent_h if iy == g - 1 else (iy + 1) * lat_cell_h
            lx1 = latent_w if ix == g - 1 else (ix + 1) * lat_cell_w
            probabilities = class_probability_map[iy, ix]
            cond[CLASS_CHANNEL_SLICE, ly0:ly1, lx0:lx1] = probabilities[:, None, None]
            cond[FP_CHANNEL_SLICE, ly0:ly1, lx0:lx1] = fp_norm[:, None, None]
            cond[CONNECTIVITY_CHANNEL_SLICE, ly0:ly1, lx0:lx1] = connectivity[:, None, None]
            cond[UNCERTAINTY_CHANNEL, ly0:ly1, lx0:lx1] = class_entropy(probabilities)

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, latent_h, dtype=np.float32),
        np.linspace(-1.0, 1.0, latent_w, dtype=np.float32),
        indexing="ij",
    )
    cond[X_COORD_CHANNEL] = xx
    cond[Y_COORD_CHANNEL] = yy
    spatial_fraction = class_probability_map.mean(axis=(0, 1)).astype(np.float32)
    return {
        "spatial_cond": cond,
        "soft_fraction": spatial_fraction,
        "fingerprint": fp_norm,
        "fingerprint_raw": fp_raw,
        "connectivity": connectivity,
        "class_assignment_entropy": np.float32(
            np.mean([class_entropy(x) for x in class_probability_map.reshape(-1, len(CLASS_NAMES))])
        ),
        "class_id": int(np.argmax(spatial_fraction)),
    }


def portable_path_parts(value):
    """Compare source metadata recorded on Linux or Windows without opening it."""
    return tuple(part for part in str(value).replace("\\", "/").split("/") if part and part != ".")


class PSIMMatchError(ValueError):
    def __init__(self, message, match_count):
        super().__init__(message)
        self.match_count = match_count


def match_psim_summary(image_path, summaries, source_stem_counts):
    """Prefer full paths, then the longest preserved parent/name suffix.

    A filename-only record is usable only if its stem is unique in the source
    dataset and PSIM catalogue. Never pair two numbered images by scan order.
    """
    parts = portable_path_parts(image_path)
    stem = Path(parts[-1]).stem
    scored = []
    for summary_path, summary in summaries:
        recorded = portable_path_parts(summary.get("image_path") or "")
        if recorded:
            score = 0
            for left, right in zip(reversed(parts), reversed(recorded)):
                if left != right:
                    break
                score += 1
            if score >= 2:
                scored.append((len(parts) + 1 if recorded == parts else score, summary_path, summary))
    if scored:
        best = max(row[0] for row in scored)
        matches = [(path, summary) for score, path, summary in scored if score == best]
        method = "full_source_path" if best == len(parts) + 1 else "preserved_parent_path_suffix"
    else:
        same_stem = []
        for path, summary in summaries:
            recorded = portable_path_parts(summary.get("image_path") or "")
            recorded_stem = Path(recorded[-1]).stem if recorded else path.parent.name
            if recorded_stem == stem:
                same_stem.append((path, summary, recorded))
        # Structured metadata with a different parent identifies another image,
        # even when its basename happens to be globally unique in this scan.
        matches = [(path, summary) for path, summary, recorded in same_stem
                   if len(recorded) <= 1] if source_stem_counts.get(stem) == 1 and len(same_stem) == 1 else []
        method = "unique_legacy_stem"
    if len(matches) != 1:
        raise PSIMMatchError(
            f"Cannot uniquely match cached PSIM for {image_path}: {len(matches)} matches. "
            "Numbered filenames need specimen-folder context, for example 169/1.jpg, "
            "in PSIM_summary.json image_path. If files were renamed, preserve the "
            "original image-to-result mapping; do not guess by file order. "
            "Check psim_output_dir. For NEW, never-analyzed training images use "
            "--prepare_psim_only to build their own cache. Existing PSIM is not rerun.",
            len(matches),
        )
    return (*matches[0], method)


def training_psim_archive(summary_path):
    exact = summary_path.parent / "PSIM_results.npz"
    archives = ([exact] if exact.is_file() else sorted(summary_path.parent.glob("PSIM_results_*.npz"),
                key=lambda p: (p.stat().st_mtime, p.name), reverse=True))
    return archives[0] if archives else None


def read_training_psim_catalog(cfg, image_paths):
    summaries = []
    for path in sorted(Path(cfg.psim_output_dir).rglob("PSIM_summary.json")):
        # A stale interruption marker must not invalidate an otherwise complete
        # numerical cache.  The archive is the decisive signal because training
        # consumes PSIM_results*.npz, not the marker file.
        incomplete = path.parent / ".training_psim_incomplete"
        if incomplete.exists() and training_psim_archive(path) is None:
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append((path, summary))
    counts = {}
    for path in image_paths:
        counts[Path(path).stem] = counts.get(Path(path).stem, 0) + 1
    return summaries, counts


def load_training_psim_engine(cfg):
    script = Path(cfg.psim_script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"PSIM program not found: {script}. Set --psim_script to the existing large-image analysis program.")
    spec = importlib.util.spec_from_file_location("_training_only_psim", script)
    lia = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lia
    spec.loader.exec_module(lia)
    if list(lia.CLASS_NAMES) != list(CLASS_NAMES):
        raise ValueError("Training PSIM class order must match the four-class generator.")
    if not np.isclose(float(lia.MICRONS_PER_SOURCE_PIXEL), cfg.microns_per_source_pixel, rtol=1e-5):
        raise ValueError("Training image/PSIM um-per-pixel mismatch. Set the correct physical calibration in both programs.")
    for field, setting in (("CHECKPOINT_DIR", cfg.psim_checkpoint_dir), ("FP_STATS_PATH", cfg.psim_fp_stats_path),
                           ("PRETRAINED_MODEL", cfg.psim_pretrained_model_name_or_path)):
        if setting:
            setattr(lia, field, setting)
    # Keep computation and numerical exports intact. Suppress supported display
    # switches only; older PSIM versions may still write diagnostic figures.
    for flag in ("SAVE_VISUALIZATIONS", "SAVE_VISUALIZATION", "SAVE_PLOTS", "SHOW_FIGURES", "SHOW_RESULTS"):
        if hasattr(lia, flag):
            setattr(lia, flag, False)
    values = lia.load_models()
    engine = lia.PhysicsAwareSpatialInferenceModule(*values, patch_size=lia.PATCH_SIZE,
                stride=lia.STRIDE, sigma=lia.SIGMA, lambda_fp=lia.LAMBDA_FINGERPRINT)
    return lia, engine


def prepare_training_psim(cfg, image_paths):
    """Explicitly analyze only missing training images, in collision-free folders."""
    if not cfg.prepare_missing_psim:
        raise ValueError("PSIM preparation requires --prepare_missing_psim or --prepare_psim_only.")
    summaries, counts = read_training_psim_catalog(cfg, image_paths)
    root = Path(cfg.psim_output_dir).resolve()
    rows, pending, problems = [], [], []
    for path in image_paths:
        path = Path(path).resolve()
        try:
            summary_path, summary, _ = match_psim_summary(path, summaries, counts)
        except PSIMMatchError as exc:
            if exc.match_count:
                problems.append(str(exc))
                rows.append({"image": str(path), "status": "ambiguous", "summary": ""})
                continue
            # A unique parent per relative source path prevents 169/1.jpg and
            # 170/1.jpg overwriting each other inside PSIM's basename directory.
            relative = path.relative_to(Path(cfg.large_image_dir).resolve()).as_posix()
            key = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
            parent = root / ("source_" + key)
            summary_path, summary = None, None
        else:
            calibration = summary.get("quantitative_morphology", {}).get("microns_per_source_pixel")
            if calibration is None or not np.isclose(float(calibration), cfg.microns_per_source_pixel, rtol=1e-5):
                problems.append(f"Missing/mismatched physical calibration: {summary_path}")
                rows.append({"image": str(path), "status": "invalid_calibration", "summary": str(summary_path)})
                continue
            if training_psim_archive(summary_path) is not None:
                # The result is complete even if a previous remote stop left a
                # marker behind.  Remove only that stale marker; never rerun the
                # expensive PSIM inference for a complete cache entry.
                (summary_path.parent / ".training_psim_incomplete").unlink(missing_ok=True)
                rows.append({"image": str(path), "status": "cached", "summary": str(summary_path)})
                continue
            # Write alongside an existing incomplete summary only when its
            # directory is exactly the path PSIM.run will use.
            parent = summary_path.parent.parent
        row = {"image": str(path), "status": "pending", "summary": str(summary_path or "")}
        rows.append(row)
        pending.append((path, parent, summary_path, row))
    root.mkdir(parents=True, exist_ok=True)
    def save_progress():
        with (root / "training_psim_preparation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "status", "summary"])
            writer.writeheader()
            writer.writerows(rows)
    save_progress()
    print(f"Training PSIM cache: {root}; {sum(r['status'] == 'cached' for r in rows)} cached, {len(pending)} missing, {len(problems)} conflicts.")
    if problems:
        raise ValueError("Resolve conflicting cached records before preparation. " + problems[0])
    if not pending:
        print("All requested training images have cached PSIM; no PSIM models loaded.")
        return
    lia, engine = load_training_psim_engine(cfg)
    try:
        for index, (path, parent, old_summary, row) in enumerate(pending, 1):
            destination = parent / lia.safe_name(path.stem)
            if old_summary is not None and destination != old_summary.parent:
                raise ValueError(f"Incomplete PSIM record uses an unexpected folder: {old_summary}; inspect it before regeneration.")
            lia.OUTPUT_DIR = str(parent)
            destination.mkdir(parents=True, exist_ok=True)
            incomplete = destination / ".training_psim_incomplete"
            incomplete.write_text(str(path), encoding="utf-8")
            row["status"] = "running"
            save_progress()
            print(f"[Training PSIM {index}/{len(pending)}] New/unprocessed image: {path}")
            try:
                engine.run(str(path))
                produced = destination / "PSIM_summary.json"
                summary = json.loads(produced.read_text(encoding="utf-8-sig"))
                if portable_path_parts(summary.get("image_path", "")) != portable_path_parts(path):
                    raise ValueError(f"PSIM result does not identify the requested source: {produced}")
                calibration = summary.get("quantitative_morphology", {}).get("microns_per_source_pixel")
                if calibration is None or not np.isclose(float(calibration), cfg.microns_per_source_pixel, rtol=1e-5):
                    raise ValueError(f"PSIM exported a missing/mismatched physical calibration: {produced}")
                archive = training_psim_archive(produced)
                if archive is None:
                    raise FileNotFoundError(f"PSIM did not export numerical fields: {destination}")
                fields = load_physical_psim_fields(str(archive))
                with Image.open(path) as source:
                    if fields[0].shape != (source.height, source.width):
                        raise ValueError(f"PSIM fields are not aligned to the training image: {path}")
                row.update(status="completed", summary=str(produced))
                incomplete.unlink()
                del fields
                load_physical_psim_fields.cache_clear()
                save_progress()
            except Exception:
                row["status"] = "failed"
                save_progress()
                raise
    finally:
        del engine
        sys.modules.pop("_training_only_psim", None)
        del lia
        load_physical_psim_fields.cache_clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def index_existing_psim(cfg, image_paths):
    """Read cached numerical records only; never import/run the PSIM model."""
    if cfg.resolution < 1 or cfg.patch_grid < 1 or cfg.latent_downsample != 8:
        raise ValueError("Positive resolution/grid and Stable Diffusion latent_downsample=8 are required.")
    if cfg.resolution % cfg.latent_downsample or (cfg.resolution // cfg.latent_downsample) % cfg.patch_grid:
        raise ValueError("resolution must be divisible by latent_downsample * patch_grid.")
    if cfg.microns_per_source_pixel <= 0:
        raise ValueError("microns_per_source_pixel must be positive.")
    if cfg.resolution >= 1024 and cfg.train_batch_size != 1:
        raise ValueError("High-resolution crops (>=1024 px) require --train_batch_size 1 to keep memory bounded.")
    summaries, source_stem_counts = read_training_psim_catalog(cfg, image_paths)
    if not summaries:
        raise FileNotFoundError(f"No training-image PSIM summaries in {cfg.psim_output_dir}. "
            "A different image dataset cannot reuse the property-study PSIM maps. "
            "Run train_large_field.py --prepare_psim_only once for the new training images, then rerun training. "
            "Existing property-study results are not recomputed.")
    records, manifest, used_archives = {}, [], {}
    for path in image_paths:
        summary_path, summary, match_method = match_psim_summary(path, summaries, source_stem_counts)
        calibration = summary.get("quantitative_morphology", {}).get("microns_per_source_pixel")
        if calibration is None or not np.isclose(float(calibration), cfg.microns_per_source_pixel, rtol=1e-5):
            raise ValueError(f"Missing/mismatched physical calibration in {summary_path}: {calibration}")
        archive = training_psim_archive(summary_path)
        if archive is None:
            raise FileNotFoundError(f"Missing PSIM numerical archive next to {summary_path}; no automatic inference.")
        archive_key = str(archive.resolve())
        if archive_key in used_archives:
            raise ValueError(f"PSIM archive {archive} would be reused for different source images: {used_archives[archive_key]} and {path}. Fix source mapping first.")
        used_archives[archive_key] = str(path)
        records[str(path)] = str(archive)
        manifest.append({"image": str(path), "archive": str(archive), "summary": str(summary_path),
                         "match_method": match_method, "image_direction": "not_used",
                         "archive_size": archive.stat().st_size, "archive_mtime_ns": archive.stat().st_mtime_ns})
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "psim_training_manifest.json"
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read existing PSIM training manifest: {manifest_path}") from exc
        if previous != manifest:
            if (output_dir / "large_field_model.pt").exists():
                raise ValueError(
                    f"Cached PSIM mapping changed for an existing checkpoint in {output_dir}. "
                    "Keep its manifest unchanged or choose a new --output_dir."
                )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return records


@lru_cache(maxsize=1)
def load_physical_psim_fields(archive):
    with np.load(archive, allow_pickle=False) as data:
        if "class_names" not in data or list(data["class_names"].astype(str)) != list(CLASS_NAMES):
            raise ValueError(f"Missing/mismatched four-class order: {archive}")
        if ("class_id_map" not in data or "mean_lath_width_um_field" not in data
                or not any(key in data for key in
                           ("dark_contrast_area_fraction_field", "beta_phase_fraction_field"))):
            raise ValueError(f"{archive} lacks physical PSIM fields/final hard map; pixel-width proxies are forbidden.")
        labels = np.asarray(data["class_id_map"])
        beta = np.asarray(read_dark_contrast_alias(
            data, "dark_contrast_area_fraction_field", "beta_phase_fraction_field"))
        width = np.asarray(data["mean_lath_width_um_field"])
    if labels.ndim != 2 or beta.shape != labels.shape or width.shape != labels.shape:
        raise ValueError(f"PSIM field dimensions disagree: {archive}")
    if (not np.isfinite(labels).all() or not np.isin(labels, np.arange(4)).all()
            or not np.isfinite(beta).all() or not np.isfinite(width).all()
            or (beta < 0).any() or (beta > 1).any() or (width < 0).any()):
        raise ValueError(f"Invalid PSIM hard IDs or physical field values: {archive}")
    labels = labels.astype(np.uint8)
    # Match the property pipeline's per-image threshold + retained-class
    # renormalization. Do not reclassify rare pixels by a fingerprint centroid.
    # Sentinel 255 is excluded from class-area denominators, not from images.
    counts = np.bincount(labels.ravel(), minlength=4) / labels.size
    rejected = np.flatnonzero((counts > 0) & (counts < HARD_CLASS_MIN_RATIO))
    for class_id in rejected:
        labels[labels == class_id] = 255
    return labels, beta.astype(np.float32), width.astype(np.float32)


def crop_image_and_psim(path, archive, cfg, rng, augment=True, crop_metadata=None):
    fields = load_physical_psim_fields(archive)
    with Image.open(path) as source:
        width, height = source.size
        if fields[0].shape != (height, width):
            raise ValueError(f"PSIM/source dimensions mismatch for {path}; no silent resizing is allowed.")
        size = cfg.resolution
        if min(width, height) < size:
            raise ValueError(f"{path} is smaller than the native training crop {size}.")
        x = int(rng.integers(width - size + 1)); y = int(rng.integers(height - size + 1))
        crop = np.asarray(source.convert("RGB").crop((x, y, x + size, y + size)))
    if crop_metadata is not None:
        crop_metadata.update({"xywh": [x, y, size, size], "source_size_wh": [width, height],
                              "augmentation_enabled": bool(augment)})
    maps = [value[y:y + size, x:x + size] for value in fields]
    if augment:
        for axis in (0, 1):
            if rng.random() < .5:
                crop = np.flip(crop, axis=axis); maps = [np.flip(value, axis=axis) for value in maps]
    return Image.fromarray(np.ascontiguousarray(crop)), tuple(np.ascontiguousarray(value) for value in maps)


def make_cached_spatial_condition(fields, cfg, fp_mean, fp_std):
    labels, beta, width = fields
    fp_raw = np.array([beta.mean(), width.mean()], dtype=np.float32)
    if fp_raw[1] <= 0:
        raise ValueError("Training crop has no valid positive lath-width measurement.")
    valid_labels = labels[labels < 4]
    if not valid_labels.size:
        raise ValueError("Crop contains only excluded rare-class pixels; no hard-class training condition is available.")
    fractions = np.bincount(valid_labels, minlength=4).astype(np.float32) / valid_labels.size
    features = dict(zip(FP_NAMES, fp_raw))
    features.update({f"{name}_fraction": float(fractions[i]) for i, name in enumerate(CLASS_NAMES)})
    result = make_inverse_spatial_condition(features, cfg, fp_mean, fp_std)
    size, grid = cfg.resolution, cfg.patch_grid
    cell = size // grid; latent_cell = size // cfg.latent_downsample // grid
    entropies = []
    for iy in range(grid):
        for ix in range(grid):
            ys, xs = slice(iy * cell, (iy + 1) * cell), slice(ix * cell, (ix + 1) * cell)
            lys, lxs = slice(iy * latent_cell, (iy + 1) * latent_cell), slice(ix * latent_cell, (ix + 1) * latent_cell)
            local = labels[ys, xs]
            valid_local = local[local < 4]
            fractions_cell = (np.bincount(valid_local, minlength=4).astype(np.float32) / valid_local.size
                              if valid_local.size else fractions)
            local_fp = normalize_fp([beta[ys, xs].mean(), width[ys, xs].mean()], fp_mean, fp_std)
            entropy = class_entropy(fractions_cell); entropies.append(entropy)
            result["spatial_cond"][CLASS_CHANNEL_SLICE, lys, lxs] = fractions_cell[:, None, None]
            result["spatial_cond"][FP_CHANNEL_SLICE, lys, lxs] = local_fp[:, None, None]
            result["spatial_cond"][UNCERTAINTY_CHANNEL, lys, lxs] = entropy
    result["class_assignment_entropy"] = np.float32(np.mean(entropies))
    return result


def physical_fp_stats(cfg, image_paths, records):
    """New normalization is measured from physical cached fields, never old pixel stats."""
    if cfg.fingerprint_stats_path:
        raise ValueError("Remove fingerprint_stats_path: physical-v3 computes its own normalization from cached PSIM fields.")
    rng = np.random.default_rng(cfg.seed)
    values = []
    for index in range(max(len(image_paths), cfg.cache_stats_samples)):
        path = image_paths[index % len(image_paths)]
        _, (_, beta, width) = crop_image_and_psim(path, records[str(path)], cfg, rng, augment=False)
        values.append([beta.mean(), width.mean()])
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or (values[:, 1] <= 0).any():
        raise ValueError("Invalid physical crop fingerprints.")
    return values.mean(0), np.maximum(values.std(0), np.array([1e-4, 1e-4], dtype=np.float32))


class LargeFieldMicrostructureDataset(Dataset):
    def __init__(self, image_paths, cfg, fp_mean, fp_std, class_means, records):
        self.image_paths = list(image_paths)
        self.cfg = cfg
        self.fp_mean = fp_mean
        self.fp_std = fp_std
        self.class_means = class_means
        self.records = records

    def __len__(self):
        return max(len(self.image_paths), 1) * 256

    def __getitem__(self, idx):
        return self._sample(idx, augment=True)

    def validation_sample(self, idx):
        return self._sample(idx, augment=False)

    def _sample(self, idx, augment):
        path = self.image_paths[idx % len(self.image_paths)]
        # One index always produces the same native crop, including after a
        # mid-epoch resume; the expanded dataset still offers many crops/image.
        rng = np.random.default_rng((self.cfg.seed + idx * 10007) % 2**32)
        metadata = {}
        crop, fields = crop_image_and_psim(path, self.records[str(path)], self.cfg, rng,
                                           augment=augment, crop_metadata=metadata)
        result = make_cached_spatial_condition(fields, self.cfg, self.fp_mean, self.fp_std)
        return {
            "pixel_values": torch.from_numpy(np.array(crop, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0),
            "spatial_cond": torch.from_numpy(result["spatial_cond"]).float(),
            "soft_fraction": torch.from_numpy(result["soft_fraction"]).float(),
            "fingerprint": torch.from_numpy(result["fingerprint"]).float(),
            "fingerprint_raw": torch.from_numpy(result["fingerprint_raw"]).float(),
            "connectivity": torch.from_numpy(result["connectivity"]).float(),
            "class_assignment_entropy": torch.tensor(result["class_assignment_entropy"], dtype=torch.float32),
            "class_id": torch.tensor(result["class_id"], dtype=torch.long),
            "path": path,
            "crop_metadata": metadata,
        }


class GlobalMorphologyToken(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        fp_dim=FP_DIM,
        connectivity_dim=CONNECTIVITY_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim + len(CLASS_NAMES) + connectivity_dim + 1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fingerprint, soft_fraction, connectivity, class_assignment_entropy):
        x = torch.cat(
            [
                fingerprint,
                soft_fraction,
                connectivity,
                class_assignment_entropy.reshape(-1, 1),
            ],
            dim=-1,
        )
        return self.net(x)


class LargeFieldSpatialConditioner(nn.Module):
    def __init__(self, in_channels=SPATIAL_CONDITION_CHANNELS, out_channels=4, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spatial_cond):
        return self.net(spatial_cond)


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=8.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
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


def base_weight_sha256(vae, unet, text_encoder):
    """Hash the loaded base weights before LoRA injection or dtype conversion."""
    digest = hashlib.sha256()
    for label, model in (("vae", vae), ("unet", unet), ("text_encoder", text_encoder)):
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{label}.{name}:{value.dtype}:{value.shape}\n".encode("utf-8"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def load_initial_lora(unet, cfg, model_path, base_digest):
    if not cfg.initial_lora_checkpoint:
        return ""
    source = os.path.abspath(os.path.expanduser(cfg.initial_lora_checkpoint))
    checkpoint = safe_torch_load(source)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("unet_lora"), dict):
        raise ValueError(f"{source} does not contain a usable unet_lora state.")
    old_cfg = checkpoint.get("args") or checkpoint.get("config") or {}
    if not isinstance(old_cfg, dict):
        old_cfg = vars(old_cfg)
    old_base = str(old_cfg.get("resolved_pretrained_model_name_or_path") or
                   old_cfg.get("pretrained_model_name_or_path") or "")
    old_digest = checkpoint.get("base_weight_sha256")
    if not old_digest:
        logger.warning("Initial micrograph checkpoint has no base hash; matching paths/metadata do not prove weight identity.")
    if old_digest:
        if old_digest != base_digest:
            raise ValueError("Initial LoRA was trained on different base weights.")
    elif not (old_base and normalized_model_path(old_base) ==
              normalized_model_path(model_path)):
        if not cfg.allow_unverified_initial_lora:
            raise ValueError(
                f"Initial LoRA base {old_base!r} cannot be verified against {model_path!r}. "
                "Supply the matching base or explicitly set --allow_unverified_initial_lora."
            )
        logger.warning("Initial LoRA base identity is unverified; visual quality cannot be assumed.")
    if "lora_rank" in old_cfg and int(old_cfg["lora_rank"]) != cfg.lora_rank:
        raise ValueError("Initial LoRA rank differs from the large-field LoRA rank.")
    if "lora_alpha" in old_cfg and float(old_cfg["lora_alpha"]) != cfg.lora_alpha:
        raise ValueError("Initial LoRA alpha differs from the large-field LoRA alpha.")
    expected = {name: p for name, p in unet.named_parameters() if "lora_" in name}
    state = checkpoint["unet_lora"]
    normalized = {}
    for name, value in state.items():
        key = name
        for prefix in ("module.", "unet_wrapper.", "unet."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in normalized:
            raise ValueError(f"Duplicate initial LoRA tensor: {key}")
        normalized[key] = value
    if set(normalized) != set(expected):
        raise ValueError(
            f"Initial LoRA tensors do not match this UNet: "
            f"missing={sorted(set(expected) - set(normalized))[:6]}, "
            f"unexpected={sorted(set(normalized) - set(expected))[:6]}"
        )
    with torch.no_grad():
        for name, target in expected.items():
            value = normalized[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"Initial LoRA shape mismatch for {name}.")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    logger.info(f"Initialized {len(expected)} LoRA tensors from {source}")
    return source


@torch.no_grad()
def merge_lora_into_unet(unet):
    """Convert inherited LoRA into ordinary Linear weights before full tuning."""
    count = 0
    for name, module in list(unet.named_modules()):
        if isinstance(module, LoRALinear):
            delta = module.lora_up.weight.float() @ module.lora_down.weight.float()
            module.base_layer.weight.add_(delta.to(module.base_layer.weight), alpha=module.scale)
            _set_nested_module(unet, name, module.base_layer)
            count += 1
    return count


def cpu_state_dict(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


class ModelEMA:
    """Track all three trained modules; store on CPU by default, not a second GPU UNet."""
    def __init__(self, modules, decay=0.999, device="cpu"):
        self.decay = float(decay)
        self.device = device
        self.num_updates = 0
        self.shadow = {
            label: {name: value.detach().to(device=device, dtype=torch.float32).clone()
                    if value.is_floating_point() else value.detach().to(device=device).clone()
                    for name, value in module.state_dict().items()}
            for label, module in modules.items()
        }

    @torch.no_grad()
    def update(self, modules):
        self.num_updates += 1
        # Early EMA must not remain close to the unfitted initialization for hundreds of steps.
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for label, module in modules.items():
            current = module.state_dict()
            if set(current) != set(self.shadow[label]):
                raise ValueError(f"EMA module structure changed: {label}")
            for name, value in current.items():
                target = self.shadow[label][name]
                value = value.detach().to(device=target.device, dtype=target.dtype)
                if target.is_floating_point():
                    target.mul_(decay).add_(value, alpha=1 - decay)
                else:
                    target.copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {label: {name: value.detach().cpu() for name, value in values.items()}
                           for label, values in self.shadow.items()}}

    def load_state_dict(self, state):
        if float(state["decay"]) != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA configuration/modules differ from saved training state.")
        for label, expected in self.shadow.items():
            values = state["shadow"][label]
            if set(expected) != set(values) or any(expected[k].shape != values[k].shape for k in expected):
                raise ValueError(f"EMA tensors differ: {label}")
            for name, target in expected.items():
                target.copy_(values[name].to(device=target.device, dtype=target.dtype))
        self.num_updates = int(state["num_updates"])

    @contextmanager
    def apply(self, modules):
        backups = {label: {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
                   for label, module in modules.items()}
        try:
            for label, module in modules.items():
                module.load_state_dict(self.shadow[label], strict=True)
            yield
        finally:
            for label, module in modules.items():
                module.load_state_dict(backups[label], strict=True)


def condition_strength(cfg, step):
    if step < cfg.condition_warmup_steps:
        return 0.0
    if cfg.condition_ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - cfg.condition_warmup_steps) / cfg.condition_ramp_steps))


def inject_global_residual(text_hidden, token, strength, scale):
    hidden = text_hidden.clone()
    hidden[:, -1, :] = hidden[:, -1, :] + float(strength * scale) * token
    return hidden


def unwrap_train_modules(accelerator, unet, global_token, spatial_conditioner):
    return {"unet": accelerator.unwrap_model(unet),
            "global_morphology_token": accelerator.unwrap_model(global_token),
            "large_field_conditioner": accelerator.unwrap_model(spatial_conditioner)}


def restore_generation_weights(checkpoint, unet, global_token, spatial_conditioner, use_ema=True):
    """Inverse callers must use this full-state loader, not the legacy LoRA loader."""
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a v4 full-UNet checkpoint; do not load it as unet_lora.")
    ema = checkpoint.get("ema_state_dict")
    if use_ema and ema is not None:
        states = ema["shadow"]
        source = "ema"
    else:
        states = {"unet": checkpoint["unet_state_dict"],
                  "global_morphology_token": checkpoint["global_morphology_token"],
                  "large_field_conditioner": checkpoint["large_field_conditioner"]}
        source = "raw"
    for label, module in (("unet", unet), ("global_morphology_token", global_token),
                          ("large_field_conditioner", spatial_conditioner)):
        module.load_state_dict(states[label], strict=True)
        module.eval()
    return {"weights": source, "active_condition_strength": checkpoint["active_condition_strength"],
            "config": checkpoint["config"], "noise_scheduler_config": checkpoint["noise_scheduler_config"]}


@torch.no_grad()
def build_generation_condition(cfg, tokenizer, text_encoder, global_token, spatial_conditioner,
                               fractions, normalized_fingerprint, connectivity, entropy,
                               spatial, latent_hw, strength, device, dtype=torch.float32):
    """Shared by verification and inverse inference; inputs use cached physical contract.

    For recipes, first use make_inverse_spatial_condition() with saved fp_mean/std.
    strength must come from checkpoint['active_condition_strength'], not a hardcoded 1.
    """
    prompt = encode_mixed_prompts(tokenizer, text_encoder, fractions, device, dtype,
                                  json.loads(cfg.class_prompts_json))
    empty = encode_empty_prompts(tokenizer, text_encoder, fractions.shape[0], device, dtype)
    token = global_token(normalized_fingerprint, fractions, connectivity, entropy)
    prompt = inject_global_residual(prompt, token.float(), strength, cfg.global_condition_scale)
    bias = spatial_conditioner(spatial).float() * strength
    if bias.shape[-2:] != tuple(latent_hw):
        bias = F.interpolate(bias, size=latent_hw, mode="bilinear", align_corners=False)
    return prompt, empty, bias


@torch.no_grad()
def sample_full_field(unet, scheduler, conditional_hidden, empty_hidden, spatial_bias,
                      latent_shape, device, dtype, steps, guidance, seed,
                      spatial_scale, autocast_context=nullcontext):
    """One whole latent field, no patch generation/fusion or reference-image latent."""
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    latent = latent * scheduler.init_noise_sigma
    for timestep in tqdm(scheduler.timesteps, desc=f"Whole-field CFG={guidance:g}", leave=False):
        model_input = scheduler.scale_model_input(latent, timestep)
        with autocast_context():
            # Separate passes avoid doubling high-resolution activation memory.
            cond_prediction = unet(model_input + spatial_scale * spatial_bias, timestep,
                                   encoder_hidden_states=conditional_hidden).sample
            if guidance > 1:
                uncond_prediction = unet(model_input, timestep,
                                         encoder_hidden_states=empty_hidden).sample
                prediction = uncond_prediction + guidance * (cond_prediction - uncond_prediction)
            else:
                prediction = cond_prediction
        latent = scheduler.step(prediction.float(), timestep, latent, generator=generator).prev_sample
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite sampling latents at timestep {timestep}")
    return latent


def capture_rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def encode_mixed_prompts(tokenizer, text_encoder, class_fractions, device, dtype, class_prompts=None):
    """Encode a soft class-mixture prompt for every crop.

    A crop with comparable Basket_weave and Discontinuous probability should
    not inherit the language embedding of whichever class wins by 0.01.  A
    weighted text embedding keeps the same semantic mixture that is supplied to
    the global and spatial numerical conditions.
    """
    fractions = torch.as_tensor(class_fractions, dtype=torch.float32, device=device)
    if fractions.ndim != 2 or fractions.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"class_fractions must have shape (batch, {len(CLASS_NAMES)}), got {tuple(fractions.shape)}"
        )
    batch_size = fractions.shape[0]
    class_prompts = list(PROMPT_DICT.values()) if class_prompts is None else class_prompts
    prompts = [class_prompts[class_id] for _ in range(batch_size) for class_id in range(len(CLASS_NAMES))]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    hidden = text_encoder(input_ids)[0]
    hidden = hidden.reshape(batch_size, len(CLASS_NAMES), hidden.shape[1], hidden.shape[2])
    return torch.sum(hidden * fractions[:, :, None, None], dim=1).to(dtype=dtype)


@torch.no_grad()
def encode_empty_prompts(tokenizer, text_encoder, batch_size, device, dtype):
    tokenized = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokenized.input_ids.to(device))[0].to(dtype=dtype)


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples]),
        "spatial_cond": torch.stack([x["spatial_cond"] for x in examples]),
        "soft_fraction": torch.stack([x["soft_fraction"] for x in examples]),
        "fingerprint": torch.stack([x["fingerprint"] for x in examples]),
        "fingerprint_raw": torch.stack([x["fingerprint_raw"] for x in examples]),
        "connectivity": torch.stack([x["connectivity"] for x in examples]),
        "class_assignment_entropy": torch.stack([x["class_assignment_entropy"] for x in examples]),
        "class_id": torch.stack([x["class_id"] for x in examples]),
        "path": [x["path"] for x in examples],
    }


def save_checkpoint(cfg, accelerator, unet, global_token, spatial_conditioner, fp_mean, fp_std,
                    class_means, step, base_digest, initial_lora_source, noise_scheduler_config, final=False, ema=None):
    if not accelerator.is_main_process:
        return
    ensure_dir(cfg.output_dir)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_token = accelerator.unwrap_model(global_token)
    unwrapped_cond = accelerator.unwrap_model(spatial_conditioner)

    full = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "requires_full_unet": True,
        "preferred_generation_weights": "ema" if ema is not None else "raw",
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "active_condition_strength": condition_strength(cfg, step),
        "structure_performance_contract": CONTRACT_VERSION,
        "training_mode": "large_field_full_unet_psim_hard_physical",
        "scientific_task": "morphology_conditioned_image_generation",
        "uses_mechanical_property_labels": False,
        "direction_conditioning": "none",
        "training_image_scope": "all_images_regardless_of_direction_or_filename",
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        # Legacy wire keys keep existing strict v4 inference loaders usable.
        "condition_schema": legacy_condition_schema(),
        "canonical_condition_schema": condition_schema(),
        "contrast_terminology": dict(CONTRAST_TERMINOLOGY),
        "microns_per_source_pixel": cfg.microns_per_source_pixel,
        "psim_manifest_sha256": hashlib.sha256(Path(cfg.output_dir, "psim_training_manifest.json").read_bytes()).hexdigest(),
        "unet_state_dict": cpu_state_dict(unwrapped_unet),
        "global_morphology_token": cpu_state_dict(unwrapped_token),
        "large_field_conditioner": cpu_state_dict(unwrapped_cond),
        "fp_mean": np.asarray(fp_mean, dtype=np.float32),
        "fp_std": np.asarray(fp_std, dtype=np.float32),
        "class_means": np.asarray(class_means, dtype=np.float32),
        "class_names": CLASS_NAMES,
        "fp_names": list(LEGACY_FP_NAMES),  # compatibility wire field only
        "fingerprint_names": list(FP_NAMES),
        "fingerprint_display_labels": list(FINGERPRINT_DISPLAY_LABELS),
        "fingerprint_stats_source": "cached_PSIM_physical_crop_means",
        "config": asdict(cfg),
        "base_weight_sha256": base_digest,
        "noise_scheduler_config": dict(noise_scheduler_config),
        "initial_lora_checkpoint": initial_lora_source,
        "initial_diffusers_model_dir": cfg.initial_diffusers_model_dir,
        "fingerprint_dim": FP_DIM,
        "connectivity_names": CONNECTIVITY_NAMES,
        "global_condition_dim": GLOBAL_CONDITION_DIM,
        "spatial_condition_channels": SPATIAL_CONDITION_CHANNELS,
        "step": int(step),
    }
    if not final and cfg.keep_step_snapshots:
        step_path = os.path.join(cfg.output_dir, f"large_field_model_step{step}.pt")
        torch.save(full, step_path + ".tmp")
        os.replace(step_path + ".tmp", step_path)
    checkpoint_path = os.path.join(cfg.output_dir, "large_field_model.pt")
    temporary_path = checkpoint_path + ".tmp"
    torch.save(full, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    np.savez_compressed(
        os.path.join(cfg.output_dir, "large_field_condition_stats.npz"),
        fp_mean=np.asarray(fp_mean, dtype=np.float32),
        fp_std=np.asarray(fp_std, dtype=np.float32),
        schema_version=np.array(CONDITION_SCHEMA_VERSION),
        fingerprint_units=np.array(["fraction", "um"]),
        fingerprint_names=np.array(FP_NAMES),
        fingerprint_display_labels=np.array(FINGERPRINT_DISPLAY_LABELS),
        legacy_fp_names=np.array(LEGACY_FP_NAMES),
        class_means=np.asarray(class_means, dtype=np.float32),
        connectivity_min=np.zeros(CONNECTIVITY_DIM, dtype=np.float32),
        connectivity_max=np.ones(CONNECTIVITY_DIM, dtype=np.float32),
    )
    with open(os.path.join(cfg.output_dir, "large_field_condition_schema.json"), "w", encoding="utf-8") as f:
        json.dump(condition_schema(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.output_dir, "large_field_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def save_training_state(cfg, accelerator, optimizer, lr_scheduler, step, epoch, next_batch,
                        epoch_generator_state, base_digest, unet, global_token, spatial_conditioner, ema,
                        initial_lora_source=""):
    if not accelerator.is_main_process:
        return
    state = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "initial_lora_checkpoint": initial_lora_source,
        "unet_state_dict": cpu_state_dict(accelerator.unwrap_model(unet)),
        "global_morphology_token": cpu_state_dict(accelerator.unwrap_model(global_token)),
        "large_field_conditioner": cpu_state_dict(accelerator.unwrap_model(spatial_conditioner)),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "step": step, "epoch": epoch, "next_batch": next_batch,
        "epoch_generator_state": epoch_generator_state,
        "rng": capture_rng_state(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "amp_scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
        "base_weight_sha256": base_digest,
        "config": {key: value for key, value in asdict(cfg).items()
                   if key not in ("resume_training_state", "initial_lora_checkpoint",
                                  "initial_diffusers_model_dir", "allow_unverified_initial_lora")},
    }
    path = os.path.join(cfg.output_dir, "training_state.pt")
    temporary_path = path + ".tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)
    if cfg.keep_step_snapshots:
        step_path = os.path.join(cfg.output_dir, f"training_state_step{step}.pt")
        torch.save(state, step_path + ".tmp")
        os.replace(step_path + ".tmp", step_path)


def load_training_state(cfg, accelerator, unet, global_token, spatial_conditioner,
                        optimizer, lr_scheduler, base_digest, ema):
    path = os.path.abspath(os.path.expanduser(cfg.resume_training_state))
    state = safe_torch_load(path)
    # Self-contained, atomic resume file: interruption between inference/resume
    # exports cannot mix model weights from one step with another optimizer.
    if state.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("Legacy LoRA training states cannot resume full-UNet optimization.")
    if state["base_weight_sha256"] != base_digest:
        raise ValueError("Saved training base differs from the loaded VAE/UNet/CLIP base.")
    current = {key: value for key, value in asdict(cfg).items()
               if key not in ("resume_training_state", "initial_lora_checkpoint",
                              "initial_diffusers_model_dir", "allow_unverified_initial_lora")}
    if state["config"] != current:
        raise ValueError("Training configuration changed; exact optimizer/scheduler resume is unsafe.")
    modules = (
        (accelerator.unwrap_model(unet), state["unet_state_dict"]),
        (accelerator.unwrap_model(global_token), state["global_morphology_token"]),
        (accelerator.unwrap_model(spatial_conditioner), state["large_field_conditioner"]),
    )
    for module, values in modules:
        module.load_state_dict(values, strict=True)
    saved_ema = state.get("ema_state_dict")
    if (ema is None) != (saved_ema is None):
        raise ValueError("EMA enablement differs from saved training state.")
    if ema is not None:
        ema.load_state_dict(saved_ema)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    saved_scaler = state.get("amp_scaler")
    if accelerator.scaler is not None:
        if saved_scaler is None:
            raise ValueError("This fp16 resume has no saved AMP scaler; use a compatible full training-state export.")
        accelerator.scaler.load_state_dict(saved_scaler)
    elif saved_scaler is not None:
        raise ValueError("Saved AMP scaler cannot be restored in the current precision mode.")
    state["initial_lora_checkpoint"] = state.get("initial_lora_checkpoint", "")
    return state


@torch.no_grad()
def save_validation_images(cfg, accelerator, dataset, vae, tokenizer, text_encoder,
                           unet, global_token, spatial_conditioner, noise_scheduler,
                           weight_dtype, step, ema=None):
    if not accelerator.is_main_process or cfg.validation_steps <= 0:
        return
    from diffusers import DDIMScheduler, DDPMScheduler

    modules = (accelerator.unwrap_model(unet), accelerator.unwrap_model(global_token),
               accelerator.unwrap_model(spatial_conditioner))
    modes = [module.training for module in modules]
    rng_state = capture_rng_state()
    ema_modules = unwrap_train_modules(accelerator, unet, global_token, spatial_conditioner)
    weights_stack = ExitStack()
    scheduler_cls = DDPMScheduler if cfg.validation_scheduler == "ddpm" else DDIMScheduler
    strength = condition_strength(cfg, step)
    try:
        if ema is not None:
            weights_stack.enter_context(ema.apply(ema_modules))
        for module in modules:
            module.eval()
        device = accelerator.device
        directory = os.path.join(cfg.output_dir, "fixed_validation", f"step_{step:06d}")
        ensure_dir(directory)
        source_count = len(dataset.image_paths)
        count = min(max(int(cfg.validation_num_samples), 1), source_count)
        indices = np.linspace(0, source_count - 1, count, dtype=int).tolist()
        report = {"step": int(step), "native_resolution": int(cfg.resolution),
                  "sampling_steps": int(cfg.validation_sampling_steps), "guidance": cfg.validation_guidance,
                  "scheduler": cfg.validation_scheduler, "weights": "ema" if ema is not None else "raw",
                  "ema_updates": ema.num_updates if ema is not None else 0,
                  "active_condition_strength": strength, "condition_schema_version": CONDITION_SCHEMA_VERSION,
                  "scheduler_config": dict(noise_scheduler.config),
                  "class_names": CLASS_NAMES, "samples": [],
                  "fingerprint_names": list(FP_NAMES),
                  "fingerprint_display_labels": list(FINGERPRINT_DISPLAY_LABELS),
                  "contrast_terminology": dict(CONTRAST_TERMINOLOGY),
                  "interpretation": "Fixed training-crop diagnostics, not held-out validation. "
                                    "Generation starts from random noise; no reference latent is used. "
                                    "Pixel MAE is reported only for the paired VAE round-trip."}
        for ordinal, index in enumerate(indices):
            sample = dataset.validation_sample(index)
            pixel = sample["pixel_values"][None].to(device=device, dtype=weight_dtype)
            fractions = sample["soft_fraction"][None].to(device=device, dtype=weight_dtype)
            fingerprint = sample["fingerprint"][None].to(device=device, dtype=weight_dtype)
            connectivity = sample["connectivity"][None].to(device=device, dtype=weight_dtype)
            entropy = sample["class_assignment_entropy"][None].to(device=device, dtype=weight_dtype)
            spatial = sample["spatial_cond"][None].to(device=device, dtype=weight_dtype)
            real_latents = vae.encode(pixel).latent_dist.mode() * vae.config.scaling_factor
            reconstruction = vae.decode(real_latents / vae.config.scaling_factor).sample
            with accelerator.autocast():
                prompt, empty, bias = build_generation_condition(
                    cfg, tokenizer, text_encoder, modules[1], modules[2], fractions, fingerprint,
                    connectivity, entropy, spatial, real_latents.shape[-2:], strength, device, weight_dtype)
            seed = cfg.seed + ordinal
            filenames = {}
            for name, tensor in (("real", pixel), ("vae_reconstruction", reconstruction)):
                image = ((tensor[0].float().clamp(-1, 1) + 1) * 127.5).byte()
                filename = f"sample_{ordinal:02d}_{name}.png"
                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(os.path.join(directory, filename))
                filenames[name] = filename
            variants = [("conditional_generation", cfg.validation_guidance)]
            if cfg.validation_compare_no_cfg and cfg.validation_guidance != 1:
                variants.append(("conditional_generation_no_cfg", 1.0))
            logger.info(f"Validation step={step}, sample={ordinal}, weights={report['weights']}, "
                        f"condition_strength={strength:.3f}, {cfg.validation_scheduler} steps={cfg.validation_sampling_steps}")
            for name, guidance in variants:
                latents = sample_full_field(
                    modules[0], scheduler_cls.from_config(noise_scheduler.config), prompt, empty, bias,
                    tuple(real_latents.shape), device, weight_dtype, cfg.validation_sampling_steps,
                    guidance, seed, cfg.spatial_condition_scale, accelerator.autocast)
                generated = vae.decode(latents.float() / vae.config.scaling_factor).sample
                image = ((generated[0].float().clamp(-1, 1) + 1) * 127.5).byte()
                filename = f"sample_{ordinal:02d}_{name}.png"
                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(os.path.join(directory, filename))
                filenames[name] = filename
                del latents, generated
            report["samples"].append({
                "dataset_index": index, "source_image": str(sample["path"]),
                "crop": sample["crop_metadata"], "seed": seed,
                "hard_class_fractions": sample["soft_fraction"].tolist(),
                "physical_fingerprint": sample["fingerprint_raw"].tolist(),
                **named_contrast_fingerprint(sample["fingerprint_raw"].tolist()),
                "vae_reconstruction_mae_0_1": float(
                    (pixel.float() - reconstruction.float().clamp(-1, 1)).abs().mean().item() / 2),
                "images": filenames,
            })
            del pixel, fractions, fingerprint, connectivity, entropy, spatial, real_latents
            del reconstruction, bias, prompt, empty
        with open(os.path.join(directory, "validation_manifest.json"), "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        logger.info(f"Fixed-condition validation saved to {directory}")
    finally:
        weights_stack.close()
        restore_rng_state(rng_state)
        for module, mode in zip(modules, modes):
            module.train(mode)


def main():
    cfg = parse_args()
    configure_offline_loading(cfg)
    ensure_dir(cfg.output_dir)
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=cfg.output_dir),
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Full UNet LR={cfg.learning_rate:g}; condition LR={cfg.condition_learning_rate:g}; "
                f"appearance warmup={cfg.condition_warmup_steps} steps; ramp={cfg.condition_ramp_steps}; "
                f"validation={cfg.validation_scheduler}/{cfg.validation_sampling_steps} steps, "
                f"CFG={cfg.validation_guidance:g}, EMA={cfg.use_ema}.")
    set_seed(cfg.seed)

    image_paths = list_image_files(cfg.large_image_dir)
    if not image_paths:
        raise RuntimeError(f"No large images found under: {cfg.large_image_dir}")
    logger.info(f"Found {len(image_paths)} large images; using ALL images without direction filtering. Numeric filenames are supported.")
    logger.info(
        f"High-resolution native crop profile: {cfg.resolution}x{cfg.resolution}px "
        f"({cfg.resolution // cfg.latent_downsample}x{cfg.resolution // cfg.latent_downsample} latent; "
        f"{cfg.patch_grid}x{cfg.patch_grid} condition grid). Original source pixels are not resized."
    )

    if cfg.prepare_missing_psim:
        if accelerator.num_processes != 1:
            raise ValueError("Prepare training PSIM in a single process first: python train_large_field.py --prepare_psim_only; then launch distributed training.")
        prepare_training_psim(cfg, image_paths)
    records = index_existing_psim(cfg, image_paths)
    logger.info(f"Reusing {len(records)} cached training-image PSIM records; no PSIM inference was run.")
    if cfg.prepare_psim_only:
        print(f"Training PSIM preparation complete for {len(records)} images. No large-field diffusion training was started.")
        return
    fp_mean, fp_std = physical_fp_stats(cfg, image_paths, records)
    # Compatibility field only. Classes come from cached final hard IDs.
    class_means = np.zeros((len(CLASS_NAMES), FP_DIM), dtype=np.float32)
    logger.info(f"Fingerprint order: {list(FINGERPRINT_DISPLAY_LABELS)}")
    logger.info(f"fp_mean={np.round(fp_mean, 5)}")
    logger.info(f"fp_std={np.round(fp_std, 5)}")
    logger.info(f"class_means={np.round(class_means, 4)}")

    dataset = LargeFieldMicrostructureDataset(image_paths, cfg, fp_mean, fp_std, class_means, records)
    if cfg.validate_psim_only:
        # Validate numerical conditioning for every image without SD imports,
        # weights, generation or PSIM inference.
        for index in range(len(image_paths)):
            dataset[index]
        print(f"Physical PSIM conditioning validated for {len(image_paths)} images; no model was loaded.")
        return
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Keep trainable master weights and VAE/CLIP in FP32. Accelerator applies
    # autocast to forward passes; casting trainable parameters to FP16 directly
    # would break GradScaler unscaling and destabilize full-UNet fine tuning.
    weight_dtype = torch.float32

    model_path = resolve_pretrained_model_name_or_path(cfg)
    logger.info(f"Building VAE, UNet, CLIP text encoder from: {model_path}")
    vae = load_diffusers_component(AutoencoderKL, model_path, "vae", cfg, "VAE")
    tokenizer = load_diffusers_component(CLIPTokenizer, model_path, "tokenizer", cfg, "CLIPTokenizer")
    text_encoder = load_diffusers_component(CLIPTextModel, model_path, "text_encoder", cfg, "CLIPTextModel")
    unet = load_diffusers_component(UNet2DConditionModel, model_path, "unet", cfg, "UNet2DConditionModel")
    noise_scheduler = load_diffusers_component(DDPMScheduler, model_path, "scheduler", cfg, "DDPMScheduler")
    if noise_scheduler.config.prediction_type not in ("epsilon", "v_prediction"):
        raise ValueError("This trainer supports epsilon or v_prediction schedulers.")
    base_digest = base_weight_sha256(vae, unet, text_encoder)
    cfg.pretrained_model_name_or_path = model_path
    logger.info(f"Loaded training-base weight SHA256: {base_digest}")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if cfg.enable_vae_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    if cfg.enable_gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        logger.info("Enabled UNet gradient checkpointing for high-resolution crops.")
    elif cfg.enable_gradient_checkpointing:
        logger.warning("UNet implementation does not expose gradient checkpointing; memory use may be high.")

    initial_lora_source = ""
    if not cfg.resume_training_state and not cfg.initial_diffusers_model_dir and cfg.initial_lora_checkpoint:
        inject_lora_into_unet(unet, rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=0.0)
        initial_lora_source = load_initial_lora(unet, cfg, model_path, base_digest)
        count = merge_lora_into_unet(unet)
        logger.info(f"Merged inherited LoRA into {count} attention Linear weights; training FULL UNet.")
    elif cfg.initial_diffusers_model_dir and not cfg.resume_training_state:
        logger.info("Inherited complete trained Diffusers UNet/VAE/CLIP; initial_lora_checkpoint is not overlaid.")
    unet.requires_grad_(True)
    logger.info("Full-UNet training: convolution + attention + normalization weights are all trainable.")

    global_token = GlobalMorphologyToken(
        embed_dim=768,
        fp_dim=FP_DIM,
        connectivity_dim=CONNECTIVITY_DIM,
    )
    spatial_conditioner = LargeFieldSpatialConditioner(
        in_channels=SPATIAL_CONDITION_CHANNELS,
        out_channels=unet.config.in_channels,
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    global_token.to(accelerator.device, dtype=weight_dtype)
    spatial_conditioner.to(accelerator.device, dtype=weight_dtype)

    unet_params = list(unet.parameters())
    condition_params = list(global_token.parameters()) + list(spatial_conditioner.parameters())
    trainable_params = unet_params + condition_params
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(
        [{"params": unet_params, "lr": cfg.learning_rate},
         {"params": condition_params, "lr": cfg.condition_learning_rate}],
        weight_decay=cfg.adam_weight_decay)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps or cfg.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, global_token, spatial_conditioner, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, global_token, spatial_conditioner, optimizer, dataloader, lr_scheduler
    )
    shuffle_generator = getattr(dataloader, "generator", None) or generator
    vae.eval()
    text_encoder.eval()
    ema = (ModelEMA(unwrap_train_modules(accelerator, unet, global_token, spatial_conditioner),
                    cfg.ema_decay, cfg.ema_device) if cfg.use_ema else None)

    state = None
    if cfg.resume_training_state:
        if cfg.dataloader_num_workers != 0 or accelerator.num_processes != 1:
            raise ValueError("Exact mid-epoch resume currently requires one process and zero data workers.")
        state = load_training_state(cfg, accelerator, unet, global_token, spatial_conditioner,
                                    optimizer, lr_scheduler, base_digest, ema)
        initial_lora_source = state["initial_lora_checkpoint"]
        logger.info(f"Resuming training from step {state['step']}, epoch {state['epoch']}, "
                    f"batch {state['next_batch']}.")
    elif os.path.isfile(os.path.join(cfg.output_dir, "large_field_model.pt")):
        raise FileExistsError(
            "Output directory already contains a checkpoint. Pass --resume_training_state "
            "<output_dir>/training_state.pt, or choose a new --output_dir. "
            "Legacy LoRA optimizer states cannot resume this full-UNet training protocol."
        )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.tracker_project_name, config=asdict(cfg))
        loss_csv = os.path.join(cfg.output_dir, "large_field_training_loss.csv")
        if state is None or not os.path.isfile(loss_csv):
            with open(loss_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "loss", "diffusion_loss", "spatial_bias_l2", "unet_lr", "condition_lr", "condition_strength"])

    global_step = int(state["step"]) if state else 0
    if global_step >= max_train_steps:
        logger.info(f"Checkpoint already reached max_train_steps={max_train_steps}; no new training needed.")
        accelerator.end_training()
        return
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    running_loss = 0.0

    accelerator.wait_for_everyone()
    save_validation_images(cfg, accelerator, dataset, vae, tokenizer, text_encoder,
                           unet, global_token, spatial_conditioner, noise_scheduler,
                           weight_dtype, global_step, ema)
    accelerator.wait_for_everyone()

    for epoch in range(int(state["epoch"]) if state else 0, cfg.num_train_epochs):
        unet.train()
        global_token.train()
        spatial_conditioner.train()
        if state is not None and epoch == state["epoch"]:
            shuffle_generator.set_state(state["epoch_generator_state"])
        epoch_generator_state = shuffle_generator.get_state().clone()

        for batch_index, batch in enumerate(dataloader):
            if state is not None and epoch == state["epoch"] and batch_index < state["next_batch"]:
                continue
            if state is not None:
                restore_rng_state(state["rng"])
                state = None
            with accelerator.accumulate(unet, global_token, spatial_conditioner):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                spatial_cond = batch["spatial_cond"].to(accelerator.device, dtype=weight_dtype)
                soft_fraction = batch["soft_fraction"].to(accelerator.device, dtype=weight_dtype)
                fingerprint = batch["fingerprint"].to(accelerator.device, dtype=weight_dtype)
                connectivity = batch["connectivity"].to(accelerator.device, dtype=weight_dtype)
                class_assignment_entropy = batch["class_assignment_entropy"].to(
                    accelerator.device, dtype=weight_dtype
                )
                bsz = pixel_values.shape[0]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    text_hidden = encode_mixed_prompts(
                        tokenizer,
                        text_encoder,
                        soft_fraction,
                        accelerator.device,
                        weight_dtype,
                        json.loads(cfg.class_prompts_json),
                    )
                    empty_hidden = encode_empty_prompts(tokenizer, text_encoder, bsz, accelerator.device, weight_dtype)

                cond_token = global_token(
                    fingerprint,
                    soft_fraction,
                    connectivity,
                    class_assignment_entropy,
                )
                strength = condition_strength(cfg, global_step)
                encoder_hidden_states = inject_global_residual(
                    text_hidden, cond_token, strength, cfg.global_condition_scale)

                spatial_bias = spatial_conditioner(spatial_cond)
                if spatial_bias.shape[-2:] != latents.shape[-2:]:
                    spatial_bias = F.interpolate(spatial_bias, size=latents.shape[-2:], mode="bilinear", align_corners=False)

                spatial_bias = spatial_bias * strength
                # Drop all morphology/text conditions together, independently per sample.
                # The empty branch at inference must also have zero spatial bias.
                dropped = torch.rand((bsz, 1, 1), device=latents.device) < cfg.condition_dropout
                encoder_hidden_states = torch.where(dropped, empty_hidden, encoder_hidden_states)
                spatial_bias = spatial_bias * (~dropped).reshape(bsz, 1, 1, 1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_input = noisy_latents + float(cfg.spatial_condition_scale) * spatial_bias
                noise_pred = unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                target = (noise_scheduler.get_velocity(latents, noise, timesteps)
                          if noise_scheduler.config.prediction_type == "v_prediction" else noise)
                loss_diff = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                loss_bias = spatial_bias.float().pow(2).mean()
                loss = loss_diff + float(cfg.spatial_bias_l2_weight) * loss_bias
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Non-finite diffusion loss; refusing to save corrupted training weights.")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                progress_bar.update(1)
                global_step += 1
                if ema is not None:
                    ema.update(unwrap_train_modules(accelerator, unet, global_token, spatial_conditioner))
                running_loss += float(loss.detach().item())
                lr = lr_scheduler.get_last_lr()[0]
                accelerator.log({
                    "train/loss": float(loss.detach().item()),
                    "train/diffusion_loss": float(loss_diff.detach().item()),
                    "train/spatial_bias_l2": float(loss_bias.detach().item()),
                    "train/lr": float(lr),
                    "train/condition_lr": float(lr_scheduler.get_last_lr()[1]),
                    "train/condition_strength": float(strength),
                }, step=global_step)

                if accelerator.is_main_process and global_step % 20 == 0:
                    avg = running_loss / 20.0
                    running_loss = 0.0
                    logger.info(
                        f"step={global_step} loss={avg:.4f} diff={loss_diff.item():.4f} "
                        f"bias_l2={loss_bias.item():.5f} lr={lr:.2e}"
                    )
                    with open(os.path.join(cfg.output_dir, "large_field_training_loss.csv"), "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            global_step,
                            float(loss.detach().item()),
                            float(loss_diff.detach().item()),
                            float(loss_bias.detach().item()),
                            float(lr),
                            float(lr_scheduler.get_last_lr()[1]),
                            float(strength),
                        ])

                if global_step % cfg.save_steps == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(cfg, accelerator, unet, global_token, spatial_conditioner,
                                    fp_mean, fp_std, class_means, global_step, base_digest,
                                    initial_lora_source, noise_scheduler.config, ema=ema)
                    save_training_state(cfg, accelerator, optimizer, lr_scheduler, global_step,
                                        epoch, batch_index + 1, epoch_generator_state, base_digest,
                                        unet, global_token, spatial_conditioner, ema, initial_lora_source)
                    accelerator.wait_for_everyone()
                if cfg.validation_steps > 0 and global_step % cfg.validation_steps == 0:
                    accelerator.wait_for_everyone()
                    save_validation_images(cfg, accelerator, dataset, vae, tokenizer, text_encoder,
                                           unet, global_token, spatial_conditioner, noise_scheduler,
                                           weight_dtype, global_step, ema)
                    accelerator.wait_for_everyone()

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break
        if state is not None:
            restore_rng_state(state["rng"])
            state = None

    accelerator.wait_for_everyone()
    save_checkpoint(cfg, accelerator, unet, global_token, spatial_conditioner,
                    fp_mean, fp_std, class_means, global_step, base_digest,
                    initial_lora_source, noise_scheduler.config, final=True, ema=ema)
    save_training_state(cfg, accelerator, optimizer, lr_scheduler, global_step,
                        epoch, batch_index + 1, epoch_generator_state, base_digest,
                        unet, global_token, spatial_conditioner, ema, initial_lora_source)
    if cfg.validation_steps > 0 and global_step % cfg.validation_steps != 0:
        save_validation_images(cfg, accelerator, dataset, vae, tokenizer, text_encoder,
                               unet, global_token, spatial_conditioner, noise_scheduler,
                               weight_dtype, global_step, ema)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Large-field training complete.")


if __name__ == "__main__":
    main()
