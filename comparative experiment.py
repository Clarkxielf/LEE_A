
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
from PIL import Image

VERSION = "four-class-benchmark-v3-scratch-default"


def _discover_project_root():
    """Find the local project without embedding a workstation path."""
    if os.environ.get("LEE_A_PROJECT_ROOT"):
        return Path(os.environ["LEE_A_PROJECT_ROOT"]).expanduser()
    script = Path(__file__).resolve()
    candidates = [script.parent, *script.parents]
    # A common layout is /data8t/LEE/LEE_A/TiClassifier/<script> while the
    # data project is the sibling /data8t/LEE/LEE_A.
    if len(script.parents) >= 3:
        candidates.append(script.parents[2] / "LEE_A")
    for candidate in candidates:
        if (candidate / "data/example_data_train_val_test").is_dir():
            return candidate
    return script.parents[1] if len(script.parents) > 1 else script.parent


_DEFAULT_PROJECT_ROOT = _discover_project_root()
CLASS_NAMES = ["Basket_weave", "Discontinuous", "Widmanstatten", "Alpha_coarsened_Widmanstatten"]
CLASS_LABELS = ["Basket weave", "Discontinuous", "Widmanstatten", "Coarsened\nWidmanstatten"]
ALIASES = {"Lamellar": "Widmanstatten", "AlphaCoarsenedWidmanstatten": CLASS_NAMES[3]}
MODEL_SPECS = {
    "resnet50": ("resnet50.a1_in1k", "ResNet50", 224),
    "densenet121": ("densenet121.ra_in1k", "DenseNet121", 224),
    "vgg16_bn": ("vgg16_bn.tv_in1k", "VGG16-BN", 224),
    "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", "MobileNetV3-L", 224),
    "efficientnetv2_m": ("tf_efficientnetv2_m.in1k", "EfficientNetV2-M", 360),
    "convnext_tiny": ("convnext_tiny.fb_in1k", "ConvNeXt-T", 224),
    "vit_base": ("vit_base_patch16_384.augreg_in1k", "ViT-B/16", 384),
    "swin_tiny": ("swin_tiny_patch4_window7_224.ms_in1k", "Swin-T", 224),
    # Optional compatibility baseline; not the same architecture as TF EfficientNetV2-M.
    "efficientnetv2_rw_m": ("efficientnetv2_rw_m.agc_in1k", "EfficientNetV2-RW-M", 360),
}
DEFAULT_MODELS = list(MODEL_SPECS)[:8]
METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = ["image_id", "path", "split", "label", "class_name", "group_id", "sha256", "width", "height"]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "Pillow", "torch", "torchvision", "timm", "scikit-learn", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-root", type=Path, default=_DEFAULT_PROJECT_ROOT,
                   help="Project root; defaults to LEE_A_PROJECT_ROOT or the directory containing this script's parent.")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--mode", choices=["run", "validate", "report"], default="run")
    p.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=DEFAULT_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    # This benchmark is intended for the complete four-class dataset.  Small-
    #-sample curves are deliberately excluded so the reported comparison is
    # not confused with the main full-data result.
    p.add_argument("--budgets", nargs="+", default=["full"], choices=["full"],
                   help="Only the complete training split is evaluated.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4], help="Equal validation-only LR search grid for each model")
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=.03)
    p.add_argument("--sampling", choices=["balanced", "shuffle"], default="balanced")
    p.add_argument("--augmentation", choices=["matched", "mild", "paper"], default="paper",
                   help="paper reproduces the supplied comparison protocol; matched reproduces train.py's resize/crop-only view.")
    p.add_argument("--mixup-alpha", type=float, default=None, help="Defaults to 0; paper augmentation selects conventional supervised Mixup alpha=.3")
    p.add_argument("--field-size", type=int, default=512, help="Common Resize(short edge) + CenterCrop view, matching train.py")
    p.add_argument("--input-size", type=int, default=0, help="0 uses registered model sizes; e.g. 384 forces a common input resolution")
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument("--scratch", dest="scratch", action="store_true",
                            help="Random initialization (default); uses a separate output namespace.")
    init_group.add_argument("--pretrained", dest="scratch", action="store_false",
                            help="Opt in to ImageNet/pretrained weights; not used for the main fair comparison.")
    p.set_defaults(scratch=True)
    p.add_argument("--weights-json", type=Path, help="Optional model-key -> local pretrained checkpoint map; see README")
    p.add_argument("--offline", action="store_true", help="Use cached/local pretrained weights only; never silently fall back to scratch")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    p.add_argument("--amp", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--allow-nondeterministic", action="store_true")
    p.add_argument("--numeric-class-order", choices=["canonical", "legacy-alphabetical"], default="canonical")
    p.add_argument("--duplicate-policy", choices=["drop_later", "keep_all", "error"], default="drop_later",
                   help="Cross-split exact duplicates: drop later copies (recommended), keep all with a leakage warning, or stop.")
    groups = p.add_mutually_exclusive_group()
    groups.add_argument("--groups-csv", type=Path, help="CSV columns image_id,group_id; image_id is relative to data-root")
    groups.add_argument("--group-regex", help="Exactly one capture group, applied to image_id after split/class")
    p.add_argument("--require-groups", action="store_true", help="Refuse to run without verified group separation")
    p.add_argument("--ours-run", action="append", default=[], metavar="SEED=RESULTS_DIR", help="Import full-data train.py test_results with path-level alignment")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.data_root = (args.data_root or args.project_root / "data/example_data_train_val_test").expanduser().resolve()
    args.output_dir = (args.output_dir or args.project_root / "Diffusion/Deep-TiDiffusion-Outputs-FourClass-v3/classification_benchmark").expanduser().resolve()
    if min(args.epochs, args.batch_size, args.grad_accum, args.field_size) < 1 or args.workers < 0:
        p.error("Epochs, batch size, accumulation and field size must be positive; workers cannot be negative.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("Seeds must be unique integers in [0, 2**32).")
    if len(set(args.models)) != len(args.models):
        p.error("Model names must be unique.")
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs) or len(set(args.lrs)) != len(args.lrs):
        p.error("Learning rates must be finite, positive and unique.")
    if not 0 <= args.label_smoothing < 1 or not 0 <= args.warmup_epochs < args.epochs or args.input_size < 0:
        p.error("Invalid smoothing, warmup or input size.")
    args.mixup_alpha = (.3 if args.augmentation == "paper" else 0.) if args.mixup_alpha is None else args.mixup_alpha
    if not math.isfinite(args.mixup_alpha) or args.mixup_alpha < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("Mixup alpha and weight decay must be finite and nonnegative.")
    if args.scratch and args.weights_json:
        p.error("--scratch cannot be combined with --weights-json.")
    if args.mode == "report" and not args.output_dir.exists():
        p.error("Report mode requires an existing output directory.")
    args.budgets = ["full"]
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return args


def canonical_label(folder, numeric_order):
    name = ALIASES.get(folder, folder)
    if name in CLASS_NAMES:
        return CLASS_NAMES.index(name)
    if folder in ("0", "1", "2", "3"):
        order = CLASS_NAMES if numeric_order == "canonical" else sorted(CLASS_NAMES)
        return CLASS_NAMES.index(order[int(folder)])
    raise ValueError(f"Unknown class folder: {folder!r}; only the four named or numeric classes are allowed.")


def build_manifest(args):
    groups = {}
    if args.groups_csv:
        with args.groups_csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key, group = row["image_id"].replace("\\", "/"), row["group_id"].strip()
                if key in groups or not group:
                    raise ValueError(f"Duplicate image_id or empty group_id in groups CSV: {key}")
                groups[key] = group
    pattern = re.compile(args.group_regex) if args.group_regex else None
    if pattern is not None and pattern.groups != 1:
        raise ValueError("--group-regex must have exactly one capture group.")
    if args.require_groups and not (groups or pattern):
        raise ValueError("Provide --groups-csv or --group-regex to verify source/specimen separation.")
    rows, hashes, group_splits = [], {}, {}
    duplicate_records = []
    for split in ("train", "val", "test"):
        root = args.data_root / split
        if not root.is_dir():
            raise FileNotFoundError(f"Missing split: {root}")
        by_label = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            label = canonical_label(folder.name, args.numeric_class_order)
            if label in by_label:
                raise ValueError(f"Multiple folders map to the same class in {split}: {folder}")
            by_label[label] = folder
        if set(by_label) != set(range(4)):
            raise ValueError(f"{split} must contain all four class folders; found {list(by_label)}.")
        for label, folder in sorted(by_label.items()):
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"Empty class: {folder}")
            for path in paths:
                image_id = path.relative_to(args.data_root).as_posix()
                group = groups.get(image_id, "")
                if groups and not group:
                    raise ValueError(f"Missing group metadata: {image_id}")
                if pattern:
                    match = pattern.search("/".join(image_id.split("/")[2:]))
                    if not match or not match.group(1):
                        raise ValueError(f"Cannot extract source group: {image_id}")
                    group = match.group(1)
                if group and group in group_splits and group_splits[group] != split:
                    raise ValueError(f"Group leakage across splits: {group}: {group_splits[group]} and {split}")
                if group:
                    group_splits[group] = split
                sha = file_digest(path)
                if sha in hashes:
                    previous = hashes[sha]
                    # A byte-identical file in another split would leak the
                    # target into validation/test. Keep the first occurrence
                    # (the scan order is train -> val -> test) and exclude the
                    # later copy from the manifest. A conflicting label is
                    # always fatal. Same-split duplicates remain indexed but
                    # are recorded as non-independent files.
                    if previous["split"] != split:
                        if previous["label"] != label:
                            raise ValueError(
                                "Exact duplicate image bytes have conflicting labels across splits: "
                                f"{previous['image_id']} ({previous['class_name']}) and {image_id} ({CLASS_NAMES[label]})."
                            )
                        if args.duplicate_policy == "error":
                            raise ValueError(
                                "Exact duplicate image bytes cross a train/val/test boundary: "
                                f"{previous['image_id']} ({previous['split']}) and {image_id} ({split})."
                            )
                        action = "retained_cross_split_duplicate_leakage" if args.duplicate_policy == "keep_all" else "dropped_cross_split_duplicate"
                        duplicate_records.append({
                            "image_id": image_id,
                            "duplicate_of": previous["image_id"],
                            "kept_split": previous["split"],
                            "split": split,
                            "class_name": CLASS_NAMES[label],
                            "sha256": sha,
                            "action": action,
                        })
                        if args.duplicate_policy != "keep_all":
                            continue
                    else:
                        if previous["label"] != label:
                            raise ValueError(
                                "Exact duplicate image bytes have conflicting labels in the same split: "
                                f"{previous['image_id']} ({previous['class_name']}) and {image_id} ({CLASS_NAMES[label]})."
                            )
                        duplicate_records.append({
                            "image_id": image_id,
                            "duplicate_of": previous["image_id"],
                            "split": split,
                            "class_name": CLASS_NAMES[label],
                            "sha256": sha,
                            "action": "retained_same_split_duplicate",
                        })
                else:
                    hashes[sha] = {"image_id": image_id, "split": split,
                                   "label": label, "class_name": CLASS_NAMES[label]}
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                rows.append(dict(image_id=image_id, path=str(path.resolve()), split=split, label=label,
                                 class_name=CLASS_NAMES[label], group_id=group, sha256=sha, width=width, height=height))
    for split in ("train", "val", "test"):
        present = {r["label"] for r in rows if r["split"] == split}
        if present != set(range(4)):
            dropped = [r for r in duplicate_records if r.get("split") == split and r.get("action") == "dropped_cross_split_duplicate"]
            raise ValueError(
                f"After duplicate removal, {split} does not contain all four classes. "
                f"Dropped duplicates: {dropped}. Restore an independent image for the missing class."
            )
    if groups and set(groups) != {r["image_id"] for r in rows}:
        raise ValueError("Group CSV contains unknown images; use metadata for exactly this dataset.")
    stable = [{k: row[k] for k in MANIFEST_FIELDS if k != "path"} for row in rows]
    stable += [{"duplicate_record": record} for record in duplicate_records]
    dropped_count = sum(r.get("action") == "dropped_cross_split_duplicate" for r in duplicate_records)
    leaked_count = sum(r.get("action") == "retained_cross_split_duplicate_leakage" for r in duplicate_records)
    retained_count = sum(r.get("action") == "retained_same_split_duplicate" for r in duplicate_records)
    audit = {"dataset_digest": digest(stable), "class_names": CLASS_NAMES,
             "counts": {s: {name: sum(r["split"] == s and r["class_name"] == name for r in rows)
                             for name in CLASS_NAMES} for s in ("train", "val", "test")},
             "group_separation_verified": bool(groups or pattern), "unique_groups": len(group_splits),
             "exact_duplicate_check": "cross-split duplicates follow duplicate_policy; conflicting labels are rejected; same-split same-class duplicates are retained and reported",
             "duplicate_policy": args.duplicate_policy,
             "duplicate_records": duplicate_records,
             "duplicate_count": len(duplicate_records),
             "cross_split_leakage_detected": bool(leaked_count),
             "numeric_class_order": args.numeric_class_order}
    if dropped_count:
        print(f"WARNING: dropped {dropped_count} cross-split exact duplicate file(s) from later splits "
              "to prevent validation/test leakage; see data_audit.json.", flush=True)
    if leaked_count:
        print(f"WARNING: retained {leaked_count} cross-split exact duplicate file(s). "
              "Validation/test metrics are leakage-contaminated and must not be reported as independent accuracy.", flush=True)
    if retained_count:
        print(f"WARNING: retained {retained_count} exact duplicate file(s) within the same split/class; "
              "they are not independent images and are listed in data_audit.json.", flush=True)
    if not audit["group_separation_verified"]:
        print("WARNING: source/specimen groups were not supplied. Exact duplicates are checked, but same-source crop leakage is unverified.", flush=True)
    return rows, audit


def select_subset(rows, budget, seed):
    train = [r for r in rows if r["split"] == "train"]
    if budget == "full":
        return train
    raise ValueError("This comparison protocol evaluates the complete training split only; budget must be 'full'.")


def classification_metrics(labels, predictions, probabilities=None):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1 or not labels.size:
        raise ValueError("Nonempty, aligned 1-D labels and predictions are required.")
    if not np.isin(labels, range(4)).all() or not np.isin(predictions, range(4)).all():
        raise ValueError("Predictions and labels must use the canonical four classes.")
    if set(labels.tolist()) != set(range(4)):
        raise ValueError("Evaluation must include real samples of all four classes.")
    cm = confusion_matrix(labels, predictions, labels=list(range(4)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=list(range(4)), zero_division=0)
    metrics = {"accuracy": float(np.mean(labels == predictions)), "balanced_accuracy": float(recall.mean()),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
               "n_test": int(labels.size)}
    per_class = [dict(class_id=i, class_name=name, precision=float(precision[i]), recall=float(recall[i]),
                      f1=float(f1[i]), support=int(support[i])) for i, name in enumerate(CLASS_NAMES)]
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape != (len(labels), 4) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError("Expected finite four-class probabilities summing to one.")
        metrics["macro_auroc_ovr"] = float(roc_auc_score(labels, probs, labels=list(range(4)), average="macro", multi_class="ovr"))
        metrics["nll"] = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
        metrics["brier_multiclass"] = float(((probs - np.eye(4)[labels]) ** 2).sum(1).mean())
    return metrics, per_class, cm


def load_torch_stack():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        import timm
        from torchvision import transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torch/torchvision/timm installation is required for training. "
                           "Use --mode validate to inspect data without those imports. " + str(exc)) from exc
    return torch, timm, transforms


def seed_everything(torch, seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class ImageRows:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def build_transform(args, spec, mean, std, transforms, train=False):
    size = args.input_size or spec[2]
    steps = [transforms.Resize(args.field_size, interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.CenterCrop(args.field_size)]
    if train and args.augmentation == "mild":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomApply([transforms.RandomRotation((90, 90))], p=.5)]
    elif train and args.augmentation == "paper":
        steps += [transforms.RandAugment(), transforms.RandomRotation(359),
                  transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.125)]
    steps += [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
              transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def load_weights_manifest(args):
    entries = read_json(args.weights_json) if args.weights_json else {}
    resolved = {}
    for key, entry in entries.items():
        if key not in MODEL_SPECS or not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError("Weights JSON must map model keys to objects with path and optional source fields.")
        path = Path(entry["path"]).expanduser()
        if not path.is_absolute():
            path = args.weights_json.resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[key] = {"path": str(path.resolve()), "sha256": file_digest(path),
                         "source": entry.get("source", "user-supplied; verify pretraining provenance")}
    return resolved


def create_model(args, key, weights, timm):
    spec = MODEL_SPECS[key]
    kwargs = {}
    if args.input_size and key in ("vit_base", "swin_tiny"):
        divisor = 16 if key == "vit_base" else 32
        if args.input_size % divisor:
            raise ValueError(f"{key} requires --input-size divisible by {divisor}.")
        kwargs["img_size"] = args.input_size
    local = weights.get(key)
    if local:
        kwargs["pretrained_cfg_overlay"] = {"file": local["path"], "url": "", "hf_hub_id": ""}
    try:
        # Load the original 1000-class backbone first, then replace its head.
        model = timm.create_model(spec[0], pretrained=not args.scratch, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Cannot initialize {spec[0]}. Supply its pretrained weights/cache or --weights-json. "
                           "Random initialization is available only via explicit --scratch. " + str(exc)) from exc
    cfg = dict(model.pretrained_cfg)
    model.reset_classifier(4)
    metadata = {"timm_name": spec[0], "display_name": spec[1], "pretrained": not args.scratch,
                "pretrained_source": local or {k: cfg.get(k) for k in ("url", "hf_hub_id", "tag", "architecture")},
                "mean": list(cfg["mean"]), "std": list(cfg["std"]), "input_size": args.input_size or spec[2],
                "parameters": sum(p.numel() for p in model.parameters())}
    return model, metadata


def loaders_for_run(args, torch, transforms, spec, metadata, train_rows, val_rows, test_rows, seed):
    def loader(rows, train):
        generator = torch.Generator().manual_seed(seed)
        sampler = None
        if train and args.sampling == "balanced":
            counts = np.bincount([r["label"] for r in rows], minlength=4)
            weights = [1. / counts[r["label"]] for r in rows]
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(rows), replacement=True,
                                                            generator=torch.Generator().manual_seed(seed))
        transform = build_transform(args, spec, metadata["mean"], metadata["std"], transforms, train)
        return torch.utils.data.DataLoader(ImageRows(rows, transform), batch_size=args.batch_size,
                shuffle=train and sampler is None, sampler=sampler, num_workers=args.workers,
                pin_memory=args.device.startswith("cuda"), worker_init_fn=seed_worker, generator=generator)
    return loader(train_rows, True), loader(val_rows, False), loader(test_rows, False)


def autocast_context(args, torch):
    if args.amp == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(model, loader, args, torch):
    model.eval()
    logits, labels, indices = [], [], []
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for images, target, index in loader:
            with autocast_context(args, torch):
                outputs = model(images.to(args.device, non_blocking=True))
            if not torch.isfinite(outputs).all():
                raise ValueError("Nonfinite evaluation logits; refusing a partial/invalid test result.")
            logits.append(outputs.float().cpu())
            labels.extend(target.tolist())
            indices.extend(index.tolist())
    elapsed = time.perf_counter() - start
    if indices != list(range(len(loader.dataset))):
        raise ValueError("Evaluation did not cover each image exactly once in manifest order.")
    tensor = torch.cat(logits)
    probabilities = tensor.softmax(1).numpy()
    predictions = probabilities.argmax(1)
    metrics, per_class, cm = classification_metrics(labels, predictions, probabilities)
    metrics["evaluation_seconds_including_loader"] = elapsed
    return metrics, per_class, cm, np.array(labels), predictions, probabilities, tensor.numpy()


def train_epoch(model, loader, optimizer, scaler, args, torch):
    model.train()
    total_loss, seen = 0., 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels, _) in enumerate(loader):
        images, labels = images.to(args.device), labels.to(args.device)
        lam, shuffled = 1., labels
        if args.mixup_alpha:
            lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            order = torch.randperm(len(labels), device=args.device)
            images = lam * images + (1. - lam) * images[order]
            shuffled = labels[order]
        with autocast_context(args, torch):
            outputs = model(images)
            loss = lam * torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=args.label_smoothing)
            loss += (1. - lam) * torch.nn.functional.cross_entropy(outputs, shuffled, label_smoothing=args.label_smoothing)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; training stopped without selecting invalid weights.")
        # Weight microbatches by sample count, including a short final batch.
        start_batch = (step // args.grad_accum) * args.grad_accum
        end_batch = min(start_batch + args.grad_accum, len(loader))
        window_samples = min(end_batch * args.batch_size, len(loader.dataset)) - start_batch * args.batch_size
        scaler.scale(loss * len(labels) / window_samples).backward()
        if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
    return total_loss / seen


def torch_load(torch, path):
    return torch.load(path, map_location="cpu", weights_only=True)


def set_epoch_lr(optimizer, args, base_lr, epoch):
    if epoch < args.warmup_epochs:
        scale = (epoch + 1) / args.warmup_epochs
    elif args.schedule == "cosine":
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs - 1)
        scale = .5 * (1 + math.cos(math.pi * progress))
    else:
        scale = 1.
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def export_evaluation(directory, result, test_rows, metadata, args):
    metrics, per_class, cm, labels, predictions, probabilities, logits = result
    write_json(directory / "metrics.json", dict(metrics, **metadata))
    write_json(directory / "class_names.json", CLASS_NAMES)
    write_csv(directory / "per_class_metrics.csv", per_class)
    write_csv(directory / "confusion_matrix.csv", [dict(true_class=name, **{c: int(cm[i, j]) for j, c in enumerate(CLASS_NAMES)})
              for i, name in enumerate(CLASS_NAMES)])
    np.save(directory / "labels.npy", labels)
    np.save(directory / "predictions.npy", predictions)
    np.save(directory / "probabilities.npy", probabilities)
    records = []
    for i, row in enumerate(test_rows):
        record = {"image_id": row["image_id"], "path": row["path"], "group_id": row["group_id"],
                  "true_label": int(labels[i]), "pred_label": int(predictions[i]),
                  "true_class": CLASS_NAMES[labels[i]], "pred_class": CLASS_NAMES[predictions[i]],
                  "confidence": float(probabilities[i].max())}
        record.update({"prob_" + name: float(probabilities[i, j]) for j, name in enumerate(CLASS_NAMES)})
        records.append(record)
    write_csv(directory / "test_predictions.csv", records)
    if not args.no_plots:
        plot_confusion(cm, metadata["display_name"], directory / "confusion_matrix")


def run_one(args, key, budget, seed, rows, audit, weights, stack, protocol_id):
    torch, timm, transforms = stack
    directory = args.output_dir / ("scratch" if args.scratch else "pretrained") / key / f"budget_{budget}" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    if status_path.exists():
        status = read_json(status_path)
        if status["protocol_id"] != protocol_id:
            raise ValueError(f"Protocol changed for existing run {directory}; choose a new --output-dir.")
        if status["state"] == "completed":
            if not all((directory / name).is_file() for name in ("metrics.json", "test_predictions.csv", "best_model.pt")):
                raise ValueError(f"Completed run has missing outputs: {directory}")
            print(f"Reusing completed run: {directory}", flush=True)
            return
        print(f"Restarting interrupted run from initialization: {directory}", flush=True)
    write_json(status_path, {"state": "running", "protocol_id": protocol_id})
    train_rows = select_subset(rows, budget, seed)
    val_rows, test_rows = [[r for r in rows if r["split"] == split] for split in ("val", "test")]
    history, selected = [], None
    best_score = (-float("inf"), -float("inf"))
    run_start = time.perf_counter()
    try:
        for lr in args.lrs:
            seed_everything(torch, seed, not args.allow_nondeterministic)
            model, metadata = create_model(args, key, weights, timm)
            model.to(args.device)
            train_loader, val_loader, test_loader = loaders_for_run(args, torch, transforms, MODEL_SPECS[key], metadata,
                                                                   train_rows, val_rows, test_rows, seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
            for epoch in range(args.epochs):
                set_epoch_lr(optimizer, args, lr, epoch)
                train_loss = train_epoch(model, train_loader, optimizer, scaler, args, torch)
                validation = evaluate(model, val_loader, args, torch)[0]
                score = (validation["accuracy"], -validation["nll"])
                history.append(dict(trial_lr=lr, epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
                                    train_loss=train_loss, val_accuracy=validation["accuracy"],
                                    val_macro_f1=validation["macro_f1"], val_nll=validation["nll"]))
                write_csv(directory / "history.csv", history)
                print(f"{key} budget={budget} seed={seed} lr={lr:g} epoch={epoch+1}/{args.epochs} "
                      f"loss={train_loss:.4f} val_acc={validation['accuracy']:.4f}", flush=True)
                if score > best_score:
                    best_score = score
                    selected = dict(metadata, model_key=key, seed=seed, budget=budget, n_train=len(train_rows),
                                    best_epoch=epoch + 1, best_lr=lr, best_val_accuracy=validation["accuracy"],
                                    best_val_nll=validation["nll"], dataset_digest=audit["dataset_digest"],
                                    protocol_id=protocol_id, selection_metric="validation accuracy, then lower validation NLL",
                                    class_names=CLASS_NAMES, group_separation_verified=audit["group_separation_verified"],
                                    training_subset_digest=digest([r["image_id"] for r in train_rows]))
                    checkpoint = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                  "metadata": selected}
                    temporary = directory / "best_model.pt.tmp"
                    torch.save(checkpoint, temporary)
                    temporary.replace(directory / "best_model.pt")
                    del checkpoint
            del optimizer, scaler, model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        # Test only once, after all epoch/LR selection is finished on validation.
        seed_everything(torch, seed, not args.allow_nondeterministic)
        model, _ = create_model(args, key, weights, timm)
        checkpoint = torch_load(torch, directory / "best_model.pt")
        if checkpoint["metadata"] != selected or selected["class_names"] != CLASS_NAMES:
            raise ValueError("Selected checkpoint metadata does not match this run.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        model.to(args.device)
        result = evaluate(model, test_loader, args, torch)
        selected["total_run_seconds_including_selection"] = time.perf_counter() - run_start
        export_evaluation(directory, result, test_rows, selected, args)
        write_json(status_path, {"state": "completed", "protocol_id": protocol_id})
        del model
    except Exception as exc:
        write_json(status_path, {"state": "failed", "protocol_id": protocol_id, "error": str(exc)})
        raise
    finally:
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def import_ours(args, rows, audit):
    test = {r["image_id"]: r for r in rows if r["split"] == "test"}
    suffixes = {key[len("test/"):]: key for key in test}
    seen_seeds = set()
    for item in args.ours_run:
        seed_text, source_text = item.split("=", 1)
        seed = int(seed_text)
        if seed in seen_seeds:
            raise ValueError(f"Duplicate Ours seed: {seed}")
        seen_seeds.add(seed)
        source = Path(source_text).expanduser()
        if (source / "test_results").is_dir():
            source = source / "test_results"
        names = read_json(source / "class_names.json")
        if len(names) != 4 or set(names) != set(CLASS_NAMES):
            raise ValueError("Ours result class names must match the same four classes.")
        protocol = read_json(source / "evaluation_protocol.json")
        if protocol.get("split") != "test" or protocol.get("selection_metric") != "validation_accuracy" or not protocol.get("checkpoint"):
            raise ValueError("Ours results must document a selected best-validation test checkpoint.")
        mapping = [CLASS_NAMES.index(name) for name in names]
        predictions = {}
        with (source / "predictions_epoch0.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                path_text = row["path"].replace("\\", "/")
                matches = [key for suffix, key in suffixes.items()
                           if path_text.endswith("/test/" + suffix) or path_text == "test/" + suffix]
                if len(matches) != 1:
                    raise ValueError(f"Cannot uniquely align Ours image to test manifest: {row['path']}")
                key = matches[0]
                raw_true, raw_pred = int(row["true_label"]), int(row["pred_label"])
                if raw_true not in range(4) or raw_pred not in range(4):
                    raise ValueError("Ours contains labels outside four-class range.")
                true, pred = mapping[raw_true], mapping[raw_pred]
                if key in predictions or true != test[key]["label"]:
                    raise ValueError(f"Duplicate or conflicting Ours test record: {key}")
                predictions[key] = pred
        if set(predictions) != set(test):
            raise ValueError("Ours results do not cover exactly the same test images.")
        keys = sorted(test)
        metrics, per_class, cm = classification_metrics([test[k]["label"] for k in keys], [predictions[k] for k in keys])
        metadata = dict(model_key="ours", display_name="Ours", budget="full", seed=seed,
                        dataset_digest=audit["dataset_digest"], source_directory=str(source.resolve()),
                        selection_metric="validation_accuracy", evaluation_protocol=protocol,
                        comparability="test paths and labels verified; image bytes, training subset and claimed training seed not independently verified")
        directory = args.output_dir / "external/ours/budget_full" / f"seed_{seed}"
        write_json(directory / "metrics.json", dict(metrics, **metadata))
        write_csv(directory / "per_class_metrics.csv", per_class)
        write_csv(directory / "test_predictions.csv", [{"image_id": key, "true_label": test[key]["label"],
                   "pred_label": predictions[key]} for key in keys])
        write_csv(directory / "confusion_matrix.csv", [dict(true_class=name, **{c: int(cm[i, j]) for j, c in enumerate(CLASS_NAMES)})
                  for i, name in enumerate(CLASS_NAMES)])
        write_json(directory / "status.json", {"state": "completed", "source": "external"})
        if not args.no_plots:
            plot_confusion(cm, "Ours (imported)", directory / "confusion_matrix")
        print(f"Imported Ours seed={seed}: {len(keys)} aligned test images. Training protocol still requires author verification.", flush=True)


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.linewidth": .7, "savefig.dpi": 300})
    return plt


def save_figure(fig, path):
    plt = figure_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_confusion(cm, title, path):
    plt = figure_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.3), layout="constrained")
    fractions = cm / cm.sum(axis=1, keepdims=True)
    image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{fractions[i,j]:.1%}\n(n={cm[i,j]})", ha="center", va="center",
                    fontsize=7, color="white" if fractions[i,j] > .55 else "black")
    ax.set_xticks(range(4), CLASS_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(4), CLASS_LABELS)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax, shrink=.8, label="Row fraction (diagonal: recall)")
    save_figure(fig, path)


def plot_comparison(runs, budget, path):
    plt = figure_style()
    keys = list(dict.fromkeys(r["model_key"] for r in runs))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, max(3.0, .35 * len(keys) + 1.1)), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ["accuracy", "balanced_accuracy", "macro_f1"], ["Accuracy", "Balanced accuracy", "Macro F1"]):
        for index, key in enumerate(keys):
            values = np.array([r[metric] for r in runs if r["model_key"] == key]) * 100
            spread = float(values.std(ddof=1)) if len(values) > 1 else 0.
            ax.scatter(values, index + np.linspace(-.12, .12, len(values)), s=13, color="#707070", alpha=.7, zorder=3)
            ax.errorbar(values.mean(), index, xerr=spread, fmt="D", color="#0072B2", markersize=4, capsize=3, zorder=4)
        ax.set(xlim=(0, 100), xlabel=f"{title} (%)", ylim=(len(keys) - .5, -.5))
        ax.grid(axis="x", alpha=.18)
    labels = []
    for key in keys:
        matched = [r for r in runs if r["model_key"] == key]
        labels.append(f"{matched[0]['display_name']} (n={len(matched)})")
    axes[0].set_yticks(range(len(keys)), labels)
    fig.suptitle(f"Training budget: {budget} | points: runs; diamonds: mean; bars: SD", fontsize=9)
    save_figure(fig, path)


def plot_low_data(runs, path):
    plt = figure_style()
    keys = list(dict.fromkeys(r["model_key"] for r in runs if r["budget"] != "full"))
    if not keys:
        return
    fig, axes = plt.subplots(math.ceil(len(keys) / 4), min(4, len(keys)), squeeze=False,
                             figsize=(7.2, 2.3 * math.ceil(len(keys) / 4)), layout="constrained")
    for ax, key in zip(axes.ravel(), keys):
        subset = [r for r in runs if r["model_key"] == key and r["budget"] != "full"]
        budgets = sorted(set(int(r["budget"]) for r in subset))
        means, stds = [], []
        for n in budgets:
            vals = np.array([r["balanced_accuracy"] for r in subset if int(r["budget"]) == n]) * 100
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.)
            ax.scatter(np.full(len(vals), n), vals, s=10, color="#707070", alpha=.6)
        ax.errorbar(budgets, means, yerr=stds, color="#0072B2", marker="o", capsize=3, markersize=3)
        ax.set(title=subset[0]["display_name"], xlabel="Training images per class", ylabel="Balanced accuracy (%)", ylim=(0, 100))
        ax.set_xticks(budgets)
    for ax in list(axes.ravel())[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Shared subsets across models; points: runs; bars: SD", fontsize=9)
    save_figure(fig, path)


def summarize(args, dataset_digest):
    runs = []
    # Ignore any legacy few-shot directories that may exist from an earlier
    # version.  The current protocol reports complete-data runs only.
    for status_path in sorted(args.output_dir.glob("*/*/budget_full/seed_*/status.json")):
        status = read_json(status_path)
        if status.get("state") != "completed":
            continue
        result = read_json(status_path.parent / "metrics.json")
        if result["dataset_digest"] != dataset_digest:
            raise ValueError("Cannot aggregate results from different data manifests.")
        result["initialization"] = status_path.relative_to(args.output_dir).parts[0]
        runs.append(result)
    if not runs:
        print("No completed runs to summarize.")
        return
    fields = ["initialization", "model_key", "display_name", "budget", "seed", "n_test"] + METRICS
    write_csv(args.output_dir / "all_runs.csv", [{k: r[k] for k in fields} for r in runs], fields)
    summary = []
    groups = sorted(set((r["initialization"], r["model_key"], r["budget"]) for r in runs))
    for initialization, key, budget in groups:
        selected = [r for r in runs if (r["initialization"], r["model_key"], r["budget"]) == (initialization, key, budget)]
        entry = {"initialization": initialization, "model": selected[0]["display_name"], "budget": budget,
                 "n_runs": len(selected), "seeds": ",".join(str(r["seed"]) for r in selected)}
        for metric in METRICS:
            values = np.array([r[metric] for r in selected])
            entry[metric + "_mean"] = float(values.mean())
            entry[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(entry)
    write_csv(args.output_dir / "comparison_summary.csv", summary)
    if not args.no_plots:
        for initialization in ("pretrained", "scratch"):
            selected = [r for r in runs if r["initialization"] in (initialization, "external")]
            if not any(r["initialization"] == initialization for r in selected):
                continue
            for budget in sorted(set(r["budget"] for r in selected)):
                plot_comparison([r for r in selected if r["budget"] == budget], budget,
                                args.output_dir / f"comparison_{initialization}_{budget}")
            plot_low_data(selected, args.output_dir / f"low_data_{initialization}")
    write_json(args.output_dir / "report_notes.json", {
        "metrics": "Fractions in CSV/JSON; percentages in figures. Macro averages give each of four classes equal weight.",
        "error_bars": "Sample SD across runs (ddof=1); unavailable with one run, not a confidence interval.",
        "low_data": "Disabled by protocol. This benchmark evaluates the complete training split only; no few-shot curves are generated.",
        "inference": "Deterministic single-view argmax; no TTA, PSIM area filtering or secondary class correction.",
        "external": "Ours import checks test paths/labels and checkpoint-selection metadata only; verify training data, pretraining and seed provenance separately.",
        "statistics": "No significance claims or tests. Repeated seeds are not additional independent test specimens.",
    })


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "report":
        audit = read_json(args.output_dir / "data_audit.json")
        with (args.output_dir / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["label"] = int(row["label"])
        import_ours(args, rows, audit)
        summarize(args, audit["dataset_digest"])
        return
    print(f"Data: {args.data_root}\nResults: {args.output_dir}", flush=True)
    rows, audit = build_manifest(args)
    audit_path = args.output_dir / "data_audit.json"
    if audit_path.exists() and read_json(audit_path)["dataset_digest"] != audit["dataset_digest"]:
        raise ValueError("Dataset changed in this output directory; choose a new --output-dir.")
    write_csv(args.output_dir / "split_manifest.csv", rows, MANIFEST_FIELDS)
    write_json(audit_path, audit)
    write_json(args.output_dir / "class_names.json", CLASS_NAMES)
    for budget in args.budgets:
        for seed in args.seeds:
            chosen = select_subset(rows, budget, seed)
            write_csv(args.output_dir / "subsets" / f"budget_{budget}_seed_{seed}.csv", chosen, MANIFEST_FIELDS)
    print(json.dumps(audit["counts"], indent=2), flush=True)
    if args.mode == "validate":
        print("Data and subset checks passed; no model imports, downloads or training.")
        return
    weights = load_weights_manifest(args)
    stack = load_torch_stack()
    torch = stack[0]
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.amp != "no" and not args.device.startswith("cuda"):
        raise ValueError("Mixed precision is supported on CUDA only; use --amp no on CPU.")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16.")
    settings = {k: v for k, v in vars(args).items() if k not in (
        "project_root", "data_root", "output_dir", "weights_json", "groups_csv", "models", "seeds", "budgets", "mode", "ours_run", "no_plots")}
    settings.update(version=VERSION, dataset_digest=audit["dataset_digest"], class_names=CLASS_NAMES,
                    weights=weights, packages=package_versions(), source_sha256=file_digest(__file__))
    protocol_id = digest(settings)
    write_json(args.output_dir / "protocols" / f"{protocol_id}.json", settings)
    print(f"{len(args.models) * len(args.budgets) * len(args.seeds)} runs, {len(args.lrs)} LR trial(s) per run. "
          "Validation selects weights; test is evaluated after selection.", flush=True)
    import_ours(args, rows, audit)
    for budget in args.budgets:
        for seed in args.seeds:
            for key in args.models:
                run_one(args, key, budget, seed, rows, audit, weights, stack, protocol_id)
                summarize(args, audit["dataset_digest"])
    print("Classification benchmark completed.")


if __name__ == "__main__":
    main()
