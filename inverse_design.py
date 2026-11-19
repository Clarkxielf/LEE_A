
# inverse_design.py
# -*- coding: utf-8 -*-
"""Native large-field generation with optional existing-image PSIM verification.

This version accepts class-ratio and physical-condition inputs, generates the
requested whole-field images, and writes generation manifests. Post-generation
Use --resume-analysis [OUTPUT_ROOT] to measure saved images without generation.
Default invocation verifies existing images. Use --series or --generate to generate images.
"""
import os
import math
import csv
import json
import random
import argparse
import hashlib
import copy
import sys
import subprocess
import importlib.util
from pathlib import Path
from dataclasses import fields
from functools import lru_cache
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

FORCE_CPU_MODE = os.environ.get("LEE_FORCE_CPU", "0") == "1"
if FORCE_CPU_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Default: two full UNet replicas; set LEE_INVERSE_GPU_COUNT=1 for the original path.
INFERENCE_GPU_COUNT = int(os.environ.get("LEE_INVERSE_GPU_COUNT", "2"))
if INFERENCE_GPU_COUNT not in (1, 2):
    raise ValueError("LEE_INVERSE_GPU_COUNT must be 1 or 2.")


def _preflight_cuda_uuids():
    """Validate isolated devices AND the final joint mapping before torch import.

    Existing CUDA_VISIBLE_DEVICES restrictions are respected. An explicit
    LEE_INVERSE_DEVICE selects the primary within that mapping; another allowed
    device may serve as the secondary. No visibility is changed after import.
    """
    requested = os.environ.get("LEE_INVERSE_DEVICE", "auto").lower()
    if FORCE_CPU_MODE or requested == "cpu":
        return
    if os.environ.get("LEE_INVERSE_SKIP_GPU_PREFLIGHT", "0") == "1":
        return
    if "torch" in sys.modules:
        raise RuntimeError("CUDA preflight must run before PyTorch import; start a fresh process.")
    explicit_index = None
    if requested == "cuda":
        explicit_index = 0
    elif requested.startswith("cuda:"):
        try:
            explicit_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("LEE_INVERSE_DEVICE must be auto, cpu or cuda:N.") from exc
        if explicit_index < 0:
            raise ValueError("CUDA index must be nonnegative.")
    elif requested != "auto":
        raise ValueError("LEE_INVERSE_DEVICE must be auto, cpu or cuda:N.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed_tokens = None
    primary_token = None
    if visible is not None:
        if visible.strip() in ("", "-1"):
            print("[GPU startup] Preserving explicit CPU-only CUDA visibility.", flush=True)
            return
        allowed_tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if any(token.startswith("MIG-") for token in allowed_tokens):
            print("[GPU startup] Preserving explicit MIG mapping; runtime will select visible devices.", flush=True)
            return
        if explicit_index is not None:
            if explicit_index >= len(allowed_tokens):
                raise ValueError("LEE_INVERSE_DEVICE is outside CUDA_VISIBLE_DEVICES.")
            primary_token = allowed_tokens[explicit_index]
        print(f"[GPU startup] Restricting probes to CUDA_VISIBLE_DEVICES={visible!r}.", flush=True)
    elif explicit_index is not None:
        primary_token = str(explicit_index)

    try:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot run nvidia-smi before CUDA startup. Check the driver or explicitly "
                           "configure known-good CUDA_VISIBLE_DEVICES with LEE_INVERSE_SKIP_GPU_PREFLIGHT=1.") from exc
    if inventory.returncode:
        raise RuntimeError("nvidia-smi failed: " + inventory.stderr.strip())

    def matches(token, index, uuid):
        return token == str(index) or uuid.startswith(token)

    minimum = float(os.environ.get("LEE_MIN_GPU_FREE_GIB", "12")) * 1024
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("LEE_MIN_GPU_FREE_GIB must be finite and nonnegative.")
    candidates = []
    for line in inventory.stdout.splitlines():
        entry = [part.strip() for part in line.split(",")]
        if len(entry) != 4 or not entry[1].startswith("GPU-"):
            continue
        try:
            index, free, total = int(entry[0]), float(entry[2]), float(entry[3])
        except ValueError:
            continue
        if not math.isfinite(free) or not math.isfinite(total) or total <= 0 or free < minimum:
            continue
        if allowed_tokens is not None and not any(matches(t, index, entry[1]) for t in allowed_tokens):
            continue
        candidates.append((free, index, entry[1]))
    candidates.sort(reverse=True)
    if primary_token is not None:
        pinned = [c for c in candidates if matches(primary_token, c[1], c[2])]
        if len(pinned) != 1:
            raise RuntimeError("The explicitly requested primary GPU is unavailable, ambiguous, or below the memory budget.")
        candidates = pinned + [c for c in candidates if c != pinned[0]]

    probe_code = (
        "import os, json, torch\n"
        "try:\n"
        "    expected=int(os.environ['LEE_CUDA_PROBE_COUNT'])\n"
        "    torch.cuda.init(); n=torch.cuda.device_count()\n"
        "    if n != expected: raise RuntimeError(f'expected {expected} visible GPUs, got {n}')\n"
        "    values=[]\n"
        "    for i in range(n):\n"
        "        torch.cuda.set_device(i)\n"
        "        x=torch.ones((8,8), device=f'cuda:{i}'); y=(x@x).sum().item()\n"
        "        if y != 512: raise RuntimeError(f'GPU {i} kernel probe returned {y}')\n"
        "        values.append(x); torch.cuda.synchronize(i)\n"
        "    if n == 2:\n"
        "        for i in range(2):\n"
        "            z=values[i].to(f'cuda:{1-i}', non_blocking=False)\n"
        "            if z.sum().item() != 64: raise RuntimeError('cross-device copy failed')\n"
        "            torch.cuda.synchronize(1-i)\n"
        "    print('LEE_CUDA_PREFLIGHT='+json.dumps({'ok':True,'count':n}))\n"
        "except Exception as exc:\n"
        "    print('LEE_CUDA_PREFLIGHT='+json.dumps({'ok':False,'error':type(exc).__name__+': '+str(exc)}))\n"
        "    raise\n")

    def probe(devices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(item[2] for item in devices)
        env["LEE_CUDA_PROBE_COUNT"] = str(len(devices))
        try:
            result = subprocess.run([sys.executable, "-c", probe_code], env=env,
                                    capture_output=True, text=True, timeout=60, check=False)
            records = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
                       if line.startswith("LEE_CUDA_PREFLIGHT=")]
            record = records[-1] if records else {"ok": False, "error": result.stderr[-1000:]}
            ok = result.returncode == 0 and record.get("ok") is True
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            ok, record = False, {"error": type(exc).__name__ + ": " + str(exc)}
        if not ok:
            print(f"[GPU probe failed] {[item[1] for item in devices]}: {record}", flush=True)
        return ok

    selected = []
    for candidate in candidates:
        if not probe([candidate]):
            if primary_token is not None and not selected:
                raise RuntimeError("The explicitly requested primary GPU failed CUDA startup.")
            continue
        if selected and not probe(selected + [candidate]):
            continue  # Try a different secondary; a healthy isolated device can still fail jointly.
        selected.append(candidate)
        if len(selected) == INFERENCE_GPU_COUNT:
            break
    if not selected:
        raise RuntimeError("No allowed GPU passed the isolated CUDA runtime probe. Check the driver and visibility mapping.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(item[2] for item in selected)
    # The selected primary was just remapped to logical zero, including explicit cuda:N requests.
    os.environ["LEE_INVERSE_DEVICE"] = "cuda:0"
    for logical, (_, physical, uuid) in enumerate(selected):
        print(f"[GPU startup] physical GPU {physical} ({uuid}) -> logical cuda:{logical}", flush=True)
    if len(selected) < INFERENCE_GPU_COUNT:
        print("[GPU startup] A second jointly usable GPU was not found; continuing on one GPU.", flush=True)
    print(f"[GPU startup] {len(selected)} runtime-tested GPU(s); startup budget does not guarantee full-canvas capacity.", flush=True)


# Existing-image verification is dispatched before generator/torch imports.
# Targets enter comparison only; they are never passed to the PSIM engine.
_V_CLASSES = ['Basket_weave', 'Discontinuous', 'Widmanstatten', 'Alpha_coarsened_Widmanstatten']
_V_REVISION = 'saved-psim-comparison-v1'


def _v_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def _v_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _v_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _v_alias(data, canonical, legacy):
    if canonical in data and legacy in data and data[canonical] != data[legacy]:
        raise ValueError(f'Conflicting fields: {canonical}, {legacy}')
    if canonical in data:
        return data[canonical]
    return data[legacy]


def _v_discover(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if root.name == 'generation_manifest.json':
            return [root]
        series = _v_read(root)
        paths = []
        for run in series.get('runs', []):
            saved = Path(run['generation_dir'])
            candidates = [saved / 'generation_manifest.json']
            candidates += [ancestor / 'reference_free_large_field' / saved.name / 'generation_manifest.json'
                           for ancestor in (root.parent, *list(root.parents)[:3])]
            found = next((p.resolve() for p in candidates if p.is_file()), None)
            if found is None:
                raise FileNotFoundError(f'Unresolved series run: {saved}')
            paths.append(found)
        if not paths:
            raise ValueError(f'No generation runs in {root}')
        return sorted(set(paths))
    paths = sorted(root.rglob('generation_manifest.json'))
    if not paths:
        raise FileNotFoundError(f'No generation_manifest.json under {root}')
    return paths


def _v_image(row, manifest):
    # Prefer the image beside its manifest after moving/copying a result tree.
    original = Path(str(row['image_path']).replace('\\', '/'))
    choices = [manifest.parent / original.name, original]
    if not original.is_absolute():
        choices.append(manifest.parent / original)
    shape = (int(row['width_px']), int(row['height_px']))
    expected = row.get('pixel_sha256')
    if not expected:
        raise ValueError('Candidate has no recorded RGB pixel hash; cannot verify provenance.')
    for path in dict.fromkeys(choices):
        if not path.is_file():
            continue
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if rgb.size == shape and digest == expected:
                return path.resolve()
    raise ValueError(f'Image missing/changed: {original}; expected size={shape} and matching RGB hash.')


def _v_measure(summary_path, archive, image, row, mpp):
    summary = _v_read(summary_path)
    if Path(summary.get('image_path', '')).resolve() != image.resolve():
        raise ValueError('PSIM summary points to a different source image.')
    physical = summary['quantitative_morphology']
    if not np.isclose(float(physical['microns_per_source_pixel']), mpp, rtol=1e-5, atol=0):
        raise ValueError('PSIM spatial calibration differs from generation metadata.')
    with np.load(archive, allow_pickle=False) as data:
        if list(data['class_names'].astype(str)) != _V_CLASSES:
            raise ValueError('PSIM archive class order mismatch.')
        labels = data['class_id_map']
        shape = (int(row['height_px']), int(row['width_px']))
        if labels.shape != shape or not np.isin(labels, [0, 1, 2, 3, 255]).all():
            raise ValueError('Invalid PSIM hard map dimensions/labels; no resizing is allowed.')
        counts = np.array([np.count_nonzero(labels == i) for i in range(4)], dtype=float)
        coverage = float(counts.sum() / labels.size)
        if not counts.sum():
            raise ValueError('No classified pixels; class fractions are unavailable.')
        fractions = counts / counts.sum()
        # Retain both denominators; no target-driven class exclusions or relabelling.
        full_fractions = counts / labels.size
    dark = float(_v_alias(physical, 'global_dark_contrast_area_fraction', 'global_beta_phase_fraction'))
    width = float(physical['global_mean_lath_width_um'])
    if not np.isfinite([dark, width]).all() or not 0 <= dark <= 1 or width <= 0:
        raise ValueError('Invalid physical measurements; no substitute algorithm is used.')
    return dict(measured_class_fractions=fractions.tolist(),
                measured_full_image_class_fractions=full_fractions.tolist(),
                classified_pixel_fraction=coverage, excluded_pixel_fraction=1-coverage,
                measured_dark_contrast_area_fraction=dark, measured_mean_lath_width_um=width,
                class_denominator='pixels with PSIM hard labels 0..3; excluded pixels reported separately',
                morphology_method='PSIM quantitative_morphology summary, unchanged')


def _v_compare(row, measured, args):
    target = np.asarray(row['target_class_fractions'], dtype=float)
    dark = float(_v_alias(row, 'target_dark_contrast_area_fraction', 'target_beta_fraction'))
    width = float(row['target_mean_lath_width_um'])
    if (target.shape != (4,) or not np.isfinite(target).all() or (target < 0).any()
            or not np.isclose(target.sum(), 1, atol=1e-5) or not np.isfinite([dark, width]).all()
            or not 0 <= dark <= 1 or width <= 0):
        raise ValueError('Invalid saved target conditions.')
    error = (np.asarray(measured['measured_class_fractions']) - target) * 100
    dark_error = (measured['measured_dark_contrast_area_fraction'] - dark) * 100
    width_error = measured['measured_mean_lath_width_um'] - width
    return dict(**measured, candidate_name=row['candidate_name'], seed=row.get('seed'),
                target_class_fractions=target.tolist(), target_dark_contrast_area_fraction=dark,
                target_mean_lath_width_um=width, class_signed_error_pp=error.tolist(),
                dark_signed_error_pp=dark_error, width_signed_error_um=width_error,
                within_tolerances=bool(np.all(abs(error) <= args.class_tolerance_pp)
                    and abs(dark_error) <= args.dark_tolerance_pp and abs(width_error) <= args.width_tolerance_um
                    and measured['excluded_pixel_fraction'] <= args.max_excluded_fraction))


def _v_plot(items, directory, pending=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = ['Basket-weave', 'Discontinuous', 'Widmanstätten', 'α-coarsened Widmanstätten',
              'Dark-contrast area fraction']
    colors = ['#0072B2', '#777777', '#D55E00', '#AA4499', '#228833']
    markers = ['o', '^', 's', 'D', 'P']
    rows = []
    for item in items:
        targets = [100*x for x in item['target_class_fractions']] + [100*item['target_dark_contrast_area_fraction'], item['target_mean_lath_width_um']]
        values = [100*x for x in item['measured_class_fractions']] + [100*item['measured_dark_contrast_area_fraction'], item['measured_mean_lath_width_um']]
        for i, (t, v) in enumerate(zip(targets, values)):
            rows.append(dict(image_path=item['image_path'], candidate_name=item['candidate_name'], seed=item.get('seed'),
                feature=(_V_CLASSES + ['dark_contrast_area_fraction', 'mean_lath_width_um'])[i],
                target=t, measured=v, signed_error=v-t, absolute_error=abs(v-t),
                value_unit='um' if i==5 else '%', error_unit='um' if i==5 else 'percentage_points',
                classified_pixel_fraction=item['classified_pixel_fraction']))
    if not rows:
        # Do not leave an old success figure in place when all current inputs fail.
        for name in ('verification_values.csv','target_vs_measured_validation.png',
                     'target_vs_measured_validation.pdf','target_vs_measured_validation.svg'):
            stale=directory/name
            if stale.is_file():os.replace(stale,stale.with_name(stale.name+'.stale'))
        return
    with (directory/'verification_values.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with plt.rc_context({'font.family':'sans-serif', 'font.size':8, 'svg.fonttype':'none', 'pdf.fonttype':42}):
        fig = plt.figure(figsize=(7.2, 4.5), facecolor='white')
        ax = fig.add_axes([.10,.19,.49,.65])
        small = fig.add_axes([.72,.57,.23,.27])
        legend = fig.add_axes([.65,.17,.34,.33]); legend.axis('off')
        handles = []
        for i in range(5):
            points = [r for r in rows if r['feature']==(_V_CLASSES+['dark_contrast_area_fraction'])[i]]
            handles.append(ax.scatter([r['target'] for r in points], [r['measured'] for r in points],
                marker=markers[i], s=30, facecolors='none', edgecolors=colors[i], linewidths=.9, label=labels[i], clip_on=False))
        ax.plot([0,100],[0,100], '--', color='.7', lw=.8, zorder=0)
        ax.set(xlim=(-3,103), ylim=(-3,103), xlabel='Target (%)', ylabel='PSIM estimated / measured (%)')
        ax.set_xticks([0,25,50,75,100]); ax.set_yticks([0,25,50,75,100])
        widths = [r for r in rows if r['feature']=='mean_lath_width_um']
        lo=min(min(r['target'],r['measured']) for r in widths); hi=max(max(r['target'],r['measured']) for r in widths)
        pad=max((hi-lo)*.15, .1); lo=max(0,lo-pad); hi+=pad
        small.plot([lo,hi],[lo,hi], '--', color='.7', lw=.8)
        small.scatter([r['target'] for r in widths],[r['measured'] for r in widths],s=23,facecolors='none',edgecolors='.4')
        small.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Target (µm)',ylabel='PSIM measured (µm)',title='Mean α-lath width')
        for axis in (ax,small): axis.spines[['top','right']].set_visible(False)
        legend.legend(handles, labels, loc='upper left', frameon=False, fontsize=7.4, labelspacing=1.05, handletextpad=.4)
        fig.suptitle('Target vs estimated morphology',fontsize=12,y=.96)
        fig.text(.1,.06,f'{len(items)} generated image(s); {pending} pending. Each point represents one image and descriptor.\n'
                 'Class fractions: classified-pixel area; overlapping points are not jittered.',fontsize=7)
        for ext in ('png','pdf','svg'):
            fig.savefig(directory/f'target_vs_measured_validation.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)


def _v_assets(lia, script):
    checkpoint = Path(lia.CHECKPOINT_DIR).expanduser().resolve()
    required = [checkpoint / (name+'_final.pth') for name in ('unet','ace','mta','dit','pqa','unified_head')]
    required.append(Path(lia.FP_STATS_PATH).expanduser().resolve())
    for path in required:
        if not path.is_file(): raise FileNotFoundError(f'PSIM model/statistics not found: {path}')
    paths=set(required)
    paths.update(p for p in checkpoint.glob('*.pth') if p.is_file())
    paths.update(p for p in (checkpoint/'test_results').glob('*.npy') if p.is_file())
    calibration=Path(getattr(lia,'DEFAULT_PAIR_CALIBRATOR_PATH',checkpoint/'pair_calibrator.json'))
    if calibration.is_file(): paths.add(calibration.resolve())
    # Fingerprint local backbone and sibling Python dependencies conservatively.
    base=Path(lia.PRETRAINED_MODEL).expanduser().resolve()
    if not base.is_dir():
        raise ValueError('Verification requires a local PSIM backbone for reproducible cache identity; use --psim-pretrained-model.')
    paths.update(p for p in base.rglob('*') if p.is_file())
    paths.update(script.parent.glob('*.py'))
    paths.add(script)
    print(f'[PSIM identity] Hashing {len(paths)} model/config/code files; no inference yet.',flush=True)
    return {str(p.resolve()):_v_hash(p) for p in sorted(paths)}


def _v_configure_paths(lia, args, project):
    """Resolve complete PSIM bundles, never the generation-only full-UNet file."""
    project=Path(project).expanduser().resolve()
    names=[name+'_final.pth' for name in ('unet','ace','mta','dit','pqa','unified_head')]
    configured=Path(args.psim_checkpoint_dir or lia.CHECKPOINT_DIR).expanduser().resolve()
    complete=lambda directory:all((directory/name).is_file() for name in names)
    selected=configured
    if not complete(configured):
        if args.psim_checkpoint_dir:
            missing=[name for name in names if not (configured/name).is_file()]
            raise FileNotFoundError(f'Explicit PSIM checkpoint directory is incomplete: {configured}; missing {missing}. '
                                    'Use --psim-checkpoint-dir with the trained PSIM bundle, not large_field_model.pt.')
        candidates=[project/'Diffusion'/'Deep-TiDiffusion-Ablation'/'full'/'seed_42',
                    project/'Diffusion'/'Deep-TiDiffusion-Outputs-FourClass-v3']
        matches=list(dict.fromkeys(p.resolve() for p in candidates if complete(p)))
        if len(matches)!=1:
            raise FileNotFoundError(f'PSIM default directory is incomplete: {configured}. '
                f'Complete local candidates: {[str(p) for p in matches]}. '
                f'Checked: {[str(p) for p in candidates]}. '
                'Specify --psim-checkpoint-dir explicitly; six *_final.pth PSIM weights are required.')
        selected=matches[0]
        print(f'[PSIM paths] Default incomplete; found complete PSIM bundle: {selected}',flush=True)
    lia.CHECKPOINT_DIR=str(selected)
    if not os.environ.get('PSIM_PAIR_CALIBRATOR'):
        lia.DEFAULT_PAIR_CALIBRATOR_PATH=str(selected/'pair_calibrator.json')
    explicit=args.psim_fp_stats_path
    statistics=([Path(explicit).expanduser().resolve()] if explicit else [
        selected/'fingerprint_stats_2d_four_class.npz',
        selected/'fp_stats_2d_beta_lath_four_class_v3.npy',
        Path(lia.FP_STATS_PATH).expanduser().resolve()])
    source=next((p for p in statistics if p.is_file()),None)
    if source is None:
        raise FileNotFoundError('PSIM fingerprint statistics missing. Use --psim-fp-stats-path with the training statistics. '
                                f'Checked: {[str(p) for p in statistics]}')
    if source.suffix.lower()=='.npz':
        with np.load(source,allow_pickle=False) as data:
            mean,std=np.asarray(data['mean']),np.asarray(data['std'])
    else:
        mean,std=np.load(source,allow_pickle=False)
    mean,std=np.asarray(mean).reshape(-1),np.asarray(std).reshape(-1)
    if mean.shape!=(2,) or std.shape!=(2,) or not np.isfinite([mean,std]).all() or (std<0).any():
        raise ValueError(f'Invalid two-descriptor PSIM training statistics: {source}')
    if source.suffix.lower()=='.npz':
        root=Path(args.root).expanduser().resolve()
        parent=root.parent if root.is_file() else root
        adapter=parent/'psim_validation'/'_model_metadata'/('fp_stats_'+_v_hash(source)[:20]+'.npy')
        adapter.parent.mkdir(parents=True,exist_ok=True)
        with adapter.with_suffix('.partial').open('wb') as stream:
            np.save(stream,np.stack([mean,std]),allow_pickle=False)
        os.replace(adapter.with_suffix('.partial'),adapter)
        lia.FP_STATS_PATH=str(adapter)
    else:lia.FP_STATS_PATH=str(source)
    lia.VERIFICATION_FP_STATS_SOURCE=str(source)
    if source.parent!=selected and not explicit:
        print(f'[PSIM statistics] Using configured legacy training statistics: {source}; training identity is not established by its path.',flush=True)
    else:print(f'[PSIM statistics] Training statistics: {source}',flush=True)
    if args.psim_pretrained_model:
        lia.PRETRAINED_MODEL=str(Path(args.psim_pretrained_model).expanduser().resolve())


def _v_load_module(args):
    # Existing GPU preflight remains before the first torch/PSIM import.
    globals()['INFERENCE_GPU_COUNT'] = 1  # sequential validation needs one model replica
    _preflight_cuda_uuids()
    import torch
    use_cuda=not FORCE_CPU_MODE and os.environ.get('LEE_INVERSE_DEVICE','auto').lower()!='cpu' and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        torch.cuda.init()
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    script = Path(args.psim_script).expanduser().resolve()
    if not script.is_file(): raise FileNotFoundError(f'PSIM script not found: {script}')
    os.environ.setdefault('PSIM_PROJECT_ROOT', os.environ.get('LEE_A_PROJECT_ROOT',str(Path(__file__).resolve().parent.parent)))
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('_saved_image_psim',script)
    lia=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lia;spec.loader.exec_module(lia)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visibility:
        raise ValueError('PSIM changed CUDA_VISIBLE_DEVICES after torch initialization; its selector must preserve the launch mapping.')
    if list(lia.CLASS_NAMES) != _V_CLASSES: raise ValueError('PSIM class order mismatch.')
    lia.DEVICE=torch.device('cuda:0' if use_cuda else 'cpu')
    _v_configure_paths(lia,args,os.environ['PSIM_PROJECT_ROOT'])
    for flag in ('SAVE_VISUALIZATIONS','SAVE_VISUALIZATION','SAVE_PLOTS','SHOW_FIGURES','SHOW_RESULTS'):
        if hasattr(lia,flag):setattr(lia,flag,False)
    config={key:value for key,value in vars(lia).items() if key.isupper()
            and isinstance(value,(str,int,float,bool,type(None))) and key not in ('OUTPUT_DIR','LARGE_IMAGE_PATH')}
    identity=dict(revision=_V_REVISION,assets=_v_assets(lia,script),config=config,
                  class_names=_V_CLASSES,measurement_seed=42)
    print(f'[PSIM] script={script}\n[PSIM] checkpoints={lia.CHECKPOINT_DIR}\n[PSIM] statistics={lia.FP_STATS_PATH}',flush=True)
    return lia,identity


def _v_valid_receipt(receipt, key):
    try:
        return receipt['key']==key and all(Path(receipt[k]).is_file() and _v_hash(receipt[k])==receipt[k+'_sha256'] for k in ('summary','archive'))
    except (KeyError,TypeError,OSError): return False


def _v_run(args):
    manifests=_v_discover(args.root)
    root=Path(args.root).expanduser().resolve()
    output=Path(args.verification_output_dir).resolve() if args.verification_output_dir else (root.parent if root.is_file() else root)/'psim_validation'
    report=dict(status='running',manifests=[str(p) for p in manifests],candidates=[],errors=[],
                tolerances=dict(class_pp=args.class_tolerance_pp,dark_pp=args.dark_tolerance_pp,width_um=args.width_tolerance_um,max_excluded_fraction=args.max_excluded_fraction))
    lia=engine=identity=None
    initialization_error=None
    print(f'[Verification] {len(manifests)} saved generation manifests; no image generation.',flush=True)
    for manifest_path in manifests:
        manifest_items=[]
        try:
            manifest=_v_read(manifest_path)
            if manifest.get('class_names')!=_V_CLASSES or manifest.get('mode')!='reference_free_physical_conditioned_generation':
                raise ValueError('Not a compatible four-class generation manifest.')
            mpp=float(manifest.get('microns_per_source_pixel') or manifest.get('verification_config',{}).get('microns_per_source_pixel') or 0)
            if not np.isfinite(mpp) or mpp<=0:raise ValueError('Missing positive source pixel calibration in manifest.')
            if not manifest.get('candidates'):raise ValueError('Manifest contains no completed candidates.')
            spec=manifest.get('run_spec',{})
            expected=len(spec.get('conditions',[]))*len(spec.get('seeds',[]))
            if expected and len(manifest['candidates']) != expected:
                report['errors'].append(dict(manifest=str(manifest_path),error=f'Generation incomplete: {len(manifest["candidates"])}/{expected} candidates saved. Only saved images will be measured.'))
        except Exception as exc:
            report['errors'].append(dict(manifest=str(manifest_path),error=str(exc)));continue
        for row in manifest['candidates']:
            try:
                image=_v_image(row,manifest_path)
                if initialization_error is not None:
                    raise RuntimeError('PSIM initialization failed earlier; not retried in this process: '+initialization_error)
                if lia is None:
                    try:lia,identity=_v_load_module(args)
                    except Exception as exc:
                        initialization_error=f'{type(exc).__name__}: {exc}'
                        report['initialization_error']=initialization_error
                        raise
                if not np.isclose(float(lia.MICRONS_PER_SOURCE_PIXEL),mpp,rtol=1e-5,atol=0):
                    raise ValueError('PSIM/source calibration mismatch; no scale is guessed.')
                key=hashlib.sha256(json.dumps(dict(image=str(image),pixels=row['pixel_sha256'],mpp=mpp,identity=identity),sort_keys=True).encode()).hexdigest()
                directory=manifest_path.parent/'psim_validation'/'cache'/key[:24]
                receipt_path=directory/'measurement_receipt.json'
                try:receipt=_v_read(receipt_path)
                except (OSError,ValueError):receipt={}
                if not args.force_verification and _v_valid_receipt(receipt,key):
                    summary_path,archive=Path(receipt['summary']),Path(receipt['archive'])
                    print(f'[PSIM reuse] {image}',flush=True)
                else:
                    if engine is None:
                        try:
                            values=lia.load_models()
                            engine=lia.PhysicsAwareSpatialInferenceModule(*values,patch_size=lia.PATCH_SIZE,stride=lia.STRIDE,sigma=lia.SIGMA,lambda_fp=lia.LAMBDA_FINGERPRINT)
                        except Exception as exc:
                            initialization_error=f'{type(exc).__name__}: {exc}'
                            report['initialization_error']=initialization_error
                            raise
                    directory.mkdir(parents=True,exist_ok=True)
                    lia.OUTPUT_DIR=str(directory)
                    # A unique attempt prevents the PSIM engine itself reusing an uncommitted export.
                    attempt=directory/('attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
                    attempt.mkdir();lia.OUTPUT_DIR=str(attempt);destination=attempt/lia.safe_name(image.stem)
                    print(f'[PSIM measure] {image}',flush=True)
                    import torch
                    random.seed(42);np.random.seed(42);torch.manual_seed(42)
                    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
                    engine.run(str(image))
                    _v_image(row,manifest_path)  # reject source changes during measurement
                    summary_path=destination/'PSIM_summary.json'
                    archives=sorted(destination.glob('PSIM_results*.npz'))
                    if len(archives)!=1:raise ValueError(f'Expected one PSIM numerical archive in {destination}; found {len(archives)}')
                    archive=archives[0]
                measured=_v_measure(summary_path,archive,image,row,mpp)
                _v_json(receipt_path,dict(key=key,summary=str(summary_path),archive=str(archive),summary_sha256=_v_hash(summary_path),archive_sha256=_v_hash(archive)))
                result=_v_compare(row,measured,args)
                result.update(image_path=str(image),manifest=str(manifest_path),summary_path=str(summary_path),archive_path=str(archive),measurement_key=key)
                _v_json(directory/'comparison.json',result)
                report['candidates'].append(result);manifest_items.append(result)
                _v_json(output/'verification_report.json',report)
                candidate_dir=manifest_path.parent/'psim_validation'/('candidate_'+hashlib.sha256(str(image).encode()).hexdigest()[:12])
                _v_plot([result],candidate_dir)
                print(f"[Verified] {image.name}; class errors (pp)={result['class_signed_error_pp']}; within tolerances={result['within_tolerances']}",flush=True)
            except Exception as exc:
                report['errors'].append(dict(manifest=str(manifest_path),candidate=row.get('candidate_name'),error=f'{type(exc).__name__}: {exc}'))
                _v_json(output/'verification_report.json',report)
                print(f'[Verification pending] {exc}',flush=True)
        if manifest_items:
            _v_json(manifest_path.parent/'psim_validation'/'verification_report.json',dict(candidates=manifest_items))
            _v_plot(manifest_items,manifest_path.parent/'psim_validation',len(manifest['candidates'])-len(manifest_items))
        else:
            _v_plot([],manifest_path.parent/'psim_validation',len(manifest['candidates']))
    report.update(status='complete' if not report['errors'] else 'partial',
                  all_within_tolerances=bool(report['candidates'] and not report['errors'] and all(r['within_tolerances'] for r in report['candidates'])))
    _v_json(output/'verification_report.json',report)
    if identity:_v_json(output/'measurement_identity.json',identity)
    _v_plot(report['candidates'],output,len(report['errors']))
    print(f"[Verification {report['status']}] {len(report['candidates'])} measured; {len(report['errors'])} pending; {output}",flush=True)
    return 0 if report['status']=='complete' else 2


def _v_main():
    global np, plt, Image
    parser=argparse.ArgumentParser(description='Measure existing generated images with trained PSIM; never generate images.')
    base=Path(os.environ.get('LEE_A_PROJECT_ROOT',str(Path(__file__).resolve().parent.parent)))
    trained=Path(os.environ.get('LEE_A_TRAINED_OUTPUT_DIR',str(base/'Diffusion'/'Deep-TiDiffusion-Outputs-FourClass-v3')))
    parser.add_argument('--resume-analysis','--verify-generated-only',dest='root',nargs='?',default=None,
                        const=str(trained/'native_large_field_regional_guidance_seed42_5440x3648'),help='Output root, generation manifest or series manifest; omitted path uses built-in output root.')
    parser.add_argument('--psim-script',default=str(Path(__file__).resolve().with_name('large_image_analysis.py')))
    parser.add_argument('--psim-checkpoint-dir')
    parser.add_argument('--psim-fp-stats-path')
    parser.add_argument('--psim-pretrained-model')
    parser.add_argument('--verification-output-dir')
    parser.add_argument('--force-verification',action='store_true',help='Explicitly recompute measurements, never images.')
    parser.add_argument('--class-tolerance-pp',type=float,default=5.)
    parser.add_argument('--dark-tolerance-pp',type=float,default=5.)
    parser.add_argument('--width-tolerance-um',type=float,default=.5)
    parser.add_argument('--max-excluded-fraction',type=float,default=0.)
    args=parser.parse_args()
    for key in ('class_tolerance_pp','dark_tolerance_pp','width_tolerance_um','max_excluded_fraction'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:parser.error(f'{key} must be finite and nonnegative')
    if args.max_excluded_fraction>1:parser.error('max-excluded-fraction must be <=1')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    try:return _v_run(args)
    except (OSError,ValueError,KeyError) as exc:
        print(f'[Verification error] {exc}',file=sys.stderr);return 2


def _dispatch_saved_image_mode(arguments):
    """Verification is the default; generation requires an explicit mode flag."""
    flags = {arg.split('=', 1)[0] for arg in arguments}
    verify = bool(flags & {'--resume-analysis', '--verify-generated-only'})
    generate = bool(flags & {'--series', '--generate'})
    if verify and generate:
        raise ValueError('Choose verification OR generation: --resume-analysis cannot be combined with --series/--generate.')
    if '--series' in flags and '--generate' in flags:
        raise ValueError('Choose either --series or --generate.')
    if generate:
        return None
    return list(arguments) if verify else ['--resume-analysis', *arguments]


if __name__ == '__main__':
    try:
        _verification_arguments = _dispatch_saved_image_mode(sys.argv[1:])
    except ValueError as error:
        print(f'[Mode error] {error}', file=sys.stderr)
        raise SystemExit(2)
    if _verification_arguments is not None:
        sys.argv[1:] = _verification_arguments
        print('[Mode] Existing-image PSIM verification; no image generation. '
              'Use --series or --generate explicitly to generate images.', flush=True)
        raise SystemExit(_v_main())

_preflight_cuda_uuids()


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = os.path.abspath(os.path.expanduser(os.environ.get(
    "LEE_A_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)))
DIFFUSION_ROOT = os.path.join(PROJECT_ROOT, "Diffusion")
TRAINED_OUTPUT_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "LEE_A_TRAINED_OUTPUT_DIR", os.path.join(DIFFUSION_ROOT, "Deep-TiDiffusion-Outputs-FourClass-v3")
)))
CHECKPOINT_FORMAT = "large-field-full-unet-v4"
CONDITION_SCHEMA_VERSION = "large-field-psim-hard-physical-full-unet-v4"
# Never silently select a legacy LoRA checkpoint.
_DEFAULT_MODEL_DIR = os.path.join(TRAINED_OUTPUT_DIR, "large_field_full_unet_1536_v4")
USE_EMA_WEIGHTS = True
VERIFY_GENERATED_WITH_PSIM = False  # Generation-only build: no post-generation verification.
SAMPLING_SCHEDULER = "training"  # Read validation_scheduler from checkpoint; ddpm/ddim override.
TRAINER_SCRIPT = os.environ.get("LEE_A_LARGE_FIELD_TRAINER", os.path.join(os.path.dirname(__file__), "train_large_field.py"))
LARGE_FIELD_MODEL_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "LEE_A_LARGE_FIELD_MODEL_DIR", _DEFAULT_MODEL_DIR
)))
LARGE_FIELD_MODEL_PATH = os.path.abspath(os.path.expanduser(os.environ.get(
    "LEE_A_LARGE_FIELD_MODEL_PATH", os.path.join(LARGE_FIELD_MODEL_DIR, "large_field_model.pt")
)))
OUTPUT_DIR = os.path.join(TRAINED_OUTPUT_DIR, "native_large_field_regional_guidance_seed42_5440x3648")
LOCAL_MODEL_SEARCH_DIRS = [
    os.path.join(DIFFUSION_ROOT, "stable-diffusion-v1-5"),
    os.path.join(DIFFUSION_ROOT, "models", "stable-diffusion-v1-5"),
    os.path.join(TRAINED_OUTPUT_DIR, "stable-diffusion-v1-5"),
]
LOCAL_FILES_ONLY = True
ALLOW_HF_DOWNLOAD = False
HF_HOME_DIR = os.environ.get("HF_HOME", "")
CLASS_NAMES = ["Basket_weave", "Discontinuous", "Widmanstatten", "Alpha_coarsened_Widmanstatten"]
N_CLASSES = len(CLASS_NAMES)
FP_NAMES = ["beta_phase_fraction", "mean_lath_width"]
# FP_NAMES retains the v4 wire contract for tensor order and cache identity.
FP_EXPORT_NAMES = ["dark_contrast_area_fraction", "mean_lath_width"]
FP_DISPLAY_LABELS = ["Dark-contrast area fraction", "Mean α-lath width"]
FP_NAME_ALIASES = {"mean_lath_width_um": "mean_lath_width",
                   "dark_contrast_area_fraction": "beta_phase_fraction"}
FP_DIM = len(FP_NAMES)
FP_EPS = 0.001
# Reserved zero channels are required by the trained tensor layout.
CONNECTIVITY_NAMES = ["connectivity_length", "fragmentation_degree", "orientation_consistency"]
CONNECTIVITY_DIM = len(CONNECTIVITY_NAMES)
SPATIAL_CONDITION_CHANNELS = N_CLASSES + FP_DIM + CONNECTIVITY_DIM + 3
SEED = 42
DEVICE = "cpu" if FORCE_CPU_MODE else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
FORCE_DISABLE_CUDNN = FORCE_CPU_MODE
REFERENCE_FREE_SAMPLES_PER_CLASS = 1
LARGE_FIELD_NUM_INFERENCE_STEPS = 500
LARGE_FIELD_GUIDANCE_START = 7.5
LARGE_FIELD_GUIDANCE_END = 7.5
RESIDUAL_DRAWS = 64
# Direct Run verifies saved images before generator imports; generation requires --series or --generate.
# Both modes reuse matching completed images; neither performs verification.
BUILTIN_RUN_MODE = "series"  # Generation-only fallback; direct run is routed to PSIM verification above.
# Four configured composition targets in CLASS_NAMES order; measured fractions are not guaranteed.
# Keep the anchor physics, seed, output directory and sampler unchanged to reuse
# matching completed images. Completed old-layout runs are reused only after strict validation.
BUILTIN_COMPOSITION_SERIES = [
    [0.00, 0.00, 0.50, 0.50],  # 80% Widmanstatten + 20% Alpha_coarsened_Widmanstatten
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
]
SERIES_SAMPLES_PER_RECIPE = 1  # Exactly one image per composition, using seed 42.
# Retained zero-valued feedback metadata keeps existing series identities.
# The feedback implementation is removed; requested class inputs are never retuned.
COMPOSITION_FEEDBACK_ROUNDS = 0  # Do not create extra adaptive attempts; exactly one draw per ratio.
COMPOSITION_FEEDBACK_GAIN = 0.5
# Experimental full-canvas prediction routing; no image/latent patch stitching.
REGIONAL_GUIDANCE_ENABLED = True
REGIONAL_MASK_SMOOTH_RADIUS = 2  # latent pixels; 16 decoded pixels at VAE factor 8
REGIONAL_GUIDANCE_REVISION = "whole_canvas_class_predictions_v1"
# Geometry changes have their own cache identity. Same layout seed across recipes
# makes paired-seed comparisons possible; it is independent of diffusion noise.
REGIONAL_LAYOUT_REVISION = "isotropic_multiscale_colonies_v1"
REGIONAL_LAYOUT_SEED = 42
REGIONAL_LAYOUT_DOMAINS = 3.0  # characteristic spatial frequency across the short side
REUSE_COMPLETED_LEGACY_LAYOUT = True  # preserve completed old images, with original targets
# Preserve requested dark-contrast area fraction / α-lath width in all branches; no implicit physical retargeting.
REGIONAL_PHYSICS_POLICY = "shared_requested_fingerprint"
BUILTIN_DIAGNOSTIC_COMPARE_SAMPLERS = False
# Used by generate/both with no explicit CLI recipe. None reuses source numeric recipes.
# Class mixture is the requested target, not a measured outcome.
# None physical values reuse the dominant class's calibrated cached PSIM representative.
# These auxiliary values are recorded, not inferred physical properties of the mixture.
# Class order: Basket_weave, Discontinuous, Widmanstatten, Alpha_coarsened_Widmanstatten.
BUILTIN_CLASS_FRACTIONS = [0.76, 0.0, 0.24, 0.0]
BUILTIN_DARK_CONTRAST_AREA_FRACTION = None  # Fraction in 0..1; None keeps the recorded default.
BUILTIN_BETA_FRACTION = None  # Legacy configuration alias; do not set conflicting values.
BUILTIN_LATH_WIDTH_UM = None
REQUIRED_NATIVE_RESOLUTION = 1536
DIAGNOSTIC_SAMPLES_PER_CLASS = 1
HEATMAP_DRAWS = 150
HEATMAP_REPEATS = 2
HEATMAP_DECODE_GAIN = 28.0  # Display gain in the supplied reference implementation.
# Publication display only; these settings do not change residuals or MSE scores.
HEATMAP_STRENGTH_CMAP = "magma"
HEATMAP_OVERLAY_COLOR = "#D81B8A"
HEATMAP_OVERLAY_OPACITY = 0.65
# Matrix source policy only; --generate/--diagnose keep measured cached recipes.
# direct-or-auto: try ALL four images without PSIM; on image-selection failure
# revert the ENTIRE matrix to original automatic cached source selection.
# direct: require the four files; fail without replacing them.
# legacy-auto: ignore specified images/crops; use original cached selection.
# cached: v24 manual selectors with verified existing physical caches.
MATRIX_SOURCE_MODE = "direct-or-auto"
# None uses checkpoint fp_mean as a common assumed condition, not an image
# measurement. Optional [dark_contrast_area_fraction, mean_lath_width_um] for all four columns.
MATRIX_REFERENCE_PHYSICS = None
# Direct mode requires four image paths in BW, DC, W, ACW order.
# Numeric filename IDs and entry:N remain available only in cached mode.
SOURCE_IMAGE_IDS = [
    os.path.join(PROJECT_ROOT, "data", "example_data", "Basket_weave", "1.jpg"),
    os.path.join(PROJECT_ROOT, "data", "example_data", "Discontinuous", "1.jpg"),
    os.path.join(PROJECT_ROOT, "data", "example_data", "Widmanstatten", "1.jpg"),
    os.path.join(PROJECT_ROOT, "data", "example_data", "Alpha_coarsened_Widmanstatten", "1.jpg"),
]
# None uses the whole image if native-size, otherwise a deterministic centre
# crop. No purity search or image resizing is performed for specified sources.
# Optional native-pixel top-left crop coordinates in the same column order.
# Example: SOURCE_CROP_XY = [(0, 0), None, (768, 512), None]
SOURCE_CROP_XY = [None, None, None, None]
# Additional existing PSIM result roots, searched recursively for PSIM_summary.json.
# The checkpoint config.psim_output_dir is also searched automatically.
SOURCE_PSIM_ROOTS = []
# Optional exact PSIM_summary.json per source column; paths remove ambiguity.
SOURCE_PSIM_SUMMARIES = [None, None, None, None]
# The checkpoint remains trained at 1536. Larger canvases are size extrapolation.
# Every UNet step and VAE decode process the entire canvas; no patch fusion.
# Dimensions must be positive multiples of 64 for the trained grid and UNet.
NATIVE_FULL_FIELD_WIDTH = 5440
NATIVE_FULL_FIELD_HEIGHT = 3648
DEFAULT_FULL_FIELD_STRIDE = 48
DEFAULT_WINDOW_BATCH_SIZE = 1
OVERLAP_WINDOW_MIN_WEIGHT = 0.10
GENERATION_CACHE_VERSION = 8
FULL_FIELD_CONDITION_POLICY = "full_unet_v4_whole_canvas_regional_prediction_v1"
SAMPLING_TIMESTEP_SPACING = "default"  # Preserve the saved schedule; never pass "default" to diffusers.
INFERENCE_DEVICE = os.environ.get("LEE_INVERSE_DEVICE", "auto")
VAE_DEVICE_POLICY = os.environ.get("LEE_VAE_DEVICE", "auto")
MIN_GPU_FREE_GIB = 12.0


def select_inference_device(requested="auto", minimum_free_gib=MIN_GPU_FREE_GIB):
    """Choose an available visible GPU without changing other processes."""
    if FORCE_CPU_MODE or requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested != "auto":
            raise RuntimeError(f"Requested {requested}, but CUDA is unavailable.")
        return "cpu"
    candidates = []
    for index in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(index)
        except RuntimeError as exc:
            print(f"[GPU] cuda:{index} memory query failed: {exc}", flush=True)
            continue
        print(f"[GPU] cuda:{index}: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
        candidates.append((int(free), index))
    if requested != "auto":
        device = torch.device(requested)
        if device.type != "cuda":
            raise ValueError("LEE_INVERSE_DEVICE must be auto, cpu or cuda:N.")
        index = torch.cuda.current_device() if device.index is None else device.index
        candidates = [entry for entry in candidates if entry[1] == index]
    if not candidates or max(candidates)[0] < minimum_free_gib * 2**30:
        raise RuntimeError(
            f"No eligible GPU has the {minimum_free_gib:g} GiB free startup budget. "
            "Other jobs must finish, or select another visible GPU. "
            "LEE_INVERSE_DEVICE=cpu explicitly permits slow CPU inference. No existing process was stopped.")
    _, selected = max(candidates)
    torch.cuda.set_device(selected)
    print(f"[GPU] Selected cuda:{selected}; startup budget is not a full-generation memory guarantee.", flush=True)
    return f"cuda:{selected}"


def estimate_vae_workspace_bytes(operation, tensor_shape, element_bytes=4):
    """Conservative activation estimate, not an exact allocator prediction."""
    batch, _, height, width = map(int, tensor_shape)
    if operation == "decode":
        height, width = height * 8, width * 8
    elif operation != "encode":
        raise ValueError("VAE operation must be encode or decode.")
    return batch * height * width * 128 * int(element_bytes) * 6 + 2 * 2**30


def configure_torch_runtime():
    if FORCE_CPU_MODE:
        print("[Runtime] CPU mode forced; CUDA is hidden for this process.")
    if FORCE_DISABLE_CUDNN:
        try:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("[Runtime] cuDNN disabled for stability.")
        except Exception as exc:
            print(f"[Runtime] cuDNN disable skipped: {exc}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def candidate_pretrained_paths():
    paths = []
    env_path = os.environ.get("SD15_LOCAL_DIR", "")
    if env_path:
        paths.append(env_path)
    paths.extend(LOCAL_MODEL_SEARCH_DIRS)
    paths.extend([
        os.path.join(TRAINED_OUTPUT_DIR, "stable-diffusion-v1-5"),
        os.path.join(TRAINED_OUTPUT_DIR, "sd-v1-5"),
        os.path.join(os.path.dirname(TRAINED_OUTPUT_DIR), "stable-diffusion-v1-5"),
        os.path.join(os.path.dirname(TRAINED_OUTPUT_DIR), "models", "stable-diffusion-v1-5"),
    ])
    seen = set()
    unique = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path)) if os.path.isabs(path) else path
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def configure_offline_loading():
    if HF_HOME_DIR:
        os.environ["HF_HOME"] = HF_HOME_DIR
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME_DIR, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME_DIR, "transformers")
    if LOCAL_FILES_ONLY and not ALLOW_HF_DOWNLOAD:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


def resolve_large_field_pretrained_model(checkpoint):
    """Resolve the saved large-field base, without switching to another prior.

    SD15_LOCAL_DIR explicitly identifies a relocated copy of the training base.
    Search directories are accepted automatically only when their model metadata
    identifies the saved model; a similarly named folder is not sufficient.
    This selects a source; digest verification happens after loading components.
    Legacy checkpoints lacking a digest remain explicitly unverified.
    """
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    declared = str(cfg.get("pretrained_model_name_or_path", "")).strip()
    if not declared:
        raise ValueError("Large-field checkpoint must record config.pretrained_model_name_or_path.")
    override = os.environ.get("SD15_LOCAL_DIR", "").strip()
    if override:
        override = os.path.abspath(os.path.expanduser(override))
        if not is_local_diffusers_model_dir(override):
            raise FileNotFoundError(f"SD15_LOCAL_DIR is not a complete training-base copy: {override}")
        print(f"[LargeField base] Explicit relocated copy: {override}; saved base={declared}. "
              "Loaded weights will be checked against the saved hash if available.")
        return override
    if is_local_diffusers_model_dir(declared):
        print(f"[LargeField base] Saved model directory: {declared}")
        return declared

    candidates = [p.strip().strip('"') for p in str(cfg.get("local_model_search_dirs", "")).split(";") if p.strip()]
    candidates.extend(candidate_pretrained_paths())
    found = []
    seen = set()
    declared_identity = declared.replace("\\", "/").rstrip("/")
    for candidate in candidates:
        if not is_local_diffusers_model_dir(candidate):
            continue
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if os.path.normcase(candidate) in seen:
            continue
        seen.add(os.path.normcase(candidate))
        found.append(candidate)
        identities = set()
        for relative in ("model_index.json", os.path.join("unet", "config.json")):
            metadata_path = os.path.join(candidate, relative)
            if not os.path.isfile(metadata_path):
                continue
            try:
                with open(metadata_path, encoding="utf-8") as file:
                    metadata = _contrast_json_load(file)
            except (OSError, ValueError):
                continue
            if isinstance(metadata, dict):
                for key in ("_name_or_path", "name_or_path"):
                    value = metadata.get(key)
                    if isinstance(value, str):
                        identities.add(value.replace("\\", "/").rstrip("/"))
        if declared_identity in identities:
            print(f"[LargeField base] Local copy whose metadata identifies {declared}: {candidate}. "
                  "Weight identity is checked separately when the checkpoint records a hash.")
            return candidate

    parts = declared_identity.split("/")
    is_repo_id = (len(parts) == 2 and all(part and part not in (".", "..") for part in parts)
                  and not any(char in declared for char in ("\\", ":")))
    if is_repo_id:
        print(f"[LargeField base] Using exactly the saved repository/cache identity: {declared}")
        if found:
            print("[LargeField base] Unverified local folders were not substituted: " + ", ".join(found)
                  + ". To relocate the original training base, set SD15_LOCAL_DIR to that exact copy.")
        return declared
    raise FileNotFoundError(
        f"Saved large-field base is unavailable: {declared}. "
        "Set SD15_LOCAL_DIR to a complete relocated copy of the same training model "
        "(VAE, UNet, tokenizer, text_encoder and scheduler). "
        f"Unverified local candidates: {found}."
    )


def base_weight_sha256(vae, unet, text_encoder):
    """Same hash as the trainer, before LoRA injection and dtype conversion."""
    digest = hashlib.sha256()
    for label, model in (("vae", vae), ("unet", unet), ("text_encoder", text_encoder)):
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{label}.{name}:{value.dtype}:{value.shape}\n".encode("utf-8"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def verify_base_digest(checkpoint, actual_digest):
    expected = checkpoint.get("base_weight_sha256")
    if expected and expected != actual_digest:
        raise ValueError(
            f"Training/inference base-weight mismatch: saved={expected}, loaded={actual_digest}. "
            "Use the same VAE, UNet and text encoder used for training.")
    status = "verified_against_training" if expected else "legacy_training_identity_unverified"
    print(f"[Base identity] {status}; loaded SHA256={actual_digest}", flush=True)
    return status


def validate_native_resolution(resolution, width=None, height=None):
    if int(resolution) != int(REQUIRED_NATIVE_RESOLUTION):
        raise ValueError(
            f"Loaded checkpoint trained at {resolution}px, required {REQUIRED_NATIVE_RESOLUTION}px. "
            "Select an actual 1536-trained checkpoint; changing output size does not change training resolution.")
    if width is not None:
        if height is None or min(int(width), int(height)) < 64 or int(width) % 64 or int(height) % 64:
            raise ValueError("Whole-field width and height must be positive multiples of 64; no resize or stitching fallback.")


@lru_cache(maxsize=1)
def load_matching_trainer():
    script = Path(TRAINER_SCRIPT).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Place the matching v4 train_large_field.py beside this script: {script}")
    spec = importlib.util.spec_from_file_location("_inverse_full_unet_v4_trainer", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # __main__ is not invoked; no training or PSIM run.
    required = ("LargeFieldConfig", "GlobalMorphologyToken", "LargeFieldSpatialConditioner",
                "restore_generation_weights", "build_generation_condition", "sample_full_field")
    if (getattr(module, "CHECKPOINT_FORMAT", None) != CHECKPOINT_FORMAT
            or getattr(module, "CONDITION_SCHEMA_VERSION", None) != CONDITION_SCHEMA_VERSION
            or any(not callable(getattr(module, name, None)) for name in required)):
        raise ValueError("The companion trainer is not the matching full-UNet v4 implementation.")
    if list(module.CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("Trainer and inference class order differ.")
    return module


def build_sampling_scheduler(model, spacing="default"):
    if spacing not in ("default", "leading", "trailing", "linspace"):
        raise ValueError("Sampling spacing must be default, leading, trailing or linspace.")
    config = dict(model.training_scheduler_config)
    if config.get("prediction_type", "epsilon") not in ("epsilon", "v_prediction"):
        raise ValueError("Only epsilon and v_prediction are supported.")
    name = getattr(model, "sampling_scheduler_name", "ddpm")
    scheduler_cls = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}[name]
    kwargs = {} if spacing == "default" else {"timestep_spacing": spacing}
    return scheduler_cls.from_config(config, **kwargs)


def sampler_run_identity(model):
    """Cache identity includes loaded base and the actual schedule, not a global label."""
    return {
        "regional_guidance": {
            "enabled": REGIONAL_GUIDANCE_ENABLED,
            "revision": REGIONAL_GUIDANCE_REVISION,
            "mask_smooth_radius": REGIONAL_MASK_SMOOTH_RADIUS,
            "physics_policy": REGIONAL_PHYSICS_POLICY,
            "layout": regional_layout_identity(),
            "scope": "target_area_partition_v1 recipes; pure recipes use trainer sampler",
        },
        "checkpoint_format": CHECKPOINT_FORMAT,
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "weights": model.generation_weight_source,
        "ema_updates": model.ema_updates,
        "active_condition_strength": model.active_condition_strength,
        "class_prompts": json.loads(model.large_field_config.class_prompts_json),
        "global_condition_scale": model.large_field_config.global_condition_scale,
        "scheduler_class": type(model.scheduler).__name__,
        "trainer_sha256": _file_sha256(os.path.expanduser(getattr(model.trainer, "__file__", TRAINER_SCRIPT))),
        "loaded_base_weight_sha256": getattr(model, "loaded_base_weight_sha256", None),
        "base_identity_status": getattr(model, "base_identity_status", "unavailable"),
        "scheduler_config": _json_ready(dict(model.scheduler.config)),
        "sampling_timesteps": model.scheduler.timesteps.detach().cpu().tolist(),
        "vae_scaling_factor": float(model.vae.config.scaling_factor),
        "inference_dtype": str(model.dtype),
    }


def _latent_statistics(tensor):
    value = tensor.detach().float()
    return {"mean": float(value.mean().item()), "std": float(value.std().item()),
            "min": float(value.min().item()), "max": float(value.max().item()),
            "rms": float(value.square().mean().sqrt().item())}


def load_diffusers_component(component_cls, model_path, subfolder, label):
    local_only = bool(LOCAL_FILES_ONLY and not ALLOW_HF_DOWNLOAD)
    try:
        return component_cls.from_pretrained(
            model_path,
            subfolder=subfolder,
            local_files_only=local_only,
        )
    except Exception as exc:
        search_lines = "\n".join(f"  - {p}" for p in candidate_pretrained_paths())
        raise RuntimeError(
            f"\nFailed to load {label} from: {model_path}\n"
            f"subfolder: {subfolder}\n"
            f"local_files_only: {local_only}\n\n"
            "This is usually caused by an incomplete local Stable Diffusion v1-5 cache or network timeout.\n"
            "Recommended fixes:\n"
            "1. Put a complete diffusers-format SD v1-5 directory on disk with:\n"
            "   vae/, unet/, tokenizer/, text_encoder/, scheduler/\n"
            "2. Set SD15_LOCAL_DIR to a relocated copy of the SAME saved training base.\n"
            "3. Or set before running: set SD15_LOCAL_DIR=D:\\path\\to\\stable-diffusion-v1-5\n"
            "4. If you really want to download online, set ALLOW_HF_DOWNLOAD=True and LOCAL_FILES_ONLY=False.\n\n"
            f"Configured search paths:\n{search_lines}\n\n"
            f"Original error:\n{repr(exc)}"
        ) from exc


def save_pil(img, path):
    ensure_dir(os.path.dirname(path))
    img.save(path)


def tensor_to_pil(x):
    x = (x.detach().float().cpu().clamp(-1, 1) + 1) / 2
    arr = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def normalize_fp(fp_raw, fp_mean, fp_std):
    return (fp_raw - fp_mean) / (fp_std + FP_EPS)


def _validate_fingerprint_stats(fp_mean, fp_std, source):
    fp_mean = np.asarray(fp_mean, dtype=np.float32).reshape(-1)
    fp_std = np.asarray(fp_std, dtype=np.float32).reshape(-1)
    if fp_mean.shape != (FP_DIM,) or fp_std.shape != (FP_DIM,):
        raise ValueError(
            f"{source} contains fingerprint stats with shapes {fp_mean.shape}/{fp_std.shape}; "
            f"expected ({FP_DIM},) for {FP_NAMES}. A legacy six-dimensional checkpoint cannot be used."
        )
    if not np.isfinite(fp_mean).all() or not np.isfinite(fp_std).all() or np.any(fp_std < 0):
        raise ValueError(f"{source} contains non-finite or negative fingerprint statistics.")
    return fp_mean, np.maximum(fp_std, 1e-6)


def build_large_field_spatial_condition(
    hard_fraction,
    class_map,
    fp_map,
    latent_h,
    latent_w,
    connectivity=None,
    class_assignment_entropy=None,
    class_valid_pixel_counts=None,
):
    frac = np.asarray(hard_fraction, dtype=np.float32).reshape(-1)
    if (frac.shape != (N_CLASSES,) or not np.isfinite(frac).all()
            or np.any(frac < 0) or not np.isclose(frac.sum(), 1.0, atol=1e-5)):
        raise ValueError("Four nonnegative finite class fractions must sum to one.")

    probabilities = np.asarray(class_map, dtype=np.float32)
    if probabilities.ndim == 3 and probabilities.shape[-1] == N_CLASSES:
        grid = probabilities.shape[0]
        if probabilities.shape[1] != grid:
            raise ValueError("Local class probability map must use a square grid.")
        probabilities = probabilities.reshape(-1, N_CLASSES)
    elif probabilities.ndim == 2 and probabilities.shape[1] == N_CLASSES:
        grid = math.isqrt(probabilities.shape[0])
        if grid * grid != probabilities.shape[0]:
            raise ValueError("Local class probability map must use a square grid.")
    else:
        raise ValueError("class_map must hold (grid, grid, 4) local class probabilities, not hard class IDs.")
    if (grid < 1 or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Local class probabilities must be nonnegative, sum to one per cell, and average to the target fractions.")
    if class_valid_pixel_counts is None:
        aggregated = probabilities.mean(axis=0)
    else:
        counts = np.asarray(class_valid_pixel_counts, dtype=np.float64)
        if (counts.shape != (len(probabilities),) or not np.isfinite(counts).all()
                or np.any(counts < 0) or counts.sum() <= 0):
            raise ValueError("Cached class maps require a valid retained-pixel count for each cell.")
        aggregated = np.average(probabilities, axis=0, weights=counts)
    if not np.allclose(aggregated, frac, atol=1e-5):
        raise ValueError("Local class probabilities must average to the target fractions using retained pixel counts.")
    if latent_h % grid or latent_w % grid:
        raise ValueError("Latent dimensions must be divisible by the trained patch grid.")

    fp_map = np.asarray(fp_map, dtype=np.float32)
    if fp_map.ndim == 1:
        fp_map = np.repeat(fp_map.reshape(1, -1), probabilities.shape[0], axis=0)
    if fp_map.ndim != 2 or fp_map.shape[0] != probabilities.shape[0]:
        raise ValueError(
            f"fp_map must have one row per class-map cell ({probabilities.shape[0]}), got {fp_map.shape}"
        )
    if fp_map.shape[1] != FP_DIM or not np.isfinite(fp_map).all():
        raise ValueError(f"fp_map must have {FP_DIM} fingerprint values per cell, got {fp_map.shape}")

    if connectivity is None:
        connectivity = np.zeros(CONNECTIVITY_DIM, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.float32).reshape(-1)
    if connectivity.shape != (CONNECTIVITY_DIM,):
        raise ValueError(f"connectivity must have shape ({CONNECTIVITY_DIM},), got {connectivity.shape}")
    connectivity = np.clip(connectivity, 0.0, 1.0)
    clipped = np.clip(probabilities, 1e-8, 1.0)
    local_entropies = -np.sum(clipped * np.log(clipped), axis=1) / np.log(float(N_CLASSES))
    mean_entropy = float(local_entropies.mean())
    if class_assignment_entropy is not None and not np.isclose(class_assignment_entropy, mean_entropy, atol=1e-5):
        raise ValueError("Global class entropy must equal the mean of local cell entropies.")

    cond = np.zeros((SPATIAL_CONDITION_CHANNELS, latent_h, latent_w), dtype=np.float32)
    cell_h = max(1, latent_h // grid)
    cell_w = max(1, latent_w // grid)
    for idx, local_fractions in enumerate(probabilities):
        iy = idx // grid
        ix = idx % grid
        y0 = iy * cell_h
        x0 = ix * cell_w
        y1 = latent_h if iy == grid - 1 else min(latent_h, (iy + 1) * cell_h)
        x1 = latent_w if ix == grid - 1 else min(latent_w, (ix + 1) * cell_w)
        cond[:N_CLASSES, y0:y1, x0:x1] = local_fractions[:, None, None]
        cond[N_CLASSES:N_CLASSES + FP_DIM, y0:y1, x0:x1] = fp_map[idx, :FP_DIM, None, None]
        conn_start = N_CLASSES + FP_DIM
        cond[conn_start:conn_start + CONNECTIVITY_DIM, y0:y1, x0:x1] = connectivity[:, None, None]
        cond[conn_start + CONNECTIVITY_DIM, y0:y1, x0:x1] = local_entropies[idx]

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, latent_h, dtype=np.float32),
        np.linspace(-1.0, 1.0, latent_w, dtype=np.float32),
        indexing="ij",
    )
    cond[-2] = xx
    cond[-1] = yy
    return cond


def regional_layout_identity():
    return {"revision": REGIONAL_LAYOUT_REVISION, "seed": REGIONAL_LAYOUT_SEED,
            "domains": REGIONAL_LAYOUT_DOMAINS, "resolution": "full_latent_canvas",
            "explicit_local_maps": "preserved",
            "meaning": "target routing geometry, not measured morphology"}


def build_irregular_regional_weights(class_fractions, latent_h, latent_w, seed,
                                     domains=3.0, radius=2):
    """Seeded smooth colonies without a preferred axis; exact SOFT input quotas.

    Rank a continuous isotropic random field at latent resolution. This avoids
    the old grid-cell staircase and left/right stripe. Colony geometry is a
    heuristic prior, not a learned grain-boundary model or output segmentation.
    """
    fractions = np.asarray(class_fractions, dtype=np.float64)
    if (fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1, atol=1e-5)):
        raise ValueError("Invalid regional class fractions.")
    if (not isinstance(latent_h, int) or not isinstance(latent_w, int)
            or min(latent_h, latent_w) < 1
            or not isinstance(seed, int) or not 0 <= seed < 2**32
            or not np.isfinite(domains) or not 1 <= domains <= 12
            or not isinstance(radius, int) or radius < 0 or radius >= min(latent_h, latent_w)):
        raise ValueError("Invalid irregular-layout dimensions, seed, scale or radius.")
    fractions = fractions / fractions.sum()
    rng = np.random.default_rng(seed)
    short = float(min(latent_h, latent_w))
    yy = np.arange(latent_h, dtype=np.float64)[:, None] / short
    xx = np.arange(latent_w, dtype=np.float64)[None, :] / short
    field = np.zeros((latent_h, latent_w), dtype=np.float64)
    for low, high, amplitude, count in ((.65, 1.35, 1., 32), (1.5, 2.8, .22, 24)):
        for _ in range(count):
            angle = rng.uniform(0, 2*np.pi)
            freq = domains * rng.uniform(low, high)
            phase = rng.uniform(0, 2*np.pi)
            field += (amplitude / np.sqrt(count)) * np.cos(
                2*np.pi*freq*(np.cos(angle)*xx + np.sin(angle)*yy) + phase)
    order = np.argsort(field.ravel(), kind="stable")
    n = latent_h * latent_w
    edges = np.r_[0., np.cumsum(fractions * n)]
    ranks = np.arange(n, dtype=np.float64)
    flat = np.zeros((N_CLASSES, n), dtype=np.float64)
    for cid in range(N_CLASSES):
        flat[cid, order] = np.maximum(
            0., np.minimum(ranks + 1., edges[cid+1]) - np.maximum(ranks, edges[cid]))
    weights = flat.reshape(N_CLASSES, latent_h, latent_w)
    # Symmetric box kernels preserve global mass and avoid wraparound at image edges.
    for axis in (1, 2):
        if not radius:
            continue
        pads = [(0, 0)] * 3
        pads[axis] = (radius, radius)
        padded = np.pad(weights, pads, mode="symmetric")
        filtered = np.zeros_like(weights)
        for offset in range(2*radius+1):
            selection = [slice(None)] * 3
            selection[axis] = slice(offset, offset+weights.shape[axis])
            filtered += padded[tuple(selection)]
        weights = filtered / (2*radius+1)
    if (not np.allclose(weights.sum(0), 1., atol=1e-6)
            or not np.allclose(weights.mean((1, 2)), fractions, atol=1e-6)):
        raise RuntimeError("Irregular routing weights failed quota conservation.")
    return weights.astype(np.float32)


def regional_weights_to_grid(weights, grid):
    """Average the SAME full-latent routing weights onto the training grid."""
    weights = np.asarray(weights)
    if weights.ndim != 3 or weights.shape[0] != N_CLASSES:
        raise ValueError("Expected C,H,W regional routing weights.")
    _, h, w = weights.shape
    if (not isinstance(grid, int) or grid < 1 or h % grid or w % grid
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(0), 1, atol=1e-6)):
        raise ValueError("Invalid routing weights or incompatible training grid.")
    return weights.reshape(N_CLASSES, grid, h//grid, grid, w//grid).mean(
        (2, 4), dtype=np.float64).transpose(1, 2, 0).reshape(-1, N_CLASSES).astype(np.float32)


def build_spatial_composition_map(class_fractions, grid):
    """Allocate a global recipe over the trained grid with exact area-weighted targets."""
    fractions = np.asarray(class_fractions, dtype=np.float64)
    grid = int(grid)
    if (grid < 1 or fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-5)):
        raise ValueError("A valid four-class recipe and positive spatial grid are required.")
    fractions = fractions / fractions.sum()

    yy, xx = np.mgrid[0:grid, 0:grid]
    # Order cells into broad wavy domains instead of broadcasting a mixture or
    # making a checkerboard. Fractional boundary cells preserve exact area totals.
    score = (xx.astype(np.float64)
             + 0.55 * np.sin(2.0 * np.pi * yy / max(grid, 2))
             + 0.12 * np.sin(2.0 * np.pi * (xx + yy) / max(grid, 2)))
    order = np.argsort(score.ravel(), kind="stable")
    cell_count = grid * grid
    class_edges = np.concatenate(([0.0], np.cumsum(fractions * cell_count)))
    result = np.zeros((cell_count, N_CLASSES), dtype=np.float32)
    for rank, cell_index in enumerate(order):
        cell_start, cell_end = float(rank), float(rank + 1)
        for class_id in range(N_CLASSES):
            overlap = max(0.0, min(cell_end, class_edges[class_id + 1])
                          - max(cell_start, class_edges[class_id]))
            result[cell_index, class_id] = overlap
        if not np.isclose(result[cell_index].sum(), 1.0, atol=1e-8):
            raise RuntimeError("Failed to allocate a complete spatial class condition cell.")
    result = result.reshape(grid, grid, N_CLASSES)
    if not np.allclose(result.mean(axis=(0, 1)), fractions, atol=1e-5):
        raise RuntimeError("Spatial class allocation does not preserve the requested global fractions.")
    return result


def make_large_field_condition_maps(condition, data, grid, seed=SEED):
    """Allocate explicit target regions, or preserve measured/legacy numerical maps."""
    del data, seed  # Retained call signature; condition maps are deterministic.
    grid = int(grid)
    if grid < 1:
        raise ValueError("The large-field patch grid must be positive.")
    n_cells = grid * grid
    fractions = np.asarray(condition["hard_fraction"], dtype=np.float32)
    if (fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-5)):
        raise ValueError("Four nonnegative finite class fractions must sum to one.")
    local = condition.get("class_probability_map")
    if local is None:
        if condition.get("spatial_class_map_strategy") == "target_area_partition_v1":
            local = build_spatial_composition_map(fractions, grid).reshape(n_cells, N_CLASSES)
        else:
            local = np.broadcast_to(fractions, (n_cells, N_CLASSES)).copy()
    else:
        local = np.asarray(local, dtype=np.float32)
        if local.shape == (grid, grid, N_CLASSES):
            local = local.reshape(n_cells, N_CLASSES)
    if (local.shape != (n_cells, N_CLASSES) or not np.isfinite(local).all()
            or np.any(local < 0) or not np.allclose(local.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Local class probabilities must be finite, nonnegative and sum to one in each cell.")
    valid_counts = condition.get("class_valid_pixel_counts")
    if valid_counts is None:
        global_from_cells = local.mean(axis=0)
    else:
        valid_counts = np.asarray(valid_counts, dtype=np.float64)
        if (valid_counts.shape != (n_cells,) or not np.isfinite(valid_counts).all()
                or np.any(valid_counts < 0) or valid_counts.sum() <= 0):
            raise ValueError("Cached class maps require a valid retained-pixel count for every cell.")
        # Training excludes rare-class pixels only from class-area denominators.
        # Physical dark-contrast / α-lath width fields still average over every source pixel.
        global_from_cells = np.average(local, axis=0, weights=valid_counts)
    if not np.allclose(global_from_cells, fractions, atol=1e-5):
        raise ValueError("Local class probabilities must average to the global four-class target using retained pixel counts when provided.")
    fingerprint = np.asarray(condition["fingerprint"], dtype=np.float32)
    if fingerprint.shape != (FP_DIM,) or not np.isfinite(fingerprint).all():
        raise ValueError(f"A finite normalized fingerprint of shape ({FP_DIM},) is required.")
    fingerprint_map = condition.get("fingerprint_map")
    if fingerprint_map is None:
        fingerprint_map = np.repeat(fingerprint[None, :], n_cells, axis=0)
    else:
        fingerprint_map = np.asarray(fingerprint_map, dtype=np.float32)
        if fingerprint_map.shape == (grid, grid, FP_DIM):
            fingerprint_map = fingerprint_map.reshape(n_cells, FP_DIM)
        if (fingerprint_map.shape != (n_cells, FP_DIM) or not np.isfinite(fingerprint_map).all()
                or not np.allclose(fingerprint_map.mean(axis=0), fingerprint, atol=1e-5)):
            raise ValueError("Local normalized fingerprints must match the training grid and average to the global fingerprint.")
    return np.asarray(local, dtype=np.float32).copy(), np.asarray(fingerprint_map, dtype=np.float32).copy()


def prepare_canvas_recipe(condition, native_resolution, output_width, output_height):
    """Keep numeric global targets; do not stretch a measured crop's spatial layout."""
    result = dict(condition)
    extrapolated = (int(output_width), int(output_height)) != (int(native_resolution), int(native_resolution))
    if extrapolated:
        preserve_target_map = result.get("spatial_class_map_strategy") == "target_area_partition_v1"
        removed_keys = ["fingerprint_map", "class_valid_pixel_counts",
                        "class_assignment_entropy", "native_psim_region"]
        if not preserve_target_map:
            removed_keys.insert(0, "class_probability_map")
        for key in removed_keys:
            result.pop(key, None)
        result["spatial_recipe_scope"] = (
            "target_area_partition_preserved_on_extrapolated_canvas" if preserve_target_map else
            "global_numeric_targets_broadcast_uniformly_to_extrapolated_canvas")
    else:
        result["spatial_recipe_scope"] = ("native_target_area_partition" if
            result.get("spatial_class_map_strategy") == "target_area_partition_v1" else
            "native_crop_local_maps_or_uniform_global_targets")
    return result


def regional_mask_weights(class_map, latent_h, latent_w, radius=2):
    """Target routing weights, not measured segmentation. Preserve soft area mass."""
    local = np.asarray(class_map, dtype=np.float64)
    if local.ndim == 2:
        grid = math.isqrt(len(local))
        local = local.reshape(grid, grid, N_CLASSES)
    if (local.ndim != 3 or local.shape[2] != N_CLASSES
            or local.shape[0] != local.shape[1] or local.shape[0] < 1):
        raise ValueError("Regional map must be a square four-class probability grid.")
    if (not np.isfinite(local).all() or np.any(local < 0)
            or not np.allclose(local.sum(-1), 1, atol=1e-6)):
        raise ValueError("Invalid regional probabilities.")
    grid = local.shape[0]
    if min(latent_h, latent_w) < 1 or latent_h % grid or latent_w % grid:
        raise ValueError("Regional latent dimensions must be divisible by the grid.")
    if not isinstance(radius, int) or radius < 0 or radius >= min(latent_h, latent_w):
        raise ValueError("Invalid regional boundary smoothing radius.")
    weights = np.repeat(np.repeat(local, latent_h // grid, axis=0), latent_w // grid, axis=1)
    weights = weights.transpose(2, 0, 1)
    for axis in (1, 2):
        if not radius:
            continue
        pads = [(0, 0)] * 3
        pads[axis] = (radius, radius)
        padded = np.pad(weights, pads, mode="symmetric")
        filtered = np.zeros_like(weights)
        for offset in range(2 * radius + 1):
            selection = [slice(None)] * 3
            selection[axis] = slice(offset, offset + weights.shape[axis])
            filtered += padded[tuple(selection)]
        weights = filtered / (2 * radius + 1)
    if (not np.allclose(weights.sum(0), 1, atol=1e-6)
            or not np.allclose(weights.mean((1, 2)), local.mean((0, 1)), atol=1e-6)):
        raise RuntimeError("Regional smoothing changed target mass or left uncovered pixels.")
    return weights.astype(np.float32)


def _select_secondary_device(primary):
    primary = torch.device(primary)
    if INFERENCE_GPU_COUNT < 2 or primary.type != "cuda":
        return None
    candidates = []
    for index in range(torch.cuda.device_count()):
        if index == primary.index:
            continue
        free, _ = torch.cuda.mem_get_info(index)
        if free >= MIN_GPU_FREE_GIB * 2**30:
            candidates.append((int(free), index))
    if not candidates:
        print("[Dual GPU] No eligible second visible GPU; using the original single-GPU sampler.", flush=True)
        return None
    return torch.device(f"cuda:{max(candidates)[1]}")


class DualGPUUNetPredictor:
    """Two persistent trained replicas; parallel forward calls, one shared scheduler.

    Each replica handles a full canvas, never a spatial tile. Worker streams are
    synchronized before returning results; reduction takes place on the primary
    in the original branch order. Thread-local no_grad is entered in each worker.
    """

    def __init__(self, unet, primary_device, secondary_device):
        self.primary = torch.device(primary_device)
        self.secondary = torch.device(secondary_device)
        if self.primary == self.secondary:
            raise ValueError("Dual GPU prediction requires two distinct logical devices.")
        self.devices = (self.primary, self.secondary)
        self.pool = None
        self.models = [unet]
        self.streams = []
        self.closed = False
        self.forward_calls = [0, 0]
        # from_config constructs on CPU: deepcopy(unet) would first allocate an
        # unwanted extra model on the primary GPU. Copy the ALREADY TRAINED state.
        replica = type(unet).from_config(unet.config).to(dtype=next(unet.parameters()).dtype)
        replica.load_state_dict(unet.state_dict(), strict=True)
        if hasattr(unet, "attn_processors"):
            replica.set_attn_processor(copy.deepcopy(unet.attn_processors))
        replica.eval().requires_grad_(False)
        try:
            replica.to(self.secondary)
        except torch.cuda.OutOfMemoryError:
            del replica
            with torch.cuda.device(self.secondary):
                torch.cuda.empty_cache()
            raise
        self.models.append(replica)
        for device in self.devices:
            with torch.cuda.device(device):
                self.streams.append(torch.cuda.Stream(device=device))
                torch.cuda.synchronize(device)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inverse-unet")
        print(f"[Dual GPU] Trained UNet replicas on {self.primary} and {self.secondary}; "
              "parallel CFG/class predictions, one scheduler and one random field. "
              "Each GPU must fit a full UNet forward; VRAM is not pooled.", flush=True)

    @torch.no_grad()
    def prepare(self, conditions):
        """Transfer static (hidden, bias-or-None) once per generated image."""
        if self.closed:
            raise RuntimeError("Dual GPU predictor is closed.")
        if not conditions:
            raise ValueError("At least one prediction condition is required.")
        groups = [[], []]
        # Equal-cost full-canvas jobs: deterministic round-robin assignment.
        for index, (hidden, bias) in enumerate(conditions):
            slot = index % 2
            device = self.devices[slot]
            with torch.cuda.device(device):
                groups[slot].append((index, hidden.detach().to(device, non_blocking=False),
                                     None if bias is None else bias.detach().to(device, non_blocking=False)))
        for device in self.devices:
            torch.cuda.synchronize(device)
        return groups

    def _forward_group(self, slot, jobs, model_input, timestep, spatial_scale):
        device, stream = self.devices[slot], self.streams[slot]
        with torch.no_grad(), torch.cuda.device(device), torch.cuda.stream(stream):
            try:
                value = model_input.to(device, non_blocking=False)
                step = timestep.to(device, non_blocking=False) if torch.is_tensor(timestep) else timestep
                outputs = []
                for index, hidden, bias in jobs:
                    conditioned = value if bias is None else value + spatial_scale * bias
                    prediction = self.models[slot](conditioned, step, encoder_hidden_states=hidden).sample
                    outputs.append((index, prediction))
                    self.forward_calls[slot] += 1
                return outputs
            finally:
                # Complete pending kernels even when a forward fails. Cross-thread
                # consumers and cleanup never race a worker's outstanding writes.
                stream.synchronize()

    def predict(self, model_input, timestep, prepared, spatial_scale):
        if self.closed:
            raise RuntimeError("Dual GPU predictor is closed.")
        # The scheduler/conditioner write on the main thread's stream. Complete
        # those writes before either independent worker stream reads them.
        torch.cuda.synchronize(self.primary)
        futures = [self.pool.submit(self._forward_group, slot, jobs, model_input,
                                    timestep, spatial_scale)
                   for slot, jobs in enumerate(prepared) if jobs]
        groups, error = [], None
        # Drain BOTH futures even if one fails, so no worker retains a live image
        # while cleanup or the error path starts.
        for future in futures:
            try:
                groups.append(future.result())
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        ordered = [None] * sum(len(jobs) for jobs in prepared)
        for group in groups:
            for index, prediction in group:
                ordered[index] = prediction.to(self.primary, non_blocking=False)
        return ordered

    def diagnostics(self):
        return {"mode": "dual_gpu_unet_predictions", "devices": [str(d) for d in self.devices],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "forward_calls_cumulative": list(self.forward_calls),
                "scheduler_device": str(self.primary), "model_replicas": 2,
                "reduction": "original_branch_order_on_primary", "vram_pooled": False}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        self.models.clear()
        self.streams.clear()


def build_dual_gpu_predictor(unet, primary_device):
    secondary = _select_secondary_device(primary_device)
    if secondary is None:
        return None
    try:
        return DualGPUUNetPredictor(unet, primary_device, secondary)
    except torch.cuda.OutOfMemoryError:
        # Replica allocation failed before sampling; safely retain the original path.
        print("[Dual GPU] Secondary UNet allocation ran out of memory; continuing with the "
              "original single-GPU sampler. Full-canvas activations may require more memory.", flush=True)
        return None


@torch.no_grad()
def sample_dual_gpu_full_field(predictor, scheduler, conditional_hidden, empty_hidden,
                              spatial_bias, latent_shape, device, dtype, steps,
                              guidance, seed, spatial_scale):
    """Same trainer CFG and random draws; only the independent forwards run together."""
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    latent = latent * scheduler.init_noise_sigma
    conditions = [(conditional_hidden, spatial_bias)]
    if guidance > 1:
        conditions.append((empty_hidden, None))
    prepared = predictor.prepare(conditions)
    for step_index, timestep in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(latent, timestep)
        predictions = predictor.predict(model_input, timestep, prepared, spatial_scale)
        cond_prediction = predictions[0]
        if guidance > 1:
            uncond_prediction = predictions[1]
            prediction = uncond_prediction + guidance * (cond_prediction - uncond_prediction)
        else:
            prediction = cond_prediction
        latent = scheduler.step(prediction.float(), timestep, latent, generator=generator).prev_sample
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite sampling latents at timestep {timestep}")
        del predictions, prediction, cond_prediction
        if guidance > 1:
            del uncond_prediction
        if step_index == 0 or (step_index + 1) % 25 == 0 or step_index + 1 == len(scheduler.timesteps):
            print(f"[Dual GPU CFG] step {step_index + 1}/{len(scheduler.timesteps)}; "
                  "one whole latent", flush=True)
    return latent


@torch.no_grad()
def sample_regional_full_field(unet, scheduler, branches, empty_hidden, latent_shape,
                               device, steps, guidance, seed, spatial_scale, predictor=None):
    """All class branches see the same full x_t; take ONE scheduler step.

    Branch: (class_id, pure_class_hidden, pure_class_bias, region_weight).
    Mix predictions in the scheduler's native epsilon/v parameterization.
    There is one random field and one DDPM random draw per step, no image stitching.
    """
    if not branches or not np.isfinite(guidance) or guidance < 1:
        raise ValueError("Regional guidance requires active branches and CFG >= 1.")
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(latent_shape, device=device, dtype=torch.float32, generator=generator)
    latent = latent * scheduler.init_noise_sigma
    initial_branch_response = {}
    prepared = None
    if predictor is not None:
        conditions = ([(empty_hidden, None)] if guidance > 1 else [])
        conditions += [(hidden, bias) for _, hidden, bias, _ in branches]
        prepared = predictor.prepare(conditions)
    for step_index, timestep in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(latent, timestep)
        mixed = torch.zeros_like(latent, dtype=torch.float32)
        predictions = (predictor.predict(model_input, timestep, prepared, spatial_scale)
                       if predictor is not None else None)
        uncond = ((predictions[0].float() if predictions is not None else
                   unet(model_input, timestep, encoder_hidden_states=empty_hidden).sample.float())
                  if guidance > 1 else None)
        for branch_index, (class_id, hidden, bias, weight) in enumerate(branches):
            prediction = (predictions[branch_index + int(guidance > 1)].float()
                          if predictions is not None else
                          unet(model_input + spatial_scale * bias, timestep,
                               encoder_hidden_states=hidden).sample.float())
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite regional prediction for class {class_id}.")
            if step_index == 0 and uncond is not None:
                initial_branch_response[CLASS_NAMES[class_id]] = float(
                    (prediction - uncond).abs().mean().item())
            mixed.add_(prediction * weight)
            del prediction
        if uncond is not None:
            mixed = uncond + guidance * (mixed - uncond)
        latent = scheduler.step(mixed, timestep, latent, generator=generator).prev_sample
        del predictions, uncond, mixed
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite regional latent at timestep {timestep}.")
        if step_index == 0 or (step_index + 1) % 25 == 0 or step_index + 1 == len(scheduler.timesteps):
            print(f"[Regional] step {step_index + 1}/{len(scheduler.timesteps)}; "
                  f"{len(branches)} class branches, one whole latent", flush=True)
    return latent, {
        "first_step_class_vs_unconditional_prediction_mae": initial_branch_response,
        "prediction_parameterization": scheduler.config.prediction_type,
        "response_interpretation": "Numerical branch response only; not morphology accuracy.",
    }


def export_regional_target_weights(image_path, weights):
    """Save routing evidence separately from generated pixels and PSIM labels."""
    path = Path(image_path)
    npz_path = path.with_name(path.stem + "_target_region_weights.npz")
    png_path = path.with_name(path.stem + "_target_regions_NOT_measurement.png")
    np.savez_compressed(npz_path, weights=weights, class_names=np.asarray(CLASS_NAMES),
                        meaning="target latent routing weights, not output measurement")
    palette = np.asarray([[0, 114, 178], [204, 121, 167], [230, 159, 0], [0, 158, 115]], dtype=np.float32)
    rgb = np.einsum("chw,cd->hwd", weights, palette).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(png_path)
    return {"target_region_weights_path": str(npz_path),
            "target_region_preview_path": str(png_path),
            "target_region_preview_is_measurement": False}


@torch.no_grad()
def large_field_sampling(model, condition, data, seed=SEED, name="large_field",
                         output_width=None, output_height=None, window_stride=None,
                         window_batch_size=DEFAULT_WINDOW_BATCH_SIZE):
    if not model.large_field_available:
        raise RuntimeError("large_field_model.pt is not loaded.")

    device = model.device
    native_px = int(model.large_field_resolution)
    width = native_px if output_width is None else int(output_width)
    height = native_px if output_height is None else int(output_height)
    validate_native_resolution(native_px, width, height)
    condition = prepare_canvas_recipe(condition, native_px, width, height)
    grid = model.large_field_patch_grid
    fractions = np.asarray(condition["hard_fraction"], dtype=np.float32)
    active_classes = np.flatnonzero(fractions > 0)
    use_regional = (REGIONAL_GUIDANCE_ENABLED and len(active_classes) > 1
                    and condition.get("spatial_class_map_strategy") == "target_area_partition_v1")
    full_weights = None
    if use_regional and condition.get("class_probability_map") is None:
        full_weights = build_irregular_regional_weights(
            fractions, height // 8, width // 8, REGIONAL_LAYOUT_SEED,
            REGIONAL_LAYOUT_DOMAINS, REGIONAL_MASK_SMOOTH_RADIUS)
        condition = dict(condition)
        condition["class_probability_map"] = regional_weights_to_grid(full_weights, int(grid))
    class_map, fp_map = make_large_field_condition_maps(condition, data, grid, seed=seed)
    hidden, _, bias = model.make_large_field_condition(
        condition["fingerprint"], fractions, class_map, fp_map,
        int(condition.get("target_class_id", np.argmax(fractions))),
        class_valid_pixel_counts=condition.get("class_valid_pixel_counts"),
        latent_h=height // 8, latent_w=width // 8)
    expected_bias_shape = (1, model.unet.config.in_channels, height // 8, width // 8)
    if tuple(bias.shape) != expected_bias_shape:
        raise ValueError(f"Spatial bias shape {tuple(bias.shape)} differs from whole latent {expected_bias_shape}.")
    if LARGE_FIELD_GUIDANCE_START != LARGE_FIELD_GUIDANCE_END:
        raise ValueError("v4 uses fixed CFG, matching the trainer's validation sampler.")
    guidance = LARGE_FIELD_GUIDANCE_START
    if guidance > 1 and not 0 < float(model.large_field_config.condition_dropout) < 1:
        raise ValueError("CFG requires a trained unconditional branch; set guidance=1.")
    empty = model.encode_prompt("")
    print(f"[LargeField] {name}: {width}x{height}, full latent, "
          f"{model.sampling_scheduler_name}/{LARGE_FIELD_NUM_INFERENCE_STEPS}, "
          f"CFG={guidance:g}, weights={model.generation_weight_source}; no stitching")
    if (width, height) != (native_px, native_px):
        ratio = width * height / float(native_px ** 2)
        print(f"[Size extrapolation] trained={native_px}x{native_px}; requested={width}x{height}; "
              f"latent area={ratio:.2f}x training. Whole-field quality and GPU capacity need validation.", flush=True)
    model.last_regional_target_weights = None
    regional_diagnostics = {"regional_guidance_applied": False}
    predictor = (model.get_prediction_parallelism()
                 if guidance > 1 or use_regional else None)
    denoising_started = perf_counter()
    try:
        if use_regional:
            weights = (full_weights if full_weights is not None else
                       regional_mask_weights(class_map, height // 8, width // 8,
                                             REGIONAL_MASK_SMOOTH_RADIUS))
            actual_layout = (regional_layout_identity() if full_weights is not None else
                             {"revision": "explicit_local_class_map"})
            print(f"[Regional layout] {actual_layout}; target weights are not measured output areas.",
                  flush=True)
            branches = []
            for class_id in active_classes:
                pure = np.eye(N_CLASSES, dtype=np.float32)[class_id]
                pure_map = np.broadcast_to(pure, (grid * grid, N_CLASSES)).copy()
                branch_hidden, _, branch_bias = model.make_large_field_condition(
                    condition["fingerprint"], pure, pure_map, fp_map, int(class_id),
                    latent_h=height // 8, latent_w=width // 8)
                if tuple(branch_bias.shape) != expected_bias_shape:
                    raise ValueError("Regional branch bias does not match the full canvas.")
                weight = torch.from_numpy(weights[class_id][None, None]).to(device=device, dtype=torch.float32)
                branches.append((int(class_id), branch_hidden, branch_bias, weight))
            print("[Regional] Pure-class full-canvas predictions: " +
                  ", ".join(CLASS_NAMES[c] for c in active_classes) +
                  "; shared requested Dark-contrast area fraction/α-lath width; no image stitching.", flush=True)
            latents, response = sample_regional_full_field(
                model.unet, model.scheduler, branches, empty, expected_bias_shape, device,
                LARGE_FIELD_NUM_INFERENCE_STEPS, guidance, seed, model.large_field_spatial_scale,
                predictor=predictor)
            regional_diagnostics.update(response)
            regional_diagnostics.update(
                regional_guidance_applied=True,
                regional_layout=actual_layout,
                regional_active_classes=[CLASS_NAMES[c] for c in active_classes],
                regional_target_weight_fractions=weights.mean((1, 2), dtype=np.float64).tolist(),
                regional_physics_policy=REGIONAL_PHYSICS_POLICY,
                regional_mask_smooth_radius=REGIONAL_MASK_SMOOTH_RADIUS,
                regional_interpretation="Experimental routing; target weights are not achieved area fractions.")
            model.last_regional_target_weights = weights
            del branches, branch_hidden, branch_bias, weight
        elif predictor is not None:
            latents = sample_dual_gpu_full_field(
                predictor, model.scheduler, hidden, empty, bias,
                expected_bias_shape, device, torch.float32,
                LARGE_FIELD_NUM_INFERENCE_STEPS, guidance, seed, model.large_field_spatial_scale)
        else:
            latents = model.trainer.sample_full_field(
                model.unet, model.scheduler, hidden, empty, bias,
                expected_bias_shape, device, torch.float32,
                LARGE_FIELD_NUM_INFERENCE_STEPS, guidance, seed, model.large_field_spatial_scale)
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            f"Whole-canvas UNet inference ran out of GPU memory at {width}x{height}. "
            "Completed candidates remain saved. A larger-memory device or explicit CPU inference is required "
            "for this size. Dual GPU prediction does not pool VRAM: both devices need a complete forward. "
            "Use LEE_INVERSE_GPU_COUNT=1 if the secondary alone lacks capacity; "
            "no patch generation, stitching or smaller-image substitution was performed.") from error
    model.last_large_field_diagnostics = {
        "spatial_bias_rms": float(bias.float().square().mean().sqrt().item()),
        "first_step_cond_uncond_epsilon_mae": None,
        "latent_trace": [{"stage": "final_latent", **_latent_statistics(latents)}],
        "sampler": sampler_run_identity(model),
    }
    model.last_large_field_diagnostics.update(regional_diagnostics)
    # Execution provenance is deliberately outside sampler_run_identity: choosing
    # different hardware does not invalidate completed image/cache checks.
    model.last_large_field_diagnostics["execution_parallelism"] = (
        predictor.diagnostics() if predictor is not None else
        {"mode": "single_device", "devices": [str(device)], "model_replicas": 1})
    denoising_seconds = perf_counter() - denoising_started
    model.last_large_field_diagnostics["denoising_wall_seconds"] = denoising_seconds
    print(f"[Denoising timing] {denoising_seconds:.2f}s; "
          f"{LARGE_FIELD_NUM_INFERENCE_STEPS / max(denoising_seconds, 1e-9):.3f} steps/s; "
          "excludes model/replica initialization and VAE decode.", flush=True)
    image = model.decode_latents(latents, expected_hw=(height, width))
    model.last_large_field_diagnostics.update(getattr(model, "last_decode_diagnostics", {}))
    return image, class_map, fp_map


# Reporting aliases do not imply crystallographic phase identification.
CONTRAST_TERMINOLOGY = {
    "dark_label": "Dark-contrast area fraction",
    "bright_label": "Bright-contrast area fraction",
    "dark_field": "dark_contrast_area_fraction",
    "bright_field": "bright_contrast_area_fraction",
    "dark_segmentation_title": "Dark-contrast segmentation",
    "bright_segmentation_title": "Bright-contrast segmentation",
    "width_label": "Mean α-lath width",
    "width_unit": "um",
    "fingerprint_feature_order": ["dark_contrast_area_fraction", "mean_lath_width"],
    "numeric_contract": "Unchanged measurements, normalization, channel order and micrometre width units.",
    "bright_scope": "Only rename existing bright measurements; no complement of normalized fingerprints or field-response maps is calculated.",
    "segmentation_scope": "Generation-only outputs; no generated-image segmentation or verification is run.",
    "legacy_wire_scope": "Legacy beta keys remain compatibility aliases; signed run_spec/identity objects retain their original representation.",
}
CONTRAST_FIELD_ALIASES = {
    "beta_phase_fraction": "dark_contrast_area_fraction",
    "global_beta_phase_fraction": "global_dark_contrast_area_fraction",
    "beta_phase_fraction_field": "dark_contrast_area_fraction_field",
    "target_beta_fraction": "target_dark_contrast_area_fraction",
    "measured_beta_fraction": "measured_dark_contrast_area_fraction",
    "condition_beta_fraction": "condition_dark_contrast_area_fraction",
    "beta_source": "dark_contrast_area_fraction_source",
    "beta_signed_error_percentage_points": "dark_contrast_area_fraction_signed_error_percentage_points",
    "beta_absolute_error_percentage_points": "dark_contrast_area_fraction_absolute_error_percentage_points",
    "beta_absolute_error": "dark_contrast_area_fraction_absolute_error",
    "beta_absolute": "dark_contrast_area_fraction_absolute",
    # Read/export historical bright measurements only when actually supplied.
    "alpha_phase_fraction": "bright_contrast_area_fraction",
    "global_alpha_phase_fraction": "global_bright_contrast_area_fraction",
    "alpha_phase_fraction_field": "bright_contrast_area_fraction_field",
}
_CONTRAST_SIGNED_KEYS = frozenset((
    "run_spec", "identity", "measurement_identity", "generation_identity",
    "sampler", "source_spec"))
_MISSING_CONTRAST_VALUE = object()


def _same_contrast_value(a, b):
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _read_contrast_alias(mapping, canonical, legacy, default=_MISSING_CONTRAST_VALUE):
    if canonical in mapping:
        value = mapping[canonical]
        if legacy in mapping and not _same_contrast_value(value, mapping[legacy]):
            raise ValueError(f"Conflicting contrast aliases: {canonical} and {legacy}.")
        return value
    if legacy in mapping:
        return mapping[legacy]
    if default is not _MISSING_CONTRAST_VALUE:
        return default
    raise KeyError(f"Missing {canonical} (legacy alias: {legacy}).")


def _legacy_contrast_view(value):
    """Read canonical/legacy reports into the existing numerical interface.

    Signed identity subtrees are opaque: never rename, add or remove their keys.
    Dual keys must agree exactly. Values and width units are never recalculated.
    """
    if isinstance(value, list):
        return [_legacy_contrast_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: item if key in _CONTRAST_SIGNED_KEYS else _legacy_contrast_view(item)
           for key, item in value.items() if key != "contrast_terminology"}
    # Two old spellings can share a canonical name; restore both for callers.
    for legacy, canonical in CONTRAST_FIELD_ALIASES.items():
        if canonical in out:
            _read_contrast_alias(out, canonical, legacy)
            out[legacy] = out[canonical]
    for canonical in set(CONTRAST_FIELD_ALIASES.values()):
        out.pop(canonical, None)
    if out.get("feature") == "dark_contrast_area_fraction":
        out["feature"] = "beta_phase_fraction"
    elif out.get("feature") == "bright_contrast_area_fraction":
        out["feature"] = "alpha_phase_fraction"
    if "legacy_feature" in out:
        if out.get("feature") != out["legacy_feature"]:
            raise ValueError("Conflicting exported feature and legacy_feature.")
        out.pop("legacy_feature")
    return out


def _export_contrast_metadata(value):
    """Expose canonical names plus legacy aliases, without mutating the input.

    No measurement is introduced here. Cache identity objects remain unchanged.
    """
    if isinstance(value, (list, tuple)):
        return [_export_contrast_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: item if key in _CONTRAST_SIGNED_KEYS else _export_contrast_metadata(item)
           for key, item in value.items()}
    for legacy, canonical in CONTRAST_FIELD_ALIASES.items():
        if legacy in out or canonical in out:
            out[canonical] = _read_contrast_alias(out, canonical, legacy)
    feature = out.get("feature")
    if feature in ("beta_phase_fraction", "alpha_phase_fraction"):
        out["legacy_feature"] = feature
        out["feature"] = CONTRAST_FIELD_ALIASES[feature]
    if "candidates" in out or "mode" in out:
        out["contrast_terminology"] = dict(CONTRAST_TERMINOLOGY)
    return out


def _contrast_json_load(handle, **kwargs):
    return _legacy_contrast_view(json.load(handle, **kwargs))


def _contrast_json_loads(value, **kwargs):
    return _legacy_contrast_view(json.loads(value, **kwargs))


def _builtin_dark_fraction():
    configured = {}
    if BUILTIN_DARK_CONTRAST_AREA_FRACTION is not None:
        configured["dark"] = BUILTIN_DARK_CONTRAST_AREA_FRACTION
    if BUILTIN_BETA_FRACTION is not None:
        configured["legacy"] = BUILTIN_BETA_FRACTION
    return _read_contrast_alias(configured, "dark", "legacy", None)


def _physical_archive_keys(archive, include_labels=True):
    dark = "dark_contrast_area_fraction_field"
    legacy = "beta_phase_fraction_field"
    _read_contrast_alias(archive, dark, legacy)  # Includes conflict detection.
    dark_key = dark if dark in archive else legacy
    keys = (("class_id_map",) if include_labels else ()) + (dark_key, "mean_lath_width_um_field")
    if not all(key in archive for key in keys):
        raise ValueError("Missing physical PSIM fields/final hard map.")
    return keys


def _write_rows_csv(rows, out_path):
    if not rows:
        return
    scalar_rows = []
    keys = set()
    for row in _export_contrast_metadata(rows):
        out = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple, np.ndarray, dict)):
                out[key] = np.asarray(value).reshape(-1).tolist() if not isinstance(value, dict) else str(value)
            else:
                out[key] = value
            keys.add(key)
        scalar_rows.append(out)
    keys = sorted(keys)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)


def load_empirical_large_field_recipes(model):
    """Read calibrated cached summaries; invalid records do not erase good ones."""
    checkpoint = getattr(model, "large_field_full", None) or {}
    config = checkpoint.get("config", {})
    try:
        manifest_path = _large_field_training_manifest_path(model)
        with open(manifest_path, "r", encoding="utf-8-sig") as handle:
            manifest = _contrast_json_load(handle)
        if not isinstance(manifest, list):
            raise ValueError("training manifest is not a list")
        expected_calibration = float(config.get("microns_per_source_pixel", float("nan")))
        if not np.isfinite(expected_calibration) or expected_calibration <= 0:
            raise ValueError("checkpoint physical pixel calibration is missing or invalid")
    except (OSError, ValueError, TypeError) as exc:
        return {}, f"training summary catalogue unavailable: {exc}"

    candidates = {cid: [] for cid in range(N_CLASSES)}
    skipped = []
    for index, entry in enumerate(manifest):
        try:
            if not isinstance(entry, dict) or not entry.get("summary"):
                raise ValueError("manifest entry lacks a summary path")
            summary_path = os.path.expanduser(str(entry["summary"]))
            if not os.path.isabs(summary_path):
                summary_path = os.path.join(os.path.dirname(manifest_path), summary_path)
            summary_path = os.path.abspath(summary_path)
            with open(summary_path, "r", encoding="utf-8-sig") as handle:
                summary = _contrast_json_load(handle)
            if not isinstance(summary, dict):
                raise ValueError("summary must be a dictionary")
            hard = summary.get("hard_global_phase_fraction", {})
            physical = summary.get("quantitative_morphology", {})
            fractions = np.asarray([hard.get(name, np.nan) for name in CLASS_NAMES], dtype=np.float64)
            fp = np.asarray([_read_contrast_alias(physical, "global_dark_contrast_area_fraction", "global_beta_phase_fraction", np.nan),
                             physical.get("global_mean_lath_width_um", np.nan)], dtype=np.float64)
            calibration = float(physical.get("microns_per_source_pixel", float("nan")))
            if (not np.isfinite(fractions).all() or np.any(fractions < 0)
                    or not np.isclose(fractions.sum(), 1.0, atol=1e-3)
                    or not np.isfinite(fp).all() or not 0 <= fp[0] <= 1 or fp[1] <= 0):
                raise ValueError("invalid hard fractions or physical descriptors")
            if not np.isfinite(calibration) or not np.isclose(calibration, expected_calibration, rtol=1e-5, atol=0):
                raise ValueError("summary physical pixel calibration differs from training")
            fractions = fractions / fractions.sum()
            cid = int(np.argmax(fractions))
            candidates[cid].append((float(fractions[cid]), fp, fractions, summary_path))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            skipped.append(f"record {index}: {exc}")
            continue

    selected = {}
    for cid, rows in candidates.items():
        if not rows:
            continue
        purity_floor = max(0.5, float(np.quantile([row[0] for row in rows], 0.75)))
        eligible = [row for row in rows if row[0] >= purity_floor]
        if not eligible:
            skipped.append(f"{CLASS_NAMES[cid]}: no representative with at least 50% hard area")
            continue
        physical_values = np.asarray([row[1] for row in eligible])
        centre = np.median(physical_values, axis=0)
        spread = np.maximum(np.std(physical_values, axis=0), [0.01, 0.05])
        _, fp, fractions, summary_path = min(
            eligible, key=lambda row: float(np.linalg.norm((row[1] - centre) / spread))
        )
        selected[cid] = (fp.astype(np.float32), fractions.astype(np.float32), summary_path)
    audit = f"{sum(map(len, candidates.values()))} calibrated training summaries from {manifest_path}"
    if skipped:
        audit += f"; {len(skipped)} skipped record/class issue(s); first: {skipped[0]}"
    return selected, audit


def build_reference_free_conditions(model, class_id=None, class_fractions=None,
                                    beta_fraction=None, lath_width_um=None, *,
                                    dark_contrast_area_fraction=None):
    """Use measured defaults or an explicit recipe; first descriptor is dark contrast.

    beta_fraction is a compatibility alias, with identical fraction units.
    """
    if dark_contrast_area_fraction is not None:
        supplied = {"dark": dark_contrast_area_fraction}
        if beta_fraction is not None:
            supplied["legacy"] = beta_fraction
        beta_fraction = _read_contrast_alias(supplied, "dark", "legacy")
    if class_id is not None:
        try:
            integer_id = int(class_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Target class must be an integer in 0..3.") from exc
        if integer_id != class_id or not 0 <= integer_id < N_CLASSES:
            raise ValueError(f"Target class must be an integer between 0 and {N_CLASSES - 1}.")
        class_id = integer_id
    if class_id is not None and class_fractions is not None:
        raise ValueError("Choose --target-class or --class-fractions, not both.")
    fp_mean, fp_std = _validate_fingerprint_stats(model.fp_mean, model.fp_std, "large-field checkpoint")
    if beta_fraction is not None and (not np.isfinite(beta_fraction) or not 0 <= beta_fraction <= 1):
        raise ValueError("Target Dark-contrast area fraction must be finite and in 0..1.")
    if lath_width_um is not None and (not np.isfinite(lath_width_um) or lath_width_um <= 0):
        raise ValueError("Target α-lath width must be finite and positive in micrometres.")
    full_physical_recipe = beta_fraction is not None and lath_width_um is not None
    if class_fractions is not None:
        fractions = np.asarray(class_fractions, dtype=np.float32)
        if (fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-5)):
            raise ValueError("--class-fractions must give four nonnegative finite values that sum to one.")
        reference_id = None
        empirical, audit = {}, "complete explicit mixture; no measured defaults required"
        if not full_physical_recipe:
            empirical, audit = load_empirical_large_field_recipes(model)
            reference_id = int(np.argmax(fractions))
            if reference_id not in empirical:
                raise ValueError(
                    "Ratio-only generation needs a calibrated cached PSIM representative for "
                    + CLASS_NAMES[reference_id] + ". Restore its training summary or set "
                    "BUILTIN_DARK_CONTRAST_AREA_FRACTION and BUILTIN_LATH_WIDTH_UM explicitly. " + audit)
            print("[Generation] Class fractions remain the user target. Missing physical conditions "
                  "reuse the cached dominant-class representative: " + CLASS_NAMES[reference_id]
                  + "; this is an auxiliary reference, not measured physics of the requested mixture.")
        recipes = [("class_mixture", reference_id, fractions / fractions.sum())]
    elif class_id is not None and full_physical_recipe:
        empirical, audit = {}, "complete explicit class recipe; no measured defaults required"
        recipes = [(CLASS_NAMES[class_id], class_id, np.eye(N_CLASSES, dtype=np.float32)[class_id])]
    else:
        empirical, audit = load_empirical_large_field_recipes(model)
        ids = list(range(N_CLASSES)) if class_id is None else [class_id]
        missing = [CLASS_NAMES[cid] for cid in ids if cid not in empirical]
        if missing:
            raise ValueError(
                "No calibrated empirical recipe for " + ", ".join(missing) + ". "
                "Restore the mapped training summaries, or supply an explicit target class/mixture "
                "together with Dark-contrast area fraction and α-lath width. Global-mean fallback is disabled. " + audit
            )
        recipes = [(CLASS_NAMES[cid], cid, empirical[cid][1]) for cid in ids]
    print(f"[Generation] Recipe catalogue: {audit}; measured classes: {sorted(empirical)}")
    result = []
    for name, cid, fractions in recipes:
        measured = empirical.get(cid) if cid is not None else None
        fp_raw = np.asarray([
            measured[0][0] if beta_fraction is None else beta_fraction,
            measured[0][1] if lath_width_um is None else lath_width_um,
        ], dtype=np.float32)
        if not np.isfinite(fp_raw).all() or not 0 <= fp_raw[0] <= 1 or fp_raw[1] <= 0:
            raise ValueError("Generation requires Dark-contrast area fraction in [0, 1] and positive α-lath width in micrometres.")
        fp_norm = normalize_fp(fp_raw, fp_mean, fp_std).astype(np.float32)
        if np.any(np.abs(fp_norm) > 3):
            print(f"[Generation] {name}: descriptor is over 3 training standard deviations from the mean.")
        result.append({
            "recipe_name": name, "hard_fraction": np.asarray(fractions, dtype=np.float32),
            "spatial_class_map_strategy": "target_area_partition_v1",
            "fingerprint": fp_norm, "fingerprint_raw": fp_raw,
            "beta_source": ("dominant_class_cached_PSIM_auxiliary_reference" if class_fractions is not None
                            else "training_PSIM_summary") if beta_fraction is None else "user_input",
            "width_source": ("dominant_class_cached_PSIM_auxiliary_reference" if class_fractions is not None
                             else "training_PSIM_summary") if lath_width_um is None else "user_input",
            "source_summary": measured[2] if measured else None,
            "target_class_id": int(np.argmax(fractions)),
        })
    return result


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path, value):
    temporary = path + f".tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(_export_contrast_metadata(value), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cached_candidate_image(row, expected_path, expected_name, width, height):
    if (not isinstance(row, dict) or row.get("candidate_name") != expected_name
            or os.path.abspath(str(row.get("image_path", ""))) != os.path.abspath(expected_path)
            or not isinstance(row.get("pixel_sha256"), str)
            or not os.path.isfile(expected_path)):
        return None
    try:
        with Image.open(expected_path) as source:
            if source.size != (width, height):
                return None
            image = source.convert("RGB")
            image.load()
        pixels = np.asarray(image, dtype=np.uint8)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]:
            return None
        return image
    except (OSError, ValueError):
        return None


def find_completed_legacy_layout(base_dir, requested_spec):
    """Reuse ONLY validated complete old stripes, differing solely in layout.

    Never borrow a different seed/checkpoint/physics/base/schedule. Return the
    original manifest and directory so its masks, targets and PSIM caches retain
    their original identities. No filename-only or orphan-PNG adoption.
    """
    wanted = copy.deepcopy(requested_spec)
    guidance = wanted.get("sampler", {}).get("regional_guidance", {})
    if guidance.get("layout") != regional_layout_identity():
        return None
    guidance.pop("layout")
    if guidance.get("revision") != "whole_canvas_class_predictions_v1":
        return None
    wanted_json = json.dumps(wanted, sort_keys=True, separators=(",", ":"), allow_nan=False)
    matches = []
    for path in sorted(Path(base_dir).glob("run_*/generation_manifest.json")):
        try:
            old = _contrast_json_loads(path.read_text(encoding="utf-8"))
            spec = old.get("run_spec")
            if (old.get("mode") != "reference_free_physical_conditioned_generation"
                    or old.get("class_names") != list(CLASS_NAMES) or not isinstance(spec, dict)):
                continue
            serialized = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if (serialized != wanted_json or hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                    != old.get("run_signature")):
                continue
            conditions, seeds = spec["conditions"], spec["seeds"]
            rows = old.get("candidates", [])
            expected_names = [f"{i+1:02d}_{c['recipe_name']}_seed{seed}"
                              for i, c in enumerate(conditions) for seed in seeds]
            if (len(rows) != len(expected_names) or
                    sorted(r.get("candidate_name", "") for r in rows) != sorted(expected_names)):
                continue
            by_name = {r["candidate_name"]: r for r in rows}
            valid = True
            for i, condition in enumerate(conditions):
                for seed in seeds:
                    name = f"{i+1:02d}_{condition['recipe_name']}_seed{seed}"
                    row = by_name[name]
                    if (row.get("seed") != seed or
                            not np.allclose(row["conditioning_class_fractions"], condition["hard_fraction"], atol=1e-7)
                            or not np.isclose(row["target_beta_fraction"], condition["fingerprint_raw"][0], atol=1e-7)
                            or not np.isclose(row["target_mean_lath_width_um"], condition["fingerprint_raw"][1], atol=1e-7)
                            or not np.allclose(row["target_class_fractions"],
                                condition.get("requested_class_fractions", condition["hard_fraction"]), atol=1e-7)
                            or _cached_candidate_image(row, str(path.parent / (name+".png")), name,
                                spec["output_width"], spec["output_height"]) is None):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                matches.append((path, old))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    if not matches:
        return None
    # Prefer the oldest saved complete run deterministically (normally one).
    chosen = min(matches, key=lambda item: (item[0].stat().st_mtime_ns, str(item[0])))
    print(f"[Resume] Preserving completed legacy-layout image(s): {chosen[0].parent}. "
          "Original layout and targets retained; missing analysis will continue.", flush=True)
    return chosen


def run_composition_series(model, cli_args):
    """Hold physical conditions fixed, vary composition, and use a common seed for all targets."""
    fractions = np.asarray(BUILTIN_COMPOSITION_SERIES, dtype=np.float32)
    if (fractions.ndim != 2 or fractions.shape[1] != N_CLASSES or not len(fractions)
            or not np.isfinite(fractions).all() or np.any(fractions < 0)
            or not np.allclose(fractions.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("BUILTIN_COMPOSITION_SERIES must contain finite nonnegative four-class vectors summing to 1.")
    if len(np.unique(fractions, axis=0)) != len(fractions):
        raise ValueError("Duplicate composition targets would reuse identical recipe/seed images.")
    if (not isinstance(COMPOSITION_FEEDBACK_ROUNDS, int) or COMPOSITION_FEEDBACK_ROUNDS < 0
            or not np.isfinite(COMPOSITION_FEEDBACK_GAIN) or not 0 < COMPOSITION_FEEDBACK_GAIN <= 1):
        raise ValueError("Feedback rounds must be a nonnegative integer and gain in (0, 1].")
    # Resolve the fixed auxiliary physics ONCE using the requested anchor recipe.
    anchor, = build_reference_free_conditions(model, class_fractions=cli_args.class_fractions,
        beta_fraction=cli_args.beta_fraction, lath_width_um=cli_args.lath_width_um)
    count = int(cli_args.samples_per_class)
    seed_start = int(cli_args.generation_seed)
    if count < 1 or seed_start < 0 or seed_start + count - 1 >= 2**32:
        raise ValueError("Series count/seeds must be positive and within the uint32 seed range.")
    model._generation_checkpoint_digest = _file_sha256(model.large_field_checkpoint_path)
    identity = {"class_fractions": fractions.tolist(), "anchor": _json_ready(anchor),
                "condition_field_policy": FULL_FIELD_CONDITION_POLICY,
                "feedback_rounds": COMPOSITION_FEEDBACK_ROUNDS, "feedback_gain": COMPOSITION_FEEDBACK_GAIN,
                "seeds": list(range(seed_start, seed_start + count)),
                "checkpoint": model._generation_checkpoint_digest, "sampler": sampler_run_identity(model),
                "width": cli_args.field_width, "height": cli_args.field_height}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    root = Path(cli_args.generation_output_dir or OUTPUT_DIR).expanduser().resolve()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") if cli_args.new_run else signature[:20]
    directory = root / "composition_series" / ("run_" + suffix)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "series_manifest.json"
    series = {"mode": "composition_series_v1", "status": "generation_running", "identity": identity,
              "expected_candidates": len(fractions) * count, "runs": [],
              "physical_policy": "same Dark-contrast area fraction and α-lath width for all composition levels; auxiliary provenance in anchor",
              "replicate_definition": "one common seed per composition; not replicated specimens"}
    print(f"[Series] {len(fractions)} targets x {count} image(s) each = {series['expected_candidates']} whole images.")
    print(f"[Series] Fixed Dark-contrast area fraction={anchor['fingerprint_raw'][0]:.6g}, α-lath width={anchor['fingerprint_raw'][1]:.6g} um.")
    _write_json_atomic(str(path), series)
    try:
        for values in fractions:
            condition = copy.deepcopy(anchor)
            condition["hard_fraction"] = values.copy()
            condition["spatial_class_map_strategy"] = "target_area_partition_v1"
            condition["target_class_id"] = int(np.argmax(values))
            for seed in identity["seeds"]:
                args = copy.copy(cli_args)
                args._resolved_generation_conditions = [condition]
                args.samples_per_class, args.generation_seed = 1, seed
                result = run_reference_free_large_field_generation(model, args)
                saved = _contrast_json_loads((Path(result) / "generation_manifest.json").read_text(encoding="utf-8"))
                actual_layout = saved["run_spec"].get("sampler", {}).get("regional_guidance", {}).get(
                    "layout", {"revision": "legacy_wavy_stripe_grid_v1"})
                series["runs"].append({"target_class_fractions": values.tolist(), "seed": seed,
                                       "generation_dir": result, "actual_layout": actual_layout,
                                       "layout_reused_from_legacy": actual_layout.get("revision") == "legacy_wavy_stripe_grid_v1"})
                _write_json_atomic(str(path), series)
        series["status"] = "generation_complete"
    except Exception as error:
        series.update(status="generation_incomplete", error=str(error))
        raise
    finally:
        _write_json_atomic(str(path), series)
        _write_json_atomic(str(root / "latest_series_manifest.json"), series)
    return str(path), [r["generation_dir"] for r in series["runs"]]


def run_reference_free_large_field_generation(model, cli_args):
    """Paired-condition sampling from noise; no reference editing or proxy PSIM."""
    if not model.large_field_available:
        raise RuntimeError("Reference-free generation requires a trained large_field_model.pt; no image-edit fallback is used.")
    field_width = int(getattr(cli_args, "field_width", model.large_field_resolution))
    field_height = int(getattr(cli_args, "field_height", model.large_field_resolution))
    validate_native_resolution(model.large_field_resolution, field_width, field_height)
    field_stride = int(getattr(cli_args, "field_stride", globals().get("DEFAULT_FULL_FIELD_STRIDE", 48)))
    use_native_recipes = all(value is None for value in (
        cli_args.target_class, cli_args.class_fractions, cli_args.beta_fraction, cli_args.lath_width_um))
    resolved = getattr(cli_args, "_resolved_generation_conditions", None)
    if resolved is not None:
        conditions = copy.deepcopy(resolved)
        use_native_recipes = False
    elif use_native_recipes:
        sources = getattr(model, "_native_recipe_sources", None)
        if sources is None:
            sources = load_large_field_matrix_sources(model, source_images=[None] * N_CLASSES, source_crops=[None] * N_CLASSES)
            model._native_recipe_sources = sources
        missing = [CLASS_NAMES[cid] for cid in range(N_CLASSES) if cid not in sources]
        if missing:
            raise ValueError("Native generation recipes lack mapped source crops for: " + ", ".join(missing))
        direct_reference = all(s.get("physics_policy") == "fixed_reference" for s in sources.values())
        conditions = []
        for cid in range(N_CLASSES):
            source = sources[cid]
            # Copy numerical conditions only: source pixels are not encoded,
            # pasted into candidates, or supplied to the noise sampler.
            condition = dict(source["condition"])
            condition.update({
                "recipe_name": CLASS_NAMES[cid],
                "beta_source": source.get("physics_origin", "training_PSIM_native_crop"),
                "width_source": source.get("physics_origin", "training_PSIM_native_crop"),
                "source_summary": source.get("summary_path"),
                "source_image_path": source.get("image_path"),
                "source_archive_path": source.get("archive_path"),
                "source_crop_xy": source.get("crop_xy"),
            })
            conditions.append(condition)
        if direct_reference:
            use_native_recipes = False  # Uniform assumed recipes, no archive or native measured fields.
    else:
        conditions = build_reference_free_conditions(
            model, class_id=cli_args.target_class, class_fractions=cli_args.class_fractions,
            beta_fraction=cli_args.beta_fraction, lath_width_um=cli_args.lath_width_um,
        )
    conditions = [prepare_canvas_recipe(c, model.large_field_resolution, field_width, field_height)
                  for c in conditions]
    extrapolated = (field_width, field_height) != (model.large_field_resolution, model.large_field_resolution)
    if extrapolated:
        use_native_recipes = False  # Numeric means only; native crop layouts are not stretched.
    count = int(cli_args.samples_per_class)
    if count < 1:
        raise ValueError("--samples-per-class must be positive.")
    seeds = [int(cli_args.generation_seed) + k for k in range(count)]
    root_dir = os.path.abspath(os.path.expanduser(cli_args.generation_output_dir or OUTPUT_DIR))
    base_dir = os.path.join(root_dir, "reference_free_large_field")
    checkpoint_path = os.path.abspath(model.large_field_checkpoint_path)
    if getattr(model, "_generation_checkpoint_digest", None) is None:
        model._generation_checkpoint_digest = _file_sha256(checkpoint_path)
    run_spec = {
        "cache_version": GENERATION_CACHE_VERSION,
        "condition_field_policy": FULL_FIELD_CONDITION_POLICY,
        "sampler": sampler_run_identity(model),
        "checkpoint_sha256": model._generation_checkpoint_digest,
        "checkpoint_step": int(model.large_field_full.get("step", -1)),
        "conditions": _json_ready(conditions),
        "seeds": seeds,
        "output_width": field_width,
        "output_height": field_height,
        "spatial_sampling_policy": "one_complete_rectangular_unet_canvas_no_window_fusion",
        "sampling_steps": int(LARGE_FIELD_NUM_INFERENCE_STEPS),
        "guidance_start": float(LARGE_FIELD_GUIDANCE_START),
        "guidance_end": float(LARGE_FIELD_GUIDANCE_END),
    }
    signature = hashlib.sha256(json.dumps(
        run_spec, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    run_name = ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                if cli_args.new_run else "run_" + signature[:20])
    out_dir = os.path.join(base_dir, run_name)
    legacy_reused = False
    if (not cli_args.new_run and REUSE_COMPLETED_LEGACY_LAYOUT
            and not os.path.isfile(os.path.join(out_dir, "generation_manifest.json"))):
        legacy = find_completed_legacy_layout(base_dir, run_spec)
        if legacy is not None:
            legacy_path, legacy_manifest = legacy
            out_dir = str(legacy_path.parent)
            run_spec = copy.deepcopy(legacy_manifest["run_spec"])
            signature = legacy_manifest["run_signature"]
            legacy_reused = True
    ensure_dir(out_dir)
    manifest_path = os.path.join(out_dir, "generation_manifest.json")
    previous = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous = _contrast_json_load(handle)
        if previous.get("run_signature") != signature:
            raise ValueError(f"Existing generation manifest does not match this run: {manifest_path}")
    previous_rows = {row.get("candidate_name"): row for row in previous.get("candidates", [])
                     if isinstance(row, dict)} if previous else {}
    if previous and previous.get("status") in ("generation_complete", "generation_complete_measurement_pending", "generation_complete_measured"):
        all_valid = all(
            _cached_candidate_image(
                previous_rows.get(f"{index + 1:02d}_{condition['recipe_name']}_seed{seed}"),
                os.path.join(out_dir, f"{index + 1:02d}_{condition['recipe_name']}_seed{seed}.png"),
                f"{index + 1:02d}_{condition['recipe_name']}_seed{seed}",
                field_width, field_height,
            ) is not None
            for index, condition in enumerate(conditions) for seed in seeds
        )
        derived = ("generation_targets.csv", "raw_candidates_grid.png",
                   "generation_condition_summary.json")
        if all_valid and all(os.path.isfile(os.path.join(out_dir, name)) for name in derived):
            previous["status"] = "generation_complete"
            previous["post_generation_verification"] = "disabled"
            _write_json_atomic(manifest_path, previous)
            _write_rows_csv(previous.get("candidates", []), os.path.join(out_dir, "generation_targets.csv"))
            condition_path = os.path.join(out_dir, "generation_condition_summary.json")
            with open(condition_path, "r", encoding="utf-8") as handle:
                saved_conditions = _contrast_json_load(handle)
            _write_json_atomic(condition_path, saved_conditions)
            print(f"[Generation] All candidate images already complete: {out_dir}", flush=True)
            return out_dir
    manifest = {
        "mode": "reference_free_physical_conditioned_generation",
        "run_signature": signature,
        "run_spec": run_spec,
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(model.large_field_full.get("step", -1)),
        "post_generation_verification": "disabled",
        "pretrained_backbone": str(model.pretrained_model_path),
        "microns_per_source_pixel": model.large_field_full.get("config", {}).get("microns_per_source_pixel"),
        "class_names": list(CLASS_NAMES),
        "resolution": int(model.large_field_resolution),
        "output_width": field_width,
        "output_height": field_height,
        "spatial_sampling_policy": "one_complete_rectangular_unet_canvas_no_window_fusion",
        "native_window": int(model.large_field_resolution),
        "output_scope": "size_extrapolation_from1536" if extrapolated else "native1536_training_crop",
        "training_resolution_px": int(model.large_field_resolution),
        "full_canvas_extrapolation": extrapolated,
        "latent_shape": [1, model.unet.config.in_channels, field_height // 8, field_width // 8],
        "paired_seeds": seeds,
        "sampling_steps": int(LARGE_FIELD_NUM_INFERENCE_STEPS),
        "guidance_start": float(LARGE_FIELD_GUIDANCE_START),
        "guidance_end": float(LARGE_FIELD_GUIDANCE_END),
        "spatial_scale": float(model.large_field_spatial_scale),
        "conditioning": ("native-pixel measured PSIM fields and retained-pixel class maps; reserved connectivity channels zero"
                         if use_native_recipes else
                         "explicit recipes use target-area spatial partitions; measured partial defaults retain summary provenance; reserved connectivity channels zero"),
        "condition_source_scope": ("native_measured_region" if use_native_recipes else "explicit_or_summary_backed_custom_recipe"),
        "condition_field_policy": FULL_FIELD_CONDITION_POLICY,
        "condition_field_interpretation": "Explicit target partitions span the full latent canvas; training-grid maps are their area averages. Cached source crop layouts are not stretched. Input quotas are not measured output areas.",
        "reference_images_used": False,
        "reference_image_pixels_encoded": False,
        "postprocessing": "none; VAE decode at requested dimensions, no resize, reference texture transfer, or repeated tiles",
        "consensus": "none; one complete latent canvas",
        "measurement_status": "not_requested",
        "interpretation": "Class proportions are conditioning targets, not measured output fractions. "
                          "This generation-only build does not evaluate target agreement.",
        "candidates": list(previous_rows.values()),
        "status": "running",
    }
    if legacy_reused:
        # The old image is never relabelled as an irregular-layout result.
        manifest = copy.deepcopy(previous)
        manifest["status"] = "running"
    _write_json_atomic(manifest_path, manifest)
    records, images = [], []
    print(f"[Generation] {len(conditions)} target recipe(s) x {count} seed(s); same seeds paired across recipes.")
    print("[Generation] Conditions: " + ("measured native-pixel PSIM fields; random-noise input"
                                         if use_native_recipes else "custom recipes with explicit or recorded measured descriptors"))
    print("[Generation] Requested class ratios are inputs; generated-image measurement is disabled.")
    print(f"[Generation] Saving raw decoded fields to {out_dir}")
    for recipe_index, condition in enumerate(conditions):
        for seed in seeds:
            name = f"{recipe_index + 1:02d}_{condition['recipe_name']}_seed{seed}"
            image_path = os.path.join(out_dir, name + ".png")
            cached_row = previous_rows.get(name)
            image = _cached_candidate_image(cached_row, image_path, name, field_width, field_height)
            if image is not None:
                print(f"[Generation] Reusing complete candidate: {image_path}", flush=True)
                records.append(cached_row)
                images.append(image)
                continue
            image, local_class_probabilities, local_fingerprints = large_field_sampling(
                model, condition, {}, seed=seed, name=name,
                output_width=field_width, output_height=field_height,
                window_stride=field_stride, window_batch_size=cli_args.window_batch_size,
            )
            if image.size != (field_width, field_height):
                raise ValueError(f"Sampler returned {image.size}; expected {(field_width, field_height)}.")
            temporary_image = image_path + ".partial.png"
            save_pil(image, temporary_image)
            os.replace(temporary_image, image_path)
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            row = {
                "candidate_name": name,
                "recipe": condition["recipe_name"], "seed": seed, "image_path": image_path,
                "target_beta_fraction": float(condition["fingerprint_raw"][0]),
                "target_mean_lath_width_um": float(condition["fingerprint_raw"][1]),
                "target_fingerprint_normalized": condition["fingerprint"].tolist(),
                "target_class_fractions": np.asarray(condition.get(
                    "requested_class_fractions", condition["hard_fraction"])).tolist(),
                "conditioning_class_fractions": condition["hard_fraction"].tolist(),
                "spatial_class_map_strategy": condition.get("spatial_class_map_strategy", "measured_or_uniform"),
                "beta_source": condition["beta_source"], "width_source": condition["width_source"],
                "source_summary": condition["source_summary"],
                "source_image_path": condition.get("source_image_path"),
                "source_archive_path": condition.get("source_archive_path"),
                "source_crop_xy": condition.get("source_crop_xy"),
                "spatial_recipe_scope": condition.get("spatial_recipe_scope"),
                "width_px": image.width, "height_px": image.height,
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "pixel_mean_0_255": float(pixels.mean()), "pixel_std_0_255": float(pixels.std()),
                "measurement_status": "not_requested",
            }
            if local_class_probabilities is not None:
                row["target_local_class_probabilities"] = np.asarray(local_class_probabilities).tolist()
            if local_fingerprints is not None:
                row["target_local_fingerprints_normalized"] = np.asarray(local_fingerprints).tolist()
            if condition.get("class_valid_pixel_counts") is not None:
                row["target_class_valid_pixel_counts"] = np.asarray(condition["class_valid_pixel_counts"]).tolist()
            row.update(getattr(model, "last_large_field_diagnostics", {}))
            routing_weights = getattr(model, "last_regional_target_weights", None)
            if routing_weights is not None:
                row.update(export_regional_target_weights(row["image_path"], routing_weights))
            records.append(row)
            images.append(image)
            previous_rows[name] = row
            manifest["candidates"] = list(previous_rows.values())
            _write_json_atomic(manifest_path, manifest)
            _write_rows_csv(records, os.path.join(out_dir, "generation_targets.csv"))
    manifest["candidates"] = records
    _write_rows_csv(records, os.path.join(out_dir, "generation_targets.csv"))
    _write_generation_condition_summary(conditions, out_dir)
    fig, axes = plt.subplots(len(conditions), count, squeeze=False,
                             figsize=(3.4 * count, 3.65 * len(conditions)), dpi=160)
    for ax, image, row in zip(axes.flat, images, records):
        ax.imshow(image)
        ax.set_title(f"Target: {row['recipe'].replace('_', ' ')}\nSeed {row['seed']}", fontsize=9, pad=5)
        ax.axis("off")
    fig.suptitle("Fields generated from random noise | target fractions are inputs", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "raw_candidates_grid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    manifest["status"] = "generation_complete"
    manifest["post_generation_verification"] = "disabled"
    _write_json_atomic(manifest_path, manifest)
    print(f"[Saved] {out_dir}")
    return out_dir


def _write_generation_condition_summary(conditions, out_dir):
    """Keep machine-readable input provenance; do not render the removed audit figure."""
    summary = [{"recipe": condition["recipe_name"],
                "target_class_fractions": np.asarray(condition.get(
                    "requested_class_fractions", condition["hard_fraction"])).tolist(),
                "conditioning_class_fractions": np.asarray(condition["hard_fraction"]).tolist(),
                "spatial_class_map_strategy": condition.get("spatial_class_map_strategy", "measured_or_uniform"),
                "target_beta_fraction": float(condition["fingerprint_raw"][0]),
                "target_mean_lath_width_um": float(condition["fingerprint_raw"][1]),
                "beta_source": condition.get("beta_source"),
                "width_source": condition.get("width_source"),
                "source_summary": condition.get("source_summary")}
               for condition in conditions]
    with open(os.path.join(out_dir, "generation_condition_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_export_contrast_metadata(_json_ready(summary)), handle, ensure_ascii=False, indent=2)


def _large_field_training_manifest_path(model):
    """Resolve explicit checkpoint-adjacent/stored paths, never guess image mappings."""
    checkpoint = model.large_field_full
    config = checkpoint.get("config", {})
    checkpoint_path = getattr(model, "large_field_checkpoint_path", None) or LARGE_FIELD_MODEL_PATH
    candidates = [os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                               "psim_training_manifest.json")]
    if config.get("output_dir"):
        candidates.append(os.path.join(os.path.expanduser(str(config["output_dir"])),
                                       "psim_training_manifest.json"))
    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if os.path.isfile(candidate):
            expected_sha = checkpoint.get("psim_manifest_sha256")
            if expected_sha and _file_sha256(candidate) != expected_sha:
                raise ValueError(
                    f"Training PSIM manifest differs from the one recorded in large_field_model.pt: {candidate}"
                )
            return candidate
    raise FileNotFoundError(
        "Training PSIM manifest is required for the real-source matrix. Checked: "
        + ", ".join(checked)
        + ". Keep the training manifest beside large_field_model.pt; its image/archive paths "
          "must still identify the original files. No PSIM is rerun and no filename mapping is guessed."
    )


def _matrix_load_physical_fields(archive_path, min_ratio):
    """Mirror train_large_field.load_physical_psim_fields, including rare pixels."""
    with np.load(archive_path, allow_pickle=False) as archive:
        if ("class_names" not in archive
                or list(archive["class_names"].astype(str)) != list(CLASS_NAMES)):
            raise ValueError(f"Missing/mismatched four-class order: {archive_path}")
        keys = _physical_archive_keys(archive, include_labels=True)
        if not all(key in archive for key in keys):
            raise ValueError(f"Missing physical PSIM fields/final hard map: {archive_path}")
        labels, beta, width = (np.asarray(archive[key]) for key in keys)
    if (labels.ndim != 2 or labels.size == 0
            or beta.shape != labels.shape or width.shape != labels.shape):
        raise ValueError(f"PSIM field dimensions disagree: {archive_path}")
    if (not np.isfinite(labels).all() or not np.isin(labels, np.arange(N_CLASSES)).all()
            or not np.isfinite(beta).all() or not np.isfinite(width).all()
            or np.any((beta < 0) | (beta > 1)) or np.any(width < 0)):
        raise ValueError(f"Invalid PSIM class IDs or physical field values: {archive_path}")
    labels = labels.astype(np.uint8)
    counts = np.bincount(labels.ravel(), minlength=N_CLASSES) / labels.size
    for rejected in np.flatnonzero((counts > 0) & (counts < min_ratio)):
        labels[labels == rejected] = 255
    return labels, beta.astype(np.float32), width.astype(np.float32)


def _matrix_physical_condition(labels, beta, width, grid, fp_mean, fp_std, class_id):
    """Summarize physical pixels; native square windows reproduce training cells.

    Rectangular full-region summaries describe targets only. UNet window inputs
    are computed separately from native square PSIM slices, never from this grid.
    """
    if (labels.ndim != 2 or beta.shape != labels.shape or width.shape != labels.shape
            or grid < 1 or min(labels.shape) < grid
            or labels.shape[0] % grid or labels.shape[1] % grid):
        raise ValueError("Physical fields must share a 2-D shape divisible by the condition grid.")
    valid = labels[(labels >= 0) & (labels < N_CLASSES)]
    if not valid.size:
        raise ValueError("Matrix crop contains only excluded rare-class pixels.")
    fractions = np.bincount(valid.astype(np.int64), minlength=N_CLASSES).astype(np.float32) / valid.size
    raw_fp = np.asarray([beta.mean(), width.mean()], dtype=np.float32)
    if not np.isfinite(raw_fp).all() or not 0 <= raw_fp[0] <= 1 or raw_fp[1] <= 0:
        raise ValueError("Matrix crop requires finite Dark-contrast area fraction and positive physical α-lath width.")
    local_classes, local_fp, valid_counts = [], [], []
    y_edges = np.linspace(0, labels.shape[0], grid + 1, dtype=int)
    x_edges = np.linspace(0, labels.shape[1], grid + 1, dtype=int)
    for iy in range(grid):
        for ix in range(grid):
            ys, xs = slice(y_edges[iy], y_edges[iy + 1]), slice(x_edges[ix], x_edges[ix + 1])
            patch = labels[ys, xs]
            local = patch[(patch >= 0) & (patch < N_CLASSES)]
            local_classes.append(
                np.bincount(local.astype(np.int64), minlength=N_CLASSES).astype(np.float32) / local.size
                if local.size else fractions.copy()
            )
            valid_counts.append(int(local.size))
            local_fp.append([float(beta[ys, xs].mean()), float(width[ys, xs].mean())])
    return {
        "hard_fraction": fractions,
        "fingerprint": normalize_fp(raw_fp, fp_mean, fp_std).astype(np.float32),
        "fingerprint_raw": raw_fp,
        "class_probability_map": np.asarray(local_classes, dtype=np.float32),
        "class_valid_pixel_counts": np.asarray(valid_counts, dtype=np.int64),
        "fingerprint_map": normalize_fp(np.asarray(local_fp, dtype=np.float32), fp_mean, fp_std).astype(np.float32),
        "target_class_id": int(class_id),
    }


def _matrix_manifest_entries(model):
    manifest_path = _large_field_training_manifest_path(model)
    with open(manifest_path, "r", encoding="utf-8-sig") as handle:
        entries = _contrast_json_load(handle)
    if not isinstance(entries, list):
        raise ValueError("The training PSIM manifest must be a list.")
    resolved = []
    for index, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict) or not all(raw.get(key) for key in ("image", "archive", "summary")):
            print(f"[Sources] Skipping manifest entry {index}: missing image/archive/summary.")
            continue
        entry = dict(raw, manifest_entry_id=index)
        for key in ("image", "archive", "summary"):
            path = os.path.expanduser(str(entry[key]))
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(manifest_path), path)
            entry[key] = os.path.abspath(path)
        resolved.append(entry)
    if not resolved:
        raise ValueError("Training PSIM manifest has no usable image/archive/summary mappings.")
    return manifest_path, resolved


def _matrix_source_entry(entries, selector):
    """Resolve IDs/paths, or verify a copied image against recorded RGB pixels.

    Never reuse a cache by basename or visual similarity
    when an explicit path was requested. The original training mapping remains
    authoritative; copied files must match size and every decoded RGB pixel.
    """
    import hashlib
    import os
    from PIL import Image

    token = str(selector).strip()
    if not token:
        raise ValueError("Empty source image selector.")
    detail = ""
    match_method, requested_path, rgb_hash = "manifest_selector", None, None
    if token.lower().startswith("entry:"):
        number = token[6:]
        matches = [entry for entry in entries if number.isdecimal() and entry["manifest_entry_id"] == int(number)]
    elif os.path.isabs(token) or os.path.dirname(token):
        path = os.path.abspath(os.path.expanduser(token))
        matches = [entry for entry in entries if os.path.normcase(entry["image"]) == os.path.normcase(path)]
        if not matches and os.path.isfile(path):
            def pixel_identity(filename):
                with Image.open(filename) as image:
                    rgb = image.convert("RGB")
                    return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()

            try:
                requested_identity = pixel_identity(path)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot read requested source image {path!r}: {exc}") from exc
            print(f"[Sources] Path is not in the training manifest; checking exact RGB identity: {path}", flush=True)
            unreadable = 0
            for entry in entries:
                candidate = entry["image"]
                try:
                    stat = os.stat(candidate)
                    stamp = (stat.st_size, stat.st_mtime_ns)
                    cached = entry.get("_source_rgb_identity_cache")
                    if cached is None or cached[0] != stamp:
                        identity = pixel_identity(candidate)
                        entry["_source_rgb_identity_cache"] = (stamp, identity)
                    else:
                        identity = cached[1]
                except (OSError, ValueError):
                    unreadable += 1
                    continue
                if identity == requested_identity:
                    matches.append(entry)
            if len(matches) == 1:
                match = matches[0]
                match_method, requested_path, rgb_hash = "exact_rgb_copy", path, requested_identity[1]
                print(f"[Sources] Verified identical RGB pixels ({requested_identity[0][0]}x"
                      f"{requested_identity[0][1]}); requested {path} -> "
                      f"entry:{match['manifest_entry_id']} {match['image']}; "
                      f"RGB SHA256={requested_identity[1]}. Reusing its original PSIM mapping.", flush=True)
            detail = (f" Exact RGB comparison found {len(matches)} matching entries; "
                      f"{unreadable} training images could not be read. "
                      "Cropped, resized or recompressed images may have different pixels and cannot reuse this mapping.")
        elif not matches:
            detail = f" The requested file does not exist: {path}."
    else:
        # Exact filenames take priority; a bare integer means numeric stem,
        # never a position in the manifest. Leading zeroes are equivalent.
        matches = [entry for entry in entries if os.path.basename(entry["image"]).casefold() == token.casefold()]
        if not matches:
            def same_stem(entry):
                stem = os.path.splitext(os.path.basename(entry["image"]))[0]
                return (int(stem) == int(token) if stem.isdecimal() and token.isdecimal()
                        else stem.casefold() == token.casefold())
            matches = [entry for entry in entries if same_stem(entry)]
    if len(matches) != 1:
        found = ", ".join(entry["image"] for entry in matches[:6])
        error = ValueError(f"Source selector {token!r} matched {len(matches)} training entries. "
                         + detail + " Run this script with --list-source-images to export source_catalog.csv, "
                         "then choose a listed full path or entry:N. A new image requires its own PSIM fields "
                         "and verified mapping; an existing file path alone is insufficient. "
                         "Do not rename another image's cache or edit the checkpoint-hashed training manifest. " + found)
        error.match_count = len(matches)
        raise error
    matched = dict(matches[0], source_match_method=match_method, source_rgb_sha256=rgb_hash,
                   cache_image=matches[0]["image"])
    if requested_path is not None:
        matched["image"] = requested_path
    return matched


def _matrix_external_catalog(model):
    """Search configured result roots only; never pair numbered files by name."""
    if hasattr(model, "_source_external_catalog"):
        return model._source_external_catalog
    roots = list(globals().get("SOURCE_PSIM_ROOTS", []))
    saved_root = model.large_field_full.get("config", {}).get("psim_output_dir")
    if saved_root:
        roots.append(saved_root)
    summaries, diagnostics, visited = [], [], set()
    model._source_external_roots = []
    for raw in roots:
        root = os.path.expanduser(str(raw))
        if not os.path.isabs(root):
            root = os.path.join(PROJECT_ROOT, root)
        root = os.path.abspath(root)
        if os.path.normcase(root) in visited:
            continue
        visited.add(os.path.normcase(root))
        model._source_external_roots.append(root)
        if not os.path.isdir(root):
            diagnostics.append(f"Missing PSIM result directory: {root}")
            continue
        print(f"[Sources] Searching PSIM metadata outside the training manifest: {root}", flush=True)
        for directory, _, files in os.walk(root, onerror=lambda exc: diagnostics.append(str(exc))):
            if "PSIM_summary.json" not in files:
                continue
            path = os.path.join(directory, "PSIM_summary.json")
            try:
                with open(path, encoding="utf-8-sig") as handle:
                    summary = _contrast_json_load(handle)
                if not isinstance(summary, dict) or not summary.get("image_path"):
                    raise ValueError("Missing image_path provenance")
                summaries.append((path, summary))
            except (OSError, ValueError) as exc:
                diagnostics.append(f"{path}: {exc}")
    model._source_external_diagnostics = diagnostics
    model._source_external_catalog = summaries
    return summaries


def _matrix_external_entry(model, selector, summary_path=None):
    """Match a requested image to its own PSIM provenance, not to training membership."""
    token = str(selector)
    if not (os.path.isabs(token) or os.path.dirname(token)):
        raise ValueError("External-cache sources require an explicit image path, not a bare filename ID or entry:N.")
    requested = os.path.abspath(os.path.expanduser(token))
    if not os.path.isfile(requested):
        raise ValueError(f"Requested source does not exist: {requested}")
    if summary_path is not None:
        path = os.path.abspath(os.path.expanduser(str(summary_path)))
        with open(path, encoding="utf-8-sig") as handle:
            summary = _contrast_json_load(handle)
        if not isinstance(summary, dict) or not summary.get("image_path"):
            raise ValueError(f"PSIM summary requires image_path provenance: {path}")
        candidates = [(path, summary)]
    else:
        candidates = _matrix_external_catalog(model)
    with Image.open(requested) as image:
        rgb = image.convert("RGB")
        requested_identity = (rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest())
    identities = getattr(model, "_source_external_image_identities", {})
    model._source_external_image_identities = identities
    matches = []
    for path, summary in candidates:
        recorded = os.path.expanduser(str(summary["image_path"]))
        if not os.path.isabs(recorded):
            recorded = os.path.join(os.path.dirname(path), recorded)
        recorded = os.path.abspath(recorded)
        method = None
        if os.path.normcase(recorded) == os.path.normcase(requested):
            method = "external_summary_exact_path"
        elif os.path.isfile(recorded):
            try:
                stat = os.stat(recorded)
                key = (recorded, stat.st_size, stat.st_mtime_ns)
                if key not in identities:
                    with Image.open(recorded) as image:
                        rgb = image.convert("RGB")
                        identities[key] = (rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest())
                if identities[key] == requested_identity:
                    method = "external_summary_exact_rgb_copy"
            except (OSError, ValueError):
                continue
        if method:
            matches.append((path, recorded, method))
    # A root listed twice must not introduce artificial ambiguity.
    matches = list({os.path.normcase(path): (path, recorded, method)
                    for path, recorded, method in matches}.values())
    if len(matches) != 1:
        extra = "; ".join(path for path, _, _ in matches[:5])
        raise ValueError(f"External PSIM lookup found {len(matches)} verified source mappings for {requested}. "
                         "Set --source-psim-root to the existing result directory, or --source-summary CLASS PATH "
                         "to this image's PSIM_summary.json. Its image_path must identify this image or an "
                         "accessible pixel-identical original. If this image has no PSIM result yet, its "
                         "physical fields must be computed first. " + extra)
    path, recorded, method = matches[0]
    directory = os.path.dirname(path)
    exact = os.path.join(directory, "PSIM_results.npz")
    archives = ([exact] if os.path.isfile(exact) else
                [os.path.join(directory, name) for name in sorted(os.listdir(directory))
                 if name.startswith("PSIM_results_") and name.endswith(".npz")])
    if len(archives) != 1:
        raise ValueError(f"Expected one numerical PSIM archive beside {path}, found {len(archives)}. "
                         "Keep an unambiguous summary/archive pair; no newest-file guess is made.")
    print(f"[Sources] Verified external PSIM mapping: {requested} -> {path}; method={method}", flush=True)
    return {"image": requested, "cache_image": recorded, "archive": archives[0], "summary": path,
            "manifest_entry_id": None, "source_match_method": method,
            "source_rgb_sha256": requested_identity[1], "cache_origin": "external_psim_summary"}


def _matrix_resolve_source(model, entries, selector, cid):
    summaries = globals().get("SOURCE_PSIM_SUMMARIES", [None] * N_CLASSES)
    if len(summaries) != N_CLASSES:
        raise ValueError("SOURCE_PSIM_SUMMARIES must have four entries in BW/DC/W/ACW order.")
    if summaries[cid] is not None:
        return _matrix_external_entry(model, selector, summaries[cid])
    try:
        return _matrix_source_entry(entries, selector)
    except ValueError as exc:
        if getattr(exc, "match_count", None) != 0:
            raise
        if not (os.path.isabs(str(selector)) or os.path.dirname(str(selector))):
            raise
        try:
            return _matrix_external_entry(model, selector)
        except (OSError, ValueError) as external_error:
            raise ValueError(f"{exc}\nExternal-cache check: {external_error}") from external_error


def _matrix_labeled_source(model, entry, cid, selector, crop_xy):
    """One user-labelled image plus aligned physical fields; no PSIM class gate."""
    size, grid = int(model.large_field_resolution), int(model.large_field_patch_grid)
    expected_calibration = float(model.large_field_full["config"]["microns_per_source_pixel"])
    for key in ("image", "archive", "summary"):
        if not os.path.isfile(entry[key]):
            raise ValueError(f"Requested source has missing {key}: {entry[key]}")
    with open(entry["summary"], "r", encoding="utf-8-sig") as handle:
        summary = _contrast_json_load(handle)
    calibration = float(summary.get("quantitative_morphology", {}).get("microns_per_source_pixel", float("nan")))
    if not np.isfinite(calibration) or not np.isclose(calibration, expected_calibration, rtol=1e-5, atol=0):
        raise ValueError(f"Requested source physical pixel calibration differs from training: {entry['summary']}")
    with np.load(entry["archive"], allow_pickle=False) as archive:
        keys = _physical_archive_keys(archive, include_labels=False)
        if not all(key in archive for key in keys):
            raise ValueError(f"Missing physical PSIM fields: {entry['archive']}")
        beta, width = (np.asarray(archive[key], dtype=np.float32) for key in keys)
        if (beta.ndim != 2 or not beta.size or width.shape != beta.shape
                or not np.isfinite(beta).all() or not np.isfinite(width).all()
                or np.any((beta < 0) | (beta > 1)) or np.any(width < 0)):
            raise ValueError(f"Invalid physical PSIM fields: {entry['archive']}")
        # Optional diagnostic only. Missing/different PSIM class labels do not
        # veto a dataset-labelled source or change its declared class.
        cached_labels = None
        if ("class_id_map" in archive and "class_names" in archive
                and list(archive["class_names"].astype(str)) == list(CLASS_NAMES)):
            candidate = np.asarray(archive["class_id_map"])
            if (candidate.shape == beta.shape and np.isfinite(candidate).all()
                    and np.isin(candidate, np.arange(N_CLASSES)).all()):
                cached_labels = candidate
    height, image_width = beta.shape
    with Image.open(entry["image"]) as source:
        if source.size != (image_width, height):
            raise ValueError("PSIM/source dimensions differ; resizing is forbidden")
        if min(height, image_width) < size:
            raise ValueError(f"Requested source is smaller than the native {size}x{size} crop; "
                             "provide a native-size source instead of silently upscaling it.")
        if crop_xy is None:
            x, y = (image_width - size) // 2, (height - size) // 2
            crop_policy = "whole_native_image" if source.size == (size, size) else "center_crop"
        else:
            x, y = crop_xy
            crop_policy = "fixed_crop"
        if x + size > image_width or y + size > height:
            raise ValueError(f"Crop {(x, y)} with size {size} exceeds image {(image_width, height)}.")
        image = source.convert("RGB").crop((x, y, x + size, y + size))
    beta_crop = beta[y:y + size, x:x + size]
    width_crop = width[y:y + size, x:x + size]
    # Broadcast the supplied image-level label as the class-condition hypothesis,
    # not as a measured segmentation. Every pixel contributes to the physics.
    label_hypothesis = np.full((size, size), cid, dtype=np.uint8)
    condition = _matrix_physical_condition(label_hypothesis, beta_crop, width_crop,
                                           grid, model.fp_mean, model.fp_std, cid)
    fraction = (float(np.mean(cached_labels[y:y + size, x:x + size] == cid))
                if cached_labels is not None else None)
    print(f"[Source {cid}] dataset label={CLASS_NAMES[cid]}; image={entry['image']}; "
          f"crop=({x}, {y}, {size}, {size}); policy={crop_policy}; PSIM purity filter=disabled", flush=True)
    return {"image": image, "condition": condition, "image_path": entry["image"],
            "cache_image_path": entry.get("cache_image", entry["image"]),
            "archive_path": entry["archive"], "summary_path": entry["summary"],
            "crop_xy": [int(x), int(y)], "crop_policy": crop_policy,
            "source_class_fraction": fraction, "source_label_origin": "user_column_label",
            "source_class_id": cid, "purity_filter_applied": False,
            "source_match_method": entry.get("source_match_method", "manifest_selector"),
            "cache_origin": entry.get("cache_origin", "training_manifest"),
            "source_rgb_sha256": entry.get("source_rgb_sha256"),
            "manifest_entry_id": entry["manifest_entry_id"], "requested_image": selector,
            "selection_mode": "specified_image_" + crop_policy}


def load_large_field_matrix_sources(model, source_images=None, source_crops=None):
    """Use specified dataset-labelled sources; auto explicitly uses legacy PSIM search."""
    checkpoint = model.large_field_full
    config = checkpoint.get("config", {})
    manifest_path, entries = _matrix_manifest_entries(model)
    source_images = list(globals().get("SOURCE_IMAGE_IDS", [None] * N_CLASSES)
                         if source_images is None else source_images)
    source_crops = list(globals().get("SOURCE_CROP_XY", [None] * N_CLASSES)
                        if source_crops is None else source_crops)
    if len(source_images) != N_CLASSES or len(source_crops) != N_CLASSES:
        raise ValueError("Source images and crop coordinates must each have four column entries.")
    requests = {}
    problems = []
    model._source_mapping_report = []
    for cid, selector in enumerate(source_images):
        if selector is not None and str(selector).strip().lower() != "auto":
            try:
                requests[cid] = _matrix_resolve_source(model, entries, selector, cid)
                model._source_mapping_report.append({"class": CLASS_NAMES[cid], "requested": selector,
                                                    "status": "matched", "summary": requests[cid]["summary"]})
            except (OSError, ValueError) as exc:
                problems.append(f"{CLASS_NAMES[cid]}: {exc}")
                model._source_mapping_report.append({"class": CLASS_NAMES[cid], "requested": selector,
                                                    "status": "failed", "error": str(exc)})
                continue
        crop = source_crops[cid]
        if crop is not None:
            if cid not in requests:
                raise ValueError(f"A fixed crop for {CLASS_NAMES[cid]} requires a specified source image.")
            if (len(crop) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                                      or v < 0 for v in crop)):
                raise ValueError("Source crop must be two nonnegative integer native-pixel coordinates.")
    if problems:
        raise ValueError("Source mapping failed:\n" + "\n\n".join(problems))
    size, grid = int(model.large_field_resolution), int(model.large_field_patch_grid)
    if size <= 0 or grid <= 0 or size % (8 * grid):
        raise ValueError("Matrix crop resolution must be divisible by 8 * training patch_grid.")
    fp_mean, fp_std = _validate_fingerprint_stats(model.fp_mean, model.fp_std, "large-field checkpoint")
    expected_calibration = float(config.get("microns_per_source_pixel", float("nan")))
    if not np.isfinite(expected_calibration) or expected_calibration <= 0:
        raise ValueError("The large-field checkpoint must record its physical pixel calibration.")
    selected = {cid: _matrix_labeled_source(model, entry, cid, source_images[cid], source_crops[cid])
                for cid, entry in requests.items()}
    if len(selected) == N_CLASSES:
        print("[Sources] All four specified images accepted by dataset label; no PSIM purity ranking or cutoff.")
        return selected
    min_ratio = float(checkpoint.get("condition_schema", {}).get("hard_class_min_ratio_per_source_image", 0.03))
    if not np.isfinite(min_ratio) or not 0 <= min_ratio < 1:
        raise ValueError("Invalid training rare-class threshold for legacy auto selection.")
    ranked = {cid: [] for cid in range(N_CLASSES)}
    problems = []
    for entry in entries:
        missing = [entry[key] for key in ("image", "archive", "summary") if not os.path.isfile(entry[key])]
        if missing:
            problems.append("missing mapped file: " + missing[0])
            continue
        try:
            with open(entry["summary"], "r", encoding="utf-8-sig") as handle:
                summary = _contrast_json_load(handle)
            hard = summary.get("hard_global_phase_fraction", {})
            fractions = np.asarray([hard.get(name, np.nan) for name in CLASS_NAMES], dtype=np.float64)
            calibration = float(summary.get("quantitative_morphology", {}).get("microns_per_source_pixel", float("nan")))
            if (not np.isfinite(fractions).all() or np.any(fractions < 0)
                    or not np.isclose(fractions.sum(), 1.0, atol=1e-3)):
                raise ValueError("invalid four-class hard fractions")
            if not np.isfinite(calibration) or not np.isclose(calibration, expected_calibration, rtol=1e-5, atol=0):
                raise ValueError("physical pixel calibration differs from training")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            problems.append(f"{entry['summary']}: {exc}")
            continue
        for cid in range(N_CLASSES):
            ranked[cid].append((float(fractions[cid]), entry))

    for cid in range(N_CLASSES):
        if cid in selected:
            continue
        best = None
        candidates = sorted(ranked[cid], key=lambda item: item[0], reverse=True)
        # Fast first pass; if a class is not found, search all mapped sources
        # on an overlapping crop grid. A top-eight shortlist is never a hard cap.
        for pass_index, candidates_this_pass in enumerate((candidates[:8], candidates)):
            if pass_index and best is not None and best[0] >= 0.50:
                break
            for _, entry in candidates_this_pass:
                try:
                    labels, beta, width = _matrix_load_physical_fields(entry["archive"], min_ratio)
                    height, image_width = labels.shape
                    if min(height, image_width) < size:
                        continue
                    with Image.open(entry["image"]) as source:
                        if source.size != (image_width, height):
                            raise ValueError("PSIM/source dimensions differ; resizing is forbidden")
                    if pass_index:
                        stride = max(1, size // 2)
                        xs = np.unique(np.append(np.arange(0, image_width - size + 1, stride), image_width - size))
                        ys = np.unique(np.append(np.arange(0, height - size + 1, stride), height - size))
                    else:
                        xs = np.unique(np.linspace(0, image_width - size, 5, dtype=int))
                        ys = np.unique(np.linspace(0, height - size, 5, dtype=int))
                    for y in ys:
                        for x in xs:
                            patch = labels[y:y + size, x:x + size]
                            valid = (patch >= 0) & (patch < N_CLASSES)
                            if not valid.any():
                                continue
                            purity = float(np.count_nonzero(patch == cid) / np.count_nonzero(valid))
                            if best is None or purity > best[0]:
                                best = (purity, entry, (int(x), int(y)), patch.copy(),
                                        beta[y:y + size, x:x + size].copy(), width[y:y + size, x:x + size].copy())
                except (OSError, ValueError, TypeError) as exc:
                    problems.append(f"{entry['archive']}: {exc}")
                    continue
                if best is not None and best[0] >= 1.0 - 1e-7:
                    break
        if best is None or best[0] < 0.50:
            details = "; ".join(dict.fromkeys(problems))[:1400]
            raise ValueError(
                f"No searched cached {CLASS_NAMES[cid]} crop has at least 50% retained hard area. "
                f"Checked {len(candidates)} mapped sources, including an overlapping-grid fallback. "
                "The matrix requires a usable real source for each class; no class or PSIM is fabricated. "
                + (f"Cache issues: {details}" if details else "Check the training-image class coverage.")
            )
        purity, entry, (x, y), labels, beta, width = best
        condition = _matrix_physical_condition(labels, beta, width, grid, fp_mean, fp_std, cid)
        with Image.open(entry["image"]) as source:
            image = source.convert("RGB").crop((x, y, x + size, y + size))
        selected[cid] = {
            "image": image, "condition": condition,
            "image_path": entry["image"], "archive_path": entry["archive"],
            "summary_path": entry["summary"],
            "crop_xy": [x, y], "source_class_fraction": purity,
            "manifest_entry_id": entry["manifest_entry_id"],
            "requested_image": source_images[cid],
            "source_label_origin": "legacy_psim_selection", "purity_filter_applied": True,
            "crop_policy": "legacy_psim_search", "source_class_id": cid,
            "selection_mode": "automatic",
        }
        print(f"[Source {cid}] {CLASS_NAMES[cid]}: {entry['image']}; crop=({x}, {y}, {size}, {size}); "
              f"cached class fraction={purity:.1%}; mode={selected[cid]['selection_mode']}")
    if problems:
        print(f"[Matrix] Skipped {len(set(problems))} invalid/missing cache entry or archive issue(s); "
              f"first issue: {problems[0]}")
    print(f"[Matrix] Four real-source crops selected from {manifest_path}; existing PSIM only.")
    return selected


def _validate_checkpoint_contract(checkpoint, source):
    """Require the complete full-UNet-v4 large-field training contract."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source}: large-field checkpoint must be a dictionary.")
    def require_value(mapping, key, expected):
        if mapping.get(key) != expected:
            raise ValueError(f'{source}: {key}={mapping.get(key)!r}; required {expected!r} for full-UNet-v4 generation.')
    def require_names(mapping, key, expected):
        values = mapping.get(key)
        if values is None:
            raise ValueError(f'{source}: missing full-UNet-v4 {key}.')
        values = [FP_NAME_ALIASES.get(str(value), str(value)) for value in values]
        expected = [FP_NAME_ALIASES.get(str(value), str(value)) for value in expected]
        if values != expected:
            raise ValueError(f'{source}: incompatible full-UNet-v4 {key}: {values!r}.')
    require_value(checkpoint, 'structure_performance_contract', 'four-class-2d-directional-v1')
    require_value(checkpoint, 'checkpoint_format', CHECKPOINT_FORMAT)
    require_value(checkpoint, 'condition_schema_version', CONDITION_SCHEMA_VERSION)
    require_value(checkpoint, 'requires_full_unet', True)
    require_value(checkpoint, 'training_mode', 'large_field_full_unet_psim_hard_physical')
    require_value(checkpoint, 'direction_conditioning', 'none')
    require_value(checkpoint, 'spatial_condition_channels', SPATIAL_CONDITION_CHANNELS)
    require_value(checkpoint, 'global_condition_dim', FP_DIM + N_CLASSES + CONNECTIVITY_DIM + 1)
    require_value(checkpoint, 'fingerprint_dim', FP_DIM)
    require_value(checkpoint, 'fingerprint_stats_source', 'cached_PSIM_physical_crop_means')
    require_names(checkpoint, 'class_names', CLASS_NAMES)
    require_names(checkpoint, 'fp_names', FP_NAMES)
    require_names(checkpoint, 'connectivity_names', CONNECTIVITY_NAMES)
    schema = checkpoint.get('condition_schema')
    if not isinstance(schema, dict):
        raise ValueError(f'{source}: missing full-UNet-v4 condition_schema dictionary.')
    require_value(schema, 'version', CONDITION_SCHEMA_VERSION)
    require_value(schema, 'global_token_injection', 'additive_residual_to_last_text_token_no_normalization')
    require_names(schema, 'global_feature_order', [*FP_NAMES, *(f'{name}_fraction' for name in CLASS_NAMES), *CONNECTIVITY_NAMES, 'class_assignment_entropy'])
    require_names(schema, 'spatial_channel_order', [*(f'{name}_probability' for name in CLASS_NAMES), *FP_NAMES, *CONNECTIVITY_NAMES, 'class_assignment_entropy', 'x_coordinate', 'y_coordinate'])
    require_names(schema, 'inverse_recipe_feature_order', [*FP_NAMES, *CONNECTIVITY_NAMES, *(f'{name}_fraction' for name in CLASS_NAMES)])
    require_value(schema, 'fingerprint_units', ['fraction', 'um'])
    require_value(schema, 'class_fraction_type', 'area_fraction_of_final_PSIM_hard_map_retained_classes')
    require_value(schema, 'image_transform', 'native_pixel_crop_and_synchronized_flips_no_resize')
    require_value(schema, 'connectivity_usage', 'reserved_zero_channels_not_measured')
    require_value(schema, 'normalised_fingerprint', True)
    for name in ('unet_state_dict', 'global_morphology_token', 'large_field_conditioner'):
        if not isinstance(checkpoint.get(name), dict) or not checkpoint[name]:
            raise ValueError(f'{source}: missing trained large-field module {name}.')
    _validate_fingerprint_stats(checkpoint.get('fp_mean'), checkpoint.get('fp_std'), source)
    cfg = checkpoint.get('config')
    if not isinstance(cfg, dict):
        raise ValueError(f'{source}: missing large-field training config.')
    for key in ('resolution', 'patch_grid', 'latent_downsample', 'class_prompts_json', 'global_condition_scale', 'condition_dropout', 'spatial_condition_scale', 'microns_per_source_pixel', 'pretrained_model_name_or_path'):
        if key not in cfg:
            raise ValueError(f'{source}: missing training config.{key}.')
    resolution, grid, downsample = (int(cfg['resolution']), int(cfg['patch_grid']), int(cfg['latent_downsample']))
    if resolution <= 0 or grid <= 0 or downsample != 8 or resolution % (downsample * grid):
        raise ValueError(f'{source}: resolution/grid must match native SD crops with latent_downsample=8.')
    prompts = _contrast_json_loads(cfg['class_prompts_json'])
    if not isinstance(prompts, list) or len(prompts) != N_CLASSES or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError(f'{source}: four recorded nonempty class prompts required.')
    strength = float(checkpoint.get('active_condition_strength', float('nan')))
    global_scale = float(cfg['global_condition_scale'])
    if not np.isfinite(strength) or not 0 <= strength <= 1 or not np.isfinite(global_scale) or global_scale < 0:
        raise ValueError(f'{source}: invalid saved condition strength/scale.')
    schedule = checkpoint.get('noise_scheduler_config')
    if not isinstance(schedule, dict) or schedule.get('prediction_type', 'epsilon') not in ('epsilon', 'v_prediction'):
        raise ValueError(f'{source}: missing/unsupported recorded training schedule.')
    if not checkpoint.get('base_weight_sha256'):
        raise ValueError(f'{source}: v4 checkpoint must include training-base weight SHA256.')
    ema = checkpoint.get('ema_state_dict')
    if ema is not None:
        shadow = ema.get('shadow', {}) if isinstance(ema, dict) else {}
        for key in ('unet', 'global_morphology_token', 'large_field_conditioner'):
            if not isinstance(shadow.get(key), dict) or not shadow[key]:
                raise ValueError(f'{source}: incomplete EMA module {key}.')
    spatial_scale = float(cfg['spatial_condition_scale'])
    if not np.isfinite(spatial_scale) or spatial_scale < 0:
        raise ValueError(f'{source}: spatial_condition_scale must be finite and nonnegative.')
    calibration = float(checkpoint.get('microns_per_source_pixel', float('nan')))
    cfg_calibration = float(cfg['microns_per_source_pixel'])
    if not np.isfinite(calibration) or calibration <= 0 or (not np.isfinite(cfg_calibration)) or (not np.isclose(calibration, cfg_calibration, rtol=1e-05, atol=0)):
        raise ValueError(f'{source}: physical pixel calibration is missing or differs from training config.')


class LargeFieldInverseBundle:
    """Load one complete full-UNet-v4 large-field checkpoint for inference."""

    def __init__(self, checkpoint_path=None, device=None, dtype=None, sampling_steps=None, guidance=None):
        self.device = torch.device(DEVICE if device is None else device)
        self.dtype = DTYPE if dtype is None else dtype
        self.large_field_available = False
        self.dual_gpu_predictor = None
        self._dual_gpu_checked = False
        self.large_field_checkpoint_path = os.path.abspath(os.path.expanduser(
            str(LARGE_FIELD_MODEL_PATH if checkpoint_path is None else checkpoint_path)
        ))
        if not os.path.isfile(self.large_field_checkpoint_path):
            raise FileNotFoundError(
                "The trained large-field checkpoint was not found: "
                + self.large_field_checkpoint_path
                + ". Set --large-field-checkpoint to the complete large_field_model.pt export."
            )
        if N_CLASSES != 4 or len(CLASS_NAMES) != 4:
            raise ValueError("The large-field full-UNet-v4 contract requires exactly four classes.")

        print(f"[Load] large-field checkpoint: {self.large_field_checkpoint_path}")
        self.large_field_full = safe_torch_load(self.large_field_checkpoint_path, map_location="cpu")
        _validate_checkpoint_contract(
            self.large_field_full, self.large_field_checkpoint_path,
        )
        config = self.large_field_full["config"]
        steps = int(config.get("validation_sampling_steps", 500) if sampling_steps is None else sampling_steps)
        cfg_scale = float(config.get("validation_guidance", 7.5) if guidance is None else guidance)
        if steps < 1 or not np.isfinite(cfg_scale) or cfg_scale < 1:
            raise ValueError("Saved/requested sampling steps and guidance are invalid.")
        globals()["LARGE_FIELD_NUM_INFERENCE_STEPS"] = steps
        globals()["LARGE_FIELD_GUIDANCE_START"] = cfg_scale
        globals()["LARGE_FIELD_GUIDANCE_END"] = cfg_scale
        self.trainer = load_matching_trainer()
        known = {field.name for field in fields(self.trainer.LargeFieldConfig)}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"Companion trainer does not understand checkpoint config keys: {sorted(unknown)}")
        self.large_field_config = self.trainer.LargeFieldConfig(**config)
        self.active_condition_strength = float(self.large_field_full["active_condition_strength"])
        self.sampling_scheduler_name = (config.get("validation_scheduler", "ddpm")
                                        if SAMPLING_SCHEDULER == "training" else SAMPLING_SCHEDULER)
        if self.sampling_scheduler_name not in ("ddpm", "ddim"):
            raise ValueError("The selected sampling scheduler must be ddpm or ddim.")
        self.large_field_resolution = int(config["resolution"])
        validate_native_resolution(self.large_field_resolution)
        self.large_field_patch_grid = int(config["patch_grid"])
        self.large_field_spatial_scale = float(config["spatial_condition_scale"])
        self.fp_mean, self.fp_std = _validate_fingerprint_stats(
            self.large_field_full["fp_mean"], self.large_field_full["fp_std"],
            self.large_field_checkpoint_path,
        )
        self.pretrained_model_path = resolve_large_field_pretrained_model(self.large_field_full)
        print(
            f"[Info] native crop={self.large_field_resolution}px, "
            f"grid={self.large_field_patch_grid}x{self.large_field_patch_grid}, "
            f"spatial scale={self.large_field_spatial_scale}"
        )

        print("[Build] Training-base VAE")
        self.vae = load_diffusers_component(
            AutoencoderKL, self.pretrained_model_path, "vae", "VAE",
        )
        if VAE_DEVICE_POLICY not in ("auto", "cpu", "cuda"):
            raise ValueError("LEE_VAE_DEVICE must be auto, cpu or cuda (the selected inference GPU).")
        self.vae_device = torch.device("cpu") if VAE_DEVICE_POLICY == "cpu" else self.device

        print("[Build] Ordinary UNet; loading the complete trained v4 state")
        self.unet = load_diffusers_component(
            UNet2DConditionModel, self.pretrained_model_path, "unet", "UNet2DConditionModel",
        )
        print("[Build] Training-base text encoder and tokenizer")
        self.tokenizer = load_diffusers_component(
            CLIPTokenizer, self.pretrained_model_path, "tokenizer", "CLIPTokenizer",
        )
        self.text_encoder = load_diffusers_component(
            CLIPTextModel, self.pretrained_model_path, "text_encoder", "CLIPTextModel",
        )
        self.loaded_base_weight_sha256 = base_weight_sha256(self.vae, self.unet, self.text_encoder)
        self.base_identity_status = verify_base_digest(self.large_field_full, self.loaded_base_weight_sha256)
        self.vae.to(self.vae_device, dtype=self.dtype)
        if hasattr(F, "scaled_dot_product_attention") and hasattr(self.vae, "set_attn_processor"):
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.vae.set_attn_processor(AttnProcessor2_0())
        print(f"[VAE] device={self.vae_device}, policy={VAE_DEVICE_POLICY}; no image tiling.")
        self.vae.eval()
        self.vae.requires_grad_(False)
        self._initialize_large_field_modules()
        self.unet.to(self.device, dtype=self.dtype)
        if hasattr(F, "scaled_dot_product_attention") and hasattr(self.unet, "set_attn_processor"):
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.unet.set_attn_processor(AttnProcessor2_0())
        self.unet.eval()
        self.unet.requires_grad_(False)

        self.text_encoder.to(self.device, dtype=self.dtype)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self._large_field_prompt_embeddings = None

        self.training_scheduler_config = dict(self.large_field_full["noise_scheduler_config"])
        self.scheduler_provenance = "recorded_training_config"
        self.scheduler = build_sampling_scheduler(self, SAMPLING_TIMESTEP_SPACING)
        if LARGE_FIELD_NUM_INFERENCE_STEPS > int(self.scheduler.config.num_train_timesteps):
            raise ValueError("Sampling steps exceed recorded training timesteps.")
        self.scheduler.set_timesteps(LARGE_FIELD_NUM_INFERENCE_STEPS, device=self.device)
        self.large_field_available = True
        print(f"[Ready] Complete v4 UNet + global/spatial conditions; weights={self.generation_weight_source}, "
              f"EMA updates={self.ema_updates}, strength={self.active_condition_strength:.3f}.")
        if self.active_condition_strength < 1:
            print("[Condition] This early checkpoint has not fully enabled numerical morphology controls.")
        # Release duplicate full/raw/EMA CPU tensors; retain metadata for manifests.
        for key in ("unet_state_dict", "global_morphology_token", "large_field_conditioner", "ema_state_dict"):
            self.large_field_full.pop(key, None)

    def get_prediction_parallelism(self):
        # Lazy initialization avoids allocating a replica for cached images,
        # matrix-only explanations, and verification-only launches.
        if not self._dual_gpu_checked:
            self._dual_gpu_checked = True
            self.dual_gpu_predictor = build_dual_gpu_predictor(self.unet, self.device)
        return self.dual_gpu_predictor

    def close(self):
        predictor = self.dual_gpu_predictor
        self.dual_gpu_predictor = None
        if predictor is not None:
            secondary = predictor.secondary
            predictor.close()
            with torch.cuda.device(secondary):
                torch.cuda.empty_cache()

    def _initialize_large_field_modules(self):
        self.large_field_token = self.trainer.GlobalMorphologyToken(embed_dim=768)
        self.large_field_conditioner = self.trainer.LargeFieldSpatialConditioner(
            in_channels=SPATIAL_CONDITION_CHANNELS, out_channels=self.unet.config.in_channels)
        if USE_EMA_WEIGHTS and self.large_field_full.get("ema_state_dict") is None:
            raise ValueError("EMA requested but not saved. Select USE_EMA_WEIGHTS=False explicitly to use raw weights.")
        loaded = self.trainer.restore_generation_weights(
            self.large_field_full, self.unet, self.large_field_token,
            self.large_field_conditioner, use_ema=USE_EMA_WEIGHTS)
        self.generation_weight_source = loaded["weights"]
        self.ema_updates = int((self.large_field_full.get("ema_state_dict") or {}).get("num_updates", 0))
        for module in (self.large_field_token, self.large_field_conditioner):
            module.to(self.device, dtype=self.dtype)
            module.eval()
            module.requires_grad_(False)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        tokens = self.tokenizer(
            [prompt], padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        hidden = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return hidden.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def run_vae(self, tensor, operation):
        if operation not in ("encode", "decode"):
            raise ValueError("VAE operation must be encode or decode.")
        if VAE_DEVICE_POLICY == "auto" and self.vae_device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.vae_device)
            estimated = estimate_vae_workspace_bytes(operation, tensor.shape, tensor.element_size())
            if free < estimated:
                print(f"[VAE] {operation}: estimated workspace {estimated / 2**30:.1f} GiB "
                      f"exceeds GPU free {free / 2**30:.1f} GiB; switching VAE to CPU. "
                      "Whole-image processing is retained and will be slower.", flush=True)
                self.vae.to("cpu")
                self.vae_device = torch.device("cpu")
                torch.cuda.empty_cache()

        def forward():
            value = tensor.to(self.vae_device, dtype=self.dtype)
            if operation == "encode":
                return self.vae.encode(value).latent_dist.mode()
            return self.vae.decode(value).sample

        try:
            return forward()
        except torch.cuda.OutOfMemoryError:
            if VAE_DEVICE_POLICY != "auto" or self.vae_device.type != "cuda":
                raise
            print(f"[VAE] CUDA OOM during {operation}; retrying the whole image on CPU.", flush=True)
        # Retry outside the except block so failed CUDA activation references
        # held by the exception traceback have already been released.
        self.vae.to("cpu")
        self.vae_device = torch.device("cpu")
        torch.cuda.empty_cache()
        return forward()

    @torch.no_grad()
    def decode_latents(self, latents, expected_hw=None):
        image = self.run_vae(latents / self.vae.config.scaling_factor, "decode")
        if expected_hw is None:
            expected_hw = (self.large_field_resolution, self.large_field_resolution)
        expected_h, expected_w = (int(expected_hw[0]), int(expected_hw[1]))
        expected = (1, 3, expected_h, expected_w)
        if tuple(image.shape) != expected or not torch.isfinite(image).all():
            raise FloatingPointError(
                f"VAE produced invalid RGB output {tuple(image.shape)}; expected {expected}. "
                "No resize or tiled-image fallback is accepted."
            )
        self.last_decode_diagnostics = {
            "raw_decoder_rgb": _latent_statistics(image),
            "raw_decoder_fraction_outside_minus1_plus1": float(
                ((image < -1) | (image > 1)).float().mean().item()),
        }
        return tensor_to_pil(image)

    @torch.no_grad()
    def make_large_field_condition(
        self, target_fp_norm, hard_fraction, class_map, fp_map, class_id,
        connectivity=None, class_assignment_entropy=None, class_valid_pixel_counts=None,
        latent_h=None, latent_w=None,
    ):
        """Recreate training's text mixture, global token and spatial input."""
        if not self.large_field_available:
            raise RuntimeError("Large-field checkpoint initialization is incomplete.")
        if not 0 <= int(class_id) < N_CLASSES:
            raise ValueError("Target class ID must be in 0..3.")
        fingerprint = np.asarray(target_fp_norm, dtype=np.float32)
        fractions = np.asarray(hard_fraction, dtype=np.float32)
        if fingerprint.shape != (FP_DIM,) or not np.isfinite(fingerprint).all():
            raise ValueError(f"A finite normalized fingerprint of shape ({FP_DIM},) is required.")
        if (fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-5)):
            raise ValueError("Four finite, nonnegative class fractions must sum to one.")
        local = np.asarray(class_map, dtype=np.float32)
        grid = self.large_field_patch_grid
        if local.shape == (grid, grid, N_CLASSES):
            local = local.reshape(grid * grid, N_CLASSES)
        if local.shape != (grid * grid, N_CLASSES):
            raise ValueError("Local class conditions must use the checkpoint's trained patch grid.")
        local_fp = np.asarray(fp_map, dtype=np.float32)
        if local_fp.shape != (grid * grid, FP_DIM):
            raise ValueError("Local fingerprints must use the checkpoint's trained patch grid.")

        if connectivity is None:
            connectivity = np.zeros(CONNECTIVITY_DIM, dtype=np.float32)
        connectivity = np.asarray(connectivity, dtype=np.float32)
        if (connectivity.shape != (CONNECTIVITY_DIM,) or not np.isfinite(connectivity).all()
                or np.any(connectivity != 0)):
            raise ValueError("Physical-v3 reserves connectivity inputs as zero; they are not measured controls.")
        clipped = np.clip(local, 1e-8, 1.0)
        mean_entropy = float((-np.sum(clipped * np.log(clipped), axis=1) / np.log(N_CLASSES)).mean())
        if class_assignment_entropy is None:
            class_assignment_entropy = mean_entropy
        if (not np.isfinite(class_assignment_entropy)
                or not np.isclose(class_assignment_entropy, mean_entropy, atol=1e-5)):
            raise ValueError("Global entropy must equal the mean entropy of the local class conditions.")
        if latent_h is None:
            latent_h = self.large_field_resolution // 8
        if latent_w is None:
            latent_w = self.large_field_resolution // 8
        spatial_array = build_large_field_spatial_condition(
            hard_fraction=fractions, class_map=local, fp_map=local_fp,
            latent_h=int(latent_h), latent_w=int(latent_w),
            connectivity=connectivity, class_assignment_entropy=class_assignment_entropy,
            class_valid_pixel_counts=class_valid_pixel_counts,
        )

        fp_tensor = torch.from_numpy(fingerprint).unsqueeze(0).to(self.device, dtype=self.dtype)
        fraction_tensor = torch.from_numpy(fractions).unsqueeze(0).to(self.device, dtype=self.dtype)
        connectivity_tensor = torch.from_numpy(connectivity).unsqueeze(0).to(self.device, dtype=self.dtype)
        entropy_tensor = torch.tensor([[float(class_assignment_entropy)]], device=self.device, dtype=self.dtype)
        spatial_tensor = torch.from_numpy(spatial_array).unsqueeze(0).to(self.device, dtype=self.dtype)
        hidden, empty, spatial_bias = self.trainer.build_generation_condition(
            self.large_field_config, self.tokenizer, self.text_encoder,
            self.large_field_token, self.large_field_conditioner,
            fraction_tensor, fp_tensor, connectivity_tensor, entropy_tensor, spatial_tensor,
            (int(latent_h), int(latent_w)), self.active_condition_strength, self.device, self.dtype)
        return hidden, fp_tensor, spatial_bias


def set_style():
    """Apply consistent plotting defaults for generated candidate grids."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "image.cmap": "gray",
    })


set_style()


def main():
    parser = argparse.ArgumentParser(
        description="Generate whole-field images from requested class ratios; no generated-image measurement or validation.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--series", action="store_true",
                       help="Generate the ratios in BUILTIN_COMPOSITION_SERIES")
    modes.add_argument("--generate", action="store_true",
                       help="Generate a single recipe supplied by arguments or built-in defaults")
    parser.add_argument("--target-class", type=int, choices=range(N_CLASSES), default=None)
    parser.add_argument("--class-fractions", type=float, nargs=N_CLASSES, default=None,
                        metavar=("BASKET", "DISCONTINUOUS", "WIDMANSTATTEN", "COARSENED"))
    parser.add_argument("--dark-contrast-area-fraction", "--beta-fraction", dest="beta_fraction",
                        type=float, default=None,
                        help="Input Dark-contrast area fraction in [0,1]; legacy --beta-fraction is accepted")
    parser.add_argument("--lath-width-um", type=float, default=None,
                        help="Input mean alpha-lath width in micrometres")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Images per recipe; default one")
    parser.add_argument("--generation-seed", type=int, default=SEED)
    parser.add_argument("--sampling-steps", type=int, default=None,
                        help="Default: checkpoint validation_sampling_steps")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Default: checkpoint validation_guidance")
    parser.add_argument("--scheduler", choices=("training", "ddpm", "ddim"), default=SAMPLING_SCHEDULER)
    parser.add_argument("--timestep-spacing", choices=("default", "leading", "trailing", "linspace"),
                        default=SAMPLING_TIMESTEP_SPACING)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=USE_EMA_WEIGHTS)
    parser.add_argument("--field-width", type=int, default=NATIVE_FULL_FIELD_WIDTH)
    parser.add_argument("--field-height", type=int, default=NATIVE_FULL_FIELD_HEIGHT)
    parser.add_argument("--field-stride", type=int, default=DEFAULT_FULL_FIELD_STRIDE, help=argparse.SUPPRESS)
    parser.add_argument("--window-batch-size", type=int, default=DEFAULT_WINDOW_BATCH_SIZE, help=argparse.SUPPRESS)
    parser.add_argument("--new-run", action="store_true",
                        help="Start a new run instead of reusing matching completed images")
    parser.add_argument("--generation-output-dir", default=None)
    parser.add_argument("--large-field-checkpoint", default=None)
    # Harmless compatibility with previous generation-only launch commands.
    parser.add_argument("--no-verify-generated", action="store_true", help=argparse.SUPPRESS)
    cli_args = parser.parse_args()
    if not (cli_args.series or cli_args.generate):
        explicit_recipe = cli_args.new_run or any(value is not None for value in (
            cli_args.target_class, cli_args.class_fractions, cli_args.beta_fraction,
            cli_args.lath_width_um, cli_args.samples_per_class))
        selected = "generate" if explicit_recipe else BUILTIN_RUN_MODE
        if selected not in ("series", "generate"):
            parser.error("BUILTIN_RUN_MODE must be series or generate in this generation-only version.")
        cli_args.series = selected == "series"
        cli_args.generate = selected == "generate"
    if cli_args.series:
        if cli_args.target_class is not None or cli_args.class_fractions is not None:
            parser.error("--series uses BUILTIN_COMPOSITION_SERIES; use --generate for a custom recipe.")
        cli_args.generate = True
        cli_args.class_fractions = BUILTIN_CLASS_FRACTIONS
        if cli_args.beta_fraction is None:
            cli_args.beta_fraction = _builtin_dark_fraction()
        if cli_args.lath_width_um is None:
            cli_args.lath_width_um = BUILTIN_LATH_WIDTH_UM
    elif all(value is None for value in (cli_args.target_class, cli_args.class_fractions,
                                         cli_args.beta_fraction, cli_args.lath_width_um)):
        cli_args.class_fractions = BUILTIN_CLASS_FRACTIONS
        cli_args.beta_fraction = _builtin_dark_fraction()
        cli_args.lath_width_um = BUILTIN_LATH_WIDTH_UM
    if cli_args.class_fractions is not None and cli_args.target_class is not None:
        parser.error("Choose --target-class or --class-fractions, not both.")
    if cli_args.class_fractions is not None:
        fractions = np.asarray(cli_args.class_fractions, dtype=np.float64)
        if (fractions.shape != (N_CLASSES,) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1., atol=1e-5)):
            parser.error("Class fractions must be four finite nonnegative values summing to one.")
    if cli_args.beta_fraction is not None and (not np.isfinite(cli_args.beta_fraction)
                                               or not 0 <= cli_args.beta_fraction <= 1):
        parser.error("Dark-contrast area fraction must be finite and in [0,1].")
    if cli_args.lath_width_um is not None and (not np.isfinite(cli_args.lath_width_um)
                                               or cli_args.lath_width_um <= 0):
        parser.error("Alpha-lath width must be finite and positive.")
    if cli_args.sampling_steps is not None and cli_args.sampling_steps < 1:
        parser.error("--sampling-steps must be positive.")
    if cli_args.guidance is not None and (not np.isfinite(cli_args.guidance) or cli_args.guidance < 1):
        parser.error("--guidance must be finite and at least one.")
    if (min(cli_args.field_width, cli_args.field_height) < 1
            or cli_args.field_width % 64 or cli_args.field_height % 64):
        parser.error("Image dimensions must be positive multiples of 64.")
    if min(cli_args.field_stride, cli_args.window_batch_size) < 1:
        parser.error("Legacy stride/batch arguments must be positive.")
    if cli_args.samples_per_class is None:
        cli_args.samples_per_class = SERIES_SAMPLES_PER_RECIPE if cli_args.series else REFERENCE_FREE_SAMPLES_PER_CLASS
    if cli_args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive.")
    if not 0 <= cli_args.generation_seed <= 2**32 - cli_args.samples_per_class:
        parser.error("All requested image seeds must be within the uint32 range.")
    globals()["SAMPLING_SCHEDULER"] = cli_args.scheduler
    globals()["USE_EMA_WEIGHTS"] = cli_args.use_ema
    globals()["SAMPLING_TIMESTEP_SPACING"] = cli_args.timestep_spacing
    if cli_args.large_field_checkpoint:
        globals()["LARGE_FIELD_MODEL_PATH"] = os.path.abspath(os.path.expanduser(cli_args.large_field_checkpoint))
    if not os.path.isfile(LARGE_FIELD_MODEL_PATH):
        raise FileNotFoundError("A complete native1536 checkpoint is required: " + LARGE_FIELD_MODEL_PATH)
    configure_offline_loading()
    configure_torch_runtime()
    globals()["DEVICE"] = select_inference_device(INFERENCE_DEVICE)
    seed_everything(cli_args.generation_seed)
    print("=" * 80)
    print("Whole-canvas inverse generation | generation only; no image verification")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {LARGE_FIELD_MODEL_PATH}")
    print(f"Output directory: {cli_args.generation_output_dir or OUTPUT_DIR}")
    print(f"Requested field: {cli_args.field_width}x{cli_args.field_height}")
    print("Mode: " + ("composition series" if cli_args.series else "single recipe"))
    print("=" * 80)
    model = LargeFieldInverseBundle(sampling_steps=cli_args.sampling_steps, guidance=cli_args.guidance)
    try:
        validate_native_resolution(model.large_field_resolution, cli_args.field_width, cli_args.field_height)
        if cli_args.series:
            _, generation_dirs = run_composition_series(model, cli_args)
        else:
            generation_dirs = [run_reference_free_large_field_generation(model, cli_args)]
        print(f"[Done] Generation complete: {len(generation_dirs)} recipe/seed run(s). "
              "No generated-image measurement, validation figure, or feedback retry was performed.", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
