import os
import shutil
import subprocess


def _is_distributed_process():
    world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        if int(world_size) > 1:
            return True
    except ValueError:
        pass
    return any(os.environ.get(key) not in (None, "") for key in ("LOCAL_RANK", "RANK"))


def query_gpus():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    command = [
        nvidia_smi,
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, memory_used, memory_total, utilization = map(float, fields)
        except ValueError:
            continue
        if memory_total <= 0:
            continue
        gpus.append({
            "index": int(index),
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
            "utilization_percent": utilization,
            "memory_ratio": memory_used / memory_total,
        })
    return gpus


def select_idle_gpu(max_memory_ratio=0.25, max_utilization_percent=20.0):
    if os.environ.get("GPU_AUTO_SELECT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return None
    if _is_distributed_process():
        return None

    gpus = query_gpus()
    if not gpus:
        return None

    idle = [
        gpu for gpu in gpus
        if gpu["memory_ratio"] <= max_memory_ratio
        and gpu["utilization_percent"] <= max_utilization_percent
    ]
    if not idle:
        return None

    selected = min(
        idle,
        key=lambda gpu: (0.75 * gpu["memory_ratio"] + 0.25 * gpu["utilization_percent"] / 100.0,
                         gpu["memory_used_mb"]),
    )
    return selected


def configure_automatic_gpu_selection():
    selected = select_idle_gpu()
    if selected is None:
        if (
            os.environ.get("GPU_AUTO_SELECT", "1").strip().lower()
            not in {"0", "false", "no", "off"}
            and os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
            and not _is_distributed_process()
        ):
            print(
                "No GPU satisfies the idle criteria; preserving the current CUDA device configuration.",
                flush=True,
            )
        return None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    print(
        "Auto-selected GPU {index}: {used:.0f}/{total:.0f} MiB, GPU utilization {util:.1f}%".format(
            index=selected["index"],
            used=selected["memory_used_mb"],
            total=selected["memory_total_mb"],
            util=selected["utilization_percent"],
        ),
        flush=True,
    )
    return selected
