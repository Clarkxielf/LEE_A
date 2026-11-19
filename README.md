# Quantitative-and-interpretable-microstructure-fingerprinting-of-Ti-6Al-4V-alloy-by-FIM-Diffusion

**Official implementation of FIM-Diffusion for four-class microstructure recognition, quantitative morphology fingerprinting, conditional generation, and inverse design.**

[Overview](#overview) · [Installation](#installation) · [Data](#data-organization) · [Usage](#usage) · [Citation](#citation)

</div>

## Overview

This repository contains the research code accompanying the manuscript **“Quantitative and Interpretable Microstructure Fingerprinting of Ti-6Al-4V Alloy by FIM-Diffusion.”** The workflow combines a latent diffusion model with interpretable morphology descriptors for Ti-6Al-4V metallographic images.

The current implementation supports:

- four-class Ti-6Al-4V microstructure recognition;
- diffusion-based microstructure representation learning and generation;
- quantitative morphology fingerprinting using dark-contrast area fraction and mean alpha-lath width;
- physics-aware spatial inference for large metallographic images;
- native large-field conditional generation and inverse design;
- CNN and Transformer classification benchmarks; and
- specimen-level morphology-property association analysis.

The canonical class order used throughout the code is:

| Class ID | Directory name | Description |
|---:|---|---|
| 0 | `Basket_weave` | Basket-weave microstructure |
| 1 | `Discontinuous` | Discontinuous microstructure |
| 2 | `Widmanstatten` | Widmanstätten microstructure |
| 3 | `Alpha_coarsened_Widmanstatten` | Alpha-coarsened Widmanstätten microstructure |

Preserve this order when preparing class vectors, datasets, checkpoints, and evaluation results.

## Repository contents

| File | Purpose |
|---|---|
| `prepare_data.py` | Builds reproducible train/validation/test splits from class-organized images |
| `config.py` | Shared arguments and defaults for four-class FIM-Diffusion training |
| `data_loader.py` | Canonical class mapping and evaluation-image loading |
| `gpu_selector.py` | Optional NVIDIA GPU selection utility |
| `train.py` | Four-class FIM-Diffusion training, testing, and ablation experiments |
| `train_large_field.py` | Native 1536-pixel full-UNet large-field training |
| `large_image_analysis.py` | Physics-aware spatial inference and morphology quantification |
| `inverse_design.py` | Conditional large-field generation and inverse design |
| `morphology_property_pipeline.py` | Morphology-property association analysis |
| `comparative experiment.py` | Multi-seed CNN/Transformer classification benchmark |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Clarkxielf/LEE_A.git
cd LEE_A
```

### 2. Create an environment

Python 3.10 is recommended. Create an isolated environment before installing the dependencies.

```bash
conda create -n lee-a python=3.10 -y
conda activate lee-a
```

Install a PyTorch build compatible with your CUDA driver, then install the remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate datasets huggingface-hub
pip install numpy pandas scipy scikit-image scikit-learn opencv-python Pillow
pip install matplotlib tqdm timm openpyxl certifi
```

The CUDA command above is an example. Select the appropriate PyTorch build for your system from the official PyTorch installation guide. For exact reproducibility, record and publish the package versions used for the final experiments.

### 3. Configure the project root

Linux or macOS:

```bash
export LEE_A_PROJECT_ROOT="$(pwd)"
```

Windows PowerShell:

```powershell
$env:LEE_A_PROJECT_ROOT = (Get-Location).Path
```

Several programs also accept explicit command-line paths. Explicit paths take precedence over inferred defaults and are recommended for server experiments.

## Data organization

Datasets are not distributed through this GitHub repository. Organize the four-class data as follows:

```text
LEE_A/
├── data/
│   ├── example_data/                         # Original class-organized images
│   │   ├── Basket_weave/
│   │   ├── Discontinuous/
│   │   ├── Widmanstatten/
│   │   └── Alpha_coarsened_Widmanstatten/
│   ├── example_data_train_val_test/
│   │   ├── train/{Basket_weave,Discontinuous,Widmanstatten,Alpha_coarsened_Widmanstatten}/
│   │   ├── val/{Basket_weave,Discontinuous,Widmanstatten,Alpha_coarsened_Widmanstatten}/
│   │   └── test/{Basket_weave,Discontinuous,Widmanstatten,Alpha_coarsened_Widmanstatten}/
│   ├── ALL-DATA/                             # Native large-field source images
│   └── Mechanical_Property.xlsx              # Optional property-analysis input
├── models/
│   └── stable-diffusion-v1-5/                # Local pretrained diffusion model
└── outputs/                                  # Checkpoints and generated results
```

To create the train/validation/test split interactively:

```bash
python prepare_data.py
```

The script records the class mapping and split manifest under the generated `_metadata` directory. Before publication, verify that images originating from the same specimen do not leak across training, validation, and test sets.

## Model weights

Pretrained Stable Diffusion weights, trained checkpoints, and large experimental datasets are not stored in this repository. Download or archive them separately, then provide their paths through command-line arguments or environment variables.

The expected local base-model directory is:

```text
models/stable-diffusion-v1-5/
```

It should contain the compatible VAE, UNet, text encoder, tokenizer, and scheduler files. Third-party models remain subject to their original licenses and usage conditions.

## Usage

Run `python <script> --help` before an experiment to view the complete interface supported by the installed version.

### Four-class FIM-Diffusion training

```bash
python train.py \
  --project-root /path/to/LEE_A \
  --pretrained_model_name_or_path /path/to/LEE_A/models/stable-diffusion-v1-5 \
  --dataset_name /path/to/LEE_A/data/example_data_train_val_test/train \
  --validation_img_dir /path/to/LEE_A/data/example_data_train_val_test/val \
  --test_img_dir /path/to/LEE_A/data/example_data_train_val_test/test \
  --output_dir /path/to/LEE_A/outputs/fim_diffusion \
  --seed 42
```

To run the controlled ablation modes across multiple random seeds:

```bash
python train.py \
  --project-root /path/to/LEE_A \
  --ablation all \
  --seeds 42,43,44,45,46 \
  --ablation-root /path/to/LEE_A/outputs/ablation
```

### Large-field morphology analysis

```bash
python large_image_analysis.py \
  --input /path/to/images \
  --output-dir /path/to/LEE_A/outputs/psim \
  --patch-size 1024 \
  --stride 256 \
  --area-mode patch-vote
```

The analysis produces image-level summaries, spatial class maps, quantitative morphology fields, patch predictions, and compressed result archives. Checkpoint paths and fingerprint statistics must be compatible with the selected trained model.

### Native large-field training

```bash
python train_large_field.py \
  --pretrained_model_name_or_path /path/to/LEE_A/models/stable-diffusion-v1-5 \
  --large_image_dir /path/to/LEE_A/data/ALL-DATA \
  --psim_output_dir /path/to/LEE_A/outputs/psim_training_ALL_DATA \
  --output_dir /path/to/LEE_A/outputs/large_field_full_unet_1536_v4
```

Large-field training operates on native 1536 × 1536-pixel crops and can require substantial GPU memory. Validate cached PSIM records and initialization checkpoints before launching a long training run.

### Conditional generation and inverse design

Generate a microstructure from a target four-class composition and morphology fingerprint:

```bash
python inverse_design.py \
  --generate \
  --class-fractions 0.76 0.00 0.24 0.00 \
  --dark-contrast-area-fraction 0.25 \
  --lath-width-um 1.20 \
  --generation-seed 42 \
  --generation-output-dir /path/to/LEE_A/outputs/inverse_design
```

The four class fractions must be finite, non-negative, and sum to one. The legacy `--beta-fraction` option is accepted as an alias by the current program, but published results should use the physically cautious term **dark-contrast area fraction** unless independent phase identification is available.

### Classification benchmark

Validate the dataset and split integrity:

```bash
python "comparative experiment.py" \
  --mode validate \
  --data-root /path/to/LEE_A/data/example_data_train_val_test \
  --output-dir /path/to/LEE_A/outputs/classification_benchmark
```

Run the default eight architectures using five random seeds:

```bash
python "comparative experiment.py" \
  --mode run \
  --data-root /path/to/LEE_A/data/example_data_train_val_test \
  --seeds 42 43 44 45 46 \
  --epochs 50 \
  --scratch \
  --output-dir /path/to/LEE_A/outputs/classification_benchmark
```

For stronger leakage control, provide a specimen/group mapping using `--groups-csv` and enable `--require-groups`.

### Morphology-property analysis

```bash
python morphology_property_pipeline.py \
  --large-image-dir /path/to/LEE_A/data/ALL-DATA \
  --property-xlsx /path/to/LEE_A/data/Mechanical_Property.xlsx \
  --psim-output-dir /path/to/LEE_A/outputs/psim \
  --analysis-output-dir /path/to/LEE_A/outputs/morphology_property
```

This pipeline evaluates specimen-level associations between measured morphology descriptors and mechanical properties. Interpret predictive results only under the grouping, sample-size, and validation assumptions recorded by the run.

## Reproducibility checklist

For each reported experiment, archive:

- the exact Git commit and release tag;
- Python, CUDA, PyTorch, and package versions;
- random seeds and command-line arguments;
- dataset version, class mapping, and specimen-level split manifest;
- microscope calibration and image preprocessing history;
- pretrained-model and checkpoint identifiers;
- generated run manifests and evaluation outputs; and
- the hardware used for training and inference.

Do not commit private data, credentials, local absolute paths, raw datasets, model weights, or generated experiment directories to GitHub.

## Outputs

Depending on the selected workflow, the programs may create:

- model checkpoints and training-state files;
- classification metrics and confusion matrices;
- PSIM class maps and morphology-field archives;
- generated large-field microstructures and run manifests;
- target-versus-measurement verification tables; and
- morphology-property statistics and publication figures.

Output directories are intentionally excluded from version control because they may contain large binary artifacts.

## Citation

If this repository contributes to your research, please cite the accompanying paper and the archived software release. The final paper citation and version-specific DOI will be added after publication and software archiving.

```bibtex
@software{fim_diffusion_lee_a,
  title   = {Quantitative and Interpretable Microstructure Fingerprinting of Ti-6Al-4V Alloy by FIM-Diffusion},
  author  = {Paper authors},
  year    = {2026},
  url     = {https://github.com/Clarkxielf/LEE_A},
  version = {v1.0.0}
}
```

Replace `Paper authors` and `v1.0.0` with the final author list and archived release version before publication.

## License

The source code in this repository is released under the [MIT License](LICENSE). Datasets, pretrained models, and third-party components may use different licenses; consult their original sources before redistribution or commercial use.

## Contact

For questions about the implementation or reproducibility, open a GitHub issue. Author contact information can be added here after the manuscript metadata are finalized.

