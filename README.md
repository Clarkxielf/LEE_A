# FIM-Diffusion

**Quantitative and interpretable microstructure fingerprinting of Ti-6Al-4V alloy**

FIM-Diffusion (fingerprint-informed morphology diffusion) connects microstructure recognition, spatial quantification, fingerprint-conditioned generation, and specimen-level structure-property analysis within a common quantitative representation.

Developed for optical micrographs of Ti-6Al-4V, the framework describes microstructural heterogeneity through morphology-class composition, dark-contrast area fraction, and mean alpha-lath width. These descriptors make experimentally observed and generated microstructures quantitatively comparable, while supporting analysis of their associations with measured mechanical properties.

## Research overview

The manuscript reports **99.7% classification accuracy on the held-out test set** across four morphology classes. Its principal contribution is a shared representation for three complementary workflows:

1. **Spatial characterization.** The physics-aware spatial inference module (PSIM) combines overlapping-patch predictions to map morphology composition and image-derived descriptors across large micrographs.
2. **Fingerprint-conditioned generation.** A dedicated diffusion generator produces large-field candidates from prescribed morphology fractions and continuous descriptors. Generated images can then be remeasured with PSIM in the same descriptor space.
3. **Microstructure-property association.** Image descriptors are aggregated by specimen and compared with experimentally measured elongation, tensile strength, and fatigue strength using Spearman rank correlations.

Together, these workflows turn expert-recognized morphology into measurable coordinates for investigating heterogeneity and specifying candidate microstructures. Generated candidates can support synthetic-data studies and future property-guided generation research; the implemented generation workflow is conditioned on morphology descriptors, not mechanical-property targets or processing parameters.

## Microstructural fingerprint

| Component | Representation | Meaning |
|---|---|---|
| Morphology composition | Four class fractions | Relative contributions of the four morphology classes |
| Dark-contrast area fraction | Fraction from 0 to 1 | Image area assigned to dark regions by contrast segmentation |
| Mean alpha-lath width | Micrometres | Image-derived lath-scale measurement using physical pixel calibration |

The composite fingerprint contains six reported values: four class fractions and two continuous descriptors. The model's continuous-descriptor encoder takes the two continuous components; class conditioning is handled separately.

Dark-contrast area fraction is an **image-contrast descriptor**, not an independently identified beta-phase fraction. Legacy names containing `beta` may remain in checkpoint fields, cache paths, or accepted argument aliases for compatibility.

### Class order

Use this order consistently for class directories, target vectors, checkpoints, and exported results:

| Class ID | Directory name | Morphology |
|---:|---|---|
| 0 | `Basket_weave` | Basket-weave |
| 1 | `Discontinuous` | Discontinuous |
| 2 | `Widmanstatten` | Widmanstatten |
| 3 | `Alpha_coarsened_Widmanstatten` | Alpha-coarsened Widmanstatten |

PSIM reports both soft and hard morphology fractions. Soft fractions average spatial class probabilities; hard fractions measure class areas in the final categorical map. The specimen-level association workflow uses hard fractions.

## Model and training

The patch model combines a pretrained latent diffusion backbone with:

- **Auxiliary Condition Encoder (ACE):** encodes continuous descriptors and morphology identity into a conditioning embedding.
- **Multi-Time-Step Feature Aggregation (MTFA):** combines U-Net mid-block features from multiple diffusion timesteps.
- **Mid-block Transformer:** refines the aggregated representation.
- **Physical Query Attention (PQA):** uses the ACE embedding to attend over spatial features and produce a channel-wise gate shared across spatial positions.

Training proceeds in two stages. Stage 1 sets the continuous inputs to ACE to zero while retaining class conditioning. Stage 2 activates continuous descriptors and the associated auxiliary objectives. Classification uses the zero-condition feature branch, whereas fingerprint and morphology-field readouts use conditioned features. The morphology-field head receives detached features, so its loss does not backpropagate through the upstream feature stream.

For this patch model, the base VAE, CLIP text encoder, and U-Net weights are frozen; U-Net attention LoRA parameters and the added modules are optimized. **Large-field generation uses a separate full-U-Net training program and checkpoint.** The two checkpoint types are not interchangeable.

## Programs

| File | Role |
|---|---|
| `train.py` | Patch-model training, evaluation, and ablations |
| `large_image_analysis.py` | PSIM spatial inference and image-derived morphology measurements |
| `train_large_field.py` | Dedicated full-U-Net generation training on native 1536 x 1536-pixel crops |
| `inverse_design.py` | Fingerprint-conditioned whole-field generation; filename retained for compatibility |
| `morphology_property_pipeline.py` | Specimen-level measured-property comparisons and Spearman correlations |
| `comparative experiment.py` | CNN/Transformer classification benchmarks |

Keep the companion modules imported by the selected scripts with the release, including `gpu_selector.py` and any required data-loading or configuration helpers. The commands below target the generation-only and measured-association versions of the programs, rather than historical property-prediction variants.

## Installation

```bash
git clone https://github.com/Clarkxielf/LEE_A.git
cd LEE_A
conda create -n fim-diffusion python=3.10 -y
conda activate fim-diffusion
```

Install PyTorch and torchvision using a build compatible with your hardware from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). The main additional dependencies are:

```bash
python -m pip install diffusers transformers accelerate datasets huggingface-hub
python -m pip install numpy pandas scipy scikit-image scikit-learn opencv-python Pillow
python -m pip install matplotlib tqdm timm openpyxl certifi
```

This is a dependency overview, not a version-locked reproduction environment. Reproducing the reported experiments requires the corresponding package versions, model weights, data splits, and run settings. GPU memory requirements depend strongly on crop size and generation canvas size.

## Data and model preparation

The manuscript describes 3,312 annotated optical micrographs divided into training, validation, and test sets in an 8:1:1 ratio, together with an independent collection of 1,088 large-field micrographs for spatial analysis. Specimen-level mechanical measurements form a separate input to the association workflow.

An example input layout is:

```text
LEE_A/
  data/
    example_data_train_val_test/
      train/<class_name>/
      val/<class_name>/
      test/<class_name>/
    ALL-DATA/
    Mechanical_Property.xlsx
  models/
    stable-diffusion-v1-5/
  outputs/
```

Use the class directory names listed above. Preserve image-to-specimen identifiers and the original split manifest. Overlapping crops and repeated fields from the same source must be traceable when evaluating split independence.

### Stable Diffusion v1.5 base model

Download the Diffusers-format Stable Diffusion v1.5 model from the [Hugging Face model page](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and place it in `models/stable-diffusion-v1-5/`. The model page identifies the repository as a mirror of the deprecated RunwayML repository and lists the CreativeML Open RAIL-M license. For reproducibility, record the exact model revision used for each run. The FIM-Diffusion patch weights, fingerprint statistics and large-field checkpoint are separate artifacts described under **Data, weights, and availability**.

**PSIM configuration:** the supplied PSIM script reads `PSIM_PROJECT_ROOT`, not simply `LEE_A_PROJECT_ROOT`. Its `PRETRAINED_MODEL`, `CHECKPOINT_DIR`, and `FP_STATS_PATH` settings must point to the matching base model, patch-model weights, and normalization statistics. Setting `--output-dir` changes the output location, not the checkpoint location. Check `MICRONS_PER_SOURCE_PIXEL` against the image calibration before measuring lath widths.

For example, set the PSIM root in the shell used to run it:

```bash
export PSIM_PROJECT_ROOT="/path/to/LEE_A"
```

Windows PowerShell equivalent:

```powershell
$env:PSIM_PROJECT_ROOT = "C:\path\to\LEE_A"
```

Other scripts have their own path defaults; use explicit path arguments where available. The multiline examples below use Bash syntax.

## Usage

### 1. Train the patch model

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

For multi-seed ablations, the training interface also supports `--ablation all`, `--seeds 42,43,44,45,46`, and `--ablation-root`. Use the same explicit data and base-model paths across runs.

### 2. Quantify large-field micrographs with PSIM

After configuring PSIM's checkpoint paths, statistics, and pixel calibration:

```bash
python large_image_analysis.py \
  --input /path/to/LEE_A/data/ALL-DATA \
  --output-dir /path/to/LEE_A/outputs/psim \
  --patch-size 1024 \
  --stride 256 \
  --area-mode patch-vote
```

PSIM produces spatial class probabilities, hard morphology maps, image-derived descriptor fields, and quantitative records. Hard-map results depend on the recorded voting, boundary-refinement, and class-presence settings. Retain quantitative cache files and source-image mappings, not only figure previews.

### 3. Train the large-field generator

Prepare PSIM caches for the actual generation-training images. A cache from another image collection cannot substitute for their spatial conditions. Use `--prepare_psim_only` to build missing training records, or `--validate_psim_only` to check existing records, with the appropriate paths for the selected dataset.

```bash
python train_large_field.py \
  --pretrained_model_name_or_path /path/to/LEE_A/models/stable-diffusion-v1-5 \
  --large_image_dir /path/to/LEE_A/data/ALL-DATA \
  --psim_output_dir /path/to/LEE_A/outputs/psim \
  --initial_lora_checkpoint /path/to/LEE_A/outputs/fim_diffusion/full_model.pt \
  --output_dir /path/to/LEE_A/outputs/large_field_full_unet_1536_v4
```

Provide a compatible initialization checkpoint and verify `microns_per_source_pixel` against the training images and PSIM caches. The trainer uses native 1536 x 1536-pixel crops with aligned spatial conditions. Its global token and spatial conditioning pathway encode morphology composition and continuous descriptors; mechanical-property labels are not required.

### 4. Generate fingerprint-conditioned microstructures

The following values illustrate the input format; they are not a recommended target or a reproduction recipe for a particular manuscript panel.

```bash
python inverse_design.py \
  --generate \
  --large-field-checkpoint /path/to/LEE_A/outputs/large_field_full_unet_1536_v4/large_field_model.pt \
  --class-fractions 0.50 0.00 0.50 0.00 \
  --dark-contrast-area-fraction 0.30 \
  --lath-width-um 3.00 \
  --field-width 5440 \
  --field-height 3648 \
  --generation-seed 42 \
  --generation-output-dir /path/to/LEE_A/outputs/generated
```

Class fractions must be non-negative and sum to one. Continuous descriptors are entered in physical units: dark-contrast area fraction from 0 to 1 and lath width in micrometres. Normalization uses the matching checkpoint statistics.

The generation-only implementation samples a whole latent canvas and decodes at the requested dimensions. A 5440 x 3648-pixel output is larger than the training crop; it is not evidence of training at that full-field resolution. The supplied generator defaults to two GPU replicas; set `LEE_INVERSE_GPU_COUNT=1` for single-GPU execution.

**Remeasurement is a separate step.** Run `large_image_analysis.py` on the generated image or image directory, using compatible calibration and reporting settings. Compare the resulting class fractions, dark-contrast fraction, and mean lath width with the requested values. The generation-only script writes images and generation metadata but does not automatically run PSIM verification. Target conditions and measured outputs should be retained as separate records.

### 5. Relate morphology to measured properties

```bash
python morphology_property_pipeline.py \
  --large-image-dir /path/to/specimen_micrographs \
  --property-xlsx /path/to/LEE_A/data/Mechanical_Property.xlsx \
  --psim-output-dir /path/to/LEE_A/outputs/specimen_psim \
  --analysis-output-dir /path/to/LEE_A/outputs/morphology_property \
  --skip-psim
```

This example reuses existing specimen-image PSIM records. The current property table uses specimen identifier `NO` and the endpoints `Elongation`, `Tensile_strength`, and `Fatigue_strength`, with no separate XY/Z endpoints. Match identifiers to the image records and retain the reported measurement units and experimental standard deviations.

Duplicate image records are checked and removed before equal-image-weight aggregation within each specimen. The per-image hard-class presence filter is retained; no second threshold is applied to the specimen average. The manuscript analysis comprises eight specimens, including six original specimens and two exploratory specimens, 375 and 961.

Spearman coefficients use pairwise-complete specimen records. A coefficient is not calculated with fewer than three paired specimens or a constant variable. Individual images are sampled fields, not independent mechanical-test replicates. Experimental property variability and between-image descriptor variability are reported separately. This workflow performs descriptive association analysis, not property-model fitting.

### 6. Run classification benchmarks

```bash
python "comparative experiment.py" \
  --mode validate \
  --data-root /path/to/LEE_A/data/example_data_train_val_test \
  --output-dir /path/to/LEE_A/outputs/classification_benchmark
```

Use `--mode run` for training and specify the intended architectures, initialization, and seeds. The interface accepts `--seeds 42 43 44 45 46` and supports specimen/group checks through `--groups-csv` and `--require-groups`. A valid group mapping is needed to establish group separation; file-level duplicate checks alone do not establish specimen independence.

## Outputs and reproducibility

Depending on the workflow, outputs include classification metrics, checkpoints, PSIM spatial fields and summaries, generated images and manifests, specimen-level descriptor tables, and correlation figures and tables. For generated-image assessment, retain both the generation manifest and the separately computed PSIM results.

Archive the code revision, dependency versions, seeds, run arguments, input manifests, specimen mapping, physical pixel calibration, normalization statistics, and checkpoint identifiers with each experiment. For large-field generation, also retain the base-model identity, sampler settings, requested dimensions, and conditioning targets. The reported 99.7% accuracy refers to the manuscript test set and is not a performance guarantee for another dataset.

## Data, weights, and availability

Code availability, experimental-image availability, and trained-weight availability are separate resources. Cloning the code does not provide the experimental datasets or model checkpoints. This README does not yet specify public dataset or trained-checkpoint download links; consult the manuscript's Data Availability and Code Availability statements for the applicable access arrangements.

Reproducing the complete workflow requires calibrated images or the appropriate quantitative records, specimen-level measurements for property analysis, compatible base-model weights, and the relevant trained checkpoints. Descriptor tables support association analysis, but do not replace source images for training or independent image remeasurement.

## Citation and licensing

Please cite the accompanying FIM-Diffusion manuscript when using this work. Final publication metadata and any archived software DOI should be used once available.

Consult the repository's explicit license file for code reuse terms. This README does not assign a license to the code, data, or trained weights. Third-party packages and pretrained models retain their own licenses.

## Contact

For implementation questions or reproducibility issues, use the [repository issue tracker](https://github.com/Clarkxielf/LEE_A/issues). Include the script version, command, relevant configuration, and error message without sharing private data or credentials.
