"""Configuration parser for the four-class tissue fingerprint training pipeline."""

import argparse
import os


PROJECT_ROOT = r"/home/user/data/2025323025029/LEE_1/"
CLASS_NAMES = [
    "Basket_weave",
    "Discontinuous",
    "Widmanstatten",
    "Alpha_coarsened_Widmanstatten",
]


def _add_boolean_argument(parser, name, default, help_text):
    """Add a BooleanOptionalAction while keeping the option explicit."""
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _add_path_arguments(parser):
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--validation_img_dir", type=str, default=None)
    parser.add_argument("--test_img_dir", type=str, default=None)
    parser.add_argument("--test_prompt_dir", type=str, default=None)
    parser.add_argument("--morphology_field_dir", type=str, default=None)


def _add_training_arguments(parser):
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--stage1_steps", type=int, default=400)
    parser.add_argument("--stage2_steps", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=400)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--caption_column", type=str, default="label")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--tracker_project_name", type=str, default="fingerprint-diffusion")
    _add_boolean_argument(parser, "--run_test", True, "Run the final test evaluation.")
    _add_boolean_argument(parser, "--auto_generate_masks", True, "Generate morphology masks when needed.")
    _add_boolean_argument(parser, "--save_at_end", True, "Save final model checkpoints.")
    _add_boolean_argument(parser, "--save_sd_pipeline", True, "Save the final diffusion pipeline.")


def _add_module_arguments(parser):
    _add_boolean_argument(parser, "--use_mtfa", True, "Enable multi-timestep feature aggregation.")
    _add_boolean_argument(parser, "--use_dit", True, "Enable the bottleneck transformer.")
    _add_boolean_argument(parser, "--use_pqa", True, "Enable physical query attention.")
    _add_boolean_argument(parser, "--use_ace", True, "Enable adaptive condition embeddings.")
    _add_boolean_argument(parser, "--use_dcl", True, "Enable diffusion contrastive learning.")
    _add_boolean_argument(parser, "--use_unet_lora", True, "Enable UNet LoRA training.")
    _add_boolean_argument(parser, "--use_connectivity_aux", True, "Enable the connectivity auxiliary head.")

    parser.add_argument("--mtfa_num_timesteps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--dcl_temperature", type=float, default=0.2)
    parser.add_argument("--dcl_output_dim", type=int, default=128)
    parser.add_argument("--dcl_warmup_steps", type=int, default=50)
    parser.add_argument("--ace_scale", type=float, default=1.0)


def _add_loss_arguments(parser):
    parser.add_argument("--stage1_guidance_dropout", type=float, default=0.1)
    parser.add_argument("--stage2_guidance_dropout", type=float, default=0.2)
    parser.add_argument("--stage1_lambda_diff", type=float, default=1.0)
    parser.add_argument("--stage1_lambda_cls", type=float, default=0.7)
    parser.add_argument("--stage1_lambda_field", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_fingerprint", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_contrast", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_separation", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_fp_decorrelation", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_fp_head_orthogonal", type=float, default=0.0)
    parser.add_argument("--stage2_lambda_diff", type=float, default=0.2)
    parser.add_argument("--stage2_lambda_cls", type=float, default=1.2)
    parser.add_argument("--stage2_lambda_field", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_fingerprint", type=float, default=0.4)
    parser.add_argument("--stage2_lambda_contrast", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_separation", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_fp_decorrelation", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_fp_head_orthogonal", type=float, default=0.05)
    parser.add_argument("--lambda_fp_cls", type=float, default=0.0)
    parser.add_argument("--lambda_connectivity", type=float, default=0.15)
    parser.add_argument("--connectivity_loss_beta", type=float, default=0.1)
    parser.add_argument("--lambda_connectivity_pair", type=float, default=0.0)
    parser.add_argument("--connectivity_pair_margin", type=float, default=0.20)


def _resolve_default_paths(args):
    """Fill project-relative paths while preserving explicit CLI overrides."""
    root = os.path.abspath(args.project_root)
    dataset_root = os.path.join(root, "data", "example_data_train_val_test")

    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or os.path.join(
        root, "models", "stable-diffusion-v1-5"
    )
    args.dataset_name = args.dataset_name or os.path.join(dataset_root, "train")
    args.output_dir = args.output_dir or os.path.join(
        root, "Diffusion", "Deep-TiDiffusion-Outputs-FourClass-v3"
    )
    args.validation_img_dir = args.validation_img_dir or os.path.join(dataset_root, "val")
    args.test_img_dir = args.test_img_dir or os.path.join(dataset_root, "test")
    args.test_prompt_dir = args.test_prompt_dir or args.test_img_dir
    args.morphology_field_dir = args.morphology_field_dir or os.path.join(
        dataset_root, "morphology_fields_tc4_2d_four_class_v3"
    )
    args.stage2_steps = args.stage2_steps or args.max_train_steps - args.stage1_steps
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Four-class tissue fingerprint diffusion training"
    )
    _add_path_arguments(parser)
    _add_training_arguments(parser)
    _add_module_arguments(parser)
    _add_loss_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args(argv)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return _resolve_default_paths(args)


__all__ = ["CLASS_NAMES", "PROJECT_ROOT", "parse_args"]
