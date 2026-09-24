"""Argument and configuration helpers used by the public CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..common import SteeringParameters
from ..config import AppConfig, get_model_checkpoint, load_config
from ..schedules import validate_steering_parameters


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint override.")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cpu, cuda, or cuda:N).")
    parser.add_argument("--device-number", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)


def add_generation_arguments(parser: argparse.ArgumentParser, *, image_to_image: bool, legacy: bool = False) -> None:
    if legacy:
        parser.add_argument("--source-prompt", "--prompt", dest="prompt", type=str, default=None)
    else:
        parser.add_argument("--prompt", "--source-prompt", dest="prompt", type=str, default=None)
    parser.add_argument("--target-prompt", type=str, default=None)
    parser.add_argument("--safe-prompt", type=str, default=None)
    parser.add_argument("--unsafe-prompt", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--cfg-src", type=float, default=None)
    parser.add_argument("--cfg-tar", type=float, default=None)
    parser.add_argument("--cfg-tar-eq-src", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--avg-target-embeddings", type=str, default=None)
    if image_to_image:
        parser.add_argument("--image", "--img-path", "--img_path", dest="image", required=True)
        parser.add_argument("--image-strength", "--img-strength", "--img_strength", dest="image_strength", type=float, default=None)


def add_mode_arguments(parser: argparse.ArgumentParser, *, legacy_steering: bool = False) -> None:
    if legacy_steering:
        parser.set_defaults(generation_mode="steer")
        parser.add_argument("--mode", "--steering-mode", dest="steering_mode", choices=("add", "replace"), default=None)
    else:
        parser.add_argument(
            "--generation-mode", "--mode", dest="generation_mode",
            choices=("baseline", "steer"), default="baseline",
        )
        parser.add_argument("--steering-mode", choices=("add", "replace"), default=None)
    parser.add_argument("--alpha-schedule", choices=("constant", "linear", "exp"), default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha-start", type=float, default=None)
    parser.add_argument("--alpha-end", type=float, default=None)
    parser.add_argument("--mu-schedule", choices=("constant", "linear", "exp"), default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu-start", type=float, default=None)
    parser.add_argument("--mu-end", type=float, default=None)


def value(args: argparse.Namespace, name: str, config: AppConfig, config_key: str, fallback: Any = None) -> Any:
    direct = getattr(args, name, None)
    return direct if direct is not None else config.get(config_key, fallback)


def resolved_config(args: argparse.Namespace, model: str) -> tuple[AppConfig, Path, Path]:
    overrides = {
        f"paths.{model}_checkpoint": args.checkpoint,
        "paths.output_root": args.output_root,
        "runtime.device": args.device,
        "runtime.device_number": args.device_number,
        "runtime.dtype": args.dtype,
    }
    config = load_config(args.config, overrides)
    checkpoint = get_model_checkpoint(config, model)
    output_root_value = config.get("paths.output_root")
    if not output_root_value:
        raise ValueError("No output root configured; use --output-root or paths.output_root")
    return config, checkpoint, Path(output_root_value)


def steering_parameters(args: argparse.Namespace, config: AppConfig, num_steps: int) -> SteeringParameters:
    params = SteeringParameters(
        mode=value(args, "steering_mode", config, "steering.mode", "add"),
        alpha_schedule=value(args, "alpha_schedule", config, "steering.alpha_schedule", "linear"),
        alpha=float(value(args, "alpha", config, "steering.alpha", 0.5)),
        alpha_start=float(value(args, "alpha_start", config, "steering.alpha_start", 0.9)),
        alpha_end=float(value(args, "alpha_end", config, "steering.alpha_end", 0.1)),
        mu_schedule=value(args, "mu_schedule", config, "steering.mu_schedule", "constant"),
        mu=float(value(args, "mu", config, "steering.mu", 0.5)),
        mu_start=float(value(args, "mu_start", config, "steering.mu_start", 0.5)),
        mu_end=float(value(args, "mu_end", config, "steering.mu_end", 0.5)),
    )
    validate_steering_parameters(params, num_steps=num_steps)
    return params


def validate_prompt_selection(args: argparse.Namespace, *, dataset_allowed: bool) -> None:
    dataset = getattr(args, "dataset", None)
    if not args.prompt and not (dataset_allowed and dataset):
        option = "--prompt or --dataset" if dataset_allowed else "--prompt"
        raise ValueError(f"Provide {option}")
    if args.prompt and dataset:
        raise ValueError("Use either --prompt or --dataset, not both")
    if args.generation_mode == "steer" and not (
        args.target_prompt or args.safe_prompt is not None or args.avg_target_embeddings
    ):
        raise ValueError("Steering requires --target-prompt, --safe-prompt, or --avg-target-embeddings")


def effective_dimensions(args: argparse.Namespace, config: AppConfig) -> tuple[int, int, int, int]:
    height = int(value(args, "height", config, "defaults.height", 1024))
    width = int(value(args, "width", config, "defaults.width", 1024))
    steps = int(value(args, "num_steps", config, "defaults.num_steps", 28))
    seed = int(value(args, "seed", config, "defaults.seed", 42))
    if min(height, width, steps) <= 0:
        raise ValueError("--height, --width, and --num-steps must be positive")
    return height, width, steps, seed
