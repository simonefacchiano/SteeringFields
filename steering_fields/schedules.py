"""Model-independent steering schedules and validation."""

from __future__ import annotations

from .common import SteeringParameters


def _scheduled_value(i: int, num_steps: int, schedule: str, constant: float, start: float, end: float) -> float:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0 <= i < num_steps:
        raise ValueError(f"step index {i} is outside [0, {num_steps})")
    if schedule == "constant":
        return float(constant)
    if num_steps == 1:
        return float(end)
    fraction = float(i) / float(num_steps - 1)
    if schedule == "linear":
        return float(start + fraction * (end - start))
    if schedule == "exp":
        if start <= 0.0 or end <= 0.0:
            raise ValueError("exponential schedule endpoints must be positive")
        return float(start * ((end / start) ** fraction))
    raise ValueError(f"Unknown schedule {schedule!r}; expected constant, linear, or exp")


def alpha_at_step(i: int, num_steps: int, schedule: str, alpha: float, alpha_start: float, alpha_end: float) -> float:
    return _scheduled_value(i, num_steps, schedule, alpha, alpha_start, alpha_end)


def mu_at_step(i: int, num_steps: int, schedule: str, mu: float, mu_start: float, mu_end: float) -> float:
    return _scheduled_value(i, num_steps, schedule, mu, mu_start, mu_end)


def clamp_alpha_value(alpha: float, mode: str, mu: float, eps: float = 1e-6) -> float:
    if mode == "add":
        return min(max(float(alpha), 0.0), 1.0)
    if mode == "replace":
        return min(max(float(alpha), 0.0), 1.0 + float(mu) - eps)
    raise ValueError(f"Unknown steering mode: {mode}")


def validate_steering_parameters(params: SteeringParameters, *, num_steps: int) -> None:
    if num_steps <= 0:
        raise ValueError("--num-steps must be >= 1")
    if params.mode not in {"add", "replace"}:
        raise ValueError("--steering-mode must be add or replace")
    if params.alpha_schedule not in {"constant", "linear", "exp"}:
        raise ValueError("Unsupported alpha schedule")
    if params.mu_schedule not in {"constant", "linear", "exp"}:
        raise ValueError("Unsupported mu schedule")
    if min(params.mu, params.mu_start, params.mu_end) < 0.0:
        raise ValueError("mu values must be non-negative")
    if params.alpha_schedule == "exp" and min(params.alpha_start, params.alpha_end) <= 0.0:
        raise ValueError("alpha endpoints must be positive for an exponential schedule")
    if params.mu_schedule == "exp" and min(params.mu_start, params.mu_end) <= 0.0:
        raise ValueError("mu endpoints must be positive for an exponential schedule")
    alpha_values = (params.alpha, params.alpha_start, params.alpha_end)
    if params.mode == "add" and any(value < 0.0 or value > 1.0 for value in alpha_values):
        raise ValueError("alpha values must be in [0, 1] for add mode")
    if params.mode == "replace" and any(value < 0.0 for value in alpha_values):
        raise ValueError("alpha values must be non-negative for replace mode")
    # Constant schedules can be validated exactly here. Scheduled combinations are
    # also clamped per step, matching the former scripts.
    if params.mode == "replace" and params.alpha_schedule == params.mu_schedule == "constant":
        if params.alpha >= 1.0 + params.mu:
            raise ValueError("replace mode requires alpha < 1 + mu")
