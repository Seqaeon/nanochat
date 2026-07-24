"""Strict architecture-and-optimizer benchmark interface."""

from .api import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
    count_model_state_elements,
)

__all__ = [
    "ModelSpec",
    "OptimizerBundle",
    "OptimizerSpec",
    "Submission",
    "assert_model_state",
    "count_model_state_elements",
]
