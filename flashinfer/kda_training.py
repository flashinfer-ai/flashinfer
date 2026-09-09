"""Paired Blackwell forward/backward training API for recurrent KDA."""

from ._kda_training_impl import (
    RecurrentKDATrainingContext,
    _load_training_module,
    _select_training_route as _select_training_route,
    _validate_forward_inputs as _validate_forward_inputs,
    recurrent_kda_training_backward,
    recurrent_kda_training_forward,
)


def _get_training_module(device):
    """Internal indirection seam used by the training API tests."""

    return _load_training_module(device)


__all__ = [
    "RecurrentKDATrainingContext",
    "recurrent_kda_training_backward",
    "recurrent_kda_training_forward",
]
