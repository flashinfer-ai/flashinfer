# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small helpers shared by explicit trace solution modules."""

from __future__ import annotations

import contextlib
import os

import torch

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_FALSE_VALUES = {"0", "false", "off", "no"}
_AUTOTUNED_SIGNATURES = set()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _parse_tuning_buckets(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(item) for item in value.replace(",", " ").split())


def _signature_value(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((key, _signature_value(item)) for key, item in value.items())
        )
    return ("value", repr(value))


@contextlib.contextmanager
def _remember_success(signature, inner_context):
    with inner_context:
        yield
    _AUTOTUNED_SIGNATURES.add(signature)


def solution_autotune(*signature_values):
    """Autotune the first successful direct solution call for a signature.

    Enabled by default. Set ``FLASHINFER_TRACE_SOLUTION_AUTOTUNE=0`` to disable.
    Optional cache/bucket controls mirror ``flashinfer.autotune`` via
    ``FLASHINFER_TRACE_SOLUTION_AUTOTUNE_CACHE``,
    ``FLASHINFER_TRACE_SOLUTION_AUTOTUNE_BUCKETS``, and
    ``FLASHINFER_TRACE_SOLUTION_AUTOTUNE_ROUND_UP``.
    """

    if not _env_flag("FLASHINFER_TRACE_SOLUTION_AUTOTUNE", True):
        return contextlib.nullcontext()

    cache = os.environ.get("FLASHINFER_TRACE_SOLUTION_AUTOTUNE_CACHE")
    tuning_buckets = _parse_tuning_buckets(
        os.environ.get("FLASHINFER_TRACE_SOLUTION_AUTOTUNE_BUCKETS")
    )
    round_up_env = os.environ.get("FLASHINFER_TRACE_SOLUTION_AUTOTUNE_ROUND_UP")
    round_up = (
        None
        if round_up_env is None
        else _env_flag("FLASHINFER_TRACE_SOLUTION_AUTOTUNE_ROUND_UP", False)
    )

    from flashinfer import autotune
    from flashinfer.autotuner import AutoTuner

    if AutoTuner.get().is_tuning_mode:
        return contextlib.nullcontext()

    signature = None
    if signature_values:
        signature = (
            ("config", cache, tuning_buckets, round_up),
            *(_signature_value(value) for value in signature_values),
        )
        if signature in _AUTOTUNED_SIGNATURES:
            return contextlib.nullcontext()

    inner_context = autotune(
        True,
        cache=cache,
        tuning_buckets=tuning_buckets,
        round_up=round_up,
    )
    if signature is None:
        return inner_context
    return _remember_success(signature, inner_context)


def default_paged_metadata(batch_size: int, num_pages: int, device):
    pages_per_seq = max(1, num_pages // max(1, batch_size))
    indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * pages_per_seq
    )
    indices = torch.arange(int(indptr[-1].item()), dtype=torch.int32, device=device)
    return indptr, indices


def full_last_page_len(kv_indptr, page_size: int):
    return torch.full(
        (kv_indptr.numel() - 1,),
        page_size,
        dtype=torch.int32,
        device=kv_indptr.device,
    )


def workspace(device):
    return torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
