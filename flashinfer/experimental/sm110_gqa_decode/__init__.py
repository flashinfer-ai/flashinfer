"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Experimental exact-SM110 GQA decode implementation.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    from .backend import sm110_gqa_decode

    return sm110_gqa_decode(
        q,
        kv,
        sequence_lengths,
        out=out,
        q_scale=q_scale,
    )


def prepare_for_launch(
    inputs: dict[str, Any], num_splits: int | None = None
) -> dict[str, Any]:
    from .prepared import prepare_for_launch as prepare

    return prepare(inputs, num_splits=num_splits)


def launch_prepared(prepared: dict[str, Any]) -> torch.Tensor:
    from .prepared import launch_prepared as launch

    return launch(prepared)


__all__ = ["decode", "prepare_for_launch", "launch_prepared"]
