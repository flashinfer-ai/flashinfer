"""
Copyright (c) 2024 by FlashInfer team.

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

import torch
from typing import Optional

try:
    from . import _kernels
except ImportError as e:
    import os
    import logging

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


class SegmentGEMMWrapper:
    r"""Wrapper for segment GEMM kernels."""

    def __init__(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.SegmentGEMMWrapper()

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        r"""Reset the workspace buffer.

        Parameters
        ----------
        new_workspace_buffer : torch.Tensor
            The new workspace buffer, the device of the new workspace buffer should
            be the same as the device of the input tensors.
        """
        self._workspace_buffer = new_workspace_buffer

    def register_problem(
        self,
        batch_size: int,
        d_in: int,
        d_out: int,
        weight_column_major: bool,
        seg_lens: Optional[torch.Tensor] = None,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
    ):
        if seg_lens is None and seg_indptr is None:
            raise ValueError("Either seg_lens or seg_indptr should be provided.")
        if seg_indptr is None:
            seg_indptr = torch.cat(
                [
                    torch.tensor([0], device=seg_lens.device, dtype=seg_lens.dtype),
                    seg_lens.cumsum(0),
                ],
                dim=0,
            )
        self._wrapper.register_problem(
            self._workspace_buffer,
            batch_size,
            d_in,
            d_out,
            weight_column_major,
            seg_indptr,
            weight_indices,
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of segment GEMM."""
        return self._wrapper.forward(x, weights)
