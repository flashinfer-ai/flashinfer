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

# mypy: disable-error-code="attr-defined"
try:
    from . import _kernels
except ImportError as e:
    import logging
    import os

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


def silu_and_mul(output: torch.Tensor, input: torch.Tensor):
    r"""Fused SiLU and Mul.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    _kernels.silu_and_mul(output, input)
