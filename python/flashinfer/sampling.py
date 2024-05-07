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


def sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor
):
    return _kernels.sampling_from_probs(probs, uniform_samples)

def top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    p: float
):
    return _kernels.top_p_sampling_from_probs(probs, uniform_samples, p)

def top_k_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    k: int
):
    return _kernels.top_k_sampling_from_probs(probs, uniform_samples, k)
