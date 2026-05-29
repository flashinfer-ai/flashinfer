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

"""FlashInfer sm90 solution for segment_gemm_run."""

from types import SimpleNamespace

import torch

from flashinfer.gemm import SegmentGEMMWrapper
from flashinfer.trace.solutions._helpers import workspace, solution_autotune

definition = "segment_gemm_run"
api = "flashinfer.gemm.gemm_base.SegmentGEMMWrapper.run"
backend = "sm90"
inputs = ("x", "weights")
outputs = ("output",)
api_kwargs = {"x": "x", "weights": "weights"}
requires_setup = True

_state = None


def _require_state():
    if _state is None:
        raise RuntimeError("Call setup(...) before benchmarking run(...).")
    return _state


def setup(x, weights):
    global _state
    batch_size = int(weights.shape[0])
    rows_per_segment = x.shape[0] // batch_size
    seg_lens = torch.full(
        (batch_size,), rows_per_segment, dtype=torch.int32, device=x.device
    )
    d_out = weights.shape[2]
    wrapper = SegmentGEMMWrapper(workspace(x.device), backend=backend)
    _state = SimpleNamespace(
        wrapper=wrapper,
        batch_size=batch_size,
        seg_lens=seg_lens,
        out=torch.empty((x.shape[0], d_out), dtype=x.dtype, device=x.device),
    )


def run(x, weights):
    with solution_autotune(
        definition,
        backend,
        x,
        weights,
    ):
        state = _require_state()
        return state.wrapper.run(
            x,
            weights,
            batch_size=state.batch_size,
            weight_column_major=False,
            out=state.out,
            seg_lens=state.seg_lens,
        )
