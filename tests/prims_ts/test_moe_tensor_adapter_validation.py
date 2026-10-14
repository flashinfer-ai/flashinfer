# Copyright (c) 2026 by FlashInfer team.
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

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.moe.tensor_adapter import (
    _check_fp8_scale_storage,
    _validate_weight_storage,
)
from flashinfer.prims_ts.moe.support import _validate_gemm1_oa_params
from flashinfer.tllm_enums import ActivationType, WeightLayout


def _major_k_cfg(weight_bits: int):
    return SimpleNamespace(
        weight_layout=int(WeightLayout.MajorK),
        weight_dtype_tma_bits=weight_bits,
    )


@pytest.mark.parametrize(
    ("dtype", "weight_bits", "shape"),
    [
        (torch.bfloat16, 16, (4, 256, 128)),
        (torch.float8_e4m3fn, 8, (4, 256, 128)),
        (torch.uint8, 4, (4, 256, 64)),
    ],
)
def test_major_k_weight_storage_accepts_expected_shape(dtype, weight_bits, shape):
    _validate_weight_storage(
        name="weights",
        tensor=torch.empty(shape, dtype=dtype),
        cfg=_major_k_cfg(weight_bits),
        num_experts=4,
        out_hidden=256,
        in_hidden=128,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 256),
        (3, 256, 128),
        (4, 255, 128),
        (4, 256, 127),
    ],
)
def test_major_k_weight_storage_rejects_malformed_shape(shape):
    with pytest.raises(ValueError, match="invalid MajorK shape"):
        _validate_weight_storage(
            name="weights",
            tensor=torch.empty(shape, dtype=torch.bfloat16),
            cfg=_major_k_cfg(16),
            num_experts=4,
            out_hidden=256,
            in_hidden=128,
        )


def test_fp4_major_k_weight_storage_requires_packed_k():
    with pytest.raises(ValueError, match="expected"):
        _validate_weight_storage(
            name="weights",
            tensor=torch.empty((4, 256, 128), dtype=torch.uint8),
            cfg=_major_k_cfg(4),
            num_experts=4,
            out_hidden=256,
            in_hidden=128,
        )


def test_mx_scale_storage_rejects_undersized_or_strided_tensor():
    with pytest.raises(ValueError, match="too small"):
        _check_fp8_scale_storage(
            "scale", torch.empty(7, dtype=torch.uint8), min_numel=8
        )
    with pytest.raises(ValueError, match="contiguous"):
        _check_fp8_scale_storage(
            "scale",
            torch.empty((4, 4), dtype=torch.uint8).t(),
            min_numel=8,
        )


def test_gemm1_oa_validation_uses_local_expert_count():
    runner = SimpleNamespace(num_local_experts=4)
    inputs = SimpleNamespace(hidden_states=torch.empty((2, 8)))
    ok, reason = _validate_gemm1_oa_params(
        runner,
        inputs,
        ActivationType.Swiglu,
        {
            "num_experts": 16,
            "gemm1_alpha": torch.ones(4, dtype=torch.float32),
        },
    )

    assert ok, reason
