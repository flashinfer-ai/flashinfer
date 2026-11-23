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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows,width,coeff_offset",
    [
        (512, 256, 0),
        (768, 256, 0),
        (512, 288, 0),
        (1, 256, 0),
        (1, 288, 0),
        (1, 288, 1),
    ],
)
def test_nvfp4_batch_m_output_requires_exact_row_pitch(rows, width, coeff_offset):
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.moe.tensor_adapter import build_nvfp4_launch_io

    cfg = map_trtllm_nvfp4_moe_tactic([1024, 0], sm_arch=107).fc2.cfg.build()
    data = torch.empty((512, 128), dtype=torch.uint8)
    scales = torch.empty((512, 16), dtype=torch.float8_e4m3fn)
    weights = torch.empty((2, 256, 128), dtype=torch.uint8)
    metadata = torch.zeros(2, dtype=torch.int32)
    scale = torch.ones(2)
    kwargs = dict(
        fc="fc2",
        cfg=cfg,
        hidden_states=data,
        hidden_states_scale=scales,
        gemm1_weights=weights,
        gemm1_weights_scale=scales,
        gemm2_weights=weights,
        gemm2_weights_scale=scales,
        gemm1_output=data,
        gemm1_output_scale=scales,
        gemm2_output=torch.empty((rows, width), dtype=torch.bfloat16),
        output1_scale_scalar=scale,
        output1_scale_gate_scalar=scale,
        output2_scale_scalar=scale,
        tile_idx=metadata,
        mn_limit=metadata,
        route_map=torch.zeros(512, dtype=torch.int32),
        num_non_exiting_ctas=metadata,
        total_num_padded_tokens=metadata,
        routed_token_capacity=512,
        activation_type=int(ActivationType.Swiglu),
        num_experts=2,
        num_tokens=17,
        top_k=2,
        intermediate_size=256,
        hidden_size=256,
    )
    if rows == 1:
        from flashinfer.prims_ts.moe.epilogue_policy import fused_finalize_config

        kwargs.update(
            cfg=fused_finalize_config(cfg),
            num_tokens=1,
            finalize_output=kwargs["gemm2_output"][:, :256],
            finalize_scales=torch.ones(2 + coeff_offset, dtype=torch.bfloat16)[
                coeff_offset:
            ].view(1, 2),
            finalize_route_map=kwargs["route_map"],
        )
    if width != 256 and rows > 1:
        with pytest.raises(
            ValueError, match="Batch-M output requires storage row stride"
        ):
            build_nvfp4_launch_io(**kwargs)
    else:
        io = build_nvfp4_launch_io(**kwargs)
        assert io["c0_dp"]._pointer == kwargs["gemm2_output"].data_ptr()
        if rows == 1:
            assert io["finalize_scale_dp"]._assumed_align == (2 if coeff_offset else 16)
