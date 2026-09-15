# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU coverage for the dtype and storage contracts of the decode compiler."""

from contextlib import nullcontext
import sys
from types import ModuleType

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

import cutlass
import cutlass.cute as cute

from flashinfer.attention.prims_ts import decode as decode_module
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    FmhaDecodeConfig,
    make_decode_config,
)


@pytest.fixture
def captured_decode_compiles(monkeypatch: pytest.MonkeyPatch):
    """Exercise real fake-tensor construction without compiling a GPU kernel."""

    kernel_name = "flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel"
    kernel_module = ModuleType(kernel_name)
    kernel_module.fmha_decode_launch = object()
    monkeypatch.setitem(sys.modules, kernel_name, kernel_module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    captured = []

    def compile_without_gpu(*args, **kwargs):
        result = object()
        captured.append((args, kwargs, result))
        return result

    monkeypatch.setattr(cute, "compile", compile_without_gpu)
    decode_module._get_compiled_decode.cache_clear()
    try:
        yield captured
    finally:
        decode_module._get_compiled_decode.cache_clear()


def _compile_spec(q_dtype, kv_dtype, output_dtype):
    config = make_decode_config(
        headdim=128,
        seq_len_q=1,
        seq_len_kv=256,
        batch_size=2,
        num_heads_q=8,
        num_heads_kv=1,
        qkv_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=output_dtype,
        qkv_layout="pagedKv",
        num_tokens_per_page=64,
        auto_tuner=False,
    )
    launch_spec = decode_module._decode_launch_spec_from_config(
        config,
        batch_size=2,
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=1,
        max_active_clusters=1,
    )
    dtype_keys = {
        cutlass.Float16: "float16",
        cutlass.BFloat16: "bfloat16",
        cutlass.Float8E4M3FN: "float8_e4m3fn",
    }
    return decode_module._make_decode_compile_spec(
        launch_spec,
        device_index=0,
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=128,
        page_size=64,
        max_kv_len=256,
        seq_len_q=1,
        q_dtype_key=dtype_keys[q_dtype],
        output_dtype_key=dtype_keys[output_dtype],
        use_packed_q=False,
        kv_prefix_mode="dynamic",
        kv_lengths_mode="dynamic",
    )


@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype", "output_dtype"),
    (
        (cutlass.Float16, cutlass.Float16, cutlass.Float16),
        (cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16),
        (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Float16),
        (cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.BFloat16),
        (cutlass.BFloat16, cutlass.Float4E2M1FN, cutlass.BFloat16),
        (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Float16),
        (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN),
    ),
)
def test_decode_compile_preserves_kv_storage_contract(
    captured_decode_compiles, q_dtype, kv_dtype, output_dtype
):
    spec = _compile_spec(q_dtype, kv_dtype, output_dtype)
    compiled_main, compiled_reducer = decode_module._get_compiled_decode(spec)

    assert len(captured_decode_compiles) == 1
    args, _, result = captured_decode_compiles[0]
    assert compiled_main is result
    assert compiled_reducer is None
    q, k, v, k_sf, v_sf, out = args[1:7]
    config = next(arg for arg in args if isinstance(arg, FmhaDecodeConfig))
    assert (config.q_dtype, config.kv_dtype, config.out_dtype) == (
        q_dtype,
        kv_dtype,
        output_dtype,
    )
    assert q.element_type == q_dtype
    assert out.element_type == output_dtype
    assert q.shape[1:] == out.shape[1:] == (8, 128)
    assert not isinstance(q.shape[0], int)
    nvfp4 = kv_dtype == cutlass.Float4E2M1FN
    storage_width = 64 if nvfp4 else 128
    for cache in (k, v):
        assert cache.element_type == (cutlass.Uint8 if nvfp4 else kv_dtype)
        assert cache.shape[1:] == (1, 64, storage_width)
        assert cache.stride[1:] == (64 * storage_width, storage_width, 1)
        assert not isinstance(cache.shape[0], int)
        assert not isinstance(cache.stride[0], int)
    for scales in (k_sf, v_sf):
        if nvfp4:
            assert scales.element_type == cutlass.Float8E4M3FN
            assert scales.shape[1:] == (1, 64, 8)
            assert scales.stride == (512, 512, 8, 1)
        else:
            assert scales.element_type == cutlass.Uint8
            assert scales.shape == (1,)


def test_decode_compile_cache_distinguishes_kv_dtype(captured_decode_compiles):
    specs = [
        _compile_spec(cutlass.BFloat16, kv_dtype, cutlass.BFloat16)
        for kv_dtype in (
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        )
    ]
    assert len(set(specs)) == 3
    compiled = [decode_module._get_compiled_decode(spec) for spec in specs]
    assert len(captured_decode_compiles) == 3
    assert len({id(main) for main, _ in compiled}) == 3
    for spec, expected in zip(specs, compiled, strict=True):
        assert decode_module._get_compiled_decode(spec) is expected
    assert len(captured_decode_compiles) == 3
