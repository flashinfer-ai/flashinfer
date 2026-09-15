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

"""CPU acceptance coverage for mixed-precision PrimTS decode bindings."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts.decode import (
    _DecodeRuntime,
    _validate_decode_output_aliasing,
)


_FP8 = torch.float8_e4m3fn


def test_attention_ts_decode_nvfp4_trace_fixed_metadata() -> None:
    """Trace packed cache scales alongside the fixed page table and lengths."""

    nvfp4_q = torch.empty((1, 8, 64), dtype=torch.bfloat16)
    nvfp4_cache = torch.empty((2, 2, 1, 64, 32), dtype=torch.uint8)
    nvfp4_sf = torch.empty((2, 1, 64, 4), dtype=torch.float8_e4m3fn)
    nvfp4_trace = batch_decode_with_paged_kv_cache.fi_trace(
        q=nvfp4_q,
        paged_kv_cache=nvfp4_cache,
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens_kv=torch.tensor([128], dtype=torch.int32),
        kv_scale_factors=(nvfp4_sf, nvfp4_sf),
    )
    assert nvfp4_trace["axes"]["head_dim"]["value"] == 64
    assert nvfp4_trace["axes"]["kv_storage_head_dim"]["value"] == 32
    assert nvfp4_trace["inputs"]["k_sf_cache"]["dtype"] == "float8_e4m3fn"
    assert nvfp4_trace["inputs"]["v_sf_cache"]["dtype"] == "float8_e4m3fn"
    assert nvfp4_trace["inputs"]["block_tables"]["dtype"] == "int32"
    assert nvfp4_trace["inputs"]["seq_lens_kv"]["dtype"] == "int32"


@pytest.mark.parametrize("entrypoint", ("validated", "trusted", "standalone"))
@pytest.mark.parametrize("cache_form", ("tuple", "combined"))
@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype"),
    (
        (torch.bfloat16, _FP8),
        (torch.bfloat16, torch.uint8),
        (_FP8, torch.uint8),
    ),
    ids=("bf16q-fp8kv", "bf16q-nvfp4kv", "fp8q-nvfp4kv"),
)
def test_attention_ts_decode_forwards_mixed_precision_runtime(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    cache_form: str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> None:
    """Wrapper and caller-workspace paths preserve K/V types and scale storage."""

    from flashinfer.attention.prims_ts import decode as decode_module

    q = torch.empty((1, 8, 64), dtype=q_dtype)
    storage_head_dim = 32 if kv_dtype == torch.uint8 else 64
    cache = torch.empty((2, 2, 1, 64, storage_head_dim), dtype=kv_dtype)
    paged_kv_cache = cache if cache_form == "combined" else (cache[:, 0], cache[:, 1])
    scale_storage = tuple(torch.empty(2048, dtype=torch.uint8) for _ in range(2))
    scales = (
        tuple(storage[:512].view(_FP8).view(2, 1, 64, 4) for storage in scale_storage)
        if kv_dtype == torch.uint8
        else None
    )
    seq_lens = torch.tensor((128,), dtype=torch.int32)
    block_tables = torch.tensor(((1, 0),), dtype=torch.int32)
    wrapper = BatchDecodePagedTSWrapper()
    wrapper._plan_state = type(
        "_MixedPrecisionDecodePlanState",
        (),
        {
            "device": q.device,
            "batch_size": 1,
            "seq_len_q": 1,
            "use_packed_q": False,
            "num_qo_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 64,
            "page_size": 64,
            "max_kv_len": 128,
            "q_dtype": q_dtype,
            "kv_dtype": kv_dtype,
            "output_dtype": q_dtype,
            "mask_type": "dense",
            "workspace_buffer": torch.empty(8, dtype=torch.uint8),
            "workspace": object(),
            "compiled_main": object(),
            "compiled_reducer": None,
            "planned_seq_lens_host": None,
            "planned_seq_lens_device": None,
        },
    )()
    # The CUDA-only entry guards are bypassed; cache/scale normalization and
    # all runtime metadata value checks still execute for validate=True.
    monkeypatch.setattr(decode_module, "_validate_q", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        decode_module,
        "_validate_block_table_metadata",
        lambda *_args: (q.device, 1, 2),
    )

    def launch(
        runtime, *, seq_lens: torch.Tensor, block_tables: torch.Tensor, **_kwargs
    ):
        assert runtime.q is q
        assert runtime.k_cache.dtype == runtime.v_cache.dtype == kv_dtype
        assert runtime.k_cache.shape[-1] == storage_head_dim
        assert runtime.k_cache.data_ptr() == cache[:, 0].data_ptr()
        assert runtime.v_cache.data_ptr() == cache[:, 1].data_ptr()
        assert runtime.out.shape == q.shape
        assert runtime.out.dtype == q_dtype
        assert runtime.bmm1_scale == 0.125
        assert runtime.bmm2_scale == 0.75
        assert seq_lens.tolist() == [128]
        assert block_tables.tolist() == [[1, 0]]
        if scales is not None:
            assert runtime.k_sf_cache is scales[0]
            assert runtime.v_sf_cache is scales[1]
        else:
            assert runtime.k_sf_cache.dtype == runtime.v_sf_cache.dtype == torch.uint8
        return runtime.out

    monkeypatch.setattr(decode_module, "_launch_decode", launch)
    if entrypoint == "standalone":
        monkeypatch.setattr(
            decode_module, "_validate_runtime_device", lambda _device: 0
        )

        def resolve_spec(*policy_args):
            assert policy_args[8:11] == (
                decode_module._dtype_key(q_dtype),
                decode_module._dtype_key(kv_dtype),
                decode_module._dtype_key(q_dtype),
            )
            return SimpleNamespace(
                scratch_shapes=((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1,)),
                config=SimpleNamespace(use_separate_reduction_kernel=False),
            )

        monkeypatch.setattr(decode_module, "_resolve_decode_launch_spec", resolve_spec)
        monkeypatch.setattr(
            decode_module,
            "_make_decode_compile_spec",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(
            decode_module, "_get_compiled_decode", lambda _spec: (object(), None)
        )

    def invoke(workspace_buffer):
        if entrypoint == "standalone":
            return decode_module.prims_ts_batch_decode_with_kv_cache(
                q,
                paged_kv_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                128,
                kv_scale_factors=scales,
                bmm1_scale=0.125,
                bmm2_scale=0.75,
            )
        wrapper._plan_state.workspace_buffer = workspace_buffer
        return wrapper.run(
            q,
            paged_kv_cache,
            seq_lens,
            block_tables,
            kv_scale_factors=scales,
            bmm1_scale=0.125,
            bmm2_scale=0.75,
            validate=entrypoint == "validated",
        )

    invoke(torch.zeros(2048, dtype=torch.uint8))
    if scales is not None and entrypoint != "trusted":
        for name, storage in zip(
            ("k_sf_cache", "v_sf_cache"), scale_storage, strict=True
        ):
            with pytest.raises(
                ValueError,
                match=rf"workspace_buffer must not overlap {name} storage",
            ):
                invoke(storage)


@pytest.mark.parametrize("aliased_name", ("k_sf_cache", "v_sf_cache"))
def test_attention_ts_decode_output_cannot_alias_nvfp4_scales(
    aliased_name: str,
) -> None:
    """Both scale allocations remain live until the decode launch finishes."""

    runtime = _DecodeRuntime(
        q=torch.empty(8),
        k_cache=torch.empty(8),
        v_cache=torch.empty(8),
        k_sf_cache=torch.empty(8),
        v_sf_cache=torch.empty(8),
        out=torch.empty(8),
        num_physical_pages=1,
        k_page_stride=8,
        v_page_stride=8,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )
    runtime = replace(runtime, **{aliased_name: runtime.out})
    with pytest.raises(
        ValueError, match=rf"out must not overlap {aliased_name} storage"
    ):
        _validate_decode_output_aliasing(
            runtime,
            seq_lens=torch.empty(8),
            qo_indptr=None,
            block_tables=torch.empty(8),
            workspace_buffer=torch.empty(8),
        )
