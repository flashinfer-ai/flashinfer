# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planning-only sparse MLA contracts, with CPU buffers and a fake compiler."""

from functools import cache
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("quantized_output", [False, True])
def test_sparse_mla_plan_without_compilation(monkeypatch, quantized_output):
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts import mla_decode as mla

    class Kernel:
        kv_cache_format = "native"
        extra_kv_cache_format = "native"
        has_nvfp4_kv = False

        def compile_signature(self):
            return (
                self.assume_valid_prefix,
                self.direct_sparse_capacities,
                self.output_quant,
                self.fuse_sparse_epilogue,
            )

    quant = (128, "fp32", 4) if quantized_output else None

    def resolve(device, rows, heads, **kwargs):
        spec = mla._MLADecodeCompileSpec(
            device_index=0,
            kernel_signature=("unfinalized",),
            num_qo_heads=heads,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            page_size=1,
            q_dtype_key="bfloat16",
            out_dtype_key="float8_e4m3fn" if quantized_output else "bfloat16",
            max_q_len=1,
            packed_query=False,
            has_kernel_workspace=False,
            split_kv=1,
            kernel=Kernel(),
        )
        return (
            device,
            SimpleNamespace(family="2cta", split_kv=1),
            spec,
            {},
            0,
            128,
            quant,
        )

    compiled = []

    @cache
    def compiler(spec):
        result = object()
        compiled.append((spec, result))
        return result

    monkeypatch.setattr(mla, "_resolve_sparse_mla_plan", resolve)
    monkeypatch.setattr(mla, "_get_compiled_mla_decode", compiler)
    wrapper = mla.BatchSparseMLADecodePagedTSWrapper()
    args = (torch.device("cpu"), 4, 64)
    kwargs = dict(max_topk=128, max_extra_topk=256, assume_valid_prefix=True)
    if quantized_output:
        kwargs.update(
            output_quant_block_size=128, out_scale_format="fp32", num_output_groups=4
        )

    wrapper.plan(*args, compile=False, **kwargs)
    spec = wrapper._state["compile_spec"]
    assert wrapper._state["compiled"] is None and not compiled
    assert compiler.cache_info().currsize == 0
    assert spec.kernel_signature != ("unfinalized",)
    assert spec.kernel.assume_valid_prefix is True
    assert spec.kernel.output_quant == quant
    with pytest.raises(RuntimeError, match=r"plan\(compile=True\)"):
        wrapper.run(None, None, None, None)

    wrapper.plan(*args, **kwargs)
    kernel = wrapper._state["compiled"]
    assert kernel is not None and compiled == [(spec, kernel)]
    wrapper.plan(*args, compile=False, **kwargs)
    assert wrapper._state["compiled"] is None
    wrapper.plan(*args, **kwargs)
    assert wrapper._state["compiled"] is kernel and len(compiled) == 1
