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

"""Exercise the real MLA launcher without importing accelerator packages.

Use --confcutdir=tests/attention --noconftest and disable plugin autoload for
standalone CPU collection. This bounds collection and excludes both conftests,
which import accelerator packages.
"""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = (
    Path(__file__).resolve().parents[2] / "flashinfer/attention/prims_ts/mla_decode.py"
)


@pytest.fixture
def launch(monkeypatch):
    original_import = builtins.__import__
    attempts = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "cuda",
            "cutlass",
            "flashinfer",
            "triton",
            "nvidia",
        }:
            attempts.append(name)
            raise AssertionError(f"accelerator import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    function = next(
        node
        for node in ast.parse(_SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_mla_decode"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(function)
    # The launcher only compares the cache dtype against torch.uint8.
    namespace = {"torch": SimpleNamespace(uint8="uint8")}
    exec(compile(module, str(_SOURCE), "exec"), namespace)
    yield namespace[function.name]
    assert not attempts


class _TensorView:
    """Only view plumbing is needed to reach the compiled-call boundary."""

    dtype = None

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        return self

    def permute(self, *dims):
        return self

    def transpose(self, *dims):
        return self

    def view(self, *shape):
        return self


@pytest.mark.parametrize("packed_query", [False, True], ids=["fixed", "packed"])
@pytest.mark.parametrize("quantized", [False, True], ids=["none", "quantized"])
def test_launch_preserves_positional_output_quantizer_slot(
    launch, packed_query, quantized
):
    output_quant_args = (object(), object()) if quantized else None
    query_shape = (6, 64, 512) if packed_query else (2, 3, 64, 512)
    runtime = SimpleNamespace(
        query=_TensorView(query_shape),
        normalized_cache=_TensorView((8, 1, 512)),
        extra_cache=_TensorView((8, 1, 512)),
        out=_TensorView(query_shape),
        sparse_inputs=tuple(object() for _ in range(6)),
        softmax_scale=0.125,
        output_scale=0.75,
        output_quant_args=output_quant_args,
    )
    workspace = SimpleNamespace(lse=_TensorView((2, 3, 64)), kernel_workspace=None)
    block_tables = _TensorView((2, 16))
    seq_lens = object()
    qo_indptr = object() if packed_query else None
    calls = []

    # A reloaded TVM-FFI Function has neither Python defaults nor keyword binding.
    def compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_offsets,
        out,
        lse,
        kernel_workspace,
        split_kv,
        cache_seqs,
        cu_seqlens_q,
        sparse_inputs,
        softmax_scale,
        output_scale,
        output_quant_args,
        max_q_len,
        /,
    ):
        calls.append(
            (
                q_latent,
                q_rope,
                c_latent,
                c_rope,
                page_offsets,
                out,
                lse,
                kernel_workspace,
                split_kv,
                cache_seqs,
                cu_seqlens_q,
                sparse_inputs,
                softmax_scale,
                output_scale,
                output_quant_args,
                max_q_len,
            )
        )

    result = launch(
        runtime,
        block_tables=block_tables,
        seq_lens_kv=seq_lens,
        qo_indptr=qo_indptr,
        packed_query=packed_query,
        kv_lora_rank=512,
        split_kv=1,
        workspace=workspace,
        compiled=compiled,
        max_q_len=3,
    )

    assert result is runtime.out
    assert calls == [
        (
            runtime.query,
            runtime.query,
            runtime.normalized_cache,
            runtime.extra_cache,
            block_tables,
            runtime.out,
            workspace.lse,
            None,
            1,
            seq_lens,
            qo_indptr,
            runtime.sparse_inputs,
            runtime.softmax_scale,
            runtime.output_scale,
            output_quant_args,
            3,
        )
    ]
    assert calls[0][-2] is output_quant_args
