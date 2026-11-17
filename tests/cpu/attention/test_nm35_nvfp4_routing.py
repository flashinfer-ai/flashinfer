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

"""Routing rules of ``backend="nm35-nvfp4"``: functions of the shape and,
for decode, of whether the kernels use the fused NVFP4 multiply."""

import logging
import threading

import pytest
import torch

from flashinfer import nm35_nvfp4 as nm35

MTP, BASE = nm35.DECODE_MTP, nm35.DECODE_BASE
MIB = 1 << 20


@pytest.mark.parametrize(
    "batch_size,q_len,max_kv_len,expected",
    [
        # A KV of at most 8192 tokens runs on trtllm-gen, whatever the rest.
        (8, 1, 1024, None),
        (1, 3, 1024, None),
        (64, 4, 4096, None),
        (2, 6, 4096, None),
        (1, 1, 8192, None),
        (256, 2, 8192, None),
        # Longer KV: 2, 4 and 8 tokens on "mtp"; 1, 3 and 5 on "base".
        (1, 1, 8193, BASE),
        (256, 1, 1 << 20, BASE),
        (32, 3, 131072, BASE),
        (32, 5, 131072, BASE),
        (1, 2, 8193, MTP),
        (256, 4, 1 << 20, MTP),
        (16, 8, 131072, MTP),
        # 6 tokens: "mtp" for one or two requests, trtllm-gen above; 7 on trtllm-gen.
        (1, 6, 131072, MTP),
        (2, 6, 131072, MTP),
        (3, 6, 131072, None),
        (1, 7, 131072, None),
        # Outside what the kernels serve.
        (257, 1, 131072, None),
        (4, 9, 131072, None),
        (4, 0, 131072, None),
        (0, 1, 131072, None),
    ],
)
def test_select_decode_kernel(batch_size, q_len, max_kv_len, expected):
    kernel = nm35.select_decode_kernel(batch_size, q_len, max_kv_len, fused=True)
    assert kernel == expected


@pytest.mark.parametrize(
    "batch_size,q_len,max_kv_len,expected",
    [
        # A KV of at most 8192 tokens runs on trtllm-gen, as on the fused build.
        (1, 1, 8192, None),
        (1, 2, 8192, None),
        # 2 tokens: "mtp" at every batch size.
        (1, 2, 8193, MTP),
        (8, 2, 131072, MTP),
        (256, 2, 1 << 20, MTP),
        # One request: 1, 3 or 5 tokens on "base", 4 or 8 on "mtp".
        (1, 1, 8193, BASE),
        (1, 3, 1 << 20, BASE),
        (1, 5, 131072, BASE),
        (1, 4, 1 << 20, MTP),
        (1, 8, 131072, MTP),
        # Two requests: only 5 and 8 tokens.
        (2, 5, 1 << 20, BASE),
        (2, 8, 8193, MTP),
        (2, 1, 131072, None),
        (2, 3, 131072, None),
        (2, 4, 131072, None),
        # 3 to 15 requests run on trtllm-gen.
        (3, 5, 131072, None),
        (8, 8, 131072, None),
        (15, 1, 1 << 20, None),
        # 16 or more requests: 8 tokens at any KV, 5 above 32K, 1 above 64K.
        (16, 8, 8193, MTP),
        (256, 8, 1 << 20, MTP),
        (16, 5, 32768, None),
        (16, 5, 32769, BASE),
        (256, 5, 1 << 20, BASE),
        (16, 1, 65536, None),
        (16, 1, 65537, BASE),
        (256, 1, 1 << 20, BASE),
        (256, 3, 1 << 20, None),
        (256, 4, 1 << 20, None),
        # 6 and 7 tokens run on trtllm-gen.
        (1, 6, 131072, None),
        (2, 6, 131072, None),
        (1, 7, 131072, None),
        # Outside what the kernels serve.
        (257, 2, 131072, None),
        (1, 9, 131072, None),
        (0, 2, 131072, None),
    ],
)
def test_select_decode_kernel_non_fused(batch_size, q_len, max_kv_len, expected):
    kernel = nm35.select_decode_kernel(batch_size, q_len, max_kv_len, fused=False)
    assert kernel == expected


def test_non_fused_routes_a_subset_of_the_fused_ones():
    for batch_size in range(1, 258):
        for q_len in range(10):
            for max_kv_len in (8192, 8193, 16384, 32769, 65537, 1 << 20):
                args = (batch_size, q_len, max_kv_len)
                kernel = nm35.select_decode_kernel(*args, fused=False)
                if kernel is not None:
                    assert kernel == nm35.select_decode_kernel(*args, fused=True)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "q_len,expected",
    [(1, BASE), (2, None), (3, BASE), (4, None), (5, BASE), (6, None), (8, None)],
)
def test_select_decode_kernel_ragged(q_len, expected, fused):
    # Requests of different query lengths never reach "mtp", which declines them.
    batch_sizes = (1, 2, 32) if fused else (1,)
    for batch_size in batch_sizes:
        kernel = nm35.select_decode_kernel(
            batch_size, q_len, 131072, ragged=True, fused=fused
        )
        assert kernel == expected
    assert nm35.select_decode_kernel(4, q_len, 8192, ragged=True, fused=fused) is None


def test_select_decode_kernel_asks_the_module_only_when_needed(monkeypatch):
    def unexpected():
        raise AssertionError("the kernel module must not be asked")

    monkeypatch.setattr(nm35, "_fused_fp4_mul", unexpected)
    assert nm35.select_decode_kernel(4, 1, 8192) is None
    assert nm35.select_decode_kernel(4, 9, 131072) is None
    assert nm35.select_decode_kernel(4, 2, 131072, ragged=True) is None

    for fused, expected in ((True, BASE), (False, None)):
        monkeypatch.setattr(nm35, "_fused_fp4_mul", lambda fused=fused: fused)
        assert nm35.select_decode_kernel(4, 1, 131072) == expected


@pytest.mark.parametrize("fused", [True, False])
def test_fused_fp4_mul_reads_the_module(monkeypatch, fused):
    class Module:
        @staticmethod
        def nm35_nvfp4_fused_fp4_mul():
            return fused

    monkeypatch.setattr(nm35, "_module", Module)
    assert nm35._fused_fp4_mul.__wrapped__() == fused
    # Without a module every call runs on trtllm-gen whatever the answer.
    monkeypatch.setattr(nm35, "_module", lambda: None)
    assert nm35._fused_fp4_mul.__wrapped__()


def test_select_prefill_kernel():
    assert nm35.select_prefill_kernel(1, 1, 8193)
    assert nm35.select_prefill_kernel(1, 32768, 1 << 20)
    assert not nm35.select_prefill_kernel(1, 32769, 1 << 20)
    assert not nm35.select_prefill_kernel(2, 1024, 1 << 20)
    assert not nm35.select_prefill_kernel(0, 1024, 1 << 20)
    # A KV of at most 8192 tokens runs on trtllm-gen.
    assert not nm35.select_prefill_kernel(1, 1024, 8192)
    assert not nm35.select_prefill_kernel(1, 1, 0)


def test_prefill_workspace_bytes():
    # 64 KiB per page, plus a 72 MiB reserve.
    assert nm35.prefill_workspace_bytes(1) == 65536 + 72 * MIB
    assert nm35.prefill_workspace_bytes(128) == 65536 + 72 * MIB
    assert nm35.prefill_workspace_bytes(129) == 2 * 65536 + 72 * MIB
    assert nm35.prefill_workspace_bytes(1 << 20) == 584 * MIB
    assert nm35.prefill_workspace_bytes((1 << 20) + 32768) == 600 * MIB
    with pytest.raises(ValueError):
        nm35.prefill_workspace_bytes(0)


def test_route_counts_are_thread_safe():
    nm35.reset_route_counts()
    calls, threads = 20000, 8

    def count():
        for _ in range(calls):
            nm35._count_route("decode", "trtllm-gen")

    workers = [threading.Thread(target=count) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert nm35.route_counts() == {("decode", "trtllm-gen"): calls * threads}
    nm35.reset_route_counts()
    assert nm35.route_counts() == {}


def _cpu_call(query_dtype=torch.float8_e4m3fn):
    query = torch.zeros((2, 32, 128), dtype=torch.uint8).view(query_dtype)
    pages = torch.zeros((4, 2, 128, 64), dtype=torch.uint8)
    scales = torch.zeros((4, 2, 128, 8), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return dict(
        out=torch.zeros((2, 32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn),
        query=query,
        k_cache=pages,
        v_cache=pages,
        k_block_scales=scales,
        v_block_scales=scales,
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
        block_tables=torch.zeros((2, 80), dtype=torch.int32),
        seq_lens=torch.full((2,), 10000, dtype=torch.int32),
        batch_size=2,
        max_q_len=1,
        max_kv_len=10000,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


@pytest.mark.parametrize("reason", ["feature", "dtype", "no block scales"])
def test_calls_outside_the_geometry_are_not_served(monkeypatch, reason):
    # Such a call is turned away before the kernel module is looked at.
    def unexpected():
        raise AssertionError("the kernel module must not be loaded")

    monkeypatch.setattr(nm35, "_module", unexpected)
    call = _cpu_call(torch.int8 if reason == "dtype" else torch.float8_e4m3fn)
    if reason == "no block scales":
        call["k_block_scales"] = None
    plain = reason != "feature"

    nm35.reset_route_counts()
    assert not nm35.try_decode(
        **call, enable_pdl=False, cum_seq_lens_q=None, plain=plain
    )
    cum_seq_lens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert not nm35.try_prefill(**call, cum_seq_lens_q=cum_seq_lens_q, plain=plain)
    assert nm35.route_counts() == {
        ("decode", "trtllm-gen"): 1,
        ("prefill", "trtllm-gen"): 1,
    }


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_prefill_with_several_requests_runs_trtllm_gen(monkeypatch, batch_size):
    # A call that the kernel serves with one request runs on trtllm-gen with
    # more, and the kernel module is not looked at.
    prefills = []

    class Module:
        def nm35_nvfp4_prefill(self, *args):
            prefills.append(args)
            return ""

    def module():
        if batch_size > 1:
            raise AssertionError("the kernel module must not be loaded")
        return Module()

    monkeypatch.setattr(nm35, "is_supported_device", lambda device: True)
    monkeypatch.setattr(nm35, "_module", module)
    call = _cpu_call()
    rows = torch.zeros((batch_size, 32, 128), dtype=torch.uint8)
    call["query"] = rows.view(torch.float8_e4m3fn)
    call["out"] = rows.clone().view(torch.float8_e4m3fn)
    call["block_tables"] = torch.zeros((batch_size, 80), dtype=torch.int32)
    call["seq_lens"] = torch.full((batch_size,), 10000, dtype=torch.int32)
    call["batch_size"] = batch_size
    cum_seq_lens_q = torch.arange(batch_size + 1, dtype=torch.int32)

    nm35.reset_route_counts()
    served = nm35.try_prefill(**call, cum_seq_lens_q=cum_seq_lens_q, plain=True)
    assert served == (batch_size == 1)
    assert len(prefills) == int(batch_size == 1)
    kernel = "prefill" if batch_size == 1 else "trtllm-gen"
    assert nm35.route_counts() == {("prefill", kernel): 1}


@pytest.mark.parametrize("offsets", [False, True])
def test_decode_with_query_offsets_is_routed_as_ragged(monkeypatch, offsets):
    # Uniform query rows prove nothing when the call passes cum_seq_lens_q: a
    # CUDA graph replays whatever offsets it holds then.
    selected = []
    monkeypatch.setattr(nm35, "is_supported_device", lambda device: True)
    monkeypatch.setattr(
        nm35, "select_decode_kernel", lambda *args: selected.append(args)
    )
    cum_seq_lens_q = torch.tensor([0, 1, 2], dtype=torch.int32) if offsets else None
    assert not nm35.try_decode(
        **_cpu_call(), enable_pdl=False, cum_seq_lens_q=cum_seq_lens_q, plain=True
    )
    assert selected == [(2, 1, 10000, offsets)]


class _Spec:
    def __init__(self, is_compiled):
        self.is_compiled = is_compiled


def _patch_toolkit(monkeypatch, is_compiled, new_enough):
    import flashinfer.jit.cpp_ext as cpp_ext
    import flashinfer.jit.nm35_nvfp4 as jit_nm35

    monkeypatch.setattr(
        jit_nm35, "gen_nm35_nvfp4_fmha_module", lambda: _Spec(is_compiled)
    )
    monkeypatch.setattr(cpp_ext, "is_cuda_version_at_least", lambda _: new_enough)
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: "12.9")


def test_build_is_blocked_only_by_an_old_toolkit(monkeypatch):
    blocker = nm35._build_blocker.__wrapped__
    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)

    _patch_toolkit(monkeypatch, is_compiled=False, new_enough=False)
    assert nm35.MIN_CUDA_TOOLKIT == "13.0"
    assert "13.0 toolkit or newer (found 12.9)" in blocker()
    # Another compiler is named explicitly: its build is attempted.
    monkeypatch.setenv("FLASHINFER_NVCC", "/opt/cuda/bin/nvcc")
    assert blocker() is None
    monkeypatch.delenv("FLASHINFER_NVCC")

    _patch_toolkit(monkeypatch, is_compiled=True, new_enough=False)
    assert blocker() is None
    _patch_toolkit(monkeypatch, is_compiled=False, new_enough=True)
    assert blocker() is None


def test_blocked_build_is_not_started(monkeypatch):
    import flashinfer.jit.nm35_nvfp4 as jit_nm35

    def unexpected():
        raise AssertionError("a build that cannot succeed must not start")

    monkeypatch.setattr(jit_nm35, "load_nm35_nvfp4_fmha_module", unexpected)
    monkeypatch.setattr(nm35, "_build_blocker", lambda: "old toolkit")
    assert nm35._module.__wrapped__() is None


def test_failed_build_falls_back(monkeypatch):
    import flashinfer.jit.nm35_nvfp4 as jit_nm35

    def failing():
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr(jit_nm35, "load_nm35_nvfp4_fmha_module", failing)
    monkeypatch.setattr(nm35, "_build_blocker", lambda: None)
    assert nm35._module.__wrapped__() is None


@pytest.mark.parametrize("fused", [True, False])
def test_slower_dequantization_path_is_logged(monkeypatch, caplog, fused):
    import flashinfer.jit.nm35_nvfp4 as jit_nm35

    class Module:
        @staticmethod
        def nm35_nvfp4_fused_fp4_mul():
            return fused

    monkeypatch.setattr(jit_nm35, "load_nm35_nvfp4_fmha_module", Module)
    monkeypatch.setattr(nm35, "_build_blocker", lambda: None)
    with caplog.at_level(logging.INFO, logger="flashinfer.nm35_nvfp4"):
        assert nm35._module.__wrapped__() is not None
    logged = any("slower NVFP4 dequantization" in m for m in caplog.messages)
    assert logged == (not fused)
