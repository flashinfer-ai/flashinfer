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

"""Routing of the Cake FMHA super_nvfp4 routes without a GPU: the gating rules,
the route selection around the kernels' own checks, and the hand-off to the
kernels and to trtllm-gen."""

import pytest
import torch

import flashinfer.cake_fmha_super_nvfp4 as super_nvfp4

MTP, BASE = super_nvfp4.DECODE_MTP, super_nvfp4.DECODE_BASE
PREFILL, TRTLLM_GEN = super_nvfp4.PREFILL, super_nvfp4.TRTLLM_GEN
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
    kernel = super_nvfp4.select_decode_kernel(batch_size, q_len, max_kv_len, fused=True)
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
    kernel = super_nvfp4.select_decode_kernel(
        batch_size, q_len, max_kv_len, fused=False
    )
    assert kernel == expected


def test_non_fused_routes_a_subset_of_the_fused_ones():
    for batch_size in range(1, 258):
        for q_len in range(10):
            for max_kv_len in (8192, 8193, 16384, 32769, 65537, 1 << 20):
                args = (batch_size, q_len, max_kv_len)
                kernel = super_nvfp4.select_decode_kernel(*args, fused=False)
                if kernel is not None:
                    assert kernel == super_nvfp4.select_decode_kernel(*args, fused=True)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "q_len,expected",
    [(1, BASE), (2, None), (3, BASE), (4, None), (5, BASE), (6, None), (8, None)],
)
def test_select_decode_kernel_ragged(q_len, expected, fused):
    # Requests of different query lengths never reach "mtp", which declines them.
    batch_sizes = (1, 2, 32) if fused else (1,)
    for batch_size in batch_sizes:
        kernel = super_nvfp4.select_decode_kernel(
            batch_size, q_len, 131072, ragged=True, fused=fused
        )
        assert kernel == expected
    assert (
        super_nvfp4.select_decode_kernel(4, q_len, 8192, ragged=True, fused=fused)
        is None
    )


def test_select_prefill_kernel():
    assert super_nvfp4.select_prefill_kernel(1, 1, 8193) == PREFILL
    assert super_nvfp4.select_prefill_kernel(1, 32768, 1 << 20) == PREFILL
    assert super_nvfp4.select_prefill_kernel(1, 32769, 1 << 20) is None
    assert super_nvfp4.select_prefill_kernel(2, 1024, 1 << 20) is None
    assert super_nvfp4.select_prefill_kernel(0, 1024, 1 << 20) is None
    # A KV of at most 8192 tokens runs on trtllm-gen.
    assert super_nvfp4.select_prefill_kernel(1, 1024, 8192) is None
    assert super_nvfp4.select_prefill_kernel(1, 1, 0) is None


def test_prefill_workspace_bytes():
    # 64 KiB per page, plus a 72 MiB reserve.
    assert super_nvfp4.prefill_workspace_bytes(1) == 65536 + 72 * MIB
    assert super_nvfp4.prefill_workspace_bytes(128) == 65536 + 72 * MIB
    assert super_nvfp4.prefill_workspace_bytes(129) == 2 * 65536 + 72 * MIB
    assert super_nvfp4.prefill_workspace_bytes(1 << 20) == 584 * MIB
    assert super_nvfp4.prefill_workspace_bytes((1 << 20) + 32768) == 600 * MIB
    with pytest.raises(ValueError):
        super_nvfp4.prefill_workspace_bytes(0)


def _call(batch_size=2, max_q_len=1, cum_seq_lens_q=None):
    """A call with the geometry, on CPU tensors (``_supported`` makes the CPU
    count as an sm_100 device)."""
    rows = torch.zeros((batch_size * max_q_len, 32, 128), dtype=torch.uint8)
    pages = torch.zeros((4, 2, 128, 64), dtype=torch.uint8)
    scales = torch.zeros((4, 2, 128, 8), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return dict(
        out=rows.clone().view(torch.float8_e4m3fn),
        query=rows.view(torch.float8_e4m3fn),
        k_cache=pages,
        v_cache=pages.clone(),
        k_block_scales=scales,
        v_block_scales=scales.clone(),
        workspace_buffer=torch.empty(16, dtype=torch.uint8),
        block_tables=torch.zeros((batch_size, 80), dtype=torch.int32),
        seq_lens=torch.full((batch_size,), 10000, dtype=torch.int32),
        cum_seq_lens_q=cum_seq_lens_q,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=10000,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def _prefill_call(max_q_len=1):
    return _call(1, max_q_len, torch.tensor([0, max_q_len], dtype=torch.int32))


class _Module:
    """Stands in for the kernel module: records its calls and declines with
    ``reason`` (empty: serves every call)."""

    def __init__(self):
        self.calls = []
        self.reason = ""

    def super_nvfp4_fused_fp4_mul(self):
        return True

    def super_nvfp4_decode_reason(self, kernel_id, *args):
        self.calls.append(("decode_reason", kernel_id, args))
        return self.reason

    def super_nvfp4_prefill_reason(self, *args):
        self.calls.append(("prefill_reason", None, args))
        return self.reason

    def super_nvfp4_decode(self, kernel_id, *args):
        self.calls.append(("decode", kernel_id, args))
        return self.reason

    def super_nvfp4_prefill(self, *args):
        self.calls.append(("prefill", None, args))
        return self.reason


@pytest.fixture
def module(monkeypatch):
    module = _Module()
    monkeypatch.setattr(super_nvfp4, "_module", lambda: module)
    return module


@pytest.fixture
def _supported(monkeypatch, module):
    monkeypatch.setattr(super_nvfp4, "_is_supported_device", lambda device: True)
    monkeypatch.setattr(super_nvfp4, "_uses_fused_fp4_mul", lambda: True)


def _select_decode(dense_causal=True, **call):
    return super_nvfp4.select_decode_route(
        **call, enable_pdl=False, dense_causal=dense_causal
    )


def _select_prefill(dense_causal=True, **call):
    return super_nvfp4.select_prefill_route(**call, dense_causal=dense_causal)


@pytest.mark.parametrize("reason", ["dtype", "no block scales", "device", "heads"])
def test_calls_outside_the_geometry_are_not_routed(monkeypatch, reason):
    # Such a call keeps Cake's other routes, and the module is not looked at.
    def unexpected():
        raise AssertionError("the kernel module must not be loaded")

    monkeypatch.setattr(super_nvfp4, "_module", unexpected)
    if reason != "device":  # a CPU tensor is on no sm_100 device
        monkeypatch.setattr(super_nvfp4, "_is_supported_device", lambda device: True)
    for call, select in ((_call(), _select_decode), (_prefill_call(), _select_prefill)):
        if reason == "dtype":
            call["query"] = call["query"].view(torch.int8)
        if reason == "no block scales":
            call["k_block_scales"] = None
        if reason == "heads":
            call["query"] = call["query"][:, :16]
        assert select(**call) is None


def test_calls_with_other_features_run_trtllm_gen(monkeypatch):
    # The geometry with a feature the kernels lack (a window, sinks, LSE): the
    # route runs trtllm-gen, and the module is not looked at.
    def unexpected():
        raise AssertionError("the kernel module must not be loaded")

    monkeypatch.setattr(super_nvfp4, "_module", unexpected)
    monkeypatch.setattr(super_nvfp4, "_is_supported_device", lambda device: True)
    assert _select_decode(dense_causal=False, **_call()) == TRTLLM_GEN
    assert _select_prefill(dense_causal=False, **_prefill_call()) == TRTLLM_GEN


@pytest.mark.parametrize(
    "batch_size,max_q_len,expected",
    [
        (2, 1, BASE),
        (2, 3, BASE),
        (4, 2, MTP),
        (2, 6, MTP),
        (4, 6, TRTLLM_GEN),
        (3, 8, MTP),
        (2, 7, TRTLLM_GEN),
    ],
)
def test_select_decode_route(_supported, module, batch_size, max_q_len, expected):
    call = _call(batch_size, max_q_len)
    assert _select_decode(**call) == expected
    # A kernel is chosen only after it accepted the call.
    asked = [kernel_id for stage, kernel_id, _ in module.calls]
    assert asked == ([] if expected == TRTLLM_GEN else [{MTP: 0, BASE: 1}[expected]])


@pytest.mark.parametrize("max_kv_len", [1, 8192, 8193])
def test_short_kv_runs_trtllm_gen(_supported, max_kv_len):
    short = max_kv_len <= super_nvfp4.TRTLLM_GEN_MAX_KV_LEN
    call = _call()
    call["max_kv_len"] = max_kv_len
    assert _select_decode(**call) == (TRTLLM_GEN if short else BASE)
    prefill = _prefill_call()
    prefill["max_kv_len"] = max_kv_len
    assert _select_prefill(**prefill) == (TRTLLM_GEN if short else PREFILL)


def test_prefill_with_several_requests_runs_trtllm_gen(_supported, module):
    call = _call(2, 1, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert _select_prefill(**call) == TRTLLM_GEN
    assert module.calls == []


def test_prefill_without_offsets_or_workspace_runs_trtllm_gen(_supported):
    call = _prefill_call()
    call["cum_seq_lens_q"] = None
    assert _select_prefill(**call) == TRTLLM_GEN
    call = _prefill_call()
    call["workspace_buffer"] = None
    assert _select_prefill(**call) == TRTLLM_GEN


def test_a_call_the_kernel_would_decline_runs_trtllm_gen(_supported, module):
    module.reason = "workspace too small"
    assert _select_decode(**_call()) == TRTLLM_GEN
    assert _select_prefill(**_prefill_call()) == TRTLLM_GEN
    assert [stage for stage, _, _ in module.calls] == [
        "decode_reason",
        "prefill_reason",
    ]


def test_unbuildable_module_runs_trtllm_gen(monkeypatch, _supported):
    monkeypatch.setattr(super_nvfp4, "_module", lambda: None)
    assert _select_decode(**_call()) == TRTLLM_GEN
    assert _select_prefill(**_prefill_call()) == TRTLLM_GEN


@pytest.mark.parametrize("offsets", [False, True])
def test_decode_with_query_offsets_is_routed_as_ragged(
    monkeypatch, _supported, offsets
):
    # Uniform query rows prove nothing when the call passes cum_seq_lens_q: a
    # CUDA graph replays whatever offsets it holds then.
    selected = []
    monkeypatch.setattr(
        super_nvfp4, "select_decode_kernel", lambda *args: selected.append(args)
    )
    cum_seq_lens_q = torch.tensor([0, 1, 2], dtype=torch.int32) if offsets else None
    assert _select_decode(**_call(2, 1, cum_seq_lens_q)) == TRTLLM_GEN
    assert selected == [(2, 1, 10000, offsets)]


def test_run_prefill_converts_a_device_bmm1_scale_to_log2(module):
    # Cake's context API passes a device bmm1_scale raw; the kernel reads log2.
    call = _prefill_call()
    call["bmm1_scale"] = torch.tensor([0.5])
    super_nvfp4.run_prefill(**call)
    ((_, _, args),) = module.calls
    assert torch.allclose(args[-2], torch.tensor([0.5 * super_nvfp4.log2e]))


def test_run_raises_when_the_kernel_declines(module):
    module.reason = "workspace too small"
    with pytest.raises(RuntimeError, match="workspace too small"):
        super_nvfp4.run_decode(BASE, **_call(), enable_pdl=False)
    with pytest.raises(RuntimeError, match="workspace too small"):
        super_nvfp4.run_prefill(**_prefill_call())


def _context_abi_args(call, **changes):
    """``run_args`` as trtllm_batch_context_with_kv_cache passes them to Cake."""
    args = dict(
        out=call["out"],
        out_scale_factor=None,
        query=call["query"],
        key_cache=call["k_cache"],
        value_cache=call["v_cache"],
        workspace_buffer=call["workspace_buffer"],
        multi_ctas_kv_counter_buffer=torch.zeros(16, dtype=torch.uint8),
        block_tables=call["block_tables"],
        seq_lens=call["seq_lens"],
        max_q_len=call["max_q_len"],
        max_kv_len=call["max_kv_len"],
        bmm1_scale=call["bmm1_scale"],
        bmm2_scale=call["bmm2_scale"],
        o_sf_scale=-1.0,
        o_sf_vec_size=-1,
        o_sf_start_index=0,
        batch_size=call["batch_size"],
        window_left=-1,
        cum_seq_lens_q=call["cum_seq_lens_q"],
        cum_seq_lens_kv=torch.tensor([0, 80], dtype=torch.int32),
        sm_count=148,
        enable_pdl=False,
        workspace_size=call["workspace_buffer"].numel(),
        attention_sinks=None,
        key_block_scales=call["k_block_scales"],
        value_block_scales=call["v_block_scales"],
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        use_fp16_softmax=False,
        uses_spcompress=False,
        is_causal=True,
        lse=None,
        lse_stride_tokens=0,
        lse_stride_heads=0,
    )
    args.update(changes)
    return list(args.values())


def _context_module(module, kernel="prefill"):
    from flashinfer.cake_fmha import (
        CakeFmhaContextRoute,
        CakeFmhaSuperNvfp4Module,
        CakeFmhaSuperNvfp4TrtllmGenModule,
    )

    route = CakeFmhaContextRoute(
        target="sm100a",
        component="context_super_nvfp4",
        num_m_blocks=0,
        num_q_heads=32,
        num_kv_heads=2,
        pack_g=0,
        page_size=128,
        l2_swizzle=0,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        reads_device_scales=True,
        super_nvfp4_kernel=kernel,
    )
    if kernel == TRTLLM_GEN:
        return CakeFmhaSuperNvfp4TrtllmGenModule(route=route)
    return CakeFmhaSuperNvfp4Module(route=route, module=module)


def test_context_abi_launches_the_prefill_kernel(module):
    # Cake's context dispatch calls every module with the same 34 arguments.
    call = _prefill_call()
    _context_module(module).cake_paged_attention_context(*_context_abi_args(call))
    ((stage, _, args),) = module.calls
    assert stage == "prefill"
    expected = [call[name] for name in ("out", "query", "k_cache", "v_cache")]
    expected += [call[name] for name in ("k_block_scales", "v_block_scales")]
    expected += [call["workspace_buffer"], call["block_tables"], call["seq_lens"]]
    expected.append(call["cum_seq_lens_q"])
    assert all(a is b for a, b in zip(args[:10], expected, strict=True))
    assert args[10:] == (1, 1, 10000, 1.0, 1.0)


def _decode_route(kernel):
    from flashinfer.cake_fmha import CakeFmhaDecodeRoute

    return CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=2,
        q_len=1,
        num_q_heads=32,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component="decode_super_nvfp4",
        page_size=128,
        super_nvfp4_kernel=kernel,
    )


def test_decode_abi_launches_the_route_kernel(module):
    # Cake's decode dispatch calls every module with the same 33 arguments.
    from flashinfer.cake_fmha import CakeFmhaSuperNvfp4Module

    call = _call()
    args = [object() for _ in range(33)]
    for index, name in (
        (0, "out"),
        (2, "query"),
        (3, "k_cache"),
        (4, "v_cache"),
        (5, "workspace_buffer"),
        (7, "block_tables"),
        (8, "seq_lens"),
        (9, "max_q_len"),
        (10, "max_kv_len"),
        (11, "bmm1_scale"),
        (12, "bmm2_scale"),
        (16, "batch_size"),
        (23, "cum_seq_lens_q"),
        (24, "k_block_scales"),
        (25, "v_block_scales"),
    ):
        args[index] = call[name]
    args[20] = True  # enable_pdl
    CakeFmhaSuperNvfp4Module(_decode_route(BASE), module).cake_paged_attention_decode(
        *args
    )
    ((stage, kernel_id, launched),) = module.calls
    assert (stage, kernel_id) == ("decode", 1)
    expected = [call[name] for name in ("out", "query", "k_cache", "v_cache")]
    expected += [call[name] for name in ("k_block_scales", "v_block_scales")]
    expected += [call[n] for n in ("workspace_buffer", "block_tables", "seq_lens")]
    expected += [2, 1, 10000, 1.0, 1.0, True, None]
    assert all(a is b or a == b for a, b in zip(launched, expected, strict=True))


@pytest.mark.parametrize("kernel", [PREFILL, TRTLLM_GEN, None])
def test_decode_module_refuses_other_kernels(module, kernel):
    from flashinfer.cake_fmha import CakeFmhaSuperNvfp4Module

    with pytest.raises(RuntimeError, match="not a decode kernel"):
        CakeFmhaSuperNvfp4Module(
            _decode_route(kernel), module
        ).cake_paged_attention_decode(*[None] * 33)
    assert module.calls == []


@pytest.mark.parametrize("kernel", [BASE, None])
def test_context_module_refuses_other_kernels(module, kernel):
    with pytest.raises(RuntimeError, match="not the prefill kernel"):
        _context_module(module, kernel).cake_paged_attention_context(
            *_context_abi_args(_prefill_call())
        )
    assert module.calls == []


class _TrtllmGen:
    """Stands in for the trtllm-gen FMHA module: records its calls."""

    def __init__(self):
        self.calls = []

    def trtllm_paged_attention_decode(self, *args):
        self.calls.append(("decode", args))

    def trtllm_paged_attention_context(self, *args):
        self.calls.append(("context", args))


@pytest.fixture
def trtllm_gen(monkeypatch):
    import flashinfer.decode
    import flashinfer.prefill

    fake = _TrtllmGen()
    for owner in (flashinfer.decode, flashinfer.prefill):
        monkeypatch.setattr(owner, "get_trtllm_gen_fmha_module", lambda: fake)
    return fake


def test_trtllm_gen_context_gets_trtllm_gen_arguments(module, trtllm_gen):
    # Cake's 34 context arguments, plus what trtllm-gen also takes: the right
    # window after window_left, the two per-token window tensors after
    # cum_seq_lens_kv, lse_scale after lse, and a log2 device bmm1 scale.
    call = _prefill_call()
    call["bmm1_scale"] = torch.tensor([0.5])
    cake_args = _context_abi_args(call, window_left=128)
    _context_module(module, TRTLLM_GEN).cake_paged_attention_context(*cake_args)
    ((stage, args),) = trtllm_gen.calls
    assert stage == "context" and len(args) == 38
    assert torch.allclose(args[11], torch.tensor([0.5 * super_nvfp4.log2e]))
    assert args[17:19] == (128, 0)  # a causal left window: right window 0
    assert args[21:23] == (None, None)
    assert args[35] == 1.0
    rest = [a for i, a in enumerate(args) if i not in (11, 18, 21, 22, 35)]
    expected = [a for i, a in enumerate(cake_args) if i != 11]
    assert all(a is b or a == b for a, b in zip(rest, expected, strict=True))


def test_trtllm_gen_decode_gets_trtllm_gen_arguments(trtllm_gen):
    # Cake's 33 decode arguments, plus lse_scale after lse and, at the end, the
    # BF16-query transform mode (0: none) and use_fp16_softmax.
    from flashinfer.cake_fmha import (
        CakeFmhaDecodeRoute,
        CakeFmhaSuperNvfp4TrtllmGenModule,
    )

    route = CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=2,
        q_len=1,
        num_q_heads=32,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component="decode_super_nvfp4",
        page_size=128,
        super_nvfp4_kernel=TRTLLM_GEN,
    )
    cake_args = [object() for _ in range(33)]
    CakeFmhaSuperNvfp4TrtllmGenModule(route=route).cake_paged_attention_decode(
        *cake_args
    )
    ((stage, args),) = trtllm_gen.calls
    assert stage == "decode"
    assert list(args) == [*cake_args[:29], 1.0, *cake_args[29:], 0, None]
