"""
Copyright (c) 2026 by FlashInfer team.

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

"""Backend selection under ``backend="auto"`` for ragged prefill.

``determine_attention_backend`` only ever answers ``"fa3"`` or ``"fa2"``, and
FA3 is gated on ``is_sm90a_supported()``, so on Blackwell ``auto`` lands on FA2.
``plan()`` then upgrades that answer where a faster kernel is known to be
equivalent -- ``fmha_v2`` on SM120, ``cutlass`` on SM100a/SM110a.

Those upgrades are the only place a caller's backend is chosen for them, and
the CUTLASS run path (``fmha_varlen``) silently ignores ``window_left``,
``logits_soft_cap``, custom masks and the multi-item-scoring pointers -- it
receives only ``causal`` and the scales. An upgrade that fired on any of those
would return quietly wrong numbers, so the negative cases below matter more
than the positive one and are tested first-class rather than as an afterthought.
"""

import pytest
import torch

import flashinfer
from flashinfer.cudnn.prefill import _cudnn_supports_direct_seqlens
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported

DTYPE = torch.bfloat16


def _cutlass_upgrade_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda")
    return is_sm100a_supported(dev) or is_sm110a_supported(dev)


def _cudnn_upgrade_available() -> bool:
    """cuDNN is preferred over CUTLASS on SM100a when it can take token indptrs directly."""
    if not torch.cuda.is_available():
        return False
    return is_sm100a_supported(torch.device("cuda")) and _cudnn_supports_direct_seqlens(
        DTYPE
    )


requires_cutlass_arch = pytest.mark.skipif(
    not _cutlass_upgrade_arch(),
    reason="the auto->cutlass upgrade targets SM100a/SM110a",
)

requires_cudnn_upgrade = pytest.mark.skipif(
    not _cudnn_upgrade_available(),
    reason="the auto->cudnn upgrade needs SM100a and cuDNN 9.24+/frontend 1.25+",
)

# What `auto` should land on for a shape both kernels serve.
BLACKWELL_DEFAULT = "cudnn" if _cudnn_upgrade_available() else "cutlass"


def _expected_auto_backend(d_qk, d_vo):
    """What `auto` should resolve to for this shape on this machine.

    Mirrors ``flashinfer.prefill``'s per-shape preference so these tests assert
    the *selection policy* rather than a frozen backend name: the order is a
    measured property (CUTLASS wins 128-head d192/128 on B200, cuDNN wins the
    d128/d256 cells), and it is expected to change as hardware is measured.
    """
    from flashinfer.prefill import _blackwell_ragged_auto_order

    cudnn_ok = _cudnn_upgrade_available()
    for backend in _blackwell_ragged_auto_order(d_qk, d_vo):
        if (
            backend == "cudnn"
            and cudnn_ok
            and (d_qk, d_vo) in {(128, 128), (192, 128), (256, 256)}
        ):
            return "cudnn"
        if (
            backend == "cutlass"
            and _cutlass_upgrade_arch()
            and ((d_qk == 128 and d_vo == 128) or d_qk == 192)
        ):
            return "cutlass"
    return "fa2"


def _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, seed=1234):
    """Seeded so two backends can be compared over byte-identical inputs."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q = torch.randn(batch * s_q, h_qo, d_qk, dtype=DTYPE, device=dev)
    k = torch.randn(batch * s_kv, h_kv, d_qk, dtype=DTYPE, device=dev)
    v = torch.randn(batch * s_kv, h_kv, d_vo, dtype=DTYPE, device=dev)
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_q
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_kv
    return q, k, v, qo_indptr, kv_indptr


def _plan_only(
    backend, batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, kv_layout="NHD", **plan_kwargs
):
    """Resolve the backend without launching a kernel.

    The choice is made in ``plan()``, so the selection tests stop there. That
    keeps a negative case from passing for the wrong reason -- e.g. an HND
    layout would make ``run()`` fail on tensor shapes, which would look like a
    pass while telling us nothing about which backend was picked.
    """
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout, backend=backend
    )
    _, _, _, qo_indptr, kv_indptr = _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    return wrapper._backend


def _plan_and_run(
    backend, batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, kv_layout="NHD", **plan_kwargs
):
    """Returns ``(resolved_backend, output)``."""
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout, backend=backend
    )
    q, k, v, qo_indptr, kv_indptr = _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    out = wrapper.run(q, k, v)
    torch.cuda.synchronize()
    return wrapper._backend, out


# ---------------------------------------------------------------------------
# The upgrade fires where it should
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo",
    [
        (64, 8, 128, 128),  # GQA, square d128
        (32, 32, 128, 128),  # MHA, square d128
        (128, 128, 192, 128),  # MLA-style rectangular (DeepSeek-V3 prefill dims)
    ],
)
def test_auto_upgrades_on_blackwell(h_qo, h_kv, d_qk, d_vo):
    """On SM100a/SM110a, ``auto`` must leave FA2: cuDNN where it can take token
    indptrs directly, CUTLASS otherwise."""
    resolved = _plan_only("auto", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    expected = _expected_auto_backend(d_qk, d_vo)
    assert resolved == expected, (
        f"auto resolved to {resolved!r}; expected {expected!r} for "
        f"h_qo={h_qo} h_kv={h_kv} d_qk={d_qk} d_vo={d_vo}"
    )
    assert resolved != "fa2", "auto must leave FA2 on Blackwell for this shape"


# ---------------------------------------------------------------------------
# The upgrade must NOT fire where CUTLASS would drop semantics
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "tag,plan_kwargs",
    [
        # fmha_varlen takes no window_left: upgrading would compute full
        # attention and silently ignore the window.
        ("sliding_window", {"window_left": 256}),
        # ... and no logits_soft_cap.
        ("logits_soft_cap", {"logits_soft_cap": 30.0}),
    ],
)
def test_auto_declines_when_semantics_would_be_dropped(tag, plan_kwargs):
    resolved = _plan_only("auto", 4, 1024, 1024, 64, 8, 128, 128, **plan_kwargs)
    assert resolved == "fa2", (
        f"auto upgraded to {resolved!r} with {tag} requested; neither the CUTLASS "
        f"nor the cuDNN run path receives it, so the result would be silently wrong"
    )


@requires_cutlass_arch
@pytest.mark.parametrize("d_qk,d_vo", [(256, 256), (64, 64)])
def test_auto_declines_outside_cutlass_head_dims(d_qk, d_vo):
    """``get_fmha_module`` serves square d128 and d192 only."""
    resolved = _plan_only("auto", 4, 1024, 1024, 32, 32, d_qk, d_vo)
    assert resolved != "cutlass", (
        f"auto upgraded to cutlass at d_qk={d_qk}/d_vo={d_vo}, which "
        f"get_fmha_module does not serve"
    )


@requires_cutlass_arch
def test_auto_declines_on_hnd_layout():
    """The CUTLASS path is only wired for NHD here."""
    resolved = _plan_only("auto", 4, 1024, 1024, 64, 8, 128, 128, kv_layout="HND")
    assert resolved != "cutlass"


# ---------------------------------------------------------------------------
# Numerics: the upgrade must not change the answer
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo", [(64, 8, 128, 128), (128, 128, 192, 128)]
)
def test_auto_matches_fa2_numerically(h_qo, h_kv, d_qk, d_vo):
    """What ``auto`` returns now must match what it returned before (FA2).

    Compared over byte-identical seeded inputs; the tolerance is the usual bf16
    accumulation slack between two independent kernels, not an exactness claim.
    """
    resolved, out_auto = _plan_and_run("auto", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    _, out_fa2 = _plan_and_run("fa2", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    assert resolved != "fa2", "precondition: this shape should have upgraded"

    diff = (out_auto.float() - out_fa2.float()).abs()
    denom = out_fa2.float().abs().max().clamp_min(1e-6)
    rel_max = (diff.max() / denom).item()
    assert rel_max < 2e-2, (
        f"auto({resolved}) diverges from fa2: rel_max={rel_max:.3e}, "
        f"max_abs={diff.max().item():.3e}, mean_abs={diff.mean().item():.3e}"
    )


# ---------------------------------------------------------------------------
# An explicit backend is never second-guessed
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize("backend", ["fa2", "cutlass"])
def test_explicit_backend_is_respected(backend):
    """The upgrade is scoped to ``auto``; it must not rewrite an explicit choice.

    Pinning this matters because the gate lives inside ``plan()`` next to the
    resolver call rather than inside ``determine_attention_backend``.
    """
    resolved = _plan_only(backend, 4, 1024, 1024, 64, 8, 128, 128)
    assert resolved == backend


# ---------------------------------------------------------------------------
# cuDNN: `auto` hands the caller's token indptrs straight to cuDNN
# ---------------------------------------------------------------------------


def _varlen_inputs(batch, s_max, h_qo, h_kv, d_qk, d_vo, seed=4321):
    """Random per-request lengths, so the padding mask (driven by the indptrs
    on the cuDNN direct path) is actually exercised."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(max(1, s_max // 4), s_max + 1, (batch,), generator=g)
    indptr = torch.zeros(batch + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, 0)
    total = int(indptr[-1])
    torch.manual_seed(seed)
    q = torch.randn(total, h_qo, d_qk, dtype=DTYPE, device=dev)
    k = torch.randn(total, h_kv, d_qk, dtype=DTYPE, device=dev)
    v = torch.randn(total, h_kv, d_vo, dtype=DTYPE, device=dev)
    return q, k, v, indptr.to(dev), lens


@requires_cutlass_arch
def test_auto_falls_back_to_cutlass_without_cudnn_direct_path(monkeypatch):
    """The preference list is cudnn -> cutlass: when cuDNN cannot take token
    indptrs directly (old cuDNN, or no cuDNN), `auto` must land on cutlass,
    never on the host-side element conversion."""
    import flashinfer.prefill as prefill_mod

    monkeypatch.setattr(prefill_mod, "_cudnn_supports_direct_seqlens", lambda *_: False)
    resolved = _plan_only("auto", 4, 1024, 1024, 64, 8, 128, 128)
    assert resolved == "cutlass"
    # d256: cutlass declines too -> fa2
    resolved = _plan_only("auto", 4, 1024, 1024, 32, 8, 256, 256)
    assert resolved == "fa2"


def _run_varlen(backend, q, k, v, indptr, h_qo, h_kv, d_qk, d_vo, **plan_kwargs):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    wrapper.plan(
        indptr,
        indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    out, lse = wrapper.run(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    return wrapper._backend, out, lse


def _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, tag):
    diff = (out.float() - out_fa2.float()).abs()
    rel_max = (diff.max() / out_fa2.float().abs().max().clamp_min(1e-6)).item()
    assert rel_max < 2e-2, f"{tag}: output diverges from fa2, rel_max={rel_max:.3e}"
    lse_diff = (lse - lse_fa2).abs().max().item()
    assert lse_diff < 2e-2, (
        f"{tag}: base-2 LSE diverges from fa2, max_abs={lse_diff:.3e}"
    )


@requires_cudnn_upgrade
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo",
    [
        (64, 8, 128, 128),
        (128, 128, 192, 128),
        (32, 8, 256, 256),  # CUTLASS declines d256; only cuDNN serves it
    ],
)
def test_auto_prefers_cudnn_on_sm100a(h_qo, h_kv, d_qk, d_vo):
    resolved = _plan_only("auto", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    assert resolved == "cudnn", f"auto resolved to {resolved!r}, expected cudnn"


@requires_cudnn_upgrade
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo",
    [(64, 8, 128, 128), (128, 128, 192, 128), (32, 8, 256, 256)],
)
def test_auto_cudnn_matches_fa2_varlen_with_lse(h_qo, h_kv, d_qk, d_vo):
    """Same token-unit indptrs to both backends; output AND packed base-2 LSE
    must agree. Random lengths exercise the cu_seq_len-driven padding mask."""
    q, k, v, indptr, _ = _varlen_inputs(6, 1024, h_qo, h_kv, d_qk, d_vo)
    resolved, out, lse = _run_varlen("auto", q, k, v, indptr, h_qo, h_kv, d_qk, d_vo)
    assert resolved == "cudnn", "precondition: this shape should have upgraded"
    _, out_fa2, lse_fa2 = _run_varlen("fa2", q, k, v, indptr, h_qo, h_kv, d_qk, d_vo)
    assert lse.shape == (q.shape[0], h_qo)
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "auto(cudnn)")


@requires_cudnn_upgrade
def test_auto_cudnn_keeps_int32_token_indptr_untouched():
    """The indptr the caller passed is what reaches cuDNN: no element rescale."""
    q, k, v, indptr, _ = _varlen_inputs(4, 512, 64, 8, 128, 128)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper.plan(indptr, indptr, 64, 8, 128, causal=True, q_data_type=DTYPE)
    assert wrapper._backend == "cudnn"
    assert torch.equal(wrapper._qo_indptr_buf, indptr)
    assert wrapper._qo_indptr_buf.dtype == torch.int32
    wrapper.run(q, k, v)


@requires_cudnn_upgrade
def test_explicit_cudnn_token_indptr_return_lse():
    """Explicit backend="cudnn" takes the same token-unit indptrs as `auto`
    (no seq_lens / max_* needed: plan() derives them), and return_lse=True
    works (the wrapper's packed [tokens, heads] LSE is addressed through a
    stats ragged offset)."""
    h_qo, h_kv, d = 64, 8, 128
    q, k, v, indptr, _ = _varlen_inputs(6, 1024, h_qo, h_kv, d, d)
    _, out_fa2, lse_fa2 = _run_varlen("fa2", q, k, v, indptr, h_qo, h_kv, d, d)
    resolved, out, lse = _run_varlen("cudnn", q, k, v, indptr, h_qo, h_kv, d, d)
    assert resolved == "cudnn"
    assert lse.shape == (q.shape[0], h_qo)
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "explicit cudnn")


@requires_cutlass_arch
def test_auto_is_reresolved_on_replan():
    """`auto` is decided per plan(), not once per wrapper: a re-plan that adds
    a sliding window must fall back to fa2 (the upgraded kernels would drop
    it), and a later re-plan without it must upgrade again."""
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    _, _, _, qo_indptr, kv_indptr = _inputs(4, 1024, 1024, 64, 8, 128, 128)
    plan_args = (qo_indptr, kv_indptr, 64, 8, 128)
    plan_kwargs = dict(causal=True, q_data_type=DTYPE, kv_data_type=DTYPE)
    wrapper.plan(*plan_args, **plan_kwargs)
    assert wrapper._backend == BLACKWELL_DEFAULT
    wrapper.plan(*plan_args, window_left=16, **plan_kwargs)
    assert wrapper._backend == "fa2"
    wrapper.plan(*plan_args, **plan_kwargs)
    assert wrapper._backend == BLACKWELL_DEFAULT


# ---------------------------------------------------------------------------
# Eligibility conditions that exist so `auto` never picks a backend which then
# fails. Each mirrors a constraint the backend enforces later, at a point where
# raising would strand a caller who only ever asked for "auto".
# ---------------------------------------------------------------------------


def _plan_only_indptr(backend, batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, **plan_kwargs):
    """Resolve the backend from indptrs alone.

    ``plan()`` never touches q/k/v, so skipping them lets the capacity case use
    a token count whose real tensors would not fit in memory.
    """
    dev = torch.device("cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_q
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_kv
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    return wrapper._backend


@requires_cutlass_arch
def test_auto_declines_cutlass_on_output_dtype_mismatch(monkeypatch):
    """bf16 in / fp16 out is outside the CUTLASS dispatch, not a slow path.

    ``DISPATCH_DTYPE_IN_OUT`` only handles ``out == in`` for fp16/bf16; anything
    else lands in its FP8 branch, which a bf16 input does not match, and the
    call dies reporting unsupported *head dimensions*. Pinned to cutlass so the
    assertion is about cutlass rather than about cuDNN taking the shape first.
    """
    monkeypatch.setenv("FLASHINFER_RAGGED_AUTO_BACKEND_ORDER", "cutlass")

    assert (
        _plan_only_indptr("auto", 2, 512, 512, 32, 32, 128, 128, o_data_type=DTYPE)
        == "cutlass"
    )
    assert (
        _plan_only_indptr(
            "auto", 2, 512, 512, 32, 32, 128, 128, o_data_type=torch.float16
        )
        == "fa2"
    )


@requires_cutlass_arch
def test_auto_declines_cutlass_beyond_plan_work_capacity(monkeypatch):
    """`fmha_varlen_plan`'s work-index buffers are a fixed 131072 entries.

    ``plan_kernel`` writes one entry per (qo_tile, head, batch) with no bounds
    check, so `auto` must not route an oversized problem there. The pair below
    straddles the limit on head count alone, which keeps the two cases identical
    in every other respect.
    """
    monkeypatch.setenv("FLASHINFER_RAGGED_AUTO_BACKEND_ORDER", "cutlass")
    batch, s_q = 64, 8192  # ceil(8192/256) = 32 tiles per request

    # 32 * 64 * 16 = 32768 work items -- inside capacity.
    assert (
        _plan_only_indptr("auto", batch, s_q, s_q, 16, 16, 128, 128, o_data_type=DTYPE)
        == "cutlass"
    )
    # 32 * 64 * 128 = 262144 work items -- would overrun the buffers.
    assert (
        _plan_only_indptr(
            "auto", batch, s_q, s_q, 128, 128, 128, 128, o_data_type=DTYPE
        )
        == "fa2"
    )


@requires_cudnn_upgrade
def test_auto_declines_cudnn_on_non_int32_indptr_under_cuda_graph():
    """Under capture, cuDNN's int32 indptr requirement is unsatisfiable.

    Outside graph mode ``plan()`` re-dtypes the buffers in place, so cuDNN stays
    eligible; with buffers registered to a graph it raises instead. `auto` has
    to route around that rather than resolve to a backend that cannot run.
    """
    dev = torch.device("cuda")
    batch, s_q, s_kv = 2, 512, 512
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    qo_indptr_buf = torch.arange(0, batch + 1, dtype=torch.int64, device=dev) * s_q
    kv_indptr_buf = torch.arange(0, batch + 1, dtype=torch.int64, device=dev) * s_kv

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        kv_indptr_buf=kv_indptr_buf,
        backend="auto",
    )
    wrapper.plan(
        qo_indptr_buf.to(torch.int32),
        kv_indptr_buf.to(torch.int32),
        32,
        32,
        128,
        head_dim_vo=128,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    # Anything but cudnn: the point is that plan() resolved instead of raising.
    assert wrapper._backend != "cudnn"
