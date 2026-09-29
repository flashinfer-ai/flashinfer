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
            and (d_qk, d_vo) in {(128, 128), (192, 128)}
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
@pytest.mark.parametrize("d_qk,d_vo", [(256, 256), (64, 64), (192, 192)])
def test_auto_declines_outside_cutlass_head_dims(d_qk, d_vo):
    """`auto` routes to CUTLASS only at (128,128) and (192,128).

    ``DISPATCH_head_dim`` accepts (192,128), (128,128) and (64,64), so 192/192
    is outside the kernel and d64 is inside it but outside `auto`'s measured
    domain. Both must stay off the CUTLASS path.
    """
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


@requires_cutlass_arch
def test_auto_declines_cutlass_under_cuda_graph(monkeypatch):
    """`fmha_varlen_plan` reallocates its work buffers per plan() call.

    A captured graph would keep pointing at the previous allocation, so CUTLASS
    is not graph-safe; `auto` must route elsewhere rather than hand back a
    backend that replays a stale plan.
    """
    monkeypatch.setenv("FLASHINFER_RAGGED_AUTO_BACKEND_ORDER", "cutlass")
    dev = torch.device("cuda")
    batch, s_q, s_kv = 2, 512, 512
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_q
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_kv

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.clone(),
        kv_indptr_buf=kv_indptr.clone(),
        backend="auto",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        32,
        32,
        128,
        head_dim_vo=128,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper._backend == "fa2"


@requires_cutlass_arch
def test_explicit_cutlass_refuses_cuda_graph():
    """An explicit `backend="cutlass"` says so plainly instead of going stale."""
    dev = torch.device("cuda")
    batch, s_q, s_kv = 2, 512, 512
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_q
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_kv

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.clone(),
        kv_indptr_buf=kv_indptr.clone(),
        backend="cutlass",
    )
    with pytest.raises(ValueError, match="not CUDA-graph safe"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            32,
            32,
            128,
            head_dim_vo=128,
            causal=True,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


# ---------------------------------------------------------------------------
# Follow-ups from review: sinks, fp8, int64 indptrs, non-contiguous q/k/v
# ---------------------------------------------------------------------------


def _new_wrapper(backend):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )


@requires_cutlass_arch
def test_auto_declines_with_wrapper_sinks():
    """vLLM's metadata builder stashes attention sinks on the wrapper (``_sinks``)
    for the fa2 route; neither upgraded kernel consumes them, so `auto` must
    stay on fa2 rather than silently drop the sink."""
    wrapper = _new_wrapper("auto")
    wrapper._sinks = torch.zeros(64, dtype=torch.float32, device="cuda")
    _, _, _, qo_indptr, kv_indptr = _inputs(4, 1024, 1024, 64, 8, 128, 128)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        64,
        8,
        128,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper._backend == "fa2"
    del wrapper._sinks
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        64,
        8,
        128,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper._backend == BLACKWELL_DEFAULT


@requires_cutlass_arch
def test_upgraded_backends_refuse_sinks_set_after_plan():
    """Sinks that appear between plan() and run() (the vLLM order) hit an
    explicit error on the upgraded backends, never a silent drop."""
    q, k, v, qo_indptr, kv_indptr = _inputs(4, 512, 512, 64, 8, 128, 128)
    wrapper = _new_wrapper("auto")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        64,
        8,
        128,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper._backend in ("cudnn", "cutlass")
    wrapper._sinks = torch.zeros(64, dtype=torch.float32, device="cuda")
    with pytest.raises(NotImplementedError, match="attention sinks"):
        wrapper.run(q, k, v)


@requires_cudnn_upgrade
def test_auto_keeps_fp8_off_cudnn():
    """`_cudnn_supports_direct_seqlens` is True for fp8 on recent cuDNN, but the
    wrapper's cuDNN run branch takes the tensors as they are and the fp8 scale
    handling lives on the paths after it: fp8 q/k/v must not resolve to cudnn.
    Checked on the eligibility function itself -- a full fp8 plan() has no
    Blackwell ragged backend to land on (fa2 rejects fp8 q at JIT time)."""
    from flashinfer.prefill import _blackwell_ragged_auto_upgrade

    dev = torch.device("cuda")
    common = dict(
        pos_encoding_mode=0,
        has_custom_mask=False,
        window_left=-1,
        logits_soft_cap=0.0,
        has_multi_item_scoring=False,
        has_sinks=False,
        cudnn_indptr_is_int32=True,
        cutlass_work_items=1,
        cuda_graph_enabled=False,
    )
    assert (
        _blackwell_ragged_auto_upgrade(
            dev, "NHD", 128, 128, DTYPE, DTYPE, DTYPE, **common
        )
        == "cudnn"
    )
    fp8 = torch.float8_e4m3fn
    assert (
        _blackwell_ragged_auto_upgrade(dev, "NHD", 128, 128, fp8, fp8, fp8, **common)
        is None
    )
    assert (
        _blackwell_ragged_auto_upgrade(
            dev, "NHD", 128, 128, fp8, fp8, torch.bfloat16, **common
        )
        is None
    )


@requires_cudnn_upgrade
def test_explicit_cudnn_int64_indptr_converts_all_four_buffers():
    """int64 token indptrs outside graph mode: plan() re-dtypes q/kv AND the o/v
    buffers that default to aliases of them, so cuDNN never sees mixed dtypes."""
    h_qo, h_kv, d = 64, 8, 128
    q, k, v, indptr, _ = _varlen_inputs(6, 1024, h_qo, h_kv, d, d)
    indptr64 = indptr.to(torch.int64)
    _, out_fa2, lse_fa2 = _run_varlen("fa2", q, k, v, indptr, h_qo, h_kv, d, d)
    wrapper = _new_wrapper("cudnn")
    wrapper.plan(
        indptr64,
        indptr64,
        h_qo,
        h_kv,
        d,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    for name in ("_qo_indptr_buf", "_kv_indptr_buf", "_o_indptr_buf", "_v_indptr_buf"):
        assert getattr(wrapper, name).dtype == torch.int32, name
    out, lse = wrapper.run(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "explicit cudnn, int64 indptr")


def _t3hd_views(batch, s_max, h, d, seed=777):
    """q/k/v as views into one packed [tokens, 3, h, d] buffer (token stride 3*h*d)."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(max(1, s_max // 4), s_max + 1, (batch,), generator=g)
    indptr = torch.zeros(batch + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, 0)
    total = int(indptr[-1])
    torch.manual_seed(seed)
    qkv = torch.randn(total, 3, h, d, dtype=DTYPE, device=dev)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    assert not q.is_contiguous() and q.stride(0) == 3 * h * d
    return q, k, v, indptr.to(dev)


@requires_cudnn_upgrade
def test_cudnn_ragged_handles_packed_t3hd_views():
    """cuDNN follows the tensors' strides; the token-unit ragged offsets are
    scaled by each tensor's real token stride (3*h*d here), and the graph cache
    keys on strides, so a contiguous call of the same shape first does not
    replay its graph for the packed views."""
    h, d = 8, 128
    q, k, v, indptr = _t3hd_views(6, 512, h, d)
    qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
    # contiguous graph first, same shapes -- must not be reused for the views
    _, out_c, lse_c = _run_varlen("cudnn", qc, kc, vc, indptr, h, h, d, d)
    _, out_fa2, lse_fa2 = _run_varlen("fa2", q, k, v, indptr, h, h, d, d)
    resolved, out, lse = _run_varlen("cudnn", q, k, v, indptr, h, h, d, d)
    assert resolved == "cudnn"
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "cudnn on packed T3HD views")
    _assert_close_to_fa2(out_c, lse_c, out_fa2, lse_fa2, "cudnn on contiguous copies")


def _single_token_rows(h_qo, h_kv, d, batch=8, seed=99):
    """Every request contributes one query token (a chunked-prefill remainder
    step) against a longer kv range: qo_indptr = arange, kv_indptr ragged."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    kv_lens = torch.randint(1, 1024, (batch,), generator=g)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_lens, 0)
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    torch.manual_seed(seed)
    q = torch.randn(batch, h_qo, d, dtype=DTYPE, device=dev)
    k = torch.randn(int(kv_indptr[-1]), h_kv, d, dtype=DTYPE, device=dev)
    v = torch.randn(int(kv_indptr[-1]), h_kv, d, dtype=DTYPE, device=dev)
    return q, k, v, qo_indptr.to(dev), kv_indptr.to(dev)


def _plan_single_token(backend, qo_indptr, kv_indptr, h_qo, h_kv, d):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d,
        head_dim_vo=d,
        causal=False,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    return wrapper


@requires_cudnn_upgrade
def test_auto_declines_single_token_gqa_rows():
    """cuDNN's s_q == 1 kernel packs a kv group's q heads and writes the LSE only
    for the first of them (cuDNN 9.26/9.27), so `auto` keeps single-token GQA
    steps off cuDNN; the results, LSE included, match fa2. MHA single-token rows
    are unaffected and may still take cuDNN."""
    h_qo, h_kv, d = 32, 8, 128
    q, k, v, qo_indptr, kv_indptr = _single_token_rows(h_qo, h_kv, d)
    w = _plan_single_token("auto", qo_indptr, kv_indptr, h_qo, h_kv, d)
    assert w._backend != "cudnn"
    out, lse = w.run(q, k, v, return_lse=True)
    wf = _plan_single_token("fa2", qo_indptr, kv_indptr, h_qo, h_kv, d)
    out_fa2, lse_fa2 = wf.run(q, k, v, return_lse=True)
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "auto on single-token GQA rows")
    # MHA: the kernel is correct, cuDNN stays eligible
    q, k, v, qo_indptr, kv_indptr = _single_token_rows(h_qo, h_qo, d)
    w = _plan_single_token("auto", qo_indptr, kv_indptr, h_qo, h_qo, d)
    assert w._backend == "cudnn"
    out, lse = w.run(q, k, v, return_lse=True)
    wf = _plan_single_token("fa2", qo_indptr, kv_indptr, h_qo, h_qo, d)
    out_fa2, lse_fa2 = wf.run(q, k, v, return_lse=True)
    _assert_close_to_fa2(out, lse, out_fa2, lse_fa2, "cudnn on single-token MHA rows")


@requires_cudnn_upgrade
def test_explicit_cudnn_refuses_single_token_gqa_lse():
    h_qo, h_kv, d = 32, 8, 128
    q, k, v, qo_indptr, kv_indptr = _single_token_rows(h_qo, h_kv, d)
    w = _plan_single_token("cudnn", qo_indptr, kv_indptr, h_qo, h_kv, d)
    with pytest.raises(NotImplementedError, match="single-token"):
        w.run(q, k, v, return_lse=True)
    # the output itself is correct, so the no-LSE call is allowed
    out = w.run(q, k, v)
    wf = _plan_single_token("fa2", qo_indptr, kv_indptr, h_qo, h_kv, d)
    out_fa2 = wf.run(q, k, v)
    torch.testing.assert_close(out.float(), out_fa2.float(), atol=2e-2, rtol=2e-2)


@requires_cudnn_upgrade
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,  # a build/execute failure is not the known LSE bug
    reason="cuDNN s_q==1 GQA kernel writes the ragged Stats only for the first head "
    "of each kv group (cuDNN 9.26/9.27, NVBug 6783545); drop the guards above when this passes",
)
def test_cudnn_single_token_gqa_lse_is_correct():
    from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache

    h_qo, h_kv, d = 32, 8, 128
    q, k, v, qo_indptr, kv_indptr = _single_token_rows(h_qo, h_kv, d)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        float(1.0 / d**0.5),
        workspace,
        max_token_per_sequence=1,
        max_sequence_kv=int((kv_indptr[1:] - kv_indptr[:-1]).max()),
        causal=False,
        return_lse=True,
        batch_offsets_q=qo_indptr,
        batch_offsets_o=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_v=kv_indptr,
        batch_offsets_stats=qo_indptr,
        batch_offsets_units="tokens",
        lse=torch.empty(q.shape[0], h_qo, dtype=torch.float32, device="cuda"),
    )
    wf = _plan_single_token("fa2", qo_indptr, kv_indptr, h_qo, h_kv, d)
    _, lse_fa2 = wf.run(q, k, v, return_lse=True)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, lse_fa2, atol=1e-2, rtol=1e-2)
