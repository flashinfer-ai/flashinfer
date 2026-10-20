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

flashinfer public-API tests of the CAKE DSA sparse-attention training package
(``flashinfer.experimental.cake_dsa_train``).

The backend suite ``test_cake_dsa_train.py`` imports only the package.  This file covers what needs flashinfer
itself: the experimental public entry points ``flashinfer.dsa_sparse_attention.dsa_sparse_attention`` /
``dsa_sparse_attention_varlen`` (their experimental marking, the arguments they forward, autograd and the varlen
glue through them) and the environment variables the wrapper resolves into the package's keyword switches --
``FLASHINFER_CAKE_DSA_DKV_DIRECT`` (``dkv_direct``), ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP`` (``check_dst_map``) and
``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY`` (``BINDING_CACHE.configure``).  The helpers shared with the
backend suite are repeated here: test modules do not import one another.
"""

import contextlib
import importlib
import warnings

import pytest
import torch

import flashinfer.dsa_sparse_attention as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.dsa_sparse_attention import (
    dsa_sparse_attention,
    dsa_sparse_attention_varlen,
)
from flashinfer.experimental.cake_dsa_train import cake_backend, cake_jit
from flashinfer.experimental.cake_dsa_train.cake_backend import (
    D_LATENT,
    D_QK,
    D_ROPE,
    DQ_PARTIAL_BYTES_PER_TOKEN,
    KEY_PASS_STAGES,
    NUM_HEADS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    KeyPassPolicy,
    derive_topk_length,
    dsa_train_workspace_size,
    generated_program_available,
    offset_gather_kv_indices,
    plan_key_passes,
    prepare_dsa_train,
    record_for,
)
from tests.test_helpers.cake_dsa_train_reference import (
    globalize_gather_indices_loop,
    make_inputs,
    reference_fp64,
    rel_l2,
)

# Accuracy gates (relative L2 vs the chunked FP64 reference of the same BF16
# inputs): the reference FA sparse-MLA numbers times 1.05.
# Forward output: within 5 % (+1e-5) of the numerics floor of a BF16-P kernel on the same inputs
# (``reference_fp64(...)["out_emu"]``); the fixed rel-L2 figures of the design brief are calibrated
# for the iid top-k-2048 configuration and are enforced by the project harness, not per test shape.
FLOOR_MARGIN = 1.05
FLOOR_ABS = 1e-5
# Backward: 1.05x the rel-L2 of the FA sparse-MLA reference kernels on the same inputs (project
# harness values); rope parts at twice the latent gate, peaked attention at one common gate.
GATE_DQ_LATENT = 0.00226
GATE_DKV_LATENT = 0.00243
GATE_ROPE_FACTOR = 2.0
GATE_PEAKED = 0.0025
GATE_LSE_ABS = 2e-5
GATE_ROW_P99_DQ = 0.004

SEED = 20260929


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, backward: bool = False):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda"), backward=backward):
        pytest.skip(
            "generated DSA training program"
            + (" with backward stages" if backward else "")
            + " not registered for this device"
        )


def _rule_passes(
    record: dict, num_queries: int, num_kv: int, topk: int = 2048, arch=None
) -> int:
    """The pass count the record's rule for ``arch`` gives a one-segment row (the test's own evaluation)."""
    policy = KeyPassPolicy.from_record(record)
    rule = policy.rule_for(arch)
    formula = policy.formula_passes(num_kv)
    if formula == 1:
        return 1
    one_chunk = num_queries <= policy.token_chunk(topk)
    tail = bool(rule["tail_rows"]) and 2 * num_queries <= num_kv
    if not (one_chunk or tail):
        return 1
    if num_kv < (rule["min_kv"] if one_chunk else rule["tail_min_kv"]):
        return 1
    passes = rule["fixed_passes"] or formula
    if tail and not one_chunk and rule["tail_cap"]:
        passes = min(passes, rule["tail_cap"])
    return passes if passes <= policy.max_passes else 1


@contextlib.contextmanager
def _quiet_experimental():
    # The experimental banner fires once per process; the API's opt-in is
    # exercised by test_public_api_is_marked_experimental.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _host_inputs(total_q=8, total_k=16, topk=5):
    q = torch.zeros(total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv = torch.zeros(total_k, D_QK, dtype=torch.bfloat16)
    idx = torch.zeros(total_q, topk, dtype=torch.int32)
    return q, kv, idx


def _floor_gate(ref) -> float:
    return FLOOR_MARGIN * rel_l2(ref["out_emu"], ref["out"]) + FLOOR_ABS


def _check_forward(inp, out, lse, ref, *, valid_rows=None):
    assert torch.isfinite(out.float()).all()
    assert rel_l2(out, ref["out"]) <= _floor_gate(ref)
    lse_ref = ref["lse"]
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    if finite.any():
        assert (
            lse.double()[finite] - lse_ref[finite]
        ).abs().max().item() <= GATE_LSE_ABS
    if (~finite).any():
        assert torch.all(lse[~finite] == float("-inf"))
        rows = (
            ~finite.any(-1) if valid_rows is None else ~valid_rows
        )  # fully masked rows
        assert torch.all(out[rows] == 0)


def _check_backward(grads, ref, *, canonical=False, peaked=False):
    """Every gradient within 1.05x its BF16-P/dS numerics floor on the same inputs; the canonical
    iid top-k-2048 configuration also within the fixed gates calibrated there."""
    names = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
    for g in grads:
        assert torch.isfinite(g.float()).all()
    for g, name in zip(grads, names, strict=True):
        got, floor = rel_l2(g, ref[name]), rel_l2(ref[f"{name}_emu"], ref[name])
        assert got <= FLOOR_MARGIN * floor + FLOOR_ABS, (
            f"{name}: rel-L2 {got:.6f} vs floor {floor:.6f}"
        )
    if canonical:
        dq_gate = GATE_PEAKED if peaked else GATE_DQ_LATENT
        dkv_gate = GATE_PEAKED if peaked else GATE_DKV_LATENT
        rope_factor = 1.0 if peaked else GATE_ROPE_FACTOR
        assert rel_l2(grads[0], ref["dq_latent"]) <= dq_gate
        assert rel_l2(grads[1], ref["dq_rope"]) <= dq_gate * rope_factor
        assert rel_l2(grads[2], ref["dkv_latent"]) <= dkv_gate
        assert rel_l2(grads[3], ref["dk_rope"]) <= dkv_gate * rope_factor


def _require_key_pass_program():
    """The registered backward program with the key-range-pass stages (skip otherwise)."""
    _require_program(backward=True)
    _, record = record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages()
    if not set(KEY_PASS_STAGES) <= set(stages):
        pytest.skip("the registered program has no key-range-pass stages")
    return record, stages


def _indices_view_inside_storage(indices: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``indices`` that starts ``offset`` elements inside a larger buffer."""
    n = indices.numel()
    buf = torch.full((n + offset + 64,), -1, dtype=indices.dtype, device=indices.device)
    view = buf[offset : offset + n].view(indices.shape)
    view.copy_(indices)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


def _require_direct_program():
    """The registered program with the natural-layout (direct accumulation) main stage (skip otherwise)."""
    _require_program(backward=True)
    _, record = record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages()
    if not cake_backend.record_direct_stages(record, stages):
        pytest.skip(
            "the registered program has no natural-layout (direct accumulation) main stage"
        )
    return record, stages


def _q_rope_as_q256_slice(q_rope):
    """``q_rope`` as the ``192:256`` channel slice of a ``[T, 64, 256]`` pre-absorption query: head stride 256, token
    stride 16384, storage offset 192 elements (384 B); the other channels are noise the kernels must not read."""
    gen = torch.Generator(device=q_rope.device).manual_seed(756)
    q256 = torch.randn(
        q_rope.shape[0], NUM_HEADS, 256, device=q_rope.device, generator=gen
    ).to(torch.bfloat16)
    q256[:, :, 192:256].copy_(q_rope)
    view = q256[:, :, 192:256]
    assert view.stride() == (16384, 256, 1) and view.storage_offset() == 192
    assert view.data_ptr() == q256.data_ptr() + 384 and view.data_ptr() % 16 == 0
    return q256, view


def _kv_as_packed_row(kv_latent, k_rope, row_stride):
    """``kv_latent`` / ``k_rope`` as the ``0:512`` / ``512:576`` column slices of a packed ``[S, row_stride]`` row
    (576 = latent + rope; 704 = with a frozen 128-channel indexer key stored alongside)."""
    gen = torch.Generator(device=kv_latent.device).manual_seed(row_stride)
    kv = torch.randn(
        kv_latent.shape[0], row_stride, device=kv_latent.device, generator=gen
    ).to(torch.bfloat16)
    kv[:, :D_LATENT].copy_(kv_latent)
    kv[:, D_LATENT:D_QK].copy_(k_rope)
    lat, rope = kv[:, :D_LATENT], kv[:, D_LATENT:D_QK]
    assert lat.stride() == (row_stride, 1) and rope.stride() == (row_stride, 1)
    assert rope.storage_offset() == D_LATENT
    return kv, lat, rope


def _forward_backward(q_latent, q_rope, kv_latent, k_rope, indices, dout):
    out, lse, o_lo = cake_backend.forward(q_latent, q_rope, kv_latent, k_rope, indices)
    grads = cake_backend.backward(
        q_latent, q_rope, kv_latent, k_rope, indices, out, o_lo, lse, dout
    )
    torch.cuda.synchronize()
    return (out, lse, o_lo), grads


def _assert_same_step(got, ref):
    """out / lse / o_lo / dq bitwise; dK / dV within the FP32 reduction spread."""
    for a, b in zip(got[0], ref[0], strict=True):
        assert torch.equal(a, b)
    assert torch.equal(got[1][0], ref[1][0]) and torch.equal(got[1][1], ref[1][1])
    assert max(rel_l2(got[1][2], ref[1][2]), rel_l2(got[1][3], ref[1][3])) < 1e-3


def _non_causal_local_indices(inp, gen, *, masked_docs=()):
    """Distinct uniform picks over the whole document (deliberately including keys after the query), 10 % of the
    slots ``-1`` anywhere, and every slot ``-1`` for the documents in ``masked_docs``."""
    local = torch.empty_like(inp.idx_local)
    cu_q = inp.cu_seqlens_q.tolist()
    topk = inp.topk
    for d, (lq, lk) in enumerate(zip(inp.seq_q, inp.seq_k, strict=True)):
        picks = torch.rand(
            lq, max(lk, topk), device=local.device, generator=gen
        ).argsort(dim=-1)[:, :topk]
        rows = torch.where(picks < lk, picks, torch.full_like(picks, -1)).to(
            torch.int32
        )
        if d in masked_docs:
            rows.fill_(-1)
        local[cu_q[d] : cu_q[d + 1]] = rows
    drop = torch.rand(local.shape, device=local.device, generator=gen) < 0.1
    return torch.where(drop, torch.full_like(local, -1), local)


# ---------------------------------------------------------------------------
# Public entry points (CPU)
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    assert dsa_sparse_attention.is_experimental is True
    assert dsa_sparse_attention_varlen.is_experimental is True
    assert "SM100" in dsa_sparse_attention.__doc__


def test_zero_query_rows_return_empty_outputs_without_binding(monkeypatch):
    """``T == 0``: the eager entry points and the autograd path return empty outputs and zero
    gradients without consulting the registry, loading a module, binding or launching."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "prepare_dsa_train", refuse)
    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    q, kv, idx = _host_inputs(total_q=0, total_k=16, topk=5)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    out, lse, o_lo = cake_backend.forward(ql, qr, kl, kr, idx)
    assert tuple(out.shape) == (0, NUM_HEADS, D_LATENT) and out.dtype == torch.bfloat16
    assert tuple(lse.shape) == (0, NUM_HEADS) and lse.dtype == torch.float32
    assert tuple(o_lo.shape) == tuple(out.shape) and o_lo.dtype == torch.bfloat16
    dout = torch.zeros(0, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    grads = cake_backend.backward(ql, qr, kl, kr, idx, out, o_lo, lse, dout)
    assert [tuple(g.shape) for g in grads] == [
        (0, NUM_HEADS, D_LATENT),
        (0, NUM_HEADS, D_ROPE),
        (16, D_LATENT),
        (16, D_ROPE),
    ]
    assert all(g.dtype == torch.bfloat16 for g in grads)
    assert torch.all(grads[2] == 0) and torch.all(grads[3] == 0)
    f32 = cake_backend.backward(
        ql, qr, kl, kr, idx, out, o_lo, lse, dout, dkv_fp32=True
    )
    assert f32[2].dtype == f32[3].dtype == torch.float32
    assert tuple(f32[2].shape) == (16, D_LATENT) and torch.all(f32[2] == 0)
    # the shape / dtype checks still apply to an empty call
    with pytest.raises(ValueError, match="int32"):
        cake_backend.forward(ql, qr, kl, kr, idx.long())
    with pytest.raises(ValueError, match="S > 0"):
        cake_backend.forward(ql, qr, kl[:0], kr[:0], idx)
    with pytest.raises(ValueError, match="lse"):
        cake_backend.backward(ql, qr, kl, kr, idx, out, o_lo, lse[:, :8], dout)
    # the autograd path
    leaves = [t.detach().clone().requires_grad_() for t in (ql, qr, kl, kr)]
    with _quiet_experimental():
        out_pub, lse_pub = dsa_sparse_attention(*leaves, idx, return_lse=True)
    assert tuple(out_pub.shape) == (0, NUM_HEADS, D_LATENT)
    assert tuple(lse_pub.shape) == (0, NUM_HEADS)
    g = torch.autograd.grad(out_pub, leaves, dout)
    assert [tuple(t.shape) for t in g] == [tuple(leaf.shape) for leaf in leaves]
    assert torch.all(g[2] == 0) and torch.all(g[3] == 0)


# ---------------------------------------------------------------------------
# Environment variables of the flashinfer wrapper (the backend takes keywords only)
# ---------------------------------------------------------------------------


def test_dkv_direct_environment_is_resolved_by_the_public_entry(monkeypatch):
    """``FLASHINFER_CAKE_DSA_DKV_DIRECT`` is read by the public entry when ``dkv_direct`` is ``None``: unset, blank or
    ``auto`` resolve to ``None`` (the record's size rule), ``1`` to ``True`` (the direct path whenever the program
    registers the natural stages), ``0`` to ``False`` (the cast's accumulate path); any other value raises
    ``ValueError`` naming the variable before the backend is reached."""
    env = public_api.DKV_DIRECT_ENV
    assert env == "FLASHINFER_CAKE_DSA_DKV_DIRECT"
    resolve = public_api._dkv_direct_from_environment
    monkeypatch.delenv(env, raising=False)
    assert resolve() is None
    for value, want in (
        (" ", None),
        ("auto", None),
        ("1", True),
        ("0", False),
        (" 1 ", True),
    ):
        monkeypatch.setenv(env, value)
        assert resolve() is want, value
    monkeypatch.setenv(env, "maybe")
    with pytest.raises(ValueError, match=env):
        resolve()
    q, kv, idx = _host_inputs()
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    with _quiet_experimental(), pytest.raises(ValueError, match=env):
        dsa_sparse_attention(ql, qr, kl, kr, idx)
    with _quiet_experimental(), pytest.raises(ValueError, match=env):
        dsa_sparse_attention_varlen(
            ql,
            qr,
            kl,
            kr,
            idx,
            torch.tensor([0, 8], dtype=torch.int32),
            torch.tensor([0, 16], dtype=torch.int32),
        )


def test_dkv_direct_environment_selects_the_backward_path(monkeypatch):
    """Through the public entry with ``dkv_acc`` and S >= 4 T: ``FLASHINFER_CAKE_DSA_DKV_DIRECT=0`` plans the cast
    path, ``1`` and ``auto`` the direct path (the record's rule picks it at S / T = 8), the three accumulate the same
    gradients, and an explicit ``dkv_direct`` keyword wins over the variable."""
    _require_direct_program()
    env = public_api.DKV_DIRECT_ENV
    inp = make_inputs([256], [2048], seed=SEED + 24, topk=128)
    S, S_dst = inp.total_k, 400
    gen = torch.Generator(device="cuda").manual_seed(3)
    dst_map = torch.randint(
        0, S_dst, (S,), device="cuda", generator=gen, dtype=torch.int32
    )
    pre = torch.randn(S_dst, D_QK, device="cuda", generator=gen)
    cache = cake_backend.BINDING_CACHE

    def step(acc, **kwargs):
        cache.clear()
        leaves = [
            t.detach().clone().requires_grad_()
            for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
        ]
        with _quiet_experimental():
            out = dsa_sparse_attention(
                *leaves, inp.idx_global, dkv_acc=acc, dkv_dst_map=dst_map, **kwargs
            )
        out.backward(inp.dout)
        torch.cuda.synchronize()
        assert leaves[2].grad is None and leaves[3].grad is None
        plans = [b.plan for b in cache._entries.values() if b.plan.backward]
        assert len(plans) == 1
        return plans[0].dkv_direct

    results = {}
    for mode, want in (("0", False), ("1", True), ("auto", True)):
        monkeypatch.setenv(env, mode)
        acc = pre.clone()
        assert step(acc) is want, mode
        results[mode] = acc
    torch.testing.assert_close(results["1"], results["0"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(results["auto"], results["1"], rtol=1e-4, atol=1e-4)
    # an explicit keyword wins over the variable; unset, the rule decides
    monkeypatch.setenv(env, "1")
    assert step(pre.clone(), dkv_direct=False) is False
    monkeypatch.setenv(env, "0")
    assert step(pre.clone(), dkv_direct=True) is True
    monkeypatch.delenv(env)
    assert step(pre.clone()) is True


def test_check_dst_map_environment_is_resolved_by_the_public_entry(monkeypatch):
    """``FLASHINFER_CAKE_DSA_CHECK_DST_MAP`` is read by the public entry when ``check_dst_map`` is ``None``: unset,
    empty or ``0`` leave the values of ``dkv_dst_map`` unchecked (the kernel does not range-check the map), anything
    else validates them in the backward and raises ``ValueError`` on a violation; an explicit keyword wins.  A call
    without query rows runs that validation on host tensors without launching."""
    env = public_api.CHECK_DST_MAP_ENV
    assert env == "FLASHINFER_CAKE_DSA_CHECK_DST_MAP"
    q, kv, idx = _host_inputs(total_q=0, total_k=3, topk=5)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    dkv_acc = torch.zeros(4, D_QK)
    good = torch.tensor([3, 0, 3], dtype=torch.int32)
    bad = torch.tensor([0, 4, 1], dtype=torch.int32)  # 4 == S_dst is outside [0, S_dst)

    def grads(dst_map, **kwargs):
        leaves = [t.detach().clone().requires_grad_() for t in (ql, qr)]
        with _quiet_experimental():
            out = dsa_sparse_attention(
                *leaves, kl, kr, idx, dkv_acc=dkv_acc, dkv_dst_map=dst_map, **kwargs
            )
        return torch.autograd.grad(out, leaves, torch.zeros_like(out))

    monkeypatch.delenv(env, raising=False)
    assert public_api._check_dst_map_from_environment() is False
    for dst_map in (good, bad):
        assert [tuple(g.shape) for g in grads(dst_map)] == [
            (0, NUM_HEADS, D_LATENT),
            (0, NUM_HEADS, D_ROPE),
        ]
    for value in ("", "0"):
        monkeypatch.setenv(env, value)
        assert public_api._check_dst_map_from_environment() is False
        grads(bad)
    for value in ("1", "yes"):
        monkeypatch.setenv(env, value)
        assert public_api._check_dst_map_from_environment() is True
        grads(good)
        with pytest.raises(
            ValueError, match=r"dkv_dst_map values must lie in \[0, 4\)"
        ):
            grads(bad)
    grads(bad, check_dst_map=False)  # the keyword wins over the variable
    monkeypatch.delenv(env)
    with pytest.raises(ValueError, match="dkv_dst_map values must lie in"):
        grads(bad, check_dst_map=True)


def test_check_dst_map_environment_checks_the_map_before_the_backward_launches(
    monkeypatch,
):
    """On the device, ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP=1`` makes the public entry's backward validate the map it
    is about to accumulate through: an in-range identity map passes and the step accumulates the same gradients as
    the unchecked step (the check is host validation), an out-of-range row raises ``ValueError`` from the backward."""
    _require_program(backward=True)
    env = public_api.CHECK_DST_MAP_ENV
    inp = make_inputs([128], [256], seed=SEED + 23, topk=64)
    S = inp.total_k
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    identity = torch.arange(S, device="cuda", dtype=torch.int32)

    def step(dst_map):
        acc = torch.zeros(S, D_QK, device="cuda")
        leaves = [t.detach().clone().requires_grad_() for t in args]
        with _quiet_experimental():
            out = dsa_sparse_attention(
                *leaves, inp.idx_global, dkv_acc=acc, dkv_dst_map=dst_map
            )
        out.backward(inp.dout)
        torch.cuda.synchronize()
        return acc

    monkeypatch.setenv(env, "1")
    checked = step(identity)
    assert torch.isfinite(checked).all() and bool((checked != 0).any())
    monkeypatch.setenv(env, "0")
    unchecked = step(identity)
    # mapped adds: FP32 reductions in another order
    assert rel_l2(checked, unchecked) < 1e-4
    bad = identity.clone()
    bad[5] = S
    monkeypatch.setenv(env, "1")
    with pytest.raises(ValueError, match=r"dkv_dst_map values must lie in \[0, 256\)"):
        step(bad)


def test_binding_cache_capacity_environment_configures_the_backend_cache(monkeypatch):
    """``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY`` is read when :mod:`flashinfer.dsa_sparse_attention` is
    imported (a malformed value fails there, naming the variable) and applied to the backend's ``BINDING_CACHE``
    through ``configure`` when the module first reaches the backend (the import itself stays deferred); unset or
    blank leaves the configured capacity alone."""
    env = public_api.BINDING_CACHE_CAPACITY_ENV
    assert env == "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY"
    cache = cake_backend.BINDING_CACHE
    before = cache.capacity
    try:
        monkeypatch.setenv(env, "5")
        importlib.reload(public_api)
        # not yet: the backend is reached on the first call
        assert cache.capacity == before
        assert public_api._backend("cake") is cake_backend
        assert cache.capacity == 5
        for blank in (" ", ""):
            monkeypatch.setenv(env, blank)
            importlib.reload(public_api)
            public_api._backend("cake")
            assert cache.capacity == 5  # blank: left as configured
        for bad in ("0", "-1", "many", "2.5"):
            monkeypatch.setenv(env, bad)
            with pytest.raises(ValueError, match=env):
                importlib.reload(public_api)
        monkeypatch.setenv(env, "7")
        importlib.reload(public_api)
        public_api._backend("cake")
        assert cache.capacity == 7
        monkeypatch.delenv(env)
        importlib.reload(public_api)
        public_api._backend("cake")
        assert cache.capacity == 7  # unset: left as configured
        with pytest.raises(ValueError, match="backend"):
            public_api._backend("other")
    finally:
        monkeypatch.delenv(env, raising=False)
        importlib.reload(public_api)
        cache.configure(capacity=before)


# ---------------------------------------------------------------------------
# Device tests through the public entry points (compute capability 10.0 / 10.3 / 10.7, registered program)
# ---------------------------------------------------------------------------


def test_varlen_multi_document_matches_flat_and_reference():
    _require_program()
    inp = make_inputs([64, 200, 120], [128, 200, 384], seed=SEED + 4, topk=128)
    out_flat, lse_flat, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    with _quiet_experimental():
        out_var, lse_var = dsa_sparse_attention_varlen(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            return_lse=True,
        )
    torch.cuda.synchronize()
    assert torch.equal(out_flat, out_var) and torch.equal(lse_flat, lse_var)
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    _check_forward(inp, out_var, lse_var, ref)


def test_backward_whole_row_policy_two_passes_through_public_entry():
    """The registered policy takes two passes at T = 4096 x S = 65,536 (top-k 2048) under the R1
    rule (the sm_100a / sm_103a records; the sm_107a rule keeps that row single-pass, so the test
    forces two there), through the public entry, against the canonical gates; the 4k x 4k row stays
    single-pass and every row follows the record's published rule for the device's architecture."""
    record, stages = _require_key_pass_program()
    device = torch.device("cuda")
    arch = cake_backend.arch_for(device)
    # without an architecture the record's constants give the R1 rule
    assert plan_key_passes(record, stages, 4096, 65536, 2048) == 2
    assert plan_key_passes(record, stages, 4096, 131072, 2048) == 3
    assert plan_key_passes(record, stages, 4096, 4096, 2048) == 1
    assert plan_key_passes(record, stages, 32768, 131072, 2048) == 1
    for T, S in (
        (4096, 65536),
        (4096, 131072),
        (4096, 4096),
        (32768, 131072),
        (8448, 131072),
    ):
        assert plan_key_passes(record, stages, T, S, 2048, arch=arch) == _rule_passes(
            record, T, S, arch=arch
        ), (T, S)
    forced = None if _rule_passes(record, 4096, 65536, arch=arch) == 2 else 2
    policy_size = dsa_train_workspace_size(4096, 65536, 2048, device, key_passes=forced)
    single_size = dsa_train_workspace_size(4096, 65536, 2048, device, key_passes=1)
    assert policy_size - single_size >= 4096 * (
        DQ_PARTIAL_BYTES_PER_TOKEN + 4 * 2048 + 4
    )
    assert policy_size == dsa_train_workspace_size(
        4096, 65536, 2048, device, key_passes=2
    )
    inp = make_inputs([4096], [65536], seed=SEED + 17, topk=2048)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out = dsa_sparse_attention(*leaves, inp.idx_global, key_passes=forced)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    torch.cuda.synchronize()
    # two passes for this row (one token chunk): per pass the compaction and the pass kernel
    runner = prepare_dsa_train(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        backward=True,
        key_passes=forced,
    )
    assert runner.key_passes == 2
    assert [k for k in runner.backward_order if isinstance(k, tuple)] == [
        ("bwd_compact", 0, 0),
        ("bwd_main_pass", 0, 0),
        ("bwd_compact", 0, 1),
        ("bwd_main_pass", 0, 1),
    ]
    del runner
    # repeating the step with the same binding is bitwise for the forward and dq
    fwd_args = tuple(t.detach() for t in leaves) + (inp.idx_global,)
    o1, l1, olo1 = cake_backend.forward(*fwd_args)
    g1 = cake_backend.backward(*fwd_args, o1, olo1, l1, inp.dout, key_passes=forced)
    g2 = cake_backend.backward(*fwd_args, o1, olo1, l1, inp.dout, key_passes=forced)
    o2, l2, olo2 = cake_backend.forward(*fwd_args)
    torch.cuda.synchronize()
    assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])
    assert torch.equal(g1[0], grads[0]) and torch.equal(o2, out.detach())
    with _quiet_experimental():
        out_single = dsa_sparse_attention(*leaves, inp.idx_global, key_passes=1)
    single = torch.autograd.grad(out_single, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out.detach(), out_single.detach())  # the forward is untouched
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_backward(grads, ref, canonical=True)
    _check_backward(single, ref, canonical=True)
    assert rel_l2(grads[0], single[0]) < 1e-3 and rel_l2(grads[1], single[1]) < 1e-3
    assert max(rel_l2(grads[2], single[2]), rel_l2(grads[3], single[3])) < 1e-2


def test_public_entry_derives_row_lengths_and_matches_full_rows():
    """Without ``topk_length`` the public entry derives the per-row length (last valid slot + 1) and the kernels skip
    the trailing invalid blocks: on documents shorter than top-k (most slots ``-1``) the forward / backward equal the
    explicit full-length call bitwise (out, lse, dq) and within the FP32 reduction spread (dkv), and match the
    reference."""
    _require_program(backward=True)
    inp = make_inputs([200, 96, 300], [200, 96, 4096], seed=SEED + 752, topk=256)
    S = inp.kv_latent.shape[0]
    derived = derive_topk_length(inp.idx_global, S)
    assert torch.equal(
        derived, inp.topk_length
    )  # the generator's rows are valid-first: last valid + 1 == count
    assert int(derived.max()) <= 256 and int((derived < 256).sum()) > 0
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    leaves_a = [t.detach().clone().requires_grad_() for t in args]
    leaves_b = [t.detach().clone().requires_grad_() for t in args]
    full = torch.full(
        (inp.q_latent.shape[0],), 256, dtype=torch.int32, device=inp.q_latent.device
    )
    with _quiet_experimental():
        out_a, lse_a = dsa_sparse_attention(*leaves_a, inp.idx_global, return_lse=True)
        out_b, lse_b = dsa_sparse_attention(
            *leaves_b, inp.idx_global, topk_length=full, return_lse=True
        )
    grads_a = torch.autograd.grad(out_a, leaves_a, inp.dout)
    grads_b = torch.autograd.grad(out_b, leaves_b, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_a.detach(), out_b.detach()) and torch.equal(lse_a, lse_b)
    assert torch.equal(grads_a[0], grads_b[0]) and torch.equal(grads_a[1], grads_b[1])
    # dkv: fp32 atomics summed in a different key-block order round to bf16 differently on a few elements
    # (the cross-path spread used throughout this file); both paths must also meet the reference gates
    assert max(rel_l2(grads_a[2], grads_b[2]), rel_l2(grads_a[3], grads_b[3])) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_forward(inp, out_a, lse_a, ref)
    _check_backward(grads_a, ref)
    _check_backward(grads_b, ref)


def test_varlen_multi_segment_row_plans_single_pass():
    """A packed two-segment key row whose total length triggers the whole-row formula (2 x 23,000 keys >
    45,511) plans the single-pass stage through the varlen entry (``num_segments = len(cu_seqlens_k) - 1``:
    whole-row ranges do not match segment-confined index rows), while the flat call on the same global
    indices plans two whole-row passes; the forward is untouched, both backwards agree (dq differs in the
    FP32 summation order only) and match the FP64 reference."""
    record, stages = _require_key_pass_program()
    device = torch.device("cuda")
    assert plan_key_passes(record, stages, 256, 46000, 128) == 2
    assert plan_key_passes(record, stages, 256, 46000, 128, num_segments=2) == 1
    assert dsa_train_workspace_size(
        256, 46000, 128, device, num_segments=2
    ) == dsa_train_workspace_size(256, 46000, 128, device, key_passes=1)
    inp = make_inputs([128, 128], [23000, 23000], seed=SEED + 751, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    leaves = [t.detach().clone().requires_grad_() for t in args]
    flat_leaves = [t.detach().clone().requires_grad_() for t in args]
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    with _quiet_experimental():
        out_v, lse_v = dsa_sparse_attention_varlen(
            *leaves,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            return_lse=True,
        )
    grads_v = torch.autograd.grad(out_v, leaves, inp.dout)
    torch.cuda.synchronize()
    remembered = [b.plan for b in cache._entries.values() if b.plan.backward]
    assert [plan.key_passes for plan in remembered] == [1]
    assert not any(isinstance(k, tuple) for k in remembered[0].backward_order)
    with _quiet_experimental():
        out_f, lse_f = dsa_sparse_attention(
            *flat_leaves, inp.idx_global, return_lse=True
        )
    grads_f = torch.autograd.grad(out_f, flat_leaves, inp.dout)
    torch.cuda.synchronize()
    # the segment count is part of the plan key: the flat call plans anew and takes the passes of the
    # device's rule for the whole row (two under the R1 rule; a one-chunk row below the sm_107a key floor
    # stays single-pass there)
    flat_passes = _rule_passes(
        record, 256, 46000, topk=128, arch=cake_backend.arch_for(device)
    )
    assert sorted(
        b.plan.key_passes for b in cache._entries.values() if b.plan.backward
    ) == sorted([1, flat_passes])
    assert torch.equal(out_v.detach(), out_f.detach()) and torch.equal(lse_v, lse_f)
    for a, b in zip(grads_v, grads_f, strict=True):
        assert rel_l2(a, b) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_forward(inp, out_v, lse_v, ref)
    _check_backward(grads_v, ref)
    _check_backward(grads_f, ref)


@pytest.mark.parametrize("offset", [1, 8, 13])
def test_indices_view_inside_a_storage_matches_the_plain_tensor(offset):
    """A contiguous ``indices`` view that starts inside its storage (a slice of a larger buffer)
    reaches the kernels as the nearest vector-aligned base plus the element offset, so their
    aligned index-tile loads (gated on the offset) stay valid: forward, backward and the autograd
    path agree with the plain tensor (bitwise where the kernels are deterministic)."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 18, topk=128)
    view = _indices_view_inside_storage(inp.idx_global, offset)
    base, element_offset = cake_backend._pointer_operand(view)
    assert element_offset == offset % 8
    assert base.data_ptr() == view.data_ptr() - 4 * element_offset
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    plain = cake_backend.forward(*args, inp.idx_global)
    got = cake_backend.forward(*args, view)
    torch.cuda.synchronize()
    for a, b in zip(plain, got, strict=True):
        assert torch.equal(a, b)
    out, lse, o_lo = plain
    g_plain = cake_backend.backward(*args, inp.idx_global, out, o_lo, lse, inp.dout)
    g_view = cake_backend.backward(*args, view, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(g_plain[0], g_view[0]) and torch.equal(g_plain[1], g_view[1])
    assert max(rel_l2(g_plain[2], g_view[2]), rel_l2(g_plain[3], g_view[3])) < 1e-3
    leaves = [t.detach().clone().requires_grad_() for t in args]
    with _quiet_experimental():
        out_pub, lse_pub = dsa_sparse_attention(*leaves, view, return_lse=True)
    grads = torch.autograd.grad(out_pub, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_pub, out) and torch.equal(lse_pub, lse)
    assert torch.equal(grads[0], g_plain[0]) and torch.equal(grads[1], g_plain[1])
    assert max(rel_l2(grads[2], g_plain[2]), rel_l2(grads[3], g_plain[3])) < 1e-3


def test_autograd_function_matches_explicit_backward():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 10, topk=128)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention(*leaves, inp.idx_global, return_lse=True)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    o2, l2, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    explicit = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        o2,
        o_lo,
        l2,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, o2) and torch.equal(lse, l2)
    assert torch.equal(grads[0], explicit[0]) and torch.equal(grads[1], explicit[1])
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_backward(grads, ref)


def test_autograd_lse_gradient_is_rejected_and_unused_out_gives_no_grad():
    _require_program(backward=True)
    inp = make_inputs([128], [256], seed=SEED + 15, topk=64)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention(*leaves, inp.idx_global, return_lse=True)
    # only out is differentiable: a gradient arriving through lse fails loudly instead of being dropped
    with pytest.raises(NotImplementedError, match="lse"):
        torch.autograd.grad(lse.sum(), leaves)
    g = torch.autograd.grad(out, leaves, inp.dout)
    torch.cuda.synchronize()
    assert all(
        t is not None and t.shape == leaf.shape
        for t, leaf in zip(g, leaves, strict=True)
    )


def test_forward_backward_strided_q_rope_256_bitwise():
    """``q_rope`` as the ``192:256`` channel slice of a ``[T, 64, 256]`` query (head stride 256, token stride 16384,
    384 B storage offset) with a contiguous ``q_latent``: forward and backward equal the contiguous path bitwise (the
    TMA descriptor takes the view's strides; the computation order is unchanged), dK / dV within the FP32 reduction
    spread; the view is consumed in place, the remembered binding serves it, and the autograd path puts the rope
    gradient into channels 192:256 of the [T, 64, 256] leaf."""
    _require_program(backward=True)
    inp = make_inputs([160, 96], [160, 320], seed=SEED + 756, topk=96)
    q256, q_rope = _q_rope_as_q256_slice(inp.q_rope)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ref = _forward_backward(*args, inp.idx_global, inp.dout)
    got = _forward_backward(
        inp.q_latent, q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, inp.dout
    )
    _assert_same_step(got, ref)
    assert q_rope.data_ptr() == q256.data_ptr() + 384 and q_rope.stride() == (
        16384,
        256,
        1,
    )
    again = cake_backend.forward(
        inp.q_latent, q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    torch.cuda.synchronize()
    for a, b in zip(again, got[0], strict=True):
        assert torch.equal(a, b)
    leaf = q256.detach().clone().requires_grad_()
    with _quiet_experimental():
        out = dsa_sparse_attention(
            inp.q_latent, leaf[:, :, 192:256], inp.kv_latent, inp.k_rope, inp.idx_global
        )
    (grad,) = torch.autograd.grad(out, [leaf], inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out, ref[0][0]) and torch.equal(grad[:, :, 192:256], ref[1][1])
    assert torch.all(grad[:, :, :192] == 0)


def test_kv_row_stride_704_bitwise():
    """``kv_latent = kv[:, :512]`` / ``k_rope = kv[:, 512:576]`` of a packed ``[S, 704]`` row (indexer key alongside)
    and of a ``[S, 576]`` row: forward and backward equal the contiguous path and each other bitwise (gather
    descriptor / rope pointer stride only); the ``[S, 704]`` autograd leaf gets its gradient in columns 0:576 only."""
    _require_program(backward=True)
    inp = make_inputs([128, 128], [128, 384], seed=SEED + 757, topk=96)
    ref = _forward_backward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, inp.dout
    )
    got = {}
    for row_stride in (576, 704):
        kv, lat, rope = _kv_as_packed_row(inp.kv_latent, inp.k_rope, row_stride)
        got[row_stride] = _forward_backward(
            inp.q_latent, inp.q_rope, lat, rope, inp.idx_global, inp.dout
        )
        _assert_same_step(got[row_stride], ref)
    _assert_same_step(got[704], got[576])
    leaf = kv.detach().clone().requires_grad_()  # the [S, 704] buffer
    with _quiet_experimental():
        out = dsa_sparse_attention(
            inp.q_latent,
            inp.q_rope,
            leaf[:, :D_LATENT],
            leaf[:, D_LATENT:D_QK],
            inp.idx_global,
        )
    (grad,) = torch.autograd.grad(out, [leaf], inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out, ref[0][0]) and torch.all(grad[:, D_QK:] == 0)
    assert (
        max(
            rel_l2(grad[:, :D_LATENT], ref[1][2]),
            rel_l2(grad[:, D_LATENT:D_QK], ref[1][3]),
        )
        < 1e-3
    )


def test_varlen_causal_tail_of_prefix_matches_reference():
    """Query segments that are the tail of their key prefix (seqlen_q < seqlen_k) with gather indices that
    deliberately select keys beyond the causal bound ``(seqlen_k - seqlen_q) + local_q``: the varlen entry masks
    exactly those slots (loop reference of the rule; the FP64 reference sees the same set), a zero-query segment
    contributes no rows, and a segment whose every slot is ``-1`` gives ``out = 0``, ``lse = -inf``, ``dq = 0`` and
    exact zeros in dK / dV for its keys."""
    _require_program(backward=True)
    seq_q, seq_k = (
        [40, 48, 0, 56, 32],
        [40, 200, 64, 320, 96],
    )  # full, tail, keys only, tail, tail (all -1)
    inp = make_inputs(seq_q, seq_k, seed=SEED + 758, topk=64)
    gen = torch.Generator(device="cuda").manual_seed(1)
    local = _non_causal_local_indices(inp, gen, masked_docs=(4,))
    cu_q, cu_k = inp.cu_seqlens_q, inp.cu_seqlens_k
    expected = globalize_gather_indices_loop(local, cu_q, cu_k, causal=True).cuda()
    assert torch.equal(
        offset_gather_kv_indices(local, cu_q, cu_k, causal=True), expected
    )
    loose = offset_gather_kv_indices(local, cu_q, cu_k, causal=False)
    assert torch.equal(
        loose, globalize_gather_indices_loop(local, cu_q, cu_k, causal=False).cuda()
    )
    assert ((expected == -1) & (loose != -1)).any(), (
        "the case must select keys beyond the causal bound"
    )
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    (out, lse, _), grads = _forward_backward(*args, expected, inp.dout)
    leaves = [t.detach().clone().requires_grad_() for t in args]
    with _quiet_experimental():
        out_v, lse_v = dsa_sparse_attention_varlen(
            *leaves,
            local,
            cu_q,
            cu_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            return_lse=True,
        )
    grads_v = torch.autograd.grad(out_v, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_v, out) and torch.equal(lse_v, lse)
    assert torch.equal(grads_v[0], grads[0]) and torch.equal(grads_v[1], grads[1])
    assert max(rel_l2(grads_v[2], grads[2]), rel_l2(grads_v[3], grads[3])) < 1e-3
    ref = reference_fp64(*args, expected, dout=inp.dout)
    _check_forward(inp, out_v, lse_v, ref)
    _check_backward(grads_v, ref)
    masked = slice(int(cu_q[-2]), int(cu_q[-1]))
    assert torch.all(lse_v[masked] == float("-inf")) and torch.all(out_v[masked] == 0)
    assert torch.all(grads_v[0][masked] == 0) and torch.all(grads_v[1][masked] == 0)
    assert torch.all(grads_v[2][int(cu_k[-2]) :] == 0) and torch.all(
        grads_v[3][int(cu_k[-2]) :] == 0
    )
    with _quiet_experimental():
        out_nc = dsa_sparse_attention_varlen(*args, local, cu_q, cu_k, causal=False)
    torch.cuda.synchronize()
    assert not torch.equal(out_nc, out_v), (
        "without the causal term the same picks attend keys after the query"
    )


def test_varlen_segment_with_every_slot_masked_gives_zero_outputs():
    """A packed segment whose gather rows are all ``-1``: ``out = 0``, ``lse = -inf``, ``dq = 0`` for its rows and
    exact zeros in dK / dV for its keys; the other segment's row-local results are those of the flat call."""
    _require_program(backward=True)
    inp = make_inputs([64, 48], [64, 160], seed=SEED + 759, topk=64)
    local = inp.idx_local.clone()
    local[64:] = -1
    cu_q, cu_k = inp.cu_seqlens_q, inp.cu_seqlens_k
    glob = offset_gather_kv_indices(local, cu_q, cu_k)
    assert torch.all(glob[64:] == -1) and torch.equal(glob[:64], inp.idx_global[:64])
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    leaves = [t.detach().clone().requires_grad_() for t in args]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention_varlen(
            *leaves, local, cu_q, cu_k, return_lse=True
        )
    grads = torch.autograd.grad(out, leaves, inp.dout)
    (out_flat, lse_flat, _), grads_flat = _forward_backward(
        *args, inp.idx_global, inp.dout
    )
    torch.cuda.synchronize()
    assert torch.all(lse[64:] == float("-inf")) and torch.all(out[64:] == 0)
    assert torch.all(grads[0][64:] == 0) and torch.all(grads[1][64:] == 0)
    assert torch.all(grads[2][64:] == 0) and torch.all(grads[3][64:] == 0)
    assert torch.equal(out[:64], out_flat[:64]) and torch.equal(lse[:64], lse_flat[:64])
    assert torch.equal(grads[0][:64], grads_flat[0][:64]) and torch.equal(
        grads[1][:64], grads_flat[1][:64]
    )
    ref = reference_fp64(*args, glob, dout=inp.dout)
    _check_forward(inp, out, lse, ref)
    _check_backward(grads, ref)


def test_varlen_packed_glm_structure_small():
    """Scaled-down GLM-5.2 packed batch: 8 segments alternating a full document and the tail of a ~32x longer key
    prefix, top-k 128, through the varlen entry -- equals the flat call on the generator's global indices (bitwise
    for out / lse / dq) and the FP64 reference; the generator's causal picks are left untouched by the rule."""
    _require_program(backward=True)
    seq_q = [177, 175, 212, 211, 212, 211, 212, 211]
    seq_k = [177, 5684, 212, 6792, 212, 6792, 212, 6792]
    inp = make_inputs(seq_q, seq_k, seed=SEED + 760, topk=128)
    assert torch.equal(
        offset_gather_kv_indices(inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k),
        inp.idx_global,
    )
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    (out, lse, _), grads = _forward_backward(*args, inp.idx_global, inp.dout)
    leaves = [t.detach().clone().requires_grad_() for t in args]
    with _quiet_experimental():
        out_v, lse_v = dsa_sparse_attention_varlen(
            *leaves,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            return_lse=True,
        )
    grads_v = torch.autograd.grad(out_v, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_v, out) and torch.equal(lse_v, lse)
    assert torch.equal(grads_v[0], grads[0]) and torch.equal(grads_v[1], grads[1])
    assert max(rel_l2(grads_v[2], grads[2]), rel_l2(grads_v[3], grads[3])) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_forward(inp, out_v, lse_v, ref)
    _check_backward(grads_v, ref)
