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

import contextlib
import warnings
from types import SimpleNamespace

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.dsa_sparse_attention import (
    dsa_sparse_attention,
    dsa_sparse_attention_varlen,
)
from flashinfer.experimental.cake_dsa_train import cake_backend, cake_jit
from flashinfer.experimental.cake_dsa_train.cake_backend import (
    ABI_CONTRACT,
    ABI_SEED,
    D_LATENT,
    D_QK,
    D_ROPE,
    DKV_ACC_LAYOUTS,
    DQ_PARTIAL_BYTES_PER_TOKEN,
    KEY_PASS_STAGES,
    NUM_HEADS,
    SUPPORTED_ABIS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    WORKSPACE_ALIGN,
    BindingCache,
    KeyPassPolicy,
    backward_binding_key,
    bind_stage,
    default_softmax_scale,
    dsa_train_workspace_size,
    forward_binding_key,
    generated_program_available,
    grid_dims,
    key_pass_dq_mode,
    key_pass_ranges,
    offset_gather_kv_indices,
    plan_key_passes,
    prepare_dsa_train,
    record_abi,
    record_dkv_acc_layout,
    record_for,
    validate_dsa_train_inputs,
    workspace_layout,
)
from tests.test_helpers.cake_dsa_train_reference import (
    calibrate_beta,
    globalize_gather_indices_loop,
    make_inputs,
    reference_fp64,
    rel_l2,
    rel_l2_rows,
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


@contextlib.contextmanager
def _quiet_experimental():
    # The experimental banner fires once per process; the API's opt-in is
    # exercised by test_public_api_is_marked_experimental.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _require_contract_abi():
    _, record = record_for(torch.device("cuda"))
    if record_abi(record) != ABI_CONTRACT:
        pytest.skip("the placeholder forward-only program does not serve this case")


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    assert dsa_sparse_attention.is_experimental is True
    assert dsa_sparse_attention_varlen.is_experimental is True
    assert "SM100" in dsa_sparse_attention.__doc__


def test_registry_records_are_well_formed():
    assert cake_jit.STAGES[0] == "fwd"
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCH_NVCC_FLAGS
        assert record_abi(record) in SUPPORTED_ABIS
        stages = cake_jit.registered_stages(name)
        assert "fwd" in stages
        if "bwd_main" in stages:
            assert record_dkv_acc_layout(record) in DKV_ACC_LAYOUTS
        if any(stage in stages for stage in KEY_PASS_STAGES):
            # the key-range-pass form comes as a pair, next to the single-pass stage, with its host policy
            assert set(KEY_PASS_STAGES) <= set(stages) and "bwd_main" in stages
            policy = KeyPassPolicy.from_record(record)
            assert policy is not None
            assert policy.token_chunk(2048) % policy.token_chunk_multiple == 0
            assert policy.passes(1, 1, 2048) == 1
        else:
            assert plan_key_passes(record, stages, 4096, 65536, 2048) == 1
        for stage in stages:
            physical = record[stage]
            assert len(physical["sources"]) == 2
            assert all(s.startswith("cake_dsa_h64_train/") for s in physical["sources"])
            assert len(physical["closure_sha256"]) == 64
            assert all(
                kind in {"buffer", "tma_buffer", "workspace", "parameter", "grid"}
                for kind, _ in physical["arg_plan"]
            )
            launch = physical.get("launch")
            if launch is not None:
                assert len(launch["cluster"]) == 3 and all(
                    int(c) >= 1 for c in launch["cluster"]
                )
                assert len(launch["block"]) == 3 and all(
                    int(b) >= 1 for b in launch["block"]
                )
            assert len(physical.get("grid", ["num_queries", 1, 1])) == 3


@pytest.mark.parametrize("abi", SUPPORTED_ABIS)
@pytest.mark.parametrize("backward", [False, True])
def test_workspace_layout(abi, backward):
    layout = workspace_layout(
        300, 1000, 96, abi=abi, backward=backward, tma_workspace_bytes=1024
    )
    regions = [k for k in layout if k != "total"]
    offsets = [layout[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert layout["total"] >= sum(layout[k][1] for k in regions)
    assert ("delta" in layout) == backward
    assert ("dkv_latent_acc" in layout) == backward
    assert ("packed_q" in layout) == (abi == ABI_SEED)
    assert layout["tma_descriptor_workspace"][1] == 1024
    assert layout["topk_length"][1] == 300 * 4


def test_workspace_layout_key_pass_regions():
    T, S, topk = 300, 70000, 96
    kw = dict(abi=ABI_CONTRACT, backward=True, tma_workspace_bytes=896)
    single = workspace_layout(T, S, topk, **kw)
    multi = workspace_layout(T, S, topk, key_passes=3, **kw)
    pass_regions = {"dq_partial", "key_scratch", "pass_counts"}
    assert not pass_regions & set(single)
    assert pass_regions <= set(multi)
    assert multi["dq_partial"][1] == T * DQ_PARTIAL_BYTES_PER_TOKEN
    assert multi["key_scratch"][1] == T * topk * 4
    assert multi["pass_counts"][1] == T * 4
    regions = [k for k in multi if k != "total"]
    offsets = [multi[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert multi["total"] >= sum(multi[k][1] for k in regions)
    assert multi["total"] - single["total"] >= T * (
        DQ_PARTIAL_BYTES_PER_TOKEN + 4 * topk + 4
    )
    forward = workspace_layout(
        T,
        S,
        topk,
        abi=ABI_CONTRACT,
        backward=False,
        tma_workspace_bytes=896,
        key_passes=3,
    )
    assert not pass_regions & set(forward)


# The policy of the registered programs: one pass per 100 MiB of FP32 dK/dV accumulator
# (2304 B per key), taken only when the whole row's pass scratch fits 640 MiB.
_POLICY = dict(
    l2_budget_bytes=100 << 20,
    key_bytes=2304,
    workspace_budget_bytes=640 << 20,
    token_chunk_multiple=128,
)
_ALL_STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
_SINGLE_PASS_STAGES = ("fwd", "bwd_delta", "bwd_main", "bwd_cast")


def test_key_pass_policy_rule():
    policy = KeyPassPolicy(**_POLICY)
    assert policy.token_chunk(2048) == 4224
    assert [
        policy.formula_passes(s)
        for s in (4096, 45511, 45512, 65536, 91022, 91023, 131072)
    ] == [1, 1, 2, 2, 2, 3, 3]
    expected = {
        (4096, 65536): 2,
        (4096, 131072): 3,
        (512, 65536): 2,
        (4224, 131072): 3,
        (4225, 131072): 1,  # the whole row exceeds the pass workspace budget
        (32768, 131072): 1,
        (65536, 65536): 1,
        (4096, 45511): 1,
        (4096, 45512): 2,
    }
    assert {k: policy.passes(k[0], k[1], 2048) for k in expected} == expected
    # a packed multi-segment key row plans one pass: whole-row ranges do not match segment-confined index rows
    assert policy.passes(4096, 65536, 2048, num_segments=1) == 2
    assert policy.passes(4096, 65536, 2048, num_segments=2) == 1
    assert policy.passes(4096, 268757, 2048, num_segments=9) == 1
    with pytest.raises(ValueError, match="num_segments"):
        policy.passes(4096, 65536, 2048, num_segments=0)
    assert key_pass_ranges(65536, 2) == ((0, 32768), (32768, 65536))
    assert key_pass_ranges(131072, 3) == ((0, 43691), (43691, 87382), (87382, 131072))
    assert key_pass_ranges(10, 1) == ((0, 10),)
    assert key_pass_dq_mode(0, 1) == 0
    assert [key_pass_dq_mode(i, 3) for i in range(3)] == [1, 2, 3]
    assert [key_pass_dq_mode(i, 2) for i in range(2)] == [1, 3]
    assert KeyPassPolicy.from_record({"arch": "sm_100a"}) is None
    with pytest.raises(ValueError, match="key_bytes"):
        KeyPassPolicy.from_record(
            {"key_pass_policy": {k: v for k, v in _POLICY.items() if k != "key_bytes"}}
        )
    with pytest.raises(ValueError, match="positive"):
        KeyPassPolicy.from_record({"key_pass_policy": dict(_POLICY, key_bytes=0)})


def test_plan_key_passes_override_and_policy():
    record = {"key_pass_policy": dict(_POLICY)}
    assert plan_key_passes(record, _ALL_STAGES, 4096, 65536, 2048) == 2
    assert plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048) == 1
    # no registered policy, or no pass stages: the single-pass stage
    assert plan_key_passes({}, _ALL_STAGES, 4096, 65536, 2048) == 1
    assert plan_key_passes(record, _SINGLE_PASS_STAGES, 4096, 65536, 2048) == 1
    # a packed multi-segment row: one pass by policy, an explicit override still counts
    assert plan_key_passes(record, _ALL_STAGES, 4096, 65536, 2048, num_segments=2) == 1
    assert (
        plan_key_passes(
            record, _ALL_STAGES, 4096, 65536, 2048, key_passes=2, num_segments=2
        )
        == 2
    )
    # explicit override
    assert plan_key_passes(record, _ALL_STAGES, 4096, 65536, 2048, key_passes=1) == 1
    assert plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048, key_passes=5) == 5
    for bad in (0, -1, True, 2.5, 4097):
        with pytest.raises(ValueError):
            plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048, key_passes=bad)
    with pytest.raises(NotImplementedError, match="bwd_compact"):
        plan_key_passes(record, _SINGLE_PASS_STAGES, 4096, 65536, 2048, key_passes=2)


def test_grid_dims():
    scalars = dict(num_queries=130, num_kv=4096)
    assert grid_dims(["num_queries", 1, 1], scalars, 148) == (130, 1, 1)
    assert grid_dims(["num_queries/64", 1, 1], scalars, 148) == (3, 1, 1)
    assert grid_dims(["sms", 1, 1], scalars, 148) == (148, 1, 1)
    assert grid_dims(["sms*2", 1, 1], scalars, 148) == (296, 1, 1)
    assert grid_dims([4, 2, 1], scalars, 148) == (4, 2, 1)
    with pytest.raises(ValueError):
        grid_dims([1, 1], scalars, 148)
    assert grid_dims(
        ["num_queries*8", 1, 1], {"num_queries": 5, "num_kv": 7, "topk": 3}, 148
    ) == (40, 1, 1)
    assert grid_dims(
        ["num_kv*36/256", 1, 1], {"num_queries": 5, "num_kv": 7, "topk": 3}, 148
    ) == (1, 1, 1)
    assert grid_dims(
        ["num_kv*36/256", 1, 1], {"num_queries": 5, "num_kv": 4096, "topk": 3}, 148
    ) == (576, 1, 1)


def test_offset_gather_kv_indices_matches_loop():
    seq_q, seq_k, topk = [3, 2, 4], [5, 2, 6], 4
    cu_q = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 7, 13], dtype=torch.int32)
    local = torch.tensor(
        [
            [0, 1, -1, -1],
            [4, 0, 2, -1],
            [1, 5, -1, 3],  # 5 >= seq_k[0]: invalid
            [0, 1, 2, -1],  # doc 1: 2 >= seq_k[1] invalid
            [1, -1, -1, -1],
            [0, 5, 3, 6],  # doc 2: 6 >= seq_k[2] invalid
            [2, 2, -1, 1],
            [-1, -1, -1, -1],
            [5, 4, 3, 0],
        ],
        dtype=torch.int32,
    )
    expected = torch.full_like(local, -1)
    row = 0
    for d in range(3):
        for _ in range(seq_q[d]):
            for j in range(topk):
                v = int(local[row, j])
                if 0 <= v < seq_k[d]:
                    expected[row, j] = v + int(cu_k[d])
            row += 1
    # causal=False: plain offsetting (the causal term has its own tests in the packed / strided layout block)
    got = offset_gather_kv_indices(local, cu_q, cu_k, causal=False)
    assert got.dtype == torch.int32
    assert torch.equal(got, expected)
    out = torch.empty_like(local)
    assert offset_gather_kv_indices(local, cu_q, cu_k, causal=False, out=out) is out
    assert torch.equal(out, expected)


def _host_inputs(total_q=8, total_k=16, topk=5):
    q = torch.zeros(total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv = torch.zeros(total_k, D_QK, dtype=torch.bfloat16)
    idx = torch.zeros(total_q, topk, dtype=torch.int32)
    return q, kv, idx


def test_validate_accepts_packed_views():
    q, kv, idx = _host_inputs()
    t, s, k = validate_dsa_train_inputs(
        q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx
    )
    assert (t, s, k) == (8, 16, 5)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT].float(),
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx,
            ),
            "bfloat16",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx.long(),
            ),
            "int32",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx[:4],
            ),
            r"\[T, topk\]",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:-1, D_LATENT:],
                idx,
            ),
            "same number of rows",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT].t().contiguous().t(),
                kv[:, D_LATENT:],
                idx,
            ),
            "contiguous",
        ),
        (
            lambda q, kv, idx: (
                q[:, :32, :D_LATENT],
                q[:, :32, D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx,
            ),
            r"\[T, 64, 512\]",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:0, :D_LATENT],
                kv[:0, D_LATENT:],
                idx,
            ),
            r"S > 0",
        ),
        (
            lambda q, kv, idx: (
                q[:0, :, :D_LATENT],
                q[:0, :, D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx[:0],
            ),
            r"T == 0",
        ),
    ],
)
def test_validate_rejects(mutate, match):
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match=match):
        validate_dsa_train_inputs(*mutate(q, kv, idx))


def test_validate_zero_length_inputs():
    """``S == 0`` is always rejected; ``T == 0`` only where a launch would follow (the runner path)."""
    q, kv, idx = _host_inputs(total_q=0)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    with pytest.raises(ValueError, match="T == 0"):
        validate_dsa_train_inputs(ql, qr, kl, kr, idx)
    assert validate_dsa_train_inputs(ql, qr, kl, kr, idx, allow_empty_queries=True) == (
        0,
        16,
        5,
    )
    with pytest.raises(ValueError, match="S > 0"):
        validate_dsa_train_inputs(ql, qr, kl[:0], kr[:0], idx, allow_empty_queries=True)
    # the prepared runner (explicit path) names the condition before anything is allocated or bound
    with pytest.raises(ValueError, match="T == 0"):
        prepare_dsa_train(ql, qr, kl, kr, idx, backward=False)


def test_zero_query_rows_return_empty_outputs_without_binding(monkeypatch):
    """``T == 0``: the eager entry points and the autograd path return empty outputs and zero
    gradients without consulting the registry, loading a module, binding or launching."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "prepare_dsa_train", refuse)
    monkeypatch.setattr(cake_backend, "load_cake_dsa_train_module", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    q, kv, idx = _host_inputs(total_q=0, total_k=16, topk=5)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    cache = cake_backend.BINDING_CACHE
    hits, misses = cache.hits, cache.misses
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
    assert (cache.hits, cache.misses) == (hits, misses)  # never consulted


def test_pointer_alias_reports_the_storage_base_of_views_inside_a_storage():
    """A contiguous view that starts inside its storage (a slice of a larger buffer) is passed as the
    whole storage viewed flat plus its element offset, like a strided view; a tensor that starts
    its storage is passed as is.  A remembered binding re-reads the alias from the current tensors."""
    buf = torch.arange(1000, dtype=torch.int32)
    for offset in (1, 8, 13):
        view = buf[offset : offset + 8 * 96].view(8, 96)
        assert view.is_contiguous() and view.storage_offset() == offset
        storage, element_offset = cake_backend._pointer_alias(view)
        assert element_offset == offset
        assert storage.dim() == 1 and storage.is_contiguous()
        assert storage.dtype == torch.int32 and storage.numel() == buf.numel()
        assert storage.data_ptr() == buf.data_ptr() != view.data_ptr()
        assert torch.equal(storage[offset : offset + 8 * 96].view(8, 96), view)
    storage, element_offset = cake_backend._pointer_alias(buf)
    assert storage is buf and element_offset == 0
    head = buf[:100]  # starts its storage: passed as is
    storage, element_offset = cake_backend._pointer_alias(head)
    assert storage is head and element_offset == 0
    kv = torch.zeros(20 * D_QK + 3, dtype=torch.bfloat16)
    k_rope = kv[3 : 3 + 16 * D_ROPE].view(16, D_ROPE)  # contiguous, offset 3
    binding = cake_backend._Binding(
        num_queries=8,
        num_kv=16,
        topk=96,
        device=buf.device,
        device_index=0,
        dkv_fp32=False,
        launches={},
        owned={},
    )
    values = binding.rebind_values(
        dict(indices=buf[13 : 13 + 8 * 96].view(8, 96), k_rope=k_rope)
    )
    assert values["indices_storage"].data_ptr() == buf.data_ptr()
    assert values["indices_offset"] == 13
    assert values["k_rope_storage"].data_ptr() == kv.data_ptr()
    assert values["k_rope_offset"] == 3
    # the single-pass placeholders alias the same storage
    assert values["key_scratch"] is values["indices_storage"]


def test_validate_rejects_bad_topk_length():
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match="topk_length"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
            idx,
            torch.zeros(3, dtype=torch.int32),
        )


def test_bind_stage_fails_closed_on_unknown_argument(monkeypatch):
    record = {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["fwd"],
        "fwd": {
            "module": "fake",
            "sources": [],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "q_latent"],
                ["parameter", "not_a_host_value"],
                ["grid", "grid_x"],
            ],
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0" * 64,
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dsa_h64_train_fake", record)
    values = {"q_latent": torch.zeros(1), "softmax_scale": 1.0}
    with pytest.raises(KeyError, match="not_a_host_value"):
        bind_stage("cake_dsa_h64_train_fake", "fwd", values, (1, 1, 1))


def test_bind_stage_records_rebind_slots(monkeypatch):
    record = {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["fwd"],
        "fwd": {
            "module": "fake",
            "sources": [],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "q_latent"],
                ["buffer", "k_rope"],
                ["parameter", "k_rope_offset"],
                ["parameter", "num_queries"],
                ["buffer", "lengths"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
            ],
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0" * 64,
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dsa_h64_train_fake", record)
    calls = []
    fake = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(
        cake_backend, "load_cake_dsa_train_module", lambda name, stage: fake
    )
    kv = torch.zeros(4, D_QK, dtype=torch.bfloat16)
    k_rope = kv[:, D_LATENT:]
    storage, offset = cake_backend._pointer_alias(k_rope)
    values = dict(
        q_latent=torch.zeros(2, NUM_HEADS, D_LATENT, dtype=torch.bfloat16),
        k_rope=k_rope,
        k_rope_storage=storage,
        k_rope_offset=offset,
        num_queries=2,
        topk_length=torch.zeros(2, dtype=torch.int32),
        softmax_scale_log2=0.5,
    )
    launch = bind_stage("cake_dsa_h64_train_fake", "fwd", values, (2, 1, 1))
    # the raw-pointer operand is bound through its storage alias; scalars derived from metadata are baked in
    assert launch.slots == (
        (0, "q_latent"),
        (1, "k_rope_storage"),
        (2, "k_rope_offset"),
        (4, "topk_length"),
    )
    assert launch.arguments[1] is storage and launch.arguments[2] == offset == D_LATENT
    assert launch.arguments[3:4] == (2,) and launch.arguments[5:] == (0.5, 2)
    template = launch.templated()
    assert [template.arguments[i] for i, _ in launch.slots] == [None] * 4
    assert not any(isinstance(a, torch.Tensor) for a in template.arguments)
    kv2 = torch.zeros(
        6, D_QK, dtype=torch.bfloat16
    )  # another packed buffer: alias and offset re-read
    storage2, offset2 = cake_backend._pointer_alias(kv2[:, D_LATENT:])
    rebound = template.arguments_for(
        dict(
            q_latent=values["q_latent"],
            k_rope_storage=storage2,
            k_rope_offset=offset2,
            topk_length=values["topk_length"],
        )
    )
    assert rebound[1] is storage2 and rebound[2] == offset2 and rebound[5:] == [0.5, 2]
    with pytest.raises(RuntimeError, match="topk_length"):
        template.arguments_for(
            dict(
                q_latent=values["q_latent"],
                k_rope_storage=storage2,
                k_rope_offset=offset2,
            )
        )


def test_bind_stage_serves_permuted_cast_plan(monkeypatch):
    """The cast of a program with permuted FP32 accumulators names its own operands
    (``src_*`` / ``dst_*`` / ``dst_*_f32`` / ``*_groups`` / ``out_f32``); the contract
    profile provides every one of them, and a backward record must declare its
    accumulator layout."""
    record = {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["fwd", "bwd_delta", "bwd_main", "bwd_cast"],
        "dkv_acc_layout": "permuted",
        "bwd_cast": {
            "module": "fake",
            "sources": [],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "src_latent"],
                ["buffer", "src_rope"],
                ["buffer", "dst_latent"],
                ["buffer", "dst_rope"],
                ["buffer", "dst_latent_f32"],
                ["buffer", "dst_rope_f32"],
                ["parameter", "latent_groups"],
                ["parameter", "rope_groups"],
                ["parameter", "out_f32"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
        },
        "bwd_main": {
            "module": "fake",
            "ffi_entry": "run",
            "arg_plan": [
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
            ],
        },
        "closure_sha256": "0" * 64,
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dsa_h64_train_fake", record)
    monkeypatch.setattr(
        cake_backend,
        "load_cake_dsa_train_module",
        lambda name, stage: SimpleNamespace(run=lambda *a: None),
    )
    S = 8
    t = dict(
        q_latent=torch.zeros(2, NUM_HEADS, D_LATENT, dtype=torch.bfloat16),
        q_rope=torch.zeros(2, NUM_HEADS, D_ROPE, dtype=torch.bfloat16),
        kv_latent=torch.zeros(S, D_LATENT, dtype=torch.bfloat16),
        k_rope=torch.zeros(S, D_ROPE, dtype=torch.bfloat16),
        indices=torch.zeros(2, 4, dtype=torch.int32),
        delta=torch.zeros(2, NUM_HEADS),
        dkv_latent_acc=torch.zeros(S, D_LATENT),
        dk_rope_acc=torch.zeros(S, D_ROPE),
        dkv_latent=torch.empty(0, dtype=torch.bfloat16),
        dk_rope=torch.empty(0, dtype=torch.bfloat16),
        dkv_latent_fp32=torch.zeros(S, D_LATENT),
        dk_rope_fp32=torch.zeros(S, D_ROPE),
    )
    scalars = dict(
        num_queries=2, num_kv=S, topk=4, softmax_scale=1.0, has_topk_length=0, out_f32=1
    )
    values = cake_backend._contract_values(t, scalars)
    assert (values["token_base"], values["token_step"]) == (0, 1)
    assert (values["latent_groups"], values["rope_groups"], values["out_f32"]) == (
        S * 16,
        S * 2,
        1,
    )
    launch = bind_stage("cake_dsa_h64_train_fake", "bwd_cast", values, (3, 1, 1))
    assert (
        launch.arguments[0] is t["dkv_latent_acc"]
        and launch.arguments[4] is t["dkv_latent_fp32"]
    )
    assert launch.arguments[2] is t["dkv_latent"] and launch.arguments[6:9] == (
        S * 16,
        S * 2,
        1,
    )
    # the FP32 outputs are re-bindable slots (fresh per call in the dkv_fp32 mode)
    assert {key for _, key in launch.slots} >= {
        "dkv_latent_acc",
        "dkv_latent_fp32",
        "dk_rope_fp32",
        "dkv_latent",
        "dk_rope",
    }
    main = bind_stage("cake_dsa_h64_train_fake", "bwd_main", values, (2, 1, 1))
    # key-range-pass operands of the single-pass program: whole key range, dq_mode 0, inert placeholders as in the
    # production launcher (delta for the dQ partials, the indices storage for the compacted keys and counts)
    assert main.arguments[:5] == (0, 1, 0, S, 0)
    assert (
        main.arguments[5] is t["delta"]
        and main.arguments[6] is t["indices"]
        and main.arguments[7] is t["indices"]
    )
    assert {key for _, key in main.slots} == {
        "dq_partial",
        "key_scratch",
        "pass_counts",
    }
    rebound = main.templated().arguments_for(
        cake_backend._inert_pass_values(
            dict(delta=t["delta"], indices_storage=t["indices"]), S
        )
    )
    assert rebound[5] is t["delta"] and rebound[6] is t["indices"]
    assert record_dkv_acc_layout(record) == "permuted"
    with pytest.raises(ValueError, match="dkv_acc_layout"):
        record_dkv_acc_layout({"arch": "sm_100a", "stages": ["fwd", "bwd_main"]})


def test_bind_stage_serves_key_pass_plans(monkeypatch):
    """The compaction and the pass form of the main stage draw every operand from the
    contract profile: the pass regions carved from the workspace, ``num_tokens`` = the
    whole row, the raw indices storage with its element offset, and the key range /
    ``dq_mode`` of each pass baked into its launch (a remembered binding re-supplies
    only the tensors)."""
    record = {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": list(_ALL_STAGES),
        "dkv_acc_layout": "permuted",
        "key_pass_policy": dict(_POLICY),
        "bwd_compact": {
            "module": "fake",
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "indices"],
                ["buffer", "topk_length"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["parameter", "num_tokens"],
                ["parameter", "topk"],
                ["parameter", "idx_stride"],
                ["parameter", "indices_offset"],
                ["parameter", "has_topk_length"],
                ["parameter", "token_base"],
                ["parameter", "token_step"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["grid", "grid_x"],
            ],
        },
        "bwd_main_pass": {
            "module": "fake",
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "delta"],
                ["buffer", "indices"],
                ["parameter", "num_queries"],
                ["parameter", "num_kv"],
                ["parameter", "pass_lo"],
                ["parameter", "pass_hi"],
                ["parameter", "dq_mode"],
                ["buffer", "dq_partial"],
                ["buffer", "key_scratch"],
                ["buffer", "pass_counts"],
                ["grid", "grid_x"],
            ],
        },
        "closure_sha256": "0" * 64,
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dsa_h64_train_fake", record)
    monkeypatch.setattr(
        cake_backend,
        "load_cake_dsa_train_module",
        lambda name, stage: SimpleNamespace(run=lambda *a: None),
    )
    T, S, topk = 6, 16, 4
    packed = torch.zeros(T, 2 * topk, dtype=torch.int32)
    indices = packed[
        :, topk:
    ]  # a strided view: bound as its storage alias + element offset
    t = dict(
        q_latent=torch.zeros(T, NUM_HEADS, D_LATENT, dtype=torch.bfloat16),
        q_rope=torch.zeros(T, NUM_HEADS, D_ROPE, dtype=torch.bfloat16),
        kv_latent=torch.zeros(S, D_LATENT, dtype=torch.bfloat16),
        k_rope=torch.zeros(S, D_ROPE, dtype=torch.bfloat16),
        indices=indices,
        topk_length=torch.full((T,), topk, dtype=torch.int32),
        delta=torch.zeros(T, NUM_HEADS),
        dq_partial=torch.zeros(T, DQ_PARTIAL_BYTES_PER_TOKEN // 4),
        key_scratch=torch.zeros(T, topk, dtype=torch.int32),
        pass_counts=torch.zeros(T, dtype=torch.int32),
    )
    scalars = dict(
        num_queries=T, num_kv=S, topk=topk, softmax_scale=1.0, has_topk_length=0
    )
    values = cake_backend._contract_values(t, scalars, key_passes=3)
    assert values["num_tokens"] == T and values["idx_stride"] == 2 * topk
    # the strided view is passed as its whole storage viewed flat plus the element offset
    storage = values["indices_storage"]
    assert storage.dim() == 1 and storage.numel() == packed.numel()
    assert storage.data_ptr() == packed.data_ptr() and values["indices_offset"] == topk
    # the pass regions are the carved tensors, not the single-pass placeholders
    assert values["dq_partial"] is t["dq_partial"]
    assert values["key_scratch"] is t["key_scratch"]
    assert "pass_lo" not in values and "dq_mode" not in values
    single = cake_backend._contract_values(t, scalars)
    assert single["dq_partial"] is t["delta"]
    assert single["key_scratch"] is single["indices_storage"]
    assert single["indices_storage"].data_ptr() == packed.data_ptr()
    assert (single["pass_lo"], single["pass_hi"], single["dq_mode"]) == (0, S, 0)
    ranges = key_pass_ranges(S, 3)
    assert ranges == ((0, 6), (6, 12), (12, 16))
    compact = bind_stage(
        "cake_dsa_h64_train_fake",
        "bwd_compact",
        cake_backend._pass_values(values, 1, 3, ranges[1]),
        (2, 1, 1),
    )
    assert compact.arguments[0] is storage
    assert compact.arguments[1] is t["topk_length"]
    assert compact.arguments[2] is t["key_scratch"]
    assert compact.arguments[3] is t["pass_counts"]
    assert compact.arguments[4:] == (T, topk, 2 * topk, topk, 0, 0, 1, 6, 12, 2)
    assert {key for _, key in compact.slots} == {
        "indices_storage",
        "indices_offset",
        "topk_length",
        "key_scratch",
        "pass_counts",
    }
    for index, (lo, hi) in enumerate(ranges):
        main = bind_stage(
            "cake_dsa_h64_train_fake",
            "bwd_main_pass",
            cake_backend._pass_values(values, index, 3, (lo, hi)),
            (T, 1, 1),
        )
        assert main.arguments[1] is storage and main.arguments[2:4] == (T, S)
        assert main.arguments[4:7] == (lo, hi, key_pass_dq_mode(index, 3))
        assert main.arguments[7] is t["dq_partial"]
        assert main.arguments[9] is t["pass_counts"]
    assert [m for m in (1, 2, 3)] == [key_pass_dq_mode(i, 3) for i in range(3)]
    # a remembered pass launch keeps its key range and dq_mode; the tensors are re-supplied
    template = main.templated()
    assert template.arguments[4:7] == (12, 16, 3) and template.arguments[7] is None
    rebound = template.arguments_for(
        dict(
            delta=t["delta"],
            indices_storage=packed,
            dq_partial=t["dq_partial"],
            key_scratch=t["key_scratch"],
            pass_counts=t["pass_counts"],
        )
    )
    assert rebound[7] is t["dq_partial"] and rebound[4:7] == [12, 16, 3]


def test_binding_keys_cover_pointer_shape_stride_dtype_scale_and_lengths():
    q, kv, idx = _host_inputs()
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    scale = default_softmax_scale()
    base = forward_binding_key(ql, qr, kl, kr, idx, None, scale)
    assert base == forward_binding_key(
        ql.view_as(ql), qr, kl, kr, idx, None, scale
    )  # another object, same binding
    assert base != forward_binding_key(ql, qr, kl, kr, idx, None, scale * 0.5)  # scale
    assert base != forward_binding_key(
        ql, qr, kl, kr, idx, torch.zeros(8, dtype=torch.int32), scale
    )  # lengths
    assert base != forward_binding_key(
        ql, qr, kl, kr, idx.clone(), None, scale
    )  # pointer
    assert base != forward_binding_key(
        ql, qr, kl[:4], kr, idx, None, scale
    )  # shape (same pointer, stride, dtype)
    same_ptr_other_stride = kv.view(-1)[: 16 * D_LATENT].view(16, D_LATENT)
    assert (
        same_ptr_other_stride.data_ptr() == kl.data_ptr()
        and same_ptr_other_stride.shape == kl.shape
    )
    assert base != forward_binding_key(
        ql, qr, same_ptr_other_stride, kr, idx, None, scale
    )  # stride
    assert base != forward_binding_key(
        ql, qr, kl.view(torch.float16), kr, idx, None, scale
    )  # dtype
    out = torch.zeros(8, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(8, NUM_HEADS)
    bwd = backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, False
    )
    assert bwd != base and bwd[0] == "bwd"
    assert bwd != backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, True
    )  # dkv_fp32
    assert bwd != backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out.clone(), None, scale, False
    )  # dout


def test_binding_cache_is_least_recently_used_and_bounded_by_capacity():
    """A hit refreshes its binding; beyond the capacity the least recently used binding goes,
    so a model whose layers cycle through up to ``capacity`` bindings per step binds each once."""
    cache = BindingCache(capacity=3)
    assert cache.enabled and len(cache) == 0

    def keys():
        return list(cache._bindings)

    for tag in ("a", "b", "c"):
        cache.remember(("fwd", tag), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "a"), ("fwd", "b"), ("fwd", "c")]
    assert cache.owned_bytes == 30
    # a hit moves the binding to the most recently used end; a miss changes nothing
    assert cache.lookup(("fwd", "a")) is not None
    assert keys() == [("fwd", "b"), ("fwd", "c"), ("fwd", "a")]
    assert cache.lookup(("fwd", "z")) is None and (cache.hits, cache.misses) == (1, 1)
    # beyond the capacity the least recently used binding goes (b, not the refreshed a)
    cache.remember(("bwd", "d"), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "c"), ("fwd", "a"), ("bwd", "d")]
    # re-remembering a key makes it the most recently used without duplicating it
    cache.remember(("fwd", "c"), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "a"), ("bwd", "d"), ("fwd", "c")] and len(cache) == 3
    # peek neither counts nor refreshes
    assert cache.peek(("fwd", "a")) is not None and keys()[0] == ("fwd", "a")
    assert (cache.hits, cache.misses) == (1, 1)
    cache.clear()
    assert len(cache) == 0 and cache.owned_bytes == 0
    # a cyclic pattern of forward + backward bindings of four layers (eight keys) in an
    # eight-binding cache: one miss per binding in the first step, hits from the second step on
    layer_keys = [(kind, layer) for layer in range(4) for kind in ("fwd", "bwd")]

    def run_steps(capacity, steps=3):
        cache = BindingCache(capacity=capacity)
        for _ in range(steps):
            for key in layer_keys:
                if cache.lookup(key) is None:
                    cache.remember(key, SimpleNamespace(owned_bytes=1))
        return cache.hits, cache.misses, len(cache)

    assert run_steps(8) == (16, 8, 8)
    assert run_steps(256) == (16, 8, 8)
    # one binding more than the capacity thrashes under a cyclic pattern (the boundary)
    assert run_steps(7) == (0, 24, 7)
    with pytest.raises(ValueError, match="capacity"):
        BindingCache(capacity=0)


def test_binding_cache_capacity_from_environment(monkeypatch):
    env = cake_backend.BINDING_CACHE_CAPACITY_ENV
    monkeypatch.delenv(env, raising=False)
    assert BindingCache().capacity == cake_backend.BINDING_CACHE_DEFAULT_CAPACITY == 256
    assert cake_backend.BINDING_CACHE.capacity >= 1
    monkeypatch.setenv(env, "5")
    assert cake_backend.binding_cache_capacity() == 5 and BindingCache().capacity == 5
    assert BindingCache(capacity=3).capacity == 3  # an explicit capacity wins
    monkeypatch.setenv(env, " ")
    assert BindingCache().capacity == 256
    for bad in ("0", "-1", "many", "2.5"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match="positive integer"):
            BindingCache()


# ---------------------------------------------------------------------------
# Device tests (compute capability 10.0 / 10.3 with a registered program)
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("topk", [128, 200])
def test_forward_iid(topk):
    _require_program()
    if topk % 64:
        _require_contract_abi()
    inp = make_inputs([384], [1024], seed=SEED, topk=topk)
    out, lse, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    _check_forward(inp, out, lse, ref)


def test_forward_accepts_packed_views_bitwise():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 1, topk=128)
    q = torch.cat([inp.q_latent, inp.q_rope], dim=-1).contiguous()
    kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1).contiguous()
    out_split, lse_split, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    out_view, lse_view, _ = cake_backend.forward(
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        inp.idx_global,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_split, out_view)
    assert torch.equal(lse_split, lse_view)


def test_forward_masked_rows_and_out_of_range():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 2, topk=128)
    idx = inp.idx_global.clone()
    idx[5] = -1  # fully masked row
    idx[17, ::3] = inp.total_k + 7  # out-of-range slots, anywhere in the row
    idx[33, :64] = -1  # invalid slots first, valid ones after
    topk_length = inp.topk_length.clone()
    topk_length[40] = 0  # masked through topk_length
    topk_length[41] = 3
    out, lse, _ = cake_backend.forward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        topk_length=topk_length,
    )
    _check_forward(inp, out, lse, ref)
    assert torch.all(out[5] == 0) and torch.all(lse[5] == float("-inf"))
    assert torch.all(out[40] == 0) and torch.all(lse[40] == float("-inf"))


def test_forward_deterministic():
    _require_program()
    inp = make_inputs([320], [640], seed=SEED + 3, topk=128)
    a = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    b = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    if a[2] is not None:
        assert torch.equal(a[2], b[2])


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


def test_runner_launches_without_allocation():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 5, topk=128)
    runner = prepare_dsa_train(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        backward=False,
    )
    runner.forward()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.forward()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]


@pytest.mark.parametrize(
    "seq_q, seq_k, topk, canonical", [(384, 1024, 200, False), (256, 4096, 2048, True)]
)
def test_backward_iid(seq_q, seq_k, topk, canonical):
    _require_program(backward=True)
    inp = make_inputs([seq_q], [seq_k], seed=SEED + 6, topk=topk)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    grads = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_forward(inp, out, lse, ref)
    _check_backward(grads, ref, canonical=canonical)


def test_backward_masked_rows_give_zero_dq():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 7, topk=128)
    idx = inp.idx_global.clone()
    idx[3] = -1
    idx[9, ::2] = -1
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx
    )
    dq_latent, dq_rope, dkv_latent, dk_rope = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.all(dq_latent[3] == 0) and torch.all(dq_rope[3] == 0)
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, dout=inp.dout
    )
    _check_backward((dq_latent, dq_rope, dkv_latent, dk_rope), ref)


def test_backward_peaked():
    """Own key carrying ~99 % of the softmax mass (the strongest calibrated peaked case).

    A saturated softmax (self weight -> 1) makes the exact dQ vanish and the relative error
    meaningless, so beta is calibrated to the target weight at this shape instead of fixed.
    """
    _require_program(backward=True)
    beta = calibrate_beta(0.99, seed=SEED + 8, probe_len=512, topk=128)
    inp = make_inputs(
        [512], [512], seed=SEED + 8, topk=128, self_including=True, beta=beta
    )
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        own_key=inp.own_key,
    )
    assert 0.9 < ref["self_weight"] < 0.999, (
        f"calibrated self weight {ref['self_weight']:.4f} outside the peaked window"
    )
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    grads = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    _check_backward(grads, ref, peaked=True)
    p99 = rel_l2_rows(grads[0], ref["dq_latent"]).quantile(0.99).item()
    assert p99 <= GATE_ROW_P99_DQ


def test_backward_deterministic_dq_and_dkv_spread():
    _require_program(backward=True)
    inp = make_inputs([320], [640], seed=SEED + 9, topk=128)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    a = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    b = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    spread = max(rel_l2(a[2], b[2]), rel_l2(a[3], b[3]))
    assert spread < 1e-2, f"dkv run-to-run spread {spread}"


def _require_key_pass_program():
    """The registered backward program with the key-range-pass stages (skip otherwise)."""
    _require_program(backward=True)
    name, record = record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages(name)
    if not set(KEY_PASS_STAGES) <= set(stages):
        pytest.skip("the registered program has no key-range-pass stages")
    return record, stages


def _masked_indices(inp):
    """Invalid slots anywhere in the row (-1 and out-of-range interleaved, a run in the
    middle), fully masked rows, and rows whose valid set ``topk_length`` cuts."""
    S = inp.kv_latent.shape[0]
    idx = inp.idx_global.clone()
    idx[3::11] = -1
    idx[4::11, ::7] = S + 5
    idx[5::11, 100:164] = -1
    topk_length = inp.topk_length.clone()
    topk_length[6::11] = 0
    topk_length[7::11] = 65
    return idx, topk_length


def test_backward_forced_key_passes_masked_matches_reference_and_is_deterministic():
    """Three forced key-range passes over 65,536 keys (T = 256) with the masking cases:
    the plan, the workspace regions, the launch order, the reference gates, bitwise dq
    across the validating and the remembered path, and agreement with the single pass."""
    _require_key_pass_program()
    inp = make_inputs([256], [65536], seed=SEED + 16, topk=2048)
    idx, topk_length = _masked_indices(inp)
    fwd_args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx)
    out, lse, o_lo = cake_backend.forward(*fwd_args, topk_length=topk_length)
    args = fwd_args + (out, o_lo, lse, inp.dout)
    runner = prepare_dsa_train(
        *fwd_args, topk_length=topk_length, dout=inp.dout, backward=True, key_passes=3
    )
    assert runner.key_passes == 3
    expected_order = ["bwd_delta"]
    for index in range(3):
        expected_order += [(stage, index) for stage in KEY_PASS_STAGES]
    if "bwd_cast" in runner.launches:
        expected_order.append("bwd_cast")
    assert list(runner.backward_order) == expected_order
    assert "bwd_main" not in runner.launches
    assert runner.tensors["dq_partial"].shape == (256, DQ_PARTIAL_BYTES_PER_TOKEN // 4)
    assert runner.tensors["key_scratch"].shape == (256, 2048)
    assert runner.tensors["pass_counts"].shape == (256,)
    assert runner.layout["dq_partial"][1] == 256 * DQ_PARTIAL_BYTES_PER_TOKEN
    assert runner.workspace.numel() == dsa_train_workspace_size(
        256, 65536, 2048, inp.q_latent.device, key_passes=3
    )
    with _cache(True) as cache:
        cache.clear()
        a = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
        b = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
        single = cake_backend.backward(*args, topk_length=topk_length, key_passes=1)
        torch.cuda.synchronize()
        assert cache.misses >= 2 and cache.hits >= 1
        binding = cache.peek(
            backward_binding_key(*args, topk_length, default_softmax_scale(), False, 3)
        )
    assert binding is not None and binding.key_passes == 3
    assert binding.holds_no_tensor()
    assert ("bwd_main_pass", 2) in binding.launches
    # the pass regions, delta and the FP32 accumulators are per-call scratch: the binding owns kilobytes
    assert not {
        "delta",
        "dkv_latent_acc",
        "dk_rope_acc",
        "dq_partial",
        "key_scratch",
        "pass_counts",
    } & set(binding.owned)
    assert set(binding.owned) <= {"topk_length", "tma_descriptor_workspace"}
    assert binding.owned_bytes < (1 << 20)
    # dq is written once per row from the carried FP32 partial: bitwise across runs and paths
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for grads in (
        a,
        single,
    ):  # fully masked rows (by -1 and by topk_length 0) give zero dq
        assert torch.all(grads[0][3::11] == 0) and torch.all(grads[0][6::11] == 0)
    ref = reference_fp64(*fwd_args, dout=inp.dout, topk_length=topk_length)
    _check_backward(a, ref)
    _check_backward(single, ref)
    # the passes re-associate the FP32 dq partial sums (agreement well inside the BF16 output);
    # the dK/dV reductions are the same reds in another order
    assert rel_l2(a[0], single[0]) < 1e-3 and rel_l2(a[1], single[1]) < 1e-3
    assert max(rel_l2(a[2], single[2]), rel_l2(a[3], single[3])) < 1e-2


def test_backward_whole_row_policy_two_passes_through_public_entry():
    """The registered policy takes two passes at T = 4096 x S = 65,536 (top-k 2048), through
    the public entry, against the canonical gates; the 4k x 4k and 32k-token rows stay single-pass."""
    record, stages = _require_key_pass_program()
    device = torch.device("cuda")
    assert plan_key_passes(record, stages, 4096, 65536, 2048) == 2
    assert plan_key_passes(record, stages, 4096, 131072, 2048) == 3
    assert plan_key_passes(record, stages, 4096, 4096, 2048) == 1
    assert plan_key_passes(record, stages, 32768, 131072, 2048) == 1
    policy_size = dsa_train_workspace_size(4096, 65536, 2048, device)
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
    with _cache(True) as cache:
        cache.clear()
        with _quiet_experimental():
            out = dsa_sparse_attention(*leaves, inp.idx_global)
        grads = torch.autograd.grad(out, leaves, inp.dout)
        torch.cuda.synchronize()
        remembered = [b for b in cache._bindings.values() if b.backward_order]
        assert len(remembered) == 1 and remembered[0].key_passes == 2
        assert [k for k in remembered[0].backward_order if isinstance(k, tuple)] == [
            ("bwd_compact", 0),
            ("bwd_main_pass", 0),
            ("bwd_compact", 1),
            ("bwd_main_pass", 1),
        ]
        # the two-pass backward's ~790 MB of scratch (dq_partial, key_scratch, the FP32 accumulators) is
        # per-call allocator memory; the remembered binding itself owns kilobytes
        assert remembered[0].owned_bytes < (1 << 20)
        assert cache.owned_bytes < len(cache) << 20
        # steady state: repeating the step with the same binding re-binds nothing
        fwd_args = tuple(t.detach() for t in leaves) + (inp.idx_global,)
        o1, l1, olo1 = cake_backend.forward(
            *fwd_args
        )  # the autograd step's forward binding: a hit
        g1 = cake_backend.backward(
            *fwd_args, o1, olo1, l1, inp.dout
        )  # fresh saved outputs: binds anew
        hits, misses, size = cache.hits, cache.misses, len(cache)
        g2 = cake_backend.backward(*fwd_args, o1, olo1, l1, inp.dout)
        o2, l2, olo2 = cake_backend.forward(*fwd_args)
        torch.cuda.synchronize()
        assert (cache.hits, cache.misses, len(cache)) == (hits + 2, misses, size)
        assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])
        assert torch.equal(g1[0], grads[0]) and torch.equal(o2, out.detach())
        misses = cache.misses
        with _quiet_experimental():
            out_single = dsa_sparse_attention(*leaves, inp.idx_global, key_passes=1)
        single = torch.autograd.grad(out_single, leaves, inp.dout)
        torch.cuda.synchronize()
        # the override is part of the binding key: the backward binds anew
        assert cache.misses > misses
        assert 1 in [b.key_passes for b in cache._bindings.values() if b.backward_order]
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
    with _cache(True) as cache:
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
        remembered = [b for b in cache._bindings.values() if b.backward_order]
        assert [b.key_passes for b in remembered] == [1]
        assert not any(isinstance(k, tuple) for k in remembered[0].backward_order)
        with _quiet_experimental():
            out_f, lse_f = dsa_sparse_attention(
                *flat_leaves, inp.idx_global, return_lse=True
            )
        grads_f = torch.autograd.grad(out_f, flat_leaves, inp.dout)
        torch.cuda.synchronize()
        # the segment count is part of the binding key: the flat call binds anew and plans the two passes
        assert sorted(
            b.key_passes for b in cache._bindings.values() if b.backward_order
        ) == [1, 2]
    assert torch.equal(out_v.detach(), out_f.detach()) and torch.equal(lse_v, lse_f)
    for a, b in zip(grads_v, grads_f, strict=True):
        assert rel_l2(a, b) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_forward(inp, out_v, lse_v, ref)
    _check_backward(grads_v, ref)
    _check_backward(grads_f, ref)


def _indices_view_inside_storage(indices: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``indices`` that starts ``offset`` elements inside a larger buffer."""
    n = indices.numel()
    buf = torch.full((n + offset + 64,), -1, dtype=indices.dtype, device=indices.device)
    view = buf[offset : offset + n].view(indices.shape)
    view.copy_(indices)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


@pytest.mark.parametrize("offset", [1, 8, 13])
def test_indices_view_inside_a_storage_matches_the_plain_tensor(offset):
    """A contiguous ``indices`` view that starts inside its storage (a slice of a larger buffer)
    reaches the kernels as the storage base plus the element offset, so their aligned index-tile
    loads (gated on the offset, not the pointer) stay valid: forward, backward and the autograd
    path agree with the plain tensor (bitwise where the kernels are deterministic)."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 18, topk=128)
    view = _indices_view_inside_storage(inp.idx_global, offset)
    storage, element_offset = cake_backend._pointer_alias(view)
    assert element_offset == offset
    assert storage.data_ptr() == view.data_ptr() - 4 * offset
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


@pytest.mark.parametrize("offset", [1, 4, 8, 13])
def test_key_pass_compaction_accepts_an_indices_view_inside_a_storage(offset):
    """The compaction stage's 8-wide index loads (whole 256-slot blocks, so top-k 256) see the
    storage base plus the element offset: two forced key-range passes with a view inside a
    larger buffer match the plain tensor and the reference."""
    _require_key_pass_program()
    inp = make_inputs([128], [1024], seed=SEED + 19, topk=256)
    view = _indices_view_inside_storage(inp.idx_global, offset)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    out, lse, o_lo = cake_backend.forward(*args, inp.idx_global)
    plain = cake_backend.backward(
        *args, inp.idx_global, out, o_lo, lse, inp.dout, key_passes=2
    )
    got = cake_backend.backward(*args, view, out, o_lo, lse, inp.dout, key_passes=2)
    torch.cuda.synchronize()
    assert torch.equal(plain[0], got[0]) and torch.equal(plain[1], got[1])
    assert max(rel_l2(plain[2], got[2]), rel_l2(plain[3], got[3])) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_backward(got, ref)


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


def test_backward_dkv_fp32_returns_natural_layout_gradients():
    """``dkv_fp32=True`` yields natural-layout FP32 dK/dV: equal to the BF16 path within BF16 rounding
    plus the ``red.global`` run-to-run spread, and never the kernels' internal accumulators."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 11, topk=128)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    args = (
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    dq_l, dq_r, dkv_latent, dk_rope = cake_backend.backward(*args, dkv_fp32=True)
    bq_l, bq_r, dkv_bf16, dkr_bf16 = cake_backend.backward(*args)
    torch.cuda.synchronize()
    S = inp.kv_latent.shape[0]
    assert dkv_latent.dtype == torch.float32 and dk_rope.dtype == torch.float32
    assert tuple(dkv_latent.shape) == (S, D_LATENT) and tuple(dk_rope.shape) == (
        S,
        D_ROPE,
    )
    assert dkv_latent.is_contiguous() and dk_rope.is_contiguous()
    assert torch.equal(dq_l, bq_l) and torch.equal(
        dq_r, bq_r
    )  # dq does not depend on the dkv mode
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    # the FP32 gradients are at least as close to the reference as their BF16 casts' floor ...
    assert (
        rel_l2(dkv_latent, ref["dkv_latent"])
        <= FLOOR_MARGIN * rel_l2(ref["dkv_latent_emu"], ref["dkv_latent"]) + FLOOR_ABS
    )
    assert (
        rel_l2(dk_rope, ref["dk_rope"])
        <= FLOOR_MARGIN * rel_l2(ref["dk_rope_emu"], ref["dk_rope"]) + FLOOR_ABS
    )
    # ... and agree with the BF16 path element by element (a permuted accumulator layout would not)
    assert (
        rel_l2(dkv_latent.to(torch.bfloat16), dkv_bf16) < 1e-2
        and rel_l2(dk_rope.to(torch.bfloat16), dkr_bf16) < 1e-2
    )
    _, record = record_for(torch.device("cuda"))
    if record_dkv_acc_layout(record) == "permuted":
        runner = prepare_dsa_train(
            *args[:5], dout=inp.dout, backward=True, dkv_fp32=True
        )
        runner.forward()
        _, _, f_lat, f_rope = runner.backward()
        torch.cuda.synchronize()
        assert f_lat.data_ptr() != runner.tensors["dkv_latent_acc"].data_ptr()
        assert f_rope.data_ptr() != runner.tensors["dk_rope_acc"].data_ptr()
        assert "bwd_cast" in runner.launches and rel_l2(f_lat, dkv_latent) < 1e-2


# ---------------------------------------------------------------------------
# Binding cache of the eager entry points (device)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _cache(enabled: bool):
    cache = cake_backend.BINDING_CACHE
    was = cache.enabled
    cache.enabled = enabled
    try:
        yield cache
    finally:
        cache.enabled = was


def test_binding_cache_forward_hits_are_bitwise_and_pin_nothing():
    _require_program()
    _require_contract_abi()
    scale = default_softmax_scale()
    inp = make_inputs([128, 128], [256, 384], seed=SEED + 12, topk=96)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    with _cache(True) as cache:
        cache.clear()
        hits0, misses0 = cache.hits, cache.misses
        first = cake_backend.forward(*args)  # validating path, remembers the binding
        second = cake_backend.forward(*args)  # remembered binding
        assert (cache.misses, cache.hits) == (misses0 + 1, hits0 + 1)
        with _cache(False):
            fresh = cake_backend.forward(*args)
        torch.cuda.synchronize()
        for a, b, c in zip(first, second, fresh, strict=True):
            assert torch.equal(a, b) and torch.equal(b, c)
        assert (
            second[0].data_ptr() != first[0].data_ptr()
        )  # outputs are fresh allocations
        binding = cache.peek(forward_binding_key(*args, None, scale))
        assert binding is not None and binding.holds_no_tensor()
        assert set(binding.owned) <= {"topk_length", "tma_descriptor_workspace"}
        # caller-provided lengths: another binding (misses once), then hits
        lengths = torch.full((256,), 96, dtype=torch.int32, device=inp.q_latent.device)
        third = cake_backend.forward(*args, topk_length=lengths, softmax_scale=scale)
        fourth = cake_backend.forward(*args, topk_length=lengths, softmax_scale=scale)
        torch.cuda.synchronize()
        assert (cache.misses, cache.hits) == (misses0 + 2, hits0 + 2)
        assert torch.equal(third[0], first[0]) and torch.equal(fourth[1], first[1])
        # packed views (strided k_rope): a hit re-reads the storage alias and offset
        q = torch.cat([inp.q_latent, inp.q_rope], dim=-1).contiguous()
        kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1).contiguous()
        pv = (
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
            inp.idx_global,
        )
        assert not pv[3].is_contiguous()
        p1 = cake_backend.forward(*pv)
        p2 = cake_backend.forward(*pv)
        torch.cuda.synchronize()
        assert cache.hits == hits0 + 3
        for a, b, r in zip(p1, p2, first, strict=True):
            assert torch.equal(a, b) and torch.equal(a, r)
        packed = cache.peek(forward_binding_key(*pv, None, scale))
        kv2 = torch.cat([kv, kv], dim=0)  # another packed buffer bound to the same plan
        values = packed.rebind_values(
            dict(k_rope=kv2[:, D_LATENT:], indices=inp.idx_global)
        )
        assert values["k_rope_offset"] == kv2[:, D_LATENT:].storage_offset() == D_LATENT
        assert (
            values["k_rope_storage"].data_ptr() == kv2.data_ptr()
            and values["indices_offset"] == 0
        )
        # an invalid topk_length never matches a remembered binding: the validating path raises
        with pytest.raises(ValueError):
            cake_backend.forward(*args, topk_length=lengths.to(torch.int64))
        with pytest.raises(ValueError):
            cake_backend.forward(*args, topk_length=lengths[:-1])
        # a smaller problem allocated after freeing the first (same addresses likely): key differs by shape
        del (
            first,
            second,
            third,
            fourth,
            fresh,
            p1,
            p2,
            q,
            kv,
            kv2,
            pv,
            values,
            args,
            binding,
            packed,
        )
        del inp
        torch.cuda.synchronize()
        small = make_inputs([64, 64], [128, 512], seed=SEED + 14, topk=64)
        misses_before = cache.misses
        out = cake_backend.forward(
            small.q_latent,
            small.q_rope,
            small.kv_latent,
            small.k_rope,
            small.idx_global,
        )
        torch.cuda.synchronize()
        assert cache.misses == misses_before + 1
        ref = reference_fp64(
            small.q_latent,
            small.q_rope,
            small.kv_latent,
            small.k_rope,
            small.idx_global,
        )
        _check_forward(small, out[0], out[1], ref)


def test_binding_cache_backward_hits_are_bitwise_and_fresh():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 13, topk=128)
    fwd_args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    with _cache(True) as cache:
        cache.clear()
        out, lse, o_lo = cake_backend.forward(*fwd_args)
        args = fwd_args + (out, o_lo, lse, inp.dout)
        hits0, misses0 = cache.hits, cache.misses
        a = cake_backend.backward(*args)  # validating path
        b = cake_backend.backward(*args)  # remembered binding
        assert (cache.misses, cache.hits) == (misses0 + 1, hits0 + 1)
        with _cache(False):
            c = cake_backend.backward(*args)
        torch.cuda.synchronize()
        # dQ is computed once per row (bitwise); dK/dV accumulate with red.global (run-to-run spread)
        for x, y, z in zip(a[:2], b[:2], c[:2], strict=True):
            assert torch.equal(x, y) and torch.equal(y, z)
        assert (
            max(
                rel_l2(a[2], b[2]),
                rel_l2(b[2], c[2]),
                rel_l2(a[3], b[3]),
                rel_l2(b[3], c[3]),
            )
            < 1e-2
        )
        assert (
            len({t.data_ptr() for t in (a[2], b[2], c[2])}) == 3
        )  # fresh outputs, not the cached accumulators
        binding = cache.peek(
            backward_binding_key(*args, None, default_softmax_scale(), False)
        )
        # delta and the FP32 accumulators are per-call scratch, not owned by the binding
        assert binding.holds_no_tensor()
        assert set(binding.owned) <= {"topk_length", "tma_descriptor_workspace"}
        assert binding.owned_bytes < (1 << 20)
        # ... and are released with the results: repeated hits leave no allocation behind
        live = torch.cuda.memory_allocated()
        for _ in range(3):
            cake_backend.backward(*args)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == live
        # the FP32 accumulators of dkv_fp32=True are outputs: fresh per call, never aliased between calls
        f1 = cake_backend.backward(*args, dkv_fp32=True)
        f2 = cake_backend.backward(*args, dkv_fp32=True)
        torch.cuda.synchronize()
        # one more binding (dkv_fp32) and one more hit on it, after the three repeated hits above
        assert cache.misses == misses0 + 2 and cache.hits == hits0 + 5
        assert f1[2].dtype == torch.float32 and f1[2].data_ptr() != f2[2].data_ptr()
        snapshot = f2[2].clone()
        f1[2].fill_(7.0)
        assert torch.equal(f2[2], snapshot) and rel_l2(snapshot, a[2].float()) < 1e-2
        fp32_binding = cache.peek(
            backward_binding_key(*args, None, default_softmax_scale(), True)
        )
        assert fp32_binding is not binding and fp32_binding.holds_no_tensor()
        # the accumulators are per-call scratch in both layouts (a permuted program serves the FP32
        # mode through its cast): never owned, never the returned gradients
        assert not {"dkv_latent_acc", "dk_rope_acc"} & set(fp32_binding.owned)
        assert f1[2].data_ptr() not in {
            t.data_ptr() for t in fp32_binding.owned.values()
        }
        assert torch.equal(f1[0], a[0]) and torch.equal(
            f2[1], a[1]
        )  # dq bitwise across modes and calls
        # the autograd path: the forward of a repeated step hits; its backward hits when the allocator hands
        # the freed forward outputs back at the same addresses (informational: the key is the binding)
        leaves = [t.detach().clone().requires_grad_() for t in fwd_args[:4]]
        grads = []
        for repeat in range(2):
            hits, misses = cache.hits, cache.misses
            with _quiet_experimental():
                o = dsa_sparse_attention(*leaves, inp.idx_global)
            assert (cache.hits, cache.misses) == (
                (hits + 1, misses) if repeat else (hits, misses + 1)
            )
            hits, misses = cache.hits, cache.misses
            grads.append(torch.autograd.grad(o, leaves, inp.dout))
            assert cache.hits + cache.misses == hits + misses + 1
            print(
                f"autograd backward repeat {repeat}: {'hit' if cache.hits > hits else 'miss'}"
            )
            del o
        torch.cuda.synchronize()
        assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(
            grads[0][1], grads[1][1]
        )
        assert torch.equal(grads[1][0], a[0]) and torch.equal(grads[1][1], a[1])


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


# --- Packed / strided trainer layouts and causal varlen offsetting (flashinfer-ai/flashinfer#5675) ---
# Packed / strided trainer layouts and the causal tail-of-prefix varlen rule (flashinfer-ai/flashinfer#5675).  The
# kernels take the head / token strides of the query operands and the key row stride from the tensors (TMA descriptors
# encoded from the view; k_rope as storage alias + element offset + row stride), so a view changes descriptor fields
# only: out, lse, o_lo and dQ must be bitwise identical to the contiguous path; dK / dV are FP32 reductions whose order
# is not fixed (FP32 tolerance).


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


# Host layer


def test_offset_gather_kv_indices_causal_tail_of_prefix():
    """The varlen rule of the Cake facade: query ``local_q`` of document ``d`` sits at key position
    ``(seqlen_k[d] - seqlen_q[d]) + local_q`` and, with ``causal=True``, attends selected keys ``<=`` that
    position only; ``causal=False`` (the default, as in the first release) is the plain offsetting; a zero-query
    document contributes no rows."""
    # documents: tail (3 queries, 5 keys), tail (2 queries, 6 keys), keys without queries, full (4 x 4)
    cu_q = torch.tensor([0, 3, 5, 5, 9], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 11, 14, 18], dtype=torch.int32)
    local = torch.tensor(
        [
            [0, 1, 2, -1, 4],  # doc 0, position (5 - 3) + 0 = 2: 4 beyond the bound
            [1, 0, 3, 4, -1],  # position 3: 4 beyond
            [2, 1, 0, 4, 5],  # position 4: 5 >= seqlen_k
            [
                4,
                5,
                3,
                -1,
                6,
            ],  # doc 1, position (6 - 2) + 0 = 4: 5 beyond, 6 >= seqlen_k
            [5, 4, 0, 6, -1],  # position 5: every key <= 5 is causal
            [0, 1, 2, 3, -1],  # doc 3, position 0
            [3, 2, 1, 0, 4],  # position 1: 3, 2 beyond, 4 >= seqlen_k
            [-1, -1, -1, -1, -1],  # fully masked row
            [
                3,
                0,
                1,
                2,
                3,
            ],  # position 3: all causal (a repeated key is a repeated slot)
        ],
        dtype=torch.int32,
    )
    strict_expected = [
        [0, 1, 2, -1, -1],
        [1, 0, 3, -1, -1],
        [2, 1, 0, 4, -1],
        [9, -1, 8, -1, -1],
        [10, 9, 5, -1, -1],
        [14, -1, -1, -1, -1],
        [-1, -1, 15, 14, -1],
        [-1, -1, -1, -1, -1],
        [17, 14, 15, 16, 17],
    ]
    loose_expected = [
        [0, 1, 2, -1, 4],
        [1, 0, 3, 4, -1],
        [2, 1, 0, 4, -1],
        [9, 10, 8, -1, -1],
        [10, 9, 5, -1, -1],
        [14, 15, 16, 17, -1],
        [17, 16, 15, 14, -1],
        [-1, -1, -1, -1, -1],
        [17, 14, 15, 16, 17],
    ]
    strict = offset_gather_kv_indices(local, cu_q, cu_k, causal=True)
    assert strict.dtype == torch.int32 and strict.tolist() == strict_expected
    loose = offset_gather_kv_indices(local, cu_q, cu_k, causal=False)
    assert loose.tolist() == loose_expected
    # the default is the plain offsetting of the first release
    assert torch.equal(offset_gather_kv_indices(local, cu_q, cu_k), loose)
    for causal, expected in ((True, strict), (False, loose)):
        assert torch.equal(
            globalize_gather_indices_loop(local, cu_q, cu_k, causal=causal), expected
        )
    out = torch.empty_like(local)
    assert offset_gather_kv_indices(local, cu_q, cu_k, causal=True, out=out) is out
    assert torch.equal(out, strict)


def _trainer_host_inputs(total_q=8, total_k=16, topk=5):
    q256 = torch.zeros(total_q, NUM_HEADS, 256, dtype=torch.bfloat16)
    q576 = torch.zeros(total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv704 = torch.zeros(total_k, 704, dtype=torch.bfloat16)
    idx = torch.zeros(total_q, topk, dtype=torch.int32)
    return q256, q576, kv704, idx


def test_validate_accepts_trainer_strided_layouts():
    """The GLM-5.2 trainer layouts: q_rope as the 192:256 slice of the [T, 64, 256] pre-absorption query (head
    stride 256, 384 B storage offset), q_latent contiguous or a view of [T, 64, 576], kv_latent / k_rope as column
    slices of a [S, 704] (indexer key alongside) or [S, 576] row -- and in general any 16-byte-multiple head / row
    stride with a 16-byte-aligned base."""
    q256, q576, kv704, idx = _trainer_host_inputs()
    q_rope = q256[:, :, 192:256]
    assert q_rope.stride() == (16384, 256, 1) and q_rope.storage_offset() == 192
    kv576 = kv704[:, :D_QK].contiguous()
    for q_latent in (q576[..., :D_LATENT], q576[..., :D_LATENT].contiguous()):
        for kv in (kv704, kv576):
            assert validate_dsa_train_inputs(
                q_latent, q_rope, kv[:, :D_LATENT], kv[:, D_LATENT:D_QK], idx
            ) == (8, 16, 5)
    # head stride 80 elements (160 B) sliced at channel 8 (16 B); key row stride 520 elements sliced at column 8
    q80 = torch.zeros(8, NUM_HEADS, 80, dtype=torch.bfloat16)
    kv520 = torch.zeros(16, 520, dtype=torch.bfloat16)
    assert validate_dsa_train_inputs(
        q576[..., :D_LATENT],
        q80[:, :, 8:72],
        kv520[:, 8:520],
        kv704[:, D_LATENT:D_QK],
        idx,
    ) == (8, 16, 5)


def _with(q_latent=None, q_rope=None, kv_latent=None, k_rope=None):
    """Validator arguments of the trainer layouts with one operand replaced by ``fn(q256, q576, kv704)``."""

    def mutate(q256, q576, kv704, idx):
        return (
            q576[..., :D_LATENT] if q_latent is None else q_latent(q256, q576, kv704),
            q256[:, :, 192:256] if q_rope is None else q_rope(q256, q576, kv704),
            kv704[:, :D_LATENT] if kv_latent is None else kv_latent(q256, q576, kv704),
            kv704[:, D_LATENT:D_QK] if k_rope is None else k_rope(q256, q576, kv704),
            idx,
        )

    return mutate


@pytest.mark.parametrize(
    "mutate, match",
    [
        # head stride 68 elements = 136 B: not a 16-byte multiple
        (
            _with(
                q_rope=lambda q256, q576, kv704: torch.zeros(
                    8, NUM_HEADS, 68, dtype=torch.bfloat16
                )[..., :D_ROPE]
            ),
            "head stride",
        ),
        # base 8 B into the row: the TMA global address must be 16-byte aligned
        (
            _with(
                q_rope=lambda q256, q576, kv704: torch.zeros(
                    8, NUM_HEADS, 80, dtype=torch.bfloat16
                )[:, :, 4:68]
            ),
            "16-byte aligned",
        ),
        # head stride below the slice width: heads would overlap
        (
            _with(
                q_rope=lambda q256, q576, kv704: torch.as_strided(
                    q576, (8, NUM_HEADS, D_ROPE), (NUM_HEADS * D_QK, 32, 1)
                )
            ),
            "head stride",
        ),
        # token stride 4100 elements: not a 16-byte multiple
        (
            _with(
                q_latent=lambda q256, q576, kv704: torch.as_strided(
                    torch.zeros(65536, dtype=torch.bfloat16),
                    (8, NUM_HEADS, D_LATENT),
                    (4100, D_LATENT, 1),
                )
            ),
            "token stride",
        ),
        # channel stride 2: the innermost dimension is not contiguous
        (
            _with(
                q_latent=lambda q256, q576, kv704: torch.zeros(
                    8, NUM_HEADS, 2 * D_LATENT, dtype=torch.bfloat16
                )[..., ::2]
            ),
            "last dimension",
        ),
        # key row stride 700 elements = 1400 B: not a 16-byte multiple
        (
            _with(
                kv_latent=lambda q256, q576, kv704: torch.zeros(
                    16, 700, dtype=torch.bfloat16
                )[:, :D_LATENT]
            ),
            "row stride",
        ),
        # key base 8 B into the row
        (_with(kv_latent=lambda q256, q576, kv704: kv704[:, 4:516]), "16-byte aligned"),
        # key row stride below the slice width: rows would overlap
        (
            _with(
                k_rope=lambda q256, q576, kv704: torch.as_strided(
                    kv704, (16, D_ROPE), (32, 1)
                )
            ),
            "row stride",
        ),
    ],
)
def test_validate_rejects_unsupported_strides(mutate, match):
    q256, q576, kv704, idx = _trainer_host_inputs()
    with pytest.raises(ValueError, match=match):
        validate_dsa_train_inputs(*mutate(q256, q576, kv704, idx))


# Device tests


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


def test_dkv_dst_map_value_check_is_opt_in(monkeypatch):
    """``_check_dkv_acc`` validates dtype / shape / contiguity / device of ``dkv_dst_map`` but not its values
    (the kernel does not range-check the map); with ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP=1`` an out-of-range
    destination row raises ``ValueError`` and an in-range map passes."""
    dkv_acc = torch.zeros(4, D_QK, dtype=torch.float32)
    good = torch.tensor([3, 0, 3], dtype=torch.int32)
    bad = torch.tensor([0, 4, 1], dtype=torch.int32)  # 4 == S_dst is outside [0, S_dst)
    monkeypatch.delenv("FLASHINFER_CAKE_DSA_CHECK_DST_MAP", raising=False)
    for dst_map in (good, bad):
        operand, row_stride = cake_backend._check_dkv_acc(dkv_acc, dst_map, num_kv=3)
        assert operand is dkv_acc and row_stride == D_QK
    monkeypatch.setenv("FLASHINFER_CAKE_DSA_CHECK_DST_MAP", "1")
    assert cake_backend._check_dkv_acc(dkv_acc, good, num_kv=3)[1] == D_QK
    with pytest.raises(ValueError, match=r"dkv_dst_map values must lie in \[0, 4\)"):
        cake_backend._check_dkv_acc(dkv_acc, bad, num_kv=3)
    with pytest.raises(ValueError, match="dkv_dst_map values must lie in"):
        cake_backend._check_dkv_acc(
            dkv_acc, torch.tensor([-1, 0, 0], dtype=torch.int32), num_kv=3
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


# --- end of the packed / strided layout block ---
# --- Packed fp32 dK/dV accumulation and destination mapping (flashinfer-ai/flashinfer#5675) ---
# Packed FP32 dK/dV accumulation into a caller-provided buffer with an optional destination-row map (issue #5675).
# Comparisons across two backward calls use a tolerance: their FP32 reductions run in another order.
ACC_REL_L2 = 1e-4


def _acc_plan_record(with_accumulate: bool) -> dict:
    """A fake record whose bwd_cast plan does (not) declare the packed-accumulate operands."""
    plan = [
        ["buffer", "src_latent"],
        ["buffer", "src_rope"],
        ["buffer", "dst_latent"],
        ["buffer", "dst_rope"],
        ["buffer", "dst_latent_f32"],
        ["buffer", "dst_rope_f32"],
        ["parameter", "latent_groups"],
        ["parameter", "rope_groups"],
        ["parameter", "out_f32"],
    ]
    if with_accumulate:
        plan += [
            ["buffer", "dst_packed"],
            ["parameter", "dst_row_stride"],
            ["buffer", "dst_map"],
            ["parameter", "has_dst_map"],
            ["parameter", "accumulate"],
        ]
    plan += [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
    return {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["fwd", "bwd_delta", "bwd_main", "bwd_cast"],
        "dkv_acc_layout": "permuted",
        "bwd_cast": {
            "module": "fake",
            "sources": [],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": plan,
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0" * 64,
    }


def _cast_operand_tensors(S: int) -> dict:
    return dict(
        q_latent=torch.zeros(2, NUM_HEADS, D_LATENT, dtype=torch.bfloat16),
        q_rope=torch.zeros(2, NUM_HEADS, D_ROPE, dtype=torch.bfloat16),
        kv_latent=torch.zeros(S, D_LATENT, dtype=torch.bfloat16),
        k_rope=torch.zeros(S, D_ROPE, dtype=torch.bfloat16),
        indices=torch.zeros(2, 4, dtype=torch.int32),
        delta=torch.zeros(2, NUM_HEADS),
        dkv_latent_acc=torch.zeros(S, D_LATENT),
        dk_rope_acc=torch.zeros(S, D_ROPE),
        dkv_latent=torch.empty(0, dtype=torch.bfloat16),
        dk_rope=torch.empty(0, dtype=torch.bfloat16),
        dkv_latent_fp32=torch.zeros(S, D_LATENT),
        dk_rope_fp32=torch.zeros(S, D_ROPE),
    )


def test_record_cast_accumulates_reads_the_plan():
    assert cake_backend.record_cast_accumulates(_acc_plan_record(True))
    assert not cake_backend.record_cast_accumulates(_acc_plan_record(False))
    assert not cake_backend.record_cast_accumulates(
        {"arch": "sm_100a", "stages": ["fwd"]}
    )
    assert set(cake_backend.CAST_ACCUMULATE_OPERANDS) <= set(
        cake_backend.CONTRACT_ALIASES
    ) | set(cake_backend.CONTRACT_SCALARS)
    for name in cake_jit.MODULES:
        assert isinstance(
            cake_backend.record_cast_accumulates(cake_jit.MODULES[name]), bool
        )


def test_check_dkv_acc_accepts_the_contract_layouts():
    S = 16
    acc576 = torch.zeros(S, D_QK)
    operand, stride = cake_backend._check_dkv_acc(acc576, None, S)
    assert operand is acc576 and stride == D_QK
    acc704 = torch.zeros(
        S + 4, 704
    )  # a frozen 128-channel indexer key stored alongside the 576 channels
    operand, stride = cake_backend._check_dkv_acc(acc704, None, S)
    assert operand is acc704 and stride == 704
    view = acc704[
        :, :D_QK
    ]  # the 576 columns as a strided view: a flat alias from its first element
    operand, stride = cake_backend._check_dkv_acc(view, None, S)
    assert stride == 704 and operand.is_contiguous() and operand.dim() == 1
    assert (
        operand.data_ptr() == view.data_ptr()
        and operand.numel() == (S + 4 - 1) * 704 + D_QK
    )
    dst_map = torch.zeros(S, dtype=torch.int32)
    operand, stride = cake_backend._check_dkv_acc(
        torch.zeros(3, D_QK), dst_map, S
    )  # fewer rows than S with a map
    assert stride == D_QK


@pytest.mark.parametrize(
    "acc, dst_map, match",
    [
        (lambda: torch.zeros(16, D_QK, dtype=torch.bfloat16), None, "float32"),
        (lambda: torch.zeros(16 * D_QK), None, "2-D"),
        (lambda: torch.zeros(16, D_LATENT), None, ">= 576"),
        (lambda: torch.zeros(D_QK, 16).t(), None, r"stride\(1\)"),
        (lambda: torch.zeros(16, 578), None, "multiple of 4"),
        (lambda: torch.zeros(16, 580)[:, 1:], None, "16-byte"),
        (lambda: torch.zeros(8, D_QK), None, "at least S"),
        (lambda: torch.zeros(16, D_QK), torch.zeros(16, dtype=torch.int64), "int32"),
        (
            lambda: torch.zeros(16, D_QK),
            torch.zeros(15, dtype=torch.int32),
            r"int32 \[S\]",
        ),
        (
            lambda: torch.zeros(16, D_QK),
            torch.zeros(32, dtype=torch.int32)[::2],
            "contiguous",
        ),
    ],
)
def test_check_dkv_acc_rejects(acc, dst_map, match):
    with pytest.raises(ValueError, match=match):
        cake_backend._check_dkv_acc(acc(), dst_map, 16)


def test_backward_binding_key_covers_dkv_acc_and_map():
    T, S = 4, 8
    q = torch.zeros(T, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv = torch.zeros(S, D_QK, dtype=torch.bfloat16)
    idx = torch.zeros(T, 5, dtype=torch.int32)
    out = torch.zeros(T, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(T, NUM_HEADS)
    args = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        idx,
        out,
        out,
        lse,
        out,
        None,
        1.0,
        False,
    )
    plain = backward_binding_key(*args)
    acc = torch.zeros(S, 704)
    with_acc = backward_binding_key(*args, dkv_acc=acc)
    assert plain != with_acc and with_acc == backward_binding_key(*args, dkv_acc=acc)
    assert with_acc != backward_binding_key(
        *args, dkv_acc=acc[:, :D_QK]
    )  # another row layout of the same storage
    dst_map = torch.zeros(S, dtype=torch.int32)
    assert with_acc != backward_binding_key(*args, dkv_acc=acc, dkv_dst_map=dst_map)


def test_bind_stage_serves_the_accumulating_cast_plan(monkeypatch):
    """The packed-accumulate operands bind from ``dkv_acc`` / ``dkv_dst_map`` (aliases ``dst_packed`` / ``dst_map``) and
    the three scalars; without them the host binds inert placeholders of the right dtypes, as the production launcher does."""
    monkeypatch.setitem(
        cake_jit.MODULES, "cake_dsa_h64_train_fake_acc", _acc_plan_record(True)
    )
    monkeypatch.setattr(
        cake_backend,
        "load_cake_dsa_train_module",
        lambda name, stage: SimpleNamespace(run=lambda *a: None),
    )
    S = 8
    t = _cast_operand_tensors(S)
    acc = torch.zeros(S, 704)
    dst_map = torch.arange(S, dtype=torch.int32)
    t["dkv_acc"], t["dkv_dst_map"] = acc, dst_map
    scalars = dict(
        num_queries=2,
        num_kv=S,
        topk=4,
        softmax_scale=1.0,
        has_topk_length=0,
        out_f32=0,
        dst_row_stride=704,
        has_dst_map=1,
        accumulate=1,
    )
    values = cake_backend._contract_values(t, scalars)
    launch = bind_stage("cake_dsa_h64_train_fake_acc", "bwd_cast", values, (3, 1, 1))
    assert launch.arguments[9] is acc and launch.arguments[10] == 704
    assert launch.arguments[11] is dst_map and launch.arguments[12:14] == (1, 1)
    assert {key for _, key in launch.slots} >= {"dkv_acc", "dkv_dst_map"}
    t.pop("dkv_acc")
    t.pop("dkv_dst_map")
    inert = cake_backend._contract_values(
        t, dict(scalars, dst_row_stride=0, has_dst_map=0, accumulate=0)
    )
    launch = bind_stage("cake_dsa_h64_train_fake_acc", "bwd_cast", inert, (3, 1, 1))
    assert (
        launch.arguments[9] is t["dkv_latent_acc"]
        and launch.arguments[11] is t["indices"]
    )
    assert launch.arguments[10] == 0 and launch.arguments[12:14] == (0, 0)
    # a remembered binding re-supplies both per call (placeholders when the call accumulates nothing)
    rebound = launch.templated().arguments_for(
        cake_backend._inert_cast_values(dict(t, indices_storage=t["indices"]))
    )
    assert rebound[9] is t["dkv_latent_acc"] and rebound[11] is t["indices"]
    # the old plan (no accumulate operands) still binds from the same values
    monkeypatch.setitem(
        cake_jit.MODULES, "cake_dsa_h64_train_fake_old", _acc_plan_record(False)
    )
    old = bind_stage("cake_dsa_h64_train_fake_old", "bwd_cast", values, (3, 1, 1))
    assert len(old.arguments) == 12 and old.arguments[0] is t["dkv_latent_acc"]


def test_backward_rejects_dkv_acc_with_dkv_fp32_and_a_map_without_acc():
    q, kv, idx = _host_inputs()
    out = torch.zeros(8, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(8, NUM_HEADS)
    args = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        idx,
        out,
        out,
        lse,
        out,
    )
    with pytest.raises(ValueError, match="dkv_fp32"):
        cake_backend.backward(*args, dkv_fp32=True, dkv_acc=torch.zeros(16, D_QK))
    with pytest.raises(ValueError, match="dkv_dst_map"):
        cake_backend.backward(*args, dkv_dst_map=torch.zeros(16, dtype=torch.int32))


def _require_accumulating_cast():
    _, record = record_for(torch.device("cuda"))
    if not cake_backend.record_cast_accumulates(record):
        pytest.skip(
            "the registered program's bwd_cast predates the packed-accumulate operands (re-export needed)"
        )


def _forward_and_natural_dkv(inp):
    """Forward outputs, the explicit backward's dq and the natural FP32 dK/dV as one packed ``[S, 576]`` reference."""
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    args = (
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    dq_l, dq_r, nat_l, nat_r = cake_backend.backward(*args, dkv_fp32=True)
    torch.cuda.synchronize()
    return args, dq_l, dq_r, torch.cat([nat_l, nat_r], dim=1)


def test_backward_dkv_acc_identity_accumulates_in_place():
    """``dkv_acc`` without a map: ``+= dkv`` into a pre-filled ``[S, 704]`` buffer (twice over two calls), columns 576:
    untouched, no dK/dV outputs, dq unchanged; the strided 576-column view of the buffer is accepted as well."""
    _require_program(backward=True)
    _require_accumulating_cast()
    inp = make_inputs([256], [512], seed=SEED + 20, topk=128)
    args, dq_l, dq_r, nat = _forward_and_natural_dkv(inp)
    S = inp.total_k
    gen = torch.Generator(device="cuda").manual_seed(1)
    pre = torch.randn(S, 704, device="cuda", generator=gen)
    acc = pre.clone()
    aq_l, aq_r, none_l, none_r = cake_backend.backward(*args, dkv_acc=acc)
    torch.cuda.synchronize()
    assert none_l is None and none_r is None
    assert torch.equal(aq_l, dq_l) and torch.equal(aq_r, dq_r)
    torch.testing.assert_close(acc[:, :D_QK], pre[:, :D_QK] + nat, rtol=1e-4, atol=1e-4)
    assert torch.equal(acc[:, D_QK:], pre[:, D_QK:])
    cake_backend.backward(
        *args, dkv_acc=acc[:, :D_QK]
    )  # second call, through the strided view: keeps adding
    torch.cuda.synchronize()
    torch.testing.assert_close(
        acc[:, :D_QK], pre[:, :D_QK] + 2 * nat, rtol=1e-4, atol=1e-4
    )
    assert torch.equal(acc[:, D_QK:], pre[:, D_QK:])


def test_bwd_cast_accumulate_without_map_is_one_fp32_add_bitwise():
    """From the SAME permuted accumulators (a prepared runner's), the accumulating cast without a map equals
    ``previous + natural`` bitwise: one FP32 add per element on both sides."""
    _require_program(backward=True)
    _require_accumulating_cast()
    inp = make_inputs([128], [256], seed=SEED + 21, topk=64)
    runner = prepare_dsa_train(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        backward=True,
        dkv_fp32=True,
    )
    runner.forward()
    _, _, nat_l, nat_r = runner.backward()
    torch.cuda.synchronize()
    S = inp.total_k
    pre = torch.randn(
        S, 704, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2)
    )
    acc = pre.clone()
    t = dict(runner.tensors, dkv_acc=acc)
    scalars = dict(
        num_queries=runner.num_queries,
        num_kv=S,
        topk=runner.topk,
        softmax_scale=runner.softmax_scale,
        has_topk_length=int(runner.has_topk_length),
        out_f32=0,
        dst_row_stride=704,
        has_dst_map=0,
        accumulate=1,
    )
    values = cake_backend._contract_values(t, scalars, key_passes=runner.key_passes)
    physical = record_for(torch.device("cuda"))[1]["bwd_cast"]
    grid = grid_dims(
        physical["grid"],
        scalars,
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    launch = bind_stage(runner.module_name, "bwd_cast", values, grid)
    with cake_backend._ffi_stream_context(runner.device_index):
        launch()
    torch.cuda.synchronize()
    assert torch.equal(acc[:, :D_LATENT], pre[:, :D_LATENT] + nat_l)
    assert torch.equal(acc[:, D_LATENT:D_QK], pre[:, D_LATENT:D_QK] + nat_r)
    assert torch.equal(acc[:, D_QK:], pre[:, D_QK:])


def test_backward_dkv_acc_destination_map_with_duplicates_matches_index_add():
    """A many-to-few map (512 source rows into 300 destination rows) sums like ``index_add_`` of the natural dK/dV;
    unmapped destination rows stay zero; a permutation lands every source row on its own destination row."""
    _require_program(backward=True)
    _require_accumulating_cast()
    inp = make_inputs([256], [512], seed=SEED + 22, topk=128)
    args, _, _, nat = _forward_and_natural_dkv(inp)
    S, S_dst = inp.total_k, 300
    gen = torch.Generator(device="cuda").manual_seed(3)
    dst_map = torch.randint(0, S_dst, (S,), device="cuda", generator=gen).to(
        torch.int32
    )
    acc = torch.zeros(S_dst, 704, device="cuda")
    _, _, none_l, none_r = cake_backend.backward(
        *args, dkv_acc=acc, dkv_dst_map=dst_map
    )
    torch.cuda.synchronize()
    assert none_l is None and none_r is None
    expect = torch.zeros(S_dst, D_QK, device="cuda").index_add_(
        0, dst_map.to(torch.int64), nat
    )
    assert rel_l2(acc[:, :D_QK], expect) < ACC_REL_L2
    assert torch.all(acc[:, D_QK:] == 0)
    untouched = torch.ones(S_dst, dtype=torch.bool, device="cuda")
    untouched[dst_map.long()] = False
    assert untouched.any() and torch.all(acc[untouched] == 0)
    perm = torch.randperm(S, device="cuda", generator=gen).to(torch.int32)
    acc_p = torch.zeros(S, D_QK, device="cuda")
    cake_backend.backward(*args, dkv_acc=acc_p, dkv_dst_map=perm)
    torch.cuda.synchronize()
    assert rel_l2(acc_p[perm.long()], nat) < ACC_REL_L2


def test_binding_cache_serves_dkv_acc_bindings_per_buffer():
    """A remembered dkv_acc binding re-supplies the caller's buffer and map per call; another buffer layout binds anew."""
    _require_program(backward=True)
    _require_accumulating_cast()
    inp = make_inputs([128], [256], seed=SEED + 24, topk=64)
    args, _, _, nat = _forward_and_natural_dkv(inp)
    S = inp.total_k
    cache = cake_backend.BINDING_CACHE
    if not cache.enabled:
        pytest.skip("binding cache disabled")
    acc = torch.zeros(S, D_QK, device="cuda")
    dst_map = torch.arange(S, device="cuda", dtype=torch.int32)
    key = backward_binding_key(
        *args, None, default_softmax_scale(), False, None, acc, dst_map
    )
    cake_backend.backward(*args, dkv_acc=acc, dkv_dst_map=dst_map)
    binding = cache.peek(key)
    assert binding is not None and binding.accumulate_dkv and binding.holds_no_tensor()
    hits = cache.hits
    cake_backend.backward(
        *args, dkv_acc=acc, dkv_dst_map=dst_map
    )  # same binding: a hit, adds again
    torch.cuda.synchronize()
    assert cache.hits == hits + 1
    assert rel_l2(acc, 2 * nat) < ACC_REL_L2
    acc2 = torch.zeros(
        S + 2, 704, device="cuda"
    )  # another layout: another binding (miss), row stride 704 baked
    cake_backend.backward(*args, dkv_acc=acc2)
    torch.cuda.synchronize()
    assert cache.hits == hits + 1
    assert (
        rel_l2(acc2[:S, :D_QK], nat) < ACC_REL_L2
        and torch.all(acc2[S:] == 0)
        and torch.all(acc2[:, D_QK:] == 0)
    )


def test_autograd_function_with_dkv_acc_returns_no_kv_grads_and_accumulates():
    _require_program(backward=True)
    _require_accumulating_cast()
    inp = make_inputs([128], [256], seed=SEED + 23, topk=64)
    args, dq_l, dq_r, nat = _forward_and_natural_dkv(inp)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    S = inp.total_k
    acc = torch.zeros(S, D_QK, device="cuda")
    out, lse = cake_backend.DSASparseAttentionFunction.apply(
        *leaves, inp.idx_global, None, default_softmax_scale(), None, acc, None
    )
    out.backward(inp.dout)
    torch.cuda.synchronize()
    assert leaves[0].grad is not None and leaves[1].grad is not None
    assert leaves[2].grad is None and leaves[3].grad is None
    assert torch.equal(leaves[0].grad, dq_l) and torch.equal(leaves[1].grad, dq_r)
    assert rel_l2(acc, nat) < ACC_REL_L2
    perm = torch.randperm(
        S, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)
    ).to(torch.int32)
    acc_p = torch.zeros(S, D_QK, device="cuda")
    out2, _ = cake_backend.DSASparseAttentionFunction.apply(
        *leaves, inp.idx_global, None, default_softmax_scale(), None, acc_p, perm
    )
    out2.backward(inp.dout)
    torch.cuda.synchronize()
    assert leaves[2].grad is None and leaves[3].grad is None
    assert rel_l2(acc_p[perm.long()], nat) < ACC_REL_L2


# --- end of the packed fp32 dK/dV block ---
