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
    NUM_HEADS,
    SUPPORTED_ABIS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    WORKSPACE_ALIGN,
    bind_stage,
    generated_program_available,
    grid_dims,
    offset_gather_kv_indices,
    prepare_dsa_train,
    record_abi,
    record_for,
    validate_dsa_train_inputs,
    workspace_layout,
)
from tests.test_helpers.cake_dsa_train_reference import (
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
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
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
        for stage in stages:
            physical = record[stage]
            assert len(physical["sources"]) == 2
            assert all(s.startswith("cake_dsa_h64_train/") for s in physical["sources"])
            assert len(physical["closure_sha256"]) == 64
            assert all(kind in {"buffer", "tma_buffer", "workspace", "parameter", "grid"} for kind, _ in physical["arg_plan"])
            launch = physical.get("launch")
            if launch is not None:
                assert len(launch["cluster"]) == 3 and all(int(c) >= 1 for c in launch["cluster"])
                assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
            assert len(physical.get("grid", ["num_queries", 1, 1])) == 3


@pytest.mark.parametrize("abi", SUPPORTED_ABIS)
@pytest.mark.parametrize("backward", [False, True])
def test_workspace_layout(abi, backward):
    layout = workspace_layout(300, 1000, 96, abi=abi, backward=backward, tma_workspace_bytes=1024)
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


def test_grid_dims():
    scalars = dict(num_queries=130, num_kv=4096)
    assert grid_dims(["num_queries", 1, 1], scalars, 148) == (130, 1, 1)
    assert grid_dims(["num_queries/64", 1, 1], scalars, 148) == (3, 1, 1)
    assert grid_dims(["sms", 1, 1], scalars, 148) == (148, 1, 1)
    assert grid_dims(["sms*2", 1, 1], scalars, 148) == (296, 1, 1)
    assert grid_dims([4, 2, 1], scalars, 148) == (4, 2, 1)
    with pytest.raises(ValueError):
        grid_dims([1, 1], scalars, 148)
    assert grid_dims(["num_queries*8", 1, 1], {"num_queries": 5, "num_kv": 7, "topk": 3}, 148) == (40, 1, 1)
    assert grid_dims(["num_kv*36/256", 1, 1], {"num_queries": 5, "num_kv": 7, "topk": 3}, 148) == (1, 1, 1)
    assert grid_dims(["num_kv*36/256", 1, 1], {"num_queries": 5, "num_kv": 4096, "topk": 3}, 148) == (576, 1, 1)


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
    got = offset_gather_kv_indices(local, cu_q, cu_k)
    assert got.dtype == torch.int32
    assert torch.equal(got, expected)
    out = torch.empty_like(local)
    assert offset_gather_kv_indices(local, cu_q, cu_k, out=out) is out
    assert torch.equal(out, expected)


def _host_inputs(total_q=8, total_k=16, topk=5):
    q = torch.zeros(total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv = torch.zeros(total_k, D_QK, dtype=torch.bfloat16)
    idx = torch.zeros(total_q, topk, dtype=torch.int32)
    return q, kv, idx


def test_validate_accepts_packed_views():
    q, kv, idx = _host_inputs()
    t, s, k = validate_dsa_train_inputs(q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx)
    assert (t, s, k) == (8, 16, 5)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda q, kv, idx: (q[..., :D_LATENT].float(), q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx), "bfloat16"),
        (lambda q, kv, idx: (q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx.long()), "int32"),
        (lambda q, kv, idx: (q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx[:4]), r"\[T, topk\]"),
        (lambda q, kv, idx: (q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:-1, D_LATENT:], idx), "same number of rows"),
        (lambda q, kv, idx: (q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT].t().contiguous().t(), kv[:, D_LATENT:], idx), "contiguous"),
        (lambda q, kv, idx: (q[:, :32, :D_LATENT], q[:, :32, D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx), r"\[T, 64, 512\]"),
    ],
)
def test_validate_rejects(mutate, match):
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match=match):
        validate_dsa_train_inputs(*mutate(q, kv, idx))


def test_validate_rejects_bad_topk_length():
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match="topk_length"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx,
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
            "arg_plan": [["buffer", "q_latent"], ["parameter", "not_a_host_value"], ["grid", "grid_x"]],
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0" * 64,
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dsa_h64_train_fake", record)
    values = {"q_latent": torch.zeros(1), "softmax_scale": 1.0}
    with pytest.raises(KeyError, match="not_a_host_value"):
        bind_stage("cake_dsa_h64_train_fake", "fwd", values, (1, 1, 1))


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
        assert (lse.double()[finite] - lse_ref[finite]).abs().max().item() <= GATE_LSE_ABS
    if (~finite).any():
        assert torch.all(lse[~finite] == float("-inf"))
        rows = ~finite.any(-1) if valid_rows is None else ~valid_rows  # fully masked rows
        assert torch.all(out[rows] == 0)


def _check_backward(grads, ref, *, peaked=False):
    dq_latent, dq_rope, dkv_latent, dk_rope = grads
    for g in grads:
        assert torch.isfinite(g.float()).all()
    dq_gate = GATE_PEAKED if peaked else GATE_DQ_LATENT
    dkv_gate = GATE_PEAKED if peaked else GATE_DKV_LATENT
    rope_factor = 1.0 if peaked else GATE_ROPE_FACTOR
    assert rel_l2(dq_latent, ref["dq_latent"]) <= dq_gate
    assert rel_l2(dq_rope, ref["dq_rope"]) <= dq_gate * rope_factor
    assert rel_l2(dkv_latent, ref["dkv_latent"]) <= dkv_gate
    assert rel_l2(dk_rope, ref["dk_rope"]) <= dkv_gate * rope_factor


@pytest.mark.parametrize("topk", [128, 200])
def test_forward_iid(topk):
    _require_program()
    if topk % 64:
        _require_contract_abi()
    inp = make_inputs([384], [1024], seed=SEED, topk=topk)
    out, lse, _ = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    torch.cuda.synchronize()
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    _check_forward(inp, out, lse, ref)


def test_forward_accepts_packed_views_bitwise():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 1, topk=128)
    q = torch.cat([inp.q_latent, inp.q_rope], dim=-1).contiguous()
    kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1).contiguous()
    out_split, lse_split, _ = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    out_view, lse_view, _ = cake_backend.forward(q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], inp.idx_global)
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
    out, lse, _ = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, topk_length=topk_length)
    torch.cuda.synchronize()
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, topk_length=topk_length)
    _check_forward(inp, out, lse, ref)
    assert torch.all(out[5] == 0) and torch.all(lse[5] == float("-inf"))
    assert torch.all(out[40] == 0) and torch.all(lse[40] == float("-inf"))


def test_forward_deterministic():
    _require_program()
    inp = make_inputs([320], [640], seed=SEED + 3, topk=128)
    a = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    b = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    if a[2] is not None:
        assert torch.equal(a[2], b[2])


def test_varlen_multi_document_matches_flat_and_reference():
    _require_program()
    inp = make_inputs([64, 200, 120], [128, 200, 384], seed=SEED + 4, topk=128)
    out_flat, lse_flat, _ = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    with _quiet_experimental():
        out_var, lse_var = dsa_sparse_attention_varlen(
            inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_local,
            inp.cu_seqlens_q, inp.cu_seqlens_k, inp.max_seqlen_q, inp.max_seqlen_k, return_lse=True,
        )
    torch.cuda.synchronize()
    assert torch.equal(out_flat, out_var) and torch.equal(lse_flat, lse_var)
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    _check_forward(inp, out_var, lse_var, ref)


def test_runner_launches_without_allocation():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 5, topk=128)
    runner = prepare_dsa_train(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, backward=False)
    runner.forward()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.forward()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]


def test_backward_iid():
    _require_program(backward=True)
    inp = make_inputs([384], [1024], seed=SEED + 6, topk=200)
    out, lse, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    grads = cake_backend.backward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, dout=inp.dout)
    _check_forward(inp, out, lse, ref)
    _check_backward(grads, ref)


def test_backward_masked_rows_give_zero_dq():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 7, topk=128)
    idx = inp.idx_global.clone()
    idx[3] = -1
    idx[9, ::2] = -1
    out, lse, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx)
    dq_latent, dq_rope, dkv_latent, dk_rope = cake_backend.backward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, out, o_lo, lse, inp.dout
    )
    torch.cuda.synchronize()
    assert torch.all(dq_latent[3] == 0) and torch.all(dq_rope[3] == 0)
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, dout=inp.dout)
    _check_backward((dq_latent, dq_rope, dkv_latent, dk_rope), ref)


def test_backward_peaked():
    _require_program(backward=True)
    inp = make_inputs([512], [512], seed=SEED + 8, topk=128, self_including=True, beta=3.0)
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, dout=inp.dout, own_key=inp.own_key)
    assert ref["self_weight"] > 0.5, "the peaked configuration must concentrate the softmax on the own key"
    out, lse, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    grads = cake_backend.backward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    _check_backward(grads, ref, peaked=True)
    p99 = rel_l2_rows(grads[0], ref["dq_latent"]).quantile(0.99).item()
    assert p99 <= GATE_ROW_P99_DQ


def test_backward_deterministic_dq_and_dkv_spread():
    _require_program(backward=True)
    inp = make_inputs([320], [640], seed=SEED + 9, topk=128)
    out, lse, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    a = cake_backend.backward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, out, o_lo, lse, inp.dout)
    b = cake_backend.backward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    spread = max(rel_l2(a[2], b[2]), rel_l2(a[3], b[3]))
    assert spread < 1e-2, f"dkv run-to-run spread {spread}"


def test_autograd_function_matches_explicit_backward():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 10, topk=128)
    leaves = [t.detach().clone().requires_grad_() for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention(*leaves, inp.idx_global, return_lse=True)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    o2, l2, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    explicit = cake_backend.backward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, o2, o_lo, l2, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out, o2) and torch.equal(lse, l2)
    assert torch.equal(grads[0], explicit[0]) and torch.equal(grads[1], explicit[1])
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, dout=inp.dout)
    _check_backward(grads, ref)


def test_backward_dkv_fp32_returns_accumulators():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 11, topk=128)
    out, lse, o_lo = cake_backend.forward(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    _, _, dkv_latent, dk_rope = cake_backend.backward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, out, o_lo, lse, inp.dout, dkv_fp32=True
    )
    torch.cuda.synchronize()
    assert dkv_latent.dtype == torch.float32 and dk_rope.dtype == torch.float32
    ref = reference_fp64(inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global, dout=inp.dout)
    assert rel_l2(dkv_latent, ref["dkv_latent"]) <= GATE_DKV_LATENT
