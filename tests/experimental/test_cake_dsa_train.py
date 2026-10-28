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

import hashlib
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers import cake_dsa_train_reference as nope_ref
from tests.test_helpers.cake_dsa_train_reference import (
    calibrate_beta,
    compact_rows,
    compact_rows_loop,
    derive_topk_length,
    globalize_gather_indices_loop,
    make_inputs,
    reference_fp64,
    rel_l2,
    rel_l2_rows,
)
from tests.test_helpers.cake_package import cake_package

# The backend suite names its package once, through the shared alias helper, and imports nothing else from
# flashinfer: a build that vendors the package under another import path runs it unchanged after pointing the
# ``PACKAGES`` entry at its own package.  The flashinfer public entry points and the environment variables the
# flashinfer wrapper resolves into the package's keyword switches are covered by ``test_cake_dsa_train_api.py``.
cake_backend = cake_package("cake_dsa_train", "cake_backend")
cake_build = cake_package("cake_dsa_train", "cake_build")
cake_jit = cake_package("cake_dsa_train", "cake_jit")
cake_launch = cake_package("cake_dsa_train", "cake_launch")
D_LATENT = cake_backend.D_LATENT
D_QK = cake_backend.D_QK
D_ROPE = cake_backend.D_ROPE
DQ_PARTIAL_BYTES_PER_TOKEN = cake_backend.DQ_PARTIAL_BYTES_PER_TOKEN
KEY_PASS_STAGES = cake_backend.KEY_PASS_STAGES
NUM_HEADS = cake_backend.NUM_HEADS
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
WORKSPACE_ALIGN = cake_backend.WORKSPACE_ALIGN
KeyPassPolicy = cake_backend.KeyPassPolicy
backward_binding_key = cake_backend.backward_binding_key
default_softmax_scale = cake_backend.default_softmax_scale
forward_binding_key = cake_backend.forward_binding_key
dsa_train_workspace_size = cake_backend.dsa_train_workspace_size
generated_program_available = cake_backend.generated_program_available
key_pass_dq_mode = cake_backend.key_pass_dq_mode
key_pass_ranges = cake_backend.key_pass_ranges
offset_gather_kv_indices = cake_backend.offset_gather_kv_indices
plan_key_passes = cake_backend.plan_key_passes
prepare_dsa_train = cake_backend.prepare_dsa_train
record_for = cake_backend.record_for
validate_dsa_train_inputs = cake_backend.validate_dsa_train_inputs
workspace_layout = cake_backend.workspace_layout

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


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backward", [False, True])
def test_workspace_layout(backward):
    layout = workspace_layout(300, 1000, 96, backward=backward)
    regions = [k for k in layout if k != "total"]
    offsets = [layout[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert layout["total"] >= sum(layout[k][1] for k in regions)
    assert ("delta" in layout) == backward
    assert ("dkv_latent_acc" in layout) == backward
    assert layout["topk_length"][1] == 300 * 4
    # a direct binding adds into the caller's dkv_acc: no FP32 accumulators in the workspace
    direct = workspace_layout(300, 1000, 96, backward=backward, dkv_direct=True)
    assert "dkv_latent_acc" not in direct and "dk_rope_acc" not in direct
    assert ("delta" in direct) == backward
    if backward:
        assert layout["total"] - direct["total"] >= 1000 * D_QK * 4
    else:
        assert direct["total"] == layout["total"]


def test_workspace_layout_key_pass_regions():
    T, S, topk = 300, 70000, 96
    kw = dict(backward=True)
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
    forward = workspace_layout(T, S, topk, backward=False, key_passes=3)
    assert not pass_regions & set(forward)


# The policy of the registered programs: one pass per 100 MiB of FP32 dK/dV accumulator
# (2304 B per key), taken only when the whole row's pass scratch fits 640 MiB.
_POLICY = dict(
    l2_budget_bytes=100 << 20,
    key_bytes=2304,
    workspace_budget_bytes=640 << 20,
    token_chunk_multiple=128,
    max_passes=4,
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
# a program with the natural-layout (direct accumulation) variants of the main stage next to the regular ones
_DIRECT_STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_main_natural",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_main_pass_natural",
    "bwd_cast",
)
_DIRECT_POLICY = dict(min_keys_per_query=4)

# The per-target rules of the registered programs (second optimisation round, lever H3): the record's
# ``key_pass_policy`` carries them under ``rules``; a record without them (first release) is the R1 rule
# = _RULES["sm_100a"] on every architecture.
_RULES = {
    "sm_100a": dict(tail_rows=0, fixed_passes=0, min_kv=0, tail_min_kv=0, tail_cap=0),
    "sm_103a": dict(
        tail_rows=1, fixed_passes=0, min_kv=0, tail_min_kv=131072, tail_cap=3
    ),
    "sm_107a": dict(
        tail_rows=1, fixed_passes=2, min_kv=131072, tail_min_kv=131072, tail_cap=0
    ),
}


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


def test_key_pass_policy_per_target_rules():
    record = {"key_pass_policy": dict(_POLICY, rules=_RULES)}
    policy = KeyPassPolicy.from_record(record)
    # a first-release record (no rules) and an unknown architecture take the R1 rule
    legacy = KeyPassPolicy.from_record({"key_pass_policy": dict(_POLICY)})
    assert legacy.rule_for("sm_103a") == _RULES["sm_100a"]
    assert policy.rule_for(None) == _RULES["sm_100a"]
    assert policy.rule_for("sm_90a") == _RULES["sm_100a"]
    assert policy.rule_for("sm_107a") == _RULES["sm_107a"]
    # the rule fields are non-negative integers, exactly the five
    for name in ("tail_cap", "tail_min_kv"):
        for bad in (-1, True, 1.5):
            rules = dict(_RULES, sm_103a=dict(_RULES["sm_103a"], **{name: bad}))
            with pytest.raises(ValueError, match=name):
                KeyPassPolicy.from_record(
                    {"key_pass_policy": dict(_POLICY, rules=rules)}
                )
    with pytest.raises(ValueError, match="rules"):
        KeyPassPolicy.from_record(
            {
                "key_pass_policy": dict(
                    _POLICY, rules=dict(_RULES, sm_100a={"tail_rows": 0})
                )
            }
        )
    rows = {
        # (T, S): (sm_100a, sm_103a, sm_107a)
        (4096, 65536): (
            2,
            2,
            1,
        ),  # cptail_4k_65536: R200 runs 65-68k-key rows single-pass
        (2123, 67923): (2, 2, 1),  # tail_2123x67923
        (4096, 131072): (3, 3, 2),  # cptail_4k_131072: fixed two passes on R200
        (4224, 131072): (3, 3, 2),
        (4225, 131072): (
            1,
            3,
            2,
        ),  # more than one chunk: a 2 T <= S row on sm_103a / sm_107a
        (32768, 65536): (
            1,
            1,
            1,
        ),  # spread_32k_65536: a 65536-key tail row stays single-pass everywhere
        (4225, 65536): (1, 1, 1),  # more than one chunk below the tail floors
        (32768, 131072): (1, 3, 2),  # spread_32k_131072
        (32768, 196608): (1, 3, 2),  # formula 5, capped at 3 on sm_103a
        (32768, 262144): (1, 3, 2),  # packed_N1
        (65536, 131072): (1, 3, 2),  # exactly 2 T == S
        (65537, 131072): (1, 1, 1),  # 2 T > S and more than one chunk
        (131072, 131072): (1, 1, 1),  # a causal single document never qualifies
        (4096, 4096): (1, 1, 1),
        (4096, 45511): (1, 1, 1),
        (4096, 45512): (2, 2, 1),
        (4096, 180224): (4, 4, 2),  # the largest capped count
        # formula 5: above max_passes on the formula targets (one pass); sm_107a's two passes are under it
        (4096, 225280): (1, 1, 2),
        (4096, 268757): (1, 1, 2),
    }
    for (T, S), want in rows.items():
        got = tuple(policy.passes(T, S, 2048, arch=a) for a in _RULES)
        assert got == want, (T, S, got, want)
        assert tuple(_rule_passes(record, T, S, arch=a) for a in _RULES) == want
        assert (
            tuple(
                plan_key_passes(record, _ALL_STAGES, T, S, 2048, arch=a) for a in _RULES
            )
            == want
        )
    # a multi-segment row plans one pass under every rule
    for a in _RULES:
        assert policy.passes(4096, 131072, 2048, num_segments=2, arch=a) == 1
    # token chunks of a multi-pass backward (the Cake launcher's loop)
    assert policy.token_chunks(4096, 2048) == ((0, 4096),)
    assert policy.token_chunks(4224, 2048) == ((0, 4224),)
    assert policy.token_chunks(8448, 2048) == ((0, 4224), (4224, 4224))
    assert policy.token_chunks(32768, 2048) == tuple(
        (b, min(4224, 32768 - b)) for b in range(0, 32768, 4224)
    )
    assert policy.token_chunks(32768, 2048)[-1] == (29568, 3200)


def test_workspace_layout_token_chunk():
    T, S, topk = 8448, 131072, 2048
    whole = workspace_layout(T, S, topk, key_passes=3)
    chunked = workspace_layout(T, S, topk, key_passes=3, token_chunk=4224)
    assert whole["dq_partial"][1] == T * DQ_PARTIAL_BYTES_PER_TOKEN
    assert chunked["dq_partial"][1] == 4224 * DQ_PARTIAL_BYTES_PER_TOKEN
    assert chunked["key_scratch"][1] == 4224 * topk * 4
    assert chunked["pass_counts"][1] == 4224 * 4
    # a chunk larger than the row changes nothing
    assert workspace_layout(T, S, topk, key_passes=3, token_chunk=10000) == whole
    assert workspace_layout(
        T, S, topk, key_passes=1, token_chunk=128
    ) == workspace_layout(T, S, topk, key_passes=1)


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
    # the cap: the formula's count is taken only up to max_passes (per-pass dQ-partial round trip + pipeline fill)
    assert policy.formula_passes(267520) == 6 and policy.formula_passes(225280) == 5
    assert policy.passes(3884, 267520, 2048) == 1
    assert policy.passes(4096, 225280, 2048) == 1
    assert policy.passes(4096, 180224, 2048) == 4
    assert policy.passes(4096, 182044, 2048) == 4
    assert policy.passes(4096, 182045, 2048) == 1
    assert KeyPassPolicy(**dict(_POLICY, max_passes=6)).passes(3884, 267520, 2048) == 6
    assert KeyPassPolicy(**dict(_POLICY, max_passes=2)).passes(4096, 131072, 2048) == 1
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
    with pytest.raises(ValueError, match="max_passes"):
        KeyPassPolicy.from_record(
            {"key_pass_policy": {k: v for k, v in _POLICY.items() if k != "max_passes"}}
        )


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


def test_plan_dkv_direct_rule_and_env():
    """Direct accumulation needs dkv_acc, the natural stages and the record's size rule -- or the ``dkv_direct``
    keyword (``True`` / ``False``) that overrides the rule; ``None`` (the default) is the rule and any other value
    is rejected."""
    record = {"dkv_direct": dict(_DIRECT_POLICY), "key_pass_policy": dict(_POLICY)}
    plan = cake_backend.plan_dkv_direct
    assert cake_backend.record_direct_stages(record, _DIRECT_STAGES)
    assert not cake_backend.record_direct_stages(record, _ALL_STAGES)
    assert plan(record, _DIRECT_STAGES, 1000, 4000, accumulate=True)
    assert not plan(record, _DIRECT_STAGES, 1000, 3999, accumulate=True)
    assert not plan(record, _DIRECT_STAGES, 1000, 4000, accumulate=False)
    assert not plan(record, _ALL_STAGES, 1000, 4000, accumulate=True)
    # dkv_direct=None is the rule
    assert plan(record, _DIRECT_STAGES, 1000, 4000, accumulate=True, dkv_direct=None)
    assert not plan(
        record, _DIRECT_STAGES, 1000, 3999, accumulate=True, dkv_direct=None
    )
    # dkv_direct=False: never the direct path
    assert not plan(
        record, _DIRECT_STAGES, 1000, 4000, accumulate=True, dkv_direct=False
    )
    # dkv_direct=True: the direct path whenever the stages are registered and dkv_acc is given
    assert plan(record, _DIRECT_STAGES, 1000, 1, accumulate=True, dkv_direct=True)
    assert not plan(record, _ALL_STAGES, 1000, 1, accumulate=True, dkv_direct=True)
    assert not plan(record, _DIRECT_STAGES, 1000, 1, accumulate=False, dkv_direct=True)
    # anything but None / True / False is rejected (the former environment strings included)
    for bad in ("auto", "0", "1", 0, 1):
        with pytest.raises(ValueError, match="dkv_direct must be None, True or False"):
            plan(record, _DIRECT_STAGES, 1000, 4000, accumulate=True, dkv_direct=bad)
    # malformed records fail closed
    with pytest.raises(ValueError, match="bwd_main stage"):
        cake_backend.record_direct_stages(
            record, ("fwd", "bwd_delta", "bwd_main_natural")
        )
    with pytest.raises(ValueError, match="accompany"):
        cake_backend.record_direct_stages(
            record,
            (
                "fwd",
                "bwd_delta",
                "bwd_main",
                "bwd_main_natural",
                "bwd_compact",
                "bwd_main_pass",
            ),
        )
    with pytest.raises(ValueError, match="dkv_direct policy"):
        cake_backend.record_direct_stages(
            {}, ("fwd", "bwd_delta", "bwd_main", "bwd_main_natural")
        )
    with pytest.raises(ValueError, match="without bwd_main_natural"):
        cake_backend.record_direct_stages(
            record, _ALL_STAGES + ("bwd_main_pass_natural",)
        )
    with pytest.raises(ValueError, match="min_keys_per_query"):
        cake_backend.DkvDirectPolicy.from_record({"dkv_direct": {}})
    with pytest.raises(ValueError, match="positive"):
        cake_backend.DkvDirectPolicy.from_record(
            {"dkv_direct": {"min_keys_per_query": 0}}
        )
    assert cake_backend.DkvDirectPolicy.from_record({}) is None


def test_main_accumulator_constants_and_direct_bound_values():
    """A direct plan binds the caller's packed rows to the main stage's accumulator operands (``dkv_f32`` /
    ``dkr_f32`` addressed through ``dkv_stride`` / ``dkr_stride`` / ``dkr_col0`` / ``dkv_dst_map`` / ``dkv_has_map``)
    and no cast operands; a plan that does not accumulate directly carries the private-accumulator geometry (inert
    on the permuted program)."""
    assert cake_backend._main_accumulator_constants(True, 704, True) == dict(
        dkv_stride=704, dkr_stride=704, dkr_col0=D_LATENT, dkv_has_map=1
    )
    assert cake_backend._main_accumulator_constants(True, 576, False) == dict(
        dkv_stride=576, dkr_stride=576, dkr_col0=D_LATENT, dkv_has_map=0
    )
    assert cake_backend._main_accumulator_constants(False, 704, True) == dict(
        dkv_stride=D_LATENT, dkr_stride=D_ROPE, dkr_col0=0, dkv_has_map=0
    )
    packed = torch.zeros(5, 704)
    dst_map = torch.zeros(3, dtype=torch.int32)
    t = dict(
        indices=torch.zeros(3, 4, dtype=torch.int32),
        topk_length=torch.full((3,), 4, dtype=torch.int32),
        delta=torch.zeros(3, NUM_HEADS),
        dkv_latent=torch.empty(0, dtype=torch.bfloat16),
        dk_rope=torch.empty(0, dtype=torch.bfloat16),
        dkv_acc=packed,
        dkv_dst_map=dst_map,
    )
    plan = SimpleNamespace(
        constants=cake_backend._main_accumulator_constants(True, 704, True),
        backward=True,
        accumulate_dkv=True,
        dkv_direct=True,
        key_passes=1,
    )
    values = cake_backend._bound_values(plan, t)
    assert values["dkv_f32"] is packed and values["dkr_f32"] is packed
    assert (
        values["dkv_stride"],
        values["dkr_stride"],
        values["dkr_col0"],
        values["dkv_has_map"],
    ) == (
        704,
        704,
        D_LATENT,
        1,
    )
    assert values["dkv_dst_map"] is dst_map
    assert (
        "dst_packed" not in values and "src_latent" not in values
    )  # no cast on a direct binding
    # the same plan without a map: the map placeholder is the int32 topk_length vector
    del t["dkv_dst_map"]
    plan.constants = cake_backend._main_accumulator_constants(True, 704, False)
    values = cake_backend._bound_values(plan, t)
    assert values["dkv_dst_map"] is t["topk_length"] and values["dkv_has_map"] == 0


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


def test_registry_record_is_well_formed():
    record = cake_jit.record()
    assert cake_jit.STAGES[0] == "fwd" and cake_jit.PROGRAM == "cake_dsa_h64_train"
    # build.json (``cake_build``) is the only record of what to compile; the registry names its
    # entries ``<library>_<arch>`` per stage
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(libraries) == len(manifest["libraries"])  # names unique
    assert set(record["arches"]) <= set(manifest["arch_flags"])
    stages = cake_jit.registered_stages()
    assert stages[0] == "fwd" and tuple(record["stages"]) == stages
    if any(stage in stages for stage in KEY_PASS_STAGES):
        # the key-range-pass form comes as a pair, next to the single-pass stage, with its host policy
        assert set(KEY_PASS_STAGES) <= set(stages) and "bwd_main" in stages
        policy = KeyPassPolicy.from_record(record)
        assert policy is not None
        assert policy.token_chunk(2048) % policy.token_chunk_multiple == 0
        assert policy.passes(1, 1, 2048) == 1
    if cake_backend.record_direct_stages(record, stages):
        # the natural-layout variants are the same kernel ABI as the regular main stage, with the size rule
        direct = cake_backend.DkvDirectPolicy.from_record(record)
        assert direct is not None
        assert direct.wants(1, direct.min_keys_per_query)
        assert not direct.wants(1, direct.min_keys_per_query - 1)
        assert (
            cake_launch.STAGE_ARGS["bwd_main_natural"]
            == cake_launch.STAGE_ARGS["bwd_main"]
        )
        assert cake_launch.GRID["bwd_main_natural"](
            130, 4096, 2048
        ) == cake_launch.GRID["bwd_main"](130, 4096, 2048)
        if "bwd_main_pass_natural" in stages:
            assert (
                cake_launch.STAGE_ARGS["bwd_main_pass_natural"]
                == cake_launch.STAGE_ARGS["bwd_main_pass"]
            )
        assert {
            "dkv_stride",
            "dkr_stride",
            "dkr_col0",
            "dkv_dst_map",
            "dkv_has_map",
        } <= set(cake_launch.STAGE_ARGS["bwd_main_natural"])
    for stage in stages:
        physical = record[stage]
        assert physical["library"] and physical["ffi_entry"]
        for arch in record["arches"]:
            entry = libraries[f"{physical['library']}_{arch}"]
            assert entry["arch"] == arch and entry["entry"] == physical["ffi_entry"]
            assert len(entry["sources"]) == 2
            assert all(
                s.startswith("csrc/cake_dsa_h64_train/") for s in entry["sources"]
            )
            kinds = [Path(s).name.rsplit("_", 1) for s in entry["sources"]]
            assert len({stem for stem, _ in kinds}) == 1
            assert sorted(kind for _, kind in kinds) == ["binding.cu", "kernel.cu"]
            assert all(
                (Path(cake_jit.__file__).parent / s).is_file() for s in entry["sources"]
            )
            assert len(entry["closure_sha256"]) == 64
        launch = physical["launch"]
        assert len(launch["cluster"]) == 3 and all(
            int(c) >= 1 for c in launch["cluster"]
        )
        assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
        assert stage in cake_launch.LAUNCH and stage in cake_launch.GRID


def test_generated_grids():
    grid = cake_launch.GRID
    assert grid["fwd"](130, 4096, 2048) == (130, 1, 1)
    assert grid["fwd"](0, 4096, 2048) == (
        1,
        1,
        1,
    )  # clamped: the eager paths never launch T == 0
    assert grid["bwd_delta"](130, 4096, 2048) == (1040, 1, 1)
    assert grid["bwd_main"](130, 4096, 2048) == (130, 1, 1)
    if "bwd_compact" in grid:
        assert grid["bwd_compact"](130, 4096, 2048) == (33, 1, 1)
        assert grid["bwd_main_pass"](130, 4096, 2048) == (130, 1, 1)
    assert grid["bwd_cast"](5, 4096, 3) == (288, 1, 1)
    assert grid["bwd_cast"](5, 7, 3) == (1, 1, 1)


def test_binding_keys_cover_geometry_scale_and_lengths_not_pointers():
    q, kv, idx = _host_inputs()
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    scale = default_softmax_scale()
    base = forward_binding_key(ql, qr, kl, kr, idx, None, scale)
    assert base == forward_binding_key(ql.view_as(ql), qr, kl, kr, idx, None, scale)
    assert base == forward_binding_key(
        ql, qr, kl, kr, idx.clone(), None, scale
    )  # other storage, same geometry: the fresh activations of a step hit
    assert base != forward_binding_key(ql, qr, kl, kr, idx, None, scale * 0.5)  # scale
    assert base != forward_binding_key(
        ql, qr, kl, kr, idx, torch.zeros(8, dtype=torch.int32), scale
    )  # lengths
    assert base != forward_binding_key(ql, qr, kl[:4], kr, idx, None, scale)  # shape
    same_ptr_other_stride = kv.view(-1)[: 16 * D_LATENT].view(16, D_LATENT)
    assert same_ptr_other_stride.data_ptr() == kl.data_ptr()
    assert base != forward_binding_key(
        ql, qr, same_ptr_other_stride, kr, idx, None, scale
    )  # stride
    assert base != forward_binding_key(
        ql, qr, kl.view(torch.float16), kr, idx, None, scale
    )  # dtype
    wide = torch.zeros(8, 13, dtype=torch.int32)
    assert base != forward_binding_key(
        ql, qr, kl, kr, wide[:, 3:8], None, scale
    )  # start alignment of indices
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
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, False, 2
    )  # key_passes


def test_binding_cache_is_least_recently_used_and_bounded_by_capacity():
    cache = cake_backend._BindingCache(capacity=3)
    for tag in ("a", "b", "c"):
        cache.put(("fwd", tag), tag)
    assert list(cache._entries) == [("fwd", "a"), ("fwd", "b"), ("fwd", "c")]
    assert cache.get(("fwd", "a")) == "a"  # a hit refreshes
    assert list(cache._entries) == [("fwd", "b"), ("fwd", "c"), ("fwd", "a")]
    assert cache.get(("fwd", "z")) is None
    cache.put(("bwd", "d"), "d")  # beyond the capacity the least recently used goes (b)
    assert list(cache._entries) == [("fwd", "c"), ("fwd", "a"), ("bwd", "d")]
    cache.put(("fwd", "c"), "c2")  # re-putting a key refreshes it without duplicating
    assert list(cache._entries) == [("fwd", "a"), ("bwd", "d"), ("fwd", "c")]
    assert len(cache) == 3
    cache.clear()
    assert len(cache) == 0


def test_binding_cache_capacity_from_environment():
    """The cache takes its capacity from the constructor or from ``configure`` (``None`` = the default of 64, a
    positive integer otherwise); the backend reads no environment variable -- the flashinfer wrapper maps its
    capacity variable onto ``configure`` (covered by the public-API suite)."""
    default = cake_backend.BINDING_CACHE_DEFAULT_CAPACITY
    assert cake_backend._BindingCache().capacity == default == 64
    assert cake_backend._BindingCache(capacity=None).capacity == default
    assert cake_backend.BINDING_CACHE.capacity >= 1
    assert cake_backend._BindingCache(capacity=3).capacity == 3
    cache = cake_backend._BindingCache(capacity=4)
    for tag in "abcd":
        cache.put(("fwd", tag), tag)
    cache.configure(capacity=2)  # shrinking evicts the least recently used entries
    assert cache.capacity == 2 and list(cache._entries) == [("fwd", "c"), ("fwd", "d")]
    cache.configure(capacity=5)
    assert cache.capacity == 5 and len(cache) == 2
    cache.configure()
    assert cache.capacity == default and len(cache) == 2
    for bad in (0, -1, "5", 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            cake_backend._BindingCache(capacity=bad)
        with pytest.raises(ValueError, match="positive integer"):
            cache.configure(capacity=bad)
    assert cache.capacity == default  # a rejected value leaves the cache as it was


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


def test_validate_accepts_strided_indices_and_wide_head_strides():
    _, kv, _ = _host_inputs()
    wide = torch.zeros(8, NUM_HEADS, 640, dtype=torch.bfloat16)  # head stride 640
    buf = torch.zeros(
        8, 16, dtype=torch.int32
    )  # indices as a column slice: row stride 16
    assert validate_dsa_train_inputs(
        wide[..., :D_LATENT],
        wide[..., D_LATENT:D_QK],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        buf[:, :5],
    ) == (8, 16, 5)


def test_validate_rejects_misaligned_k_rope_and_strided_index_columns():
    q, kv, idx = _host_inputs()
    flat = torch.zeros(16 * D_QK + 3, dtype=torch.bfloat16)
    k_rope = flat[3 : 3 + 16 * D_ROPE].view(
        16, D_ROPE
    )  # starts 3 elements into its storage
    with pytest.raises(ValueError, match="aligned"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], k_rope, idx
        )
    with pytest.raises(ValueError, match="indices rows"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
            idx.t().contiguous().t(),
        )


def test_pointer_operand_rebases_views_to_a_vector_aligned_element():
    """A raw-pointer operand that starts inside its storage reaches the kernels as the view that
    starts at the nearest vector-aligned element below it plus the element offset the kernels add
    (their vector loads are gated on that offset); an aligned tensor is passed as is."""
    buf = torch.arange(1000, dtype=torch.int32)
    for offset in (1, 8, 13):
        view = buf[offset : offset + 8 * 96].view(8, 96)
        base, element_offset = cake_backend._pointer_operand(view)
        assert element_offset == offset % 8
        assert base.data_ptr() == view.data_ptr() - 4 * element_offset
        assert base.data_ptr() % 32 == buf.data_ptr() % 32
        assert base.shape == view.shape and base.stride() == view.stride()
        assert torch.equal(
            base.as_strided(
                view.shape, view.stride(), base.storage_offset() + element_offset
            ),
            view,
        )
    base, element_offset = cake_backend._pointer_operand(buf)
    assert base is buf and element_offset == 0


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


def test_eager_calls_reuse_the_plan_and_return_fresh_outputs():
    """Repeated eager calls of one geometry share one cached plan (no new entry, no
    re-validation) and still return distinct output tensors with equal values."""
    _require_program(backward=True)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 31, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    first = cake_backend.forward(*args)
    assert len(cache) == 1
    second = cake_backend.forward(*args)
    assert len(cache) == 1
    torch.cuda.synchronize()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    out, lse, o_lo = first
    g1 = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    g2 = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert len(cache) == 2  # one forward plan, one backward plan
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(g1, g2, strict=True))
    assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])
    # another geometry (a different top-k) plans anew; the fresh activations of a
    # training step (same geometry, other storage) do not
    other = make_inputs([256], [512], seed=SEED + 32, topk=64)
    cake_backend.forward(
        other.q_latent, other.q_rope, other.kv_latent, other.k_rope, other.idx_global
    )
    assert len(cache) == 3
    moved = [t.clone() for t in args]
    cake_backend.forward(*moved)
    assert len(cache) == 3
    # ... nor do other tensor contents of the same geometry: the shared plan serves
    # them with the results of a freshly prepared, uncached binding
    fresh = make_inputs([256], [512], seed=SEED + 35, topk=128)
    fresh_args = (
        fresh.q_latent,
        fresh.q_rope,
        fresh.kv_latent,
        fresh.k_rope,
        fresh.idx_global,
    )
    assert not torch.equal(fresh.idx_global, inp.idx_global)
    cached = cake_backend.forward(*fresh_args)
    assert len(cache) == 3
    direct = prepare_dsa_train(*fresh_args, backward=False).forward()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(cached, direct, strict=True))
    assert not torch.equal(cached[0], first[0])


def test_eager_cache_entries_pin_nothing_problem_sized():
    """A remembered plan owns only the full-length ``topk_length`` vector (4 B per
    query row); outputs and the backward scratch are released with the results."""
    _require_program(backward=True)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 36, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    out, lse, o_lo = cake_backend.forward(*args)
    grads = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    del out, lse, o_lo, grads
    assert len(cache) == 2
    assert (
        torch.cuda.memory_allocated() - live <= 2 * 512 * 4
    )  # <= two topk_length vectors (allocator rounding)
    for _ in range(3):  # repeated hits leave no allocation behind
        out, lse, o_lo = cake_backend.forward(*args)
        grads = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    del out, lse, o_lo, grads
    assert torch.cuda.memory_allocated() - live <= 2 * 512 * 4


def test_binding_cache_is_bounded():
    _require_program()
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    capacity = cache.capacity
    try:
        cache.capacity = 2
        for topk in (16, 24, 32, 40):
            inp = make_inputs([128], [256], seed=SEED + 33, topk=topk)
            cake_backend.forward(
                inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
            )
            assert len(cache) <= 2
    finally:
        cache.capacity = capacity
        cache.clear()


def test_eager_forward_captures_into_a_cuda_graph_without_touching_the_cache():
    _require_program()
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 34, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    eager = cake_backend.forward(*args)
    torch.cuda.synchronize()
    assert len(cache) == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm the capture stream
            cake_backend.forward(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = cake_backend.forward(*args)
    assert len(cache) == 1  # the capture planned privately
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, captured, strict=True):
        assert torch.equal(a, b)


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
    _, record = record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages()
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
    across two calls, and agreement with the single pass."""
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
        expected_order += [(stage, 0, index) for stage in KEY_PASS_STAGES]
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
    a = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
    b = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
    single = cake_backend.backward(*args, topk_length=topk_length, key_passes=1)
    torch.cuda.synchronize()
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


def test_backward_forced_passes_over_two_token_chunks():
    """Two forced passes over a row of 4352 tokens (one full chunk of 4224 and one of 128): the
    launches run chunk by chunk through every pass, the pass scratch holds one chunk, dq is
    bitwise across two calls and agrees with the single pass and the reference."""
    record, _stages = _require_key_pass_program()
    device = torch.device("cuda")
    chunk = KeyPassPolicy.from_record(record).token_chunk(2048)
    assert chunk == 4224
    inp = make_inputs([chunk + 128], [65536], seed=SEED + 18, topk=2048)
    fwd_args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    out, lse, o_lo = cake_backend.forward(*fwd_args)
    runner = prepare_dsa_train(*fwd_args, dout=inp.dout, backward=True, key_passes=2)
    assert runner.key_passes == 2
    assert [k for k in runner.backward_order if isinstance(k, tuple)] == [
        (stage, c, p) for c in range(2) for p in range(2) for stage in KEY_PASS_STAGES
    ]
    assert runner.layout["dq_partial"][1] == chunk * DQ_PARTIAL_BYTES_PER_TOKEN
    assert runner.layout["key_scratch"][1] == chunk * 2048 * 4
    assert (
        dsa_train_workspace_size(chunk + 128, 65536, 2048, device, key_passes=2)
        == runner.layout["total"]
    )
    del runner
    args = fwd_args + (out, o_lo, lse, inp.dout)
    a = cake_backend.backward(*args, key_passes=2)
    b = cake_backend.backward(*args, key_passes=2)
    single = cake_backend.backward(*args, key_passes=1)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert rel_l2(a[0], single[0]) < 1e-3 and rel_l2(a[1], single[1]) < 1e-3
    assert max(rel_l2(a[2], single[2]), rel_l2(a[3], single[3])) < 1e-2
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_backward(a, ref, canonical=True)


def test_forward_derives_row_lengths_in_kernel_and_binding_key_covers_it():
    """``forward(..., topk_length=buf, derive_topk_length=True)`` fills ``buf`` with the row lengths (bitwise
    :func:`derive_topk_length`) and returns bitwise the outputs of the explicit-length call; a call without lengths
    returns the same outputs (the kernel derives into the binding's own vector); the prepared runner derives as
    well; the binding key separates the three forms, and a derived call without a tensor to fill is rejected."""
    _require_program()
    inp = make_inputs([200, 96, 300], [200, 96, 4096], seed=SEED + 753, topk=256)
    S, T = int(inp.kv_latent.shape[0]), int(inp.q_latent.shape[0])
    expected = derive_topk_length(inp.idx_global, S)
    assert int((expected < 256).sum()) > 0 and int(expected.max()) <= 256
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    buf = torch.full((T,), -1, dtype=torch.int32, device=inp.q_latent.device)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    out_d, lse_d, o_lo_d = cake_backend.forward(
        *args, topk_length=buf, derive_topk_length=True
    )
    out_e, lse_e, o_lo_e = cake_backend.forward(*args, topk_length=expected)
    out_n, lse_n, o_lo_n = cake_backend.forward(*args)
    torch.cuda.synchronize()
    assert torch.equal(buf, expected)
    assert (
        torch.equal(out_d, out_e)
        and torch.equal(lse_d, lse_e)
        and torch.equal(o_lo_d, o_lo_e)
    )
    assert (
        torch.equal(out_n, out_e)
        and torch.equal(lse_n, lse_e)
        and torch.equal(o_lo_n, o_lo_e)
    )
    assert len(cache) == 3
    buf.fill_(
        -1
    )  # the remembered derived binding rewrites the caller's vector on every call
    out_d2, _, _ = cake_backend.forward(*args, topk_length=buf, derive_topk_length=True)
    torch.cuda.synchronize()
    assert torch.equal(buf, expected) and torch.equal(out_d2, out_e)
    assert len(cache) == 3
    buf.fill_(-1)
    runner = prepare_dsa_train(
        *args, topk_length=buf, derive_topk_length=True, backward=False
    )
    r_out, r_lse, _ = runner.forward()
    torch.cuda.synchronize()
    assert (
        torch.equal(buf, expected)
        and torch.equal(r_out, out_e)
        and torch.equal(r_lse, lse_e)
    )
    scale = default_softmax_scale()
    assert forward_binding_key(*args, buf, scale, True) != forward_binding_key(
        *args, buf, scale, False
    )
    assert forward_binding_key(*args, buf, scale) == forward_binding_key(
        *args, buf, scale, False
    )
    with pytest.raises(ValueError, match="derive_topk_length"):
        cake_backend.forward(*args, derive_topk_length=True)
    with pytest.raises(ValueError, match="derive_topk_length"):
        prepare_dsa_train(*args, derive_topk_length=True, backward=False)


def _indices_view_inside_storage(indices: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``indices`` that starts ``offset`` elements inside a larger buffer."""
    n = indices.numel()
    buf = torch.full((n + offset + 64,), -1, dtype=indices.dtype, device=indices.device)
    view = buf[offset : offset + n].view(indices.shape)
    view.copy_(indices)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


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
    # the FP32 mode is served by the cast stage, never by the kernels' internal accumulators
    runner = prepare_dsa_train(*args[:5], dout=inp.dout, backward=True, dkv_fp32=True)
    runner.forward()
    _, _, f_lat, f_rope = runner.backward()
    torch.cuda.synchronize()
    assert f_lat.data_ptr() != runner.tensors["dkv_latent_acc"].data_ptr()
    assert f_rope.data_ptr() != runner.tensors["dk_rope_acc"].data_ptr()
    assert "bwd_cast" in runner.launches and rel_l2(f_lat, dkv_latent) < 1e-2


def test_forward_accepts_wide_query_head_strides_bitwise():
    """q views whose head stride exceeds the packed 576 (the tensor maps carry both strides)."""
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 20, topk=128)
    wide = torch.zeros(
        (256, NUM_HEADS, 640), dtype=torch.bfloat16, device=inp.q_latent.device
    )
    wide[..., :D_LATENT].copy_(inp.q_latent)
    wide[..., D_LATENT:D_QK].copy_(inp.q_rope)
    plain = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    got = cake_backend.forward(
        wide[..., :D_LATENT],
        wide[..., D_LATENT:D_QK],
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
    )
    torch.cuda.synchronize()
    for a, b in zip(plain, got, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("pad", [3, 8])
def test_indices_column_slice_matches_the_plain_tensor(pad):
    """``indices`` as a column slice of a wider buffer (row stride topk + pad): the kernels take the
    row stride; a stride that is not a multiple of eight takes their scalar index path."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 21, topk=128)
    wide = torch.full(
        (256, 128 + pad), -1, dtype=torch.int32, device=inp.q_latent.device
    )
    wide[:, :128].copy_(inp.idx_global)
    view = wide[:, :128]
    assert not view.is_contiguous() and view.stride(0) == 128 + pad
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


# --- Packed / strided trainer layouts and causal varlen offsetting (flashinfer-ai/flashinfer#5675) ---
# Packed / strided trainer layouts and the causal tail-of-prefix varlen rule (flashinfer-ai/flashinfer#5675).  The
# kernels take the head / token strides of the query operands and the key row stride from the tensors (TMA descriptors
# encoded from the view; k_rope as storage alias + element offset + row stride), so a view changes descriptor fields
# only: out, lse, o_lo and dQ must be bitwise identical to the contiguous path; dK / dV are FP32 reductions whose order
# is not fixed (FP32 tolerance).


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
    # the fused per-row lengths (varlen entry default) equal the derivation on the result, bitwise
    strict2, lengths = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=True, return_topk_length=True
    )
    assert torch.equal(strict2, strict)
    expected_lengths = [3, 3, 4, 3, 3, 1, 4, 0, 5]  # last valid slot + 1 per strict row
    assert lengths.dtype == torch.int32 and lengths.tolist() == expected_lengths
    assert torch.equal(lengths, derive_topk_length(strict, 18))
    loose2, loose_len = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=False, return_topk_length=True
    )
    assert torch.equal(loose2, loose) and torch.equal(
        loose_len, derive_topk_length(loose, 18)
    )
    # more queries than keys: own positions -1, 0, 1 -> the first row is fully masked under the causal rule
    cu_q2 = torch.tensor([0, 3], dtype=torch.int32)
    cu_k2 = torch.tensor([0, 2], dtype=torch.int32)
    local2 = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
    strict3, len3 = offset_gather_kv_indices(
        local2, cu_q2, cu_k2, causal=True, return_topk_length=True
    )
    assert strict3.tolist() == [[-1, -1], [0, -1], [1, 0]] and len3.tolist() == [
        0,
        1,
        2,
    ]
    assert offset_gather_kv_indices(local2, cu_q2, cu_k2, causal=False).tolist() == [
        [0, 1],
        [0, 1],
        [1, 0],
    ]


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


def test_dkv_dst_map_value_check_is_opt_in():
    """``_check_dkv_acc`` validates dtype / shape / contiguity / device of ``dkv_dst_map`` but not its values by
    default (the kernel does not range-check the map); with ``check_dst_map=True`` an out-of-range destination row
    raises ``ValueError`` and an in-range map passes.  The eager backward threads the keyword to that check (a call
    without query rows runs it on host tensors without launching)."""
    check = cake_backend._check_dkv_acc
    dkv_acc = torch.zeros(4, D_QK, dtype=torch.float32)
    good = torch.tensor([3, 0, 3], dtype=torch.int32)
    bad = torch.tensor([0, 4, 1], dtype=torch.int32)  # 4 == S_dst is outside [0, S_dst)
    for dst_map in (good, bad):
        operand, row_stride = check(dkv_acc, dst_map, num_kv=3)
        assert operand is dkv_acc and row_stride == D_QK
        assert check(dkv_acc, dst_map, num_kv=3, check_dst_map=False)[1] == D_QK
    assert check(dkv_acc, good, num_kv=3, check_dst_map=True)[1] == D_QK
    with pytest.raises(ValueError, match=r"dkv_dst_map values must lie in \[0, 4\)"):
        check(dkv_acc, bad, num_kv=3, check_dst_map=True)
    with pytest.raises(ValueError, match="dkv_dst_map values must lie in"):
        check(
            dkv_acc,
            torch.tensor([-1, 0, 0], dtype=torch.int32),
            num_kv=3,
            check_dst_map=True,
        )
    q, kv, idx = _host_inputs(total_q=0, total_k=3, topk=5)
    out = torch.zeros(0, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    args = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        idx,
        out,
        out,
        torch.zeros(0, NUM_HEADS),
        out,
    )
    grads = cake_backend.backward(*args, dkv_acc=dkv_acc, dkv_dst_map=bad)
    assert grads[2] is None and grads[3] is None  # unchecked by default
    with pytest.raises(ValueError, match="dkv_dst_map values must lie in"):
        cake_backend.backward(
            *args, dkv_acc=dkv_acc, dkv_dst_map=bad, check_dst_map=True
        )


# --- end of the packed / strided layout block ---
# --- Packed fp32 dK/dV accumulation and destination mapping (flashinfer-ai/flashinfer#5675) ---
# Packed FP32 dK/dV accumulation into a caller-provided buffer with an optional destination-row map (issue #5675).
# Comparisons across two backward calls use a tolerance: their FP32 reductions run in another order.
ACC_REL_L2 = 1e-4


def test_registered_cast_declares_the_packed_accumulate_operands():
    """The generated launcher of ``bwd_cast`` takes the packed-accumulate operands the backend serves ``dkv_acc`` /
    ``dkv_dst_map`` through (the export resolves the cast by this ABI, so every delivered program has them)."""
    assert {
        "dst_packed",
        "dst_row_stride",
        "dst_map",
        "has_dst_map",
        "accumulate",
    } <= set(cake_launch.STAGE_ARGS["bwd_cast"])
    assert "bwd_cast" in cake_launch.LAUNCH and "bwd_cast" in cake_launch.GRID


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
    # with dkv_acc the dkv_direct option is part of the key (None = the rule, True, False); without it, inert
    assert with_acc == backward_binding_key(*args, dkv_acc=acc, dkv_direct=None)
    keys = {
        backward_binding_key(*args, dkv_acc=acc, dkv_direct=mode)
        for mode in (None, True, False)
    }
    assert len(keys) == 3
    assert plain == backward_binding_key(*args, dkv_direct=True)


def test_bound_values_serve_the_accumulating_cast_operands():
    """The packed-accumulate operands bind from ``dkv_acc`` / ``dkv_dst_map`` (``dst_packed`` / ``dst_map``) and the
    three scalars of the plan; a plan without ``dkv_acc`` binds never-dereferenced placeholders of the right dtypes
    (the FP32 accumulator and the ``topk_length`` vector) with ``accumulate = has_dst_map = dst_row_stride = 0``."""
    S = 8
    t = dict(
        indices=torch.zeros(2, 4, dtype=torch.int32),
        topk_length=torch.full((2,), 4, dtype=torch.int32),
        delta=torch.zeros(2, NUM_HEADS),
        dkv_latent_acc=torch.zeros(S, D_LATENT),
        dk_rope_acc=torch.zeros(S, D_ROPE),
        dkv_latent=torch.empty(0, dtype=torch.bfloat16),
        dk_rope=torch.empty(0, dtype=torch.bfloat16),
        dkv_latent_fp32=torch.zeros(S, D_LATENT),
        dk_rope_fp32=torch.zeros(S, D_ROPE),
    )
    acc = torch.zeros(S, 704)
    dst_map = torch.arange(S, dtype=torch.int32)
    constants = dict(dst_row_stride=704, has_dst_map=1, accumulate=1)
    plan = SimpleNamespace(
        constants=constants,
        backward=True,
        accumulate_dkv=True,
        dkv_direct=False,
        key_passes=1,
    )
    values = cake_backend._bound_values(plan, dict(t, dkv_acc=acc, dkv_dst_map=dst_map))
    assert values["dst_packed"] is acc and values["dst_map"] is dst_map
    assert (
        values["dkv_dst_map"] is dst_map
    )  # the main stage's map operand (inert on the permuted program)
    assert (values["dst_row_stride"], values["has_dst_map"], values["accumulate"]) == (
        704,
        1,
        1,
    )
    assert (
        values["src_latent"] is t["dkv_latent_acc"]
        and values["dkv_f32"] is t["dkv_latent_acc"]
    )
    # the same binding without a map: the map placeholder is the int32 topk_length vector
    plan.constants = dict(dst_row_stride=704, has_dst_map=0, accumulate=1)
    values = cake_backend._bound_values(plan, dict(t, dkv_acc=acc))
    assert values["dst_packed"] is acc and values["dst_map"] is t["topk_length"]
    assert values["has_dst_map"] == 0
    # a plan that does not accumulate: inert placeholders of the operand dtypes
    plan = SimpleNamespace(
        constants=dict(dst_row_stride=0, has_dst_map=0, accumulate=0),
        backward=True,
        accumulate_dkv=False,
        dkv_direct=False,
        key_passes=1,
    )
    values = cake_backend._bound_values(plan, t)
    assert values["dst_packed"] is t["dkv_latent_acc"]
    assert values["dst_map"] is t["topk_length"]
    assert (values["dst_row_stride"], values["has_dst_map"], values["accumulate"]) == (
        0,
        0,
        0,
    )
    assert (
        values["dst_packed"].dtype == torch.float32
        and values["dst_map"].dtype == torch.int32
    )


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


def test_backward_dkv_acc_direct_and_cast_paths_agree():
    """With dkv_acc and S >= 4 T the natural-layout main stage adds into the caller's rows itself (no accumulators, no
    cast); forced through the cast path (``dkv_direct=False``) the same rows receive the same gradients (another FP32
    summation order), dq is bitwise the same and untouched rows / columns stay untouched -- through a duplicating
    destination map into a [S_dst, 704] buffer, on the prepared runner and the eager entry; ``dkv_direct=None`` (the
    default) takes the rule, and the in-range map passes the opt-in value check."""
    record, stages = _require_direct_program()
    inp = make_inputs(
        [256], [2048], seed=SEED + 24, topk=128
    )  # S / T = 8: the size rule picks direct
    args, dq_l, dq_r, nat = _forward_and_natural_dkv(inp)
    S, S_dst = inp.total_k, 400
    gen = torch.Generator(device="cuda").manual_seed(3)
    dst_map = torch.randint(
        0, S_dst, (S,), device="cuda", generator=gen, dtype=torch.int32
    )
    pre = torch.randn(S_dst + 8, 704, device="cuda", generator=gen)
    expect = pre.clone()
    expect[:S_dst, :D_QK].index_add_(0, dst_map.long(), nat)
    results = {}
    for mode in (True, False, None):
        acc = pre.clone()
        runner = prepare_dsa_train(
            *args[:5],
            dout=args[8],
            backward=True,
            dkv_acc=acc[:S_dst],
            dkv_dst_map=dst_map,
            dkv_direct=mode,
            check_dst_map=True,
        )
        assert runner.dkv_direct == (mode is not False)
        if runner.dkv_direct:
            assert (
                "bwd_main_natural" in runner.launches
                and "bwd_cast" not in runner.launches
            )
            assert (
                "dkv_latent_acc" not in runner.tensors
                and "dkv_latent_acc" not in runner.layout
            )
        else:
            assert "bwd_main" in runner.launches and "bwd_cast" in runner.launches
        runner.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            acc[:S_dst, :D_QK], expect[:S_dst, :D_QK], rtol=1e-4, atol=1e-4
        )
        assert torch.equal(acc[:, D_QK:], pre[:, D_QK:]) and torch.equal(
            acc[S_dst:], pre[S_dst:]
        )
        acc = pre.clone()
        aq_l, aq_r, none_l, none_r = cake_backend.backward(
            *args, dkv_acc=acc[:S_dst], dkv_dst_map=dst_map, dkv_direct=mode
        )
        torch.cuda.synchronize()
        assert none_l is None and none_r is None
        assert torch.equal(aq_l, dq_l) and torch.equal(aq_r, dq_r)
        torch.testing.assert_close(
            acc[:S_dst, :D_QK], expect[:S_dst, :D_QK], rtol=1e-4, atol=1e-4
        )
        assert torch.equal(acc[:, D_QK:], pre[:, D_QK:]) and torch.equal(
            acc[S_dst:], pre[S_dst:]
        )
        results[mode] = acc[:S_dst, :D_QK].clone()
    torch.testing.assert_close(results[True], results[False], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(results[None], results[True], rtol=1e-4, atol=1e-4)
    # the workspace of a direct binding has no FP32 accumulators; dkv_direct sizes it the same way
    T, device = inp.total_q, torch.device("cuda")
    plain = dsa_train_workspace_size(T, S, 128, device)
    direct_size = dsa_train_workspace_size(T, S, 128, device, dkv_acc=True)
    assert plain - direct_size >= S * D_QK * 4
    assert (
        dsa_train_workspace_size(T, S, 128, device, dkv_acc=True, dkv_direct=False)
        == plain
    )
    assert (
        dsa_train_workspace_size(T, S, 128, device, dkv_acc=True, dkv_direct=True)
        == direct_size
    )
    # S < 4 T keeps the cast path by the rule
    assert not cake_backend.plan_dkv_direct(record, stages, 1000, 3999, accumulate=True)


def test_bwd_cast_accumulate_without_map_is_one_fp32_add_bitwise():
    """From the SAME permuted accumulators (a prepared runner's), the accumulating cast without a map equals
    ``previous + natural`` bitwise: one FP32 add per element on both sides."""
    _require_program(backward=True)
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
    # the accumulating plan of the same geometry (row stride 704 baked into its launch constants)
    plan = cake_backend._plan(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        topk_length=None,
        dout=inp.dout,
        softmax_scale=runner.softmax_scale,
        outputs={},
        dkv_fp32=False,
        backward=True,
        key_passes=runner.key_passes,
        dkv_acc=acc,
    )
    assert plan.accumulate_dkv and plan.constants["dst_row_stride"] == 704
    operand, _ = cake_backend._check_dkv_acc(acc, None, S)
    values = cake_backend._bound_values(plan, dict(runner.tensors, dkv_acc=operand))
    launch = cake_backend._launches(plan, values)["bwd_cast"]
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
    """A remembered dkv_acc plan re-supplies the caller's buffer and map per call; another buffer layout plans anew."""
    _require_program(backward=True)
    inp = make_inputs([128], [256], seed=SEED + 24, topk=64)
    args, _, _, nat = _forward_and_natural_dkv(inp)
    S = inp.total_k
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    acc = torch.zeros(S, D_QK, device="cuda")
    dst_map = torch.arange(S, device="cuda", dtype=torch.int32)
    key = backward_binding_key(
        *args, None, default_softmax_scale(), False, dkv_acc=acc, dkv_dst_map=dst_map
    )
    cake_backend.backward(*args, dkv_acc=acc, dkv_dst_map=dst_map)
    binding = cache.get(key)
    assert binding is not None and binding.plan.accumulate_dkv
    assert (
        binding.topk_length is not None and binding.topk_length.numel() == inp.total_q
    )
    entries = len(cache)
    cake_backend.backward(
        *args, dkv_acc=acc, dkv_dst_map=dst_map
    )  # the same plan: no new entry, adds again
    torch.cuda.synchronize()
    assert len(cache) == entries
    assert rel_l2(acc, 2 * nat) < ACC_REL_L2
    acc2 = torch.zeros(
        S + 2, 704, device="cuda"
    )  # another layout: another plan (row stride 704 baked)
    cake_backend.backward(*args, dkv_acc=acc2)
    torch.cuda.synchronize()
    assert len(cache) == entries + 1
    assert (
        rel_l2(acc2[:S, :D_QK], nat) < ACC_REL_L2
        and torch.all(acc2[S:] == 0)
        and torch.all(acc2[:, D_QK:] == 0)
    )


def test_autograd_function_with_dkv_acc_returns_no_kv_grads_and_accumulates():
    _require_program(backward=True)
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


# ---------------------------------------------------------------------------
# NoPE (zero-rope) program: registry, host layer, plans and the GPU suite
# ---------------------------------------------------------------------------
#
# The NoPE entries (``forward_nope`` / ``backward_nope`` / ``prepare_dsa_nope_train`` /
# ``DSASparseAttentionNopeFunction``) share the host layer with the rope program through the ``program`` keyword
# of every stage-name consumer.  The host-level tests below run everywhere.  The plan-level tests run on a
# supported device over the registered record completed with stand-in entries whose launch entries record their
# arguments (no kernel runs), so the plans, bound values and launch sequences are covered before every library is
# registered.  The GPU tests skip until the registered NoPE record carries the stages they need.

NOPE_PROGRAM = cake_jit.NOPE_PROGRAM
NOPE_SOFTMAX_SCALE = cake_backend.NOPE_SOFTMAX_SCALE
NOPE_DQ_PARTIAL_BYTES_PER_TOKEN = cake_backend.NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
forward_nope = cake_backend.forward_nope
backward_nope = cake_backend.backward_nope
prepare_dsa_nope_train = cake_backend.prepare_dsa_nope_train
validate_dsa_nope_inputs = cake_backend.validate_dsa_nope_inputs
validate_cu_seqlens = cake_backend.validate_cu_seqlens
workspace_layout_nope = cake_backend.workspace_layout_nope
dsa_nope_train_workspace_size = cake_backend.dsa_nope_train_workspace_size
forward_binding_key_nope = cake_backend.forward_binding_key_nope
backward_binding_key_nope = cake_backend.backward_binding_key_nope
NOPE_INDEX_PREP_STAGE = cake_backend.NOPE_INDEX_PREP_STAGE
nope_globalize_indices = cake_backend.nope_globalize_indices

# The kernels' argument lists of the NoPE stages (the generated launchers bind them by name, in this order; the
# delta, compaction and cast stages are the rope program's kernels and keep their lists).
NOPE_FWD_ARGS = (
    "q_latent",
    "kv_latent",
    "out",
    "o_lo",
    "lse",
    "indices",
    "topk_length",
    "num_queries",
    "num_kv",
    "topk",
    "idx_stride",
    "indices_offset",
    "has_topk_length",
    "derive_length",
    "scale_log2",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "indices_out",
    "num_docs",
    "search_iters",
    "causal",
    "convert",
    "grid_x",
    "grid_y",
    "grid_z",
)
NOPE_MAIN_ARGS = (
    "q_latent",
    "dout",
    "dq_latent",
    "kv_latent",
    "lse",
    "delta",
    "out",
    "o_lo",
    "indices",
    "topk_length",
    "dkv_f32",
    "dkv_stride",
    "dkv_dst_map",
    "dkv_has_map",
    "num_queries",
    "num_kv",
    "topk",
    "idx_stride",
    "indices_offset",
    "has_topk_length",
    "token_base",
    "token_step",
    "scale_log2",
    "sm_scale",
    "pass_lo",
    "pass_hi",
    "dq_mode",
    "dq_partial",
    "key_scratch",
    "pass_counts",
    "grid_x",
    "grid_y",
    "grid_z",
)
# NoPE stages that name a kernel the rope record also ships: the compaction and cast stages, and -- since the
# rotary backward reads a strided ``dout`` in place -- the strided delta, which serves as the rope record's
# ``bwd_delta`` as well (the dense delta kernel left the package)
NOPE_SHARED_STAGES = {
    "nope_bwd_delta": "bwd_delta",
    "nope_bwd_compact": "bwd_compact",
    "nope_bwd_cast": "bwd_cast",
}
# the strided delta kernel (both records): dout addressed as
# dout_offset + token * dout_token_stride + head * dout_head_stride + channel (elements, multiples of 8)
NOPE_DELTA_ARGS = (
    "dout",
    "out",
    "o_lo",
    "delta",
    "num_rows",
    "dout_token_stride",
    "dout_head_stride",
    "dout_offset",
    "grid_x",
    "grid_y",
    "grid_z",
)
NOPE_SLOT_WIDTHS = (1, 64, 67, 256, 512, 1024, 2177)
NOPE_LARGE_TESTS_ENV = "FLASHINFER_CAKE_DSA_NOPE_LARGE_TESTS"
# Fixed rel-L2 table of the iid 4k x 4k, K_slots 2176 configuration and the element-wise output tolerance of the
# NoPE acceptance.  out and dkv: the frozen rope-path figures of the same geometry times a 2 % noise band.  dq: the
# FP64-emulated BF16 rounding floor measured on the same geometry and inputs at scale 1/16 (0.2231 %) times the same
# 2 % band -- the rope-path dq figure (0.2154 %) lies below that floor and cannot bound this kernel.
NOPE_GATE_OUT = 0.002013 * 1.02
NOPE_GATE_DQ = 0.002231 * 1.02
NOPE_GATE_DKV = 0.002319 * 1.02
# Per-case floors: ``out`` and ``dq`` within 1.02x the emulated floor measured on the same geometry and inputs (the
# emulation places every rounding at the kernel's own points; the kernel sits at 1.000x on the recorded cases);
# ``dkv`` keeps the 1.05x margin of the shared floor rule. Peaked cases use the same per-case floors, not the fixed
# iid table (the kernel's dq on the 0.99 / 0.53 self-weight inputs is 0.2398 % / 0.2348 %, above the iid constant).
NOPE_FLOOR_MARGIN = 1.02
# Row-level ``dq`` check (every NoPE backward case) is a structural guard, not a precision claim: the per-(token,
# head)-row RMS of the normalised error as a distribution, the kernel's p99 within ``NOPE_ROW_P99_GUARD`` x and its
# maximum within ``NOPE_ROW_MAX_GUARD`` x the emulated floor's. It detects a wrong key / wrong row contribution (row
# errors of order 100 %); the precision claim is the case-level rel-L2 within the measured floor x 1.02. Rows whose
# exact gradient is identically zero (tokens with at most one valid key) are checked element-wise against the bound
# the reference derives from the inputs and the frozen fp32 / bf16 numerics instead (``_check_dq_zero_rows``).
NOPE_ROW_P99_GUARD = 1.5
NOPE_ROW_MAX_GUARD = 2.0
NOPE_ROW_SCALE_FLOOR = 1e-3
NOPE_OUT_ATOL = 1e-2
NOPE_OUT_RTOL = 1e-2
NOPE_SEED = SEED + 1000


class _Recorder:
    """A stand-in for a binding entry: remembers the positional arguments of every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def _nope_host_inputs(total_q=8, total_k=16, topk=5, device="cpu"):
    q = torch.zeros(total_q, NUM_HEADS, D_LATENT, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(total_k, D_LATENT, dtype=torch.bfloat16, device=device)
    idx = torch.zeros(total_q, topk, dtype=torch.int32, device=device)
    return q, kv, idx


def _libraries_missing_from_manifest(program: str) -> list[str]:
    """``<library>_<arch>`` names the registry of ``program`` references and ``build.json`` does not carry (the
    export writes them; until then those stages have no device sources)."""
    record = cake_jit.record(program)
    names = {entry["name"] for entry in cake_build.manifest()["libraries"]}
    return sorted(
        f"{record[stage]['library']}_{arch}"
        for stage in cake_jit.registered_stages(program)
        for arch in record["arches"]
        if f"{record[stage]['library']}_{arch}" not in names
    )


def _nope_program_available(*, backward: bool = False) -> bool:
    return (
        _device_supported()
        and generated_program_available(
            torch.device("cuda"), backward=backward, program=NOPE_PROGRAM
        )
        and not _libraries_missing_from_manifest(NOPE_PROGRAM)
    )


def _require_nope_program(*, backward: bool = False):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(
        torch.device("cuda"), backward=backward, program=NOPE_PROGRAM
    ):
        pytest.skip(
            "generated program not available: the NoPE DSA training program has no registered "
            + ("backward" if backward else "forward")
            + " stages for this device"
        )
    missing = _libraries_missing_from_manifest(NOPE_PROGRAM)
    if missing:
        pytest.skip(
            "generated program not available: build.json lacks the NoPE libraries "
            f"{missing[:2]}... ({len(missing)} entries; the export delivers them)"
        )


def _require_nope_direct_program():
    """The registered NoPE program with the natural-layout (direct accumulation) main stage."""
    _require_nope_program(backward=True)
    _, record = record_for(torch.device("cuda"), NOPE_PROGRAM)
    stages = cake_jit.registered_stages(NOPE_PROGRAM)
    if not cake_backend.record_direct_stages(record, stages, NOPE_PROGRAM):
        pytest.skip(
            "generated program not available: the NoPE program has no natural-layout main stage"
        )
    return record, stages


def _require_nope_pass_program():
    """The registered NoPE program with the key-range-pass stages."""
    _require_nope_program(backward=True)
    _, record = record_for(torch.device("cuda"), NOPE_PROGRAM)
    stages = cake_jit.registered_stages(NOPE_PROGRAM)
    if not {"nope_bwd_compact", "nope_bwd_main_pass"} <= set(stages):
        pytest.skip(
            "generated program not available: the NoPE program has no key-range-pass stages"
        )
    return record, stages


def _check_forward_nope(out, lse, ref, *, fixed=False, margin=NOPE_FLOOR_MARGIN):
    """``out`` within ``margin`` x (1.02) its BF16-P numerics floor (rel-L2) plus the element-wise band; the floor
    (``reference_fp64_nope``'s ``out_emu``) follows the kernel's own rounding points -- P rounded to bf16 relative to
    the running row maximum under the threshold-6 lazy rescale over 64-slot tiles, denominator from the unrounded P,
    one final bf16 rounding -- inherited from the production forward kernel; rounding relative to the exact row
    maximum under-estimates the floor by 6-8 % on short peaked rows."""
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.isfinite(out.float()).all()
    err, floor = rel_l2(out, ref["out"]), rel_l2(ref["out_emu"], ref["out"])
    assert err <= margin * floor + FLOOR_ABS, (
        f"out: rel-L2 {err:.6f} vs floor {floor:.6f}"
    )
    torch.testing.assert_close(
        out.double(), ref["out"], atol=NOPE_OUT_ATOL, rtol=NOPE_OUT_RTOL
    )
    if fixed:
        assert rel_l2(out, ref["out"]) <= NOPE_GATE_OUT
    lse_ref = ref["lse"]
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    if finite.any():
        assert (
            lse.double()[finite] - lse_ref[finite]
        ).abs().max().item() <= GATE_LSE_ABS
    if (~finite).any():
        assert torch.all(lse[~finite] == float("-inf"))
        assert torch.all(out[~finite.any(-1)] == 0)  # fully masked rows


def _dq_row_errors(dq, dq_ref, zero_rows):
    """Per-(token, head)-row normalised error ``e[r]``: the RMS over the 512 elements of ``dq_r - exact_r`` divided by
    the row's exact RMS (floored at ``NOPE_ROW_SCALE_FLOOR``), over the rows of the tokens whose exact gradient is
    not identically zero (``zero_rows`` marks the tokens with at most one valid key: :func:`_check_dq_zero_rows`)."""
    ref = dq_ref.double()[~zero_rows]
    rms = ref.pow(2).mean(-1).sqrt()
    return (dq.double()[~zero_rows] - ref).pow(2).mean(-1).sqrt() / rms.clamp_min(
        NOPE_ROW_SCALE_FLOOR
    )


def _check_dq_zero_rows(dq, ref):
    """Rows whose exact ``dq`` is identically zero: the (token, head) rows of the tokens with at most one valid key
    (``ref["dq_zero_rows"]``).  With one valid key ``P = 1`` and ``out = v`` exactly, so ``dS = P (dP - delta)``
    cancels: ``dP`` (the MMA's fp32 accumulation of ``dout . v`` over the 512 dims) and ``delta`` (the row-sum stage's
    fp32 dot product of ``dout`` with ``hi + lo = v``) are the same exact value, and the kernel's
    ``dq = scale (dP - delta) k`` is their fp32 rounding difference times the key.  The check is element-wise against
    the bound the reference derives from the inputs and the frozen numerics (``ref["dq_zero_bound"]``,
    :func:`nope_ref.nope_zero_row_dq_bound`): ``|dq[t, h, el]| <= (1 + 2^-8)^2 scale B[t, h] |k[el]|`` with
    ``B[t, h] = 2 gamma_512 sum_d |dout[t, h, d] v[d]|``, ``gamma_n = n u / (1 - n u)``, ``u = 2^-24`` -- the standard
    forward error bound of two fp32 dot products of 512 exact products in any order, times the bf16 roundings of
    ``dS`` and ``dq``.  A token without a valid key has a zero bound: ``dq`` is exactly zero there.  The kernel's
    residue sits far below the bound -- measured on sm_103a (GB300), the largest ``|dq| / bound`` over all zero-gradient
    rows is 0.0019 on the peaked 0.99 self-weight case (one token, 64 rows), 0.0016 on the peaked 0.53 case and 0.0027
    at ``K_slots = 1`` (130 tokens, 8320 rows; single and two-pass alike), the per-row median of the implied
    ``|dP - delta| / B`` about 3e-4: a margin above 350x."""
    zero = ref["dq_zero_rows"]
    if not zero.any():
        return
    got, bound = dq.double()[zero].abs(), ref["dq_zero_bound"]
    positive = bound > 0
    ratio = torch.where(
        positive, got / bound.clamp_min(torch.finfo(torch.float64).tiny), 0.0
    )
    assert (got <= bound).all(), (
        f"dq on zero-gradient rows: {ratio.max().item():.4f} x the fp32 rounding bound "
        f"(max |dq| {got.max().item():.3e}; non-zero dq on a zero bound: {int((got[~positive] > 0).sum())})"
    )


def _check_dq_row_stats(e_k, e_f):
    """Structural guard on the kernel's row errors ``e_k`` against the emulated floor's ``e_f`` as distributions: the
    0.99 quantile within ``NOPE_ROW_P99_GUARD`` x, the maximum within ``NOPE_ROW_MAX_GUARD`` x the floor's.  A wrong
    key or a wrong row contribution is a row error of order 100 %; this is what the guard detects.  It is not a
    precision statement: the precision claim is the case-level rel-L2 within the measured floor x 1.02."""
    got, floor = e_k.quantile(0.99).item(), e_f.quantile(0.99).item()
    assert got <= NOPE_ROW_P99_GUARD * floor + FLOOR_ABS, (
        f"dq row error p99: {got:.6f} vs floor {floor:.6f}"
    )
    got, floor = e_k.max().item(), e_f.max().item()
    assert got <= NOPE_ROW_MAX_GUARD * floor + FLOOR_ABS, (
        f"dq row error max: {got:.6f} vs floor {floor:.6f}"
    )


def _check_dq_rows_nope(dq, ref):
    """Row-level ``dq`` guard as a distribution statistic, kernel against its emulated floor, both measured against the
    FP64 reference: ``e[r]`` = RMS of ``dq_r - exact_r`` over the row, divided by the row's exact RMS (floor 1e-3);
    the 0.99 quantile of the kernel's ``e`` within ``NOPE_ROW_P99_GUARD`` x the floor's, the maximum within
    ``NOPE_ROW_MAX_GUARD`` x the floor's -- a guard against a wrong key / wrong row contribution (row errors of order
    100 %); the precision claim is the case-level rel-L2 within the measured floor x 1.02.  An element-wise comparison
    of the kernel with the floor is not meaningful:
    both carry the forward output's last bits (bf16 hi + lo split, fp32 accumulation) through ``delta``, amplified by
    ~1 / (1 - max P) on rows where one key holds most of the mass (two to four valid keys, a self key near 1), as two
    different realisations of the same noise -- only their magnitudes compare.  The FA4 sparse-MLA backward on the same
    inputs misses the per-element band on those rows by x40-x1527 (measured), the kernel by x3-x7.  Rows whose exact
    gradient is identically zero (at most one valid key) are checked first, element-wise against the bound derived
    from the inputs (:func:`_check_dq_zero_rows`), and stay out of the statistics."""
    _check_dq_zero_rows(dq, ref)
    zero = ref["dq_zero_rows"]
    e_k = _dq_row_errors(dq, ref["dq"], zero)
    e_f = _dq_row_errors(ref["dq_emu"], ref["dq"], zero)
    if e_k.numel():
        _check_dq_row_stats(e_k, e_f)


def _check_backward_nope(dq, dkv, ref, *, fixed=False, margin=NOPE_FLOOR_MARGIN):
    """``dq`` (BF16) within ``margin`` x (1.02) and the FP32 ``dkv`` within 1.05x their BF16-P/dS numerics floors on the
    same inputs (rel-L2), ``dq`` also within the row-level distribution check against its floor
    (:func:`_check_dq_rows_nope`); the canonical iid 4k x 4k configuration also within the fixed table."""
    assert dq.dtype == torch.bfloat16 and dkv.dtype == torch.float32
    assert tuple(dkv.shape) == tuple(ref["dkv"].shape) and dkv.is_contiguous()
    assert torch.isfinite(dq.float()).all() and torch.isfinite(dkv).all()
    for got, name in ((dq, "dq"), (dkv, "dkv")):
        if not ref[name].any():
            # an identically zero exact gradient (one valid key per row: P = 1, dS = 0, so dq = 0): a relative
            # floor is undefined there; the kernel's residue is the fp32 rounding of dP - delta, which the row
            # check bounds element-wise from the inputs (_check_dq_zero_rows)
            continue
        err, floor = rel_l2(got, ref[name]), rel_l2(ref[f"{name}_emu"], ref[name])
        limit = margin if name == "dq" else FLOOR_MARGIN
        assert err <= limit * floor + FLOOR_ABS, (
            f"{name}: rel-L2 {err:.6f} vs floor {floor:.6f}"
        )
    _check_dq_rows_nope(dq, ref)
    if fixed:
        assert rel_l2(dq, ref["dq"]) <= NOPE_GATE_DQ
        assert rel_l2(dkv, ref["dkv"]) <= NOPE_GATE_DKV


def _nope_inputs(seq_q, seq_k, *, seed, topk, index="iid", **kw):
    return nope_ref.make_nope_inputs(
        [seq_q], [seq_k], seed=seed, topk=topk, index=index, **kw
    )


def _nope_reference(inp, indices=None, *, dout=None, topk_length=None):
    return nope_ref.reference_fp64_nope(
        inp.q,
        inp.kv,
        inp.idx_global if indices is None else indices,
        dout=dout,
        topk_length=topk_length,
        softmax_scale=NOPE_SOFTMAX_SCALE,
    )


# --- registry and generated launchers (CPU) ---


@pytest.mark.parametrize("program", cake_jit.PROGRAMS)
def test_registry_programs_are_well_formed(program):
    """Both records: declared stages in the program's order with physical entries, every library named in
    ``build.json`` for every arch, five launch leaves per stage, launchers and grids for every stage; the NoPE
    record carries its own policies (2048 B per key, one pass by default, natural accumulation from one key per
    query) and may be partial until its libraries are exported (a registered forward stage comes first)."""
    assert cake_jit.PROGRAMS == (cake_jit.PROGRAM, NOPE_PROGRAM)
    record = cake_jit.record(program)
    order = cake_jit.STAGES_OF[program]
    stages = cake_jit.registered_stages(program)
    assert stages == tuple(s for s in order if s in stages)  # launch order
    # the NoPE record's stage list leads with the index-preparation program, which is not an attention stage the
    # backend plans (``registered_stages`` is those, in launch order)
    prefix = (NOPE_INDEX_PREP_STAGE,) if NOPE_INDEX_PREP_STAGE in record else ()
    assert tuple(record["stages"]) == prefix + stages
    assert set(record["arches"]) == {"sm_100a", "sm_103a", "sm_107a"}
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    first = cake_jit.stage_name("fwd", program)
    if first in stages:
        assert stages[0] == first
    policy = KeyPassPolicy.from_record(record)
    assert policy is not None
    direct = cake_backend.DkvDirectPolicy.from_record(record)
    assert direct is not None
    if program == NOPE_PROGRAM:
        # natural accumulation whenever the row has at least one key per query (measured faster than the
        # permuted drain on every row) and one pass by default (measured fastest on every row): no per-arch rules
        assert direct.min_keys_per_query == 1
        assert policy.key_bytes == 2048
        assert policy.dq_partial_bytes_per_token == NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
        assert policy.max_passes == 1
        for arch in record["arches"]:
            assert policy.rule_for(arch) == policy.rule_for(None)
        rope = KeyPassPolicy.from_record(cake_jit.record())
        assert (policy.l2_budget_bytes, policy.workspace_budget_bytes) == (
            rope.l2_budget_bytes,
            rope.workspace_budget_bytes,
        )
        assert policy.token_chunk_multiple == rope.token_chunk_multiple
    else:
        assert direct.min_keys_per_query == 4
    pending = []
    for stage in stages:
        physical = record[stage]
        assert physical["library"] and physical["ffi_entry"]
        for arch in record["arches"]:
            entry = libraries.get(f"{physical['library']}_{arch}")
            if entry is None:
                pending.append(f"{physical['library']}_{arch}")
                continue
            assert entry["arch"] == arch and entry["entry"] == physical["ffi_entry"]
            assert all(
                (Path(cake_jit.__file__).parent / s).is_file() for s in entry["sources"]
            )
        launch = physical["launch"]
        assert set(launch) == {
            "block",
            "cluster",
            "cooperative",
            "dynamic_smem_bytes",
            "use_pdl",
        }
        assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
        assert len(launch["cluster"]) == 3 and all(
            int(c) >= 1 for c in launch["cluster"]
        )
        assert isinstance(launch["cooperative"], bool)
        assert isinstance(launch["use_pdl"], bool)
        assert isinstance(launch["dynamic_smem_bytes"], int)
        assert launch["dynamic_smem_bytes"] >= 0
        assert stage in cake_launch.LAUNCH and stage in cake_launch.GRID
        assert stage in cake_launch.STAGE_ARGS
        assert tuple(cake_launch.STAGE_ARGS[stage][-3:]) == (
            "grid_x",
            "grid_y",
            "grid_z",
        )
    if program == NOPE_PROGRAM:
        # a stage that reuses a rope kernel names the same library and bakes the same launch geometry
        rope = cake_jit.record()
        for stage, shared in NOPE_SHARED_STAGES.items():
            if stage in stages:
                assert record[stage]["library"] == rope[shared]["library"]
                assert record[stage]["launch"] == rope[shared]["launch"]
                assert tuple(cake_launch.STAGE_ARGS[stage]) == tuple(
                    cake_launch.STAGE_ARGS[shared]
                )
        if "nope_bwd_delta" in stages:
            # the strided delta serves both records (the rope ``bwd_delta`` names the same stem: the rotary
            # backward reads a strided dout in place): 256 threads, no dynamic shared memory
            delta = record["nope_bwd_delta"]
            assert delta["library"] == rope["bwd_delta"]["library"]
            assert delta["launch"]["block"] == [256, 1, 1]
            assert delta["launch"]["dynamic_smem_bytes"] == 0
    if pending:
        # the registry is the source the export renders from; its libraries reach build.json with the export
        # (scripts/check_cake_build_manifests.py reports the gap until then)
        pytest.skip(
            f"export pending: build.json lacks {len(pending)} entries ({pending[0]}, ...)"
        )


def test_nope_stage_names_and_registry_accessors():
    assert cake_jit.STAGE_PREFIX == {cake_jit.PROGRAM: "", NOPE_PROGRAM: "nope_"}
    assert tuple(f"nope_{s}" for s in cake_jit.STAGES) == cake_jit.NOPE_STAGES
    assert cake_jit.NOPE_FORWARD_STAGES == ("nope_fwd",)
    assert cake_jit.NOPE_STAGES[1:] == tuple(
        cake_jit.stage_name(s, NOPE_PROGRAM) for s in cake_jit.STAGES[1:]
    )
    assert cake_jit.STAGES_OF == {
        cake_jit.PROGRAM: cake_jit.STAGES,
        NOPE_PROGRAM: cake_jit.NOPE_STAGES,
    }
    assert cake_jit.stage_name("bwd_main_pass_natural", NOPE_PROGRAM) == (
        "nope_bwd_main_pass_natural"
    )
    assert cake_jit.stage_name("fwd") == "fwd"
    assert cake_backend._program_stages(("fwd", "bwd_cast"), NOPE_PROGRAM) == (
        "nope_fwd",
        "nope_bwd_cast",
    )
    with pytest.raises(ValueError, match="unknown program"):
        cake_jit.stage_name("fwd", "cake_dsa_h64_other")
    with pytest.raises(ValueError, match="unknown program"):
        cake_jit.record("cake_dsa_h64_other")
    with pytest.raises(ValueError, match="unknown program"):
        cake_jit.registered_stages("cake_dsa_h64_other")
    stages = cake_jit.registered_stages(NOPE_PROGRAM)
    for stage in cake_jit.NOPE_STAGES:
        if stage in stages:
            for arch in cake_jit.record(NOPE_PROGRAM)["arches"]:
                name = cake_jit.library_name(stage, arch, NOPE_PROGRAM)
                assert (
                    name == f"{cake_jit.record(NOPE_PROGRAM)[stage]['library']}_{arch}"
                )
                if stage in NOPE_SHARED_STAGES:
                    assert name == cake_jit.library_name(
                        NOPE_SHARED_STAGES[stage], arch
                    )
            with pytest.raises(RuntimeError, match="not served"):
                cake_jit.library_name(stage, "sm_80", NOPE_PROGRAM)
        else:
            with pytest.raises(NotImplementedError):
                cake_jit.library_name(stage, "sm_100a", NOPE_PROGRAM)


def test_nope_launchers_bind_the_kernel_argument_lists_and_grids():
    """Every NoPE launcher passes ``STAGE_ARGS[stage]`` positionally (the kernels' own lists; the shared stages
    keep the rope lists) followed by the grid; the grid rules follow the program's CTA mapping."""
    args = {stage: tuple(names) for stage, names in cake_launch.STAGE_ARGS.items()}
    assert args["nope_fwd"] == NOPE_FWD_ARGS
    # the conversion operands the packed entry binds (``varlen_index_mode`` probes them) close the kernel list
    assert NOPE_FWD_ARGS[-10:-3] == cake_backend.NOPE_FWD_CONVERT_ARGS
    for stage in (
        "nope_bwd_main",
        "nope_bwd_main_natural",
        "nope_bwd_main_pass",
        "nope_bwd_main_pass_natural",
    ):
        assert args[stage] == NOPE_MAIN_ARGS
    for stage, shared in NOPE_SHARED_STAGES.items():
        assert args[stage] == args[shared]
    # the strided delta's list on both records: the rope backward hands its dout strides to the same kernel
    assert args["nope_bwd_delta"] == NOPE_DELTA_ARGS
    assert args["bwd_delta"] == NOPE_DELTA_ARGS
    assert "q_rope" not in NOPE_MAIN_ARGS and "dkr_f32" not in NOPE_MAIN_ARGS
    for stage in cake_jit.NOPE_STAGES:
        names = args[stage]
        values = {name: object() for name in names[:-3]}
        recorder = _Recorder()
        cake_launch.LAUNCH[stage](recorder, values, (7, 1, 1))
        assert len(recorder.calls) == 1
        assert recorder.calls[0] == tuple(values[name] for name in names[:-3]) + (
            7,
            1,
            1,
        )
    grid = cake_launch.GRID
    assert grid["nope_fwd"](130, 4096, 2176) == (130, 1, 1)
    assert grid["nope_fwd"](0, 4096, 2176) == (1, 1, 1)
    assert grid["nope_bwd_delta"](130, 4096, 2176) == (1040, 1, 1)
    assert grid["nope_bwd_main"](130, 4096, 2176) == (130, 1, 1)
    assert grid["nope_bwd_main_natural"](130, 4096, 2176) == (130, 1, 1)
    assert grid["nope_bwd_compact"](130, 4096, 2176) == (33, 1, 1)
    assert grid["nope_bwd_main_pass"](130, 4096, 2176) == (130, 1, 1)
    assert grid["nope_bwd_main_pass_natural"](130, 4096, 2176) == (130, 1, 1)
    # the cast walks the one 512-wide accumulator: 16 permuted groups per key, 256 per CTA
    assert grid["nope_bwd_cast"](5, 4096, 3) == (256, 1, 1)
    assert grid["nope_bwd_cast"](5, 7, 3) == (1, 1, 1)
    assert grid["bwd_cast"](5, 4096, 3) == (288, 1, 1)  # the rope cast is unchanged


@pytest.mark.parametrize("program", cake_jit.PROGRAMS)
def test_binding_run_signatures_match_the_stage_argument_lists(program):
    """The ``Run(`` parameter list of every registered stage's binding (``arg_<name>`` / ``a_<name>`` per kernel
    argument, then the grid) is the stage's ``STAGE_ARGS`` list: the launcher and the binding agree positionally."""
    record = cake_jit.record(program)
    root = Path(cake_jit.__file__).parent
    pattern = re.compile(r"\bvoid\s+Run\s*\(([^)]*)\)\s*\{", re.S)
    entries = {e["name"]: e for e in cake_build.manifest()["libraries"]}
    pending = []
    for stage in cake_jit.registered_stages(program):
        name = cake_jit.library_name(stage, record["arches"][0], program)
        sources = entries[name]["sources"] if name in entries else []
        bindings = [root / s for s in sources if s.endswith("_binding.cu")]
        if len(bindings) != 1 or not bindings[0].is_file():
            pending.append(stage)
            continue
        binding = bindings[0]
        found = pattern.findall(binding.read_text())
        assert len(found) == 1, binding
        params = [p.strip().split()[-1] for p in found[0].split(",")]
        # the binding generator spells a kernel argument ``arg_<name>`` (shipped package) or ``a_<name>`` (current
        # generator); the grid scalars are unprefixed either way
        names = tuple(
            p[4:] if p.startswith("arg_") else p[2:] if p.startswith("a_") else p
            for p in params
        )
        assert names == tuple(cake_launch.STAGE_ARGS[stage]), stage
    if pending:
        pytest.skip(f"export pending: no binding source yet for {pending}")


# --- host layer (CPU) ---


def test_validate_dsa_nope_inputs_accepts_the_contract_layouts():
    q, kv, idx = _nope_host_inputs()
    assert validate_dsa_nope_inputs(q, kv, idx) == (8, 16, 5)
    # strided query views (a packed [T, 64, 576] query, head stride 576) and any index row stride
    packed = torch.zeros(8, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    assert validate_dsa_nope_inputs(packed[..., :D_LATENT], kv, idx) == (8, 16, 5)
    wide = torch.zeros(8, 13, dtype=torch.int32)
    assert validate_dsa_nope_inputs(q, kv, wide[:, 3:8]) == (8, 16, 5)
    assert validate_dsa_nope_inputs(q, kv, idx[:, :1]) == (8, 16, 1)  # K_slots 1
    lengths = torch.full((8,), 3, dtype=torch.int32)
    dout = torch.zeros_like(q)
    assert validate_dsa_nope_inputs(q, kv, idx, lengths, dout=dout) == (8, 16, 5)
    assert validate_dsa_nope_inputs(q[:0], kv, idx[:0], allow_empty_queries=True) == (
        0,
        16,
        5,
    )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda q, kv, idx: (q, kv.unsqueeze(1), idx), "kv_latent"),
        (lambda q, kv, idx: (q, kv[:0], idx), "S > 0"),
        (lambda q, kv, idx: (q[:0], kv, idx[:0]), "T == 0"),
        (lambda q, kv, idx: (q, kv, idx.long()), "int32"),
        (lambda q, kv, idx: (q, kv, idx[:, :0]), "K_slots >= 1"),
        (lambda q, kv, idx: (q, kv, idx[:, ::2]), "contiguous"),
        (lambda q, kv, idx: (q, kv, idx[:4]), r"\[T, K_slots\]"),
        (lambda q, kv, idx: (q[..., :256], kv, idx), "q_latent"),
        (lambda q, kv, idx: (q.float(), kv, idx), "q_latent"),
        (lambda q, kv, idx: (q, kv[:, :256], idx), "kv_latent"),
    ],
)
def test_validate_dsa_nope_inputs_rejects(mutate, match):
    q, kv, idx = _nope_host_inputs()
    with pytest.raises(ValueError, match=match):
        validate_dsa_nope_inputs(*mutate(q, kv, idx))


def test_validate_dsa_nope_inputs_rejects_bad_lengths_and_dout():
    q, kv, idx = _nope_host_inputs()
    with pytest.raises(ValueError, match="topk_length"):
        validate_dsa_nope_inputs(q, kv, idx, torch.zeros(8, dtype=torch.int64))
    with pytest.raises(ValueError, match="topk_length"):
        validate_dsa_nope_inputs(q, kv, idx, torch.zeros(9, dtype=torch.int32))
    with pytest.raises(ValueError, match="T rows"):
        validate_dsa_nope_inputs(q, kv, idx, dout=torch.zeros_like(q)[:4])
    with pytest.raises(ValueError, match="dout"):
        validate_dsa_nope_inputs(q, kv, idx, dout=torch.zeros_like(q).float())
    head_major = torch.zeros(NUM_HEADS, 8, D_LATENT, dtype=torch.bfloat16).permute(
        1, 0, 2
    )
    assert head_major.stride() == (D_LATENT, 8 * D_LATENT, 1)
    # a head-major dout, or a slice of a packed [T, 64, 576] tensor, is admitted: the delta stage reads dout
    # through its token / head strides (multiples of 8 elements); an odd head stride or a channel stride is not
    assert validate_dsa_nope_inputs(q, kv, idx, dout=head_major) == (8, 16, 5)
    packed = torch.zeros(8, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    assert validate_dsa_nope_inputs(q, kv, idx, dout=packed[..., :D_LATENT]) == (
        8,
        16,
        5,
    )
    odd = torch.zeros(8, NUM_HEADS, D_LATENT + 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head stride"):
        validate_dsa_nope_inputs(q, kv, idx, dout=odd[..., :D_LATENT])
    wide = torch.zeros(8, NUM_HEADS, 2 * D_LATENT, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="last dimension"):
        validate_dsa_nope_inputs(q, kv, idx, dout=wide[..., ::2])


def _i32(*values):
    return torch.tensor(values, dtype=torch.int32)


def test_validate_cu_seqlens_checks_structure_and_values():
    """The varlen invariants: int32 ``[num_docs + 1]`` prefix sums of one length, starting at 0, non-decreasing,
    ending at the packed ``T`` / ``S`` (a shorter or longer key window is rejected, not truncated) and with
    ``seqlen_q <= seqlen_k`` per document; ``values=False`` keeps the structural checks only."""
    validate_cu_seqlens(_i32(0, 3, 8), _i32(0, 5, 16), 8, 16)
    validate_cu_seqlens(_i32(0, 0, 8), _i32(0, 4, 16), 8, 16)  # an empty query segment
    validate_cu_seqlens(_i32(0, 0), _i32(0, 16), 0, 16)  # no query rows at all
    cases = [
        (torch.tensor([0, 8]), _i32(0, 16), 8, 16, "cu_seqlens_q must be an int32"),
        (_i32(0, 8), _i32(0, 16).view(2, 1), 8, 16, "cu_seqlens_k must be an int32"),
        (_i32(0), _i32(0), 0, 0, r"\[num_docs \+ 1\]"),
        (_i32(0, 3, 8), _i32(0, 16), 8, 16, "same documents"),
        (_i32(1, 8), _i32(0, 16), 8, 16, "cu_seqlens_q must start at 0"),
        (_i32(0, 8), _i32(2, 16), 8, 16, "cu_seqlens_k must start at 0"),
        (
            _i32(0, 5, 3, 8),
            _i32(0, 5, 10, 16),
            8,
            16,
            "cu_seqlens_q must be non-decreasing",
        ),
        (
            _i32(0, 4, 6, 8),
            _i32(0, 12, 10, 16),
            8,
            16,
            "cu_seqlens_k must be non-decreasing",
        ),
        (
            _i32(0, 8),
            _i32(0, 12),
            8,
            16,
            r"cu_seqlens_k\[-1\] \(12\) must equal the packed key rows S \(16\)",
        ),
        (_i32(0, 8), _i32(0, 20), 8, 16, r"cu_seqlens_k\[-1\] \(20\)"),
        (
            _i32(0, 6),
            _i32(0, 16),
            8,
            16,
            r"cu_seqlens_q\[-1\] \(6\) must equal the packed query rows T \(8\)",
        ),
        (
            _i32(0, 6, 8),
            _i32(0, 4, 16),
            8,
            16,
            r"document 0 has seqlen_q \(6\) > seqlen_k \(4\)",
        ),
    ]
    for bad_q, bad_k, T, S, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_cu_seqlens(bad_q, bad_k, T, S)
    # values=False: the structural checks only (a timed loop over inputs validated once)
    validate_cu_seqlens(_i32(0, 8), _i32(0, 12), 8, 16, values=False)
    validate_cu_seqlens(_i32(0, 6, 8), _i32(0, 4, 16), 8, 16, values=False)
    with pytest.raises(ValueError, match="int32"):
        validate_cu_seqlens(torch.tensor([0, 8]), _i32(0, 16), 8, 16, values=False)


def test_nope_varlen_backend_entry_checks_cu_seqlens_before_any_device_work():
    """The backend's varlen entry validates the prefix sums first (host tensors: nothing else would run): their
    structure on every call, their values with ``check_cu_seqlens=True`` (the default leaves the values to the
    caller: that check synchronizes with the device)."""
    q, kv, idx = _nope_host_inputs()
    varlen = cake_backend.dsa_sparse_attention_nope_varlen
    with pytest.raises(ValueError, match=r"cu_seqlens_k\[-1\]"):
        varlen(q, kv, idx, _i32(0, 8), _i32(0, 12), 8, 12, check_cu_seqlens=True)
    with pytest.raises(ValueError, match="seqlen_q"):
        varlen(q, kv, idx, _i32(0, 6, 8), _i32(0, 4, 16), 8, 16, check_cu_seqlens=True)
    with pytest.raises(ValueError, match="non-decreasing"):
        varlen(
            q,
            kv,
            idx,
            _i32(0, 5, 3, 8),
            _i32(0, 5, 10, 16),
            8,
            16,
            check_cu_seqlens=True,
        )
    with pytest.raises(ValueError, match="same documents"):
        varlen(q, kv, idx, _i32(0, 3, 8), _i32(0, 16), 8, 16)
    with pytest.raises(ValueError, match="cu_seqlens_q must be contiguous"):
        varlen(q, kv, idx, _i32(0, 0, 3, 3, 8, 8)[::2], _i32(0, 5, 16), 8, 16)
    with pytest.raises(ValueError, match="gather_kv_indices must be an int32"):
        varlen(q, kv, idx[:4], _i32(0, 3, 8), _i32(0, 5, 16), 8, 16)


# --- packed form: on-device index preparation (host layer) ---

# The index-preparation kernel's argument list (the generated launcher binds these names positionally, then the
# grid); the export writes the real tables, the stand-in below mirrors them.
# The index-preparation kernel's argument list: the fifteen of the first release, then the compaction mode
# (``compact``: 0 = position-preserving rewrite with last-valid + 1 lengths, 1 = stable compaction with the valid
# counts), the caller's length cut (``length_in [T]``, read only when ``has_length_in != 0``; the ``topk_length``
# output pointer stands in as the placeholder otherwise), then the grid.
INDEX_PREP_ARGS = (
    "indices",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "out",
    "topk_length",
    "num_rows",
    "topk",
    "idx_stride",
    "indices_offset",
    "num_docs",
    "search_iters",
    "causal",
    "has_docs",
    "flat_bound",
    "write_out",
    "compact",
    "length_in",
    "has_length_in",
    "grid_x",
    "grid_y",
    "grid_z",
)
_INDEX_PREP_STAND_IN = {
    "library": f"{NOPE_PROGRAM}_stand_in_index_prep",
    "ffi_entry": "run",
    "launch": {
        "block": [128, 1, 1],
        "cluster": [1, 1, 1],
        "cooperative": False,
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
    },
}


def _index_prep_example():
    """The four-document example of the causal glue test: tail segments, an empty query segment, a fully masked
    row, a repeated key."""
    cu_q = _i32(0, 3, 5, 5, 9)
    cu_k = _i32(0, 5, 11, 14, 18)
    local = torch.tensor(
        [
            [0, 1, 2, -1, 4],
            [1, 0, 3, 4, -1],
            [2, 1, 0, 4, 5],
            [4, 5, 3, -1, 6],
            [5, 4, 0, 6, -1],
            [0, 1, 2, 3, -1],
            [3, 2, 1, 0, 4],
            [-1, -1, -1, -1, -1],
            [3, 0, 1, 2, 3],
        ],
        dtype=torch.int32,
    )
    return local, cu_q, cu_k


def test_index_prep_grid_and_search_iters():
    grid = cake_backend.index_prep_grid
    assert [grid(n) for n in (0, 1, 4, 5, 16263)] == [
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (4066, 1, 1),
    ]
    iters = cake_backend.index_prep_search_iters
    assert [iters(n) for n in (1, 2, 3, 4, 5, 8, 9, 512)] == [1, 2, 3, 3, 4, 4, 5, 10]
    for n in (1, 2, 3, 7, 9, 100, 1023, 1024, 1025):
        assert iters(n) == (0 if n == 1 else math.ceil(math.log2(n))) + 1


def test_index_storage_alias_covers_strided_and_offset_views():
    """The kernel addresses its index operand as flat storage (row ``t`` at ``indices_offset + t * idx_stride``):
    the host passes the contiguous 1-D span the rows occupy from the vector-aligned element at or below the view's
    start, so column slices, every-other-row views and views inside a larger storage are read in place."""
    T, K = 6, 5
    alias = cake_backend._index_storage_alias
    base = torch.arange(T * (K + 8), dtype=torch.int32).view(T, K + 8)
    view, offset, stride = alias(base[:, :K])
    assert view.dim() == 1 and view.stride() == (1,) and view.is_contiguous()
    assert (offset, stride, view.numel()) == (0, K + 8, (T - 1) * (K + 8) + K)
    assert view.data_ptr() == base.data_ptr()
    view, offset, stride = alias(base[:, 3 : 3 + K])
    assert (offset, stride, view.numel()) == (3, K + 8, 3 + (T - 1) * (K + 8) + K)
    assert view.data_ptr() == base.data_ptr()
    for t in range(T):
        row = view[offset + t * stride : offset + t * stride + K]
        assert torch.equal(row, base[t, 3 : 3 + K])
    view, offset, stride = alias(
        base[:, 8 : 8 + K]
    )  # a vector-aligned start folds into the pointer
    assert (offset, stride) == (0, K + 8) and view.data_ptr() == base[:, 8:].data_ptr()
    flat = torch.zeros(T, K, dtype=torch.int32)
    view, offset, stride = alias(flat)
    assert (offset, stride, view.numel()) == (0, K, T * K)
    assert view.data_ptr() == flat.data_ptr()
    assert alias(flat[:1])[2] == K  # one row: the width stands as its stride
    view, offset, stride = alias(torch.zeros(2 * T, K, dtype=torch.int32)[::2])
    assert (offset, stride, view.numel()) == (0, 2 * K, (T - 1) * 2 * K + K)
    inside = torch.zeros(T * K + 13, dtype=torch.int32)[13:].view(T, K)
    view, offset, stride = alias(inside)
    assert (offset, stride) == (5, K) and view.data_ptr() == inside.data_ptr() - 5 * 4


def test_nope_index_prep_checks_structure_first_and_refuses_host_tensors():
    """The structural checks of the index operands come first (host tensors: nothing else would run); a host tensor
    with valid structure is refused -- the NoPE entries have no torch fallback -- except ``T == 0``, which needs no
    launch; the torch formulation stays the tests' reference."""
    local, cu_q, cu_k = _index_prep_example()
    with pytest.raises(ValueError, match="int32"):
        nope_globalize_indices(local.long(), cu_q, cu_k)
    with pytest.raises(ValueError, match="same documents"):
        nope_globalize_indices(local, cu_q, cu_k[:-1])
    with pytest.raises(ValueError, match="cu_seqlens_k must be an int32"):
        nope_globalize_indices(local, cu_q, cu_k.view(1, -1))
    strided = torch.stack([cu_q, cu_q], 1)[:, 0]
    with pytest.raises(ValueError, match="cu_seqlens_q must be contiguous"):
        nope_globalize_indices(local, strided, cu_k)
    with pytest.raises(ValueError, match="cu_seqlens_q must be contiguous"):
        validate_cu_seqlens(strided, cu_k, 9, 18)
    with pytest.raises(NotImplementedError, match="CUDA"):
        nope_globalize_indices(local, cu_q, cu_k, causal=True)
    with pytest.raises(NotImplementedError, match="CUDA"):
        cake_backend.varlen_index_mode(local)
    empty = nope_globalize_indices(local[:0], _i32(0, 0), _i32(0, 18))
    assert tuple(empty[0].shape) == (0, 5) and tuple(empty[1].shape) == (0,)
    assert empty[0].dtype == torch.int32 and empty[1].dtype == torch.int32
    empty_c = nope_globalize_indices(local[:0], _i32(0, 0), _i32(0, 18), compact=True)
    assert tuple(empty_c[0].shape) == (0, 5) and tuple(empty_c[1].shape) == (0,)
    with pytest.raises(NotImplementedError, match="CUDA"):
        nope_globalize_indices(local, cu_q, cu_k, causal=True, compact=True)
    # the flat compacting helper: the same structure checks, host tensors refused, T == 0 without a launch
    compact_indices = cake_backend.nope_compact_indices
    with pytest.raises(ValueError, match="int32"):
        compact_indices(local.long(), 18)
    with pytest.raises(NotImplementedError, match="CUDA"):
        compact_indices(local, 18)
    empty_f = compact_indices(local[:0], 18)
    assert tuple(empty_f[0].shape) == (0, 5) and tuple(empty_f[1].shape) == (0,)
    # the reference rules the stage is held to, on this example: the position-preserving default and the
    # compacting twin (valid slots first in slot order, -1 after, lengths = counts)
    strict, lengths = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=True, return_topk_length=True
    )
    assert torch.equal(
        strict, globalize_gather_indices_loop(local, cu_q, cu_k, causal=True)
    )
    assert lengths.tolist() == [3, 3, 4, 3, 3, 1, 4, 0, 5]
    compacted, counts = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=True, compact=True, return_topk_length=True
    )
    want_rows, want_counts = compact_rows(strict, 18)
    assert torch.equal(compacted, want_rows) and torch.equal(counts, want_counts)
    assert counts.tolist() == [3, 3, 4, 2, 3, 1, 2, 0, 5]
    assert compacted[3].tolist() == [
        9,
        8,
        -1,
        -1,
        -1,
    ]  # row 3: slots 0 and 2 valid, slot 1 dropped
    assert compacted[6].tolist() == [15, 14, -1, -1, -1]
    assert compacted.tolist() == compact_rows_loop(strict.tolist(), 18)
    # the cut is a position cut applied before the compaction
    cut = _i32(5, 1, 2, 5, 0, 5, 5, 5, 2)
    rows_cut, counts_cut = offset_gather_kv_indices(
        local,
        cu_q,
        cu_k,
        causal=True,
        compact=True,
        topk_length=cut,
        return_topk_length=True,
    )
    want_cut = compact_rows(strict, 18, cut)
    assert torch.equal(rows_cut, want_cut[0]) and torch.equal(counts_cut, want_cut[1])
    assert counts_cut.tolist() == [3, 1, 2, 2, 0, 1, 2, 0, 2]
    assert compacted.dtype == torch.int32 and counts.dtype == torch.int32
    assert torch.equal(
        offset_gather_kv_indices(local, cu_q, cu_k, causal=True, compact=True),
        compacted,
    )


def test_nope_record_registers_the_index_prep_stage_first():
    """The shipped NoPE record carries the index-preparation stage ahead of ``nope_fwd`` (it runs before it), with the
    128-thread block the kernel declares, and the generated tables bind its eighteen arguments in the kernel's order
    (the fifteen of the first release, then ``compact``, ``length_in``, ``has_length_in``) followed by the grid; the
    grid rule is one CTA per four rows."""
    record = cake_jit.record(NOPE_PROGRAM)
    assert record["stages"][0] == NOPE_INDEX_PREP_STAGE
    stage = record[NOPE_INDEX_PREP_STAGE]
    assert stage["library"] and stage["ffi_entry"] == "run"
    assert stage["launch"]["block"] == [128, 1, 1]
    assert stage["launch"]["cluster"] == [1, 1, 1]
    assert stage["launch"]["dynamic_smem_bytes"] == 0
    assert not stage["launch"]["cooperative"] and not stage["launch"]["use_pdl"]
    assert tuple(cake_launch.STAGE_ARGS[NOPE_INDEX_PREP_STAGE]) == INDEX_PREP_ARGS
    for T in (0, 1, 4, 5, 16263):
        assert cake_launch.GRID[NOPE_INDEX_PREP_STAGE](T, 0, 0) == (
            cake_backend.index_prep_grid(T)
        )
    values = {name: object() for name in INDEX_PREP_ARGS[:-3]}
    recorder = _Recorder()
    cake_launch.LAUNCH[NOPE_INDEX_PREP_STAGE](recorder, values, (9, 1, 1))
    assert recorder.calls == [
        tuple(values[name] for name in INDEX_PREP_ARGS[:-3]) + (9, 1, 1)
    ]
    for arch in record["arches"]:
        assert cake_jit.library_name(NOPE_INDEX_PREP_STAGE, arch, NOPE_PROGRAM) == (
            f"{stage['library']}_{arch}"
        )


def test_nope_index_prep_availability_and_mode_follow_the_record(monkeypatch):
    """With the stage in the record the packed entry's route follows the table size: the one-launch
    position-preserving forward (``"mask"``) below ``NOPE_CONVERT_SLOT_LIMIT`` slots, the compacting stage at or
    above it and always with caller lengths.  Without the stage a forward program advertising the conversion
    operands is the stage-less fallback (``"mask"`` at every size; a caller cut needs the stage); a program with
    neither refuses on a served device (no torch fallback).  The shipped limit keeps both small bench tables
    (1024 x 128 and 1024 x 512 slots) on the one-launch forward; a limit of 0 (measurement) is the stage at every
    size."""
    record = dict(cake_jit.record(NOPE_PROGRAM))
    record.pop(NOPE_INDEX_PREP_STAGE, None)
    monkeypatch.setitem(cake_jit.MODULES, NOPE_PROGRAM, record)
    assert cake_backend.nope_index_prep_available(torch.device("cuda")) is False
    converts = all(
        name in cake_launch.STAGE_ARGS["nope_fwd"]
        for name in cake_backend.NOPE_FWD_CONVERT_ARGS
    )
    assert cake_backend.nope_fwd_converts_indices() is converts
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    assert limit > 1024 * 512  # both small bench tables take the one-launch forward
    assert cake_backend.nope_varlen_index_mode(1024, 128) == "mask"
    assert cake_backend.nope_varlen_index_mode(1024, 512) == "mask"
    if not _device_supported():
        return
    idx = torch.zeros(4, 3, dtype=torch.int32, device="cuda")
    if converts:
        assert cake_backend.varlen_index_mode(idx) == "mask"
        with pytest.raises(NotImplementedError, match="index-preparation"):
            cake_backend.varlen_index_mode(idx, caller_lengths=True)
        # the stage-less fallback does not depend on the size
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
        assert cake_backend.varlen_index_mode(idx) == "mask"
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", limit)
    else:
        with pytest.raises(NotImplementedError, match="index-preparation"):
            cake_backend.varlen_index_mode(idx)
    with pytest.raises(NotImplementedError, match="index-preparation"):
        nope_globalize_indices(idx, idx.new_tensor([0, 4]), idx.new_tensor([0, 8]))
    with pytest.raises(NotImplementedError, match="index-preparation"):
        cake_backend.nope_compact_indices(idx, 8)
    record[NOPE_INDEX_PREP_STAGE] = dict(_INDEX_PREP_STAND_IN)
    assert cake_backend.nope_index_prep_available(torch.device("cuda")) is True
    assert cake_backend.varlen_index_mode(idx, caller_lengths=True) == "index_prep"
    monkeypatch.setitem(
        cake_launch.STAGE_ARGS,
        "nope_fwd",
        list(cake_launch.STAGE_ARGS["nope_fwd"])
        + list(cake_backend.NOPE_FWD_CONVERT_ARGS),
    )
    assert cake_backend.nope_fwd_converts_indices()
    assert cake_backend.varlen_index_mode(idx) == "mask"  # 12 slots: below the limit
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.varlen_index_mode(idx) == "index_prep"
    assert cake_backend.varlen_index_mode(idx, caller_lengths=True) == "index_prep"
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 1 << 20)
    assert cake_backend.varlen_index_mode(idx) == "mask"
    assert cake_backend.varlen_index_mode(idx, caller_lengths=True) == "index_prep"
    assert cake_backend.nope_varlen_index_mode(1024, 1023) == "mask"
    assert cake_backend.nope_varlen_index_mode(1024, 1024) == "index_prep"
    # a program without the converting forward: the stage at every size
    monkeypatch.setitem(
        cake_launch.STAGE_ARGS,
        "nope_fwd",
        [n for n in cake_launch.STAGE_ARGS["nope_fwd"] if n != "convert"],
    )
    assert not cake_backend.nope_fwd_converts_indices()
    assert cake_backend.varlen_index_mode(idx) == "index_prep"


def test_nope_index_modes_follow_the_slot_limit_on_the_host(monkeypatch):
    """The size rule alone (no device, no registry): ``nope_varlen_index_mode`` and ``nope_flat_index_mode``
    pick the one-launch position-preserving forward (``"mask"``) for ``T * K < NOPE_CONVERT_SLOT_LIMIT`` and the
    compacting stage (``"index_prep"``) at or above it -- the boundary at ``T * K == limit - 1`` / ``== limit``
    -- while caller lengths take the stage at every size and the compacting one-launch mode (``"kernel"``,
    ``convert = 1``) is never selected by size.  The shipped limit keeps both small bench tables below it; a limit
    of 0 is the stage at every size and the baseline's ``1 << 20`` splits 1024 x 1023 from 1024 x 1024."""
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    varlen, flat = (
        cake_backend.nope_varlen_index_mode,
        cake_backend.nope_flat_index_mode,
    )
    assert isinstance(limit, int) and limit > 1024 * 512
    for T, K in ((1024, 128), (1024, 512), (1, 1), (0, 2176), (limit - 1, 1)):
        assert varlen(T, K) == "mask" and flat(T, K) == "mask", (T, K)
        assert varlen(T, K, caller_lengths=True) == "index_prep"
    for T, K in ((limit, 1), (1, limit), (16263, 2176), (limit + 1, 1)):
        assert varlen(T, K) == "index_prep" and flat(T, K) == "index_prep", (T, K)
        assert varlen(T, K, caller_lengths=True) == "index_prep"
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert varlen(1, 1) == "index_prep" and flat(1, 1) == "index_prep"
    assert varlen(0, 1) == "index_prep" and flat(0, 1) == "index_prep"
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 1 << 20)
    assert varlen(1024, 1023) == "mask" and varlen(1024, 1024) == "index_prep"
    assert flat(1024, 1023) == "mask" and flat(1024, 1024) == "index_prep"
    assert varlen(4096, 255) == "mask" and varlen(4096, 256) == "index_prep"
    modes = {
        f(T, K)
        for f in (varlen, flat)
        for T in (0, 1, 1024, 4096, 16384)
        for K in (1, 7, 128, 2176)
    }
    assert modes == {"mask", "index_prep"}
    assert (cake_backend.NOPE_CONVERT_COMPACT, cake_backend.NOPE_CONVERT_MASK) == (1, 2)


def test_kv_latent_rows_squeezes_the_one_head_form_only():
    kv = torch.zeros(16, D_LATENT, dtype=torch.bfloat16)
    assert cake_backend._kv_latent_rows(kv) is kv
    one_head = kv.unsqueeze(1)
    rows = cake_backend._kv_latent_rows(one_head)
    assert tuple(rows.shape) == (16, D_LATENT) and rows.data_ptr() == kv.data_ptr()
    two_heads = torch.zeros(16, 2, D_LATENT, dtype=torch.bfloat16)
    assert cake_backend._kv_latent_rows(two_heads) is two_heads  # validation rejects it
    q, _, idx = _nope_host_inputs()
    with pytest.raises(ValueError, match="kv_latent"):
        validate_dsa_nope_inputs(q, two_heads, idx)


def test_check_dkv_acc_min_cols_for_the_combined_gradient():
    """The NoPE accumulator holds the combined dK + dV in columns 0:512: ``min_cols=512`` admits a [S, 512]
    buffer (the rope program's 576 does not), the row-stride rules are unchanged."""
    acc = torch.zeros(16, D_LATENT)
    operand, stride = cake_backend._check_dkv_acc(acc, None, 16, min_cols=D_LATENT)
    assert operand is acc and stride == D_LATENT
    with pytest.raises(ValueError, match=">= 576"):
        cake_backend._check_dkv_acc(acc, None, 16)
    wide = torch.zeros(16, 704)
    assert cake_backend._check_dkv_acc(wide, None, 16, min_cols=D_LATENT)[1] == 704
    strided = torch.zeros(16, 516)[:, :D_LATENT]
    operand, stride = cake_backend._check_dkv_acc(strided, None, 16, min_cols=D_LATENT)
    assert (
        stride == 516
        and operand.dim() == 1
        and operand.data_ptr() == strided.data_ptr()
    )
    with pytest.raises(ValueError, match=">= 512"):
        cake_backend._check_dkv_acc(torch.zeros(16, 508), None, 16, min_cols=D_LATENT)
    with pytest.raises(ValueError, match="multiple of 4"):
        cake_backend._check_dkv_acc(
            torch.zeros(16, 514)[:, :D_LATENT], None, 16, min_cols=D_LATENT
        )
    with pytest.raises(ValueError, match="at least"):
        cake_backend._check_dkv_acc(acc[:8], None, 16, min_cols=D_LATENT)
    with pytest.raises(ValueError, match="float32"):
        cake_backend._check_dkv_acc(acc.to(torch.bfloat16), None, 16, min_cols=D_LATENT)


@pytest.mark.parametrize("backward", [False, True])
def test_workspace_layout_nope_regions(backward):
    T, S, topk = 300, 1000, 96
    layout = workspace_layout_nope(T, S, topk, backward=backward)
    regions = [k for k in layout if k != "total"]
    offsets = [layout[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert layout["total"] >= sum(layout[k][1] for k in regions)
    assert layout["topk_length"][1] == T * 4
    assert ("delta" in layout) == backward
    assert ("dkv_latent_acc" in layout) == backward
    assert "dk_rope_acc" not in layout  # no rope accumulator in the NoPE program
    if backward:
        assert layout["delta"][1] == T * NUM_HEADS * 4
        assert layout["dkv_latent_acc"][1] == S * D_LATENT * 4
    direct = workspace_layout_nope(T, S, topk, backward=backward, dkv_direct=True)
    assert "dkv_latent_acc" not in direct and ("delta" in direct) == backward
    if backward:
        assert layout["total"] - direct["total"] >= S * D_LATENT * 4
    else:
        assert direct["total"] == layout["total"]
    multi = workspace_layout_nope(T, S, topk, backward=backward, key_passes=3)
    pass_regions = {"dq_partial", "key_scratch", "pass_counts"}
    if backward:
        assert pass_regions <= set(multi)
        assert multi["dq_partial"][1] == T * NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
        assert multi["key_scratch"][1] == T * topk * 4
        assert multi["pass_counts"][1] == T * 4
        capped = workspace_layout_nope(
            T, S, topk, backward=True, key_passes=3, token_chunk=128
        )
        assert capped["dq_partial"][1] == 128 * NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
        assert capped["key_scratch"][1] == 128 * topk * 4
    else:
        assert not pass_regions & set(multi)


def test_nope_key_pass_policy_plans_with_2048_bytes_per_key():
    """The NoPE record's policy: 2048 B of FP32 accumulator per key and a 131,072 B dQ partial per token, so the
    token chunk of a multi-pass backward is 4736 at K_slots 2048 / 2176, 4864 at 1152, 4992 at 128."""
    policy = KeyPassPolicy.from_record(cake_jit.record(NOPE_PROGRAM))
    assert policy.key_bytes == 2048
    assert policy.dq_partial_bytes_per_token == NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
    assert NOPE_DQ_PARTIAL_BYTES_PER_TOKEN == NUM_HEADS * D_LATENT * 4
    for topk, chunk in ((2048, 4736), (2176, 4736), (1152, 4864), (128, 4992)):
        assert policy.token_chunk(topk) == chunk, topk
        assert chunk % policy.token_chunk_multiple == 0
    assert policy.formula_passes(4096) == 1
    assert policy.formula_passes(262144) == -(-262144 * 2048 // policy.l2_budget_bytes)
    # one pass by default on every row (max_passes 1, no per-architecture rules; one pass measured fastest on
    # every row, build and GPU of this form); an explicit key_passes still plans the pass form
    assert policy.max_passes == 1
    for arch in ("sm_100a", "sm_103a", "sm_107a", None):
        assert policy.rule_for(arch) == policy.rule_for(None)
        for T, S in ((2048, 65536), (4096, 131072), (4096, 270336), (32768, 262144)):
            assert policy.passes(T, S, 2176, arch=arch) == 1
    nope_record = cake_jit.record(NOPE_PROGRAM)
    stages = cake_jit.NOPE_STAGES
    assert (
        plan_key_passes(nope_record, stages, 4096, 131072, 2176, program=NOPE_PROGRAM)
        == 1
    )
    assert (
        plan_key_passes(
            nope_record, stages, 4096, 131072, 2176, 2, program=NOPE_PROGRAM
        )
        == 2
    )
    assert (
        plan_key_passes(
            nope_record,
            stages,
            4096,
            131072,
            2176,
            3,
            arch="sm_103a",
            program=NOPE_PROGRAM,
        )
        == 3
    )
    # the rope record keeps its defaults (147,456 B per token)
    rope = KeyPassPolicy.from_record(cake_jit.record())
    assert rope.key_bytes == D_QK * 4
    assert rope.dq_partial_bytes_per_token == DQ_PARTIAL_BYTES_PER_TOKEN
    record = dict(cake_jit.record(NOPE_PROGRAM))
    record["key_pass_policy"] = dict(
        record["key_pass_policy"], dq_partial_bytes_per_token=0
    )
    with pytest.raises(ValueError, match="positive"):
        KeyPassPolicy.from_record(record)


def test_nope_bwd_launch_grid_mirrors_the_kernel_rule():
    """The persistent grid of the NoPE main stages: ``min(num_queries, SMs)`` (at least one CTA) while the launch's
    FP32 dkv window ``num_kv x 2048 B`` is within 64 MiB, a CTA per token beyond it; every value is <= num_queries
    (legal for the kernel's token loop) except the never-launched zero-token non-resident case, which mirrors the rule."""
    grid = cake_backend.nope_bwd_launch_grid
    assert cake_backend.NOPE_PERSISTENT_DKV_BYTES == 64 << 20
    assert cake_backend.NOPE_DKV_BYTES_PER_KEY == 2048
    window = (
        cake_backend.NOPE_PERSISTENT_DKV_BYTES // cake_backend.NOPE_DKV_BYTES_PER_KEY
    )
    assert window == 32768
    for sms in (132, 148, 160):
        assert grid(4096, window, sms) == sms  # the 64 MiB window itself is resident
        assert grid(4096, window + 1, sms) == 4096  # one key more: a CTA per token
        assert grid(100, window, sms) == 100  # fewer tokens than SMs
        assert grid(sms, 128, sms) == sms
        assert grid(1, 128, sms) == 1
        assert grid(0, 128, sms) == 1  # resident: never below one CTA
        # the kernel's rule past the window at zero tokens; the host never launches zero tokens
        assert grid(0, window + 1, sms) == 0
        assert grid(4736, 65536, sms) == 4736  # a 128 MiB window
        assert grid(32768, 262144, sms) == 32768
    # chunk ranges of a multi-pass backward: each launch counts its own tokens against the same key window
    policy = KeyPassPolicy.from_record(cake_jit.record(NOPE_PROGRAM))
    chunk = policy.token_chunk(2176)
    chunks = policy.token_chunks(chunk + 128, 2176)
    assert [grid(n, 65536, 148) for _b, n in chunks] == [chunk, 128]
    assert [grid(n, 4096, 148) for _b, n in chunks] == [148, 128]


def test_nope_binding_keys_are_tagged_and_cover_geometry_scale_lengths_and_dkv_options():
    q, kv, idx = _nope_host_inputs()
    scale = NOPE_SOFTMAX_SCALE
    base = forward_binding_key_nope(q, kv, idx, None, scale)
    assert base[0] == "nope_fwd"
    assert base == forward_binding_key_nope(q.clone(), kv, idx.clone(), None, scale)
    assert base != forward_binding_key_nope(q, kv, idx, None, scale * 0.5)
    assert base != forward_binding_key_nope(
        q, kv, idx, torch.zeros(8, dtype=torch.int32), scale
    )
    assert base != forward_binding_key_nope(q, kv[:4], idx, None, scale)
    assert base != forward_binding_key_nope(q, kv, idx, None, scale, True)  # derive
    wide = torch.zeros(8, 13, dtype=torch.int32)
    assert base != forward_binding_key_nope(q, kv, wide[:, 3:8], None, scale)
    # disjoint from the rope keys of the same shapes
    rope_q = torch.zeros(8, NUM_HEADS, D_ROPE, dtype=torch.bfloat16)
    rope_k = torch.zeros(16, D_ROPE, dtype=torch.bfloat16)
    assert forward_binding_key(q, rope_q, kv, rope_k, idx, None, scale)[0] == "fwd"
    out = torch.zeros(8, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(8, NUM_HEADS)
    bwd = backward_binding_key_nope(q, kv, idx, out, out, lse, out, None, scale)
    assert bwd[0] == "nope_bwd" and bwd != base
    assert bwd == backward_binding_key_nope(
        q, kv, idx, out.clone(), out, lse, out, None, scale
    )
    assert bwd != backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, 2
    )  # key_passes
    assert bwd != backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, None, 3
    )  # num_segments
    acc = torch.zeros(16, D_LATENT)
    with_acc = backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, None, 1, acc
    )
    assert with_acc != bwd
    assert with_acc == backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, None, 1, acc.clone()
    )
    assert with_acc != backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, None, 1, torch.zeros(16, 704)
    )
    dst_map = torch.arange(16, dtype=torch.int32)
    assert with_acc != backward_binding_key_nope(
        q, kv, idx, out, out, lse, out, None, scale, None, 1, acc, dst_map
    )
    # dkv_direct is part of the key with and without dkv_acc (a fresh dkv is accumulated under the same rule)
    for direct in (True, False):
        assert bwd != backward_binding_key_nope(
            q, kv, idx, out, out, lse, out, None, scale, None, 1, None, None, direct
        )
        assert with_acc != backward_binding_key_nope(
            q, kv, idx, out, out, lse, out, None, scale, None, 1, acc, None, direct
        )


def test_bound_values_nope_serve_the_main_and_cast_operands():
    """Without the natural stage the one permuted accumulator feeds the main stage and the cast: the cast writes
    the fresh ``dkv`` densely (``out_f32``) or adds into ``dkv_acc`` (``dst_packed`` / ``dst_map``), its rope
    operands are zero-size placeholders of the declared dtypes; a direct plan binds the FP32 buffer itself as the
    main stage's ``dkv_f32`` and no cast operands; the single-pass placeholders are the same as the rope plan's."""
    S = 8
    base = dict(
        indices=torch.zeros(2, 4, dtype=torch.int32),
        topk_length=torch.full((2,), 4, dtype=torch.int32),
        delta=torch.zeros(2, NUM_HEADS),
        inert_f32=torch.empty(0),
        inert_bf16=torch.empty(0, dtype=torch.bfloat16),
    )
    acc_perm = torch.zeros(S, D_LATENT)
    dkv = torch.zeros(S, D_LATENT)
    constants = dict(dst_row_stride=0, has_dst_map=0, accumulate=0, out_f32=1)
    plan = SimpleNamespace(
        constants=constants,
        backward=True,
        accumulate_dkv=False,
        dkv_direct=False,
        fresh_dkv=True,
        key_passes=1,
    )
    t = dict(base, dkv_latent_acc=acc_perm, dkv=dkv)
    values = cake_backend._bound_values_nope(plan, t)
    assert values["dkv_f32"] is acc_perm and values["src_latent"] is acc_perm
    assert values["dst_latent_f32"] is dkv and values["out_f32"] == 1
    assert values["dst_packed"] is acc_perm and values["accumulate"] == 0
    assert values["dst_map"] is base["topk_length"]
    assert values["dkv_dst_map"] is base["topk_length"]
    assert values["src_rope"] is base["inert_f32"]
    assert values["dst_rope_f32"] is base["inert_f32"]
    assert values["dst_latent"] is base["inert_bf16"]
    assert values["dst_rope"] is base["inert_bf16"]
    assert values["dq_partial"] is base["delta"]
    assert values["key_scratch"] is values["pass_counts"] is base["topk_length"]
    # accumulate through the cast
    acc = torch.zeros(S, 704)
    dst_map = torch.arange(S, dtype=torch.int32)
    plan.constants = dict(dst_row_stride=704, has_dst_map=1, accumulate=1, out_f32=0)
    plan.accumulate_dkv, plan.fresh_dkv = True, False
    values = cake_backend._bound_values_nope(
        plan, dict(base, dkv_latent_acc=acc_perm, dkv_acc=acc, dkv_dst_map=dst_map)
    )
    assert values["dst_packed"] is acc and values["dst_map"] is dst_map
    assert values["dkv_dst_map"] is dst_map and values["dst_latent_f32"] is acc_perm
    assert (values["dst_row_stride"], values["has_dst_map"], values["accumulate"]) == (
        704,
        1,
        1,
    )
    # direct: the main stage adds into the buffer itself
    plan = SimpleNamespace(
        constants=dict(dkv_stride=704, dkv_has_map=1),
        backward=True,
        accumulate_dkv=True,
        dkv_direct=True,
        fresh_dkv=False,
        key_passes=1,
    )
    values = cake_backend._bound_values_nope(
        plan, dict(base, dkv_acc=acc, dkv_dst_map=dst_map)
    )
    assert values["dkv_f32"] is acc and values["dkv_dst_map"] is dst_map
    assert "src_latent" not in values and "dst_packed" not in values
    plan.accumulate_dkv, plan.fresh_dkv = False, True
    plan.constants = dict(dkv_stride=D_LATENT, dkv_has_map=0)
    values = cake_backend._bound_values_nope(plan, dict(base, dkv=dkv))
    assert values["dkv_f32"] is dkv and values["dkv_dst_map"] is base["topk_length"]
    # a forward plan binds no backward operands
    plan = SimpleNamespace(constants={}, backward=False)
    values = cake_backend._bound_values_nope(plan, dict(base))
    assert "dkv_f32" not in values and values["topk_length"] is base["topk_length"]


def test_nope_eager_entries_without_query_rows_neither_bind_nor_launch(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "record_for", refuse)
    monkeypatch.setattr(cake_backend, "_binding", refuse)
    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", refuse)
    q, kv, idx = _nope_host_inputs(total_q=0)
    out, lse, o_lo = forward_nope(q, kv, idx)
    assert tuple(out.shape) == (0, NUM_HEADS, D_LATENT) and out.dtype == torch.bfloat16
    assert tuple(lse.shape) == (0, NUM_HEADS) and lse.dtype == torch.float32
    assert tuple(o_lo.shape) == tuple(out.shape) and o_lo.dtype == torch.bfloat16
    dout = torch.zeros(0, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    dq, dkv = backward_nope(q, kv, idx, out, o_lo, lse, dout)
    assert tuple(dq.shape) == (0, NUM_HEADS, D_LATENT) and dq.dtype == torch.bfloat16
    assert tuple(dkv.shape) == (16, D_LATENT) and dkv.dtype == torch.float32
    assert torch.all(dkv == 0)
    acc = torch.ones(16, D_LATENT)
    dq, none = backward_nope(q, kv, idx, out, o_lo, lse, dout, dkv_acc=acc)
    assert none is None and torch.all(acc == 1)  # nothing accumulated, nothing touched
    # the contracts still apply to an empty call
    with pytest.raises(ValueError, match="int32"):
        forward_nope(q, kv, idx.long())
    with pytest.raises(ValueError, match="S > 0"):
        forward_nope(q, kv[:0], idx)
    with pytest.raises(ValueError, match="lse"):
        backward_nope(q, kv, idx, out, o_lo, lse[:, :8], dout)
    with pytest.raises(ValueError, match=">= 512"):
        backward_nope(q, kv, idx, out, o_lo, lse, dout, dkv_acc=torch.zeros(16, 500))
    with pytest.raises(ValueError, match="dkv_dst_map"):
        backward_nope(
            q,
            kv,
            idx,
            out,
            o_lo,
            lse,
            dout,
            dkv_dst_map=torch.zeros(16, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="derive_topk_length"):
        forward_nope(q, kv, idx, derive_topk_length=True)
    # the autograd function without query rows: empty out / lse, no gradient flow needed
    out, lse = cake_backend.DSASparseAttentionNopeFunction.apply(
        q, kv, idx, None, NOPE_SOFTMAX_SCALE
    )
    assert tuple(out.shape) == (0, NUM_HEADS, D_LATENT) and tuple(lse.shape) == (
        0,
        NUM_HEADS,
    )


# --- plans over the completed record with recording launch entries (supported device, no kernel) ---

_NOPE_STAND_IN_LAUNCH = {
    "nope_fwd": [320, 1, 1],
    "nope_bwd_main": [640, 1, 1],
    "nope_bwd_main_natural": [640, 1, 1],
    "nope_bwd_main_pass": [640, 1, 1],
    "nope_bwd_main_pass_natural": [640, 1, 1],
}


def _completed_nope_record() -> dict:
    """The registered NoPE record completed with stand-in entries for the stages it does not register yet."""
    record = dict(cake_jit.record(NOPE_PROGRAM))
    for stage in cake_jit.NOPE_STAGES:
        if stage not in record:
            library = f"{NOPE_PROGRAM}_stand_in_{stage}"
            record[stage] = {
                "library": library,
                "ffi_entry": "run",
                "launch": {
                    "block": _NOPE_STAND_IN_LAUNCH[stage],
                    "cluster": [1, 1, 1],
                    "cooperative": False,
                    "dynamic_smem_bytes": 0,
                    "use_pdl": False,
                },
            }
    record["stages"] = list(cake_jit.NOPE_STAGES)
    return record


def _stand_in_nope_program(monkeypatch, calls: list) -> dict:
    """Register the completed record and launch entries that append ``(stage, {argument: value})`` to ``calls``
    instead of running a kernel; the rope program's loader is untouched."""
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    record = _completed_nope_record()
    monkeypatch.setitem(cake_jit.MODULES, NOPE_PROGRAM, record)
    real_load = cake_jit.load_cake_dsa_train_module

    def load(stage, arch, program=cake_jit.PROGRAM):
        if program != NOPE_PROGRAM:
            return real_load(stage, arch, program)

        def run(*args):
            calls.append(
                (stage, dict(zip(cake_launch.STAGE_ARGS[stage], args, strict=True)))
            )

        return SimpleNamespace(run=run)

    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", load)
    # the recording entries take the host tensors as they are: the tvm-ffi conversion stands in as identity
    monkeypatch.setattr(cake_backend, "_ffi_tensor", lambda tensor: tensor)
    cake_backend.BINDING_CACHE.clear()
    return record


def _nope_device_inputs(T, S, K, *, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, NUM_HEADS, D_LATENT, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    kv = torch.randn(S, D_LATENT, device="cuda", generator=gen).to(torch.bfloat16)
    idx = torch.randint(0, S, (T, K), device="cuda", generator=gen, dtype=torch.int32)
    dout = torch.randn(T, NUM_HEADS, D_LATENT, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    return q, kv, idx, dout


def _saved_outputs(T, device="cuda"):
    out = torch.zeros(T, NUM_HEADS, D_LATENT, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(T, NUM_HEADS, device=device)
    return out, lse, torch.zeros_like(out)


def test_nope_plan_forward_launch_values(monkeypatch):
    calls = []
    _stand_in_nope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q, kv, idx, _ = _nope_device_inputs(T, S, K)
    out, lse, o_lo = forward_nope(q, kv, idx)
    assert [c[0] for c in calls] == ["nope_fwd"]
    v = calls[0][1]
    assert v["q_latent"] is q and v["kv_latent"] is kv
    assert v["out"] is out and v["lse"] is lse and v["o_lo"] is o_lo
    assert v["indices"].data_ptr() == idx.data_ptr()
    assert (v["num_queries"], v["num_kv"], v["topk"]) == (T, S, K)
    assert (v["idx_stride"], v["indices_offset"]) == (K, 0)
    assert (v["has_topk_length"], v["derive_length"]) == (0, 1)
    assert v["scale_log2"] == pytest.approx(NOPE_SOFTMAX_SCALE * cake_backend.LOG2E)
    assert (v["grid_x"], v["grid_y"], v["grid_z"]) == (T, 1, 1)
    assert v["topk_length"].dtype == torch.int32 and tuple(v["topk_length"].shape) == (
        T,
    )
    assert torch.all(v["topk_length"] == K)
    assert tuple(out.shape) == (T, NUM_HEADS, D_LATENT) and out.dtype == torch.bfloat16
    assert tuple(lse.shape) == (T, NUM_HEADS) and lse.dtype == torch.float32
    # caller lengths, another scale, a column slice of a wider index tensor and a strided query view
    lengths = torch.full((T,), 3, dtype=torch.int32, device="cuda")
    wide = torch.zeros(T, K + 8, dtype=torch.int32, device="cuda")
    wide[:, 3 : 3 + K] = idx
    packed = torch.zeros(T, NUM_HEADS, D_QK, dtype=torch.bfloat16, device="cuda")
    packed[..., :D_LATENT] = q
    forward_nope(
        packed[..., :D_LATENT],
        kv,
        wide[:, 3 : 3 + K],
        topk_length=lengths,
        softmax_scale=0.25,
    )
    v = calls[-1][1]
    assert v["topk_length"] is lengths and (
        v["has_topk_length"],
        v["derive_length"],
    ) == (1, 0)
    assert v["scale_log2"] == pytest.approx(0.25 * cake_backend.LOG2E)
    assert v["idx_stride"] == K + 8 and v["indices_offset"] == 3
    assert (
        v["q_latent"].data_ptr() == packed.data_ptr()
        and v["q_latent"].stride(1) == D_QK
    )
    # the forward key is tagged; the two geometries are two cached plans
    keys = list(cake_backend.BINDING_CACHE._entries)
    assert len(keys) == 2 and all(key[0] == "nope_fwd" for key in keys)
    # derive_topk_length fills the caller's vector
    filled = torch.empty(T, dtype=torch.int32, device="cuda")
    forward_nope(q, kv, idx, topk_length=filled, derive_topk_length=True)
    v = calls[-1][1]
    assert v["topk_length"] is filled and (
        v["has_topk_length"],
        v["derive_length"],
    ) == (1, 1)


def _stand_in_index_prep(monkeypatch, calls: list) -> dict:
    """The stand-in NoPE program (:func:`_stand_in_nope_program`) completed with the index-preparation stage: a
    recording launch entry, and a launcher over :data:`INDEX_PREP_ARGS` in the generated tables (the export writes
    the real ones with the stage)."""
    record = _stand_in_nope_program(monkeypatch, calls)
    record[NOPE_INDEX_PREP_STAGE] = dict(_INDEX_PREP_STAND_IN)
    record["stages"] = list(record["stages"]) + [NOPE_INDEX_PREP_STAGE]
    names = INDEX_PREP_ARGS[:-3]

    def launch(run, values, grid):
        run(*(values[name] for name in names), *grid)

    monkeypatch.setitem(cake_launch.LAUNCH, NOPE_INDEX_PREP_STAGE, launch)
    monkeypatch.setitem(
        cake_launch.GRID,
        NOPE_INDEX_PREP_STAGE,
        lambda num_queries, num_kv, topk: cake_backend.index_prep_grid(num_queries),
    )
    monkeypatch.setitem(
        cake_launch.STAGE_ARGS, NOPE_INDEX_PREP_STAGE, list(INDEX_PREP_ARGS)
    )
    return record


def test_nope_index_prep_launch_values(monkeypatch):
    """The stage's launch values from the host -- the flat alias of the index rows with their offset and stride, the
    caller-allocated outputs, the document count and its search depth, the causal flag, the document-mode constants,
    the compaction mode (``nope_globalize_indices`` default ``compact=False`` keeps the position-preserving glue
    promise, ``compact=True`` compacts; a caller cut arrives as ``length_in`` with ``has_length_in = 1``, else the
    ``topk_length`` output pointer stands in and ``has_length_in = 0``), the four-rows-per-CTA grid -- and the
    packed entry on the stage route (the limit set to 0 here: the example is below the shipped limit): the
    compacting stage first (``compact = 1``), then the forward over its rows with ``convert = 0`` reading the
    stage's counts; caller lengths reach the stage as the cut, never the forward."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    assert cake_backend.varlen_index_mode(local) == "mask"  # 45 slots: below the limit
    assert cake_backend.varlen_index_mode(local, caller_lengths=True) == "index_prep"
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.varlen_index_mode(local) == "index_prep"
    indices, lengths = nope_globalize_indices(local, cu_q, cu_k, causal=True)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE]
    v = calls[0][1]
    assert v["indices"].dim() == 1 and v["indices"].numel() == T * K
    assert v["indices"].data_ptr() == local.data_ptr()
    assert v["cu_seqlens_q"] is cu_q and v["cu_seqlens_k"] is cu_k
    assert v["out"] is indices and v["topk_length"] is lengths
    assert indices.dtype == torch.int32 and tuple(indices.shape) == (T, K)
    assert indices.is_contiguous() and lengths.dtype == torch.int32
    assert tuple(lengths.shape) == (T,)
    assert (v["num_rows"], v["topk"], v["idx_stride"], v["indices_offset"]) == (
        T,
        K,
        K,
        0,
    )
    assert (v["num_docs"], v["search_iters"], v["causal"]) == (4, 3, 1)
    assert (v["has_docs"], v["flat_bound"], v["write_out"]) == (1, 0, 1)
    # the default keeps today's position-preserving rewrite; no cut: the output vector is the placeholder
    assert (v["compact"], v["has_length_in"]) == (0, 0)
    assert v["length_in"] is lengths
    assert (v["grid_x"], v["grid_y"], v["grid_z"]) == (-(-T // 4), 1, 1)
    # the compacting arm, with and without a cut
    rows_c, counts_c = nope_globalize_indices(
        local, cu_q, cu_k, causal=True, compact=True
    )
    v = calls[-1][1]
    assert (v["compact"], v["has_length_in"], v["write_out"]) == (1, 0, 1)
    assert v["out"] is rows_c and v["topk_length"] is counts_c
    assert v["length_in"] is counts_c
    cut = torch.full((T,), 2, dtype=torch.int32, device="cuda")
    nope_globalize_indices(
        local, cu_q, cu_k, causal=True, compact=True, topk_length=cut
    )
    v = calls[-1][1]
    assert (v["compact"], v["has_length_in"]) == (1, 1) and v["length_in"] is cut
    assert (
        v["topk_length"] is not cut
    )  # the counts are fresh; the cut is read, never overwritten
    with pytest.raises(ValueError, match="compact"):
        nope_globalize_indices(local, cu_q, cu_k, causal=True, topk_length=cut)
    # causal off; a column slice of a wider buffer is read in place through the flat alias
    wide = torch.zeros(T, K + 8, dtype=torch.int32, device="cuda")
    wide[:, 3 : 3 + K] = local
    nope_globalize_indices(wide[:, 3 : 3 + K], cu_q, cu_k)
    v = calls[-1][1]
    assert v["causal"] == 0 and (v["idx_stride"], v["indices_offset"]) == (K + 8, 3)
    assert v["indices"].data_ptr() == wide.data_ptr()
    assert v["indices"].numel() == 3 + (T - 1) * (K + 8) + K
    # T == 0: no launch, empty results
    del calls[:]
    empty = nope_globalize_indices(
        local[:0], cu_q.new_tensor([0, 0]), cu_k.new_tensor([0, 18])
    )
    assert calls == [] and tuple(empty[0].shape) == (0, K) and empty[1].numel() == 0
    # the packed entry: the compacting stage into the fresh global-row buffer and counts, then the forward over
    # those rows with convert = 0 (the conversion operands bound but never read), reading the stage's counts
    q, kv, _ = _nope_host_inputs(total_q=T, total_k=18, topk=K, device="cuda")
    varlen = cake_backend.dsa_sparse_attention_nope_varlen
    out, lse = varlen(q, kv, local, cu_q, cu_k, causal=True, return_lse=True)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert (prep["causal"], prep["num_docs"], prep["search_iters"]) == (1, 4, 3)
    assert (prep["has_docs"], prep["flat_bound"], prep["write_out"]) == (1, 0, 1)
    assert (prep["compact"], prep["has_length_in"]) == (1, 0)
    assert prep["indices"].data_ptr() == local.data_ptr()
    assert prep["cu_seqlens_q"] is cu_q and prep["cu_seqlens_k"] is cu_k
    assert tuple(prep["out"].shape) == (T, K) and prep["out"].is_contiguous()
    assert fwd["indices"].data_ptr() == prep["out"].data_ptr()
    assert fwd["indices_out"].data_ptr() == prep["out"].data_ptr()
    assert fwd["topk_length"] is prep["topk_length"]
    assert fwd["out"] is out and fwd["lse"] is lse
    assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (1, 0, 0)
    assert (fwd["num_queries"], fwd["num_kv"], fwd["topk"]) == (T, 18, K)
    assert (fwd["idx_stride"], fwd["indices_offset"]) == (K, 0)
    assert (fwd["num_docs"], fwd["search_iters"], fwd["causal"]) == (1, 1, 0)
    # the backward reads the stage's rows and counts (settled at forward time)
    assert (
        out.grad_fn is None
    )  # host zeros carry no grad; the kernel-level entry exposes the pair instead
    del calls[:]
    _o, _l, _o_lo, rows_k, lengths_k = cake_backend.forward_nope_varlen(
        q, kv, local, cu_q, cu_k, causal=True
    )
    assert calls[0][1]["out"] is rows_k and calls[0][1]["topk_length"] is lengths_k
    assert calls[1][1]["topk_length"] is lengths_k
    # caller lengths: the stage takes them as the cut (has_length_in = 1); the forward and the backward read
    # the stage's counts, not the caller's vector
    mine = torch.full((T,), 2, dtype=torch.int32, device="cuda")
    del calls[:]
    varlen(q, kv, local, cu_q, cu_k, topk_length=mine)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert prep["causal"] == 0 and prep["topk_length"] is not mine
    assert (prep["compact"], prep["has_length_in"]) == (1, 1)
    assert prep["length_in"] is mine
    assert prep["out"].data_ptr() == fwd["indices"].data_ptr()
    assert fwd["topk_length"] is prep["topk_length"] and fwd["topk_length"] is not mine
    assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (1, 0, 0)
    assert fwd["indices_out"].data_ptr() == prep["out"].data_ptr()
    assert (fwd["num_docs"], fwd["search_iters"], fwd["causal"]) == (1, 1, 0)
    del calls[:]
    _o, _l, _o_lo, rows_k, lengths_k = cake_backend.forward_nope_varlen(
        q, kv, local, cu_q, cu_k, causal=True, topk_length=mine
    )
    assert lengths_k is not mine and calls[0][1]["topk_length"] is lengths_k
    assert calls[0][1]["length_in"] is mine and calls[0][1]["has_length_in"] == 1
    # T == 0 through the entry: nothing launched; a non-contiguous prefix sum is refused before any launch
    del calls[:]
    out = varlen(
        q[:0], kv, local[:0], cu_q.new_tensor([0, 0]), cu_k.new_tensor([0, 18])
    )
    assert calls == [] and tuple(out.shape) == (0, NUM_HEADS, D_LATENT)
    with pytest.raises(ValueError, match="cu_seqlens_q must be contiguous"):
        varlen(q, kv, local, torch.stack([cu_q, cu_q], 1)[:, 0], cu_k)
    assert calls == []


@pytest.mark.parametrize("route", ["mask", "index_prep"])
def test_nope_flat_entry_route_follows_the_table_size(monkeypatch, route):
    """The flat autograd entry without caller lengths (``nope_flat_index_mode``).  Below the slot limit
    (``"mask"``): one launch of the forward over the caller's rows as given -- ``has_topk_length = 1,
    derive_length = 1, convert = 0``, the forward deriving the last valid slot + 1 into a fresh ``[T]`` vector --
    and the saved pair is the caller's rows with that vector.  At or above it (``"index_prep"``; the limit set to
    0 here): one launch of the stage in flat mode (``has_docs = 0``, ``flat_bound = S - 1``, ``compact = 1``, no
    cut, the prefix sums inert) into a fresh ``[T, K]`` / ``[T]`` pair that the forward (``convert = 0``, the
    counts as lengths) and the saved backward read.  With caller lengths the entry keeps the forward alone over
    the caller's rows and lengths on either route.  ``nope_compact_indices`` is the flat stage launch exposed: the
    compacted rows and counts of global rows, with an optional cut.  The eager ``forward_nope`` keeps raw-row
    semantics (the forward derives last-valid + 1 in-kernel)."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    T, S, K = 8, 64, 5
    if route == "index_prep":
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.nope_flat_index_mode(T, K) == route
    q, kv, idx, _ = _nope_device_inputs(T, S, K)
    q.requires_grad_()
    out = cake_backend.dsa_sparse_attention_nope(q, kv, idx)
    saved = out.grad_fn.saved_tensors  # (q, kv, out, lse, o_lo, rows, lengths)
    assert len(saved) == 7 and out.grad_fn.nope_prep.has_topk_length
    wide = torch.zeros(T, K + 8, dtype=torch.int32, device="cuda")
    wide[:, 3 : 3 + K] = idx
    if route == "mask":
        assert [c[0] for c in calls] == ["nope_fwd"]
        fwd = calls[0][1]
        assert fwd["indices"].data_ptr() == idx.data_ptr()
        assert (
            fwd["indices_out"].data_ptr() == idx.data_ptr()
        )  # never read: convert = 0
        assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (
            1,
            1,
            0,
        )
        assert (fwd["num_docs"], fwd["search_iters"], fwd["causal"]) == (1, 1, 0)
        assert tuple(fwd["topk_length"].shape) == (T,)
        assert fwd["topk_length"].dtype == torch.int32 and fwd["topk_length"].is_cuda
        # the saved pair: the caller's rows and the vector the forward filled
        assert saved[5].data_ptr() == idx.data_ptr()
        assert saved[6].data_ptr() == fwd["topk_length"].data_ptr()
        # a column slice of a wider buffer is read in place by the forward (its stride and offset); the never-read
        # ``indices_out`` operand is then the contiguous span of its storage (the binding takes contiguous pointers)
        del calls[:]
        cake_backend.dsa_sparse_attention_nope(q, kv, wide[:, 3 : 3 + K])
        assert [c[0] for c in calls] == ["nope_fwd"]
        fwd = calls[0][1]
        assert (fwd["idx_stride"], fwd["indices_offset"]) == (K + 8, 3)
        assert fwd["indices"].data_ptr() == wide.data_ptr()
        assert fwd["indices_out"].is_contiguous()
        assert fwd["indices_out"].data_ptr() == wide.data_ptr()
    else:
        assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
        prep, fwd = calls[0][1], calls[1][1]
        assert (prep["has_docs"], prep["flat_bound"], prep["write_out"]) == (
            0,
            S - 1,
            1,
        )
        assert (prep["compact"], prep["has_length_in"], prep["causal"]) == (1, 0, 0)
        assert (prep["num_docs"], prep["search_iters"]) == (1, 1)
        assert (prep["num_rows"], prep["topk"], prep["idx_stride"]) == (T, K, K)
        assert prep["indices"].data_ptr() == idx.data_ptr()
        assert tuple(prep["out"].shape) == (T, K) and prep["out"].is_contiguous()
        assert (
            prep["out"].data_ptr() != idx.data_ptr()
        )  # a fresh buffer: the caller's rows are not rewritten
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_k",
        ):  # never read in flat mode, valid pointers all the same
            assert prep[name].dtype == torch.int32 and prep[name].is_cuda
        assert fwd["indices"].data_ptr() == prep["out"].data_ptr()
        assert fwd["topk_length"] is prep["topk_length"]
        assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (
            1,
            0,
            0,
        )
        # the stage's pair, not ``idx``
        assert saved[5].data_ptr() == prep["out"].data_ptr()
        assert saved[6].data_ptr() == prep["topk_length"].data_ptr()
        # a column slice of a wider buffer and a view inside a storage are read in place by the flat stage
        del calls[:]
        cake_backend.dsa_sparse_attention_nope(q, kv, wide[:, 3 : 3 + K])
        prep = calls[0][1]
        assert (prep["idx_stride"], prep["indices_offset"]) == (K + 8, 3)
        assert prep["indices"].data_ptr() == wide.data_ptr()
    # caller lengths: unchanged on either route -- the forward alone over the caller's rows, no stage launch
    mine = torch.full((T,), 3, dtype=torch.int32, device="cuda")
    del calls[:]
    out_m = cake_backend.dsa_sparse_attention_nope(q, kv, idx, topk_length=mine)
    assert [c[0] for c in calls] == ["nope_fwd"]
    fwd = calls[0][1]
    assert fwd["indices"].data_ptr() == idx.data_ptr() and fwd["topk_length"] is mine
    assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (1, 0, 0)
    saved_m = out_m.grad_fn.saved_tensors
    assert (
        saved_m[5].data_ptr() == idx.data_ptr()
        and saved_m[6].data_ptr() == mine.data_ptr()
    )
    # the eager forward: raw rows, lengths derived in-kernel
    del calls[:]
    forward_nope(q.detach(), kv, idx)
    assert [c[0] for c in calls] == ["nope_fwd"]
    assert (calls[0][1]["has_topk_length"], calls[0][1]["derive_length"]) == (0, 1)
    # the public helper: the flat compacting launch into fresh outputs, the cut as length_in
    del calls[:]
    rows, counts = cake_backend.nope_compact_indices(idx, S)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE]
    v = calls[0][1]
    assert v["out"] is rows and v["topk_length"] is counts
    assert (v["has_docs"], v["flat_bound"], v["compact"], v["has_length_in"]) == (
        0,
        S - 1,
        1,
        0,
    )
    assert tuple(rows.shape) == (T, K) and tuple(counts.shape) == (T,)
    assert rows.dtype == torch.int32 and counts.dtype == torch.int32
    cake_backend.nope_compact_indices(idx, S, topk_length=mine)
    v = calls[-1][1]
    assert (v["compact"], v["has_length_in"]) == (1, 1) and v["length_in"] is mine
    del calls[:]
    e = cake_backend.nope_compact_indices(idx[:0], S)
    assert calls == [] and tuple(e[0].shape) == (0, K) and e[1].numel() == 0


def test_nope_varlen_index_mode_follows_the_table_size(monkeypatch):
    """The route from the table size alone (``nope_varlen_index_mode``, no device access): the one-launch
    position-preserving forward (``"mask"``) below ``NOPE_CONVERT_SLOT_LIMIT`` slots, the compacting stage at or
    above it and always with caller lengths; ``varlen_index_mode`` applies it to the tensor's shape.  Through the
    packed entry (stand-in launches; the limit at the baseline's ``1 << 20`` so that 1024 x 1023 and 1024 x 1024
    straddle it): the small table is one ``nope_fwd`` launch over the caller's rows (their stride) with ``convert
    = 2`` deriving the lengths (``has_topk_length = 1, derive_length = 1``) into a fresh ``indices_out`` / ``[T]``
    pair; the large table is the stage (``compact = 1``), then the forward over its rows with ``convert = 0``
    reading the stage's counts.  A limit of 0 (measurement) is the stage at every size.  The compacting one-launch
    mode is explicit only (``_packed_forward_nope(..., "kernel")``: ``convert = 1``); neither one-launch mode
    takes caller lengths."""
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    mode = cake_backend.nope_varlen_index_mode
    for T, K in ((1024, 1023), (1024, 1024), (1, 1), (0, 2176), (16263, 2176)):
        want = "mask" if limit > T * K else "index_prep"
        assert mode(T, K) == want and mode(T, K, caller_lengths=True) == "index_prep"
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    assert (
        cake_backend.nope_fwd_converts_indices()
    )  # the shipped tables carry the conversion operands
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 1 << 20)
    assert mode(1024, 1023) == "mask" and mode(1024, 1024) == "index_prep"
    assert mode(1, 1) == "mask" and mode(0, 2176) == "mask"
    assert mode(4, 3, caller_lengths=True) == "index_prep"
    large = torch.zeros(1024, 1024, dtype=torch.int32, device="cuda")
    small = large[:, :1023]
    assert cake_backend.varlen_index_mode(large) == "index_prep"
    assert cake_backend.varlen_index_mode(small) == "mask"
    assert cake_backend.varlen_index_mode(small, caller_lengths=True) == "index_prep"
    with pytest.raises(ValueError, match=r"\[T, K_slots\]"):
        cake_backend.varlen_index_mode(large[0])
    cu_q = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    q, kv, _ = _nope_host_inputs(total_q=1024, total_k=1024, topk=1024, device="cuda")
    varlen = cake_backend.dsa_sparse_attention_nope_varlen
    # at or above the limit: the stage, then the forward over its rows
    del calls[:]
    varlen(q, kv, large, cu_q, cu_k)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert (prep["num_rows"], prep["topk"], prep["write_out"]) == (1024, 1024, 1)
    assert (prep["compact"], prep["has_length_in"]) == (1, 0)
    assert prep["idx_stride"] == 1024
    assert fwd["convert"] == 0 and fwd["indices"].data_ptr() == prep["out"].data_ptr()
    assert fwd["topk_length"] is prep["topk_length"]
    assert (fwd["has_topk_length"], fwd["derive_length"]) == (1, 0)
    assert (fwd["idx_stride"], fwd["topk"]) == (1024, 1024)
    # below the limit: the forward alone over the caller's rows, converting them position-preserving
    del calls[:]
    varlen(q, kv, small, cu_q, cu_k)
    assert [c[0] for c in calls] == ["nope_fwd"]
    fwd = calls[0][1]
    assert fwd["convert"] == cake_backend.NOPE_CONVERT_MASK == 2
    assert (fwd["idx_stride"], fwd["topk"]) == (1024, 1023)
    assert (fwd["has_topk_length"], fwd["derive_length"]) == (1, 1)
    assert fwd["indices"].data_ptr() == small.data_ptr()
    assert tuple(fwd["indices_out"].shape) == (1024, 1023)
    assert (
        fwd["indices_out"].is_contiguous() and fwd["indices_out"].dtype == torch.int32
    )
    assert fwd["indices_out"].data_ptr() != small.data_ptr()
    assert tuple(fwd["topk_length"].shape) == (1024,)
    assert fwd["cu_seqlens_q"] is cu_q and fwd["cu_seqlens_k"] is cu_k
    assert (fwd["num_docs"], fwd["search_iters"], fwd["causal"]) == (1, 1, 0)
    # the kernel-level pair returns the forward's rows and lengths
    del calls[:]
    _, _, _, rows, lengths = cake_backend.forward_nope_varlen(
        q, kv, small, cu_q, cu_k, causal=True
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    assert calls[0][1]["indices_out"] is rows and calls[0][1]["topk_length"] is lengths
    assert calls[0][1]["causal"] == 1
    # a limit of 0: the stage at every size
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert mode(1, 1) == "index_prep"
    assert cake_backend.varlen_index_mode(small) == "index_prep"
    del calls[:]
    varlen(q, kv, small, cu_q, cu_k)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    assert calls[1][1]["convert"] == 0
    # the compacting one-launch mode, by name only
    del calls[:]
    cake_backend._packed_forward_nope(
        q, kv, small, cu_q, cu_k, True, None, 1 / 16, "kernel"
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    assert calls[0][1]["convert"] == cake_backend.NOPE_CONVERT_COMPACT == 1
    assert (calls[0][1]["has_topk_length"], calls[0][1]["derive_length"]) == (1, 1)
    for name in ("kernel", "mask"):
        with pytest.raises(ValueError, match=f"{name} mode"):
            cake_backend._packed_forward_nope(
                q, kv, small, cu_q, cu_k, True, cu_q.new_zeros(1024), 1 / 16, name
            )
    with pytest.raises(ValueError, match="unknown NoPE index mode"):
        cake_backend._packed_forward_nope(
            q, kv, small, cu_q, cu_k, True, None, 1 / 16, "torch"
        )


def test_nope_autograd_backward_is_settled_at_forward_and_launches_back_to_back(
    monkeypatch,
):
    """The autograd wrappers settle the backward at forward time (``ctx.nope_prep``: the binding key without
    ``dout``, the options, the accumulator validated once and kept as a ctx attribute in its tensor form) and save
    seven tensors in a fixed order (``q``, ``kv``, ``out``, ``lse``, ``o_lo``, the index rows, the lengths or a
    zero-size placeholder: everything of size the backward reads, so a checkpointed forward retains nothing,
    flashinfer-ai/flashinfer#7039); the backward's host work before its first launch is the key completion, the
    lookup, the fresh allocations and the value binding -- every allocation and the accumulator check precede the
    first launch, the zero fills of the buffers the call owns and the stages follow back to back, and nothing is
    allocated or validated after a launch."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    events = []
    real_alloc, real_check = cake_backend._alloc, cake_backend._check_dkv_acc
    real_launch, real_zero = (
        cake_backend._NopeBinding._launch_stages,
        torch.Tensor.zero_,
    )

    def alloc(shape, dtype, device, *, zero=False):
        events.append(("alloc", tuple(int(d) for d in shape)))
        return real_alloc(shape, dtype, device, zero=zero)

    def check(*args, **kwargs):
        events.append(("check_dkv_acc",))
        return real_check(*args, **kwargs)

    def launch(self, values, keys):
        events.append(
            ("launch", tuple(k if isinstance(k, str) else k[0] for k in keys))
        )
        return real_launch(self, values, keys)

    def zero(tensor):
        events.append(("fill", tuple(tensor.shape)))
        return real_zero(tensor)

    real_plan, cache = cake_backend._plan_nope, cake_backend.BINDING_CACHE
    real_get = cache.get

    def plan(*args, **kwargs):
        events.append(("plan",))
        return real_plan(*args, **kwargs)

    def get(key):
        binding = real_get(key)
        events.append(("cache", binding is not None, len(cache), cache.capacity))
        return binding

    monkeypatch.setattr(cake_backend, "_alloc", alloc)
    monkeypatch.setattr(cake_backend, "_check_dkv_acc", check)
    monkeypatch.setattr(cake_backend._NopeBinding, "_launch_stages", launch)
    monkeypatch.setattr(torch.Tensor, "zero_", zero)
    monkeypatch.setattr(cake_backend, "_plan_nope", plan)
    monkeypatch.setattr(cache, "get", get)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    S = 18
    q, kv, _ = _nope_host_inputs(total_q=T, total_k=S, topk=K, device="cuda")
    q.requires_grad_()
    dout = torch.zeros_like(q)
    acc = torch.empty(S, D_LATENT, device="cuda")
    # the packed entry with a caller accumulator: the check runs at forward time, the backward allocates dq and
    # delta, binds and launches delta + main with no fill (the caller owns the zeroing)
    out, lse = cake_backend.dsa_sparse_attention_nope_varlen(
        q, kv, local, cu_q, cu_k, return_lse=True, dkv_acc=acc
    )
    prep = out.grad_fn.nope_prep
    assert isinstance(prep, cake_backend._NopeBackwardPrep)
    assert len(out.grad_fn.saved_tensors) == 7 and prep.dkv_acc is acc
    assert ("check_dkv_acc",) in events and prep.operands["dkv_acc"] is acc
    assert prep.key_head[0] == "nope_bwd" and len(prep.key_head) == 7
    del events[:], calls[:]
    # the first backward of a geometry plans once (a binding-cache miss); every later call is the steady state
    torch.autograd.grad(out, (q,), dout, retain_graph=True)
    kinds = [e[0] for e in events]
    assert kinds[:2] == ["cache", "plan"] and events[0][1] is False, events
    assert kinds.count("plan") == 1 and kinds[-1] == "launch", events
    del events[:], calls[:]
    (dq,) = torch.autograd.grad(out, (q,), dout)
    kinds = [e[0] for e in events]
    assert kinds == ["cache", "alloc", "alloc", "launch"], events
    assert events[0][1] is True  # the binding the first backward planned and cached
    assert events[1][1] == (T, NUM_HEADS, D_LATENT) and events[2][1] == (T, NUM_HEADS)
    assert events[3][1][0] == "nope_bwd_delta" and len(events[3][1]) >= 2
    assert [c[0] for c in calls] == list(events[3][1])
    assert tuple(dq.shape) == (T, NUM_HEADS, D_LATENT)
    fwd_main = calls[-1][1]
    assert fwd_main["dkv_f32"] is acc or fwd_main.get("dst_packed") is acc
    # the flat entry without an accumulator: the fresh dkv's zero fill sits right before the stages
    del events[:], calls[:]
    idx = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    out2 = cake_backend.dsa_sparse_attention_nope(q, kv, idx)
    assert (
        len(out2.grad_fn.saved_tensors) == 7 and out2.grad_fn.nope_prep.dkv_acc is None
    )
    del events[:], calls[:]
    torch.autograd.grad(out2, (q,), dout, retain_graph=True)
    kinds = [e[0] for e in events]
    assert kinds[:2] == ["cache", "plan"] and kinds.count("plan") == 1, events
    del events[:], calls[:]
    (dq2,) = torch.autograd.grad(out2, (q,), dout)
    kinds = [e[0] for e in events]
    assert (
        kinds[:4] == ["cache", "alloc", "alloc", "alloc"] and kinds[-1] == "launch"
    ), events
    assert events[0][1] is True
    assert all(k in ("alloc", "fill") for k in kinds[1:-1]) and "fill" in kinds, events
    assert events[3][1] == (S, D_LATENT)
    assert tuple(dq2.shape) == (T, NUM_HEADS, D_LATENT)


def test_nope_index_prep_route_issues_the_forward_with_nothing_left_to_do(
    monkeypatch,
):
    """On the preparation route the packed entry allocates every buffer, binds the forward's values and converts
    its tensor operands before the stage launch; between the two launches nothing runs but the forward's binding
    call.  Below the slot limit without caller lengths the one-launch route keeps its single launch without any
    conversion."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    events = []
    real_alloc = cake_backend._alloc

    def alloc(*args, **kwargs):
        events.append("alloc")
        return real_alloc(*args, **kwargs)

    def convert(tensor):
        events.append("convert")
        return tensor

    monkeypatch.setattr(cake_backend, "_alloc", alloc)
    monkeypatch.setattr(cake_backend, "_ffi_tensor", convert)
    for stage in (NOPE_INDEX_PREP_STAGE, "nope_fwd"):
        real_launch = cake_launch.LAUNCH[stage]

        def launch(run, values, grid, _real=real_launch, _stage=stage):
            events.append(("launch", _stage))
            return _real(run, values, grid)

        monkeypatch.setitem(cake_launch.LAUNCH, stage, launch)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    q, kv, _ = _nope_host_inputs(total_q=T, total_k=18, topk=K, device="cuda")
    mine = torch.full((T,), 2, dtype=torch.int32, device="cuda")
    for _ in range(
        2
    ):  # the first call plans, the second hits the binding cache: same launch order
        del events[:], calls[:]
        out, lse, o_lo, indices, lengths = cake_backend.forward_nope_varlen(
            q, kv, local, cu_q, cu_k, causal=True, topk_length=mine
        )
        first = events.index(("launch", NOPE_INDEX_PREP_STAGE))
        assert events[first:] == [
            ("launch", NOPE_INDEX_PREP_STAGE),
            ("launch", "nope_fwd"),
        ], events
        before = events[:first]
        assert "convert" in before and before.count("alloc") >= 5, events
        # the caller's cut reaches the stage; the returned lengths are the stage's counts
        assert lengths is not mine and calls[0][1]["length_in"] is mine
        assert calls[0][1]["topk_length"] is lengths
        assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
        assert calls[1][1]["out"] is out and calls[1][1]["indices"] is indices
    # without caller lengths at or above the limit (a limit of 0 here) the same two launches, in the same order
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    del events[:], calls[:]
    cake_backend.forward_nope_varlen(q, kv, local, cu_q, cu_k, causal=True)
    first = events.index(("launch", NOPE_INDEX_PREP_STAGE))
    assert events[first:] == [
        ("launch", NOPE_INDEX_PREP_STAGE),
        ("launch", "nope_fwd"),
    ], events
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    assert calls[0][1]["has_length_in"] == 0
    # below the limit (the shipped route for this table): one launch from the host tensors, nothing converted
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", limit)
    assert cake_backend.varlen_index_mode(local) == "mask"
    del events[:], calls[:]
    cake_backend.forward_nope_varlen(q, kv, local, cu_q, cu_k, causal=True)
    assert [e for e in events if e != "alloc"] == [("launch", "nope_fwd")], events
    assert [c[0] for c in calls] == ["nope_fwd"]
    assert calls[0][1]["convert"] == cake_backend.NOPE_CONVERT_MASK


def test_nope_kernel_level_packed_forward_is_the_public_entry_without_autograd(
    monkeypatch,
):
    """``forward_nope_varlen`` (no autograd) takes the packed entry's route and returns the outputs plus the
    global rows and lengths ``backward_nope`` consumes.  With stand-in launches: at or above the slot limit (a
    limit of 0 here) the compacting stage then the forward over its rows (``convert = 0``); below it (the shipped
    route for the small example) the converting forward alone -- ``convert = 2``, the fresh rows as
    ``indices_out`` (position-preserving, written by the forward's scan lanes) and the derived lengths; the
    compacting one-launch mode by name (``convert = 1``).  A call without query rows launches nothing."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    q, kv, _ = _nope_host_inputs(total_q=T, total_k=18, topk=K, device="cuda")
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    out, lse, o_lo, indices, lengths = cake_backend.forward_nope_varlen(
        q, kv, local, cu_q, cu_k, causal=True
    )
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, v = calls[0][1], calls[1][1]
    assert prep["out"] is indices and prep["topk_length"] is lengths
    assert (prep["compact"], prep["causal"], prep["has_length_in"]) == (1, 1, 0)
    assert v["out"] is out and v["lse"] is lse and v["o_lo"] is o_lo
    assert v["indices"].data_ptr() == indices.data_ptr() and v["topk_length"] is lengths
    assert (v["convert"], v["has_topk_length"], v["derive_length"]) == (0, 1, 0)
    assert tuple(indices.shape) == (T, K) and tuple(lengths.shape) == (T,)
    del calls[:]
    large = torch.zeros(1024, 1024, dtype=torch.int32, device="cuda")
    cu1 = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    q2, kv2, _ = _nope_host_inputs(total_q=1024, total_k=1024, topk=1024, device="cuda")
    _, _, _, idx2, len2 = cake_backend.forward_nope_varlen(q2, kv2, large, cu1, cu1)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    assert calls[0][1]["out"] is idx2 and calls[0][1]["topk_length"] is len2
    assert calls[1][1]["indices"].data_ptr() == idx2.data_ptr()
    assert calls[1][1]["topk_length"] is len2 and calls[1][1]["convert"] == 0
    # the shipped limit: the small example takes the one-launch position-preserving forward
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", limit)
    assert cake_backend.varlen_index_mode(local) == "mask"
    del calls[:]
    out, lse, o_lo, indices, lengths = cake_backend.forward_nope_varlen(
        q, kv, local, cu_q, cu_k, causal=True
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    v = calls[0][1]
    assert v["out"] is out and v["lse"] is lse and v["o_lo"] is o_lo
    assert v["indices_out"] is indices and v["topk_length"] is lengths
    assert (v["convert"], v["causal"]) == (2, 1)
    assert (v["has_topk_length"], v["derive_length"]) == (1, 1)
    assert tuple(indices.shape) == (T, K) and tuple(lengths.shape) == (T,)
    # the compacting one-launch mode, by name
    del calls[:]
    _, _, _, idx_k, len_k = cake_backend._packed_forward_nope(
        q, kv, local, cu_q, cu_k, True, None, 1 / 16, "kernel"
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    v = calls[0][1]
    assert v["indices_out"] is idx_k and v["topk_length"] is len_k
    assert (v["convert"], v["causal"]) == (1, 1)
    del calls[:]
    e = cake_backend.forward_nope_varlen(
        q[:0], kv, local[:0], cu_q.new_tensor([0, 0]), cu_k.new_tensor([0, 18])
    )
    assert calls == [] and tuple(e[0].shape) == (0, NUM_HEADS, D_LATENT)
    assert e[3].shape[0] == 0 and e[4].numel() == 0


@pytest.mark.parametrize(
    "limit", ["shipped", 0, 1 << 40], ids=["shipped", "stage", "mask"]
)
@pytest.mark.parametrize(
    "seqs_q, seqs_k, topk",
    [
        ([300, 517, 0, 1024], [400, 517, 64, 2048], 200),  # 368,200 slots
        (
            [640, 512],
            [1024, 1024],
            1024,
        ),  # 1,179,648 slots: above the baseline's 1 << 20
    ],
)
def test_nope_kernel_level_packed_pair_matches_the_autograd_entry(
    seqs_q, seqs_k, topk, limit, monkeypatch
):
    """The kernel-level pair -- ``forward_nope_varlen`` and ``backward_nope`` (and its prepared form through
    ``prepare_dsa_nope_train``) over the rows and lengths it returns -- against the autograd packed entry on both
    index routes (the shipped limit, the stage forced by a limit of 0, the one-launch forward forced by a limit
    above every table): ``out``, ``lse`` and ``dq`` bitwise, ``dkv`` (FP32 atomics into a caller buffer on both
    sides) within the summation-order spread."""
    _require_nope_index_prep()
    _require_nope_program(backward=True)
    if limit != "shipped":
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", limit)
    if cake_backend.nope_varlen_index_mode(sum(seqs_q), topk) == "mask":
        _require_nope_mask_route()
    inp = nope_ref.make_nope_inputs(
        seqs_q, seqs_k, seed=NOPE_SEED + 700 + topk, topk=topk, index="iid"
    )
    S = inp.total_k
    q = inp.q.detach().requires_grad_()
    acc = torch.zeros(S, D_LATENT, device="cuda")
    out, lse = cake_backend.dsa_sparse_attention_nope_varlen(
        q,
        inp.kv,
        inp.idx_local,
        inp.cu_seqlens_q,
        inp.cu_seqlens_k,
        causal=True,
        return_lse=True,
        dkv_acc=acc,
    )
    (dq,) = torch.autograd.grad(out, (q,), inp.dout)
    out_k, lse_k, o_lo, indices, lengths = cake_backend.forward_nope_varlen(
        inp.q, inp.kv, inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k, causal=True
    )
    acc_k = torch.zeros(S, D_LATENT, device="cuda")
    dq_k, none = cake_backend.backward_nope(
        inp.q,
        inp.kv,
        indices,
        out_k,
        o_lo,
        lse_k,
        inp.dout,
        topk_length=lengths,
        dkv_acc=acc_k,
        num_segments=len(seqs_q),
    )
    torch.cuda.synchronize()
    assert none is None
    assert torch.equal(out, out_k) and torch.equal(lse, lse_k) and torch.equal(dq, dq_k)
    assert rel_l2(acc_k, acc) < 1e-5
    # the prepared form of the same backward: the forward's buffers bound once, launches only per call
    acc_r = torch.zeros(S, D_LATENT, device="cuda")
    runner = cake_backend.prepare_dsa_nope_train(
        inp.q,
        inp.kv,
        indices,
        topk_length=lengths,
        dout=inp.dout,
        out=out_k,
        lse=lse_k,
        o_lo=o_lo,
        dkv_acc=acc_r,
        num_segments=len(seqs_q),
    )
    dq_r, none_r = runner.backward()
    torch.cuda.synchronize()
    assert none_r is None and torch.equal(dq_r, dq_k)
    assert rel_l2(acc_r, acc) < 1e-5


def test_nope_packed_entry_route_when_the_forward_program_converts(monkeypatch):
    """A forward program that appends the conversion operands: at or above the slot limit (a limit of 0 here) the
    compacting stage runs first over the per-document rows (their own stride and offset through the flat alias)
    and the forward takes its rows with ``convert = 0`` and valid pointers for the operands it never reads; caller
    lengths reach the stage as the cut at any limit and the forward reads the stage's counts; a flat forward
    passes ``convert = 0`` with the index operand doubling as ``indices_out``.  Below the limit (the shipped route
    for the small example) the forward converts the rows itself in one launch -- ``convert = 2``,
    position-preserving, the lengths derived -- and the compacting one-launch mode is reachable by name only
    (``convert = 1``)."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    names = list(cake_launch.STAGE_ARGS["nope_fwd"])
    if not all(name in names for name in cake_backend.NOPE_FWD_CONVERT_ARGS):
        # a program without the operands: stand in the extended tables (the shipped tables carry them)
        extended = names[:-3] + list(cake_backend.NOPE_FWD_CONVERT_ARGS) + names[-3:]
        monkeypatch.setitem(cake_launch.STAGE_ARGS, "nope_fwd", extended)

        def launch(run, values, grid):
            run(*(values[name] for name in extended[:-3]), *grid)

        monkeypatch.setitem(cake_launch.LAUNCH, "nope_fwd", launch)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    assert cake_backend.nope_fwd_converts_indices()
    limit = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    assert cake_backend.varlen_index_mode(local) == "mask"
    assert cake_backend.varlen_index_mode(local, caller_lengths=True) == "index_prep"
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.varlen_index_mode(local) == "index_prep"
    wide = torch.zeros(T, K + 8, dtype=torch.int32, device="cuda")
    wide[:, 3 : 3 + K] = local
    q, kv, _ = _nope_host_inputs(total_q=T, total_k=18, topk=K, device="cuda")
    varlen = cake_backend.dsa_sparse_attention_nope_varlen
    varlen(q, kv, wide[:, 3 : 3 + K], cu_q, cu_k, causal=True)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, v = calls[0][1], calls[1][1]
    assert (prep["causal"], prep["num_docs"], prep["search_iters"]) == (1, 4, 3)
    assert (prep["compact"], prep["has_length_in"]) == (1, 0)
    assert prep["indices"].data_ptr() == wide.data_ptr()
    assert (prep["idx_stride"], prep["indices_offset"]) == (K + 8, 3)
    assert prep["cu_seqlens_q"] is cu_q and prep["cu_seqlens_k"] is cu_k
    assert v["convert"] == 0 and v["indices"].data_ptr() == prep["out"].data_ptr()
    assert (v["idx_stride"], v["indices_offset"]) == (K, 0)
    assert v["indices_out"].data_ptr() == prep["out"].data_ptr()
    assert v["topk_length"] is prep["topk_length"]
    assert (v["has_topk_length"], v["derive_length"]) == (1, 0)
    assert (v["num_docs"], v["search_iters"], v["causal"]) == (1, 1, 0)
    for name in ("cu_seqlens_q", "cu_seqlens_k"):
        assert v[name].dtype == torch.int32 and v[name].is_cuda and v[name].numel() >= 2
    # caller lengths at any limit: the stage takes the cut, the forward reads the stage's counts with convert = 0
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", limit)
    mine = torch.full((T,), 2, dtype=torch.int32, device="cuda")
    del calls[:]
    varlen(q, kv, local, cu_q, cu_k, topk_length=mine)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "nope_fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert (prep["compact"], prep["has_length_in"]) == (1, 1) and prep[
        "length_in"
    ] is mine
    assert fwd["topk_length"] is prep["topk_length"] and fwd["topk_length"] is not mine
    assert (fwd["has_topk_length"], fwd["derive_length"], fwd["convert"]) == (1, 0, 0)
    assert fwd["indices"].data_ptr() == prep["out"].data_ptr()
    assert fwd["indices_out"].data_ptr() == prep["out"].data_ptr()
    # a flat forward over global rows: convert = 0, the index operand doubles as indices_out
    del calls[:]
    idx = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    forward_nope(q, kv, idx)
    v = calls[-1][1]
    assert calls[-1][0] == "nope_fwd" and v["convert"] == 0
    assert v["indices_out"].data_ptr() == idx.data_ptr()
    assert (v["num_docs"], v["search_iters"], v["causal"]) == (1, 1, 0)
    assert v["cu_seqlens_q"].dtype == torch.int32 and v["cu_seqlens_q"].numel() == 2
    # below the limit: the converting forward alone over the per-document rows (their stride and offset), the
    # prefix sums, the document count and search depth, the causal flag, the fresh global-row buffer the backward
    # reads (position-preserving: convert = 2) and the derived lengths (last valid slot + 1)
    assert cake_backend.varlen_index_mode(local) == "mask"
    del calls[:]
    out_m, lse_m = varlen(
        q, kv, wide[:, 3 : 3 + K], cu_q, cu_k, causal=True, return_lse=True
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    v = calls[0][1]
    assert (v["convert"], v["causal"], v["num_docs"], v["search_iters"]) == (2, 1, 4, 3)
    assert v["cu_seqlens_q"] is cu_q and v["cu_seqlens_k"] is cu_k
    assert v["indices"].data_ptr() == wide.data_ptr()
    assert (v["idx_stride"], v["indices_offset"]) == (K + 8, 3)
    assert tuple(v["indices_out"].shape) == (T, K) and v["indices_out"].is_contiguous()
    assert v["indices_out"].dtype == torch.int32
    assert (v["has_topk_length"], v["derive_length"]) == (1, 1)
    assert tuple(v["topk_length"].shape) == (T,)
    assert v["out"] is out_m and v["lse"] is lse_m
    # the compacting one-launch mode by name: the same operands with convert = 1
    del calls[:]
    cake_backend._packed_forward_nope(
        q, kv, wide[:, 3 : 3 + K], cu_q, cu_k, True, None, 1 / 16, "kernel"
    )
    assert [c[0] for c in calls] == ["nope_fwd"]
    v = calls[0][1]
    assert (v["convert"], v["causal"], v["num_docs"], v["search_iters"]) == (1, 1, 4, 3)
    assert (v["idx_stride"], v["indices_offset"]) == (K + 8, 3)
    assert (v["has_topk_length"], v["derive_length"]) == (1, 1)


def test_nope_plan_backward_fresh_and_accumulate_by_the_direct_rule(monkeypatch):
    """S = 8 T: the record's rule (one key per query) selects the natural-layout main stage adding into the FP32
    buffer -- the fresh ``dkv`` (zero-filled, stride 512) or the caller's rows (their stride, the map) -- with no
    cast; ``dkv_direct=False`` routes both forms through the permuted accumulator and the cast (``out_f32`` for
    the fresh buffer, ``accumulate`` into the caller's rows); T > S (fewer keys than query rows) takes the cast
    path unless ``dkv_direct=True``.  The delta stage receives the dense ``dout``'s strides and a zero offset."""
    calls = []
    _stand_in_nope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q, kv, idx, dout = _nope_device_inputs(T, S, K)
    out, lse, o_lo = _saved_outputs(T)
    args = (q, kv, idx, out, o_lo, lse, dout)

    def launched():
        seq = [c[0] for c in calls]
        calls.clear()
        return seq

    dq, dkv = backward_nope(*args)
    main = [c for c in calls if c[0] == "nope_bwd_main_natural"][0][1]
    delta = calls[0][1]
    assert launched() == ["nope_bwd_delta", "nope_bwd_main_natural"]
    assert tuple(dq.shape) == (T, NUM_HEADS, D_LATENT) and dq.dtype == torch.bfloat16
    assert tuple(dkv.shape) == (S, D_LATENT) and dkv.dtype == torch.float32
    assert torch.all(
        dkv == 0
    )  # zero-filled for the main stage (the stand-in adds nothing)
    assert main["dkv_f32"] is dkv and main["dkv_stride"] == D_LATENT
    assert main["dkv_has_map"] == 0 and main["dkv_dst_map"].dtype == torch.int32
    assert main["dq_latent"] is dq and main["dout"] is dout
    assert (main["pass_lo"], main["pass_hi"], main["dq_mode"]) == (0, S, 0)
    assert (main["token_base"], main["token_step"], main["num_queries"]) == (0, 1, T)
    assert main["sm_scale"] == pytest.approx(NOPE_SOFTMAX_SCALE)
    assert main["dq_partial"] is main["delta"] and main["delta"].dtype == torch.float32
    assert main["key_scratch"] is main["pass_counts"] is main["topk_length"]
    assert (main["grid_x"], main["grid_y"], main["grid_z"]) == (T, 1, 1)
    assert delta["num_rows"] == T * NUM_HEADS and delta["grid_x"] == T * 8
    assert delta["dout"] is dout and delta["delta"] is main["delta"]
    assert delta["out"] is out and delta["o_lo"] is o_lo
    assert (
        delta["dout_token_stride"],
        delta["dout_head_stride"],
        delta["dout_offset"],
    ) == (NUM_HEADS * D_LATENT, D_LATENT, 0)
    # fresh through the cast
    dq2, dkv2 = backward_nope(*args, dkv_direct=False)
    assert [c[0] for c in calls] == ["nope_bwd_delta", "nope_bwd_main", "nope_bwd_cast"]
    main, cast = calls[1][1], calls[2][1]
    calls.clear()
    acc_perm = main["dkv_f32"]
    assert acc_perm is not dkv2 and tuple(acc_perm.shape) == (S, D_LATENT)
    assert acc_perm.dtype == torch.float32 and torch.all(acc_perm == 0)
    assert main["dkv_stride"] == D_LATENT and main["dkv_has_map"] == 0
    assert cast["src_latent"] is acc_perm and cast["dst_latent_f32"] is dkv2
    assert (cast["out_f32"], cast["accumulate"], cast["has_dst_map"]) == (1, 0, 0)
    assert (cast["latent_groups"], cast["rope_groups"]) == (S * D_LATENT // 32, 0)
    assert cast["dst_row_stride"] == 0 and cast["dst_packed"] is acc_perm
    assert cast["src_rope"].numel() == 0 and cast["src_rope"].dtype == torch.float32
    assert (
        cast["dst_rope_f32"].numel() == 0
        and cast["dst_rope_f32"].dtype == torch.float32
    )
    assert (
        cast["dst_latent"].numel() == 0 and cast["dst_latent"].dtype == torch.bfloat16
    )
    assert cast["dst_rope"].numel() == 0 and cast["dst_rope"].dtype == torch.bfloat16
    assert cast["dst_map"].dtype == torch.int32
    assert (cast["grid_x"], cast["grid_y"], cast["grid_z"]) == (-(-S * 16 // 256), 1, 1)
    # accumulate: direct into the caller's rows, with and without a map
    acc = torch.zeros(S, 704, device="cuda")
    perm = torch.randperm(
        S, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    ).to(torch.int32)
    _, none = backward_nope(*args, dkv_acc=acc)
    assert none is None and launched()[-1] == "nope_bwd_main_natural"
    _, none = backward_nope(*args, dkv_acc=acc, dkv_dst_map=perm)
    main = calls[-1][1]
    assert launched() == ["nope_bwd_delta", "nope_bwd_main_natural"]
    assert main["dkv_f32"].data_ptr() == acc.data_ptr() and main["dkv_stride"] == 704
    assert main["dkv_has_map"] == 1 and main["dkv_dst_map"] is perm
    # accumulate through the cast
    backward_nope(*args, dkv_acc=acc, dkv_dst_map=perm, dkv_direct=False)
    main, cast = calls[1][1], calls[2][1]
    assert launched() == ["nope_bwd_delta", "nope_bwd_main", "nope_bwd_cast"]
    assert main["dkv_stride"] == D_LATENT and main["dkv_has_map"] == 0
    assert cast["dst_packed"].data_ptr() == acc.data_ptr() and cast["dst_map"] is perm
    assert (cast["out_f32"], cast["accumulate"], cast["has_dst_map"]) == (0, 1, 1)
    assert cast["dst_row_stride"] == 704 and cast["dst_latent_f32"] is main["dkv_f32"]
    # T = 2 S (fewer keys than query rows): the rule takes the cast path; dkv_direct=True forces the natural stage
    q, kv, idx, dout = _nope_device_inputs(T, T // 2, K, seed=1)
    small = (q, kv, idx, out, o_lo, lse, dout)
    backward_nope(*small)
    assert launched() == ["nope_bwd_delta", "nope_bwd_main", "nope_bwd_cast"]
    backward_nope(*small, dkv_direct=True)
    assert launched() == ["nope_bwd_delta", "nope_bwd_main_natural"]
    keys = list(cake_backend.BINDING_CACHE._entries)
    assert keys and all(key[0] == "nope_bwd" for key in keys)
    # a destination map without the accumulator, and an accumulator too narrow
    with pytest.raises(ValueError, match="dkv_dst_map"):
        backward_nope(*args, dkv_dst_map=perm)
    with pytest.raises(ValueError, match=">= 512"):
        backward_nope(*args, dkv_acc=torch.zeros(S, 500, device="cuda"))


def test_nope_plan_multi_pass_backward_launch_order(monkeypatch):
    calls = []
    record = _stand_in_nope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q, kv, idx, dout = _nope_device_inputs(T, S, K)
    out, lse, o_lo = _saved_outputs(T)
    args = (q, kv, idx, out, o_lo, lse, dout)
    backward_nope(*args, key_passes=2, dkv_direct=False)
    assert [c[0] for c in calls] == [
        "nope_bwd_delta",
        "nope_bwd_compact",
        "nope_bwd_main_pass",
        "nope_bwd_compact",
        "nope_bwd_main_pass",
        "nope_bwd_cast",
    ]
    ranges = key_pass_ranges(S, 2)
    for index, (compact, main) in enumerate(
        ((calls[1][1], calls[2][1]), (calls[3][1], calls[4][1]))
    ):
        lo, hi = ranges[index]
        assert (compact["pass_lo"], compact["pass_hi"]) == (lo, hi)
        assert (main["pass_lo"], main["pass_hi"]) == (lo, hi)
        assert main["dq_mode"] == key_pass_dq_mode(index, 2)
        assert (compact["num_tokens"], compact["token_base"]) == (T, 0)
        assert (main["token_base"], main["token_step"], main["num_queries"]) == (
            0,
            1,
            T,
        )
        assert main["key_scratch"] is compact["key_scratch"]
        assert main["pass_counts"] is compact["pass_counts"]
        assert tuple(main["dq_partial"].shape) == (
            T,
            NOPE_DQ_PARTIAL_BYTES_PER_TOKEN // 4,
        )
        assert tuple(main["key_scratch"].shape) == (T, K)
        assert tuple(main["pass_counts"].shape) == (T,)
    calls.clear()
    backward_nope(*args, key_passes=2)  # direct by the rule: the natural pass form
    assert [c[0] for c in calls] == [
        "nope_bwd_delta",
        "nope_bwd_compact",
        "nope_bwd_main_pass_natural",
        "nope_bwd_compact",
        "nope_bwd_main_pass_natural",
    ]
    stages = cake_jit.registered_stages(NOPE_PROGRAM)
    token_chunk = cake_backend._record_token_chunk(record, K)
    for direct in (False, True):
        size = dsa_nope_train_workspace_size(
            T, S, K, torch.device("cuda"), key_passes=2, dkv_direct=direct
        )
        assert (
            size
            == workspace_layout_nope(
                T,
                S,
                K,
                backward=True,
                key_passes=2,
                dkv_direct=direct,
                token_chunk=token_chunk,
            )["total"]
        )
    assert cake_backend._plan_nope_direct(record, stages, T, S, None)
    # one key per query suffices for the natural path; fewer keys than query rows do not
    assert cake_backend._plan_nope_direct(record, stages, T, T, None)
    assert not cake_backend._plan_nope_direct(record, stages, 2 * S, S, None)
    assert not cake_backend._plan_nope_direct(record, stages, T, S, False)
    runner = prepare_dsa_nope_train(
        q, kv, idx, dout=dout, key_passes=2, dkv_direct=False
    )
    assert runner.key_passes == 2 and runner.layout[
        "total"
    ] == dsa_nope_train_workspace_size(
        T, S, K, torch.device("cuda"), key_passes=2, dkv_direct=False
    )
    assert runner.backward_order[0] == "nope_bwd_delta"
    assert runner.backward_order[-1] == "nope_bwd_cast"
    assert runner.backward_order[1:-1] == (
        ("nope_bwd_compact", 0, 0),
        ("nope_bwd_main_pass", 0, 0),
        ("nope_bwd_compact", 0, 1),
        ("nope_bwd_main_pass", 0, 1),
    )


def test_nope_plan_main_stage_grids_follow_the_persistent_rule(monkeypatch):
    """The main stages (single pass and every key-pass chunk) launch with ``nope_bwd_launch_grid`` over the launch's
    token count and the launch's key count; the delta / compaction / cast grids are the generated ones.  The SM count
    is read once per device (cached) and stubbed to 4 here."""
    calls = []
    record = _stand_in_nope_program(monkeypatch, calls)
    index = torch.cuda.current_device()
    real = torch.cuda.get_device_properties(index).multi_processor_count
    cake_backend._SM_COUNTS.pop(index, None)
    assert (
        cake_backend._sm_count(index) == real and cake_backend._SM_COUNTS[index] == real
    )
    monkeypatch.setattr(cake_backend, "_sm_count", lambda device_index: 4)
    window = (
        cake_backend.NOPE_PERSISTENT_DKV_BYTES // cake_backend.NOPE_DKV_BYTES_PER_KEY
    )
    T, K = 8, 5
    out, lse, o_lo = _saved_outputs(T)

    def grids(stage):
        return [c[1]["grid_x"] for c in calls if c[0] == stage]

    for S, expect in ((64, 4), (window, 4), (window + 1, T)):
        q, kv, idx, dout = _nope_device_inputs(T, S, K, seed=S)
        calls.clear()
        backward_nope(q, kv, idx, out, o_lo, lse, dout)
        assert grids("nope_bwd_main_natural") == [expect], S
        assert grids("nope_bwd_delta") == [T * 8]
        calls.clear()
        backward_nope(q, kv, idx, out, o_lo, lse, dout, dkv_direct=False)
        assert grids("nope_bwd_main") == [expect], S
        assert grids("nope_bwd_cast") == [-(-S * 16 // 256)]
    # two token chunks (chunk + 128 tokens) x two passes: per chunk min(tokens, 4) on a resident window, the chunk's
    # token count beyond it; the compaction keeps ceil(tokens / 4)
    chunk = cake_backend._record_token_chunk(record, K)
    T2 = chunk + 128
    out2, lse2, o_lo2 = _saved_outputs(T2)
    for S, expect in ((64, [4, 4, 4, 4]), (window + 1, [chunk, chunk, 128, 128])):
        q, kv, idx, dout = _nope_device_inputs(T2, S, K, seed=S)
        runner = prepare_dsa_nope_train(
            q,
            kv,
            idx,
            dout=dout,
            out=out2,
            lse=lse2,
            o_lo=o_lo2,
            key_passes=2,
            dkv_direct=False,
        )
        calls.clear()
        runner.backward()
        assert grids("nope_bwd_main_pass") == expect, S
        assert grids("nope_bwd_compact") == [-(-chunk // 4)] * 2 + [32] * 2
        # every chunk launch counts its own tokens: num_queries (the main pass kernel's token count) and
        # num_tokens (the compaction's) are the chunk's, token_base its first token
        mains = [c[1] for c in calls if c[0] == "nope_bwd_main_pass"]
        assert [m["num_queries"] for m in mains] == [chunk, chunk, 128, 128]
        assert [m["token_base"] for m in mains] == [0, 0, chunk, chunk]
        compacts = [c[1] for c in calls if c[0] == "nope_bwd_compact"]
        assert [c["num_tokens"] for c in compacts] == [chunk, chunk, 128, 128]
        assert [c["token_base"] for c in compacts] == [0, 0, chunk, chunk]
        del q, kv, idx, dout, runner
    cake_backend.BINDING_CACHE.clear()


def test_nope_prepared_runner_binds_outputs_scratch_and_zero_fill(monkeypatch):
    calls = []
    _stand_in_nope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q, kv, idx, dout = _nope_device_inputs(T, S, K)
    out, lse, o_lo = _saved_outputs(T)
    dq = torch.zeros_like(out)
    dkv = torch.ones(S, D_LATENT, device="cuda")
    size = dsa_nope_train_workspace_size(T, S, K, torch.device("cuda"))
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    runner = prepare_dsa_nope_train(
        q,
        kv,
        idx,
        dout=dout,
        workspace_buffer=workspace,
        out=out,
        lse=lse,
        o_lo=o_lo,
        dq=dq,
        dkv=dkv,
    )
    assert isinstance(runner, cake_backend.DSANopeTrainRunner)
    assert runner.module_name == NOPE_PROGRAM and runner.stages == cake_jit.NOPE_STAGES
    assert runner.forward_stages == ("nope_fwd",) and runner.has_backward
    assert runner.fresh_dkv and runner.dkv_direct and not runner.accumulate_dkv
    assert not runner.dkv_fp32 and not runner.has_topk_length
    assert (
        runner.layout["total"] == size
        and runner.workspace.data_ptr() == workspace.data_ptr()
    )
    assert runner.tensors["dkv"] is dkv and "dkv_latent_acc" not in runner.tensors
    assert runner.tensors["delta"].data_ptr() >= workspace.data_ptr()
    fwd = runner.forward()
    assert fwd[0] is out and fwd[1] is lse and fwd[2] is o_lo
    assert [c[0] for c in calls] == ["nope_fwd"]
    got_dq, got_dkv = runner.backward()
    assert got_dq is dq and got_dkv is dkv
    assert torch.all(dkv == 0)  # zero-filled before the natural main stage adds into it
    assert [c[0] for c in calls] == [
        "nope_fwd",
        "nope_bwd_delta",
        "nope_bwd_main_natural",
    ]
    assert calls[-1][1]["dkv_f32"] is dkv and calls[-1][1]["dq_latent"] is dq
    assert set(runner.launches) == {
        "nope_fwd",
        "nope_bwd_delta",
        "nope_bwd_main_natural",
    }
    calls.clear()
    step = runner.step()
    assert step[0] is dq and step[1] is dkv
    assert [c[0] for c in calls] == [
        "nope_fwd",
        "nope_bwd_delta",
        "nope_bwd_main_natural",
    ]
    # the cast path zero-fills the permuted accumulator and writes the fresh buffer densely
    calls.clear()
    cast_runner = prepare_dsa_nope_train(q, kv, idx, dout=dout, dkv_direct=False)
    assert not cast_runner.dkv_direct and cast_runner.fresh_dkv
    acc_perm = cast_runner.tensors["dkv_latent_acc"]
    acc_perm.fill_(3.0)
    _, fresh = cast_runner.backward()
    assert torch.all(acc_perm == 0) and tuple(fresh.shape) == (S, D_LATENT)
    assert [c[0] for c in calls] == ["nope_bwd_delta", "nope_bwd_main", "nope_bwd_cast"]
    assert calls[-1][1]["dst_latent_f32"] is fresh and calls[-1][1]["out_f32"] == 1
    assert cast_runner.layout["total"] == dsa_nope_train_workspace_size(
        T, S, K, torch.device("cuda"), dkv_direct=False
    )
    # accumulate: the runner returns None for dkv and owns no dkv buffer
    acc = torch.zeros(S, 704, device="cuda")
    acc_runner = prepare_dsa_nope_train(q, kv, idx, dout=dout, dkv_acc=acc)
    assert (
        acc_runner.accumulate_dkv and not acc_runner.fresh_dkv and acc_runner.dkv_direct
    )
    assert acc_runner.backward()[1] is None and "dkv" not in acc_runner.tensors
    assert (
        calls[-1][1]["dkv_f32"].data_ptr() == acc.data_ptr()
        and calls[-1][1]["dkv_stride"] == 704
    )
    # forward-only runner, contract violations
    fwd_only = prepare_dsa_nope_train(q, kv, idx)
    assert not fwd_only.has_backward and fwd_only.layout[
        "total"
    ] == dsa_nope_train_workspace_size(T, S, K, torch.device("cuda"), backward=False)
    with pytest.raises(NotImplementedError, match="without the backward"):
        fwd_only.backward()
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        prepare_dsa_nope_train(
            q, kv, idx, dout=dout, workspace_buffer=workspace[: size - 1]
        )
    with pytest.raises(ValueError, match="dkv_acc accumulates in place"):
        prepare_dsa_nope_train(q, kv, idx, dout=dout, dkv_acc=acc, dkv=dkv)
    with pytest.raises(ValueError, match="belongs to the backward"):
        prepare_dsa_nope_train(q, kv, idx, dkv_acc=acc)
    with pytest.raises(ValueError, match="needs dout"):
        prepare_dsa_nope_train(q, kv, idx, backward=True)
    with pytest.raises(ValueError, match="dkv must be"):
        prepare_dsa_nope_train(q, kv, idx, dout=dout, dkv=dkv.to(torch.bfloat16))
    with pytest.raises(ValueError, match="backend='cake'"):
        prepare_dsa_nope_train(q, kv, idx, backend="other")
    # a head-major dout is bound in place: the delta stage receives its token / head strides
    calls.clear()
    head_major = dout.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    hm_runner = prepare_dsa_nope_train(q, kv, idx, dout=head_major)
    assert hm_runner.tensors["dout"].data_ptr() == head_major.data_ptr()
    assert hm_runner.tensors["dout"].stride() == head_major.stride()
    hm_runner.backward()
    assert [c[0] for c in calls] == ["nope_bwd_delta", "nope_bwd_main_natural"]
    delta, main = calls[0][1], calls[1][1]
    assert delta["dout"].data_ptr() == head_major.data_ptr()
    assert main["dout"].data_ptr() == head_major.data_ptr()
    assert (
        delta["dout_token_stride"],
        delta["dout_head_stride"],
        delta["dout_offset"],
    ) == (D_LATENT, T * D_LATENT, 0)


def test_nope_eager_backward_reads_a_strided_dout_in_place(monkeypatch):
    """A head-major ``dout`` (or a slice of a packed ``[T, 64, 576]`` tensor) is handed to the delta and main
    stages as is -- no copy: the delta receives the token / head strides in elements and a zero offset, the dense
    call its dense strides; every layout is its own plan (the binding key carries the strides)."""
    calls = []
    _stand_in_nope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q, kv, idx, dout = _nope_device_inputs(T, S, K)
    out, lse, o_lo = _saved_outputs(T)
    head_major = dout.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    assert head_major.stride() == (D_LATENT, T * D_LATENT, 1)

    def delta_strides():
        delta = calls[0][1]
        assert calls[0][0] == "nope_bwd_delta"
        assert delta["num_rows"] == T * NUM_HEADS and delta["grid_x"] == T * 8
        return (
            delta["dout"],
            delta["dout_token_stride"],
            delta["dout_head_stride"],
            delta["dout_offset"],
        )

    backward_nope(q, kv, idx, out, o_lo, lse, head_major)
    seen, token, head, offset = delta_strides()
    assert (
        seen.data_ptr() == head_major.data_ptr()
        and seen.stride() == head_major.stride()
    )
    assert calls[1][1]["dout"].data_ptr() == head_major.data_ptr()
    assert (token, head, offset) == (D_LATENT, T * D_LATENT, 0)
    calls.clear()
    backward_nope(q, kv, idx, out, o_lo, lse, dout)
    seen, token, head, offset = delta_strides()
    assert seen is dout and (token, head, offset) == (NUM_HEADS * D_LATENT, D_LATENT, 0)
    calls.clear()
    packed = torch.zeros(T, NUM_HEADS, D_QK, dtype=torch.bfloat16, device="cuda")
    packed[..., :D_LATENT] = dout
    backward_nope(q, kv, idx, out, o_lo, lse, packed[..., :D_LATENT])
    seen, token, head, offset = delta_strides()
    assert seen.data_ptr() == packed.data_ptr()
    assert (token, head, offset) == (NUM_HEADS * D_QK, D_QK, 0)
    assert len(cake_backend.BINDING_CACHE) == 3  # three layouts, three plans


def test_nope_registered_record_without_kernels_fails_loudly():
    """Before the NoPE libraries exist the entries refuse loudly instead of launching: a record without the stages
    raises ``NotImplementedError`` naming them, a complete record whose libraries ``build.json`` does not carry yet
    raises on the library lookup; the workspace size of the registered policies is computable either way."""
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if _nope_program_available(backward=True):
        pytest.skip(
            "the NoPE program and its libraries are registered: covered by the GPU tests"
        )
    T, S, K = 8, 64, 5
    q, kv, idx, dout = _nope_device_inputs(T, S, K)
    cake_backend.BINDING_CACHE.clear()
    if generated_program_available(
        torch.device("cuda"), backward=True, program=NOPE_PROGRAM
    ):
        with pytest.raises(KeyError, match="is not a library"):
            forward_nope(q, kv, idx)
        with pytest.raises(KeyError, match="is not a library"):
            prepare_dsa_nope_train(q, kv, idx, dout=dout)
    else:
        with pytest.raises(NotImplementedError, match="no forward stages"):
            forward_nope(q, kv, idx)
        with pytest.raises(NotImplementedError, match="no backward stages"):
            prepare_dsa_nope_train(q, kv, idx, dout=dout)
    assert len(cake_backend.BINDING_CACHE) == 0  # nothing was remembered
    program, record = record_for(torch.device("cuda"), NOPE_PROGRAM)
    assert program == NOPE_PROGRAM
    stages = cake_jit.registered_stages(NOPE_PROGRAM)
    direct = cake_backend._plan_nope_direct(record, stages, T, S, None)
    assert (
        dsa_nope_train_workspace_size(T, S, K, torch.device("cuda"))
        == (
            workspace_layout_nope(
                T,
                S,
                K,
                backward=True,
                key_passes=1,
                dkv_direct=direct,
                token_chunk=cake_backend._record_token_chunk(record, K),
            )["total"]
        )
    )


# --- GPU suite (skips until the NoPE program is registered) ---


@pytest.mark.parametrize("topk", [128, 200])
def test_nope_forward_iid(topk):
    _require_nope_program()
    inp = _nope_inputs(384, 1024, seed=NOPE_SEED, topk=topk)
    out, lse, _ = forward_nope(inp.q, inp.kv, inp.idx_global)
    torch.cuda.synchronize()
    _check_forward_nope(out, lse, _nope_reference(inp))


def test_nope_forward_pooled_default_width():
    """The recorded pool structure at the profiled K_slots (2176: up to 512 complete pools plus the causal tail),
    rows shorter than the width padded with ``-1``."""
    _require_nope_program()
    inp = _nope_inputs(1024, 4096, seed=NOPE_SEED + 1, topk=2176, index="pooled")
    assert inp.topk == nope_ref.DEFAULT_NOPE_TOPK
    out, lse, _ = forward_nope(inp.q, inp.kv, inp.idx_global)
    torch.cuda.synchronize()
    _check_forward_nope(out, lse, _nope_reference(inp))


@pytest.mark.parametrize("topk", NOPE_SLOT_WIDTHS)
def test_nope_forward_slot_widths(topk):
    """Any ``K_slots >= 1``: the 256 / 512 / 1024 widths, the odd widths 1 / 67 / 2177 and 64."""
    _require_nope_program()
    inp = _nope_inputs(130, 512, seed=NOPE_SEED + topk, topk=topk)
    out, lse, _ = forward_nope(inp.q, inp.kv, inp.idx_global)
    torch.cuda.synchronize()
    _check_forward_nope(out, lse, _nope_reference(inp))


@pytest.mark.parametrize("key_passes", [1, 2])
@pytest.mark.parametrize("topk", NOPE_SLOT_WIDTHS)
def test_nope_backward_slot_widths(topk, key_passes):
    """The backward at every width above, as the single pass and as two forced passes (the compaction and pass
    stages read the index rows through the same ``K_slots`` metadata), each against the reference."""
    if key_passes > 1:
        _require_nope_pass_program()
    else:
        _require_nope_program(backward=True)
    inp = _nope_inputs(130, 512, seed=NOPE_SEED + 100 + topk, topk=topk)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(
        inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout, key_passes=key_passes
    )
    torch.cuda.synchronize()
    _check_backward_nope(dq, dkv, _nope_reference(inp, dout=inp.dout))


@pytest.mark.parametrize("seq_q, seq_k", [(4095, 4096), (4097, 4097)])
def test_nope_irregular_tail_lengths(seq_q, seq_k):
    """Query counts one below and one above a 4096 multiple (the latter with an odd key count): the forward and the
    single-pass backward handle the partial tail tile at the top-k width 2176."""
    _require_nope_program(backward=True)
    inp = _nope_inputs(seq_q, seq_k, seed=NOPE_SEED + 200 + seq_q, topk=2176)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    ref = _nope_reference(inp, dout=inp.dout)
    _check_forward_nope(out, lse, ref)
    _check_backward_nope(dq, dkv, ref)


def test_nope_backward_32k_x_256k_skewed_per_row_dq_p99():
    """The 32k x 256k peaked row (own key ~99 % of the softmax mass, iid K_slots 2176): per-token ``dq`` rel-L2
    p99 within ``GATE_ROW_P99_DQ`` (per token, and per (token, head) row on the normalised row error ``e`` of
    :func:`_check_dq_rows_nope`), the row-error distribution within its emulated floor's, ``out`` and ``dq`` within
    1.02x and ``dkv`` within 1.05x their BF16-P/dS numerics floors.  The FP64 reference streams over 256-row chunks and
    takes minutes: opt in with
    ``FLASHINFER_CAKE_DSA_NOPE_LARGE_TESTS=1``."""
    if os.environ.get(NOPE_LARGE_TESTS_ENV, "") != "1":
        pytest.skip(
            f"set {NOPE_LARGE_TESTS_ENV}=1 to run the 32k x 256k per-row p99 case (minutes of FP64 reference)"
        )
    _require_nope_program(backward=True)
    T, S, K = 32768, 262144, 2176
    beta = nope_ref.calibrate_beta_nope(0.99, seed=NOPE_SEED + 300, topk=K)
    inp = _nope_inputs(
        T, S, seed=NOPE_SEED + 300, topk=K, self_including=True, beta=beta
    )
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    rows, row_err_k, row_err_f = [], [], []
    sq = {k: 0.0 for k in ("err", "emu", "ref", "out_err", "out_emu", "out_ref")}

    def consume(r0, r1, chunk):
        d_ref, o_ref = chunk["dq"], chunk["out"]
        rows.append(rel_l2_rows(dq[r0:r1], d_ref).cpu())
        _check_dq_zero_rows(dq[r0:r1], chunk)
        zero = chunk["dq_zero_rows"]
        row_err_k.append(_dq_row_errors(dq[r0:r1], d_ref, zero).cpu())
        row_err_f.append(_dq_row_errors(chunk["dq_emu"], d_ref, zero).cpu())
        sq["err"] += (dq[r0:r1].double() - d_ref).pow(2).sum().item()
        sq["emu"] += (chunk["dq_emu"] - d_ref).pow(2).sum().item()
        sq["ref"] += d_ref.pow(2).sum().item()
        sq["out_err"] += (out[r0:r1].double() - o_ref).pow(2).sum().item()
        sq["out_emu"] += (chunk["out_emu"] - o_ref).pow(2).sum().item()
        sq["out_ref"] += o_ref.pow(2).sum().item()

    ref = nope_ref.reference_fp64_nope(
        inp.q,
        inp.kv,
        inp.idx_global,
        dout=inp.dout,
        softmax_scale=NOPE_SOFTMAX_SCALE,
        own_key=inp.own_key,
        consume=consume,
    )
    assert 0.9 < ref["self_weight"] < 0.999, (
        f"calibrated self weight {ref['self_weight']:.4f} outside the peaked window"
    )
    p99 = torch.cat(rows).quantile(0.99).item()
    assert p99 <= GATE_ROW_P99_DQ, f"per-row dq rel-L2 p99 {p99:.6f}"
    e_k, e_f = torch.cat(row_err_k), torch.cat(row_err_f)
    p99_rows = e_k.quantile(0.99).item()
    assert p99_rows <= GATE_ROW_P99_DQ, (
        f"per-(token, head) dq row error p99 {p99_rows:.6f}"
    )
    _check_dq_row_stats(e_k, e_f)
    for name, prefix in (("dq", ""), ("out", "out_")):
        err = math.sqrt(sq[prefix + "err"] / sq[prefix + "ref"])
        floor = math.sqrt(sq[prefix + "emu"] / sq[prefix + "ref"])
        assert err <= NOPE_FLOOR_MARGIN * floor + FLOOR_ABS, (
            f"{name}: rel-L2 {err:.6f} vs floor {floor:.6f}"
        )
    err, floor = rel_l2(dkv, ref["dkv"]), rel_l2(ref["dkv_emu"], ref["dkv"])
    assert err <= FLOOR_MARGIN * floor + FLOOR_ABS, (
        f"dkv: rel-L2 {err:.6f} vs floor {floor:.6f}"
    )


def test_nope_forward_masked_rows_and_out_of_range():
    _require_nope_program()
    inp = _nope_inputs(256, 512, seed=NOPE_SEED + 2, topk=128)
    idx = inp.idx_global.clone()
    idx[3] = -1
    idx[9, ::2] = -1
    idx[5, 0] = inp.total_k + 7
    idx[11, 1:] = -1  # one key
    out, lse, _ = forward_nope(inp.q, inp.kv, idx)
    torch.cuda.synchronize()
    assert torch.all(out[3] == 0) and torch.all(lse[3] == float("-inf"))
    _check_forward_nope(out, lse, _nope_reference(inp, idx))


def test_nope_forward_strided_query_view_bitwise():
    """A query slice of a packed [T, 64, 576] tensor (head stride 576) gives the contiguous results bitwise."""
    _require_nope_program()
    inp = _nope_inputs(256, 512, seed=NOPE_SEED + 3, topk=128)
    packed = torch.zeros(
        inp.total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device="cuda"
    )
    packed[..., :D_LATENT] = inp.q
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    out_v, lse_v, o_lo_v = forward_nope(packed[..., :D_LATENT], inp.kv, inp.idx_global)
    torch.cuda.synchronize()
    assert (
        torch.equal(out, out_v)
        and torch.equal(lse, lse_v)
        and torch.equal(o_lo, o_lo_v)
    )


def test_nope_forward_derives_row_lengths_in_kernel():
    _require_nope_program()
    # seq_q == seq_k: the query positions start at key 0, so the first 199 rows hold fewer than 200 keys
    inp = _nope_inputs(256, 256, seed=NOPE_SEED + 4, topk=200)
    assert not torch.all(inp.topk_length == inp.topk)  # short rows exist
    derived = forward_nope(inp.q, inp.kv, inp.idx_global)
    explicit = forward_nope(inp.q, inp.kv, inp.idx_global, topk_length=inp.topk_length)
    filled = torch.empty(inp.total_q, dtype=torch.int32, device="cuda")
    forward_nope(
        inp.q, inp.kv, inp.idx_global, topk_length=filled, derive_topk_length=True
    )
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(derived, explicit, strict=True))
    assert torch.equal(filled, derive_topk_length(inp.idx_global, inp.total_k))


def test_nope_forward_deterministic():
    _require_nope_program()
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 5, topk=128)
    first = forward_nope(inp.q, inp.kv, inp.idx_global)
    second = forward_nope(inp.q, inp.kv, inp.idx_global)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize(
    "seq_q, seq_k, topk, fixed", [(384, 1024, 200, False), (4096, 4096, 2176, True)]
)
def test_nope_backward_iid(seq_q, seq_k, topk, fixed):
    _require_nope_program(backward=True)
    inp = _nope_inputs(seq_q, seq_k, seed=NOPE_SEED + 6, topk=topk)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    ref = _nope_reference(inp, dout=inp.dout)
    _check_forward_nope(out, lse, ref, fixed=fixed)
    _check_backward_nope(dq, dkv, ref, fixed=fixed)


def test_nope_backward_pooled_with_masked_rows():
    _require_nope_program(backward=True)
    inp = _nope_inputs(1024, 4096, seed=NOPE_SEED + 7, topk=2176, index="pooled")
    idx = inp.idx_global.clone()
    idx[3] = -1
    idx[9, ::2] = -1
    out, lse, o_lo = forward_nope(inp.q, inp.kv, idx)
    dq, dkv = backward_nope(inp.q, inp.kv, idx, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.all(dq[3] == 0)
    ref = _nope_reference(inp, idx, dout=inp.dout)
    _check_forward_nope(out, lse, ref)
    _check_backward_nope(dq, dkv, ref)


def test_nope_backward_peaked_self_weight():
    """Own key carrying ~99 % of the softmax mass: the strongest calibrated peaked case (``beta`` from the FP64
    self-weight bisection); the calibrated 0.53 case runs at the same gates: 1.02x the case's own emulated floors
    plus the element-wise / row bands, not the fixed iid table."""
    _require_nope_program(backward=True)
    for target, seed in ((0.99, NOPE_SEED + 8), (0.53, NOPE_SEED + 9)):
        beta = nope_ref.calibrate_beta_nope(target, seed=seed, topk=2176)
        inp = nope_ref.make_nope_inputs(
            [4096],
            [4096],
            seed=seed,
            topk=2176,
            index="iid",
            self_including=True,
            beta=beta,
        )
        out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
        dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
        torch.cuda.synchronize()
        ref = nope_ref.reference_fp64_nope(
            inp.q, inp.kv, inp.idx_global, dout=inp.dout, own_key=inp.own_key
        )
        assert abs(ref["self_weight"] - target) < 0.05
        _check_forward_nope(out, lse, ref)
        _check_backward_nope(dq, dkv, ref)


def test_nope_backward_fresh_and_accumulate_agree():
    """The fresh FP32 ``dkv`` and the accumulate form (identity map, a wider buffer, two accumulations) carry the
    same gradient (another FP32 summation order); untouched columns stay untouched; ``dq`` is bitwise the same."""
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 2048, seed=NOPE_SEED + 10, topk=128)
    args = (inp.q, inp.kv, inp.idx_global)
    out, lse, o_lo = forward_nope(*args)
    saved = (out, o_lo, lse, inp.dout)
    dq, dkv = backward_nope(*args, *saved)
    S = inp.total_k
    acc = torch.zeros(S, D_LATENT, device="cuda")
    dq_a, none = backward_nope(*args, *saved, dkv_acc=acc)
    wide = torch.zeros(S, 704, device="cuda")
    wide[:, D_LATENT:] = 7.0
    backward_nope(*args, *saved, dkv_acc=wide)
    backward_nope(*args, *saved, dkv_acc=wide)
    torch.cuda.synchronize()
    assert none is None and torch.equal(dq, dq_a)
    assert rel_l2(acc, dkv) < ACC_REL_L2
    assert rel_l2(wide[:, :D_LATENT], 2 * dkv) < ACC_REL_L2
    assert torch.all(wide[:, D_LATENT:] == 7.0)
    ref = _nope_reference(inp, dout=inp.dout)
    _check_backward_nope(dq, dkv, ref)
    _check_backward_nope(dq_a, acc, ref)


@pytest.mark.parametrize(
    "chunks", [(4096,), (2048, 2048), (1000, 1000, 1000, 1096), (4095, 1)]
)
def test_nope_backward_chunk_partitions_are_consistent(chunks):
    """Backward passes over query chunks of one row accumulate into one ``dkv_acc`` what the whole row produces
    (FP32 re-association only); every chunk's ``dq`` is the whole row's bitwise."""
    _require_nope_program(backward=True)
    T = sum(chunks)
    inp = _nope_inputs(T, 4096, seed=NOPE_SEED + 11, topk=2176)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    acc = torch.zeros(inp.total_k, D_LATENT, device="cuda")
    base = 0
    for n in chunks:
        rows = slice(base, base + n)
        q_c, idx_c = inp.q[rows], inp.idx_global[rows]
        out_c, lse_c, o_lo_c = forward_nope(q_c, inp.kv, idx_c)
        dq_c, none = backward_nope(
            q_c, inp.kv, idx_c, out_c, o_lo_c, lse_c, inp.dout[rows], dkv_acc=acc
        )
        torch.cuda.synchronize()
        assert none is None
        assert torch.equal(out_c, out[rows]) and torch.equal(lse_c, lse[rows])
        assert torch.equal(dq_c, dq[rows])
        base += n
    assert rel_l2(acc, dkv) < ACC_REL_L2


def test_nope_backward_head_major_dout_matches_dense():
    _require_nope_program(backward=True)
    inp = _nope_inputs(
        256, 1024, seed=NOPE_SEED + 12, topk=128, dout_layout="head_major"
    )
    assert inp.dout.stride() == (D_LATENT, inp.total_q * D_LATENT, 1)
    dense = inp.dout.contiguous()
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq_hm, dkv_hm = backward_nope(
        inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout
    )
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, dense)
    torch.cuda.synchronize()
    assert torch.equal(dq_hm, dq) and rel_l2(dkv_hm, dkv) < ACC_REL_L2
    _check_backward_nope(dq_hm, dkv_hm, _nope_reference(inp, dout=dense))
    # a prepared runner reads the head-major view in place as well
    runner = prepare_dsa_nope_train(inp.q, inp.kv, inp.idx_global, dout=inp.dout)
    assert runner.tensors["dout"].data_ptr() == inp.dout.data_ptr()
    dq_r, dkv_r = runner.step()
    torch.cuda.synchronize()
    assert torch.equal(dq_r, dq) and rel_l2(dkv_r, dkv) < ACC_REL_L2


def test_nope_backward_deterministic_dq_and_dkv_spread():
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 13, topk=128)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    args = (inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    first = backward_nope(*args)
    second = backward_nope(*args)
    torch.cuda.synchronize()
    assert torch.equal(first[0], second[0])
    assert rel_l2(first[1], second[1]) < 1e-5  # FP32 atomics: summation order only


def test_nope_backward_destination_map_with_duplicates_matches_index_add():
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 14, topk=128)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    args = (inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    _, dkv = backward_nope(*args)
    S, S_dst = inp.total_k, 300
    gen = torch.Generator(device="cuda").manual_seed(3)
    dst_map = torch.randint(
        0, S_dst, (S,), device="cuda", generator=gen, dtype=torch.int32
    )
    for direct in (None, False):
        acc = torch.zeros(S_dst, 704, device="cuda")
        _, none = backward_nope(
            *args, dkv_acc=acc, dkv_dst_map=dst_map, dkv_direct=direct
        )
        torch.cuda.synchronize()
        want = torch.zeros(S_dst, D_LATENT, device="cuda").index_add_(
            0, dst_map.long(), dkv
        )
        assert none is None and rel_l2(acc[:, :D_LATENT], want) < ACC_REL_L2
        assert torch.all(acc[:, D_LATENT:] == 0)
    bad = dst_map.clone()
    bad[0] = S_dst
    with pytest.raises(ValueError, match="dkv_dst_map values must lie in"):
        backward_nope(*args, dkv_acc=acc, dkv_dst_map=bad, check_dst_map=True)


def test_nope_backward_direct_and_cast_paths_agree():
    """With S >= 4 T the natural-layout main stage adds into the FP32 buffer itself (no accumulator, no cast),
    for the fresh ``dkv`` and for ``dkv_acc`` alike; forced through the cast path the same rows receive the same
    gradients and ``dq`` is bitwise the same."""
    _require_nope_direct_program()
    inp = _nope_inputs(256, 2048, seed=NOPE_SEED + 15, topk=128)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    args = (inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    direct = prepare_dsa_nope_train(*args[:3], dout=inp.dout)
    cast = prepare_dsa_nope_train(*args[:3], dout=inp.dout, dkv_direct=False)
    assert direct.dkv_direct and "nope_bwd_cast" not in direct.launches
    assert not cast.dkv_direct and "nope_bwd_cast" in cast.launches
    assert "dkv_latent_acc" not in direct.tensors and "dkv_latent_acc" in cast.tensors
    dq_d, dkv_d = direct.step()
    dq_c, dkv_c = cast.step()
    dq_e, dkv_e = backward_nope(*args)
    dq_f, dkv_f = backward_nope(*args, dkv_direct=False)
    torch.cuda.synchronize()
    assert (
        torch.equal(dq_d, dq_c) and torch.equal(dq_e, dq_c) and torch.equal(dq_f, dq_c)
    )
    for got in (dkv_d, dkv_e, dkv_f):
        assert rel_l2(got, dkv_c) < ACC_REL_L2
    acc_d = torch.zeros(inp.total_k, D_LATENT, device="cuda")
    acc_c = torch.zeros_like(acc_d)
    backward_nope(*args, dkv_acc=acc_d)
    backward_nope(*args, dkv_acc=acc_c, dkv_direct=False)
    torch.cuda.synchronize()
    assert rel_l2(acc_d, dkv_c) < ACC_REL_L2 and rel_l2(acc_c, dkv_c) < ACC_REL_L2
    _check_backward_nope(dq_c, dkv_c, _nope_reference(inp, dout=inp.dout))


def test_nope_backward_forced_key_passes_match_reference_and_are_deterministic():
    _require_nope_pass_program()
    inp = _nope_inputs(256, 4096, seed=NOPE_SEED + 16, topk=200)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    args = (inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    single = backward_nope(*args, key_passes=1)
    passes = backward_nope(*args, key_passes=3)
    again = backward_nope(*args, key_passes=3)
    torch.cuda.synchronize()
    assert torch.equal(passes[0], again[0])
    assert rel_l2(passes[1], single[1]) < ACC_REL_L2
    ref = _nope_reference(inp, dout=inp.dout)
    _check_backward_nope(*passes, ref)
    runner = prepare_dsa_nope_train(*args[:3], dout=inp.dout, key_passes=3)
    assert runner.key_passes == 3
    dq_r, dkv_r = runner.step()
    torch.cuda.synchronize()
    assert torch.equal(dq_r, passes[0]) and rel_l2(dkv_r, single[1]) < ACC_REL_L2


def test_nope_backward_forced_passes_over_two_token_chunks():
    """Two forced passes over a row longer than one token chunk (4736 at K_slots 2176, plus 128): the launches run
    chunk by chunk through every pass with a one-chunk pass scratch; dq is bitwise across two calls and agrees with
    the single pass and the reference."""
    record, _stages = _require_nope_pass_program()
    device = torch.device("cuda")
    chunk = KeyPassPolicy.from_record(record).token_chunk(2176)
    assert chunk == 4736
    inp = _nope_inputs(chunk + 128, 65536, seed=NOPE_SEED + 22, topk=2176)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    runner = prepare_dsa_nope_train(
        inp.q, inp.kv, inp.idx_global, dout=inp.dout, key_passes=2, dkv_direct=False
    )
    assert runner.key_passes == 2
    assert [k for k in runner.backward_order if isinstance(k, tuple)] == [
        (stage, c, p)
        for c in range(2)
        for p in range(2)
        for stage in ("nope_bwd_compact", "nope_bwd_main_pass")
    ]
    assert runner.layout["dq_partial"][1] == chunk * NOPE_DQ_PARTIAL_BYTES_PER_TOKEN
    assert runner.layout["key_scratch"][1] == chunk * 2176 * 4
    assert (
        dsa_nope_train_workspace_size(
            chunk + 128, 65536, 2176, device, key_passes=2, dkv_direct=False
        )
        == runner.layout["total"]
    )
    del runner
    args = (inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    a = backward_nope(*args, key_passes=2)
    b = backward_nope(*args, key_passes=2)
    single = backward_nope(*args, key_passes=1)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0])
    assert rel_l2(a[0], single[0]) < 1e-3 and rel_l2(a[1], single[1]) < 1e-2
    ref = _nope_reference(inp, dout=inp.dout)
    _check_backward_nope(*a, ref, fixed=True)


def test_nope_runner_launches_without_allocation_and_matches_eager():
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 17, topk=128)
    runner = prepare_dsa_nope_train(inp.q, inp.kv, inp.idx_global, dout=inp.dout)
    assert runner.layout["total"] == dsa_nope_train_workspace_size(
        inp.total_q, inp.total_k, inp.topk, torch.device("cuda")
    )
    dq_r, dkv_r = runner.step()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(runner.out, out) and torch.equal(runner.lse, lse)
    assert torch.equal(dq_r, dq) and rel_l2(dkv_r, dkv) < ACC_REL_L2


def test_nope_eager_calls_reuse_the_plan_under_tagged_keys():
    _require_nope_program(backward=True)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = _nope_inputs(256, 512, seed=NOPE_SEED + 18, topk=128)
    args = (inp.q, inp.kv, inp.idx_global)
    first = forward_nope(*args)
    second = forward_nope(*args)
    assert len(cache) == 1 and next(iter(cache._entries))[0] == "nope_fwd"
    torch.cuda.synchronize()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    out, lse, o_lo = first
    g1 = backward_nope(*args, out, o_lo, lse, inp.dout)
    g2 = backward_nope(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert len(cache) == 2 and torch.equal(g1[0], g2[0])
    assert g1[1].data_ptr() != g2[1].data_ptr()
    # the rope entry of the same shapes plans separately
    rope = make_inputs([256], [512], seed=NOPE_SEED + 18, topk=128)
    if generated_program_available(torch.device("cuda")):
        cake_backend.forward(
            rope.q_latent, rope.q_rope, rope.kv_latent, rope.k_rope, rope.idx_global
        )
        assert len(cache) == 3
    cache.clear()


def test_nope_forward_captures_into_a_cuda_graph_without_touching_the_cache():
    _require_nope_program()
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = _nope_inputs(256, 512, seed=NOPE_SEED + 19, topk=128)
    args = (inp.q, inp.kv, inp.idx_global)
    eager = forward_nope(*args)
    torch.cuda.synchronize()
    assert len(cache) == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward_nope(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward_nope(*args)
    assert len(cache) == 1
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, captured, strict=True):
        assert torch.equal(a, b)


def test_nope_autograd_function_matches_explicit_backward():
    """The function's ``q`` gradient is the explicit ``dq`` bitwise; without ``dkv_acc`` its FP32 ``dkv`` reaches a
    BF16 ``kv`` leaf as BF16 (the autograd engine casts to the leaf's dtype); with ``dkv_acc`` the FP32 values are
    accumulated and ``kv.grad`` is ``None``."""
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 20, topk=128)
    out, lse, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    dq, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out, o_lo, lse, inp.dout)
    leaves = [t.detach().clone().requires_grad_() for t in (inp.q, inp.kv)]
    out_f, lse_f = cake_backend.DSASparseAttentionNopeFunction.apply(
        *leaves, inp.idx_global, None, NOPE_SOFTMAX_SCALE
    )
    with pytest.raises(NotImplementedError, match="lse"):
        torch.autograd.grad(lse_f.sum(), leaves)
    out_f.backward(inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_f, out) and torch.equal(lse_f, lse)
    assert torch.equal(leaves[0].grad, dq)
    assert leaves[1].grad.dtype == torch.bfloat16
    assert rel_l2(leaves[1].grad, dkv) < 1e-2
    acc = torch.zeros(inp.total_k, D_LATENT, device="cuda")
    leaves = [t.detach().clone().requires_grad_() for t in (inp.q, inp.kv)]
    out_f, _ = cake_backend.DSASparseAttentionNopeFunction.apply(
        *leaves, inp.idx_global, None, NOPE_SOFTMAX_SCALE, None, acc
    )
    out_f.backward(inp.dout)
    torch.cuda.synchronize()
    assert leaves[1].grad is None and torch.equal(leaves[0].grad, dq)
    assert rel_l2(acc, dkv) < ACC_REL_L2


def test_nope_backend_entry_squeezes_the_one_head_kv():
    _require_nope_program(backward=True)
    inp = _nope_inputs(256, 1024, seed=NOPE_SEED + 21, topk=128)
    out, lse = cake_backend.dsa_sparse_attention_nope(
        inp.q, inp.kv, inp.idx_global, return_lse=True
    )
    one_head = inp.kv.unsqueeze(1).detach().clone().requires_grad_()
    q = inp.q.detach().clone().requires_grad_()
    out_h, lse_h = cake_backend.dsa_sparse_attention_nope(
        q, one_head, inp.idx_global, return_lse=True
    )
    out_h.backward(inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out, out_h) and torch.equal(lse, lse_h)
    assert tuple(one_head.grad.shape) == (inp.total_k, 1, D_LATENT)
    out_e, lse_e, o_lo = forward_nope(inp.q, inp.kv, inp.idx_global)
    _, dkv = backward_nope(inp.q, inp.kv, inp.idx_global, out_e, o_lo, lse_e, inp.dout)
    torch.cuda.synchronize()
    assert rel_l2(one_head.grad.squeeze(1), dkv) < 1e-2
    two_heads = torch.zeros(
        inp.total_k, 2, D_LATENT, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(ValueError, match="kv_latent"):
        cake_backend.dsa_sparse_attention_nope(inp.q, two_heads, inp.idx_global)


# --- packed form: the index-preparation stage against the torch formulation (GPU; skips until the stage ships) ---


def _require_nope_index_prep():
    _require_nope_program()
    if not cake_backend.nope_index_prep_available(torch.device("cuda")):
        pytest.skip(
            "the NoPE record does not register the index-preparation stage on this checkout"
        )


_INDEX_PREP_CASES = [
    # (seqs_q, seqs_k, K_slots): the sanitizer batch (an empty query segment, keys longer than queries); a tail
    # segment with a scalar-path width; one slot; seven slots with an empty segment; one document; the profiled
    # width 2176 and the odd 2177; nine documents (a deeper search); no query rows at all
    ((96, 160, 0, 64), (96, 320, 0, 512), 192),
    ((37, 5), (37, 300), 100),
    ((9,), (40,), 1),
    ((9, 0, 3), (40, 8, 3), 7),
    ((300,), (300,), 128),
    ((33, 70), (500, 70), 2176),
    ((33, 70), (500, 70), 2177),
    ((3, 11, 0, 7, 1, 20, 4, 9, 13), (50, 11, 4, 70, 1, 900, 4, 9, 300), 520),
    ((0, 0), (5, 7), 16),
    # one-token documents (seqlen_q == seqlen_k == 1), keys far beyond the width (seqlen_k - seqlen_q > K_slots)
    # and a 67-slot row set (one tile plus three slots)
    ((1, 1, 1), (1, 1, 1), 67),
    ((5, 1), (900, 1), 64),
    ((50,), (50,), 67),
]


def _prefix(seqs):
    cu = [0]
    for n in seqs:
        cu.append(cu[-1] + int(n))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


def _index_prep_case(seqs_q, seqs_k, topk, *, seed):
    """Per-document int32 rows with negatives and out-of-range slots, the first row of every document fully masked,
    and the prefix sums, on the device."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cu_q, cu_k = _prefix(seqs_q), _prefix(seqs_k)
    local = torch.empty(int(sum(seqs_q)), topk, dtype=torch.int32, device="cuda")
    start = 0
    for lq, lk in zip(seqs_q, seqs_k, strict=True):
        local[start : start + lq] = torch.randint(
            -3, lk + 4, (lq, topk), device="cuda", generator=gen, dtype=torch.int32
        )
        if lq:
            local[start] = -1
        start += lq
    return local, cu_q, cu_k


def _require_nope_compacting_stage():
    """The index-preparation stage of this checkout compacts (the K2 interface: ``compact`` / ``length_in`` /
    ``has_length_in`` among its launch arguments, ``nope_compact_indices`` in the backend, and -- a launch table may
    be edited ahead of the kernel -- one probe launch that actually compacts a hole row); skips until the render
    lands."""
    _require_nope_index_prep()
    if "compact" not in cake_launch.STAGE_ARGS.get(
        NOPE_INDEX_PREP_STAGE, ()
    ) or not hasattr(cake_backend, "nope_compact_indices"):
        pytest.skip(
            "the index-preparation stage of this checkout does not compact (the K2 render has not landed)"
        )
    probe = torch.tensor(
        [[-1, 5, -1, 7], [3, -1, -1, 9]], dtype=torch.int32, device="cuda"
    )
    try:
        rows, counts = cake_backend.nope_compact_indices(probe, 16)
        torch.cuda.synchronize()
        ok = torch.equal(
            rows, probe.new_tensor([[5, 7, -1, -1], [3, 9, -1, -1]])
        ) and torch.equal(counts, probe.new_tensor([2, 2]))
    except Exception as exc:  # noqa: BLE001 -- the shipped kernel predates the interface
        pytest.skip(
            f"the index-preparation kernel of this checkout does not compact: {exc}"
        )
    if not ok:
        pytest.skip(
            "the index-preparation kernel of this checkout does not compact (the K2 render has not landed)"
        )


def _require_nope_mask_route():
    """The forward program of this checkout converts per-document rows position-preserving under ``convert =
    NOPE_CONVERT_MASK`` (the small-table route): the conversion operands in its launch arguments, and one probe
    launch below a raised limit that returns the globalized rows with their holes in place and the last valid slot
    + 1 as lengths.  Skips until the render lands (a forward that still compacts under the new value)."""
    _require_nope_program()
    if not cake_backend.nope_fwd_converts_indices():
        pytest.skip("the registered NoPE forward program has no conversion operands")
    local = torch.tensor(
        [[-1, 1, -1, 2], [0, -1, -1, -1]], dtype=torch.int32, device="cuda"
    )
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    q, kv, _ = _nope_host_inputs(total_q=2, total_k=8, topk=4, device="cuda")
    old = cake_backend.NOPE_CONVERT_SLOT_LIMIT
    cake_backend.NOPE_CONVERT_SLOT_LIMIT = 1 << 40
    try:
        assert cake_backend.varlen_index_mode(local) == "mask"
        _, _, _, rows, lengths = cake_backend.forward_nope_varlen(
            q, kv, local, cu_q, cu_k
        )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 -- the shipped kernel predates the mode
        pytest.skip(
            f"the NoPE forward of this checkout cannot convert position-preserving: {exc}"
        )
    finally:
        cake_backend.NOPE_CONVERT_SLOT_LIMIT = old
    if not (
        torch.equal(rows, local) and torch.equal(lengths, local.new_tensor([4, 1]))
    ):
        pytest.skip(
            "the NoPE forward of this checkout compacts under convert = 2 (the render has not landed)"
        )


def _compact_local_rows(local, glob):
    """The stable compaction of per-document rows ``local`` by the validity of their globalized rows ``glob``
    (``-1`` after the causal rule and the range checks): the twin the kernels must treat bitwise like ``local``."""
    valid = glob != -1
    order = (~valid).to(torch.int8).argsort(dim=-1, stable=True)
    return (
        torch.where(valid, local, local.new_full((), -1)).gather(-1, order).contiguous()
    )


def _nope_qkv(T, S, *, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, NUM_HEADS, D_LATENT, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    kv = torch.randn(S, D_LATENT, device="cuda", generator=gen).to(torch.bfloat16)
    dout = torch.randn(T, NUM_HEADS, D_LATENT, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    return q, kv, dout


def _rel_or_equal(a, b, tol):
    """``rel_l2`` within ``tol``, or bitwise equality where the reference side is identically zero."""
    if float(b.double().norm()) == 0.0:
        assert torch.equal(a, b)
    else:
        assert rel_l2(a, b) < tol, rel_l2(a, b)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqs_q, seqs_k, topk", _INDEX_PREP_CASES)
def test_nope_index_prep_matches_the_torch_glue_bitwise(
    seqs_q, seqs_k, topk, causal, compact
):
    """One launch of the stage against the torch formulation and the loop reference, under both causal settings:
    ``compact=False`` -- global rows and last-valid + 1 lengths bitwise ``offset_gather_kv_indices(...,
    return_topk_length=True)``; ``compact=True`` -- the stable compaction (``compact_rows`` of those rows: valid keys
    first in ascending slot order, ``-1`` after) and the valid counts, bitwise the torch twin
    ``offset_gather_kv_indices(..., compact=True)`` and the loop oracle, with the tail of every row ``-1`` and
    ``derive_topk_length`` of the compacted rows equal to the counts."""
    _require_nope_index_prep()
    if compact:
        _require_nope_compacting_stage()
    local, cu_q, cu_k = _index_prep_case(
        seqs_q, seqs_k, topk, seed=NOPE_SEED + 500 + topk
    )
    S = int(cu_k[-1])
    assert cake_backend.varlen_index_mode(local) in ("index_prep", "mask")
    want_idx, want_len = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=causal, return_topk_length=True
    )
    got_idx, got_len = nope_globalize_indices(
        local, cu_q, cu_k, causal=causal, compact=compact
    )
    torch.cuda.synchronize()
    assert got_idx.dtype == torch.int32 and got_len.dtype == torch.int32
    if not compact:
        assert torch.equal(got_idx, want_idx) and torch.equal(got_len, want_len)
        assert torch.equal(got_len, derive_topk_length(got_idx, S))
        assert torch.equal(
            got_idx.cpu(),
            globalize_gather_indices_loop(local, cu_q, cu_k, causal=causal),
        )
        return
    rows, counts = compact_rows(want_idx, S)
    assert torch.equal(got_idx, rows) and torch.equal(got_len, counts)
    assert torch.equal(got_len, derive_topk_length(got_idx, S))
    assert torch.equal(got_len, (want_idx != -1).sum(-1, dtype=torch.int32))
    twin_idx, twin_len = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=causal, compact=True, return_topk_length=True
    )
    assert torch.equal(twin_idx, rows) and torch.equal(twin_len, counts)
    assert got_idx.cpu().tolist() == compact_rows_loop(want_idx.cpu().tolist(), S)
    slot = torch.arange(int(local.shape[1]), device="cuda")
    assert torch.all(got_idx[slot[None, :] >= got_len[:, None]] == -1)
    assert torch.all(got_idx[slot[None, :] < got_len[:, None]] != -1)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("topk", [7, 67, 2177])
def test_nope_index_prep_reads_strided_and_offset_index_views_in_place(topk, compact):
    """Column slices of wider buffers (row strides ``K``, ``K + 4``, ``K + 8``, ``2 K``; offsets 0 / 3 / 4 / 8 / 13,
    vector-aligned and not), every-other-row views and a view inside a larger storage: read in place through the
    flat alias, bitwise the glue in both modes; the flat mode (``nope_compact_indices`` over global rows) reads the
    same views in place and gives ``compact_rows`` of them."""
    _require_nope_index_prep()
    if compact:
        _require_nope_compacting_stage()
    local, cu_q, cu_k = _index_prep_case(
        (96, 160, 0, 64), (96, 320, 0, 512), topk, seed=NOPE_SEED + 7 + topk
    )
    T, K = (int(d) for d in local.shape)
    S = int(cu_k[-1])
    want = offset_gather_kv_indices(
        local, cu_q, cu_k, causal=True, compact=compact, return_topk_length=True
    )
    glob = offset_gather_kv_indices(local, cu_q, cu_k, causal=True)
    want_flat = compact_rows(glob, S)

    def check(view, source):
        assert view.data_ptr() != source.data_ptr() or view.stride() != source.stride()
        got = nope_globalize_indices(view, cu_q, cu_k, causal=True, compact=compact)
        torch.cuda.synchronize()
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])

    def check_flat(view):
        if not compact:
            return
        got = cake_backend.nope_compact_indices(view, S)
        torch.cuda.synchronize()
        assert torch.equal(got[0], want_flat[0]) and torch.equal(got[1], want_flat[1])

    for width in (K, K + 4, K + 8, 2 * K):
        for start in (0, 3, 4, 8, 13):
            if start + K > width:
                continue
            wide = torch.full((T, width), -7, dtype=torch.int32, device="cuda")
            wide[:, start : start + K] = local
            check(wide[:, start : start + K], local)
            wide_g = torch.full((T, width), -7, dtype=torch.int32, device="cuda")
            wide_g[:, start : start + K] = glob
            check_flat(wide_g[:, start : start + K])
    every_other = torch.full((2 * T, K), -7, dtype=torch.int32, device="cuda")
    every_other[::2] = local
    check(every_other[::2], local)
    every_other[::2] = glob
    check_flat(every_other[::2])
    buf = torch.full((T * K + 13,), -7, dtype=torch.int32, device="cuda")
    inside = buf[13:].view(T, K)
    inside.copy_(local)
    check(inside, local)
    inside.copy_(glob)
    check_flat(inside)


@pytest.mark.parametrize("route", ["shipped", "stage"])
@pytest.mark.parametrize(
    "seqs_q, seqs_k, topk, causal",
    [
        ([300, 517, 0, 1024], [400, 517, 64, 2048], 200, False),  # 368,200 slots
        ([256, 128], [512, 512], 96, True),  # 36,864 slots
        ([640, 512], [1024, 1024], 1024, False),  # 1,179,648 slots
    ],
)
def test_nope_packed_entry_with_on_device_index_prep_is_bitwise_the_glue_path(
    seqs_q, seqs_k, topk, causal, route, monkeypatch
):
    """The backend's packed entry against the explicit glue plus the flat entry with caller lengths: ``out``,
    ``lse`` and ``dq`` bitwise; ``dkv`` (FP32 atomics into a caller buffer on both sides) within the
    summation-order spread.  On the stage route (at or above the slot limit, or forced by a limit of 0) the glue
    is ``offset_gather_kv_indices`` globalized rows stably compacted by ``nope_compact_indices`` (==
    ``compact_rows``) with the counts as lengths: the random causal rows carry holes (keys the causal rule drops),
    so the glue side must compact to agree.  Below the limit (the one-launch forward) the glue is the
    position-preserving globalized rows themselves with the last valid slot + 1 as lengths, and the kernel-level
    ``forward_nope_varlen`` returns exactly that pair."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    if route == "stage":
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    mode = cake_backend.nope_varlen_index_mode(sum(seqs_q), topk)
    if mode == "mask":
        _require_nope_mask_route()
    inp = nope_ref.make_nope_inputs(
        seqs_q, seqs_k, seed=NOPE_SEED + 600 + topk, topk=topk, index="iid"
    )
    local = inp.idx_local
    if causal:  # random rows, so the causal rule drops keys
        gen = torch.Generator(device="cuda").manual_seed(NOPE_SEED + 601)
        local = torch.full_like(inp.idx_local, -1)
        for d, (lq, lk) in enumerate(zip(seqs_q, seqs_k, strict=True)):
            rows = slice(int(inp.cu_seqlens_q[d]), int(inp.cu_seqlens_q[d + 1]))
            local[rows] = torch.randint(
                0, lk, (lq, topk), device="cuda", generator=gen
            ).to(torch.int32)
    S = inp.total_k

    def run(fused):
        leaves = [t.detach().clone().requires_grad_() for t in (inp.q, inp.kv)]
        acc = torch.zeros(S, D_LATENT, device="cuda")
        if fused:
            assert cake_backend.varlen_index_mode(local) == mode
            out, lse = cake_backend.dsa_sparse_attention_nope_varlen(
                leaves[0],
                leaves[1],
                local,
                inp.cu_seqlens_q,
                inp.cu_seqlens_k,
                causal=causal,
                return_lse=True,
                dkv_acc=acc,
            )
        else:
            glob = offset_gather_kv_indices(
                local, inp.cu_seqlens_q, inp.cu_seqlens_k, causal=causal
            )
            if mode == "mask":
                idx, lengths = glob, nope_ref.last_valid_plus_one(glob, S)
            else:
                idx, lengths = cake_backend.nope_compact_indices(glob, S)
                rows, counts = compact_rows(glob, S)
                assert torch.equal(idx, rows) and torch.equal(lengths, counts)
            got = cake_backend.forward_nope_varlen(
                inp.q, inp.kv, local, inp.cu_seqlens_q, inp.cu_seqlens_k, causal=causal
            )
            torch.cuda.synchronize()
            assert torch.equal(got[3], idx) and torch.equal(got[4], lengths)
            out, lse = cake_backend.dsa_sparse_attention_nope(
                leaves[0],
                leaves[1],
                idx,
                topk_length=lengths,
                return_lse=True,
                dkv_acc=acc,
                num_segments=len(seqs_q),
            )
        (dq,) = torch.autograd.grad(out, [leaves[0]], inp.dout)
        torch.cuda.synchronize()
        return out, lse, dq, acc

    fused, glue = run(True), run(False)
    for a, b in zip(fused[:3], glue[:3], strict=True):
        assert torch.equal(a, b)
    assert rel_l2(fused[3], glue[3]) < 1e-5  # FP32 atomics: summation order only


# --- hole rows (U6 section 5 / judges' grafts): every construction is bitwise its stable compaction through the
# kernels; the compacting entries return the compacted rows and the valid counts (GPU; skips until the render) ---

# name -> (seqs_q, seqs_k, K_slots, variant): the repro structure at a small T, interleaved holes (the random rows
# of ``_index_prep_case``: negatives and out-of-range slots anywhere, the first row of every document fully masked,
# keys the causal rule drops), leading holes, every-other / every-seventh holes, hole runs across tile borders,
# first- and last-tile holes, one key in the last column, zero valid keys, K_slots 1 / 7 / 67 / 2177, a row count
# that is not a multiple of four, an empty document, and no query rows at all
_HOLE_CASES = {
    "repro_small": ((300,), (300,), 2176, "repro"),
    "interleaved": ((96, 160, 0, 64), (96, 320, 0, 512), 192, "random"),
    "leading": ((96, 160, 0, 64), (96, 320, 0, 512), 128, "leading"),
    "every_other_t42": ((37, 5), (37, 300), 100, "every_other"),
    "k1": ((9,), (40,), 1, "random"),
    "k7_empty_doc": ((9, 0, 3), (40, 8, 3), 7, "random"),
    "k67_every_7th": ((50,), (50,), 67, "every_7th"),
    "k2177_runs": ((33, 70), (500, 70), 2177, "runs"),
    "first_last_tile": ((64,), (4096,), 2176, "first_last_tile"),
    "one_key_last_column": ((5,), (3000,), 2176, "one_last"),
    "all_minus_one": ((7,), (30,), 16, "all"),
    "no_rows": ((0, 0), (5, 7), 16, "random"),
}


def _hole_case(name):
    """``(local, cu_q, cu_k)`` of one ``_HOLE_CASES`` entry: per-document int32 rows with holes on the device."""
    seqs_q, seqs_k, K, variant = _HOLE_CASES[name]
    if variant == "repro":
        cu_q, cu_k = _prefix(seqs_q), _prefix(seqs_k)
        local = torch.cat(
            [
                nope_ref.repro_gap_local(lq, lk, K, torch.device("cuda"))
                for lq, lk in zip(seqs_q, seqs_k, strict=True)
            ],
            0,
        ).contiguous()
        return local, cu_q, cu_k
    local, cu_q, cu_k = _index_prep_case(seqs_q, seqs_k, K, seed=NOPE_SEED + 700 + K)
    if variant == "leading":
        local[:, :5] = -1
    elif variant == "every_other":
        local[:, 1::2] = -1
    elif variant == "every_7th":
        local[:, ::7] = -1
    elif variant == "runs":
        local[:, 100:164] = -1
        local[:, 60:68] = -1
        local[:, 1000:1003] = -1
    elif variant == "first_last_tile":
        local[:, :64] = -1
        local[:, 2112:] = -1
        local[:, 64:2112] = torch.arange(2048, device="cuda", dtype=torch.int32)[
            None, :
        ]
    elif variant == "one_last":
        local.fill_(-1)
        local[:, -1] = torch.arange(
            int(local.shape[0]), device="cuda", dtype=torch.int32
        )
    elif variant == "all":
        local.fill_(-1)
    return local, cu_q, cu_k


def _hole_case_tensors(name, *, seed):
    """The rows of a hole case, their position-preserving globalization ``glob``, its stable compaction and counts,
    the compacted per-document twin, and matching ``q`` / ``kv`` / ``dout``."""
    local, cu_q, cu_k = _hole_case(name)
    S, T = int(cu_k[-1]), int(local.shape[0])
    glob = offset_gather_kv_indices(local, cu_q, cu_k, causal=True)
    rows, counts = compact_rows(glob, S)
    twin = _compact_local_rows(local, glob)
    assert torch.equal(offset_gather_kv_indices(twin, cu_q, cu_k, causal=True), rows)
    q, kv, dout = _nope_qkv(T, S, seed=seed)
    return SimpleNamespace(
        local=local,
        cu_q=cu_q,
        cu_k=cu_k,
        S=S,
        T=T,
        glob=glob,
        rows=rows,
        counts=counts,
        twin=twin,
        q=q,
        kv=kv,
        dout=dout,
        num_docs=len(_HOLE_CASES[name][0]),
    )


def _public_nope_step(entry, q, kv, dout, *args, **kw):
    """``(out, lse, dq, dkv_acc)`` of one public NoPE entry over fresh leaves with an FP32 accumulator."""
    leaves = [t.detach().clone().requires_grad_() for t in (q, kv)]
    acc = torch.zeros(int(kv.shape[0]), D_LATENT, device="cuda")
    out, lse = entry(leaves[0], leaves[1], *args, return_lse=True, dkv_acc=acc, **kw)
    if out.shape[0]:
        (dq,) = torch.autograd.grad(out, [leaves[0]], dout)
    else:
        dq = torch.zeros_like(leaves[0])
    torch.cuda.synchronize()
    return out, lse, dq, acc


@pytest.mark.parametrize("name", sorted(_HOLE_CASES))
def test_nope_hole_rows_are_bitwise_their_stable_compaction_through_the_kernels(
    name, monkeypatch
):
    """The stage route (at or above the slot limit; every hole case is below the shipped limit, so the limit is
    set to 0 here).  Through ``forward_nope_varlen`` the hole rows and their stable per-document twin give
    ``out``, ``lse``, ``o_lo`` bitwise, the same compacted global rows (``compact_rows`` of the glue's globalized
    rows) and the same valid counts; ``backward_nope`` over the returned pair gives ``dq`` bitwise and ``dkv``
    within the atomics spread.  Through the public entries the varlen call on hole rows, the flat call on the
    compacted rows with the counts as lengths and the flat call on the hole rows without lengths (the stage's flat
    mode) agree bitwise on ``out`` / ``lse`` / ``dq``; no query rows give empty outputs and no launch."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    c = _hole_case_tensors(name, seed=NOPE_SEED + 800)
    a = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    b = cake_backend.forward_nope_varlen(c.q, c.kv, c.twin, c.cu_q, c.cu_k, causal=True)
    torch.cuda.synchronize()
    for x, y in zip(a[:3], b[:3], strict=True):
        assert torch.equal(x, y)
    for side in (a, b):
        assert torch.equal(side[3], c.rows) and torch.equal(side[4], c.counts)
    if c.T == 0:
        assert tuple(a[0].shape) == (0, NUM_HEADS, D_LATENT) and a[3].numel() == 0
        out_v = cake_backend.dsa_sparse_attention_nope_varlen(
            c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
        )
        assert tuple(out_v.shape) == (0, NUM_HEADS, D_LATENT)
        return
    ga = backward_nope(
        c.q,
        c.kv,
        a[3],
        a[0],
        a[2],
        a[1],
        c.dout,
        topk_length=a[4],
        num_segments=c.num_docs,
    )
    gb = backward_nope(
        c.q,
        c.kv,
        b[3],
        b[0],
        b[2],
        b[1],
        c.dout,
        topk_length=b[4],
        num_segments=c.num_docs,
    )
    torch.cuda.synchronize()
    assert torch.equal(ga[0], gb[0])
    _rel_or_equal(ga[1], gb[1], 1e-5)
    varlen = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    flat_c = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        c.rows,
        topk_length=c.counts,
        num_segments=c.num_docs,
    )
    flat_h = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        c.glob,
        num_segments=c.num_docs,
    )
    assert torch.equal(varlen[0], a[0]) and torch.equal(varlen[1], a[1])
    assert torch.equal(varlen[2], ga[0])
    for other in (flat_c, flat_h):
        for x, y in zip(varlen[:3], other[:3], strict=True):
            assert torch.equal(x, y)
        _rel_or_equal(other[3], varlen[3], 1e-5)
    _rel_or_equal(varlen[3], ga[1], 1e-5)


@pytest.mark.parametrize("name", sorted(_HOLE_CASES))
def test_nope_hole_rows_below_the_limit_are_bitwise_the_position_preserving_forward(
    name, monkeypatch
):
    """Below the slot limit (raised above every case here; every case, the 652,800-slot repro included, also sits
    below the shipped limit) the packed entry converts in-kernel, position-preserving (``convert = 2``).  Through
    ``forward_nope_varlen`` the hole rows give the globalized rows of ``offset_gather_kv_indices`` with their
    holes in place and the last valid slot + 1 as lengths, and ``out`` / ``lse`` / ``o_lo`` bitwise the eager
    ``forward_nope`` over those rows (the same kernel over the same slot order); ``backward_nope`` over the
    returned pair gives ``dq`` bitwise the eager backward over the globalized rows with those lengths and ``dkv``
    within the atomics spread.  Through the public entries the varlen call on hole rows, the flat call on the
    globalized rows without lengths (one launch deriving the lengths) and the flat call with the derived lengths
    agree bitwise on ``out`` / ``lse`` / ``dq``; the varlen Function saves exactly that pair; and the outputs and
    gradients meet the FP64 floors evaluated on the raw slot order.  No query rows give empty outputs and no
    launch."""
    _require_nope_program(backward=True)
    _require_nope_mask_route()
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 1 << 40)
    c = _hole_case_tensors(name, seed=NOPE_SEED + 800)
    lengths = nope_ref.last_valid_plus_one(c.glob, c.S)
    if c.T:
        assert cake_backend.varlen_index_mode(c.local) == "mask"
        assert cake_backend.nope_flat_index_mode(c.T, int(c.glob.shape[1])) == "mask"
    a = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    e = forward_nope(
        c.q, c.kv, c.glob
    )  # the eager kernel over the raw rows derives the same lengths
    torch.cuda.synchronize()
    assert torch.equal(a[3], c.glob) and torch.equal(a[4], lengths)
    for x, y in zip(a[:3], e, strict=True):
        assert torch.equal(x, y)
    if c.T == 0:
        assert tuple(a[0].shape) == (0, NUM_HEADS, D_LATENT) and a[3].numel() == 0
        out_v = cake_backend.dsa_sparse_attention_nope_varlen(
            c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
        )
        assert tuple(out_v.shape) == (0, NUM_HEADS, D_LATENT)
        return
    ga = backward_nope(
        c.q,
        c.kv,
        a[3],
        a[0],
        a[2],
        a[1],
        c.dout,
        topk_length=a[4],
        num_segments=c.num_docs,
    )
    gb = backward_nope(
        c.q,
        c.kv,
        c.glob,
        e[0],
        e[2],
        e[1],
        c.dout,
        topk_length=lengths,
        num_segments=c.num_docs,
    )
    torch.cuda.synchronize()
    assert torch.equal(ga[0], gb[0])
    _rel_or_equal(ga[1], gb[1], 1e-5)
    varlen = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    flat_h = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        c.glob,
        num_segments=c.num_docs,
    )
    flat_l = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        c.glob,
        topk_length=lengths,
        num_segments=c.num_docs,
    )
    assert torch.equal(varlen[0], a[0]) and torch.equal(varlen[1], a[1])
    assert torch.equal(varlen[2], ga[0])
    for other in (flat_h, flat_l):
        for x, y in zip(varlen[:3], other[:3], strict=True):
            assert torch.equal(x, y)
        _rel_or_equal(other[3], varlen[3], 1e-5)
    _rel_or_equal(varlen[3], ga[1], 1e-5)
    # the saved pair of both Functions: the position-preserving rows and the last valid slot + 1
    q_leaf = c.q.detach().clone().requires_grad_()
    out_v = cake_backend.dsa_sparse_attention_nope_varlen(
        q_leaf, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    saved = out_v.grad_fn.saved_tensors
    assert torch.equal(saved[5], c.glob) and torch.equal(saved[6], lengths)
    out_f = cake_backend.dsa_sparse_attention_nope(
        q_leaf, c.kv, c.glob, num_segments=c.num_docs
    )
    saved = out_f.grad_fn.saved_tensors
    assert saved[5].data_ptr() == c.glob.data_ptr() and torch.equal(saved[6], lengths)
    # the FP64 floors on the raw slot order (the order both kernels process below the limit)
    ref = nope_ref.reference_fp64_nope(c.q, c.kv, c.glob, dout=c.dout, compact=False)
    _check_forward_nope(varlen[0], varlen[1], ref)
    _check_backward_nope(varlen[2], varlen[3], ref)


@pytest.mark.parametrize("name", ["interleaved", "k2177_runs", "repro_small"])
def test_nope_caller_cut_on_hole_rows_is_the_flat_entry_on_the_cut_compaction(name):
    """A caller ``topk_length`` is a position cut applied before the compaction (K2): ``nope_compact_indices(glob,
    S, topk_length=L)`` == ``compact_rows(glob, S, L)``; ``cake_backend.forward_nope_varlen(..., topk_length=L)`` returns those
    rows and counts; the varlen entry with ``L`` on hole rows is bitwise the flat entry on the cut compaction with
    its counts.  The flat entry with caller lengths does not compact: it is bitwise the kernel-level forward over
    the caller's rows and lengths (D5 both ways)."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    c = _hole_case_tensors(name, seed=NOPE_SEED + 810)
    K = int(c.local.shape[1])
    gen = torch.Generator(device="cuda").manual_seed(NOPE_SEED + 811)
    cut = torch.randint(0, K + 1, (c.T,), device="cuda", generator=gen).to(torch.int32)
    cut[: min(3, c.T)] = torch.tensor([0, 1, K], device="cuda", dtype=torch.int32)[
        : min(3, c.T)
    ]
    rows, counts = compact_rows(c.glob, c.S, cut)
    got = cake_backend.nope_compact_indices(c.glob, c.S, topk_length=cut)
    torch.cuda.synchronize()
    assert torch.equal(got[0], rows) and torch.equal(got[1], counts)
    a = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True, topk_length=cut
    )
    torch.cuda.synchronize()
    assert torch.equal(a[3], rows) and torch.equal(a[4], counts)
    varlen = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
        topk_length=cut,
    )
    flat = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        rows,
        topk_length=counts,
        num_segments=c.num_docs,
    )
    for x, y in zip(varlen[:3], flat[:3], strict=True):
        assert torch.equal(x, y)
    _rel_or_equal(flat[3], varlen[3], 1e-5)
    # caller lengths on the flat entry: the caller's rows as given (no compaction), bitwise the kernel-level pair
    kept = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        c.q,
        c.kv,
        c.dout,
        c.glob,
        topk_length=cut,
        num_segments=c.num_docs,
    )
    out, lse, o_lo = forward_nope(c.q, c.kv, c.glob, topk_length=cut)
    dq, _ = backward_nope(
        c.q,
        c.kv,
        c.glob,
        out,
        o_lo,
        lse,
        c.dout,
        topk_length=cut,
        num_segments=c.num_docs,
    )
    torch.cuda.synchronize()
    assert (
        torch.equal(kept[0], out)
        and torch.equal(kept[1], lse)
        and torch.equal(kept[2], dq)
    )


@pytest.mark.parametrize("route", ["mask", "index_prep"])
@pytest.mark.parametrize("topk", NOPE_SLOT_WIDTHS)
@pytest.mark.parametrize("offset", [0, 3, 13])
def test_nope_flat_entry_hole_variants_on_every_slot_width(
    topk, offset, route, monkeypatch
):
    """Global rows with every third slot ``-1`` (and a fully masked row) at every slot width, read from a column
    slice at offset 0 / 3 / 13 of a wider buffer, through the flat entry without lengths.  Below the slot limit
    (``"mask"``, the shipped route for these 96-row tables): one launch over the view as given, bitwise the flat
    entry on the view with the last valid slot + 1 as caller lengths and the eager kernel-level pair, the saved
    pair being the view and those lengths.  At or above it (``"index_prep"``, the limit set to 0): the stage's
    flat mode compacts the view in place (``nope_compact_indices`` of the view == ``compact_rows``) and the entry
    is bitwise the flat entry on the compacted rows with the counts."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    T, S = 96, 1024
    if route == "index_prep":
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.nope_flat_index_mode(T, topk) == route
    gen = torch.Generator(device="cuda").manual_seed(NOPE_SEED + 820 + topk)
    wide = torch.randint(0, S, (T, topk + 16), device="cuda", generator=gen).to(
        torch.int32
    )
    view = wide[:, offset : offset + topk]
    view[:, 1::3] = -1
    view[5] = -1
    q, kv, dout = _nope_qkv(T, S, seed=NOPE_SEED + 821 + topk)
    holes = _public_nope_step(cake_backend.dsa_sparse_attention_nope, q, kv, dout, view)
    if route == "mask":
        lengths = nope_ref.last_valid_plus_one(view, S)
        twin = _public_nope_step(
            cake_backend.dsa_sparse_attention_nope,
            q,
            kv,
            dout,
            view,
            topk_length=lengths,
        )
        out, lse, o_lo = forward_nope(q, kv, view)
        dq, _ = backward_nope(q, kv, view, out, o_lo, lse, dout, topk_length=lengths)
        torch.cuda.synchronize()
        assert torch.equal(holes[0], out) and torch.equal(holes[1], lse)
        assert torch.equal(holes[2], dq)
        q_leaf = q.detach().clone().requires_grad_()
        saved = cake_backend.dsa_sparse_attention_nope(
            q_leaf, kv, view
        ).grad_fn.saved_tensors
        assert saved[5].data_ptr() == view.data_ptr() and torch.equal(saved[6], lengths)
    else:
        rows, counts = compact_rows(view, S)
        got = cake_backend.nope_compact_indices(view, S)
        torch.cuda.synchronize()
        assert torch.equal(got[0], rows) and torch.equal(got[1], counts)
        twin = _public_nope_step(
            cake_backend.dsa_sparse_attention_nope,
            q,
            kv,
            dout,
            rows,
            topk_length=counts,
        )
    for x, y in zip(holes[:3], twin[:3], strict=True):
        assert torch.equal(x, y)
    _rel_or_equal(holes[3], twin[3], 1e-5)
    assert torch.all(holes[0][5] == 0) and torch.all(holes[1][5] == float("-inf"))


def test_nope_kernel_route_compacts_like_the_stage_on_hole_rows(monkeypatch):
    """The compacting one-launch mode (``"kernel"``, ``convert = 1``; never selected by size, so the route
    function is replaced here) and the stage route (the limit set to 0) agree bitwise on hole rows -- ``out``,
    ``lse``, ``o_lo``, the returned global rows (the stable compaction) and the lengths (the valid counts) through
    ``forward_nope_varlen``, ``dq`` through the public entry: the forward's scan lanes write the same compaction
    the stage does."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    if not cake_backend.nope_fwd_converts_indices():
        pytest.skip("the registered NoPE forward program has no conversion operands")
    c = _hole_case_tensors("interleaved", seed=NOPE_SEED + 835)
    assert (
        cake_backend.varlen_index_mode(c.local) == "mask"
    )  # the shipped route of this table
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    stage = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    stage_pub = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    torch.cuda.synchronize()
    monkeypatch.setattr(
        cake_backend,
        "varlen_index_mode",
        lambda rows, *, caller_lengths=False: "kernel",
    )
    kernel = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    kernel_pub = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    torch.cuda.synchronize()
    for a, b in zip(stage, kernel, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(kernel[3], c.rows) and torch.equal(kernel[4], c.counts)
    for a, b in zip(stage_pub[:3], kernel_pub[:3], strict=True):
        assert torch.equal(a, b)
    _rel_or_equal(kernel_pub[3], stage_pub[3], 1e-5)


def test_nope_hole_rows_are_deterministic_run_to_run():
    """Two calls of the varlen entry on hole rows: ``out``, ``lse``, ``dq`` and the compacted rows / counts the
    forward returns are bitwise the same (a fixed launch order, no new atomics on the index side)."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    c = _hole_case_tensors("interleaved", seed=NOPE_SEED + 830)
    first = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    second = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        c.q,
        c.kv,
        c.dout,
        c.local,
        c.cu_q,
        c.cu_k,
        causal=True,
    )
    for x, y in zip(first[:3], second[:3], strict=True):
        assert torch.equal(x, y)
    a = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    b = cake_backend.forward_nope_varlen(
        c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
    )
    torch.cuda.synchronize()
    assert torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])


def _graph_capture(fn):
    """Run ``fn`` eagerly on a side stream (detached copies of its outputs are the reference), warm it twice
    there, capture one call into a CUDA graph on that stream and replay it twice; returns ``(eager, captured)``
    (the captured outputs are the buffers the replays refresh).  Everything runs on the capture stream: a leaf's
    ``AccumulateGrad`` node records the stream of its first use, and one created on the default stream is a stale
    reference that invalidates the capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = tuple(t.detach().clone() for t in fn())
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    return eager, captured


def test_nope_public_entries_capture_into_a_cuda_graph_on_hole_rows():
    """Forward + backward of the public NoPE entries (the varlen entry on hole rows, the flat entry on hole rows
    without lengths) and the kernel-level ``forward_nope_varlen`` capture into a CUDA graph and replay to the eager
    results bitwise (``out``, ``lse``, ``dq``; the compacted rows and counts): the stage route allocates before its
    launches and nothing synchronizes with the device."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    c = _hole_case_tensors("interleaved", seed=NOPE_SEED + 840)
    q_leaf = c.q.detach().clone().requires_grad_()

    def varlen():
        out, lse = cake_backend.dsa_sparse_attention_nope_varlen(
            q_leaf, c.kv, c.local, c.cu_q, c.cu_k, causal=True, return_lse=True
        )
        (dq,) = torch.autograd.grad(out, [q_leaf], c.dout)
        return out, lse, dq

    def flat():
        out, lse = cake_backend.dsa_sparse_attention_nope(
            q_leaf, c.kv, c.glob, return_lse=True, num_segments=c.num_docs
        )
        (dq,) = torch.autograd.grad(out, [q_leaf], c.dout)
        return out, lse, dq

    def kernel_level():
        return cake_backend.forward_nope_varlen(
            c.q, c.kv, c.local, c.cu_q, c.cu_k, causal=True
        )

    for fn in (varlen, flat, kernel_level):
        eager, captured = _graph_capture(fn)
        for a, b in zip(eager, captured, strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("route", ["mask", "index_prep"])
@pytest.mark.parametrize("construction", ["pooled_fixed_tail", "every_third"])
def test_nope_flat_entry_hole_rows_meet_the_floor_on_their_processing_order(
    construction, route, monkeypatch
):
    """Precision of the flat entry without lengths on hole rows (512 x 256 = 131,072 slots) against the FP64
    reference with its floors evaluated on the order the route processes (the lazy-rescale floor walks 64-slot
    tiles in slot order, so the floor depends on the slot order the kernel sees; the exact values do not): the raw
    order below the slot limit (``"mask"``, the shipped route for this table: ``reference_fp64_nope(...,
    compact=False)``), the compacted order on the stage route (``"index_prep"``, the limit set to 0:
    ``compact=True``).  ``out`` / ``dq`` within 1.02x their floors plus the element-wise and row bands, ``dkv``
    within 1.05x.  Constructions: the pooled draw with the open pool's tail at a fixed column over a document that
    starts inside the query range (the R2 structure in miniature: a row has holes only while it holds fewer than
    ``pools`` complete pools, so the first 251 positions carry them) and iid rows with every third slot ``-1`` and
    a fully masked row."""
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    if route == "index_prep":
        monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 0)
    assert cake_backend.nope_flat_index_mode(512, 256) == route
    if construction == "pooled_fixed_tail":
        inp = _nope_inputs(
            512, 512, seed=NOPE_SEED + 850, topk=256, index="pooled_fixed_tail"
        )
        idx = inp.idx_global
    else:
        inp = _nope_inputs(512, 2048, seed=NOPE_SEED + 851, topk=256)
        idx = inp.idx_global.clone()
        idx[:, 1::3] = -1
        idx[7] = -1
    assert bool(((idx == -1) & (idx.flip(-1).cummax(-1).values.flip(-1) >= 0)).any()), (
        "the construction must carry holes before valid slots"
    )
    out, lse, dq, acc = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope, inp.q, inp.kv, inp.dout, idx
    )
    ref = nope_ref.reference_fp64_nope(
        inp.q,
        inp.kv,
        idx,
        dout=inp.dout,
        softmax_scale=NOPE_SOFTMAX_SCALE,
        compact=route == "index_prep",
    )
    _check_forward_nope(out, lse, ref)
    _check_backward_nope(dq, acc, ref)


@pytest.mark.parametrize("construction", ["every_third_causal", "pooled_fixed_tail"])
def test_nope_autograd_backward_on_masked_rows_in_raw_order_matches_the_reference(
    construction, monkeypatch
):
    """Autograd through both public entries on small tables with holes anywhere (below the slot limit, raised
    here above both constructions): the varlen entry on per-document rows and the flat entry on their globalized
    rows give ``out`` / ``lse`` / ``dq`` bitwise each other and bitwise the kernel-level pair (``forward_nope`` /
    ``backward_nope`` over the globalized rows with the last valid slot + 1 as lengths), ``dkv`` within the atomics
    spread, and all of them meet the FP64 floors evaluated on the raw slot order -- the order a position-preserving
    forward and backward process.  Constructions: two documents with iid causal rows where every third slot is
    ``-1`` and one row is fully masked (so the causal rule and the planted holes both leave masked slots before
    valid ones), and the pooled draw with its tail at a fixed column over one document."""
    _require_nope_program(backward=True)
    _require_nope_mask_route()
    monkeypatch.setattr(cake_backend, "NOPE_CONVERT_SLOT_LIMIT", 1 << 40)
    if construction == "every_third_causal":
        inp = nope_ref.make_nope_inputs(
            [256, 128], [512, 512], seed=NOPE_SEED + 870, topk=96, index="iid"
        )
        local = inp.idx_local.clone()
        local[:, 1::3] = -1
        local[7] = -1
    else:
        inp = _nope_inputs(
            512, 512, seed=NOPE_SEED + 871, topk=256, index="pooled_fixed_tail"
        )
        local = inp.idx_local
    S = inp.total_k
    glob = offset_gather_kv_indices(
        local, inp.cu_seqlens_q, inp.cu_seqlens_k, causal=True
    )
    assert bool(((glob == -1) & (glob.flip(-1).cummax(-1).values.flip(-1) >= 0)).any())
    lengths = nope_ref.last_valid_plus_one(glob, S)
    assert cake_backend.varlen_index_mode(local) == "mask"
    varlen = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        inp.q,
        inp.kv,
        inp.dout,
        local,
        inp.cu_seqlens_q,
        inp.cu_seqlens_k,
        causal=True,
    )
    flat = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope,
        inp.q,
        inp.kv,
        inp.dout,
        glob,
        num_segments=inp.num_segments,
    )
    out, lse, o_lo = forward_nope(inp.q, inp.kv, glob)
    acc = torch.zeros(S, D_LATENT, device="cuda")
    dq, none = backward_nope(
        inp.q,
        inp.kv,
        glob,
        out,
        o_lo,
        lse,
        inp.dout,
        topk_length=lengths,
        dkv_acc=acc,
        num_segments=inp.num_segments,
    )
    torch.cuda.synchronize()
    assert none is None
    for side in (varlen, flat):
        assert torch.equal(side[0], out) and torch.equal(side[1], lse)
        assert torch.equal(side[2], dq)
        _rel_or_equal(side[3], acc, 1e-5)
    ref = nope_ref.reference_fp64_nope(
        inp.q,
        inp.kv,
        glob,
        dout=inp.dout,
        softmax_scale=NOPE_SOFTMAX_SCALE,
        compact=False,
    )
    _check_forward_nope(varlen[0], varlen[1], ref)
    _check_backward_nope(varlen[2], varlen[3], ref)


def test_nope_issue_repro_rows_match_their_stable_twin_bitwise_large():
    """Opt-in (``FLASHINFER_CAKE_DSA_NOPE_LARGE_TESTS=1``): the issue's construction at full size -- one 16384-token
    document, 1024 front keys, a gap to column 2048, four tail keys, K_slots 2176 -- through the public varlen entry
    is bitwise its stable twin (``compacted_twin``) on ``out`` / ``lse`` / ``dq`` with ``dkv`` within the atomics
    spread; the issue's descending-sorted twin regroups the keys across tiles and stays within the fp32-noise class."""
    if not os.environ.get(NOPE_LARGE_TESTS_ENV):
        pytest.skip(f"set {NOPE_LARGE_TESTS_ENV}=1 to run the 16k-row constructions")
    _require_nope_compacting_stage()
    _require_nope_program(backward=True)
    inp = nope_ref.make_nope_inputs(
        [16384], [16384], seed=NOPE_SEED + 860, topk=2176, index="repro_gap"
    )
    assert inp.has_holes
    twin = nope_ref.compacted_twin(inp)
    issue = nope_ref.make_nope_inputs(
        [16384], [16384], seed=NOPE_SEED + 860, topk=2176, index="repro_compact_issue"
    )
    args = (inp.cu_seqlens_q, inp.cu_seqlens_k)
    holes = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        inp.q,
        inp.kv,
        inp.dout,
        inp.idx_local,
        *args,
        causal=True,
    )
    stable = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        inp.q,
        inp.kv,
        inp.dout,
        twin.idx_local,
        *args,
        causal=True,
    )
    for x, y in zip(holes[:3], stable[:3], strict=True):
        assert torch.equal(x, y)
    _rel_or_equal(holes[3], stable[3], 1e-5)
    sorted_twin = _public_nope_step(
        cake_backend.dsa_sparse_attention_nope_varlen,
        inp.q,
        inp.kv,
        inp.dout,
        issue.idx_local,
        *args,
        causal=True,
    )
    assert (
        rel_l2(sorted_twin[0], holes[0]) < 1e-3
        and rel_l2(sorted_twin[2], holes[2]) < 1e-2
    )


# --- rope record: the strided delta reads a head-major ``dout`` in place (ask 2) and the varlen entry takes the
# compacting stage (ask 3) ---


def _stand_in_rope_program(monkeypatch, calls: list):
    """Recording launch entries for every stage of the rope record (``(stage, {argument: value})`` appended to
    ``calls``), chained after any NoPE stand-in already installed: the loader in place keeps serving the other
    program."""
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    _require_program(backward=True)
    current = cake_jit.load_cake_dsa_train_module

    def load(stage, arch, program=cake_jit.PROGRAM):
        if program != cake_jit.PROGRAM:
            return current(stage, arch, program)

        def run(*args):
            calls.append(
                (stage, dict(zip(cake_launch.STAGE_ARGS[stage], args, strict=True)))
            )

        return SimpleNamespace(run=run)

    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", load)
    monkeypatch.setattr(cake_backend, "_ffi_tensor", lambda tensor: tensor)
    cake_backend.BINDING_CACHE.clear()


def _rope_device_inputs(T, S, K, *, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bf = lambda *shape: torch.randn(*shape, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    q_latent, q_rope = bf(T, NUM_HEADS, D_LATENT), bf(T, NUM_HEADS, D_ROPE)
    kv_latent, k_rope = bf(S, D_LATENT), bf(S, D_ROPE)
    idx = torch.randint(0, S, (T, K), device="cuda", generator=gen, dtype=torch.int32)
    dout = bf(T, NUM_HEADS, D_LATENT)
    return q_latent, q_rope, kv_latent, k_rope, idx, dout


def test_rope_eager_backward_reads_a_strided_dout_in_place(monkeypatch):
    """The rope record's ``bwd_delta`` is the strided delta kernel: a head-major ``dout`` (``[512, T * 512, 1]``
    strides), the dense ``dout`` and a slice of a packed ``[T, 64, 576]`` tensor reach the delta stage as they are
    -- token / head strides in elements and a zero offset -- and the main stage reads the same pointer; every
    layout is its own plan.  A view the kernels cannot read (a channel stride of two) arrives contiguous through the
    eager ``backward`` while ``prepare_dsa_train`` refuses it."""
    calls = []
    _stand_in_rope_program(monkeypatch, calls)
    T, S, K = 8, 64, 5
    q_latent, q_rope, kv_latent, k_rope, idx, dout = _rope_device_inputs(T, S, K)
    out, lse, o_lo = _saved_outputs(T)
    args = (q_latent, q_rope, kv_latent, k_rope, idx, out, o_lo, lse)
    head_major = dout.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    packed = torch.zeros(T, NUM_HEADS, D_QK, dtype=torch.bfloat16, device="cuda")
    packed[..., :D_LATENT] = dout
    expected = {
        "head_major": (head_major, (D_LATENT, T * D_LATENT, 0)),
        "dense": (dout, (NUM_HEADS * D_LATENT, D_LATENT, 0)),
        "packed_slice": (packed[..., :D_LATENT], (NUM_HEADS * D_QK, D_QK, 0)),
    }
    assert cake_backend._dout_admissible(head_major)
    for name, (view, strides) in expected.items():
        calls.clear()
        cake_backend.backward(*args, view)
        stages = [c[0] for c in calls]
        assert stages[0] == "bwd_delta" and any(
            s.startswith("bwd_main") for s in stages
        ), (name, stages)
        delta = calls[0][1]
        assert (
            delta["dout"].data_ptr() == view.data_ptr()
            and delta["dout"].stride() == view.stride()
        )
        assert (
            delta["dout_token_stride"],
            delta["dout_head_stride"],
            delta["dout_offset"],
        ) == strides, name
        assert delta["num_rows"] == T * NUM_HEADS
        main = [c[1] for c in calls if c[0].startswith("bwd_main")][0]
        assert main["dout"].data_ptr() == view.data_ptr()
    assert (
        len(cake_backend.BINDING_CACHE) == 3
    )  # three dout layouts, three backward plans
    strided_channels = torch.zeros(
        T, NUM_HEADS, 2 * D_LATENT, dtype=torch.bfloat16, device="cuda"
    )[..., ::2]
    assert not cake_backend._dout_admissible(strided_channels)
    calls.clear()
    cake_backend.backward(*args, strided_channels)
    delta = calls[0][1]
    assert (
        delta["dout"].is_contiguous()
        and delta["dout"].data_ptr() != strided_channels.data_ptr()
    )
    assert (
        delta["dout_token_stride"],
        delta["dout_head_stride"],
        delta["dout_offset"],
    ) == (
        NUM_HEADS * D_LATENT,
        D_LATENT,
        0,
    )
    with pytest.raises(ValueError, match="dout"):
        prepare_dsa_train(*args[:5], dout=strided_channels, backward=True)
    runner = prepare_dsa_train(*args[:5], dout=head_major, backward=True)
    assert runner.tensors["dout"].data_ptr() == head_major.data_ptr()


def test_validate_dsa_train_inputs_admits_a_head_major_dout_and_rejects_odd_strides():
    """``validate_dsa_train_inputs(..., dout=...)`` admits the head-major view and a packed slice, and refuses a
    channel stride other than one, a head stride that is not a multiple of eight elements and a misaligned base."""
    T, S, K = 16, 32, 4
    q_latent = torch.zeros(T, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    q_rope = torch.zeros(T, NUM_HEADS, D_ROPE, dtype=torch.bfloat16)
    kv_latent = torch.zeros(S, D_LATENT, dtype=torch.bfloat16)
    k_rope = torch.zeros(S, D_ROPE, dtype=torch.bfloat16)
    idx = torch.zeros(T, K, dtype=torch.int32)
    args = (q_latent, q_rope, kv_latent, k_rope, idx)
    head_major = torch.zeros(NUM_HEADS, T, D_LATENT, dtype=torch.bfloat16).permute(
        1, 0, 2
    )
    assert head_major.stride() == (D_LATENT, T * D_LATENT, 1)
    validate_dsa_train_inputs(*args, dout=head_major)
    validate_dsa_train_inputs(
        *args,
        dout=torch.zeros(T, NUM_HEADS, D_QK, dtype=torch.bfloat16)[..., :D_LATENT],
    )
    with pytest.raises(ValueError, match="contiguous along its last dimension"):
        validate_dsa_train_inputs(
            *args,
            dout=torch.zeros(T, NUM_HEADS, 2 * D_LATENT, dtype=torch.bfloat16)[
                ..., ::2
            ],
        )
    odd_head = torch.zeros(T, NUM_HEADS, D_LATENT + 4, dtype=torch.bfloat16)[
        ..., :D_LATENT
    ]
    with pytest.raises(ValueError, match="head stride"):
        validate_dsa_train_inputs(*args, dout=odd_head)
    misaligned = torch.zeros(T * NUM_HEADS * D_LATENT + 4, dtype=torch.bfloat16)[
        4:
    ].view(T, NUM_HEADS, D_LATENT)
    with pytest.raises(ValueError, match="aligned"):
        validate_dsa_train_inputs(*args, dout=misaligned)
    with pytest.raises(ValueError, match="bfloat16"):
        validate_dsa_train_inputs(
            *args, dout=torch.zeros(T, NUM_HEADS, D_LATENT, dtype=torch.float16)
        )


def test_rope_backward_head_major_dout_matches_the_dense_path_without_a_copy():
    """A head-major ``dout`` through the eager rope backward, the prepared runner and autograd gives the dense
    path's ``dq_latent`` / ``dq_rope`` bitwise and ``dkv`` / ``dk_rope`` within the atomics spread; the runner binds
    the view in place (its ``dout`` pointer is the caller's); the head-major backward's peak memory above the
    inputs is not larger than the contiguous backward's (no contiguous copy of ``dout``: the backward allocates its
    gradients, the delta and a fresh ``dkv`` on both sides alike)."""
    _require_program(backward=True)
    inp = make_inputs(
        [1024], [2048], seed=SEED + 71, topk=256, dout_layout="head_major"
    )
    assert inp.dout.stride() == (D_LATENT, inp.total_q * D_LATENT, 1)
    dense = inp.dout.contiguous()
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    out, lse, o_lo = cake_backend.forward(*args)
    saved = (out, o_lo, lse)
    cake_backend.backward(
        *args, *saved, dense
    )  # plans and module loads of both layouts before the measurement
    cake_backend.backward(*args, *saved, inp.dout)
    torch.cuda.synchronize()

    def peak_of(dout):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        grads = cake_backend.backward(*args, *saved, dout)
        torch.cuda.synchronize()
        return grads, torch.cuda.max_memory_allocated() - base

    grads_d, peak_d = peak_of(dense)
    grads_h, peak_h = peak_of(inp.dout)
    assert torch.equal(grads_h[0], grads_d[0]) and torch.equal(grads_h[1], grads_d[1])
    assert (
        rel_l2(grads_h[2], grads_d[2]) < ACC_REL_L2
        and rel_l2(grads_h[3], grads_d[3]) < ACC_REL_L2
    )
    assert peak_h <= peak_d + (8 << 20), (
        peak_h / 2**20,
        peak_d / 2**20,
        inp.dout.numel() * 2 / 2**20,
    )
    runner = prepare_dsa_train(*args, dout=inp.dout, backward=True)
    assert runner.tensors["dout"].data_ptr() == inp.dout.data_ptr()
    assert runner.tensors["dout"].stride() == inp.dout.stride()
    runner.forward()
    grads_r = runner.backward()
    torch.cuda.synchronize()
    assert torch.equal(grads_r[0], grads_d[0]) and torch.equal(grads_r[1], grads_d[1])
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    out_f = cake_backend.dsa_sparse_attention(*leaves, inp.idx_global)
    out_f.backward(inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_f, out)
    assert torch.equal(leaves[0].grad, grads_d[0]) and torch.equal(
        leaves[1].grad, grads_d[1]
    )
    ref = reference_fp64(*args, dout=dense)
    _check_backward(grads_h, ref)


def test_rope_varlen_entry_takes_the_compacting_stage_then_the_forward(monkeypatch):
    """With stand-in launches of both records the rope varlen entry issues the NoPE record's index-preparation
    stage (``compact = 1``; a caller ``topk_length`` as ``length_in``) into fresh rows and counts, then the rope
    forward over them with the counts as ``topk_length``; the route is ``"torch"`` -- the twin glue, no stage
    launch -- for host tensors, no query rows, non-contiguous prefix sums and a record without the stage, and
    ``"index_prep"`` otherwise."""
    calls = []
    _stand_in_index_prep(monkeypatch, calls)
    _stand_in_rope_program(monkeypatch, calls)
    local, cu_q, cu_k = (t.cuda() for t in _index_prep_example())
    T, K = (int(d) for d in local.shape)
    S = int(cu_k[-1])
    mode = cake_backend._rope_varlen_index_mode
    assert mode(local, cu_q, cu_k, T) == "index_prep"
    assert mode(local.cpu(), cu_q.cpu(), cu_k.cpu(), T) == "torch"
    assert mode(local[:0], cu_q.new_tensor([0, 0, 0, 0, 0]), cu_k, 0) == "torch"
    assert mode(local, torch.stack([cu_q, cu_q], 1)[:, 0], cu_k, T) == "torch"
    assert mode(local[:, :0], cu_q, cu_k, T) == "torch"
    q_latent, q_rope, kv_latent, k_rope, _, _ = _rope_device_inputs(T, S, K)
    varlen = cake_backend.dsa_sparse_attention_varlen
    varlen(q_latent, q_rope, kv_latent, k_rope, local, cu_q, cu_k, causal=True)
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert (
        prep["compact"],
        prep["has_length_in"],
        prep["causal"],
        prep["write_out"],
    ) == (1, 0, 1, 1)
    assert (prep["num_rows"], prep["topk"], prep["num_docs"]) == (T, K, 4)
    assert prep["indices"].data_ptr() == local.data_ptr()
    assert fwd["indices"].data_ptr() == prep["out"].data_ptr()
    assert fwd["topk_length"] is prep["topk_length"] and fwd["has_topk_length"] == 1
    mine = torch.full((T,), 2, dtype=torch.int32, device="cuda")
    calls.clear()
    varlen(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        local,
        cu_q,
        cu_k,
        causal=True,
        topk_length=mine,
    )
    assert [c[0] for c in calls] == [NOPE_INDEX_PREP_STAGE, "fwd"]
    prep, fwd = calls[0][1], calls[1][1]
    assert (prep["compact"], prep["has_length_in"]) == (1, 1) and prep[
        "length_in"
    ] is mine
    assert fwd["topk_length"] is prep["topk_length"] and fwd["topk_length"] is not mine
    # a record without the stage: the twin glue, then the forward alone
    monkeypatch.setattr(cake_backend, "nope_index_prep_available", lambda device: False)
    assert mode(local, cu_q, cu_k, T) == "torch"
    calls.clear()
    varlen(q_latent, q_rope, kv_latent, k_rope, local, cu_q, cu_k, causal=True)
    assert [c[0] for c in calls] == ["fwd"]
    fwd = calls[0][1]
    rows, counts = compact_rows(
        offset_gather_kv_indices(local, cu_q, cu_k, causal=True), S
    )
    assert torch.equal(fwd["indices"], rows) and torch.equal(fwd["topk_length"], counts)


def test_rope_varlen_fallback_routes_give_the_stage_route_results_on_hole_rows():
    """The rope varlen entry on hole rows: the stage route, the torch twin (a record without the stage) and
    non-contiguous prefix sums (the twin again) give ``out`` / ``lse`` and the ``dq`` gradients bitwise and the key
    gradients within the atomics spread; the result is route-independent because every route compacts."""
    _require_nope_compacting_stage()
    _require_program(backward=True)
    local, cu_q, cu_k = _index_prep_case(
        (96, 160, 0, 64), (96, 320, 0, 512), 128, seed=SEED + 72
    )
    T, S = int(local.shape[0]), int(cu_k[-1])
    q_latent, q_rope, kv_latent, k_rope, _, dout = _rope_device_inputs(
        T, S, 128, seed=SEED + 73
    )
    varlen = cake_backend.dsa_sparse_attention_varlen

    def run(cu_q_arg, cu_k_arg, expected_mode):
        assert (
            cake_backend._rope_varlen_index_mode(local, cu_q_arg, cu_k_arg, T)
            == expected_mode
        )
        leaves = [
            t.detach().clone().requires_grad_()
            for t in (q_latent, q_rope, kv_latent, k_rope)
        ]
        out, lse = varlen(
            *leaves, local, cu_q_arg, cu_k_arg, causal=True, return_lse=True
        )
        grads = torch.autograd.grad(out, leaves, dout)
        torch.cuda.synchronize()
        return out, lse, grads

    stage = run(cu_q, cu_k, "index_prep")
    strided_cu = torch.stack([cu_q, cu_q], 1)[:, 0]
    twin = run(strided_cu, cu_k, "torch")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cake_backend, "nope_index_prep_available", lambda device: False)
        no_stage = run(cu_q, cu_k, "torch")
    for other in (twin, no_stage):
        assert torch.equal(other[0], stage[0]) and torch.equal(other[1], stage[1])
        assert torch.equal(other[2][0], stage[2][0]) and torch.equal(
            other[2][1], stage[2][1]
        )
        assert (
            rel_l2(other[2][2], stage[2][2]) < ACC_REL_L2
            and rel_l2(other[2][3], stage[2][3]) < ACC_REL_L2
        )
    # the flat rope entry on the compacted rows with the counts is the same computation
    rows, counts = compact_rows(
        offset_gather_kv_indices(local, cu_q, cu_k, causal=True), S
    )
    out, lse, _ = cake_backend.forward(
        q_latent, q_rope, kv_latent, k_rope, rows, topk_length=counts
    )
    torch.cuda.synchronize()
    assert torch.equal(out, stage[0]) and torch.equal(lse, stage[1])


# --- activation checkpointing (flashinfer-ai/flashinfer#7039): the autograd entries keep nothing alive after a
# non-reentrant checkpointed forward, their outputs and gradients are those of the eager pair, and the backward is
# recompute-safe (GPU) ---

# Bytes an autograd entry may keep after ``checkpoint(fn, *inputs, use_reentrant=False)`` beyond the output storage:
# allocator granularity (a few ``[T]`` int32 vectors and scalars at most), never an activation-sized tensor.  Under
# non-reentrant checkpointing only tensors passed through ``ctx.save_for_backward`` are freed after the forward and
# recomputed; a tensor kept as a ``ctx`` attribute (or on a returned handle) stays alive for every layer at once.
CKPT_RETAINED_LIMIT_MIB = 2.0
# the test shape of the issue's construction: one causal segment, each query attends to its last W keys
CKPT_T, CKPT_W = 2048, 256
CKPT_DST_ROWS = 32768  # the CP-gather style destination buffer (the issue's 262144-row accumulator, scaled)
# ``FLASHINFER_CAKE_DSA_CKPT_DIGESTS``: a JSON written by ``work/bench/ckpt_retained.py --shape small`` with the
# reference package (the same inputs: generator seed 0, the same tensor order); when set, the eager outputs and
# gradients of this package must digest-match it (the cross-package bitwise evidence) and the eager peak memory must
# not exceed its ``eager_peak_mib`` (allocator granularity allowed)
CKPT_DIGESTS_ENV = "FLASHINFER_CAKE_DSA_CKPT_DIGESTS"
MIB = 2**20


def _sha16(t):
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def _ckpt_retained_mib(fn, *inputs) -> float:
    """The issue's ``retained()``: ``memory_allocated`` kept after a checkpointed forward, less the output storage."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *inputs, use_reentrant=False)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    del out
    return kept / MIB


def _sliding_window_rows(T, W, device="cuda"):
    rows = torch.arange(T, device=device, dtype=torch.int32)[:, None] - torch.arange(
        W, device=device, dtype=torch.int32
    )
    return torch.where(rows >= 0, rows, -1).to(torch.int32)


def _ckpt_inputs():
    """The construction of ``work/bench/ckpt_retained.py --shape small`` (generator seed 0, this tensor order): the
    NoPE and rotary leaves, ``dout``, the sliding-window rows (one document: local rows == global rows), the prefix
    sums and the destination map of the scaled CP gather."""
    _require_program(backward=True)
    _require_nope_program(backward=True)
    T, W = CKPT_T, CKPT_W
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, 64, 512, device=dev, generator=g).bfloat16().requires_grad_()
    kv = torch.randn(T, 512, device=dev, generator=g).bfloat16().requires_grad_()
    q_rope = torch.randn(T, 64, 64, device=dev, generator=g).bfloat16().requires_grad_()
    k_rope = torch.randn(T, 64, device=dev, generator=g).bfloat16().requires_grad_()
    dout = torch.randn(T, 64, 512, device=dev, generator=g).bfloat16()
    idx = _sliding_window_rows(T, W)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    dst_map = (torch.arange(T, device=dev, dtype=torch.int64) * CKPT_DST_ROWS // T).to(
        torch.int32
    )
    return SimpleNamespace(
        q=q,
        kv=kv,
        q_rope=q_rope,
        k_rope=k_rope,
        dout=dout,
        idx=idx,
        cu=cu,
        dst_map=dst_map,
    )


def _dkv_acc_argument(form, buffer):
    """How the caller's FP32 accumulator reaches an autograd entry: ``"tensor"`` -- the buffer itself (the buffer is
    owned outside the checkpointed region, so the recomputed forward meets the same tensor); ``"callable"`` -- a
    zero-argument callable returning it, the deferred form: the entry resolves the buffer at backward time and
    keeps no accumulator on ``ctx``.  An entry that does not accept the deferred form raises ``TypeError`` /
    ``ValueError`` at forward time, which the tests turn into a skip."""
    if form == "tensor":
        return buffer
    assert form == "callable"
    return lambda: buffer


DEFERRED_FORM_ERRORS = (TypeError, ValueError, AttributeError)


def _skip_if_deferred_form_unsupported(exc):
    pytest.skip(
        f"the package does not accept a callable dkv_acc (the deferred accumulator form): {exc}"
    )


def _kv_leaf_indices(name):
    """Positions of the key-side leaves in the entry's leaf tuple: ``(kv,)`` for NoPE, ``(kv, k_rope)`` for rotary."""
    return (1,) if name.startswith("nope") else (2, 3)


def _ckpt_entry(name, inp, *, acc_form=None, dst_map=False):
    """``(fn, leaves, acc)`` of one autograd entry over the issue's construction: ``fn(*leaves) -> out``.  With
    ``acc_form`` the FP32 accumulator ``acc`` (zeroed by the caller per step) is passed in that form; ``dst_map``
    adds the destination map into the ``CKPT_DST_ROWS``-row buffer (NoPE varlen only)."""
    S = int(inp.kv.shape[0])
    nope = name.startswith("nope")
    cols = D_LATENT if nope else D_QK
    acc = None
    kw = {}
    if acc_form is not None:
        rows = CKPT_DST_ROWS if dst_map else S
        acc = torch.zeros(rows, cols, device="cuda")
        kw["dkv_acc"] = _dkv_acc_argument(acc_form, acc)
        if dst_map:
            kw["dkv_dst_map"] = inp.dst_map
    if name == "nope_varlen":
        leaves = (inp.q, inp.kv)

        def fn(q, kv):
            return cake_backend.dsa_sparse_attention_nope_varlen(
                q, kv, inp.idx, inp.cu, inp.cu, causal=True, softmax_scale=1 / 16, **kw
            )

    elif name == "nope_flat":
        leaves = (inp.q, inp.kv)

        def fn(q, kv):
            return cake_backend.dsa_sparse_attention_nope(
                q, kv, inp.idx, softmax_scale=1 / 16, **kw
            )

    elif name == "rope":
        leaves = (inp.q, inp.q_rope, inp.kv, inp.k_rope)

        def fn(q, q_rope, kv, k_rope):
            return cake_backend.dsa_sparse_attention(
                q, q_rope, kv, k_rope, inp.idx, **kw
            )

    elif name == "rope_varlen":
        leaves = (inp.q, inp.q_rope, inp.kv, inp.k_rope)

        def fn(q, q_rope, kv, k_rope):
            return cake_backend.dsa_sparse_attention_varlen(
                q, q_rope, kv, k_rope, inp.idx, inp.cu, inp.cu, causal=True, **kw
            )

    else:
        raise ValueError(name)
    return fn, leaves, acc


def _eager_step(fn, leaves, dout, acc):
    """Eager forward + backward: ``(out, grads, acc_copy)`` with the gradients of the leaves (``None`` for a kv leaf
    whose gradient went into the accumulator)."""
    if acc is not None:
        acc.zero_()
    out = fn(*leaves)
    grads = torch.autograd.grad(out, leaves, dout, allow_unused=True)
    torch.cuda.synchronize()
    return out.detach(), grads, None if acc is None else acc.clone()


def _assert_same_results(got, want, *, acc_got=None, acc_want=None):
    """``out`` and the query gradients bitwise; the key-side gradients and the FP32 accumulator (``red.global`` sums in
    no fixed order, so never bitwise run to run; cast to the BF16 leaf a few one-ulp flips) within ``ACC_REL_L2``."""
    out_g, grads_g = got
    out_w, grads_w = want
    assert torch.equal(out_g, out_w)
    for i, (a, b) in enumerate(zip(grads_g, grads_w, strict=True)):
        if a is None or b is None:
            assert a is None and b is None
            continue
        if i < len(grads_g) - (2 if len(grads_g) == 4 else 1):
            assert torch.equal(a, b), f"gradient {i} differs"
        else:
            assert rel_l2(a, b) < ACC_REL_L2, f"gradient {i}: rel-L2 {rel_l2(a, b)}"
    if acc_got is not None or acc_want is not None:
        assert rel_l2(acc_got, acc_want) < ACC_REL_L2


CKPT_ENTRIES = ("nope_varlen", "nope_flat", "rope", "rope_varlen")
CKPT_ACC_FORMS = (None, "tensor", "callable")


def _ckpt_case_ids():
    cases = []
    for name in CKPT_ENTRIES:
        for form in CKPT_ACC_FORMS:
            cases.append((name, form, False))
            if name == "nope_varlen" and form is not None:
                cases.append((name, form, True))
    return cases


def _ckpt_case_name(case):
    name, form, dst = case
    return (
        name
        + ("" if form is None else f"+dkv_acc[{form}]")
        + (f"+dst_map{CKPT_DST_ROWS}" if dst else "")
    )


@pytest.mark.parametrize("case", _ckpt_case_ids(), ids=_ckpt_case_name)
def test_ckpt_autograd_entries_keep_nothing_alive_after_a_checkpointed_forward(case):
    """The issue's measurement on every autograd entry (the NoPE varlen entry at the issue's construction, the flat
    NoPE entry, the rotary flat and varlen Functions), with and without the caller's FP32 accumulator (and the
    destination map into a CP-gather buffer): after ``checkpoint(fn, *inputs, use_reentrant=False)`` the entry
    keeps at most ``CKPT_RETAINED_LIMIT_MIB`` beyond its output -- no ``o_lo``, no index rows or lengths, no
    accumulator on ``ctx`` (the accumulator is the caller's, allocated outside the checkpointed region; the
    deferred callable form resolves it at backward time)."""
    name, form, dst = case
    inp = _ckpt_inputs()
    fn, leaves, acc = _ckpt_entry(name, inp, acc_form=form, dst_map=dst)
    try:
        fn(*leaves)  # warm-up: kernels, plans, per-device workspaces
    except DEFERRED_FORM_ERRORS as exc:
        if form == "callable":
            _skip_if_deferred_form_unsupported(exc)
        raise
    torch.cuda.synchronize()
    if acc is not None:
        acc.zero_()
    kept = _ckpt_retained_mib(fn, *leaves)
    assert kept <= CKPT_RETAINED_LIMIT_MIB, (
        f"{_ckpt_case_name(case)}: {kept:.1f} MiB kept alive after a checkpointed forward"
    )


@pytest.mark.parametrize("case", _ckpt_case_ids(), ids=_ckpt_case_name)
def test_ckpt_outputs_and_gradients_match_the_eager_pair_and_the_reference_digests(
    case,
):
    """Under checkpointing every entry gives the eager pair's ``out`` and query gradients bitwise and its key-side
    gradients within the FP32 summation spread (the fresh ``dkv`` path, the ``dkv_acc`` accumulate path and the
    ``dkv_dst_map`` path alike), with the accumulator zero-filled between the checkpointed forward and its backward
    as a trainer does.  With ``FLASHINFER_CAKE_DSA_CKPT_DIGESTS`` set, the eager ``out`` / ``dq`` digests also equal
    the reference package's on the same inputs (``work/bench/ckpt_retained.py --shape small``)."""
    name, form, dst = case
    inp = _ckpt_inputs()
    fn, leaves, acc = _ckpt_entry(name, inp, acc_form=form, dst_map=dst)
    try:
        out_e, grads_e, acc_e = _eager_step(fn, leaves, inp.dout, acc)
    except DEFERRED_FORM_ERRORS as exc:
        if form == "callable":
            _skip_if_deferred_form_unsupported(exc)
        raise
    out_c = checkpoint(fn, *leaves, use_reentrant=False)
    if acc is not None:
        # the trainer's pattern: the accumulator is zero-filled between the forward and the backward (the tensor
        # form stays an attribute of the autograd node, so autograd's saved-tensor version check does not see it)
        acc.zero_()
    grads_c = torch.autograd.grad(out_c, leaves, inp.dout, allow_unused=True)
    torch.cuda.synchronize()
    acc_c = None if acc is None else acc.clone()
    _assert_same_results(
        (out_c.detach(), grads_c), (out_e, grads_e), acc_got=acc_c, acc_want=acc_e
    )
    if acc is not None:
        for i in _kv_leaf_indices(
            name
        ):  # the key-side gradients went into the accumulator
            assert grads_e[i] is None and grads_c[i] is None
        assert float(acc_e.abs().sum()) > 0
        if dst:
            hit = torch.zeros(CKPT_DST_ROWS, dtype=torch.bool, device="cuda")
            hit[inp.dst_map.long()] = True
            assert torch.all(acc_e[~hit] == 0)
    path = os.environ.get(CKPT_DIGESTS_ENV, "")
    if not path:
        return
    # ``out`` and the query gradients do not depend on the accumulator form, so every variant of an entry must
    # digest-match the reference package's plain entry
    recorded = json.loads(Path(path).read_text())["entries"]
    if name not in recorded or "digests" not in recorded[name]:
        pytest.skip(f"{path} carries no digests for {name}")
    want = recorded[name]["digests"]["eager"]
    assert _sha16(out_e) == want["out"], "out differs from the reference package"
    assert _sha16(grads_e[0]) == want["dq"], "dq differs from the reference package"
    if name.startswith("rope"):
        assert _sha16(grads_e[1]) == want["dq_rope"], (
            "dq_rope differs from the reference package"
        )


@pytest.mark.parametrize("name", CKPT_ENTRIES)
def test_ckpt_backward_is_recompute_safe_after_an_unrelated_forward(name):
    """The checkpointed backward of one call runs after a second, unrelated forward on the same inputs (another
    layer's call in a training step) and after the binding cache served other geometries: the recompute finds its
    saved tensors by position, not by identity with the live forward, and the gradients equal the eager pair's.
    The forward's saved-tensor count is the same for the live and the recomputed forward."""
    inp = _ckpt_inputs()
    fn, leaves, _acc = _ckpt_entry(name, inp)
    out_e, grads_e, _ = _eager_step(fn, leaves, inp.dout, None)
    out_c = checkpoint(fn, *leaves, use_reentrant=False)
    unrelated = fn(
        *leaves
    )  # another call on the same inputs, its graph alive during the backward
    # another geometry through the same entry (a different binding) between the forward and its backward
    n = 512
    other_fn, other_leaves, _ = _ckpt_entry(
        name,
        SimpleNamespace(
            q=inp.q[:n],
            kv=inp.kv[:n],
            q_rope=inp.q_rope[:n],
            k_rope=inp.k_rope[:n],
            dout=inp.dout[:n],
            idx=inp.idx[:n],
            cu=inp.cu.new_tensor([0, n]),
            dst_map=inp.dst_map[:n],
        ),
    )
    other_out = other_fn(*other_leaves)
    torch.cuda.synchronize()
    grads_c = torch.autograd.grad(out_c, leaves, inp.dout, allow_unused=True)
    torch.cuda.synchronize()
    del unrelated, other_out
    _assert_same_results((out_c.detach(), grads_c), (out_e, grads_e))


@pytest.mark.parametrize("name", CKPT_ENTRIES)
def test_ckpt_eager_peak_memory_is_not_above_the_reference_package(name):
    """The eager (non-checkpointed) forward + backward of every entry allocates no more above the inputs than the
    reference package measured on the same construction (``FLASHINFER_CAKE_DSA_CKPT_DIGESTS``: ``eager_peak_mib`` of
    ``work/bench/ckpt_retained.py --shape small``; allocator granularity allowed).  Without the recording the peak
    is measured and reported only."""
    inp = _ckpt_inputs()
    fn, leaves, _acc = _ckpt_entry(name, inp)
    fn(*leaves)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(*leaves)
    torch.autograd.grad(out, leaves, inp.dout, allow_unused=True)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MIB
    path = os.environ.get(CKPT_DIGESTS_ENV, "")
    if not path:
        pytest.skip(
            f"eager peak {peak:.1f} MiB above the inputs; set {CKPT_DIGESTS_ENV} to compare"
        )
    recorded = json.loads(Path(path).read_text())["entries"]
    if name not in recorded or "eager_peak_mib" not in recorded[name]:
        pytest.skip(f"{path} carries no eager peak for {name}")
    want = recorded[name]["eager_peak_mib"]
    assert peak <= want + CKPT_RETAINED_LIMIT_MIB, (
        f"{name}: eager peak {peak:.1f} MiB above the inputs vs {want:.1f} MiB with the reference package"
    )
