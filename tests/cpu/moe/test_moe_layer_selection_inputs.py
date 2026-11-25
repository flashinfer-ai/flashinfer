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

``MoELayer`` selects its per-bucket backend on serving-like inputs.

Frameworks make the first call of every token count with a warm-up batch on
static, often zero-filled CUDA-graph buffers, so every token in it is identical:
router logits are identical across tokens and every token picks the same
``top_k`` experts. A kernel that is fast when tokens share experts wins that
call although it loses on served traffic, and the captured graph keeps it.
These tests pin down the selection inputs
(``flashinfer.fused_moe.selection``) and the layer contract around them, with
fake runners whose cost depends on the routing they are timed on.
"""

import hashlib
import os
import subprocess
import sys
from collections import OrderedDict

import pytest
import torch

import flashinfer.fused_moe.layer as layer_module
from flashinfer.autotuner import TuningConfig
from flashinfer.fused_moe import (
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe import selection
from flashinfer.tllm_enums import RoutingMethodType

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS, TOP_K, HIDDEN = 256, 8, 64
# GLM-5.2-style routing: DeepSeekV3 scoring with a correction bias, one group.
GLM_ROUTING = RoutingConfig(
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    method=RoutingMethodType.DeepSeekV3,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=2.5,
)
GROUPED_ROUTING = RoutingConfig(
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    method=RoutingMethodType.DeepSeekV3,
    n_group=8,
    topk_group=4,
)


def _bias():
    return torch.linspace(-0.05, 0.05, NUM_EXPERTS)


def _logits_pack(logits, *, hidden=None, bias=None):
    tokens = logits.shape[0]
    if hidden is None:
        hidden = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16)
    return MoEActivationPack(
        hidden_states_q=hidden,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=_bias() if bias is None else bias,
    )


def _degenerate(tokens):
    """A warm-up-like batch: identical tokens (all-zero rows here), so every
    token picks the same ``top_k`` experts."""
    return _logits_pack(torch.zeros(tokens, NUM_EXPERTS))


def _served(tokens, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return _logits_pack(
        torch.randn(tokens, NUM_EXPERTS, generator=generator) * 1.2 - 2.8,
        hidden=torch.randn(tokens, HIDDEN, generator=generator).bfloat16(),
    )


def _selected(pack, routing=GLM_ROUTING):
    """The experts a DeepSeekV3 router (one group) selects for ``pack``."""
    scores = torch.sigmoid(pack.routing_logits.float())
    if pack.routing_bias is not None:
        scores = scores + pack.routing_bias.float()
    return torch.topk(scores, routing.top_k, dim=-1).indices


def _tensors(pack):
    return {
        name: value
        for name, value in vars(pack).items()
        if isinstance(value, torch.Tensor)
    }


def _assert_same_values(a, b):
    ta, tb = _tensors(a), _tensors(b)
    assert ta.keys() == tb.keys()
    for name in ta:
        assert torch.equal(ta[name], tb[name]), name


# ---------------------------------------------------------------------------
# Selection inputs
# ---------------------------------------------------------------------------


def test_selection_pack_keeps_geometry_weights_and_bias():
    base = torch.zeros(6, 2 * HIDDEN, dtype=torch.bfloat16)
    caller = _logits_pack(torch.zeros(6, NUM_EXPERTS), hidden=base[:, ::2])
    synthetic = selection.representative_pack(caller, GLM_ROUTING, QuantFormat.BF16)

    for name, value in _tensors(caller).items():
        other = getattr(synthetic, name)
        assert (other.shape, other.dtype, other.device) == (
            value.shape,
            value.dtype,
            value.device,
        ), name
        assert other.stride() == value.stride(), name
    assert synthetic.routing_input_mode is RoutingInputMode.FromLogits
    # The bias is a model parameter: the same tensor, not a copy.
    assert synthetic.routing_bias is caller.routing_bias
    assert synthetic.routing_logits is not caller.routing_logits
    assert synthetic.hidden_states_q.data_ptr() != caller.hidden_states_q.data_ptr()


@pytest.mark.parametrize("tokens", [1, 2, 6, 10, 64])
def test_selection_pack_spreads_degenerate_routing(tokens):
    caller = _degenerate(tokens)
    assert _selected(caller).unique().numel() == TOP_K  # one shared expert set

    synthetic = selection.representative_pack(caller, GLM_ROUTING, QuantFormat.BF16)
    distinct = _selected(synthetic).unique().numel()
    # iid logits (+ the real bias): as many distinct experts as a load-balanced
    # router, e.g. ~44 of 48 slots at 6 tokens.
    assert distinct >= 0.75 * min(tokens * TOP_K, NUM_EXPERTS)
    hidden = synthetic.hidden_states_q.float()
    assert hidden.abs().max() > 0
    assert abs(hidden.std().item() - 1.0) < 0.35


def test_selection_pack_is_independent_of_caller_values_and_rng():
    rng_state = torch.random.get_rng_state()
    from_warmup = selection.representative_pack(
        _degenerate(6), GLM_ROUTING, QuantFormat.BF16
    )
    from_served = selection.representative_pack(
        _served(6), GLM_ROUTING, QuantFormat.BF16
    )
    again = selection.representative_pack(_degenerate(6), GLM_ROUTING, QuantFormat.BF16)
    # The private generator leaves the caller's global RNG stream alone.
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    # Only the bias object differs between the two callers' packs.
    from_served.routing_bias = from_warmup.routing_bias
    _assert_same_values(from_warmup, from_served)
    _assert_same_values(from_warmup, again)


_DIGEST_SCRIPT = """
import hashlib, torch
from flashinfer.fused_moe import MoEActivationPack, QuantFormat, RoutingConfig, RoutingInputMode
from flashinfer.fused_moe.selection import representative_pack
from flashinfer.tllm_enums import RoutingMethodType
routing = RoutingConfig(num_experts=256, top_k=8, method=RoutingMethodType.DeepSeekV3,
                        n_group=1, topk_group=1, routed_scaling_factor=2.5)
pack = MoEActivationPack(hidden_states_q=torch.zeros(6, 64, dtype=torch.bfloat16),
                         hidden_states_scale=None, routing_input_mode=RoutingInputMode.FromLogits,
                         routing_logits=torch.zeros(6, 256), routing_bias=torch.linspace(-0.05, 0.05, 256))
synthetic = representative_pack(pack, routing, QuantFormat.BF16)
digest = hashlib.sha256()
for tensor in (synthetic.hidden_states_q, synthetic.routing_logits):
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
print(digest.hexdigest())
"""


def test_selection_pack_is_identical_across_processes():
    synthetic = selection.representative_pack(
        _degenerate(6), GLM_ROUTING, QuantFormat.BF16
    )
    digest = hashlib.sha256()
    for tensor in (synthetic.hidden_states_q, synthetic.routing_logits):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=600,
    )
    assert out.stdout.strip().splitlines()[-1] == digest.hexdigest()


@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
@pytest.mark.parametrize(
    "routing", [GLM_ROUTING, GROUPED_ROUTING], ids=["one_group", "grouped"]
)
def test_precomputed_selection_routing_is_distinct_and_grouped(mode, routing):
    tokens = 512
    weights_dtype = (
        torch.bfloat16
        if mode is RoutingInputMode.UnpackedPrecomputed
        else torch.float32
    )
    caller = MoEActivationPack(
        torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16),
        None,
        torch.arange(TOP_K, dtype=torch.int32).expand(tokens, TOP_K).contiguous(),
        torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=weights_dtype),
        routing_input_mode=mode,
    )
    synthetic = selection.representative_pack(caller, routing, QuantFormat.BF16)
    ids = synthetic.topk_ids.long()
    assert synthetic.topk_ids.dtype == torch.int32
    assert synthetic.topk_weights.dtype == weights_dtype
    assert ids.min() >= 0 and ids.max() < NUM_EXPERTS
    assert all(len(set(row.tolist())) == TOP_K for row in ids)
    assert ids.unique().numel() == NUM_EXPERTS
    weights = synthetic.topk_weights.float()
    assert (weights > 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(tokens), atol=1e-2, rtol=0)
    if routing is GROUPED_ROUTING:
        groups = ids // (NUM_EXPERTS // routing.n_group)
        assert max(len(set(row.tolist())) for row in groups) <= routing.topk_group


def test_selection_pack_scales_are_units_in_the_activation_encoding():
    tokens = 4
    routing = RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K)
    ids = torch.zeros(tokens, TOP_K, dtype=torch.int32)
    weights = torch.ones(tokens, TOP_K)

    nvfp4 = MoEActivationPack(
        torch.zeros(tokens, HIDDEN // 2, dtype=torch.uint8),
        torch.zeros(tokens, HIDDEN // 16, dtype=torch.uint8).view(torch.float8_e4m3fn),
        ids,
        weights,
        per_token_scale=torch.zeros(tokens),
    )
    synthetic = selection.representative_pack(nvfp4, routing, QuantFormat.NVFP4)
    assert synthetic.hidden_states_scale.float().unique().tolist() == [1.0]
    assert synthetic.per_token_scale.unique().tolist() == [1.0]
    # Two E2M1 codes per byte, every code equally likely (no NaN encoding).
    assert synthetic.hidden_states_q.unique().numel() > 64

    mxfp8 = MoEActivationPack(
        torch.zeros(tokens, HIDDEN, dtype=torch.float8_e4m3fn),
        torch.zeros(tokens, HIDDEN // 32, dtype=torch.uint8),
        ids,
        weights,
    )
    synthetic = selection.representative_pack(mxfp8, routing, QuantFormat.MXFP8)
    assert synthetic.hidden_states_scale.unique().tolist() == [127]  # UE8M0 1.0
    assert torch.isfinite(synthetic.hidden_states_q.float()).all()

    deepseek = MoEActivationPack(
        torch.zeros(tokens, 256, dtype=torch.float8_e4m3fn),
        torch.zeros(256 // 128, tokens),
        ids,
        weights,
    )
    synthetic = selection.representative_pack(
        deepseek, routing, QuantFormat.DeepSeekFp8
    )
    assert synthetic.hidden_states_scale.unique().tolist() == [1.0]


def test_expert_load_steers_the_drawn_experts():
    tokens = 2048
    load = torch.zeros(NUM_EXPERTS)
    load[:32] = torch.arange(1, 33, dtype=torch.float32)
    # FromLogits: the drawn experts must survive the router, bias included.
    synthetic = selection.representative_pack(
        _degenerate(tokens), GLM_ROUTING, QuantFormat.BF16, expert_load=load
    )
    selected = _selected(synthetic)
    assert selected.max() < 32  # zero-load experts are never selected
    counts = torch.bincount(selected.flatten(), minlength=32)[:32].float()
    assert torch.corrcoef(torch.stack([counts, load[:32]]))[0, 1] > 0.9

    caller = MoEActivationPack(
        torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16),
        None,
        torch.zeros(tokens, TOP_K, dtype=torch.int32),
        torch.ones(tokens, TOP_K),
    )
    grouped = selection.representative_pack(
        caller, GROUPED_ROUTING, QuantFormat.BF16, expert_load=load
    )
    assert grouped.topk_ids.max() < 32


@pytest.mark.parametrize(
    "load,match",
    [
        (torch.ones(NUM_EXPERTS - 1), "shape"),
        (-torch.ones(NUM_EXPERTS), "non-negative"),
        (torch.full((NUM_EXPERTS,), float("nan")), "finite"),
        (
            torch.cat([torch.ones(TOP_K - 1), torch.zeros(NUM_EXPERTS - TOP_K + 1)]),
            "top_k",
        ),
    ],
)
def test_expert_load_is_validated(load, match):
    with pytest.raises(ValueError, match=match):
        selection.validate_expert_load(load, GLM_ROUTING)


# ---------------------------------------------------------------------------
# MoELayer contract, with fake runners
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Times itself by the routing it is packed with, like a real kernel:
    ``shared_experts`` is fast while tokens share experts (warm-up routing),
    ``spread`` costs the same whatever the routing."""

    supported_routing_modes = (RoutingInputMode.FromLogits,)

    def __init__(self, backend_key, fixed, per_expert):
        self.backend_key = backend_key
        self.fixed, self.per_expert = fixed, per_expert
        self.forwarded = []

    def accepts(self, act, weights):
        return True

    def pack_inputs(self, act, weights):
        return [act]

    def launch_kwargs_for(self, inputs):
        return {}

    def tuning_config_for(self, inputs):
        return TuningConfig()

    def forward(self, inputs, *, tactic=-1, **kwargs):
        act = inputs[0]
        self.forwarded.append(act)
        distinct = _selected(act).unique().numel()
        _FakeRunner.cost = self.fixed + self.per_expert * distinct
        return act.hidden_states_q


def _fake_bench(fn, **kwargs):
    fn()
    return [_FakeRunner.cost]


def _layer(monkeypatch, *runners):
    from flashinfer.testing import utils as testing_utils

    monkeypatch.setattr(testing_utils, "bench_gpu_time", _fake_bench)
    layer = MoELayer.__new__(MoELayer)
    layer.config = MoEConfig(
        routing=GLM_ROUTING,
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=256),
        execution=ExecutionConfig(tune_max_num_tokens=64),
    )
    layer.device = torch.device("cpu")
    layer.tuner = type("Tuner", (), {"choose_one": staticmethod(_first_tactic)})()
    layer.runners = list(runners) or [
        _FakeRunner("shared_experts", fixed=10.0, per_expert=4.0),
        _FakeRunner("spread", fixed=60.0, per_expert=0.0),
    ]
    layer._winners = OrderedDict()
    layer._last_winner_backend = None
    return layer


def _first_tactic(*, runners, **kwargs):
    return runners[0], -1


WEIGHTS = MoEWeightPack()


def test_warmup_routing_selects_the_served_winner(monkeypatch):
    monkeypatch.delenv(selection.SELECT_ON_CALL_INPUTS_ENV, raising=False)
    # On the warm-up batch itself shared_experts wins (10 + 4 * 8 < 60) ...
    warm = _layer(monkeypatch)
    out = warm(_degenerate(6), WEIGHTS)
    # ... but the candidates were timed on serving-like routing, as for a first
    # call on served traffic (10 + 4 * ~44 > 60).
    served = _layer(monkeypatch)
    served(_served(6), WEIGHTS)
    assert warm.winner_backend == served.winner_backend == "spread"
    # The call itself computes on its own tensors.
    assert out is warm.runners[1].forwarded[-1].hidden_states_q
    assert warm.runners[1].forwarded[-1].routing_logits.abs().max() == 0


def test_select_on_call_inputs_env_restores_first_call_selection(monkeypatch):
    monkeypatch.setenv(selection.SELECT_ON_CALL_INPUTS_ENV, "1")
    warm = _layer(monkeypatch)
    warm(_degenerate(6), WEIGHTS)
    assert warm.winner_backend == "shared_experts"


def test_selection_is_deterministic_across_calls_and_layers(monkeypatch):
    monkeypatch.delenv(selection.SELECT_ON_CALL_INPUTS_ENV, raising=False)
    timed = []
    for _ in range(2):
        layer = _layer(monkeypatch)
        original = layer._select_winner

        def record(pack, weights, runners, _original=original):
            timed.append(pack)
            return _original(pack, weights, runners)

        monkeypatch.setattr(layer, "_select_winner", record)
        layer(_degenerate(6), WEIGHTS)
        layer.reset_winner()
        layer(_degenerate(6), WEIGHTS)
    assert len(timed) == 4
    for pack in timed[1:]:
        _assert_same_values(timed[0], pack)


def test_single_candidate_times_the_callers_pack(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("one candidate needs no synthetic pack")

    monkeypatch.setattr(layer_module, "representative_pack", fail)
    layer = _layer(monkeypatch, _FakeRunner("only", fixed=1.0, per_expert=0.0))
    caller = _degenerate(6)
    seen = []
    original = layer._select_winner
    monkeypatch.setattr(
        layer,
        "_select_winner",
        lambda pack, weights, runners: seen.append(pack)
        or original(pack, weights, runners),
    )
    layer(caller, WEIGHTS)
    assert seen == [caller]


def test_nothing_is_selected_or_synthesized_inside_capture(monkeypatch):
    monkeypatch.delenv(selection.SELECT_ON_CALL_INPUTS_ENV, raising=False)
    layer = _layer(monkeypatch)
    layer(_degenerate(6), WEIGHTS)  # eager: selects the 6-token bucket
    assert layer.winner_backend == "spread"

    selected = []
    original = layer._select_winner
    monkeypatch.setattr(
        layer,
        "_select_winner",
        lambda pack, weights, runners: selected.append(pack)
        or original(pack, weights, runners),
    )
    synthesized = []
    monkeypatch.setattr(
        layer_module,
        "representative_pack",
        lambda *args, **kwargs: synthesized.append(args),
    )
    monkeypatch.setattr(layer_module, "is_capturing", lambda act: True)

    # A captured call of a selected bucket replays the cached winner.
    layer(_degenerate(6), WEIGHTS)
    assert selected == [] and layer.winner_backend == "spread"
    # A first call under capture generates nothing into the caller's graph and
    # keeps selecting on its own tensors (real candidates cannot be timed
    # inside a capture, so a framework warms every captured size up first).
    caller = _degenerate(12)
    layer(caller, WEIGHTS)
    assert selected == [caller] and synthesized == []
    # The explicit hook refuses to run at all.
    with pytest.raises(RuntimeError, match="capture"):
        layer.warmup(_degenerate(24), WEIGHTS)
    assert selected == [caller]


def test_warmup_selects_ahead_and_replaces_the_cached_winner(monkeypatch):
    monkeypatch.delenv(selection.SELECT_ON_CALL_INPUTS_ENV, raising=False)
    layer = _layer(monkeypatch)
    assert layer.warmup(_degenerate(6), WEIGHTS) == "spread"
    assert len(layer._winners) == 1

    # A recorded load concentrated on eight experts makes tokens share them,
    # which is where shared_experts wins; the hook replaces the cached winner.
    load = torch.zeros(NUM_EXPERTS)
    load[:TOP_K] = 1.0
    assert layer.warmup(_degenerate(6), WEIGHTS, expert_load=load) == "shared_experts"
    assert len(layer._winners) == 1
    layer(_served(6), WEIGHTS)
    assert layer.winner_backend == "shared_experts"

    # use_pack_values times the caller's own values (here, served logits).
    assert layer.warmup(_served(6), WEIGHTS, use_pack_values=True) == "spread"
    assert layer.warmup(_degenerate(6), WEIGHTS, use_pack_values=True) == (
        "shared_experts"
    )

    with pytest.raises(ValueError, match="not both"):
        layer.warmup(_degenerate(6), WEIGHTS, expert_load=load, use_pack_values=True)
    with pytest.raises(ValueError, match="shape"):
        layer.warmup(_degenerate(6), WEIGHTS, expert_load=load[:-1])
