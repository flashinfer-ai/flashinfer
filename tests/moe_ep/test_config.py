"""B2 — dataclass + AlgoKnob unit tests (no CUDA, no comms)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    EpAlgorithm,
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobQuantization,
    FleetParams,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobUserStream,
    QuantType,
)
from flashinfer.moe_ep.algo_knobs import _index_knobs


class TestFleetParams:
    def test_happy_path(self) -> None:
        p = FleetParams(
            num_experts=8,
            max_tokens_per_rank=128,
            token_hidden_size=4096,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
        )
        assert p.num_experts == 8
        assert p.algorithm is EpAlgorithm.LOW_LATENCY

    @pytest.mark.parametrize(
        "field,value",
        [
            ("num_experts", 0),
            ("num_experts", -1),
            ("max_tokens_per_rank", 0),
            ("token_hidden_size", 0),
            ("dtype_bytes", 0),
        ],
    )
    def test_validation_rejects_nonpositive(self, field: str, value: int) -> None:
        kwargs = dict(num_experts=8, max_tokens_per_rank=128, token_hidden_size=4096)
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            FleetParams(**kwargs)

    def test_replace_round_trip(self) -> None:
        p1 = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=4096)
        p2 = replace(p1, num_experts=16)
        assert p1.num_experts == 8 and p2.num_experts == 16
        assert p2.max_tokens_per_rank == 128


class TestBootstrapConfig:
    def test_rank_in_range(self) -> None:
        BootstrapConfig(world_size=8, rank=0)
        BootstrapConfig(world_size=8, rank=7)

    @pytest.mark.parametrize("rank", [-1, 8, 9])
    def test_rank_out_of_range(self, rank: int) -> None:
        with pytest.raises(ValueError, match=r"rank -?\d+ not in"):
            BootstrapConfig(world_size=8, rank=rank)

    def test_world_size_positive(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            BootstrapConfig(world_size=0, rank=0)


class TestAlgoKnobs:
    def test_index_one(self) -> None:
        idx = _index_knobs([HandleAlgoKnobUserStream(stream=42)])
        assert HandleAlgoKnobUserStream in idx
        knob = idx[HandleAlgoKnobUserStream]
        assert isinstance(knob, HandleAlgoKnobUserStream)
        assert knob.stream == 42

    def test_index_split_marker(self) -> None:
        # SplitOperation acts as a flag — presence in dict signals "set".
        idx = _index_knobs([HandleAlgoKnobSplitOperation()])
        assert HandleAlgoKnobSplitOperation in idx

    def test_index_quantization(self) -> None:
        idx = _index_knobs(
            [
                FleetAlgoKnobQuantization(
                    quants=frozenset({QuantType.FP8E4M3, QuantType.UE8M0})
                )
            ]
        )
        q = idx[FleetAlgoKnobQuantization]
        assert isinstance(q, FleetAlgoKnobQuantization)
        assert QuantType.FP8E4M3 in q.quants
        assert QuantType.UE8M0 in q.quants

    def test_later_wins(self) -> None:
        idx = _index_knobs(
            [
                FleetAlgoKnobNumChannelsPerRank(n=4),
                FleetAlgoKnobNumChannelsPerRank(n=8),
            ]
        )
        assert idx[FleetAlgoKnobNumChannelsPerRank].n == 8  # type: ignore[attr-defined]

    def test_reject_non_knob(self) -> None:
        with pytest.raises(TypeError, match="AlgoKnob"):
            _index_knobs(["not a knob"])  # type: ignore[list-item]
