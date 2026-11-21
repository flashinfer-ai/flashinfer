"""CPU tests for building a shard of cuDNN plans, with a fake cuDNN graph."""

import sys
import types

import pytest

import flashinfer.gemm._cudnn_plan_build as pb


class FakeGraph:
    def __init__(self, engines, bad_lookup=(), bad_serialize=()):
        self.engines = engines
        self.bad_lookup = set(bad_lookup)
        self.bad_serialize = set(bad_serialize)
        self.selected = None
        self.built = set()

    def validate(self):
        pass

    def build_operation_graph(self):
        pass

    def create_execution_plans(self, heur_modes):
        pass

    def check_support(self):
        pass

    def get_execution_plan_count(self):
        return len(self.engines)

    def get_engine_and_knobs_at_index(self, index):
        if index in self.bad_lookup:
            raise RuntimeError("no engine config")
        return self.engines[index], {}

    def select_plan(self, index):
        self.selected = index

    def build_plan_at_index(self, index):
        self.built.add(index)

    def serialize(self, as_bytes):
        if self.selected in self.bad_serialize:
            raise RuntimeError("cannot serialize")
        assert self.selected in self.built
        return f"plan{self.selected}".encode()


@pytest.fixture
def use_graph(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", types.SimpleNamespace(heur_mode=int))

    def use(graph):
        monkeypatch.setattr(pb, "build_bf16_override_graph", lambda spec, h: graph)
        return graph

    return use


def test_skipped_lookup_keeps_each_tactic_on_its_own_plan(use_graph):
    # cuDNN plan 1 has no readable engine config, so candidates[1] is plan 2:
    # building by list position would export plan 1 under plan 2's tactic.
    use_graph(FakeGraph([10, 11, 12], bad_lookup={1}))
    candidates, plans, build_here = pb.build_shard(None, None, [0], (0, 1))
    assert candidates == [(10, ()), (12, ())]
    assert plans == [(0, (10, ()), b"plan0"), (1, (12, ()), b"plan2")]
    assert build_here == []


def test_plan_that_cannot_be_serialized_is_built_in_process(use_graph):
    # It built, so in-process it would be usable: it must not be reported as
    # an unbuildable (failed) candidate.
    use_graph(FakeGraph([10, 11, 12], bad_serialize={0}))
    candidates, plans, build_here = pb.build_shard(None, None, [0], (0, 1))
    assert candidates == [(10, ()), (11, ()), (12, ())]
    assert [index for index, _, _ in plans] == [1, 2]
    assert build_here == [0]
