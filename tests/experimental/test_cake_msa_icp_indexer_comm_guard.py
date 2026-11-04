"""The decode path validates a comm once and re-validates it when a field changes (CPU only).

``MSAIcpIndexerPlan.decode`` runs ``check_comm`` on the first use of a comm, then compares each comm field with the
tensor ``check_comm`` validated (object, address, shape, strides, device).  A swapped, re-allocated, resized or
re-strided field must re-run ``check_comm`` (and drop the slot views made from the old field) before any launch.
"""

from __future__ import annotations

import importlib
import os
import types

import pytest
import torch

PACKAGE = os.environ.get(
    "CAKE_MSA_ICP_PACKAGE", "flashinfer.experimental.cake_msa_icp_indexer"
)
cake_indexer = importlib.import_module(f"{PACKAGE}.cake_indexer")


def _comm() -> types.SimpleNamespace:
    storage = torch.zeros(4096, dtype=torch.int32)
    return types.SimpleNamespace(
        recv=storage[:1024].view(2, 2, 256),
        tagged_recv=storage[1024:3072].view(torch.int64).view(2, 2, 256),
        recv_bases=torch.zeros(2, dtype=torch.int64),
        tagged_recv_bases=torch.zeros(2, dtype=torch.int64),
        publish_generations=torch.zeros(3, 4, 8, dtype=torch.int32),
        expected_generations=torch.zeros(3, 2, 8, dtype=torch.int32),
        errors=torch.zeros(1, dtype=torch.int32),
    )


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch):
    """A plan without a device: ``check_comm`` is replaced by a counter that records the comm as validated."""
    p = object.__new__(cake_indexer.MSAIcpIndexerPlan)
    p._checked_comms = {}
    p._slot_views = {}
    p.validated = []

    def check_comm(comm):
        p.validated.append(comm)
        p._remember_comm(comm)

    monkeypatch.setattr(p, "check_comm", check_comm)
    return p


def test_a_comm_is_validated_once_while_unchanged(plan) -> None:
    comm = _comm()
    for _ in range(3):
        plan._ensure_comm(comm)
    assert len(plan.validated) == 1


@pytest.mark.parametrize(
    "change",
    [
        "swap recv_bases",
        "swap expected_generations",
        "swap recv",
        "reallocate errors in place",
        "resize publish_generations in place",
        "restride recv in place",
    ],
)
def test_a_changed_field_is_validated_again(plan, change: str) -> None:
    comm = _comm()
    plan._ensure_comm(comm)
    plan._slot_views[(id(comm), 0, True)] = (
        comm.tagged_recv[0],
        comm.expected_generations[0],
    )
    if change.startswith("swap"):
        field = change.split()[1]
        setattr(comm, field, getattr(comm, field).clone())
    elif change == "reallocate errors in place":
        comm.errors.set_(torch.zeros(1, dtype=torch.int32).untyped_storage())
    elif change == "resize publish_generations in place":
        comm.publish_generations.resize_(3, 4, 16)
    else:
        comm.recv.as_strided_((2, 2, 256), (256, 512, 1))
    plan._ensure_comm(comm)
    assert len(plan.validated) == 2
    assert (id(comm), 0, True) not in plan._slot_views
    plan._ensure_comm(comm)
    assert len(plan.validated) == 2


def test_another_comm_object_at_the_same_id_is_validated(plan) -> None:
    comm = _comm()
    plan._ensure_comm(comm)
    plan._checked_comms[id(comm)] = (object(), plan._checked_comms[id(comm)][1])
    plan._ensure_comm(comm)
    assert len(plan.validated) == 2
