"""Host-only admission of the dense MQA family through ``flashinfer.dense_mqa`` (the engine-facing surface).

The engine (sglang ``supports_fp8_mqa_logits``) decides per request whether to take the Cake route or keep
stock DeepGEMM by asking ``dense_route_available``; the producer publishes the 64-head family per tier
(``policy.dense_admission``): admitted tiers have catalog records, withheld tiers have none.  These tests pin
that contract against the shipped catalog itself, so they hold for every export (all admitted, some
withheld, or the 64-head family not exported at all).  No device, no JIT.
"""

import copy
import functools

import pytest

from flashinfer import dense_mqa
from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as _runtime

ADMITTED_FIXTURE = ("fp8:h64:partial:any", "fp8:h64:full:le64")
REASON_FIXTURE = "not converged, a follow-up issue sub-issue"


def _h64_catalog(shipped):
    """The shipped catalog plus a 64-head family with two admitted tiers and the rest withheld (the shape of
    the export's ``policy.dense_admission``); the records are table-only, no program is ever loaded."""
    catalog = copy.deepcopy(shipped)
    policy = catalog["policy"]
    policy["heads"] = sorted({*(int(h) for h in policy["heads"]), 64})
    policy["kv_alignment"] = {**policy["kv_alignment"], "64": 1}
    template = next(iter(shipped["routes"].values()))
    for route in ADMITTED_FIXTURE:
        catalog["routes"][route] = {
            **template,
            "num_heads": 64,
            "block_q": 2,
            "kv_alignment": 1,
            "clean_logits": None,
            "stages": [["logits", "cake_deepgemm_dense_mqa_fixture"]],
        }
    withheld = sorted(H64_ROUTES - set(ADMITTED_FIXTURE))
    policy["dense_admission"] = dict(
        admitted_routes=sorted(ADMITTED_FIXTURE), withheld_routes=withheld, reason=REASON_FIXTURE
    )
    return catalog


@pytest.fixture(params=["shipped", "h64_fixture"])
def catalog_variant(request, monkeypatch):
    """Run every test against the shipped catalog and against the synthetic partially admitted 64-head one."""
    if request.param == "h64_fixture":
        synthetic = _h64_catalog(_runtime._catalog())
        monkeypatch.setattr(_runtime, "_catalog", functools.cache(lambda: synthetic))
    return request.param

H64_TIERS = ("le8", "le64", "le1024", "any")
H64_ROUTES = frozenset(
    {"fp8:h64:q1", *(f"fp8:h64:{kind}:{tier}" for kind in ("full", "partial") for tier in H64_TIERS)}
)
# One query count per 64-head route: Q = 1 (q1); 2 / 3 (le8 = 1-4 blocks of 2); 16 / 37 (le64); 128 / 1023
# (le1024); 4096 / 16231 (any).  K = 4137 is a legal 64-head KV length (any K >= 1).
QUERIES_BY_ROUTE = {
    "fp8:h64:q1": (1,),
    "fp8:h64:full:le8": (2, 8),
    "fp8:h64:partial:le8": (3, 7),
    "fp8:h64:full:le64": (16, 64),
    "fp8:h64:partial:le64": (37, 65 - 2),
    "fp8:h64:full:le1024": (128, 1024),
    "fp8:h64:partial:le1024": (129, 1023),
    "fp8:h64:full:any": (4096, 16232),
    "fp8:h64:partial:any": (1025, 16231),
}


def test_dense_admission_is_disjoint_and_complete(catalog_variant):
    """admitted and withheld are disjoint 64-head route names; a reason accompanies a non-empty withheld
    list; when the 64-head family is exported, every tier name is either admitted or withheld."""
    admission = dense_mqa.dense_admission()
    admitted, withheld = set(admission["admitted_routes"]), set(admission["withheld_routes"])
    assert admitted <= H64_ROUTES and withheld <= H64_ROUTES
    assert not admitted & withheld
    assert bool(withheld) == (admission["reason"] is not None)
    if 64 in _runtime.heads():
        assert admitted | withheld == H64_ROUTES
    else:
        assert not admitted and not withheld
    if catalog_variant == "h64_fixture":
        assert admitted == set(ADMITTED_FIXTURE) and admission["reason"] == REASON_FIXTURE


@pytest.mark.parametrize("route", sorted(H64_ROUTES))
def test_dense_route_available_follows_the_admitted_tiers(catalog_variant, route):
    """Every query count of an admitted tier is available, every query count of a withheld (or unexported)
    tier is not; the engine-facing wrapper agrees with the backend's table."""
    admission = dense_mqa.dense_admission()
    admitted = route in admission["admitted_routes"]
    for queries in QUERIES_BY_ROUTE[route]:
        if 64 in _runtime.heads():
            assert _runtime.route_name("fp8", queries, 4137, 64) == route, queries
        available = dense_mqa.dense_route_available(64, queries, 4137)
        assert available == admitted, (route, queries)
        assert available == _runtime.dense_route_available(64, queries, 4137)


def test_dense_route_available_rejects_out_of_contract_points(catalog_variant):
    """Points no catalog can serve are rejected without raising: zero queries, zero keys, foreign head counts,
    an unaligned 32-head KV length."""
    assert not dense_mqa.dense_route_available(64, 0, 4137)
    assert not dense_mqa.dense_route_available(64, 37, 0)
    assert not dense_mqa.dense_route_available(16, 16, 4096)
    assert not dense_mqa.dense_route_available(32, 16, 4100)
    assert dense_mqa.dense_route_available(32, 16, 4096)


def test_dense_admission_without_a_published_set_is_the_routes_table(monkeypatch):
    """A 64-head catalog whose policy predates ``admitted_routes`` admits exactly the ``fp8:h64:*`` routes it
    ships (the engine admission and the table agree by construction)."""
    synthetic = _h64_catalog(_runtime._catalog())
    del synthetic["policy"]["dense_admission"]["admitted_routes"]
    monkeypatch.setattr(_runtime, "_catalog", functools.cache(lambda: synthetic))
    admission = dense_mqa.dense_admission()
    assert admission["admitted_routes"] == sorted(ADMITTED_FIXTURE)
    assert admission["reason"] == REASON_FIXTURE
    assert dense_mqa.dense_route_available(64, 16231, 4137) and not dense_mqa.dense_route_available(64, 37, 4137)
