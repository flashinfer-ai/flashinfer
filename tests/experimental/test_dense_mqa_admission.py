"""Host-only admission of the dense MQA family through ``flashinfer.dense_mqa`` (the engine-facing surface).

The engine (sglang ``supports_fp8_mqa_logits``) decides per request whether to take the Cake route or keep
stock DeepGEMM by asking ``dense_route_available(H, Q, K, arch=device_arch)``; the producer publishes the
64-head family per tier and per architecture (``policy.dense_admission``): an admitted tier is served on that
arch, a withheld tier is not (its record may exist for another arch).  These tests pin that contract against
the shipped catalog itself and against a synthetic catalog whose architectures disagree.  No device, no JIT.
"""

import copy
import functools

import pytest

from flashinfer import dense_mqa
from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as _runtime

H64_TIERS = ("le8", "le64", "le1024", "any")
H64_ROUTES = frozenset(
    {
        "fp8:h64:q1",
        *(
            f"fp8:h64:{kind}:{tier}"
            for kind in ("full", "partial")
            for tier in H64_TIERS
        ),
    }
)
# One or two query counts per 64-head route (K = 4137 is a legal 64-head KV length: any K >= 1).
QUERIES_BY_ROUTE = {
    "fp8:h64:q1": (1,),
    "fp8:h64:full:le8": (2, 8),
    "fp8:h64:partial:le8": (3, 7),
    "fp8:h64:full:le64": (16, 64),
    "fp8:h64:partial:le64": (37, 63),
    "fp8:h64:full:le1024": (128, 1024),
    "fp8:h64:partial:le1024": (129, 1023),
    "fp8:h64:full:any": (4096, 16232),
    "fp8:h64:partial:any": (1025, 16231),
}
# Synthetic per-arch admission: sm_100a admits nothing, sm_103a admits two tiers (the shape of the export's
# ``policy.dense_admission``); one tier is therefore admitted on some architectures only.
ADMITTED_FIXTURE = {
    "sm_100a": (),
    "sm_103a": ("fp8:h64:partial:any", "fp8:h64:full:le64"),
}
REASON_FIXTURE = {
    "sm_100a": "not converged: partial:any blocked by one row",
    "sm_103a": "not converged",
}


def _h64_catalog(shipped):
    """The shipped catalog plus a 64-head family with per-arch admission; the records are table-only."""
    catalog = copy.deepcopy(shipped)
    catalog["arches"] = sorted({*catalog["arches"], *ADMITTED_FIXTURE})
    policy = catalog["policy"]
    policy["heads"] = sorted({*(int(h) for h in policy["heads"]), 64})
    policy["kv_alignment"] = {**policy["kv_alignment"], "64": 1}
    template = next(iter(shipped["routes"].values()))
    published = {route for routes in ADMITTED_FIXTURE.values() for route in routes}
    for route in published:
        catalog["routes"][route] = {
            **template,
            "num_heads": 64,
            "block_q": 2,
            "kv_alignment": 1,
            "clean_logits": "raw",
            "stages": [["logits", "cake_deepgemm_dense_mqa_fixture"]],
        }
    policy["dense_admission"] = dict(
        admitted_routes={
            arch: sorted(routes) for arch, routes in ADMITTED_FIXTURE.items()
        },
        withheld_routes={
            arch: sorted(H64_ROUTES - set(routes))
            for arch, routes in ADMITTED_FIXTURE.items()
        },
        reason=dict(REASON_FIXTURE),
    )
    return catalog


@pytest.fixture(params=["shipped", "h64_fixture"])
def catalog_variant(request, monkeypatch):
    """Run every test against the shipped catalog and against the synthetic per-arch 64-head one."""
    if request.param == "h64_fixture":
        synthetic = _h64_catalog(_runtime._catalog())
        monkeypatch.setattr(_runtime, "_catalog", functools.cache(lambda: synthetic))
    return request.param


def _arches():
    return sorted(_runtime._catalog()["arches"])


def test_dense_admission_is_per_arch_disjoint_and_complete(catalog_variant):
    """``dense_admission()`` is keyed by every catalogued arch; per arch the admitted and withheld lists are
    disjoint 64-head tier names, a reason accompanies a non-empty withheld list, admitted routes have catalog
    records, and (when the family is exported) the two lists together name every tier."""
    everything = dense_mqa.dense_admission()
    assert sorted(everything) == _arches()
    for arch in _arches():
        admission = dense_mqa.dense_admission(arch)
        assert admission == everything[arch]
        admitted, withheld = (
            set(admission["admitted_routes"]),
            set(admission["withheld_routes"]),
        )
        assert (
            admitted <= H64_ROUTES
            and withheld <= H64_ROUTES
            and not admitted & withheld
        )
        assert admitted <= set(_runtime._catalog()["routes"])
        assert bool(withheld) == (admission["reason"] is not None)
        if 64 in _runtime.heads():
            assert admitted | withheld == H64_ROUTES
        else:
            assert not admitted and not withheld
    if catalog_variant == "h64_fixture":
        assert (
            everything["sm_100a"]["admitted_routes"] == []
            and everything["sm_100a"]["reason"] == REASON_FIXTURE["sm_100a"]
        )
        assert everything["sm_103a"]["admitted_routes"] == sorted(
            ADMITTED_FIXTURE["sm_103a"]
        )


@pytest.mark.parametrize("route", sorted(H64_ROUTES))
def test_dense_route_available_follows_the_per_arch_admission(catalog_variant, route):
    """On every arch, every query count of an admitted tier is available and every query count of a withheld
    (or unexported) tier is not; the engine-facing wrapper agrees with the backend; without ``arch`` the call
    answers only where the architectures agree and raises otherwise."""
    verdicts = {}
    for arch in _arches():
        admitted = route in dense_mqa.dense_admission(arch)["admitted_routes"]
        verdicts[arch] = admitted
        for queries in QUERIES_BY_ROUTE[route]:
            if 64 in _runtime.heads():
                assert _runtime.route_name("fp8", queries, 4137, 64) == route, queries
            available = dense_mqa.dense_route_available(64, queries, 4137, arch=arch)
            assert available == admitted, (arch, route, queries)
            assert available == _runtime.dense_route_available(
                64, queries, 4137, arch=arch
            )
    queries = QUERIES_BY_ROUTE[route][0]
    if len(set(verdicts.values())) == 1:
        assert dense_mqa.dense_route_available(64, queries, 4137) == next(
            iter(verdicts.values())
        )
    else:
        with pytest.raises(ValueError, match="pass arch"):
            dense_mqa.dense_route_available(64, queries, 4137)
    with pytest.raises(ValueError, match="arch must be one of"):
        dense_mqa.dense_admission("sm_999x")


def test_dense_route_available_rejects_out_of_contract_points(catalog_variant):
    """Points no catalog can serve are rejected without raising on every arch: zero queries, zero keys,
    foreign head counts, an unaligned 32-head KV length; 32-head routes need no arch."""
    for arch in (None, *_arches()):
        assert not dense_mqa.dense_route_available(64, 0, 4137, arch=arch)
        assert not dense_mqa.dense_route_available(64, 37, 0, arch=arch)
        assert not dense_mqa.dense_route_available(16, 16, 4096, arch=arch)
        assert not dense_mqa.dense_route_available(32, 16, 4100, arch=arch)
        assert dense_mqa.dense_route_available(32, 16, 4096, arch=arch)


def test_plan_refuses_a_withheld_route_by_name(monkeypatch):
    """``route_record`` (the plan's gate) names the withheld arch instead of a generic missing-route error."""
    synthetic = _h64_catalog(_runtime._catalog())
    monkeypatch.setattr(_runtime, "_catalog", functools.cache(lambda: synthetic))
    assert (
        _runtime.route_record("fp8", 16231, 4137, 64, arch="sm_103a")["num_heads"] == 64
    )
    with pytest.raises(ValueError, match="sm_100a"):
        _runtime.route_record("fp8", 16231, 4137, 64, arch="sm_100a")
