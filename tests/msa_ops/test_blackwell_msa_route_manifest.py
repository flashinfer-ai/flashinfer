"""CPU-only coverage contract for exported Blackwell MSA routes."""

import json
from collections import Counter
from pathlib import Path


_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "blackwell_msa"
    / "route_manifest.json"
)


def _manifest():
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_route_manifest_covers_exactly_the_production_specializations():
    manifest = _manifest()
    routes = manifest["reachable_specializations"]

    assert manifest["schema_version"] == 1
    assert manifest["attention_topk"] == 16
    assert manifest["reachable_specialization_count"] == 32
    assert len(routes) == 32
    assert len({route["id"] for route in routes}) == 32
    assert Counter(route["family"] for route in routes) == {
        "topk_select": 1,
        "union_prefill": 12,
        "m64_prefill_override": 1,
        "long_bf16_reverse_prefill": 7,
        "direct_m16_decode": 6,
        "fp8_kv_q1_exact_override": 2,
        "fp8_kv_q1_xform2": 2,
        "uniform_fp8_qkv_decode": 1,
    }
    assert set(manifest["family_predicates"]) == {route["family"] for route in routes}
    assert all(manifest["family_predicates"].values())
    assert sum(len(route["architectures"]) for route in routes) == 57
    assert all(route["source_units"] for route in routes)


def test_route_manifest_explicitly_excludes_unreachable_helpers():
    manifest = _manifest()
    excluded = manifest["excluded_unreachable_helpers"]
    routes = manifest["reachable_specializations"]

    assert {item["id"] for item in excluded} == {
        "reverse_prefill_qagg_pdl",
        "reverse_prefill_j21",
        "reverse_prefill_atomic",
        "paged_bf16_qload4_exact_graph",
    }
    assert {item["reason"] for item in excluded} == {
        "no production dispatcher callsite"
    }
    exported_ids = {route["id"] for route in routes}
    exported_units = {
        source_unit for route in routes for source_unit in route["source_units"]
    }
    excluded_ids = {item["id"] for item in excluded}
    assert exported_ids.isdisjoint(excluded_ids)
    assert exported_units.isdisjoint(excluded_ids)


def test_route_manifest_contains_no_private_provenance_fields():
    manifest = _manifest()

    assert "source_revision" not in manifest
    assert "source_repository" not in manifest
    assert "change_request" not in manifest
