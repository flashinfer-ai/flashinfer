# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Source catalog compatibility, shared metadata and export/selection tooling."""

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from flashinfer.fused_moe.backends.cudnn_frost.manifest import (
    MANIFEST_FILENAME,
    expand_manifest,
    normalize_manifest,
    read_manifest,
    write_manifest,
)
from flashinfer.fused_moe.backends.cudnn_frost.runtime import artifact_root
from flashinfer.fused_moe.backends.cudnn_frost.source_template import render_source


@pytest.fixture
def payload():
    original = read_manifest(artifact_root("bf16") / MANIFEST_FILENAME)
    first = deepcopy(original["kernels"][0])
    first["id"] = "first"
    first["contract"].update(n=256, k=128, experts=8, groups=8)
    second = deepcopy(first)
    second["id"] = "second"
    second["contract"]["n"] *= 2
    return {**original, "kernels": [first, second]}


@pytest.mark.cpu_only
def test_catalog_shares_implementations_without_merging_shape_contracts(payload):
    before = deepcopy(payload)
    catalog = normalize_manifest(payload)
    assert payload == before
    assert len(catalog["sources"]) == len(catalog["parameters"]) == 1
    assert len(catalog["implementations"]) == 1
    assert len(catalog["contracts"]) == 2
    assert expand_manifest(catalog) == before
    assert normalize_manifest(catalog) == catalog

    template = (
        artifact_root("bf16") / before["kernels"][0]["source"]["path"]
    ).read_text()
    for old, new in zip(
        before["kernels"], expand_manifest(catalog)["kernels"], strict=True
    ):
        assert render_source(template, old["source"]["parameters"]) == render_source(
            template, new["source"]["parameters"]
        )


@pytest.mark.cpu_only
def test_catalog_preserves_ordered_repeated_constants(payload):
    for record in payload["kernels"]:
        record["source"]["parameters"]["constants"] = [
            ["ab_stages", "10"],
            ["ab_stages", "6"],
        ]
    assert expand_manifest(normalize_manifest(payload)) == payload


@pytest.mark.cpu_only
def test_catalog_references_are_stable_when_other_kernels_are_added(payload):
    catalog = normalize_manifest(payload)
    third = deepcopy(payload["kernels"][0])
    third["id"] = "third"
    third["source"]["path"] = "sources/another.py"
    expanded = normalize_manifest({**payload, "kernels": [third, *payload["kernels"]]})
    assert expanded["kernels"][1:] == catalog["kernels"]
    for table in ("sources", "parameters", "implementations", "contracts"):
        assert catalog[table].items() <= expanded[table].items()


@pytest.mark.cpu_only
def test_expanded_kernel_metadata_does_not_alias_shared_tables(payload):
    catalog = normalize_manifest(payload)
    records = expand_manifest(catalog)["kernels"]
    records[0]["source"]["parameters"]["constants"][0][1] = "modified"
    records[0]["tactic"]["cta_tile"]["m"] = 1
    assert records[1]["source"] == payload["kernels"][1]["source"]
    assert records[1]["tactic"] == payload["kernels"][1]["tactic"]
    assert expand_manifest(catalog) == payload


@pytest.mark.cpu_only
def test_catalog_supports_sources_without_template_parameters(payload):
    for record in payload["kernels"]:
        del record["source"]["parameters"]
    catalog = normalize_manifest(payload)
    assert catalog["parameters"] == {}
    assert expand_manifest(catalog) == payload


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "table", ["sources", "parameters", "implementations", "contracts"]
)
def test_catalog_rejects_missing_reference_tables(payload, table):
    catalog = normalize_manifest(payload)
    del catalog[table]
    with pytest.raises(RuntimeError, match=f"'{table}' must be a table"):
        expand_manifest(catalog)


@pytest.mark.cpu_only
@pytest.mark.parametrize("reference", ["missing", None, 0, []])
@pytest.mark.parametrize(
    "table", ["sources", "parameters", "implementations", "contracts"]
)
def test_catalog_rejects_dangling_or_invalid_references(payload, table, reference):
    catalog = normalize_manifest(payload)
    entry = catalog["kernels"][0]
    if table in ("implementations", "contracts"):
        entry[
            "implementation_ref" if table == "implementations" else "contract_ref"
        ] = reference
    else:
        implementation = catalog["implementations"][entry["implementation_ref"]]
        implementation["source_ref" if table == "sources" else "parameters_ref"] = (
            reference
        )
    with pytest.raises(RuntimeError, match=f"unknown cuDNN Frost {table} reference"):
        expand_manifest(catalog)


@pytest.mark.cpu_only
@pytest.mark.parametrize("field", ["arch", "abi", "source", "contract"])
def test_catalog_entries_cannot_override_shared_implementation(payload, field):
    catalog = normalize_manifest(payload)
    catalog["kernels"][0][field] = "override"
    with pytest.raises(RuntimeError, match="invalid cuDNN Frost catalog kernel entry"):
        expand_manifest(catalog)


@pytest.mark.cpu_only
def test_manifest_writer_upgrades_v2_and_does_not_replace_on_invalid_input(
    payload, tmp_path
):
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps(payload))
    path.chmod(0o644)
    assert read_manifest(path) == payload
    write_manifest(path, read_manifest(path))
    assert path.stat().st_mode & 0o777 == 0o644
    assert json.loads(path.read_text())["schema_version"] == 3
    assert read_manifest(path) == payload
    before = path.read_bytes()
    write_manifest(path, read_manifest(path))
    assert path.read_bytes() == before
    broken = json.loads(path.read_text())
    broken["kernels"][0]["implementation_ref"] = "missing"
    with pytest.raises(
        RuntimeError, match="unknown cuDNN Frost implementations reference"
    ):
        write_manifest(path, broken)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.cpu_only
def test_export_adds_and_replaces_kernels_in_legacy_and_shared_catalogs(
    payload, tmp_path
):
    from flashinfer.fused_moe.backends.cudnn_frost.export import _write_manifest

    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps({**payload, "kernels": payload["kernels"][:1]}))
    second = payload["kernels"][1]
    _write_manifest(tmp_path, second, replace=False)
    assert json.loads(path.read_text())["schema_version"] == 3
    assert read_manifest(path) == payload
    with pytest.raises(RuntimeError, match="already exists"):
        _write_manifest(tmp_path, second, replace=False)
    updated = deepcopy(second)
    updated["workspace_bytes"] += 128
    _write_manifest(tmp_path, updated, replace=True)
    assert read_manifest(path)["kernels"] == [payload["kernels"][0], updated]


@pytest.mark.cpu_only
@pytest.mark.parametrize("schema_version", [2, 3])
def test_selection_copies_sources_and_prunes_unreferenced_metadata(
    payload, tmp_path, monkeypatch, schema_version
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks"))
    from bench_cudnn_frost_moe_bf16 import copy_selected

    candidate = tmp_path / "candidate"
    selected = tmp_path / "selected"
    candidate.mkdir()
    source_relative = payload["kernels"][0]["source"]["path"]
    source = candidate / source_relative
    source.parent.mkdir()
    source.write_bytes((artifact_root("bf16") / source_relative).read_bytes())
    manifest = normalize_manifest(payload) if schema_version == 3 else payload
    (candidate / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    copy_selected(candidate, selected, {"second"}, io.StringIO())
    assert (selected / source_relative).read_bytes() == source.read_bytes()
    assert read_manifest(selected / MANIFEST_FILENAME)["kernels"] == [
        payload["kernels"][1]
    ]
    catalog = json.loads((selected / MANIFEST_FILENAME).read_text())
    assert catalog["schema_version"] == 3
    assert all(
        len(catalog[table]) == 1
        for table in ("sources", "parameters", "implementations", "contracts")
    )
    before = (selected / MANIFEST_FILENAME).read_bytes()
    copy_selected(candidate, selected, {"second"}, io.StringIO())
    assert (selected / MANIFEST_FILENAME).read_bytes() == before


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"])
def test_packaged_catalogs_share_verified_sources_and_preserve_shortlist_ids(dtype):
    root = artifact_root(dtype)
    catalog = json.loads((root / MANIFEST_FILENAME).read_text())
    assert catalog["schema_version"] == 4
    assert "kernels" not in catalog and "implementations" not in catalog
    sources = [v["source"] for f in catalog["families"] for v in f["variants"]]
    assert len(sources) < len(read_manifest(root / MANIFEST_FILENAME)["kernels"])
    for source in sources:
        assert (
            hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )
    ids = {
        record["id"] for record in read_manifest(root / MANIFEST_FILENAME)["kernels"]
    }
    shortlists = json.loads((root / "moe_shortlists.json").read_text())
    assert all(
        identity in ids
        for entry in shortlists["entries"]
        for stage in ("fc1", "fc2")
        for identity in entry[stage]
    )
