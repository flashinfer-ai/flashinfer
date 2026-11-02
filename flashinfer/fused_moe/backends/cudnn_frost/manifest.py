# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Read and write the shared Frost source catalog without importing cuDNN.

Schema v4 factors tile, activation and store axes; v3 interns expanded metadata.
Readers expand v2/v3/v4 to the same descriptors for dtype-specific validation.
Logical kernel IDs are independent of device resources; source/compiler hashes
qualify persisted tactics.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "cudnn_frost_selected_kernels.json"
_TABLES = ("sources", "parameters", "implementations", "contracts")
_IMPLEMENTATION_FIELDS = (
    "op",
    "arch",
    "abi",
    "workspace_bytes",
    "launch",
    "tactic",
)
_ENTRY_REFS = {"implementation_ref", "contract_ref"}


def expand_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return independent, expanded records from either source catalog schema."""
    if isinstance(payload, dict) and payload.get("schema_version") == 4:
        from .dimension_catalog import expand_catalog

        return expand_catalog(payload)
    if not isinstance(payload, dict) or payload.get("schema_version") not in (2, 3):
        raise RuntimeError("unsupported cuDNN Frost manifest schema")
    kernels = payload.get("kernels")
    if not isinstance(kernels, list) or any(not isinstance(k, dict) for k in kernels):
        raise RuntimeError("cuDNN Frost manifest 'kernels' must be a list of records")
    if payload["schema_version"] == 2:
        return deepcopy(payload)

    for name in _TABLES:
        table = payload.get(name)
        if not isinstance(table, dict) or any(
            not isinstance(value, dict) for value in table.values()
        ):
            raise RuntimeError(
                f"cuDNN Frost manifest '{name}' must be a table of records"
            )

    def resolve(table: str, reference: Any) -> dict[str, Any]:
        if not isinstance(reference, str) or reference not in payload[table]:
            raise RuntimeError(f"unknown cuDNN Frost {table} reference: {reference!r}")
        return deepcopy(payload[table][reference])

    records = []
    for entry in kernels:
        if (_ENTRY_REFS - entry.keys()) or (
            set(_IMPLEMENTATION_FIELDS) | {"source", "contract"}
        ) & entry.keys():
            raise RuntimeError("invalid cuDNN Frost catalog kernel entry")
        implementation = resolve("implementations", entry["implementation_ref"])
        fields = set(_IMPLEMENTATION_FIELDS) | {"source_ref"}
        if not fields <= implementation.keys() or implementation.keys() - (
            fields | {"parameters_ref"}
        ):
            raise RuntimeError("invalid cuDNN Frost catalog implementation")
        source = resolve("sources", implementation.pop("source_ref"))
        if "parameters" in source:
            raise RuntimeError("cuDNN Frost catalog sources must reference parameters")
        if "parameters_ref" in implementation:
            source["parameters"] = resolve(
                "parameters", implementation.pop("parameters_ref")
            )
        record = {k: deepcopy(v) for k, v in entry.items() if k not in _ENTRY_REFS}
        record.update(implementation)
        record["source"] = source
        record["contract"] = resolve("contracts", entry["contract_ref"])
        records.append(record)
    return {
        **{k: deepcopy(v) for k, v in payload.items() if k not in _TABLES},
        "schema_version": 2,
        "kernels": records,
    }


def read_manifest(path: Path) -> dict[str, Any]:
    """Read source metadata; discovery caches are owned by dtype runtimes."""
    try:
        return expand_manifest(json.loads(path.read_text()))
    except RuntimeError as exc:
        raise RuntimeError(f"{path}: {exc}") from exc


def normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a v3 catalog, preserving record order, IDs and all source constants."""
    expanded = expand_manifest(payload)
    tables: dict[str, dict[str, Any]] = {name: {} for name in _TABLES}

    def intern(table: str, value: dict[str, Any]) -> str:
        if not isinstance(value, dict):
            raise RuntimeError(f"cuDNN Frost {table} entries must be records")
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        # Content references stay stable when other records are added or removed.
        reference = hashlib.sha256(encoded).hexdigest()[:16]
        if reference in tables[table] and tables[table][reference] != value:
            raise RuntimeError(f"cuDNN Frost {table} reference collision")
        tables[table][reference] = value
        return reference

    entries = []
    for record in expanded["kernels"]:
        if _ENTRY_REFS & record.keys():
            raise RuntimeError("reserved cuDNN Frost catalog reference field")
        source = record.pop("source")
        implementation = {field: record.pop(field) for field in _IMPLEMENTATION_FIELDS}
        implementation["source_ref"] = intern(
            "sources", {k: v for k, v in source.items() if k != "parameters"}
        )
        if "parameters" in source:
            # Keep the ordered assignments, including repeated constant names.
            implementation["parameters_ref"] = intern(
                "parameters", source["parameters"]
            )
        record["implementation_ref"] = intern("implementations", implementation)
        record["contract_ref"] = intern("contracts", record.pop("contract"))
        entries.append(record)
    return {
        **{k: v for k, v in expanded.items() if k not in ("schema_version", "kernels")},
        "schema_version": 3,
        **tables,
        "kernels": entries,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Atomically write a catalog and return its input-to-persisted ID mapping."""
    if payload.get("schema_version") != 4 and path.exists():
        current = json.loads(path.read_text())
        if current.get("schema_version") == 4:
            from .dimension_catalog import write_catalog

            expanded = expand_manifest(payload)
            return write_catalog(
                path,
                expanded["kernels"],
                current["dtype"],
                producer=expanded.get("producer"),
            )
    if payload.get("schema_version") == 4:
        expanded = expand_manifest(payload)
        catalog = payload
    else:
        expanded = expand_manifest(payload)
        catalog = normalize_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as file:
            file.write(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {record["id"]: record["id"] for record in expanded["kernels"]}
