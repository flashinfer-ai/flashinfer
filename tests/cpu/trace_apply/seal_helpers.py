# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seal packages on the fly for the sealed-package tests (no GPU required)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).with_name("data") / "kernel_package"
GOLDEN_DIGEST = "59a6a6628b685f1becd920fd2fbf07845abe72bacce7f1935c6ff197c41a984d"

OPERATION = "test_kernel_op"
SOURCE = b"def run(x, out):\n    out.copy_(x)\n"
SOLUTION = {
    "name": "test_solution",
    "definition": OPERATION,
    "author": "sealed package tests",
    "spec": {
        "language": "python",
        "target_hardware": ["sm100"],
        "entry_point": "kernel.py::run",
        "destination_passing_style": True,
    },
    "sources": [{"path": "kernel.py", "content": SOURCE.decode()}],
}
#: Opaque sealed sidecar content: the sealed-package layer verifies its bytes
#: and never interprets it (a consumer like a qualification reader may).
QUALIFICATION = {
    "operation": OPERATION,
    "fi_api": "flashinfer.test_api",
    "flashinfer_version": "0.7.0.post1",
    "target_hardware": ["sm100"],
    "inputs": {"x": "float32", "out": "float32"},
    "receipt_sha256": "c" * 64,
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def listing_sha256(files: Mapping[str, str]) -> str:
    listing = "".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(listing.encode()).hexdigest()


def seal_package(
    root: Path,
    files: Mapping[str, bytes],
    *,
    sidecar: object | None = None,
    manifest_fields: Mapping[str, Any] | None = None,
) -> str:
    """Build the producer's manifest/listing envelope.

    ``sidecar`` is sealed via the ``qualification_sha256`` legacy key; tests
    that exercise the generic ``sealed_files`` spelling pass it through
    ``manifest_fields``.
    """
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    manifest = {"files": digests, "bundle_sha256": listing_sha256(digests)}
    if sidecar is not None:
        manifest["qualification_sha256"] = hashlib.sha256(
            json_bytes(sidecar)
        ).hexdigest()
    manifest.update(manifest_fields or {})
    manifest_bytes = json_bytes(manifest)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    directory = root / digest
    (directory / "bundle").mkdir(parents=True)
    (directory / "manifest.json").write_bytes(manifest_bytes)
    if sidecar is not None:
        (directory / "qualification.json").write_bytes(json_bytes(sidecar))
    for name, data in files.items():
        path = directory / "bundle" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return digest


def seal_flashinfer_fixture(
    root: Path, *, solution: object = SOLUTION, sidecar: object = QUALIFICATION
) -> str:
    """Seal the standard source; preserve supplied metadata unchanged."""
    return seal_package(
        root,
        {"solution.json": json_bytes(solution), "kernel.py": SOURCE},
        sidecar=sidecar,
    )
