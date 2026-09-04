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

from __future__ import annotations

import hashlib
import json

import pytest

from flashinfer.jit.generated_program_promotion import (
    MANIFEST_KIND,
    PromotionIntegrityError,
    import_promotion,
    load_manifest,
    parse_manifest,
    verify_promotion,
)


def _artifact(source: str, destination: str, data: bytes, *, executable=False):
    return {
        "destination": destination,
        "executable": executable,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "source": source,
    }


def _manifest(mode: str, artifacts: list[dict]):
    return {
        "artifacts": sorted(artifacts, key=lambda item: item["destination"]),
        "kind": MANIFEST_KIND,
        "mode": mode,
        "name": "example-program",
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    ("mode", "filename", "data"),
    (
        ("cuda", "kernel.cu", b'extern "C" __global__ void kernel() {}\n'),
        ("cubin", "kernel.cubin", b"\x7fELF\x02\x01generated-program"),
    ),
)
def test_import_and_rehash_preserve_exact_bytes(tmp_path, mode, filename, data):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    source = payload_root / "program" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    manifest = parse_manifest(
        _manifest(
            mode,
            [_artifact(f"program/{filename}", f"generated/{filename}", data)],
        )
    )

    import_promotion(
        manifest,
        payload_root=payload_root,
        output_root=output_root,
        mode=mode,
    )

    destination = output_root / "generated" / filename
    assert destination.read_bytes() == data
    verify_promotion(
        manifest,
        payload_root=payload_root,
        output_root=output_root,
        mode=mode,
    )

    destination.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(PromotionIntegrityError, match="identity mismatch"):
        verify_promotion(
            manifest,
            payload_root=payload_root,
            output_root=output_root,
            mode=mode,
        )
    import_promotion(
        manifest,
        payload_root=payload_root,
        output_root=output_root,
        mode=mode,
        replace=True,
    )
    assert destination.read_bytes() == data


def test_import_preflights_all_sources_before_writing(tmp_path):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    payload_root.mkdir()
    first = b"first"
    second = b"second"
    (payload_root / "first.cu").write_bytes(first)
    (payload_root / "second.cu").write_bytes(b"tampered")
    manifest = parse_manifest(
        _manifest(
            "cuda",
            [
                _artifact("first.cu", "generated/first.cu", first),
                _artifact("second.cu", "generated/second.cu", second),
            ],
        )
    )

    with pytest.raises(PromotionIntegrityError, match="identity mismatch"):
        import_promotion(
            manifest,
            payload_root=payload_root,
            output_root=output_root,
            mode="cuda",
        )
    assert not (output_root / "generated" / "first.cu").exists()


def test_manifest_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"kind":"unused",'
        '"name":"unused","mode":"cuda","artifacts":[]}',
        encoding="utf-8",
    )
    with pytest.raises(PromotionIntegrityError, match="duplicate JSON key"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda payload: payload.update(extra=True), "manifest keys"),
        (
            lambda payload: payload["artifacts"][0].update(source="../kernel.cu"),
            "normalized relative path",
        ),
        (
            lambda payload: payload["artifacts"][0].update(sha256="ABC"),
            "full lowercase SHA-256",
        ),
    ),
)
def test_manifest_rejects_ambiguous_or_unverifiable_fields(mutate, message):
    data = b"kernel"
    payload = _manifest("cuda", [_artifact("kernel.cu", "kernel.cu", data)])
    mutate(payload)
    with pytest.raises(PromotionIntegrityError, match=message):
        parse_manifest(payload)


def test_manifest_requires_canonical_artifact_order():
    payload = _manifest(
        "cuda",
        [
            _artifact("a.cu", "a.cu", b"a"),
            _artifact("z.cu", "z.cu", b"z"),
        ],
    )
    payload["artifacts"].reverse()
    with pytest.raises(PromotionIntegrityError, match="sorted by destination"):
        parse_manifest(payload)


def test_import_rejects_symlinked_payload_file(tmp_path):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    payload_root.mkdir()
    external = tmp_path / "external.cubin"
    data = b"cubin"
    external.write_bytes(data)
    (payload_root / "kernel.cubin").symlink_to(external)
    manifest = parse_manifest(
        _manifest(
            "cubin",
            [_artifact("kernel.cubin", "generated/kernel.cubin", data)],
        )
    )

    with pytest.raises(PromotionIntegrityError, match="file symlink"):
        import_promotion(
            manifest,
            payload_root=payload_root,
            output_root=output_root,
            mode="cubin",
        )


def test_import_preserves_declared_executable_bit(tmp_path):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    payload_root.mkdir()
    data = b"#!/bin/sh\nexit 0\n"
    (payload_root / "launcher").write_bytes(data)
    manifest = parse_manifest(
        _manifest(
            "cuda",
            [_artifact("launcher", "generated/launcher", data, executable=True)],
        )
    )

    import_promotion(
        manifest,
        payload_root=payload_root,
        output_root=output_root,
        mode="cuda",
    )

    assert (output_root / "generated" / "launcher").stat().st_mode & 0o111


def test_import_rejects_extra_payload_file(tmp_path):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    payload_root.mkdir()
    data = b"kernel"
    (payload_root / "kernel.cu").write_bytes(data)
    (payload_root / "undeclared.txt").write_text("extra", encoding="utf-8")
    manifest = parse_manifest(
        _manifest("cuda", [_artifact("kernel.cu", "kernel.cu", data)])
    )

    with pytest.raises(PromotionIntegrityError, match="extra=.*undeclared.txt"):
        import_promotion(
            manifest,
            payload_root=payload_root,
            output_root=output_root,
            mode="cuda",
        )


def test_import_requires_explicit_matching_mode(tmp_path):
    payload_root = tmp_path / "payload"
    output_root = tmp_path / "checkout"
    payload_root.mkdir()
    data = b"kernel"
    (payload_root / "kernel.cu").write_bytes(data)
    manifest = parse_manifest(
        _manifest("cuda", [_artifact("kernel.cu", "kernel.cu", data)])
    )

    with pytest.raises(PromotionIntegrityError, match="does not match manifest"):
        import_promotion(
            manifest,
            payload_root=payload_root,
            output_root=output_root,
            mode="cubin",
        )


def test_manifest_file_round_trip(tmp_path):
    data = b"kernel"
    payload = _manifest("cuda", [_artifact("kernel.cu", "kernel.cu", data)])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_manifest(path) == parse_manifest(payload)
