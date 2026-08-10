# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate SM100a+SM103a H12 receipts and emit the promotion gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kda_h12_evidence import load_preset, reduce_dual_arch_receipts


BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET = BENCHMARKS_DIR / "presets" / "recurrent_kda_prefill_h12_phase_a.json"


def _load_receipt(path: Path) -> tuple[dict, str]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"receipt root must be an object: {path}")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm100a", type=Path, required=True)
    parser.add_argument("--sm103a", type=Path, required=True)
    parser.add_argument("--expected-flashinfer-commit", required=True)
    parser.add_argument("--expected-fla-commit", required=True)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    sm100a, sm100a_sha256 = _load_receipt(args.sm100a)
    sm103a, sm103a_sha256 = _load_receipt(args.sm103a)
    result = reduce_dual_arch_receipts(
        sm100a_report=sm100a,
        sm103a_report=sm103a,
        sm100a_receipt_sha256=sm100a_sha256,
        sm103a_receipt_sha256=sm103a_sha256,
        expected_candidate_commit=args.expected_flashinfer_commit,
        expected_fla_commit=args.expected_fla_commit,
        preset=load_preset(args.preset),
    )
    result["receipts"]["sm100a"]["path"] = str(args.sm100a.resolve())
    result["receipts"]["sm103a"]["path"] = str(args.sm103a.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
