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

"""Combine successful SM100a and SM103a H12 KDA benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET = BENCHMARKS_DIR / "presets" / "recurrent_kda_prefill_h12.json"
FLASH_KDA_PEER_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
REQUIRED_ARCHITECTURES = ("sm100a", "sm103a")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("H12 preset schema_version must be 1")
    if payload.get("name") != "recurrent_kda_prefill_h12":
        raise ValueError("unexpected H12 preset name")
    common = payload.get("common")
    if not isinstance(common, dict) or common.get("num_heads") != 12:
        raise ValueError("H12 preset must use 12 heads")
    if common.get("head_dim_qk") != 128 or common.get("head_dim_vo") != 128:
        raise ValueError("H12 preset must use K=V=128")
    if common.get("dtype") != "bfloat16" or common.get("lower_bound") != -5.0:
        raise ValueError("H12 preset must use BF16 and lower_bound=-5")
    if payload.get("aggregation") != "per_case_only":
        raise ValueError("H12 benchmark reports per-case results only")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("H12 preset must contain six cases")
    if any(
        not isinstance(case, dict) or case.get("layout") not in {"fixed", "packed"}
        for case in cases
    ):
        raise ValueError("H12 preset cases must use fixed or packed layout")
    names = [case.get("name") for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("H12 preset case names must be unique")
    return cases


def _positive_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _validate_arch_report(
    payload: Any,
    *,
    architecture: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{architecture} benchmark result must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError(f"{architecture} benchmark contains a malformed row")
        if row["name"] in rows:
            raise ValueError(f"{architecture} benchmark contains duplicate cases")
        rows[row["name"]] = row

    selected = []
    for case in cases:
        name = case["name"]
        try:
            row = rows[name]
        except KeyError as error:
            raise ValueError(f"{architecture} benchmark is missing {name}") from error
        hardware = row.get("hardware")
        if not isinstance(hardware, dict) or hardware.get("cuda_arch") != architecture:
            raise ValueError(f"{name} has the wrong architecture")
        if row.get("num_heads") != 12 or row.get("variant") != "m128_n16":
            raise ValueError(f"{name} did not use the H12 m128_n16 route")
        if row.get("seq_lens") != case["seq_lens"]:
            raise ValueError(f"{name} sequence lengths differ from the preset")
        if row.get("layout") != case["layout"] or row.get("seed") != case["seed"]:
            raise ValueError(f"{name} identity differs from the preset")
        if row.get("correctness_peer") != "passed":
            raise ValueError(f"{name} did not pass FlashKDA correctness")
        if row.get("timing_backend") != "cupti" or row.get("cold_l2") is not True:
            raise ValueError(f"{name} must use cold-L2 CUPTI timing")
        if row.get("timing_scope") != "public_recurrent_kda_with_inplace_state_update":
            raise ValueError(f"{name} did not time the public recurrent_kda call")
        provenance = row.get("flash_kda_peer_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("source_commit") != FLASH_KDA_PEER_COMMIT
        ):
            raise ValueError(f"{name} used the wrong FlashKDA peer")
        selected.append(
            {
                "name": name,
                "median_ms": _positive_number(row.get("median_ms"), "median_ms"),
                "flash_kda_peer_raw_ms": _positive_number(
                    row.get("flash_kda_peer_raw_ms"), "flash_kda_peer_raw_ms"
                ),
                "flash_kda_peer_adapted_ms": _positive_number(
                    row.get("flash_kda_peer_adapted_ms"),
                    "flash_kda_peer_adapted_ms",
                ),
                "speedup_vs_flash_kda_peer_adapted": _positive_number(
                    row.get("speedup_vs_flash_kda_peer_adapted"),
                    "speedup_vs_flash_kda_peer_adapted",
                ),
            }
        )
    return selected


def reduce_reports(
    sm100a: Any,
    sm103a: Any,
    *,
    preset: Path = DEFAULT_PRESET,
) -> dict[str, Any]:
    cases = _load_cases(preset)
    architectures = {
        "sm100a": _validate_arch_report(sm100a, architecture="sm100a", cases=cases),
        "sm103a": _validate_arch_report(sm103a, architecture="sm103a", cases=cases),
    }
    return {
        "schema_version": 1,
        "suite": "recurrent_kda_prefill_h12_dual_arch",
        "complete_dual_arch_h12": True,
        "case_order": [case["name"] for case in cases],
        "architectures": architectures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm100a", type=Path, required=True)
    parser.add_argument("--sm103a", type=Path, required=True)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    result = reduce_reports(
        _load_json(args.sm100a),
        _load_json(args.sm103a),
        preset=args.preset,
    )
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
