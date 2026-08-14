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

import importlib.util
import json
from pathlib import Path

import pytest


BENCHMARKS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "reduce_kda_h12", BENCHMARKS_DIR / "reduce_kda_h12.py"
)
assert SPEC is not None and SPEC.loader is not None
REDUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REDUCER)


def _report(architecture: str) -> list[dict]:
    cases = REDUCER._load_cases(REDUCER.DEFAULT_PRESET)
    rows = []
    for index, case in enumerate(cases):
        median_ms = 0.1 + index * 0.01
        rows.append(
            {
                "name": case["name"],
                "num_heads": 12,
                "variant": "m128_n16",
                "seq_lens": case["seq_lens"],
                "layout": case["layout"],
                "seed": case["seed"],
                "hardware": {"cuda_arch": architecture},
                "correctness_peer": "passed",
                "timing_backend": "cupti",
                "cold_l2": True,
                "timing_scope": "public_recurrent_kda_with_inplace_state_update",
                "median_ms": median_ms,
                "flash_kda_peer_raw_ms": median_ms * 1.5,
                "flash_kda_peer_adapted_ms": median_ms * 1.6,
                "speedup_vs_flash_kda_peer_adapted": 1.6,
                "flash_kda_peer_provenance": {
                    "source_commit": REDUCER.FLASH_KDA_PEER_COMMIT
                },
            }
        )
    return rows


def test_reduce_reports_accepts_complete_dual_arch_results():
    result = REDUCER.reduce_reports(_report("sm100a"), _report("sm103a"))

    assert result["complete_dual_arch_h12"] is True
    assert tuple(result["architectures"]) == REDUCER.REQUIRED_ARCHITECTURES
    assert len(result["case_order"]) == 6
    assert len(result["architectures"]["sm100a"]) == 6
    assert len(result["architectures"]["sm103a"]) == 6


def test_reduce_reports_rejects_wrong_architecture():
    sm103a = _report("sm103a")
    sm103a[0]["hardware"]["cuda_arch"] = "sm100a"

    with pytest.raises(ValueError, match="wrong architecture"):
        REDUCER.reduce_reports(_report("sm100a"), sm103a)


def test_reduce_reports_rejects_missing_h12_case():
    with pytest.raises(ValueError, match="missing"):
        REDUCER.reduce_reports(_report("sm100a")[:-1], _report("sm103a"))


def test_load_cases_rejects_unknown_layout(tmp_path):
    preset = json.loads(REDUCER.DEFAULT_PRESET.read_text())
    preset["cases"][0]["layout"] = "unknown"
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset))

    with pytest.raises(ValueError, match="fixed or packed"):
        REDUCER._load_cases(preset_path)
