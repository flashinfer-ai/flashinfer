"""
Copyright (c) 2026 by NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "flashinfer" / "experimental" / "union_gqa"
_CSRC = _PACKAGE / "csrc" / "sm100a"


def test_union_gqa_is_a_standalone_source_bundle() -> None:
    expected = {
        "union_gqa_mma_noelect.cuh",
        "union_gqa_prefill.cu",
        "union_gqa_prefill_api.cuh",
        "union_gqa_prefill_binding.cu",
        "union_gqa_utils.cuh",
    }
    assert {path.name for path in _CSRC.iterdir()} == expected

    all_source = "\n".join(
        (_CSRC / name).read_text(encoding="utf-8") for name in sorted(expected)
    )
    forbidden = ("torch::", "torch/extension")
    for marker in forbidden:
        assert marker not in all_source
    assert "namespace flashinfer::union_gqa" in all_source


def test_block_index_contract_is_64_tokens() -> None:
    source = (_CSRC / "union_gqa_prefill.cu").read_text(encoding="utf-8")
    assert "static constexpr int BLOCK_SIZE = 64;" in source
    assert "static constexpr int B_N = BLOCK_SIZE;" in source
    assert "params.topk_blocks" in source
    assert "params.topk / BLOCK_SIZE" not in source
    assert "blk <= last_valid_kv / BLOCK_SIZE" in source
    assert "r_pair[j] = nvutil::ex2_emulation_3deg_pair(x_pair);" in source


def test_partial_epilogue_uses_64_bit_output_offset() -> None:
    source = (_CSRC / "union_gqa_prefill.cu").read_text(encoding="utf-8")
    assert "const int64_t out_idx" in source
    assert "static_cast<int64_t>(global_token) * params.h_q * D_V" in source
    assert "params.out[out_idx] = sO_copy(head, dim, token);" in source


def test_cutlass_derived_mma_atoms_retain_bsd_license() -> None:
    source = (_CSRC / "union_gqa_mma_noelect.cuh").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: BSD-3-Clause" in source
    assert "Derived from CuTe's SM100_MMA_F16BF16_{SS,TS} atoms" in source


def test_experimental_backend_is_not_registered_for_aot() -> None:
    source = (_ROOT / "flashinfer" / "aot.py").read_text(encoding="utf-8")
    assert "union_gqa" not in source


def test_package_data_includes_cuda_sources() -> None:
    source = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"flashinfer.experimental.union_gqa" = ["csrc/**"]' in source


def test_jit_module_uses_body_and_binding_translation_units(tmp_path: Path) -> None:
    # Construct the JIT spec without importing the top-level package, which
    # initializes unrelated GPU backends.
    script = textwrap.dedent(
        f"""
        import importlib
        import json
        import sys
        import types
        from pathlib import Path

        package_root = Path({str(_ROOT / "flashinfer")!r})
        packages = {{
            "flashinfer": package_root,
            "flashinfer.jit": package_root / "jit",
            "flashinfer.experimental": package_root / "experimental",
            "flashinfer.experimental.union_gqa": package_root / "experimental" / "union_gqa",
        }}
        for name, path in packages.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package

        core = importlib.import_module("flashinfer.jit.core")
        union_gqa = importlib.import_module("flashinfer.experimental.union_gqa.jit")
        core.current_compilation_context.TARGET_CUDA_ARCHS = {{(10, "0a")}}
        spec = union_gqa.gen_union_gqa_module()
        print(json.dumps({{
            "name": spec.name,
            "sources": [source.name for source in spec.sources],
            "cuda_flags": spec.extra_cuda_cflags,
        }}))
        """
    )
    env = os.environ.copy()
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    spec = json.loads(result.stdout)

    assert spec["name"] == "union_gqa_prefill_sm100a"
    assert spec["sources"] == [
        "union_gqa_prefill.cu",
        "union_gqa_prefill_binding.cu",
    ]
    assert "-std=c++20" in spec["cuda_flags"]
    assert "-gencode=arch=compute_100a,code=sm_100a" in spec["cuda_flags"]
    assert "-Xptxas" in spec["cuda_flags"]
    assert "-O2" in spec["cuda_flags"]
