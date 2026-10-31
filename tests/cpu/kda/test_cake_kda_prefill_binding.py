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

"""CPU checks of the rendered host bindings of the generated KDA prefill modules."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from flashinfer.jit import cake_kda_tf32
from flashinfer.jit import core as jit_core
from flashinfer.jit import env as jit_env
from flashinfer.jit.cake_kda_prefill_binding import BINDINGS, _DEFAULT_SMEM_LIMIT

PREPARED_EXPORTS = ("run", "run_prepare_tma", "run_prepared")
# The shipped modules (the hand-owned rows); generated programs ship their own
# binding and are checked by test_cake_kda_prefill_generated.py.
SHIPPED = {name: cake_kda_tf32.MODULES[name] for name in cake_kda_tf32._MODULE_ROWS}


@pytest.mark.cpu_only
def test_every_module_has_one_binding_row_and_one_kernel_source():
    assert sorted(BINDINGS) == sorted(SHIPPED)
    for record in SHIPPED.values():
        # The fold test pins which body a module compiles.
        (source,) = record["sources"]
        assert re.fullmatch(
            r"csrc/kda/(bf16|tf32)/cake_kda_\1_[0-9a-f]{64}_kernel\.cu", source
        ), source


@pytest.mark.cpu_only
def test_bindings_render_to_the_pinned_text():
    # The sidecar pins the sha256 of each rendered binding (keyed by module).
    sidecar = Path(__file__).with_name("cake_kda_prefill_binding_sha256.json")
    pinned = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sorted(pinned) == sorted(SHIPPED)
    symbols = set()
    namespaces = set()
    for name, record in SHIPPED.items():
        text = cake_kda_tf32.render_binding(name)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == pinned[name], name
        symbol = f"kernel_{record['cache_name']}"
        assert f'extern "C" __global__ void {symbol}(' in text
        (namespace,) = re.findall(
            r"^namespace (cake_host_shim_[0-9a-f]{64}) \{$", text, re.MULTILINE
        )
        exports = re.findall(
            r"^TVM_FFI_DLL_EXPORT_TYPED_FUNC\((\w+), (\w+)::\w+\);$", text, re.MULTILINE
        )
        expected = PREPARED_EXPORTS if record["tma_workspace_bytes"] else ("run",)
        assert exports == [(entry, namespace) for entry in expected], name
        symbols.add(symbol)
        namespaces.add(namespace)
    # Every module keeps its own kernel symbol and host-shim namespace, so all
    # modules can load into one process.
    assert len(symbols) == len(SHIPPED)
    assert len(namespaces) == len(SHIPPED)


@pytest.mark.cpu_only
def test_prepared_launch_skips_descriptor_encoding():
    # Only the preparing actions encode and upload the TMA descriptors and opt
    # the kernel into its dynamic shared memory; the launch-only action reuses
    # both. Modules without descriptors export run only and keep the opt-in.
    for name, record in SHIPPED.items():
        _signature, smem_bytes, _block, _launch, encoders, _strides = BINDINGS[name]
        text = cake_kda_tf32.render_binding(name)
        assert "ScopedCudaDevice" not in text
        assert "CUtensorMap h_" not in text
        opt_ins = text.count("EnsureMaxDynamicSharedMemory(")
        assert opt_ins == int(smem_bytes > _DEFAULT_SMEM_LIMIT)
        if not record["tma_workspace_bytes"]:
            assert "EncodeTma_" not in text
            assert "descriptor_action" not in text
            continue
        start = text.index("  if (descriptor_action != 0) {\n")
        end = text.index("  if (descriptor_action == 1) return;\n", start)
        preparing = text[start:end]
        assert text.count("EncodeTma_") == len(encoders)
        assert preparing.count("EncodeTma_") == len(encoders)
        assert preparing.count("EnsureMaxDynamicSharedMemory(") == opt_ins


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", cake_kda_tf32.ARCHES)
def test_spec_lists_the_kernel_body_and_the_rendered_binding(
    monkeypatch, tmp_path, arch
):
    minor = {"sm_100a": "0", "sm_103a": "3"}[arch]
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, f"{minor}a")}
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    name = next(name for name, record in SHIPPED.items() if arch in record["arches"])
    record = SHIPPED[name]
    cake_kda_tf32.spec.cache_clear()
    try:
        spec = cake_kda_tf32.spec(name, arch)
        assert spec.name == f"{record['cache_name']}_{arch}"
        kernel, binding = spec.sources
        assert kernel.name == Path(record["sources"][0]).name
        assert kernel.is_file()
        assert binding == tmp_path / spec.name / f"{record['cache_name']}_binding.cu"
        assert binding.read_text(encoding="utf-8") == cake_kda_tf32.render_binding(name)
        assert spec.extra_include_dirs[0] == kernel.parent
        assert [
            flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
        ] == [f"-gencode=arch=compute_10{minor}a,code=sm_10{minor}a"]
    finally:
        cake_kda_tf32.spec.cache_clear()
