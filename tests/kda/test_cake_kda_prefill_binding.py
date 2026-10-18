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
from flashinfer.jit.cake_kda_prefill_binding import BINDINGS

PREPARED_EXPORTS = ("run", "run_prepare_tma", "run_prepared")


def test_every_module_has_one_binding_row_and_one_kernel_source():
    assert sorted(BINDINGS) == sorted(cake_kda_tf32.MODULES)
    for record in cake_kda_tf32.MODULES.values():
        assert [Path(path).name for path in record["sources"]] == [
            f"{record['cache_name']}_kernel.cu"
        ]


def test_bindings_render_to_the_former_per_module_text():
    # The sidecar pins each rendered binding to the sha256 of the checked-in
    # binding file it replaced (keyed by module).
    sidecar = Path(__file__).with_name("cake_kda_prefill_binding_sha256.json")
    pinned = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sorted(pinned) == sorted(cake_kda_tf32.MODULES)
    symbols = set()
    namespaces = set()
    for name, record in cake_kda_tf32.MODULES.items():
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
    assert len(symbols) == len(cake_kda_tf32.MODULES)
    assert len(namespaces) == len(cake_kda_tf32.MODULES)


@pytest.mark.parametrize("arch", cake_kda_tf32.ARCHES)
def test_spec_lists_the_kernel_body_and_the_rendered_binding(
    monkeypatch, tmp_path, arch
):
    minor = {"sm_100a": "0", "sm_103a": "3"}[arch]
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, f"{minor}a")}
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    name = next(
        name
        for name, record in cake_kda_tf32.MODULES.items()
        if arch in record["arches"]
    )
    record = cake_kda_tf32.MODULES[name]
    cake_kda_tf32.spec.cache_clear()
    try:
        spec = cake_kda_tf32.spec(name, arch)
        assert spec.name == f"{record['cache_name']}_{arch}"
        kernel, binding = spec.sources
        assert kernel.name == f"{record['cache_name']}_kernel.cu"
        assert kernel.is_file()
        assert binding == tmp_path / spec.name / f"{record['cache_name']}_binding.cu"
        assert binding.read_text(encoding="utf-8") == cake_kda_tf32.render_binding(name)
        assert spec.extra_include_dirs[0] == kernel.parent
        assert [
            flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
        ] == [f"-gencode=arch=compute_10{minor}a,code=sm_10{minor}a"]
    finally:
        cake_kda_tf32.spec.cache_clear()
