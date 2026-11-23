"""
Copyright (c) 2026 by FlashInfer team.

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

# State pools of the Cake SSDCombined backend: with ``initial_state_slots`` /
# ``final_state_slots`` a packed-varlen call reads its initial states from and
# writes its final and checkpoint states to rows of caller pools, through the
# state-slot twins of the dense programs (``_SLOT_MODULES``); the slot vectors
# may be strided, ``initial_state_flags`` gates each sequence's initial read
# and ``null_slot_id`` names the slot of no row.  The CPU tests check the
# program tables, the slot launchers, that the dense artifacts are unchanged,
# the route of every call form to a slot program, the bound values and the
# validation; the GPU tests check each slot program bitwise against its dense
# twin on the gathered states, the -1, null and out-of-range slots, flags,
# checkpoint pools (the state pool, a separate pool of another row stride and
# row count, a dense checkpoint tensor), in-place pools, a vLLM-shaped paged
# cache, 64-bit pool offsets and CUDA-graph replays with mutated slot vectors,
# flags and tokens.  The look-back slot twin is expected only where the route
# serves it (``_LOOKBACK_STATE_SLOTS`` and the architecture's slot table).
# On sm_107a the claim and wrap programs have slot twins of their own
# (``_SM107_SLOT_MODULES`` alone): a slot call runs one exactly when the same
# call without slot vectors runs that program, and the sm_107a GPU cells of
# those routes check them against the dense call as above.

import hashlib
import importlib
import inspect
import re
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Optional

import pytest
import torch

from flashinfer.mamba import SSDCombined
from flashinfer.utils import get_compute_capability

from .test_cake_ssd_combined import (
    _HOST_PLACEHOLDERS,
    _assert_cake_accuracy,
    _cpu_forwarding_runner,
    _cpu_public_run_inputs,
    _cpu_route_call,
    _host_prepare_arguments,
    _host_run_arguments,
    _public_runner_without_constructor,
    _route_shapes,
    _signature_contract,
)

_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"
_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
_SM_COUNTS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
_CAKE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_STATES = ("bf16", "f16", "f32")
_STATE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_ROW = 64 * 128  # elements of one head's state

# The seventeen state-slot programs, the same keys on every architecture: the
# dense key plus the ``slots`` variant (``_slots`` without a dense variant).
_SLOT_KEYS = frozenset(
    [f"exactfused_{state}_varlen_slots" for state in _STATES]
    + [f"exactfused_{state}_varlen_noringslots" for state in ("bf16", "f16")]
    + [f"exactpre_{state}_varlen_slots" for state in _STATES]
    + [f"exactpre_{state}_varlen_tailslots" for state in _STATES]
    + [f"chunkpar_{state}_varlen_slots" for state in _STATES]
    + [f"chunkpar3_{state}_varlen_slots" for state in _STATES]
)
# The four sm_107a-only state-slot programs, in ``_SM107_SLOT_MODULES`` alone:
# the twins of the claim and wrap programs of the bf16 and f32 states.
_SM107_ONLY_SLOT_KEYS = frozenset(
    f"exactpre_{state}_varlen_{variant}slots"
    for variant in ("claim", "wrap")
    for state in ("bf16", "f32")
)
_ALL_SLOT_KEYS = _SLOT_KEYS | _SM107_ONLY_SLOT_KEYS
_SLOT_FAMILIES = frozenset({"exactfused", "exactpre", "chunkpar", "chunkpar3"})
# Main dynamic SMEM of the slot programs by (family, dense variant): the dense
# twin's literal; the sm_107a tail pair render keeps 8 more bytes.
_SLOT_SMEM_BYTES = {
    ("exactfused", ""): 232320,
    ("exactfused", "noring"): 232192,
    ("exactpre", ""): 232320,
    ("exactpre", "tail"): 232320,
    ("exactpre", "claim"): 232448,
    ("exactpre", "wrap"): 232448,
    ("chunkpar", ""): 231808,
    ("chunkpar3", ""): 232448,
}
# The twelve slot arguments in the kernels' order and the main-argument
# counts of the slot launchers (exact scan, three-phase, look-back).
_SLOT_ARGS = (
    "initial_state_slots",
    "final_state_slots",
    "initial_state_slot_stride",
    "final_state_slot_stride",
    "initial_state_rows",
    "final_state_rows",
    "checkpoint_state_slot_stride",
    "initial_state_flags",
    "initial_state_slots_step",
    "final_state_slots_step",
    "checkpoint_state_slots_step",
    "null_slot",
)
_EXACT_SLOT_ARGS, _CHUNKPAR3_SLOT_ARGS, _CHUNKPAR_SLOT_ARGS = 56, 57, 59
# The look-back family's slot programs follow ``_LOOKBACK_STATE_SLOTS``.
_LOOKBACK_SWITCH = "_LOOKBACK_STATE_SLOTS"
_SM107_TAIL_SMEM_BYTES = 232448

# sha256 of the dense host templates and of the dense device units (one
# digest over the sorted ``"<file name> <sha256>\n"`` lines of the 123
# units): the slot programs add files and table entries, never edit these.
_DENSE_HOST_TEMPLATE_SHA256 = {
    "mamba_ssd_combined_sequence.cpp": (
        "a992780866f3b70c21f009dfdc74753acecc8573925d219dad5e822821771344"
    ),
    "mamba_ssd_combined_fused_sequence.cpp": (
        "c734c348b3558c2ba1cf74d2536daf2cc64e4288c8579774538af1fe1936fabc"
    ),
    "mamba_ssd_combined_fused_metadata_sequence.cpp": (
        "8bb56cacf3481563597a4465fd9cb089427063a2ff4a46cec78b5b150d704e02"
    ),
    "mamba_ssd_combined_chunk_parallel_sequence.cpp": (
        "f3e0987ad000b58e432994e5dc2cc1bf5180c41d6f4681452bf6ce02ea4f36ee"
    ),
    "mamba_ssd_combined_chunk_parallel3_sequence.cpp": (
        "872749559c0d099875b44bf020f712536b66918ad10ca2e6e4c29eb3142d8bea"
    ),
}
_DENSE_DEVICE_UNITS = 123
_DENSE_DEVICE_UNITS_SHA256 = (
    "fb0be2e0dedff31a748285f52a17174ccb3ea2c9aa276de6dda93a91dfd3ed08"
)


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


@pytest.fixture(autouse=True)
def _without_chunk_parallel_override(monkeypatch):
    """Every test starts with ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` unset
    (``auto``); the tests of an override set it through ``monkeypatch``."""

    monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)


def _dense_twin(name):
    """The dense program a state-slot program is the twin of."""

    if name.endswith("_slots"):
        return name[: -len("_slots")]
    assert name.endswith("slots"), name
    return name[: -len("slots")]


def _slot_key(dense):
    """The state-slot twin of dense program ``dense``."""

    return dense + ("slots" if dense.count("_") == 3 else "_slots")


def _lookback_slot_family(module, state_dtype, arch):
    """The family a slot call the chunk-parallel rule sends to the look-back
    program runs on ``arch``: ``chunkpar`` with ``_LOOKBACK_STATE_SLOTS`` on
    and the look-back slot twin of the state dtype in the slot table the
    architecture builds (``_SLOT_PROGRAMS``, and ``_SM107_SLOT_MODULES`` on
    sm_107a), else ``chunkpar3``."""

    key = f"chunkpar_{module._STATE_DTYPE_KEYS[state_dtype]}_varlen_slots"
    served = (
        getattr(module, _LOOKBACK_SWITCH)
        and key in module._SLOT_PROGRAMS
        and (arch != "sm_107a" or key in module._SM107_SLOT_MODULES)
    )
    return "chunkpar" if served else "chunkpar3"


def _served_slot_keys(module, arch):
    """The slot programs calls on ``arch`` can reach: ``_SLOT_KEYS`` without
    the look-back twins the route does not serve there, and on sm_107a the
    claim and wrap twins (``_SM107_ONLY_SLOT_KEYS``)."""

    served = frozenset(
        key
        for key in _SLOT_KEYS
        if not key.startswith("chunkpar_")
        or _lookback_slot_family(
            module, _STATE_DTYPES[_STATES.index(key.split("_")[1])], arch
        )
        == "chunkpar"
    )
    return served | (_SM107_ONLY_SLOT_KEYS if arch == "sm_107a" else frozenset())


def _slot_key_arches(key):
    """The architectures that build slot program ``key``."""

    return _ARCHES if key in _SLOT_KEYS else ("sm_107a",)


def _kernel_parameters(text, kernel):
    """Parameter names of ``kernel``'s signature in a device source."""

    match = re.search(rf"\b{kernel}\(([^)]*)\)", text)
    assert match is not None, kernel
    return tuple(
        parameter.split()[-1].lstrip("*&") for parameter in match.group(1).split(",")
    )


def _metadata(lengths, *, seq_idx_dtype=torch.int32, device="cpu"):
    """Packed-varlen metadata of ``lengths`` (a length may be 0): ``seq_idx``
    ``[1, total]``, the chunk-128 logical segments (a segment starts at every
    chunk start and at every sequence start inside a chunk), the exclusive
    prefix sum of per-sequence logical chunk counts and ``cu_seqlens``."""

    starts, start = [], 0
    for length in lengths:
        starts.append(start)
        start += length
    total = start
    boundaries = {
        first for first, length in zip(starts, lengths, strict=True) if length
    }
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-total // 128)):
        low = chunk * 128
        inside = {first - low for first in boundaries if low < first < low + 128}
        for offset in sorted({0} | inside):
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    cumsum = [0]
    for first, length in zip(starts, lengths, strict=True):
        end = first + length
        chunks = (-(-end // 128) - first // 128) if length else 0
        cumsum.append(cumsum[-1] + chunks)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.int64)
    ).reshape(1, total)

    def as_i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return dict(
        seq_idx=seq_idx.to(dtype=seq_idx_dtype, device=device),
        chunk_indices=as_i32(chunk_indices),
        chunk_offsets=as_i32(chunk_offsets),
        seq_chunk_cumsum=as_i32(cumsum),
        cu_seqlens=as_i32([0, *accumulate(lengths)]),
    )


def _form_arguments(metadata, form, num_sequences):
    """The metadata arguments of one call form: ``cu_seqlens``, the seq_idx
    triple deriving its table (``derived``), the triple with the caller's
    trusted table (``trusted``) or the triple publishing the table into the
    caller's buffer (``publishing``)."""

    if form == "cu_seqlens":
        return {"cu_seqlens": metadata["cu_seqlens"]}
    arguments = {
        "seq_idx": metadata["seq_idx"],
        "chunk_indices": metadata["chunk_indices"],
        "chunk_offsets": metadata["chunk_offsets"],
        "num_seqs": num_sequences,
    }
    if form == "trusted":
        arguments["seq_chunk_cumsum"] = metadata["seq_chunk_cumsum"]
    elif form == "publishing":
        arguments["seq_chunk_cumsum"] = torch.full_like(
            metadata["seq_chunk_cumsum"], -1
        )
        arguments["update_seq_chunk_cumsum"] = True
    else:
        assert form == "derived", form
    return arguments


def _cpu_varlen_call(lengths, nheads, ngroups, has_z, form):
    """CPU tensors and keyword arguments of one packed-varlen call."""

    total = sum(lengths)
    x = torch.empty((1, total, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, total, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((1, total, ngroups, 128), dtype=torch.bfloat16)
    kwargs = {"z": torch.empty_like(x)} if has_z else {}
    kwargs.update(_form_arguments(_metadata(lengths), form, len(lengths)))
    return (x, dt, A, B, torch.empty_like(B)), kwargs


def _cpu_pool(rows, nheads, state_dtype, *, pad=64):
    """A CPU state pool of ``rows`` rows ``pad`` elements apart."""

    row = nheads * _ROW
    storage = torch.empty(rows * (row + pad), dtype=state_dtype)
    return storage.as_strided((rows, nheads, 64, 128), (row + pad, _ROW, 128, 1))


# ---------------------------------------------------------------------------
# CPU: tables, launchers and unchanged dense artifacts.


def test_slot_tables_bind_the_dense_twins_without_gpu():
    """Both slot tables hold the seventeen keys, and the sm_107a table the
    four claim and wrap twins beside them; each unit is its dense main stem
    plus ``_slots`` under its own content hash (one source for sm_100a and
    sm_103a, one for sm_107a; the claim and wrap twins an sm_107a source
    alone), takes the dense twin's arguments plus the twelve slot arguments
    and drops ``__restrict__`` from the three state pools; each program
    launches with its dense twin's SMEM, cluster shape, preprocess and
    dependent-launch mode on every architecture that builds it.  The dense
    table is unchanged."""

    module = _module()
    assert len(_SLOT_KEYS) == 17 and len(_SM107_ONLY_SLOT_KEYS) == 4
    assert set(module._SLOT_MODULES) == _SLOT_KEYS
    assert set(module._SM107_SLOT_MODULES) == _ALL_SLOT_KEYS
    assert module._SM107_ONLY_SLOT_PROGRAMS == _SM107_ONLY_SLOT_KEYS
    assert set(module._SLOT_PROGRAMS) == _ALL_SLOT_KEYS
    assert not _ALL_SLOT_KEYS & set(module._PROGRAMS)
    assert set(module._SM107_SLOT_TWIN_VARIANTS) == {"claim", "wrap"}
    assert len(module._PROGRAMS) == 84
    assert len(module._SM107_MODULES) == 68
    assert module._SLOT_ARGS == _SLOT_ARGS
    assert module._HAS_INITIAL_FLAGS == 2
    assert module._MAIN_ARGS_SLOTS == module._MAIN_ARGS + module._SLOT_ARGS
    assert (
        module._MAIN_ARGS_CHUNKPAR3_SLOTS
        == module._MAIN_ARGS_CHUNKPAR3 + module._SLOT_ARGS
    )
    assert (
        module._MAIN_ARGS_CHUNKPAR_SLOTS
        == module._MAIN_ARGS_CHUNKPAR + module._SLOT_ARGS
    )
    assert len(module._MAIN_ARGS_SLOTS) == _EXACT_SLOT_ARGS
    assert len(module._MAIN_ARGS_CHUNKPAR3_SLOTS) == _CHUNKPAR3_SLOT_ARGS
    assert len(module._MAIN_ARGS_CHUNKPAR_SLOTS) == _CHUNKPAR_SLOT_ARGS
    source_root = module._source_dir() / module._DEVICE_DIR
    for key in sorted(_ALL_SLOT_KEYS):
        twin = _dense_twin(key)
        assert _slot_key(twin) == key
        dense = module._PROGRAMS[twin]
        arches = _slot_key_arches(key)
        tables = (
            (module._SLOT_MODULES, module._SM107_SLOT_MODULES)
            if key in _SLOT_KEYS
            else (module._SM107_SLOT_MODULES,)
        )
        family = key.split("_")[0]
        main_args = {
            "chunkpar": module._MAIN_ARGS_CHUNKPAR_SLOTS,
            "chunkpar3": module._MAIN_ARGS_CHUNKPAR3_SLOTS,
        }.get(family, module._MAIN_ARGS_SLOTS)
        dense_stem = dense.main.module.rsplit("_", 1)[0]
        for table in tables:
            identity = table[key]
            stem, digest = identity.rsplit("_", 1)
            assert stem == f"{dense_stem}_slots", (key, identity)
            assert re.fullmatch(r"[0-9a-f]{10}", digest), identity
            source = source_root / f"{identity}.cu"
            assert source.is_file(), source
            data = source.read_bytes()
            assert hashlib.sha256(data).hexdigest()[:10] == digest, identity
            text = data.decode("utf-8")
            assert re.search(rf"\bkernel_{stem}\(", text), identity
            assert text.count("#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n") == 1
            assert _kernel_parameters(text, f"kernel_{stem}") == main_args, identity
            assert not re.search(
                r"__restrict__\s+(?:initial|final|checkpoint)_states\b", text
            ), identity
        if key in _SLOT_KEYS:
            assert module._SLOT_MODULES[key] != module._SM107_SLOT_MODULES[key]
        else:
            # An sm_107a-only dense program: its sm_107a unit is its main
            # module on every architecture, and so is its twin's.
            assert dense.main.module == module._SM107_MODULES[twin]
            slot_module = module._SLOT_PROGRAMS[key].main.module
            assert slot_module == module._SM107_SLOT_MODULES[key]
        for arch in arches:
            program = module._arch_program(key, arch)
            dense_program = module._arch_program(twin, arch)
            assert program.state_slots and not dense_program.state_slots
            assert program.family == dense_program.family == family
            assert program.variant == dense_program.variant
            assert program.cluster_dims == dense_program.cluster_dims
            assert program.main_smem_bytes == dense_program.main_smem_bytes, (
                key,
                arch,
            )
            assert program.preprocess == dense_program.preprocess, (key, arch)
            assert program.preprocess_args == dense_program.preprocess_args
            assert program.state_dtype_code == dense_program.state_dtype_code
            assert program.state_dtype_bits == dense_program.state_dtype_bits
            assert module._main_pdl(key, arch) == module._main_pdl(twin, arch)
            table = (
                module._SM107_SLOT_MODULES
                if arch == "sm_107a"
                else module._SLOT_MODULES
            )
            assert module._main_module(key, arch) == table[key]
            assert program.main.module == table[key]
            assert program.main.kernel == f"kernel_{dense_stem}_slots"
            assert program.main_args == main_args
            assert program.host_template == module._SLOT_FAMILY_HOST[family][0]
            assert program.main_args == module._SLOT_FAMILY_HOST[family][1]
            smem = _SLOT_SMEM_BYTES[(family, program.variant)]
            if arch == "sm_107a" and program.variant == "tail":
                smem = _SM107_TAIL_SMEM_BYTES
            assert program.main_smem_bytes == smem, (key, arch)
            assert program.cluster_dims == (
                (2, 1, 1) if program.variant == "tail" else (1, 1, 1)
            )
            # sm_107a: the two-kernel slot programs launch with programmatic
            # dependent launch, the three-phase ones behind the early-trigger
            # preprocess, as their dense twins (the look-back ones behind the
            # preprocess that triggers at CTA exit).
            assert module._main_pdl(key, arch) == (
                arch == "sm_107a" and family != "exactfused"
            )
            if arch == "sm_107a" and family == "chunkpar3":
                assert program.preprocess is module._SM107_SEGMENT_PREPROCESS_TRIGGER
            if arch == "sm_107a" and family == "chunkpar":
                assert program.preprocess is module._SM107_SEGMENT_PREPROCESS


def test_slot_host_templates_bind_the_slot_arguments_without_gpu():
    """The three slot launchers take the dense launcher's arguments plus the
    twelve slot arguments in the loader order (``kargs[56]`` / ``[57]`` /
    ``[59]``), check the three state tensors as pools instead of contiguous
    tensors, the three slot vectors at their steps, the flags and the null
    slot, keep the dense preprocess stage, the dependent-launch statement and
    every placeholder, and render for every slot program and architecture."""

    module = _module()
    source_dir = module._source_dir()
    templates = {
        module._EXACTFUSED_SLOTS_HOST_TEMPLATE: (
            module._MAIN_ARGS_SLOTS,
            _EXACT_SLOT_ARGS,
        ),
        module._CHUNKPAR3_SLOTS_HOST_TEMPLATE: (
            module._MAIN_ARGS_CHUNKPAR3_SLOTS,
            _CHUNKPAR3_SLOT_ARGS,
        ),
        module._CHUNKPAR_SLOTS_HOST_TEMPLATE: (
            module._MAIN_ARGS_CHUNKPAR_SLOTS,
            _CHUNKPAR_SLOT_ARGS,
        ),
    }
    assert set(templates) == {host for host, _ in module._SLOT_FAMILY_HOST.values()}
    texts = {}
    for path, (main_args, kargs) in templates.items():
        template = (source_dir / path).read_text(encoding="utf-8")
        texts[path] = template
        assert _host_prepare_arguments(template, "main") == main_args, path
        assert _host_run_arguments(template, "main") == main_args, path
        for stage_arguments in (_host_prepare_arguments, _host_run_arguments):
            assert stage_arguments(template, "preprocess") == module._PREPROCESS_ARGS
        main_stage = template[template.index("namespace stage_main {") :]
        assert f"void* kargs[{kargs}] = {{}};" in main_stage, path
        assert f"prepared.kargs[{kargs}]" not in main_stage, path
        for index, name in enumerate(module._SLOT_ARGS):
            pointer = name.endswith(("_slots", "_flags"))
            field = ("p_" if pointer else "v_") + name
            entry = f"prepared.kargs[{kargs - 12 + index}] = &prepared.{field};"
            assert entry in main_stage, entry
        assert template.count("  CAKE_SSD_MAIN_PDL_LAUNCH\n") == 1, path
        assert template.count("stream, 5248u));") == 1, path
        assert ("if (preprocess_grid_x != 0)" in template) == (
            path == module._EXACTFUSED_SLOTS_HOST_TEMPLATE
        )
        for side in ("initial", "final", "checkpoint"):
            assert f"check_contiguous(arg_{side}_states" not in main_stage
            assert f"check_contiguous(arg_{side}_state_slots" not in main_stage
            rows = (
                "arg_checkpoint_state_count"
                if side == "checkpoint"
                else f"arg_{side}_state_rows"
            )
            assert re.search(
                rf"check_state_slots\(arg_{side}_state_slots, {rows},"
                rf"\s*arg_sequence_count,\s*arg_{side}_state_slots_step,"
                rf"\s*\"{side}_state_slots\"\);",
                template,
            ), (path, side)
            assert re.search(
                rf"check_state_pool\(arg_{side}_states, arg_{side}_state_slot_stride,"
                rf"\s*{rows}, arg_nheads,\s*\"{side}_states\"\);",
                template,
            ), (path, side)
        assert re.search(
            r"check_state_flags\(arg_initial_state_flags, arg_has_initial,"
            r"\s*arg_sequence_count, \"initial_state_flags\"\);",
            template,
        ), path
        assert "check_dtype(t, DLDataType{kDLUInt, 8, 1}, name);" in template
        assert "arg_null_slot >= -1 && arg_null_slot <= 2147483647LL" in template
        assert "prepared.retained.push_back(arg_initial_state_flags);" in main_stage
        # The slot vectors are checked at their step, not as contiguous
        # tensors; the dense twin's checkpoint token vector stays contiguous.
        assert "t.stride(0) == step" in template
        assert (
            'check_contiguous(arg_checkpoint_token_indices, "checkpoint_token_indices");'
            in main_stage
        )
        assert "Generated by cake.runtime.host_shim" not in template
        assert all(placeholder in template for placeholder in _HOST_PLACEHOLDERS)
    for key in sorted(_ALL_SLOT_KEYS):
        for arch in _slot_key_arches(key):
            program = module._arch_program(key, arch)
            main_pdl = module._main_pdl(key, arch)
            rendered = module._render_host_source(
                texts[program.host_template], key, program, main_pdl=main_pdl
            )
            assert not any(
                placeholder in rendered for placeholder in _HOST_PLACEHOLDERS
            )
            assert "CAKE_SSD_MAIN_PDL_LAUNCH" not in rendered
            assert f"namespace cake_mamba_ssd_combined_host_{key} {{" in rendered
            assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered
            assert f'"{program.main.kernel}"' in rendered
            assert f"stream, {program.main_smem_bytes}u" in rendered
            assert ("LaunchEx(" in rendered) == main_pdl, (key, arch)
            for state_tensor in ("initial_states", "final_states", "checkpoint_states"):
                state_check = (
                    f"check_dtype(arg_{state_tensor}, DLDataType{{"
                    f"{program.state_dtype_code}, {program.state_dtype_bits}, 1}}, "
                    f'"{state_tensor}");'
                )
                assert state_check in rendered, (key, state_check)


def test_dense_host_templates_and_device_units_are_unchanged_without_gpu():
    """The slot programs add launchers, units and table entries only: the
    five dense host templates and the 123 dense device units keep their
    bytes."""

    module = _module()
    source_dir = module._source_dir()
    host_dir = source_dir / module._HOST_TEMPLATE.parent
    for name, digest in _DENSE_HOST_TEMPLATE_SHA256.items():
        data = (host_dir / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
    slot_sources = {
        f"{identity}.cu"
        for table in (module._SLOT_MODULES, module._SM107_SLOT_MODULES)
        for identity in table.values()
    }
    units = sorted(
        path
        for path in (source_dir / module._DEVICE_DIR).glob("*.cu")
        if path.name not in slot_sources
    )
    assert len(units) == _DENSE_DEVICE_UNITS
    lines = "".join(
        f"{path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for path in units
    )
    assert hashlib.sha256(lines.encode()).hexdigest() == _DENSE_DEVICE_UNITS_SHA256


# ---------------------------------------------------------------------------
# CPU: the route of every call form to a slot program.

_SLOT_FORMS = ("cu_seqlens", "derived", "trusted", "publishing")
# Anchor calls beyond the shape grid, (lengths, nheads, ngroups): vLLM's
# 1024 one-token prefills, 8 x 100 tokens, the ``_noring`` window (2 and 3 x
# 64 tokens at 128 heads), the tail pair program's long items (4 x 2048; the
# sm_107a varlen pair call of 2 x 1024), one 32768-token sequence at 8 heads
# and a two-sequence call at 4 heads (the chunk-parallel calls), and the
# sm_107a one-chunk calls of 2048 items (claim / initial-state bulk), and the
# sm_107a claim and wrap cells of 256 x 128 tokens at 16 heads and 8 x 2048
# tokens at 32 heads.
_ANCHOR_CELLS = (
    ((1,) * 1024, 128, 8),
    ((100,) * 8, 128, 8),
    ((64,) * 2, 128, 8),
    ((64,) * 3, 128, 8),
    ((2048,) * 4, 128, 8),
    ((1024,) * 2, 128, 8),
    ((32768,), 8, 8),
    ((1500, 2600), 4, 4),
    ((100,) * 16, 128, 8),
    ((1,) * 2048, 1, 1),
    ((128,) * 256, 16, 2),
    ((2048,) * 8, 32, 4),
)
# The base slot program of the sm_107a forms the slot route leaves:
# initial-state bulk and one-CTA-cluster calls run the ring program, one-CTA
# pair calls the 2-CTA pair program.  The claim and wrap programs run their
# own twins.
_BASE_VARIANTS = {
    "ibulk": ("",),
    "unit": ("",),
    "tailone": ("tail",),
}


def _route_cells():
    """``(lengths, nheads, ngroups, has_z, state_dtype, overrides)`` of the
    packed-varlen grid and the anchors; the chunk-parallel overrides only
    replace the family, so they run on two small grid cells."""

    for items, tokens, nheads, mode_varlen, has_z, state_dtype in _route_shapes():
        if not mode_varlen or (nheads != 128 and tokens > 384):
            continue
        overrides = (None,)
        if not has_z and nheads == 128 and (items, tokens) in ((2, 64), (2, 384)):
            overrides = (None, "always", "three_phase", "never")
        yield (tokens,) * items, nheads, 8, has_z, state_dtype, overrides
    for lengths, nheads, ngroups in _ANCHOR_CELLS:
        for has_z in (False, True):
            for state_dtype in _STATE_DTYPES:
                yield lengths, nheads, ngroups, has_z, state_dtype, (None,)


def _route_call(lengths, nheads, ngroups, has_z, form):
    if len(set(lengths)) == 1 and ngroups == 8 and form != "publishing":
        return _cpu_route_call(len(lengths), lengths[0], nheads, True, has_z, form)
    if len(set(lengths)) == 1 and ngroups == 8:
        tensors, kwargs = _cpu_route_call(
            len(lengths), lengths[0], nheads, True, has_z, "trusted"
        )
        return tensors, {**kwargs, "update_seq_chunk_cumsum": True}
    return _cpu_varlen_call(lengths, nheads, ngroups, has_z, form)


def test_slot_calls_route_to_the_slot_twins_without_gpu(monkeypatch):
    """Every packed-varlen call form on every architecture (its SM count),
    state dtype, ``z`` choice and chunk-parallel override runs a state-slot
    program of the call's state dtype when it passes slot vectors: the twin
    of the program the same call runs without them where that program has a
    twin (a varlen ``exactfused`` / ``exactpre`` / ``chunkpar`` /
    ``chunkpar3`` program of the ring, ``_noring`` or ``_tail`` schedule),
    else the twin the slot route maps it to -- the retained start program to
    ``exactpre``, the SM100 metadata and SM103 / SM107 one-launch programs to
    the two-kernel program, the sm_107a initial-state bulk, one-CTA pair and
    one-CTA-cluster forms to their base program, the look-back program to its
    twin where the route serves it (:func:`_lookback_slot_family`), else to
    the three-phase twin.  The sm_107a claim and wrap programs (bf16 and f32
    states) run their own twins: a slot call runs one exactly when the same
    call without slot vectors runs that program.  The same call with strided
    slot vectors (columns of a slot table), flags, a checkpoint pool (the
    state pool itself) and ``null_slot_id=0`` runs the same program.  Each
    architecture reaches every slot program it serves (all seventeen with the
    look-back twins shipped, and the four claim and wrap twins on
    sm_107a)."""

    module = _module()
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)
    for arch in _ARCHES:
        sm_count = _SM_COUNTS[arch]
        reached = set()
        for lengths, nheads, ngroups, has_z, state_dtype, overrides in _route_cells():
            runners = {}
            for slots in (True, False):
                # Keep no launch records: nothing reads them, and each
                # runner's launcher stays on the monkeypatch undo stack until
                # teardown, so a list would keep every launch's CPU tensors
                # of the whole grid alive (hundreds of GiB).
                runners[slots] = _cpu_forwarding_runner(
                    module,
                    monkeypatch,
                    deque(maxlen=0),
                    arch=arch,
                    nheads=nheads,
                    ngroups=ngroups,
                    state_dtype=state_dtype,
                    has_varlen=True,
                    has_z=has_z,
                    has_initial_states=slots,
                )
            monkeypatch.setattr(module, "_sm_count", lambda _, n=sm_count: n)
            pool = torch.empty((2, nheads, 64, 128), dtype=state_dtype)
            vector = torch.full((len(lengths),), -1, dtype=torch.int32)
            # vLLM-shaped vectors: columns of a slot table, flags and a
            # checkpoint pool that is the state pool.
            table = torch.full((len(lengths), 3), -1, dtype=torch.int32)
            vllm_states = dict(
                initial_states=pool,
                initial_state_slots=table[:, 0],
                initial_state_flags=torch.ones(len(lengths), dtype=torch.bool),
                final_states=pool,
                final_state_slots=table[:, 0],
                checkpoint_token_indices=torch.full_like(vector, -1),
                checkpoint_state_slots=table[:, 2],
                checkpoint_states=pool,
                null_slot_id=0,
            )
            state_key = module._STATE_DTYPE_KEYS[state_dtype]
            for override in overrides:
                if override is None:
                    monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
                else:
                    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override)
                for form in _SLOT_FORMS:
                    cell = (arch, override, form, lengths[:4], len(lengths))
                    cell += (nheads, has_z, state_key)
                    tensors, kwargs = _route_call(lengths, nheads, ngroups, has_z, form)
                    runners[True].run(
                        *tensors,
                        **kwargs,
                        initial_states=pool,
                        initial_state_slots=vector,
                        final_states=pool,
                        final_state_slots=vector,
                    )
                    tensors, kwargs = _route_call(lengths, nheads, ngroups, has_z, form)
                    runners[False].run(*tensors, **kwargs, return_final_states=False)
                    name = runners[True].last_program_name
                    dense_name = runners[False].last_program_name
                    tensors, kwargs = _route_call(lengths, nheads, ngroups, has_z, form)
                    runners[True].run(*tensors, **kwargs, **vllm_states)
                    assert runners[True].last_program_name == name, cell
                    assert name in module._SLOT_PROGRAMS, (name, cell)
                    program = module._SLOT_PROGRAMS[name]
                    assert program.state_slots, cell
                    assert program.family in _SLOT_FAMILIES, (name, cell)
                    family, name_state, mode, _variant = name.split("_")
                    assert (name_state, mode) == (state_key, "varlen"), (name, cell)
                    assert program.variant in ("", "noring", "tail") or (
                        arch == "sm_107a"
                        and program.variant in module._SM107_SLOT_TWIN_VARIANTS
                    ), (name, cell)
                    assert (name in _SM107_ONLY_SLOT_KEYS) == (
                        dense_name.endswith(("_claim", "_wrap"))
                    ), (name, dense_name, cell)
                    dense_family, _, dense_mode, *rest = dense_name.split("_")
                    dense_variant = rest[0] if rest else ""
                    lookback = _lookback_slot_family(module, state_dtype, arch)
                    expected = None
                    if dense_family == "chunkpar":
                        expected = (f"{lookback}_{state_key}_varlen_slots",)
                    elif (
                        dense_mode == "varlen"
                        and dense_family in _SLOT_FAMILIES
                        and dense_variant in ("", "noring", "tail", "claim", "wrap")
                    ):
                        expected = (_slot_key(dense_name),)
                    elif dense_family == "exact":
                        expected = (f"exactpre_{state_key}_varlen_slots",)
                    elif dense_family in ("exactmeta", "exactone"):
                        expected = tuple(
                            f"exactpre_{state_key}_varlen_{variant}"
                            for variant in ("slots", "tailslots")
                        )
                    elif dense_mode == "varlen" and dense_variant in _BASE_VARIANTS:
                        expected = tuple(
                            _slot_key(
                                module._program_name(
                                    dense_family, state_dtype, True, base
                                )
                            )
                            for base in _BASE_VARIANTS[dense_variant]
                        )
                    if expected is not None:
                        assert name in expected, (name, dense_name, cell)
                    if override == "always":
                        assert family == lookback, (name, cell)
                    if override == "three_phase":
                        assert family == "chunkpar3", (name, cell)
                    reached.add(name)
        served = _served_slot_keys(module, arch)
        assert reached == served, (arch, sorted(served ^ reached))


def test_lookback_switch_routes_to_the_three_phase_twin_without_gpu(monkeypatch):
    """A slot call the chunk-parallel rule sends to the look-back program runs
    the look-back slot twin only with ``_LOOKBACK_STATE_SLOTS`` on and that
    twin in the slot table of the architecture (``_SLOT_PROGRAMS``, and
    ``_SM107_SLOT_MODULES`` on sm_107a); otherwise it runs the three-phase
    slot twin (the earlier route, kept for A/B timing and for tables without
    the look-back twin).  Every other slot call keeps its program.  The
    choice is read per new call shape.  Where the tables do not ship the
    twin, a stand-in binding exercises the served route (nothing is built or
    launched)."""

    module = _module()
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)
    lengths, nheads = (1500, 2600), 4
    key = "chunkpar_f32_varlen_slots"
    stand_in = "mamba_ssd_chunk_parallel_lookback_f32_varlen_slots_0123456789"
    shipped = (dict(module._SLOT_PROGRAMS), dict(module._SM107_SLOT_MODULES))
    present = (
        {**shipped[0], key: shipped[0].get(key) or module._program(key, stand_in)},
        {**shipped[1], key: shipped[1].get(key, stand_in)},
    )
    absent = tuple(
        {name: value for name, value in table.items() if name != key}
        for table in shipped
    )
    sm107_absent = (present[0], absent[1])

    def route(arch, override, lookback, tables):
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override)
        monkeypatch.setattr(module, _LOOKBACK_SWITCH, lookback)
        monkeypatch.setattr(module, "_SLOT_PROGRAMS", tables[0])
        monkeypatch.setattr(module, "_SM107_SLOT_MODULES", tables[1])
        runner = _cpu_forwarding_runner(
            module,
            monkeypatch,
            [],
            arch=arch,
            nheads=nheads,
            ngroups=4,
            state_dtype=torch.float32,
            has_varlen=True,
            has_z=False,
            has_initial_states=True,
        )
        monkeypatch.setattr(module, "_sm_count", lambda _, n=_SM_COUNTS[arch]: n)
        pool = torch.empty((2, nheads, 64, 128), dtype=torch.float32)
        vector = torch.full((2,), -1, dtype=torch.int32)
        tensors, kwargs = _cpu_varlen_call(lengths, nheads, 4, False, "cu_seqlens")
        runner.run(
            *tensors,
            **kwargs,
            initial_states=pool,
            initial_state_slots=vector,
            final_states=pool,
            final_state_slots=vector,
        )
        return runner.last_program_name

    for arch in _ARCHES:
        for override, lookback, tables, family in (
            ("always", True, present, "chunkpar"),
            ("always", False, present, "chunkpar3"),
            ("always", True, absent, "chunkpar3"),
            (
                "always",
                True,
                sm107_absent,
                "chunkpar3" if arch == "sm_107a" else "chunkpar",
            ),
            ("three_phase", True, present, "chunkpar3"),
            ("three_phase", False, present, "chunkpar3"),
            ("never", False, present, None),
        ):
            name = route(arch, override, lookback, tables)
            cell = (arch, override, lookback, tables is present)
            if family is None:
                assert name.split("_")[0] not in ("chunkpar", "chunkpar3"), name
            else:
                assert name == f"{family}_f32_varlen_slots", (cell, name)
        # The shipped tables: the effective choice of switch and presence.
        for lookback in (True, False):
            name = route(arch, "always", lookback, shipped)
            monkeypatch.setattr(module, _LOOKBACK_SWITCH, lookback)
            family = _lookback_slot_family(module, torch.float32, arch)
            assert name == f"{family}_f32_varlen_slots", (arch, lookback, name)


def test_gpu_twin_cells_route_as_the_gpu_tests_expect_without_gpu(monkeypatch):
    """The routes the GPU twin tests rely on, checked on CPU: on sm_107a
    (208, 212 and 216 SMs) the dense call of every sm_107a cell runs the
    claim or wrap program and its slot call that program's twin, while the
    same cells with an f16 state, and every cell on sm_100a and sm_103a, run
    no claim or wrap twin; the twin calls (with the sm_107a cells on sm_107a)
    reach every slot program the architecture serves."""

    module = _module()
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)

    def route(call, arch, sm_count, slots):
        _set_override(monkeypatch, call)
        ngroups = call.ngroups or min(call.nheads, 8)
        runner = _cpu_forwarding_runner(
            module,
            monkeypatch,
            deque(maxlen=0),
            arch=arch,
            nheads=call.nheads,
            ngroups=ngroups,
            state_dtype=call.state_dtype,
            has_varlen=True,
            has_z=call.has_z,
            has_initial_states=slots,
        )
        monkeypatch.setattr(module, "_sm_count", lambda _, n=sm_count: n)
        tensors, kwargs = _route_call(
            call.lengths, call.nheads, ngroups, call.has_z, call.form
        )
        if slots:
            pool = torch.empty((2, call.nheads, 64, 128), dtype=call.state_dtype)
            vector = torch.full((len(call.lengths),), -1, dtype=torch.int32)
            runner.run(
                *tensors,
                **kwargs,
                initial_states=pool,
                initial_state_slots=vector,
                final_states=pool,
                final_state_slots=vector,
            )
        else:
            runner.run(*tensors, **kwargs, return_final_states=False)
        return runner.last_program_name

    for arch in _ARCHES:
        sm107 = arch == "sm_107a"
        for sm_count in (208, 212, 216) if sm107 else (_SM_COUNTS[arch],):
            for call in _SM107_TWIN_CALLS:
                name = route(call, arch, sm_count, True)
                if sm107:
                    _sm107_twin_route(call, name, route(call, arch, sm_count, False))
                else:
                    assert name not in _SM107_ONLY_SLOT_KEYS, (arch, str(call), name)
                f16 = replace(call, state_dtype=torch.float16)
                name = route(f16, arch, sm_count, True)
                assert name not in _SM107_ONLY_SLOT_KEYS, (arch, str(f16), name)
            if sm_count == 208:
                continue
            calls = (*_TWIN_CALLS, *(_SM107_TWIN_CALLS if sm107 else ()))
            reached = {route(call, arch, sm_count, True) for call in calls}
            served = _served_slot_keys(module, arch)
            assert reached == served, (arch, sm_count, sorted(served ^ reached))


# ---------------------------------------------------------------------------
# CPU: bound values, validation and preparation.


def _slot_runner(module, monkeypatch, calls, **constructor):
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)
    return _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=constructor.pop("nheads", 8),
        ngroups=8,
        has_varlen=constructor.pop("has_varlen", True),
        has_initial_states=constructor.pop("has_initial_states", True),
        **constructor,
    )


def _launch_length(module, name, launch):
    program = module._SLOT_PROGRAMS[name]
    return len(
        module._sequence_arguments(
            program.main_args,
            launch["preprocess"],
            launch["preprocess_grid"],
            launch["main"],
            launch["main_grid"],
            launch["cuda_stream"],
            preprocess_args=program.preprocess_args,
        )
    )


def test_slot_call_binds_pools_and_slot_vectors_without_gpu(monkeypatch):
    """A pool side binds the pool as given (row-strided, never packed), the
    caller's slot vector, its row stride in elements as a Python int (one
    row for a one-row pool) and its row count; a dense side of a slot call
    binds its contiguous states with the workspace's identity slots
    ``arange(num_seqs)``, the dense row stride and ``num_seqs`` rows; an
    absent side binds the dummies with stride and rows 0.  Without flags,
    checkpoints or ``null_slot_id`` the call binds the one-byte flag dummy
    with ``has_initial`` 0 / 1, one row as the checkpoint row stride, steps
    of 1 and null slot -1.  The launch takes the 56 (exact scan), 57
    (three-phase) or 59 (look-back) main arguments; a call without slot
    vectors binds none of them."""

    module = _module()
    calls = []
    nheads, lengths = 8, (96, 160)
    row = nheads * _ROW
    runner = _slot_runner(module, monkeypatch, calls, nheads=nheads)
    tensors, kwargs = _cpu_varlen_call(lengths, nheads, 8, False, "cu_seqlens")
    pool = _cpu_pool(5, nheads, torch.bfloat16)
    initial_slots = torch.tensor([3, -1], dtype=torch.int32)
    final_slots = torch.tensor([3, 0], dtype=torch.int32)

    # Both sides one pool, updated in place.
    out, final = runner.run(
        *tensors,
        **kwargs,
        initial_states=pool,
        initial_state_slots=initial_slots,
        final_states=pool,
        final_state_slots=final_slots,
    )
    name, launch = calls[-1]
    main = launch["main"]
    assert name == runner.last_program_name == "exactfused_bf16_varlen_slots"
    assert final is pool and tuple(out.shape) == tuple(tensors[0].shape)
    assert main["initial_states"] is pool and main["final_states"] is pool
    assert main["initial_state_slots"] is initial_slots
    assert main["final_state_slots"] is final_slots
    for side in ("initial", "final"):
        stride = main[f"{side}_state_slot_stride"]
        assert type(stride) is int and stride == pool.stride(0) == row + 64
        assert main[f"{side}_state_rows"] == 5
    assert main["has_initial"] == 1 and main["write_final_states"] == 1
    assert main["checkpoint_state_count"] == 0
    assert main["checkpoint_state_slot_stride"] == row
    flags = main["initial_state_flags"]
    assert flags.dtype == torch.uint8 and flags.numel() == 1
    for side in ("initial", "final", "checkpoint"):
        assert main[f"{side}_state_slots_step"] == 1
    assert main["null_slot"] == -1
    assert set(main) == set(module._SLOT_PROGRAMS[name].main_args)
    assert _launch_length(module, name, launch) == 27 + 3 + _EXACT_SLOT_ARGS + 3 + 1
    assert not any(
        key[0] in ("initial_states", "final_states")
        for key in runner._workspace["contiguous"]
    )

    # A one-row pool: its row stride is one row whatever stride(0) says.
    one_row = torch.empty(row, dtype=torch.bfloat16).as_strided(
        (1, nheads, 64, 128), (12345, _ROW, 128, 1)
    )
    runner.run(
        *tensors,
        **kwargs,
        initial_states=one_row,
        initial_state_slots=torch.tensor([0, -1], dtype=torch.int32),
        return_final_states=False,
    )
    main = calls[-1][1]["main"]
    assert main["initial_states"] is one_row
    assert main["initial_state_slot_stride"] == row
    assert main["initial_state_rows"] == 1
    # ... and the absent final side: the dummies, stride and rows 0.
    assert main["final_states"].numel() == 1
    assert main["final_states"].dtype == torch.bfloat16
    assert main["final_state_slots"].numel() == 1
    assert main["final_state_slots"].dtype == torch.int32
    assert main["final_state_slot_stride"] == 0 and main["final_state_rows"] == 0
    assert main["write_final_states"] == 0

    # A dense initial side with a final pool.
    dense_initial = torch.empty((2, nheads, 64, 128), dtype=torch.bfloat16)
    runner.run(
        *tensors,
        **kwargs,
        initial_states=dense_initial,
        final_states=pool,
        final_state_slots=final_slots,
    )
    main = calls[-1][1]["main"]
    identity = runner._workspace["identity_slots"]
    assert torch.equal(identity, torch.arange(2, dtype=torch.int32))
    assert main["initial_states"] is dense_initial
    assert main["initial_state_slots"] is identity
    assert main["initial_state_slot_stride"] == row
    assert main["initial_state_rows"] == 2
    assert main["final_states"] is pool and main["final_state_rows"] == 5

    # An initial pool with allocated dense final states.
    _, final = runner.run(
        *tensors, **kwargs, initial_states=pool, initial_state_slots=initial_slots
    )
    main = calls[-1][1]["main"]
    assert tuple(final.shape) == (2, nheads, 64, 128)
    assert main["final_states"] is final
    assert main["final_state_slots"] is runner._workspace["identity_slots"]
    assert main["final_state_slot_stride"] == row
    assert main["final_state_rows"] == 2

    # An absent initial side with a final pool.
    no_initial = _slot_runner(
        module, monkeypatch, calls, nheads=nheads, has_initial_states=False
    )
    _, final = no_initial.run(
        *tensors, **kwargs, final_states=pool, final_state_slots=final_slots
    )
    main = calls[-1][1]["main"]
    assert final is pool
    assert main["initial_states"].numel() == 1 and main["has_initial"] == 0
    assert main["initial_state_slots"].numel() == 1
    assert main["initial_state_slot_stride"] == 0
    assert main["initial_state_rows"] == 0

    # The derived table runs the two-kernel program; the three-phase program
    # binds 57 main arguments, the look-back program 59 (where the route
    # serves its slot twin, else the call runs the three-phase twin).
    tensors_derived, kwargs_derived = _cpu_varlen_call(
        lengths, nheads, 8, False, "derived"
    )
    runner.run(
        *tensors_derived,
        **kwargs_derived,
        initial_states=pool,
        initial_state_slots=initial_slots,
        final_states=pool,
        final_state_slots=final_slots,
    )
    name, launch = calls[-1]
    assert name == "exactpre_bf16_varlen_slots"
    assert launch["preprocess_grid"][0] >= 1
    assert launch["preprocess"]["write_seq_chunk_cumsum"] == 1
    assert _launch_length(module, name, launch) == 27 + 3 + _EXACT_SLOT_ARGS + 3 + 1
    slot_args = {"chunkpar3": _CHUNKPAR3_SLOT_ARGS, "chunkpar": _CHUNKPAR_SLOT_ARGS}
    lookback = _lookback_slot_family(module, torch.bfloat16, "sm_103a")
    for override, family in (("three_phase", "chunkpar3"), ("always", lookback)):
        count = slot_args[family]
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override)
        runner.run(
            *tensors,
            **kwargs,
            initial_states=pool,
            initial_state_slots=initial_slots,
            final_states=pool,
            final_state_slots=final_slots,
        )
        name, launch = calls[-1]
        assert name == f"{family}_bf16_varlen_slots", override
        assert set(launch["main"]) == set(module._SLOT_PROGRAMS[name].main_args)
        assert _launch_length(module, name, launch) == 27 + 3 + count + 3 + 1
    monkeypatch.delenv(_CHUNK_PARALLEL_ENV)

    # A call without slot vectors binds none of the slot arguments.
    runner.run(*tensors, **kwargs, initial_states=dense_initial)
    name, launch = calls[-1]
    assert name == "exactfused_bf16_varlen"
    assert not set(module._SLOT_ARGS) & set(launch["main"])


def test_vllm_shaped_slot_call_binds_everything_as_given_without_gpu(monkeypatch):
    """A vLLM-shaped call: one row-strided pool for the initial, final and
    checkpoint states, column 0 of a ``[num_seqs, 1 + num_spec]`` slot table
    as both slot vectors, a column of a block table as the checkpoint slots
    (0 the null block), bool flags and ``null_slot_id=0``.  Everything is
    bound as given: the three state tensors are the pool, the slot vectors
    the caller's views with their element steps, the flags the bool tensor's
    uint8 view (the same bytes) with ``has_initial`` 2, the pool's row count
    as ``checkpoint_state_count`` and its row stride as
    ``checkpoint_state_slot_stride``, null slot 0, and no packed copy is
    made.  A one-entry vector binds step 1 whatever its stride."""

    module = _module()
    calls = []
    nheads, lengths = 8, (96, 160, 40)
    runner = _slot_runner(module, monkeypatch, calls, nheads=nheads)
    tensors, kwargs = _cpu_varlen_call(lengths, nheads, 8, False, "cu_seqlens")
    pool = _cpu_pool(9, nheads, torch.bfloat16)
    blocks = torch.tensor(
        [[6, 4, 0, 0, 0], [8, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.int32
    )
    checkpoint_slots = blocks[:, 1]
    tokens = torch.tensor([50, -1, 276], dtype=torch.int32)
    flags = torch.tensor([True, False, True])
    for num_spec in (0, 1, 3):
        table = torch.zeros((3, 1 + num_spec), dtype=torch.int32)
        table[:, 0] = torch.tensor([3, 5, 2], dtype=torch.int32)
        slots = table[:, 0]
        runner.run(
            *tensors,
            **kwargs,
            initial_states=pool,
            initial_state_slots=slots,
            initial_state_flags=flags,
            final_states=pool,
            final_state_slots=slots,
            checkpoint_token_indices=tokens,
            checkpoint_state_slots=checkpoint_slots,
            checkpoint_states=pool,
            null_slot_id=0,
        )
        name, launch = calls[-1]
        main = launch["main"]
        assert name in module._SLOT_PROGRAMS, (name, num_spec)
        for state in ("initial_states", "final_states", "checkpoint_states"):
            assert main[state] is pool, state
        assert main["initial_state_slots"] is slots
        assert main["final_state_slots"] is slots
        assert main["checkpoint_state_slots"] is checkpoint_slots
        assert main["initial_state_slots_step"] == 1 + num_spec
        assert main["final_state_slots_step"] == 1 + num_spec
        assert main["checkpoint_state_slots_step"] == 5
        bound_flags = main["initial_state_flags"]
        assert bound_flags.dtype == torch.uint8
        assert bound_flags.data_ptr() == flags.data_ptr()
        assert torch.equal(bound_flags, flags.to(torch.uint8))
        assert main["has_initial"] == module._HAS_INITIAL_FLAGS
        assert main["checkpoint_state_count"] == 9
        assert launch["preprocess"]["checkpoint_state_count"] == 9
        assert main["checkpoint_state_slot_stride"] == pool.stride(0)
        assert type(main["checkpoint_state_slot_stride"]) is int
        assert main["checkpoint_token_indices"] is tokens
        assert launch["preprocess"]["checkpoint_token_indices"] is tokens
        assert main["null_slot"] == 0
        assert main["initial_state_rows"] == main["final_state_rows"] == 9
        assert not runner._workspace["contiguous"], runner._workspace["contiguous"]

    # uint8 flags bind the same bytes too.
    byte_flags = flags.to(torch.uint8)
    runner.run(
        *tensors,
        **kwargs,
        initial_states=pool,
        initial_state_slots=table[:, 0],
        initial_state_flags=byte_flags,
        final_states=pool,
        final_state_slots=table[:, 0],
    )
    assert calls[-1][1]["main"]["initial_state_flags"].data_ptr() == (
        byte_flags.data_ptr()
    )

    # One sequence: a stride-0 vector binds step 1.
    single, single_kwargs = _cpu_varlen_call((96,), nheads, 8, False, "cu_seqlens")
    one = torch.tensor([3], dtype=torch.int32).as_strided((1,), (0,))
    runner.run(
        *single,
        **single_kwargs,
        initial_states=pool,
        initial_state_slots=one,
        final_states=pool,
        final_state_slots=one,
    )
    main = calls[-1][1]["main"]
    assert main["initial_state_slots"] is one
    assert main["initial_state_slots_step"] == main["final_state_slots_step"] == 1


def _pool_from(storage, rows, nheads, row_stride, offset=0):
    return storage.as_strided(
        (rows, nheads, 64, 128), (row_stride, _ROW, 128, 1), offset
    )


def test_occupied_overlaps_counts_rows_not_gaps_without_gpu():
    """``_occupied_overlaps`` of a row-strided tensor (contiguous rows, a
    positive row stride of at least one row) meets only its rows: a tensor on
    or straddling a row overlaps, one in the gap rows, before the first row or
    after the last does not; the other tensor's own gaps still count.  Any
    other geometry falls back to the gap-inclusive ``_storage_overlaps``."""

    module = _module()
    storage = torch.empty((9, 2, 4), dtype=torch.float32)
    flat = storage.view(-1)
    rows = storage[1::4]  # rows 1 and 5 of ``storage``
    for other in (
        storage[1],
        storage[5],
        storage[:2],
        storage[4:6],
        storage[5:],
        flat[15:17],  # the last element of row 1 and the next one
        flat[39:41],  # the element before row 5 and its first one
        storage[::6],  # rows 0 and 6: its extent covers row 1
    ):
        assert module._occupied_overlaps(rows, other)
    for other in (storage[0], storage[2:5], storage[2:5:2], storage[6:], flat[:8]):
        assert not module._occupied_overlaps(rows, other)
    # One row: only that row counts.
    assert module._occupied_overlaps(storage[3:4], storage[3])
    assert not module._occupied_overlaps(storage[3:4], storage[4:])
    # Other geometries count their whole extent, gap rows included.
    for fallback in (
        rows.transpose(1, 2),  # rows not contiguous
        flat.as_strided((3, 8), (4, 1)),  # rows overlap each other
        flat.as_strided((2, 8), (0, 1)),  # zero row stride
    ):
        for other in (storage[2:5], storage[0], storage[6:]):
            expected = module._storage_overlaps(fallback, other)
            assert module._occupied_overlaps(fallback, other) == expected
    assert module._occupied_overlaps(rows.transpose(1, 2), storage[2:5])


def test_slot_call_validation_without_gpu(monkeypatch):
    """Slot vectors need packed varlen mode and their state tensor; they are
    int32 vectors of one slot per sequence on the device of ``x`` with a
    positive stride.  A pool is ``[rows, nheads, 64, 128]`` of the state
    dtype with dense rows, a row stride of at least one row and a 16-byte
    aligned base and row stride.  A final pool must not overlap a dense
    initial_states (for a strided one, its rows; its gap rows are free); it
    may be the initial pool when both slot vectors are given.  Flags need an
    initial slot vector and are contiguous bool / uint8 ``[num_seqs]``;
    ``null_slot_id`` is an int in ``[-1, 2**31 - 1]`` and, unless -1, needs a
    slot vector on every state side the call binds (a dense initial side or
    returned dense final states are refused; an absent side is fine).  A slot
    call's checkpoint states are a pool (strided slot vector allowed) that may
    only be the identical initial or final pool tensor, never a partial
    overlap, also against a dense initial side (for a strided one, its rows);
    a call without slot vectors refuses a row-strided checkpoint pool and a
    strided checkpoint slot vector, and ``prepare`` refuses its checkpoint
    outputs."""

    module = _module()
    calls = []
    nheads, lengths = 8, (96, 160)
    row = nheads * _ROW
    runner = _slot_runner(module, monkeypatch, calls, nheads=nheads)
    tensors, kwargs = _cpu_varlen_call(lengths, nheads, 8, False, "cu_seqlens")
    pool = _cpu_pool(4, nheads, torch.bfloat16)
    slots = torch.tensor([1, 0], dtype=torch.int32)

    def run_initial(initial_states=pool, initial_state_slots=slots, **extra):
        return runner.run(
            *tensors,
            **kwargs,
            initial_states=initial_states,
            initial_state_slots=initial_state_slots,
            **extra,
        )

    # Batched mode.
    batched = _slot_runner(module, monkeypatch, calls, nheads=nheads, has_varlen=False)
    x = torch.empty((2, 128, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((2, 128, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((2, 128, 8, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="require packed varlen mode"):
        batched.run(
            x,
            dt,
            A,
            B,
            torch.empty_like(B),
            initial_states=pool,
            initial_state_slots=slots,
        )
    with pytest.raises(ValueError, match="require packed varlen mode"):
        batched.run(
            x,
            dt,
            A,
            B,
            torch.empty_like(B),
            initial_states=torch.empty((2, nheads, 64, 128), dtype=torch.bfloat16),
            final_states=pool,
            final_state_slots=slots,
        )

    # Slot vectors.
    for bad in (
        slots.to(torch.int64),
        torch.tensor([1], dtype=torch.int32).as_strided((2,), (0,)),
        torch.tensor([1, 0, 2], dtype=torch.int32),
        slots.reshape(1, 2),
    ):
        with pytest.raises(ValueError, match="initial_state_slots"):
            run_initial(initial_state_slots=bad)
        with pytest.raises(ValueError, match="final_state_slots"):
            run_initial(
                final_states=_cpu_pool(4, nheads, torch.bfloat16),
                final_state_slots=bad,
            )

    # Pools.
    flat = torch.empty(4 * row + 64, dtype=torch.bfloat16)
    for bad in (
        torch.empty((4, nheads, _ROW), dtype=torch.bfloat16),
        torch.empty((4, nheads, 64, 64), dtype=torch.bfloat16),
        torch.empty((4, nheads + 1, 64, 128), dtype=torch.bfloat16),
        torch.empty((0, nheads, 64, 128), dtype=torch.bfloat16),
        torch.empty((4, nheads, 128, 64), dtype=torch.bfloat16).transpose(2, 3),
        _pool_from(flat, 4, nheads, row - 128),
        _pool_from(flat, 4, nheads, row, offset=1),
        _pool_from(
            torch.empty(4 * (row + 1), dtype=torch.bfloat16), 4, nheads, row + 1
        ),
        _cpu_pool(4, nheads, torch.float16),
    ):
        with pytest.raises(ValueError, match="initial_states"):
            run_initial(initial_states=bad)
        if bad.dtype == torch.bfloat16:
            with pytest.raises(ValueError, match="final_states"):
                run_initial(final_states=bad, final_state_slots=slots)

    # A slot vector without its state tensor or with nothing to return.
    no_initial = _slot_runner(
        module, monkeypatch, calls, nheads=nheads, has_initial_states=False
    )
    with pytest.raises(ValueError, match="initial_state_slots"):
        no_initial.run(*tensors, **kwargs, initial_state_slots=slots)
    with pytest.raises(ValueError, match="final_state_slots requires final_states"):
        run_initial(final_state_slots=slots)
    with pytest.raises(ValueError, match="return_final_states"):
        run_initial(
            final_states=_cpu_pool(4, nheads, torch.bfloat16),
            final_state_slots=slots,
            return_final_states=False,
        )

    # Overlaps.
    storage = torch.empty((5, nheads, 64, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must not overlap initial_states"):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=storage[:2],
            final_states=storage[1:4],
            final_state_slots=slots,
        )
    # A strided dense initial side (rows 0 and 4 of ``storage``, packed before
    # the launch) refuses final states, pool or dense, on one of its rows and
    # accepts a final pool in the gap rows between them.
    strided_initial = storage[::4]
    for final_state_slots in (slots, None):
        with pytest.raises(ValueError, match="must not overlap initial_states"):
            runner.run(
                *tensors,
                **kwargs,
                initial_states=strided_initial,
                final_states=storage[:2],
                final_state_slots=final_state_slots,
            )
    gap = storage[2:4]
    _, final = runner.run(
        *tensors,
        **kwargs,
        initial_states=strided_initial,
        final_states=gap,
        final_state_slots=slots,
    )
    assert final is gap
    checkpoint_storage = torch.empty((5, nheads, 64, 128), dtype=torch.bfloat16)
    tokens = torch.tensor([95, 255], dtype=torch.int32)
    with pytest.raises(
        ValueError, match="checkpoint_states must not overlap final_states"
    ):
        run_initial(
            final_states=checkpoint_storage[1:4],
            final_state_slots=slots,
            checkpoint_token_indices=tokens,
            checkpoint_state_slots=torch.tensor([0, 1], dtype=torch.int32),
            checkpoint_states=checkpoint_storage[:2],
        )
    _, final = run_initial(final_states=pool, final_state_slots=slots)
    assert final is pool

    # Strided slot vectors (a column of a slot table) are accepted.
    strided = torch.tensor([1, 7, 0, 7], dtype=torch.int32)[::2]
    _, final = run_initial(
        initial_state_slots=strided, final_states=pool, final_state_slots=strided
    )
    assert final is pool

    # Flags.
    flags = torch.tensor([True, False])
    dense_initial = torch.empty((2, nheads, 64, 128), dtype=torch.bfloat16)
    with pytest.raises(
        ValueError, match="initial_state_flags requires initial_state_slots"
    ):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=dense_initial,
            final_states=_cpu_pool(4, nheads, torch.bfloat16),
            final_state_slots=slots,
            initial_state_flags=flags,
        )
    for bad in (
        flags.int(),
        flags.float(),
        torch.ones(3, dtype=torch.bool),
        flags.reshape(1, 2),
        torch.ones(4, dtype=torch.bool)[::2],
    ):
        with pytest.raises(ValueError, match="initial_state_flags"):
            run_initial(initial_state_flags=bad)
    for good in (flags, flags.to(torch.uint8)):
        run_initial(initial_state_flags=good)

    # The null slot.
    for bad in (-2, 2**31, True, 1.0, None):
        with pytest.raises(ValueError, match="null_slot_id"):
            run_initial(null_slot_id=bad)
    with pytest.raises(ValueError, match="null_slot_id requires"):
        runner.run(*tensors, **kwargs, initial_states=dense_initial, null_slot_id=0)
    for good in (-1, 0, 3, 2**31 - 1):
        run_initial(null_slot_id=good, final_states=pool, final_state_slots=slots)
    # A null slot other than -1 with a dense state side (identity slots
    # ``0 .. num_seqs - 1``, which the null slot could name) is refused: a
    # dense initial side with a final pool, an initial pool with returned
    # dense final states.
    for null in (0, 1, 3):
        with pytest.raises(
            ValueError,
            match=r"requires state pools with slot vectors on both state sides.*"
            r"initial_states without initial_state_slots",
        ):
            runner.run(
                *tensors,
                **kwargs,
                initial_states=dense_initial,
                final_states=pool,
                final_state_slots=slots,
                null_slot_id=null,
            )
        with pytest.raises(
            ValueError,
            match=r"requires state pools with slot vectors on both state sides.*"
            r"return_final_states=True without final_state_slots",
        ):
            run_initial(null_slot_id=null)
    # -1 keeps the dense sides; a side without states needs no slot vector.
    run_initial()
    runner.run(
        *tensors,
        **kwargs,
        initial_states=dense_initial,
        final_states=pool,
        final_state_slots=slots,
    )
    run_initial(null_slot_id=0, return_final_states=False)
    assert calls[-1][1]["main"]["null_slot"] == 0
    _, final = no_initial.run(
        *tensors, **kwargs, final_states=pool, final_state_slots=slots, null_slot_id=0
    )
    assert final is pool and calls[-1][1]["main"]["null_slot"] == 0

    # Checkpoint pools of a slot call: validated as pools, strided slot
    # vectors allowed, the identical pool tensor shared.
    checkpoint = dict(
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=torch.tensor([3, 9, 2, 9], dtype=torch.int32)[::2],
    )
    _, final = run_initial(
        **checkpoint,
        checkpoint_states=pool,
        final_states=pool,
        final_state_slots=slots,
    )
    assert final is pool
    run_initial(**checkpoint, checkpoint_states=_cpu_pool(4, nheads, torch.bfloat16))
    for bad in (
        torch.empty((4, nheads, 64, 64), dtype=torch.bfloat16),
        _pool_from(flat, 4, nheads, row - 128),
        _pool_from(flat, 4, nheads, row, offset=1),
        _cpu_pool(4, nheads, torch.float16),
    ):
        with pytest.raises(ValueError, match="checkpoint_states"):
            run_initial(**checkpoint, checkpoint_states=bad)
    with pytest.raises(ValueError, match="checkpoint_state_slots"):
        run_initial(
            checkpoint_token_indices=tokens,
            checkpoint_state_slots=torch.tensor([1], dtype=torch.int32).as_strided(
                (2,), (0,)
            ),
            checkpoint_states=_cpu_pool(4, nheads, torch.bfloat16),
        )
    # Partial overlaps: a view into the initial pool, and a checkpoint pool
    # over a dense initial side of a final-pool call.
    with pytest.raises(
        ValueError, match="checkpoint_states must not overlap initial_states"
    ):
        run_initial(**checkpoint, checkpoint_states=pool[1:3])
    with pytest.raises(
        ValueError, match="checkpoint_states must not overlap initial_states"
    ):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=storage[:2],
            final_states=_cpu_pool(4, nheads, torch.bfloat16),
            final_state_slots=slots,
            **checkpoint,
            checkpoint_states=storage[1:4],
        )
    # A checkpoint pool on a row of a strided dense initial side is refused;
    # one in its gap rows is accepted.
    with pytest.raises(
        ValueError, match="checkpoint_states must not overlap initial_states"
    ):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=strided_initial,
            final_states=_cpu_pool(4, nheads, torch.bfloat16),
            final_state_slots=slots,
            **checkpoint,
            checkpoint_states=storage[3:5],
        )
    runner.run(
        *tensors,
        **kwargs,
        initial_states=strided_initial,
        final_states=_cpu_pool(4, nheads, torch.bfloat16),
        final_state_slots=slots,
        **checkpoint,
        checkpoint_states=storage[1:4],
    )

    # A call without slot vectors keeps the dense checkpoint rules.
    dense_checkpoint = dict(
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=torch.tensor([0, 1], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="a checkpoint pool requires"):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=dense_initial,
            **dense_checkpoint,
            checkpoint_states=_cpu_pool(4, nheads, torch.bfloat16),
        )
    with pytest.raises(ValueError, match="checkpoint_state_slots must be a contiguous"):
        runner.run(
            *tensors,
            **kwargs,
            initial_states=dense_initial,
            **checkpoint,
            checkpoint_states=checkpoint_storage[:4],
        )
    x, dt, A, B, C = tensors
    with pytest.raises(ValueError, match=r"prepare\(\) takes checkpoint outputs"):
        runner.prepare(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            **kwargs,
            initial_states=dense_initial,
            **dense_checkpoint,
            checkpoint_states=checkpoint_storage[:2],
        )


def test_prepared_slot_call_is_a_pure_launch_without_gpu(monkeypatch):
    """``prepare`` of a slot call keys its workspace with ``state_slots``,
    allocates no final-state buffer for a final pool (or for any given
    ``final_states``) and resolves the program the run launches; the run
    with a static ``out`` allocates nothing, with or without flags,
    checkpoints and ``null_slot_id`` (prepared with them or not: every slot
    preparation allocates the flag dummy, so a run without flags after a
    preparation with them allocates nothing either).  The slot-free call of
    the same shape is another workspace."""

    module = _module()
    loads = []
    calls = []
    nheads, lengths = 8, (96, 160)
    runner = _slot_runner(module, monkeypatch, calls, nheads=nheads)
    monkeypatch.setattr(
        module, "_load_generated_program", lambda name, arch: loads.append(name)
    )
    (x, dt, A, B, C), kwargs = _cpu_varlen_call(lengths, nheads, 8, False, "cu_seqlens")
    out = torch.empty_like(x)
    pool = _cpu_pool(4, nheads, torch.bfloat16)
    slots = torch.tensor([2, -1], dtype=torch.int32)
    states = dict(
        initial_states=pool,
        initial_state_slots=slots,
        final_states=pool,
        final_state_slots=slots,
    )
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs, **states, out=out)
    assert prepared.workspace_key.state_slots is True
    assert prepared.program_name in module._SLOT_PROGRAMS
    assert loads == [prepared.program_name]
    workspace = runner._workspaces.get(prepared.workspace_key).workspace
    assert workspace.get("final_static") is None
    assert torch.equal(workspace["identity_slots"], torch.arange(2, dtype=torch.int32))
    out_run, final = runner.run(x, dt, A, B, C, **kwargs, **states, out=out)
    assert out_run is out and final is pool
    assert runner._allocations_in_last_run == 0
    assert runner.last_program_name == calls[-1][0] == prepared.program_name

    # Flags, a checkpoint pool (the state pool), strided slots and a null
    # slot: the same workspace and program, prepared or not.
    table = torch.tensor([[2, 0], [-1, 0]], dtype=torch.int32)
    extras = dict(
        initial_state_flags=torch.tensor([True, False]),
        null_slot_id=0,
        checkpoint_token_indices=torch.tensor([40, -1], dtype=torch.int32),
        checkpoint_state_slots=torch.tensor([[1, 3], [0, 0]], dtype=torch.int32)[:, 1],
        checkpoint_states=pool,
    )
    strided_states = dict(
        states, initial_state_slots=table[:, 0], final_state_slots=table[:, 0]
    )
    with_extras = runner.prepare(
        x=x, dt=dt, A=A, B=B, C=C, **kwargs, **strided_states, **extras, out=out
    )
    assert with_extras.workspace_key == prepared.workspace_key
    assert with_extras.program_name == prepared.program_name
    runner.run(x, dt, A, B, C, **kwargs, **strided_states, **extras, out=out)
    assert runner._allocations_in_last_run == 0
    main = calls[-1][1]["main"]
    assert main["has_initial"] == module._HAS_INITIAL_FLAGS
    assert main["checkpoint_states"] is pool and main["null_slot"] == 0
    runner.run(x, dt, A, B, C, **kwargs, **states, out=out)
    assert runner._allocations_in_last_run == 0

    # Prepared with flags, run without: the preparation allocated the flag
    # dummy all the same, so the run allocates nothing.
    flagged = _slot_runner(module, monkeypatch, calls, nheads=nheads)
    flagged.prepare(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        **kwargs,
        **states,
        initial_state_flags=extras["initial_state_flags"],
        out=out,
    )
    flagged.run(x, dt, A, B, C, **kwargs, **states, out=out)
    assert flagged._allocations_in_last_run == 0
    assert calls[-1][1]["main"]["has_initial"] == 1

    dense = runner.prepare(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        **kwargs,
        initial_states=torch.empty((2, nheads, 64, 128), dtype=torch.bfloat16),
        out=out,
    )
    assert dense.workspace_key.state_slots is False
    assert dense.workspace_key != prepared.workspace_key
    assert dense.program_name == _dense_twin(prepared.program_name)

    final_only = _slot_runner(
        module, monkeypatch, calls, nheads=nheads, has_initial_states=False
    )
    prepared = final_only.prepare(
        x=x, dt=dt, A=A, B=B, C=C, **kwargs, final_states=pool, final_state_slots=slots
    )
    workspace = final_only._workspaces.get(prepared.workspace_key).workspace
    assert prepared.workspace_key.state_slots is True
    assert workspace.get("final_static") is None


def test_public_entries_forward_the_slot_vectors_without_gpu():
    """The public runner and functional entry take both slot vectors, the
    flags and the null slot after ``final_states`` and forward them to the
    Cake runner; ``prepare`` also takes the checkpoint arguments.  The CuTe
    backend refuses them."""

    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    tail = (
        ("final_states", positional, None),
        ("initial_state_slots", positional, None),
        ("final_state_slots", positional, None),
        ("initial_state_flags", positional, None),
        ("null_slot_id", positional, -1),
    )
    assert _signature_contract(SSDCombined.run, drop_self=True)[-5:] == tail
    assert _signature_contract(module.ssd_combined_fwd)[-5:] == tail
    assert tuple(
        name for name, *_ in _signature_contract(SSDCombined.prepare, drop_self=True)
    )[-8:] == (
        "final_states",
        "initial_state_slots",
        "final_state_slots",
        "initial_state_flags",
        "null_slot_id",
        "checkpoint_token_indices",
        "checkpoint_state_slots",
        "checkpoint_states",
    )
    x, dt, A, B, C = _cpu_public_run_inputs()
    slots = torch.zeros(1, dtype=torch.int32)
    flags = torch.ones(1, dtype=torch.bool)
    cute = _public_runner_without_constructor("cute")
    refusal = re.escape("state slots require SSDCombined backend='cake'")
    for extra in (
        {"initial_state_slots": slots},
        {"final_state_slots": slots},
        {"initial_state_flags": flags},
        {"null_slot_id": 0},
    ):
        with pytest.raises(ValueError, match=refusal):
            cute.run(x, dt, A, B, C, **extra)
    cake = _public_runner_without_constructor("cake", cake_result=("out", "final"))
    final_slots = torch.zeros(1, dtype=torch.int32)
    assert cake.run(
        x,
        dt,
        A,
        B,
        C,
        initial_state_slots=slots,
        final_state_slots=final_slots,
        initial_state_flags=flags,
        null_slot_id=0,
    ) == ("out", "final")
    _, forwarded = cake._cake_runner.calls[-1]
    assert forwarded["initial_state_slots"] is slots
    assert forwarded["final_state_slots"] is final_slots
    assert forwarded["initial_state_flags"] is flags
    assert forwarded["null_slot_id"] == 0


# ---------------------------------------------------------------------------
# GPU: slot programs against their dense twins.


def _cake_device():
    return (
        torch.cuda.is_available()
        and get_compute_capability(torch.device("cuda")) in _CAKE_CAPABILITIES
    )


requires_cake_device = pytest.mark.skipif(
    not _cake_device(), reason="Cake SSDCombined requires SM100, SM103 or SM107"
)


def _device_arch():
    capability = tuple(get_compute_capability(torch.device("cuda")))
    return dict(zip(_CAKE_CAPABILITIES, _ARCHES, strict=True))[capability]


requires_sm107_device = pytest.mark.skipif(
    not (_cake_device() and _device_arch() == "sm_107a"),
    reason="the claim and wrap slot programs run on SM107 only",
)


def _sm107_params(*calls):
    """``calls`` as parameters that run on an SM107 device alone."""

    return tuple(
        pytest.param(call, marks=requires_sm107_device, id=str(call)) for call in calls
    )


@dataclass(frozen=True)
class _Call:
    """One packed-varlen call of the GPU tests."""

    lengths: tuple
    nheads: int
    state_dtype: torch.dtype
    has_z: bool = True
    d_layout: str = "matrix"  # "none", "vector" or "matrix"
    form: str = "cu_seqlens"  # or "derived", "trusted", "publishing"
    seq_idx_dtype: torch.dtype = torch.int32
    override: Optional[str] = None
    ngroups: Optional[int] = None
    seed: int = 0

    def __str__(self):
        state = _STATES[_STATE_DTYPES.index(self.state_dtype)]
        if len(set(self.lengths)) == 1:
            lengths = f"{len(self.lengths)}x{self.lengths[0]}"
        else:
            lengths = "-".join(map(str, self.lengths))
        parts = [lengths, f"h{self.nheads}", state, self.form, self.d_layout]
        parts.append("z" if self.has_z else "noz")
        if self.seq_idx_dtype == torch.int64 and self.form != "cu_seqlens":
            parts.append("i64")
        if self.override:
            parts.append(self.override)
        return "_".join(parts)


def _inputs(call, *, has_initial_states=True):
    """Constructor, positional tensors and keyword arguments of ``call`` on
    the current device (no state arguments), and its int64 ``seq_idx`` for
    the fp64 recurrence."""

    torch.manual_seed(call.seed)
    nheads = call.nheads
    ngroups = call.ngroups or min(nheads, 8)
    total = sum(call.lengths)
    x = torch.randn(1, total, nheads, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(1, total, nheads, device="cuda", dtype=torch.float32)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(1, total, ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    arguments = dict(
        dt_bias=torch.rand(nheads, device="cuda", dtype=torch.float32) - 4.0,
        dt_softplus=True,
        dt_limit=(0.001, 0.1),
    )
    if call.has_z:
        arguments["z"] = torch.randn_like(x)
    if call.d_layout == "matrix":
        arguments["D"] = torch.randn(nheads, 64, device="cuda").to(torch.bfloat16)
    elif call.d_layout == "vector":
        arguments["D"] = torch.randn(nheads, device="cuda").to(torch.bfloat16)
    metadata = _metadata(call.lengths, seq_idx_dtype=call.seq_idx_dtype, device="cuda")
    arguments.update(_form_arguments(metadata, call.form, len(call.lengths)))
    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=call.state_dtype,
        has_d=call.d_layout != "none",
        d_has_hdim=call.d_layout == "matrix",
        has_initial_states=has_initial_states,
        has_varlen=True,
        has_z=call.has_z,
        seq_idx_dtype=call.seq_idx_dtype,
    )
    return constructor, (x, dt, A, B, C), arguments, metadata["seq_idx"].long()


# A pool page laid out like a paged cache entry: a convolution-state region,
# the [nheads, 64, 128] SSM state, then padding; the buffer starts with a
# 16-byte guard.  Every byte outside the state rows must survive a call.
_CONV_BYTES = 4096
_PAD_BYTES = 48
_GUARD_BYTES = 16


class _Page:
    """A raw uint8 page buffer and the state pool viewed into it (``pad``
    bytes after each state row, a multiple of 16)."""

    def __init__(
        self, rows, nheads, state_dtype, *, seed, fill="random", pad=_PAD_BYTES
    ):
        self.rows, self.nheads, self.state_dtype = rows, nheads, state_dtype
        self.element = torch.empty((), dtype=state_dtype).element_size()
        self.page_bytes = _CONV_BYTES + nheads * _ROW * self.element + pad
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.raw = torch.randint(
            0,
            256,
            (2 * _GUARD_BYTES + rows * self.page_bytes,),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        self.pool = self.view(self.raw)
        if fill == "random":
            values = torch.randn(
                self.pool.shape, device="cuda", generator=generator
            ).to(state_dtype)
            self.pool.copy_(values)
        else:
            assert fill == "nan", fill
            self.pool.fill_(float("nan"))

    def view(self, raw):
        """The pool inside ``raw`` (this page's buffer or a copy of it)."""

        return raw.view(self.state_dtype).as_strided(
            (self.rows, self.nheads, 64, 128),
            (self.page_bytes // self.element, _ROW, 128, 1),
            (_GUARD_BYTES + _CONV_BYTES) // self.element,
        )


def _slot_vectors(num_sequences, rows, *, seed):
    """Initial and final slot vectors over ``rows``: distinct rows in random
    order for up to ``rows - 2`` sequences, -1 for the rest, one initial slot
    -1 and one out of range, one final slot out of range (call sizes
    permitting)."""

    generator = torch.Generator().manual_seed(seed)
    valid = min(num_sequences, rows - 2)
    order = torch.randperm(num_sequences, generator=generator)[:valid]
    initial = torch.full((num_sequences,), -1, dtype=torch.int32)
    final = torch.full((num_sequences,), -1, dtype=torch.int32)
    initial[order] = torch.randperm(rows, generator=generator)[:valid].int()
    final[order] = torch.randperm(rows, generator=generator)[:valid].int()
    if valid >= 2:
        initial[order[0]] = -1
    if valid >= 3:
        initial[order[1]] = rows
        final[order[2]] = rows + 5
    return initial.cuda(), final.cuda()


def _valid(slots, rows):
    indices = slots.long()
    return (indices >= 0) & (indices < rows), indices


def _gather(pool, slots):
    """The dense initial states a slot vector selects (zero outside the pool)."""

    valid, indices = _valid(slots, pool.shape[0])
    gathered = torch.zeros(
        (slots.numel(), *pool.shape[1:]), dtype=pool.dtype, device=pool.device
    )
    gathered[valid] = pool[indices[valid]]
    return gathered


def _scatter(pool, slots, values):
    """Store ``values[i]`` into row ``slots[i]`` of ``pool`` where it is a row."""

    valid, indices = _valid(slots, pool.shape[0])
    pool[indices[valid]] = values[valid]


def _run(constructor, tensors, arguments, **states):
    runner = SSDCombined(**constructor, backend="cake")
    out, final = runner.run(*tensors, **arguments, **states)
    torch.cuda.synchronize()
    return runner._cake_runner.last_program_name, out, final


@contextmanager
def _twin_route(twin):
    """Route the dense call to ``twin``, the dense program of the slot route:
    without the SM100 metadata and SM103 / SM107 one-launch forms, the
    batched views of single-sequence calls and the sm_107a claim,
    initial-state bulk, wrap, one-CTA pair and one-CTA-cluster forms; with
    ``exactpre`` for the retained start program (and its ring schedule) and
    the look-back or three-phase chunk-parallel program."""

    module = _module()
    family = twin.split("_")[0]
    with pytest.MonkeyPatch.context() as patch:
        for rule in (
            "claim_selected",
            "ibulk_selected",
            "fused_ibulk_selected",
            "wrap_long_selected",
            "tail_one_cta_selected",
            "single_sequence_two_kernel_batched",
        ):
            patch.setattr(module, rule, lambda **_: False)
        patch.setattr(module, "single_sequence_batched_variant", lambda **_: None)
        patch.setattr(
            module, "unit_cluster_variant", lambda **kwargs: kwargs["variant"]
        )
        patch.setattr(module, "_LIGHT_STAGE_CAPABILITIES", ())
        patch.setattr(module, "_ONE_LAUNCH_CAPABILITIES", ())
        if family == "exactpre":
            two_kernel_family = module.two_kernel_family
            exact_scan_variant = module._exact_scan_variant
            patch.setattr(
                module,
                "two_kernel_family",
                lambda **kwargs: "exactpre"
                if two_kernel_family(**kwargs) == "exact"
                else two_kernel_family(**kwargs),
            )
            patch.setattr(
                module,
                "_exact_scan_variant",
                lambda **kwargs: ""
                if exact_scan_variant(**kwargs) == "noring"
                else exact_scan_variant(**kwargs),
            )
        if family == "chunkpar3":
            patch.setenv(_CHUNK_PARALLEL_ENV, "three_phase")
        if family == "chunkpar":
            patch.setenv(_CHUNK_PARALLEL_ENV, "always")
        yield


def _dense_twin_run(twin, constructor, tensors, arguments, **states):
    """Run the dense call on ``twin``: ``((out, final), default)`` where
    ``default`` is ``None`` when the default route runs ``twin``, else the
    default route's ``(out, final)`` (another family, judged against the fp64
    recurrence)."""

    name, out, final = _run(constructor, tensors, arguments, **states)
    if name == twin:
        return (out, final), None
    default = (out, final)
    with _twin_route(twin):
        name, out, final = _run(constructor, tensors, arguments, **states)
    assert name == twin, (name, twin)
    return (out, final), default


def _oracle_arguments(arguments, seq_idx, initial_states):
    return dict(
        seq_idx=seq_idx,
        initial_states=initial_states,
        D=arguments.get("D"),
        z=arguments.get("z"),
        dt_bias=arguments["dt_bias"],
        dt_softplus=arguments["dt_softplus"],
        dt_limit=arguments["dt_limit"],
    )


# Calls whose slot programs cover the seventeen keys on every architecture:
# the fused window (ring and ``_noring``), the two-kernel ring and tail pair
# programs, the look-back and three-phase chunk-parallel calls, and calls
# whose slot route changes family (the derived table, one-token prefills, the
# sm_107a claim cells).
_TWIN_CALLS = (
    _Call((96, 160), 8, torch.bfloat16),
    _Call(
        (96, 160),
        8,
        torch.float16,
        has_z=False,
        d_layout="vector",
        form="trusted",
        seq_idx_dtype=torch.int64,
    ),
    _Call((96, 160), 8, torch.float32, d_layout="none"),
    _Call((64, 64), 128, torch.bfloat16),
    _Call((64, 64), 128, torch.float16, d_layout="vector", form="trusted"),
    _Call((600,) * 8, 32, torch.bfloat16, form="derived"),
    _Call(
        (600,) * 8,
        32,
        torch.float16,
        has_z=False,
        d_layout="vector",
        form="publishing",
        seq_idx_dtype=torch.int64,
    ),
    _Call((600,) * 8, 32, torch.float32, d_layout="none"),
    _Call((2048,) * 4, 128, torch.bfloat16, has_z=False, d_layout="none"),
    _Call(
        (2048,) * 4,
        128,
        torch.float16,
        has_z=False,
        form="trusted",
        seq_idx_dtype=torch.int64,
    ),
    _Call((2048,) * 4, 128, torch.float32, has_z=False, d_layout="vector"),
    _Call((32768,), 8, torch.bfloat16),
    _Call(
        (1500, 2600),
        4,
        torch.float16,
        has_z=False,
        d_layout="vector",
        form="derived",
        seq_idx_dtype=torch.int64,
        override="three_phase",
    ),
    _Call((1500, 2600), 4, torch.float32, form="trusted", override="three_phase"),
    _Call((96, 160), 8, torch.bfloat16, form="derived"),
    _Call((96, 160), 8, torch.float32, form="publishing", d_layout="none"),
    _Call((1,) * 1024, 32, torch.bfloat16, has_z=False),
    _Call((100,) * 16, 128, torch.bfloat16, has_z=False, d_layout="none"),
    _Call((1500, 2600), 4, torch.bfloat16, override="always"),
    _Call((1500, 2600), 4, torch.bfloat16, has_z=False, override="three_phase"),
    _Call(
        (1500, 2600),
        4,
        torch.float16,
        has_z=False,
        form="trusted",
        seq_idx_dtype=torch.int64,
        override="always",
    ),
    _Call((1500, 2600), 4, torch.float32, d_layout="vector", override="always"),
)


def _set_override(monkeypatch, call):
    if call.override is None:
        monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
    else:
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, call.override)


@requires_cake_device
@pytest.mark.parametrize("call", _TWIN_CALLS, ids=str)
def test_slot_program_is_bitwise_its_dense_twin(monkeypatch, call):
    """A call reading a row-strided initial pool and writing a separate final
    pool through permuted slot vectors (with -1 and out-of-range slots) is
    bitwise the dense twin's call on the gathered initial states: the same
    output, every valid final slot holding the twin's final state and every
    other byte of both page buffers (guard, convolution region, padding,
    unused rows) unchanged.  Where the slot route changes family the slot
    call is also at least as accurate against the fp64 recurrence as the
    default dense route."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, seq_idx = _inputs(call)
    num_sequences = len(call.lengths)
    rows = min(num_sequences + 3, 64)
    initial = _Page(rows, call.nheads, call.state_dtype, seed=call.seed + 1)
    final = _Page(rows, call.nheads, call.state_dtype, seed=call.seed + 2, fill="nan")
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=call.seed)
    initial_before, final_before = initial.raw.clone(), final.raw.clone()
    gathered = _gather(initial.pool, initial_slots)

    runner = SSDCombined(**constructor, backend="cake")
    out, returned = runner.run(
        *tensors,
        **arguments,
        initial_states=initial.pool,
        initial_state_slots=initial_slots,
        final_states=final.pool,
        final_state_slots=final_slots,
    )
    torch.cuda.synchronize()
    key = runner._cake_runner.last_program_name
    assert key in _ALL_SLOT_KEYS, key
    assert returned is final.pool

    (twin_out, twin_final), default = _dense_twin_run(
        _dense_twin(key), constructor, tensors, arguments, initial_states=gathered
    )
    assert torch.equal(out, twin_out), key
    assert torch.equal(initial.raw, initial_before), "the initial pool was written"
    expected = final_before.clone()
    _scatter(final.view(expected), final_slots, twin_final)
    assert torch.equal(final.raw, expected), key
    if default is not None:
        _assert_cake_accuracy(
            (out, twin_final),
            default,
            constructor,
            tensors,
            _oracle_arguments(arguments, seq_idx, gathered),
        )


@requires_cake_device
def test_twin_calls_reach_every_slot_program(monkeypatch):
    """On this device the twin calls run every slot program the route serves
    here (all seventeen with the look-back twins shipped, and with the
    sm_107a cells the four claim and wrap twins on SM107; routes only,
    nothing launched)."""

    module = _module()
    monkeypatch.setattr(module, "_launch_program", lambda *_, **__: None)
    reached = {}
    sm107 = _SM107_TWIN_CALLS if _device_arch() == "sm_107a" else ()
    for call in (*_TWIN_CALLS, *sm107):
        _set_override(monkeypatch, call)
        constructor, tensors, arguments, _ = _inputs(call)
        pool = torch.empty(
            (2, call.nheads, 64, 128), dtype=call.state_dtype, device="cuda"
        )
        slots = torch.full((len(call.lengths),), -1, dtype=torch.int32, device="cuda")
        runner = SSDCombined(**constructor, backend="cake")
        runner.run(
            *tensors,
            **arguments,
            initial_states=pool,
            initial_state_slots=slots,
            final_states=pool,
            final_state_slots=slots,
        )
        reached.setdefault(runner._cake_runner.last_program_name, []).append(str(call))
    served = _served_slot_keys(module, _device_arch())
    assert set(reached) == served, (sorted(served ^ set(reached)), reached)


# sm_107a cells whose dense call runs the claim or wrap program (the bf16
# claim cell of 1024 one-token prefills at 32 heads; the f32 claim cells of
# 1024 one-token and 256 x 128-token prefills at 8 and 16 heads; the wrap
# cells of 8 x 2048 tokens at 32 heads without z, both states): their slot
# calls run the claim and wrap twins.
_SM107_TWIN_CALLS = (
    _Call((1,) * 1024, 32, torch.bfloat16, has_z=False),
    _Call((1,) * 1024, 8, torch.float32, has_z=False, ngroups=1),
    _Call((128,) * 256, 16, torch.float32, has_z=False, ngroups=2),
    _Call(
        (128,) * 256,
        16,
        torch.float32,
        has_z=False,
        ngroups=2,
        d_layout="vector",
        form="trusted",
    ),
    _Call((2048,) * 8, 32, torch.bfloat16, has_z=False, ngroups=2, d_layout="none"),
    _Call((2048,) * 8, 32, torch.float32, has_z=False, ngroups=4),
    _Call(
        (2048,) * 8,
        32,
        torch.float32,
        has_z=False,
        ngroups=4,
        form="trusted",
        seq_idx_dtype=torch.int64,
    ),
)
# The cu_seqlens cells among them (the checkpoint and capture layouts), and
# their wrap cells (the capture calls hold the claim cells already).
_SM107_CU_SEQLENS_CALLS = tuple(
    call for call in _SM107_TWIN_CALLS if call.form == "cu_seqlens"
)
_SM107_WRAP_CU_SEQLENS_CALLS = tuple(
    call for call in _SM107_CU_SEQLENS_CALLS if call.lengths[0] > 128
)


def _sm107_twin_route(call, name, dense_name):
    """Check that the dense call runs the claim or wrap program and the slot
    call its twin."""

    assert dense_name.endswith(("_claim", "_wrap")), (str(call), dense_name)
    assert name == _slot_key(dense_name), (str(call), name, dense_name)
    assert name in _SM107_ONLY_SLOT_KEYS, name


@pytest.mark.parametrize("call", _sm107_params(*_SM107_TWIN_CALLS))
def test_sm107_claim_and_wrap_twins_are_bitwise_the_dense_call(monkeypatch, call):
    """A call the dense route sends to the claim or wrap program, reading a
    row-strided initial pool and writing a separate final pool through
    permuted slot vectors (with -1 and out-of-range slots), runs that
    program's slot twin and is bitwise the dense call on the gathered initial
    states: the same output, every valid final slot holding the dense final
    state and every other byte of both page buffers unchanged.  A bf16 claim
    call is bitwise the dense claim program, whose update rounds twice."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=call.seed + 1)
    final = _Page(rows, call.nheads, call.state_dtype, seed=call.seed + 2, fill="nan")
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=call.seed)
    initial_before, final_before = initial.raw.clone(), final.raw.clone()
    gathered = _gather(initial.pool, initial_slots)

    name, out, returned = _run(
        constructor,
        tensors,
        arguments,
        initial_states=initial.pool,
        initial_state_slots=initial_slots,
        final_states=final.pool,
        final_state_slots=final_slots,
    )
    assert returned is final.pool
    dense_name, dense_out, dense_final = _run(
        constructor, tensors, arguments, initial_states=gathered
    )
    _sm107_twin_route(call, name, dense_name)
    assert torch.equal(out, dense_out), name
    assert torch.equal(initial.raw, initial_before), "the initial pool was written"
    expected = final_before.clone()
    _scatter(final.view(expected), final_slots, dense_final)
    assert torch.equal(final.raw, expected), name


@pytest.mark.parametrize("call", _sm107_params(*_SM107_TWIN_CALLS))
def test_sm107_in_place_claim_and_wrap_pools_are_bitwise_the_dense_call(
    monkeypatch, call
):
    """One in-place pool (each sequence's final slot its own initial slot,
    permuted rows, some -1 or out of range): over 32 fresh seeds the claim or
    wrap twin leaves the page buffer bitwise the dense call's final states
    on the gathered initial rows scattered back, and the outputs equal."""

    _set_override(monkeypatch, call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    runner = SSDCombined(**_inputs(call)[0], backend="cake")
    dense = SSDCombined(**_inputs(call)[0], backend="cake")
    for seed in range(32):
        _, tensors, arguments, _ = _inputs(replace(call, seed=seed))
        page = _Page(rows, call.nheads, call.state_dtype, seed=200 + seed)
        slots, _ = _slot_vectors(num_sequences, rows, seed=300 + seed)
        before = page.raw.clone()
        gathered = _gather(page.pool, slots)
        dense_out, dense_final = dense.run(
            *tensors, **arguments, initial_states=gathered
        )
        dense_out, dense_final = dense_out.clone(), dense_final.clone()
        out, returned = runner.run(
            *tensors,
            **arguments,
            initial_states=page.pool,
            initial_state_slots=slots,
            final_states=page.pool,
            final_state_slots=slots,
        )
        torch.cuda.synchronize()
        assert returned is page.pool
        _sm107_twin_route(
            call,
            runner._cake_runner.last_program_name,
            dense._cake_runner.last_program_name,
        )
        expected = before.clone()
        _scatter(page.view(expected), slots, dense_final)
        assert torch.equal(out, dense_out), seed
        assert torch.equal(page.raw, expected), seed


_SEMANTICS_CALLS = (
    _Call((96, 160), 8, torch.bfloat16),
    _Call((600,) * 8, 32, torch.float32, form="derived"),
    _Call((1500, 2600), 4, torch.float16, override="three_phase"),
    _Call((1500, 2600), 4, torch.bfloat16, override="always"),
)


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_SEMANTICS_CALLS, *_sm107_params(*_SM107_TWIN_CALLS)), ids=str
)
def test_invalid_slots_start_from_zero_and_store_nothing(monkeypatch, call):
    """Initial slots all -1 start every sequence from zero: bitwise the
    dense call without initial states.  Final slots all -1 store nothing:
    the page buffer keeps every byte and the output is unchanged.  A slot
    of ``rows``, ``rows + 5``, ``2**31 - 1`` or -2 acts as -1 on either
    side."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=11)
    final = _Page(rows, call.nheads, call.state_dtype, seed=12, fill="nan")
    initial_slots, _ = _slot_vectors(num_sequences, rows, seed=13)
    final_slots = torch.randperm(rows)[:num_sequences].int().cuda()
    none = torch.full_like(initial_slots, -1)
    final_before = final.raw.clone()

    def slot_run(initial_vector, final_vector):
        final.raw.copy_(final_before)
        name, out, returned = _run(
            constructor,
            tensors,
            arguments,
            initial_states=initial.pool,
            initial_state_slots=initial_vector,
            final_states=final.pool,
            final_state_slots=final_vector,
        )
        assert name in _ALL_SLOT_KEYS and returned is final.pool
        return name, out, final.raw.clone()

    # All -1 initial: the dense call without initial states.
    name, out_zero, raw_zero = slot_run(none, final_slots)
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(name),
        {**constructor, "has_initial_states": False},
        tensors,
        arguments,
    )
    expected = final_before.clone()
    _scatter(final.view(expected), final_slots, twin_final)
    assert torch.equal(out_zero, twin_out) and torch.equal(raw_zero, expected)

    # All -1 final: nothing stored, the same output.
    _, out_store, _ = slot_run(initial_slots, final_slots)
    _, out_none, raw_none = slot_run(initial_slots, none)
    assert torch.equal(out_none, out_store)
    assert torch.equal(raw_none, final_before)

    # Out-of-range slots act as -1 on either side.
    for value in (rows, rows + 5, 2**31 - 1, -2):
        _, out, raw = slot_run(torch.full_like(none, value), final_slots)
        assert torch.equal(out, out_zero) and torch.equal(raw, raw_zero), value
        _, out, raw = slot_run(initial_slots, torch.full_like(none, value))
        assert torch.equal(out, out_none) and torch.equal(raw, raw_none), value


@requires_cake_device
@pytest.mark.parametrize("form", ("cu_seqlens", "trusted"))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32), ids=str)
def test_empty_sequences_carry_their_initial_slot(form, state_dtype):
    """A sequence without tokens stores its initial row into its final row
    (zero when its initial slot is -1); the sequences around it are bitwise
    the dense call's."""

    call = _Call((96, 0, 160, 0), 8, state_dtype, form=form)
    constructor, tensors, arguments, _ = _inputs(call)
    initial = _Page(8, 8, state_dtype, seed=21)
    final = _Page(8, 8, state_dtype, seed=22, fill="nan")
    initial_slots = torch.tensor([0, 1, 2, -1], dtype=torch.int32, device="cuda")
    final_slots = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device="cuda")
    name, out, _ = _run(
        constructor,
        tensors,
        arguments,
        initial_states=initial.pool,
        initial_state_slots=initial_slots,
        final_states=final.pool,
        final_state_slots=final_slots,
    )
    assert name in _ALL_SLOT_KEYS
    assert torch.equal(final.pool[4], initial.pool[1])
    assert not final.pool[6].any()
    gathered = _gather(initial.pool, initial_slots)
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(name), constructor, tensors, arguments, initial_states=gathered
    )
    assert torch.equal(out, twin_out)
    assert torch.equal(final.pool[final_slots.long()], twin_final)


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_SEMANTICS_CALLS, *_sm107_params(*_SM107_TWIN_CALLS)), ids=str
)
def test_mixed_pool_and_dense_sides_are_bitwise_the_dense_call(monkeypatch, call):
    """One side a pool, the other dense: a dense initial side with a final
    pool, and an initial pool with dense (allocated) final states, are
    bitwise the dense call."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=31)
    final = _Page(rows, call.nheads, call.state_dtype, seed=32, fill="nan")
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=33)
    gathered = _gather(initial.pool, initial_slots)

    final_before = final.raw.clone()
    name, out, returned = _run(
        constructor,
        tensors,
        arguments,
        initial_states=gathered,
        final_states=final.pool,
        final_state_slots=final_slots,
    )
    assert name in _ALL_SLOT_KEYS and returned is final.pool
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(name), constructor, tensors, arguments, initial_states=gathered
    )
    expected = final_before.clone()
    _scatter(final.view(expected), final_slots, twin_final)
    assert torch.equal(out, twin_out) and torch.equal(final.raw, expected)

    name, out, dense_final = _run(
        constructor,
        tensors,
        arguments,
        initial_states=initial.pool,
        initial_state_slots=initial_slots,
    )
    assert name in _ALL_SLOT_KEYS
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(name), constructor, tensors, arguments, initial_states=gathered
    )
    assert torch.equal(out, twin_out) and torch.equal(dense_final, twin_final)


_IN_PLACE_CALLS = (
    _Call((32768,), 8, torch.bfloat16, override="three_phase"),
    _Call((1500, 2600), 4, torch.float32, override="three_phase"),
    _Call((32768,), 8, torch.bfloat16, override="always"),
    _Call((1500, 2600), 4, torch.float32, override="always"),
    _Call((600,) * 8, 32, torch.bfloat16, form="derived"),
    _Call((2048,) * 4, 128, torch.bfloat16, has_z=False, d_layout="none"),
    _Call((96, 160), 8, torch.float16),
)


@requires_cake_device
@pytest.mark.parametrize("call", _IN_PLACE_CALLS, ids=str)
def test_in_place_pool_is_bitwise_the_two_pool_call(monkeypatch, call):
    """``final_states`` is the ``initial_states`` pool and each sequence's
    final slot its own initial slot (permuted rows, some -1): over 32 seeds
    the page buffer after the call is bitwise the final pool of the same call
    with a separate final pool, and the outputs are equal."""

    _set_override(monkeypatch, call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    runner = SSDCombined(**_inputs(call)[0], backend="cake")
    for seed in range(32):
        constructor, tensors, arguments, _ = _inputs(replace(call, seed=seed))
        page = _Page(rows, call.nheads, call.state_dtype, seed=100 + seed)
        slots, _ = _slot_vectors(num_sequences, rows, seed=seed)
        before = page.raw.clone()
        separate_initial = _Page(rows, call.nheads, call.state_dtype, seed=0)
        separate_initial.raw.copy_(before)
        separate_final = _Page(rows, call.nheads, call.state_dtype, seed=0)
        separate_final.raw.copy_(before)
        out_two, _ = runner.run(
            *tensors,
            **arguments,
            initial_states=separate_initial.pool,
            initial_state_slots=slots,
            final_states=separate_final.pool,
            final_state_slots=slots,
        )
        out_two = out_two.clone()
        two_pool_program = runner._cake_runner.last_program_name
        out_one, returned = runner.run(
            *tensors,
            **arguments,
            initial_states=page.pool,
            initial_state_slots=slots,
            final_states=page.pool,
            final_state_slots=slots,
        )
        torch.cuda.synchronize()
        assert returned is page.pool
        assert runner._cake_runner.last_program_name == two_pool_program
        assert two_pool_program in _ALL_SLOT_KEYS
        assert torch.equal(out_one, out_two), seed
        assert torch.equal(page.raw, separate_final.raw), seed


def _restore(*pages_and_copies):
    for page, copy in pages_and_copies:
        page.raw.copy_(copy)


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_SEMANTICS_CALLS, *_sm107_params(*_SM107_TWIN_CALLS)), ids=str
)
def test_strided_flagged_and_null_slots_are_bitwise_their_plain_forms(
    monkeypatch, call
):
    """Columns of slot tables (steps 2 and 4) are bitwise the contiguous slot
    vectors; ``initial_state_flags`` is bitwise the host-masked vector
    ``where(flags, initial_slots, -1)`` (bool and uint8 alike); a
    ``null_slot_id`` naming a row that is both an initial and a final slot
    is bitwise the vectors with that slot replaced by -1 on every side, and
    the null row of the final pool keeps its bytes."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=71)
    final = _Page(rows, call.nheads, call.state_dtype, seed=72, fill="nan")
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=73)
    # The null slot: a row some sequence reads, made a row some sequence
    # writes too, so the null run suppresses an initial read and a final store.
    valid = (initial_slots >= 0) & (initial_slots < rows)
    null = int(initial_slots[valid][0])
    if not bool((final_slots == null).any()):
        stores = ((final_slots >= 0) & (final_slots < rows)).nonzero().flatten()
        final_slots[int(stores[0])] = null
    assert bool((initial_slots == null).any()) and bool((final_slots == null).any())
    pages = ((initial, initial.raw.clone()), (final, final.raw.clone()))
    runner = SSDCombined(**constructor, backend="cake")

    def slot_run(**states):
        _restore(*pages)
        out, returned = runner.run(
            *tensors,
            **arguments,
            initial_states=initial.pool,
            final_states=final.pool,
            **states,
        )
        torch.cuda.synchronize()
        assert returned is final.pool
        assert runner._cake_runner.last_program_name in _ALL_SLOT_KEYS
        return out.clone(), initial.raw.clone(), final.raw.clone()

    plain = slot_run(initial_state_slots=initial_slots, final_state_slots=final_slots)
    for step in (2, 4):
        table = torch.full(
            (num_sequences, step), rows + 9, dtype=torch.int32, device="cuda"
        )
        initial_table, final_table = table.clone(), table.clone()
        initial_table[:, 0] = initial_slots
        final_table[:, 0] = final_slots
        strided = slot_run(
            initial_state_slots=initial_table[:, 0],
            final_state_slots=final_table[:, 0],
        )
        assert all(map(torch.equal, strided, plain)), step

    generator = torch.Generator(device="cuda").manual_seed(74)
    flags = torch.rand(num_sequences, device="cuda", generator=generator) < 0.5
    flags[0] = True
    masked = slot_run(
        initial_state_slots=torch.where(flags, initial_slots, -1),
        final_state_slots=final_slots,
    )
    for flag_tensor in (flags, flags.to(torch.uint8)):
        flagged = slot_run(
            initial_state_slots=initial_slots,
            final_state_slots=final_slots,
            initial_state_flags=flag_tensor,
        )
        assert all(map(torch.equal, flagged, masked))

    without_null = slot_run(
        initial_state_slots=torch.where(initial_slots == null, -1, initial_slots),
        final_state_slots=torch.where(final_slots == null, -1, final_slots),
    )
    with_null = slot_run(
        initial_state_slots=initial_slots,
        final_state_slots=final_slots,
        null_slot_id=null,
    )
    assert all(map(torch.equal, with_null, without_null))
    # The final pool is NaN-filled: compare the null row's bits, not values.
    bits = torch.int32 if final.element == 4 else torch.int16
    assert torch.equal(
        final.view(with_null[2])[null].view(bits),
        final.view(pages[1][1])[null].view(bits),
    )


# vLLM-shaped paged caches: each page holds the convolution state, the SSM
# state at a byte offset and padding; the ``ssm_state`` view is a row-strided
# pool over the page buffer: (nheads, page stride in state elements, SSM
# byte offset in the page, state dtype).  ``super_bf16`` keeps the byte size
# of the f32 Super page (1064960 f32 elements = 2129920 bf16 elements).
_VLLM_CACHES = {
    "nano": (64, 540672, 36864, torch.float32),
    "super": (128, 1064960, 61440, torch.float32),
    "super_bf16": (128, 2129920, 61440, torch.bfloat16),
}


class _PagedCache:
    """A raw uint8 page buffer with random bytes and the SSM state pool
    viewed into it (random finite states, row 0 the null block)."""

    def __init__(self, pages, model, *, seed):
        nheads, stride, offset, state_dtype = _VLLM_CACHES[model]
        self.pages, self.nheads, self.state_dtype = pages, nheads, state_dtype
        self.stride = stride
        self.element = torch.empty((), dtype=state_dtype).element_size()
        assert offset + nheads * _ROW * self.element <= stride * self.element
        self.offset = (_GUARD_BYTES + offset) // self.element
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.raw = torch.randint(
            0,
            256,
            (2 * _GUARD_BYTES + pages * stride * self.element,),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        self.pool = self.view(self.raw)
        self.pool.copy_(
            torch.randn(self.pool.shape, device="cuda", generator=generator).to(
                state_dtype
            )
        )

    def view(self, raw):
        return raw.view(self.state_dtype).as_strided(
            (self.pages, self.nheads, 64, 128),
            (self.stride, _ROW, 128, 1),
            self.offset,
        )


def _checkpoint_tokens(cu_seqlens, offsets):
    """vLLM's per-step conversion of relative checkpoint offsets (0: none)
    into absolute exclusive packed tokens, masked by the offset only."""

    return torch.where(offsets > 0, cu_seqlens[:-1] + offsets, -1).to(torch.int32)


_VLLM_LENGTHS = (300, 1, 129, 1000, 64, 2048)
# Relative checkpoint offsets per sequence: mid-sequence, none, one chunk in,
# the sequence end (the final state), an offset on a null checkpoint block
# (no store; an extra boundary only), one token in.
_VLLM_OFFSETS = (150, 0, 128, 1000, 7, 1)
_VLLM_NULL_CHECKPOINT = 4
_VLLM_CELLS = tuple(
    (model, num_spec, None) for model in _VLLM_CACHES for num_spec in (0, 1, 3)
) + (
    ("nano", 1, "always"),
    ("nano", 1, "three_phase"),
    ("super", 0, "always"),
)


@requires_cake_device
@pytest.mark.parametrize("model,num_spec,override", _VLLM_CELLS)
def test_vllm_paged_cache_is_bitwise_the_gather_dense_scatter_flow(
    monkeypatch, model, num_spec, override
):
    """A vLLM-shaped prefill: the ``ssm_state`` view of a paged cache as the
    initial, final and checkpoint pool, column 0 of a ``[num_reqs, 1 +
    num_spec]`` slot table as both slot vectors, ``has_initial_states`` as
    flags, a block-table column as the checkpoint slots (block 0 the null
    block, ``null_slot_id=0``) and the offset-masked checkpoint tokens.  The
    output is bitwise the current flow emulated in torch (gather the flagged
    initial rows, the dense call with dense checkpoint states, scatter the
    final and the non-null checkpoint rows), every other byte of the page
    buffer -- convolution states, padding, guards, the null block and unused
    pages -- is unchanged, and the checkpoint taken at a sequence's end is
    its final row."""

    monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
    if override is not None:
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override)
    nheads, _, _, state_dtype = _VLLM_CACHES[model]
    call = _Call(_VLLM_LENGTHS, nheads, state_dtype, override=override)
    constructor, tensors, arguments, seq_idx = _inputs(call)
    num_sequences = len(_VLLM_LENGTHS)
    cache = _PagedCache(2 * num_sequences + 4, model, seed=81 + num_spec)
    generator = torch.Generator().manual_seed(82 + num_spec)
    rows = torch.randperm(cache.pages - 1, generator=generator) + 1
    state_rows = rows[:num_sequences].int()
    checkpoint_rows = rows[num_sequences : 2 * num_sequences].int()
    # Slot table: column 0 the request's state row, the speculative columns
    # other rows the kernel must not read.
    table = torch.zeros((num_sequences, 1 + num_spec), dtype=torch.int32)
    table[:, 0] = state_rows
    table[:, 1:] = checkpoint_rows.flip(0).unsqueeze(1)
    slots = table.cuda()[:, 0]
    offsets = torch.tensor(_VLLM_OFFSETS, dtype=torch.int32, device="cuda")
    blocks = torch.zeros((num_sequences, 4), dtype=torch.int32)
    blocks[:, 2] = torch.where(
        torch.tensor(_VLLM_OFFSETS) > 0,
        checkpoint_rows,
        torch.zeros_like(checkpoint_rows),
    )
    blocks[_VLLM_NULL_CHECKPOINT, 2] = 0
    checkpoint_slots = blocks.cuda()[:, 2]
    flags = torch.tensor([True, False, True, True, False, True], device="cuda")
    tokens = _checkpoint_tokens(arguments["cu_seqlens"], offsets)
    before = cache.raw.clone()

    runner = SSDCombined(**constructor, backend="cake")
    out, returned = runner.run(
        *tensors,
        **arguments,
        initial_states=cache.pool,
        initial_state_slots=slots,
        initial_state_flags=flags,
        final_states=cache.pool,
        final_state_slots=slots,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=checkpoint_slots,
        checkpoint_states=cache.pool,
        null_slot_id=0,
    )
    torch.cuda.synchronize()
    key = runner._cake_runner.last_program_name
    assert key in _ALL_SLOT_KEYS and returned is cache.pool
    if override == "always":
        family = _lookback_slot_family(_module(), state_dtype, _device_arch())
        assert key.startswith(f"{family}_"), key
    if override == "three_phase":
        assert key.startswith("chunkpar3_"), key

    # The current flow: gather, dense call, scatter.
    gathered = _gather(cache.view(before), torch.where(flags, slots, -1))
    dense_checkpoints = torch.zeros(
        (num_sequences, nheads, 64, 128), dtype=state_dtype, device="cuda"
    )
    dense_checkpoint_slots = torch.where(
        tokens >= 0,
        torch.arange(num_sequences, dtype=torch.int32, device="cuda"),
        -1,
    ).to(torch.int32)
    (twin_out, twin_final), default = _dense_twin_run(
        _dense_twin(key),
        constructor,
        tensors,
        arguments,
        initial_states=gathered,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=dense_checkpoint_slots,
        checkpoint_states=dense_checkpoints,
    )
    expected = before.clone()
    expected_pool = cache.view(expected)
    _scatter(expected_pool, slots, twin_final)
    stored = (tokens >= 0) & (checkpoint_slots != 0)
    _scatter(
        expected_pool, torch.where(stored, checkpoint_slots, -1), dense_checkpoints
    )
    assert torch.equal(out, twin_out), key
    assert torch.equal(cache.raw, expected), key
    assert torch.equal(cache.pool[0], cache.view(before)[0])
    end = _VLLM_OFFSETS.index(_VLLM_LENGTHS[3])
    assert torch.equal(
        cache.pool[int(checkpoint_rows[end])], cache.pool[int(state_rows[end])]
    )
    if default is not None:
        _assert_cake_accuracy(
            (out, twin_final),
            default,
            constructor,
            tensors,
            _oracle_arguments(arguments, seq_idx, gathered),
        )


_LOOKBACK_IN_PLACE_CALLS = (
    _Call((8192, 8192, 300), 8, torch.bfloat16, override="always"),
    _Call((8192, 300, 8192), 8, torch.float32, d_layout="none", override="always"),
    _Call((5000, 20000), 4, torch.float16, has_z=False, override="always"),
)


@requires_cake_device
@pytest.mark.parametrize("call", _LOOKBACK_IN_PLACE_CALLS, ids=str)
def test_lookback_in_place_pool_with_flags_is_bitwise_the_two_pool_call(
    monkeypatch, call
):
    """The look-back slot program (the three-phase one where the route does
    not serve the look-back twin) with one in-place pool, mixed flags and
    items spanning many ranges (every range of an item reads its initial
    row, the range of its last tile overwrites it): over 16 seeds the page
    after the call is bitwise the final pool of the same call with separate
    initial and final pools, the outputs are equal, and both are the dense
    twin's call on the host-masked gathered states."""

    _set_override(monkeypatch, call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    family = _lookback_slot_family(_module(), call.state_dtype, _device_arch())
    runner = SSDCombined(**_inputs(call)[0], backend="cake")
    for seed in range(16):
        constructor, tensors, arguments, _ = _inputs(replace(call, seed=seed))
        page = _Page(rows, call.nheads, call.state_dtype, seed=200 + seed)
        generator = torch.Generator().manual_seed(300 + seed)
        # Every sequence in place on its own row; flags mixed.
        slots = torch.randperm(rows, generator=generator)[:num_sequences]
        slots = slots.int().cuda()
        flags = torch.rand(num_sequences, generator=generator) < 0.5
        flags[seed % num_sequences] = True
        flags[(seed + 1) % num_sequences] = False
        flags = flags.cuda()
        before = page.raw.clone()
        separate_initial = _Page(rows, call.nheads, call.state_dtype, seed=0)
        separate_initial.raw.copy_(before)
        separate_final = _Page(rows, call.nheads, call.state_dtype, seed=0)
        separate_final.raw.copy_(before)
        states = dict(initial_state_slots=slots, initial_state_flags=flags)
        out_two, _ = runner.run(
            *tensors,
            **arguments,
            **states,
            initial_states=separate_initial.pool,
            final_states=separate_final.pool,
            final_state_slots=slots,
        )
        out_two = out_two.clone()
        assert runner._cake_runner.last_program_name.startswith(f"{family}_")
        out_one, _ = runner.run(
            *tensors,
            **arguments,
            **states,
            initial_states=page.pool,
            final_states=page.pool,
            final_state_slots=slots,
        )
        torch.cuda.synchronize()
        key = runner._cake_runner.last_program_name
        assert key.startswith(f"{family}_") and key in _ALL_SLOT_KEYS, key
        assert torch.equal(out_one, out_two), seed
        assert torch.equal(page.raw, separate_final.raw), seed
        gathered = _gather(page.view(before), torch.where(flags, slots, -1))
        (twin_out, twin_final), _ = _dense_twin_run(
            _dense_twin(key), constructor, tensors, arguments, initial_states=gathered
        )
        expected = before.clone()
        _scatter(page.view(expected), slots, twin_final)
        assert torch.equal(out_one, twin_out), seed
        assert torch.equal(page.raw, expected), seed


# cu_seqlens calls of every chunk-parallel choice and the exact-scan route
# for the checkpoint layouts of a slot call.
_CHECKPOINT_CALLS = (
    _Call((96, 160, 40), 8, torch.bfloat16),
    _Call((600,) * 8, 32, torch.float16, has_z=False, d_layout="vector"),
    _Call((1500, 2600), 4, torch.float32, override="three_phase"),
    _Call((1500, 2600), 4, torch.bfloat16, override="always"),
)


def _checkpoint_token_vector(lengths, *, seed):
    """Absolute exclusive checkpoint tokens of a cu_seqlens call: a random
    token inside the sequence, none, the sequence end (the final state), in
    turn; -1 for an empty sequence."""

    generator = torch.Generator().manual_seed(seed)
    starts = [0, *accumulate(lengths)]
    tokens = []
    for index, length in enumerate(lengths):
        if length == 0 or index % 3 == 1:
            tokens.append(-1)
        elif index % 3 == 2:
            tokens.append(starts[index + 1])
        else:
            offset = int(torch.randint(1, length + 1, (1,), generator=generator))
            tokens.append(starts[index] + offset)
    return torch.tensor(tokens, dtype=torch.int32, device="cuda")


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_CHECKPOINT_CALLS, *_sm107_params(*_SM107_CU_SEQLENS_CALLS)), ids=str
)
def test_separate_checkpoint_pool_is_bitwise_the_gather_dense_scatter_flow(
    monkeypatch, call
):
    """A slot call whose checkpoint pool is a page buffer of its own, with
    another row stride (1 KiB more padding per page) and row count than the
    state pools, is bitwise the emulated dense flow: the dense twin's call on
    the gathered initial states with dense checkpoint states, its final
    states scattered into the final pool and its checkpoint states into the
    checkpoint rows (a -1 checkpoint slot stores nothing).  Every other byte
    of the three page buffers is unchanged."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=111)
    final = _Page(rows, call.nheads, call.state_dtype, seed=112, fill="nan")
    checkpoint = _Page(
        2 * num_sequences + 5,
        call.nheads,
        call.state_dtype,
        seed=113,
        fill="nan",
        pad=_PAD_BYTES + 1024,
    )
    assert checkpoint.pool.stride(0) != final.pool.stride(0)
    assert checkpoint.rows != final.rows
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=114)
    tokens = _checkpoint_token_vector(call.lengths, seed=115)
    generator = torch.Generator().manual_seed(116)
    checkpoint_slots = torch.randperm(checkpoint.rows, generator=generator)
    checkpoint_slots = checkpoint_slots[:num_sequences].int()
    if num_sequences > 3:
        checkpoint_slots[3] = -1
    checkpoint_slots = checkpoint_slots.cuda()
    initial_before = initial.raw.clone()
    final_before = final.raw.clone()
    checkpoint_before = checkpoint.raw.clone()

    runner = SSDCombined(**constructor, backend="cake")
    out, returned = runner.run(
        *tensors,
        **arguments,
        initial_states=initial.pool,
        initial_state_slots=initial_slots,
        final_states=final.pool,
        final_state_slots=final_slots,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=checkpoint_slots,
        checkpoint_states=checkpoint.pool,
    )
    torch.cuda.synchronize()
    key = runner._cake_runner.last_program_name
    assert key in _ALL_SLOT_KEYS and returned is final.pool, key

    gathered = _gather(initial.view(initial_before), initial_slots)
    dense_checkpoints = torch.zeros(
        (num_sequences, call.nheads, 64, 128), dtype=call.state_dtype, device="cuda"
    )
    dense_checkpoint_slots = torch.where(
        tokens >= 0,
        torch.arange(num_sequences, dtype=torch.int32, device="cuda"),
        -1,
    ).to(torch.int32)
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(key),
        constructor,
        tensors,
        arguments,
        initial_states=gathered,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=dense_checkpoint_slots,
        checkpoint_states=dense_checkpoints,
    )
    expected_final = final_before.clone()
    _scatter(final.view(expected_final), final_slots, twin_final)
    expected_checkpoint = checkpoint_before.clone()
    _scatter(
        checkpoint.view(expected_checkpoint),
        torch.where(tokens >= 0, checkpoint_slots, -1),
        dense_checkpoints,
    )
    assert torch.equal(out, twin_out), key
    assert torch.equal(initial.raw, initial_before), key
    assert torch.equal(final.raw, expected_final), key
    assert torch.equal(checkpoint.raw, expected_checkpoint), key


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_CHECKPOINT_CALLS, *_sm107_params(*_SM107_CU_SEQLENS_CALLS)), ids=str
)
def test_dense_checkpoint_tensor_in_a_slot_call_is_bitwise_the_dense_twin(
    monkeypatch, call
):
    """A slot call may pass a contiguous ``[num_seqs, nheads, 64, 128]``
    checkpoint tensor (a pool of one-row stride) with one checkpoint slot
    per sequence: its output, final pool and checkpoint tensor are bitwise
    the dense twin's call with the same checkpoint tensor and slots on the
    gathered initial states.  A ``null_slot_id`` naming one of its stored
    checkpoint slots is bitwise the same call with that value replaced by -1
    on every slot vector, and that checkpoint row is not written."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = num_sequences + 3
    initial = _Page(rows, call.nheads, call.state_dtype, seed=121)
    final = _Page(rows, call.nheads, call.state_dtype, seed=122, fill="nan")
    initial_slots, final_slots = _slot_vectors(num_sequences, rows, seed=123)
    tokens = _checkpoint_token_vector(call.lengths, seed=124)
    checkpoint_slots = torch.where(
        tokens >= 0,
        torch.arange(num_sequences, dtype=torch.int32, device="cuda"),
        -1,
    ).to(torch.int32)
    # A finite sentinel: rows no call writes keep it (torch.equal on NaN fails).
    sentinel = torch.full(
        (num_sequences, call.nheads, 64, 128),
        -3.0,
        dtype=call.state_dtype,
        device="cuda",
    )
    pages = ((initial, initial.raw.clone()), (final, final.raw.clone()))
    runner = SSDCombined(**constructor, backend="cake")

    def slot_run(initial_vector, final_vector, checkpoint_vector, **extra):
        _restore(*pages)
        checkpoints = sentinel.clone()
        out, returned = runner.run(
            *tensors,
            **arguments,
            initial_states=initial.pool,
            initial_state_slots=initial_vector,
            final_states=final.pool,
            final_state_slots=final_vector,
            checkpoint_token_indices=tokens,
            checkpoint_state_slots=checkpoint_vector,
            checkpoint_states=checkpoints,
            **extra,
        )
        torch.cuda.synchronize()
        key = runner._cake_runner.last_program_name
        assert key in _ALL_SLOT_KEYS and returned is final.pool, key
        return key, out.clone(), final.raw.clone(), checkpoints

    key, out, final_raw, checkpoints = slot_run(
        initial_slots, final_slots, checkpoint_slots
    )
    gathered = _gather(initial.view(pages[0][1]), initial_slots)
    dense_checkpoints = sentinel.clone()
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(key),
        constructor,
        tensors,
        arguments,
        initial_states=gathered,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=checkpoint_slots,
        checkpoint_states=dense_checkpoints,
    )
    expected_final = pages[1][1].clone()
    _scatter(final.view(expected_final), final_slots, twin_final)
    assert torch.equal(out, twin_out), key
    assert torch.equal(final_raw, expected_final), key
    assert torch.equal(checkpoints, dense_checkpoints), key

    # The null slot names a stored checkpoint row (and whatever initial or
    # final slot holds the same value): bitwise the -1 replacement.
    null = int((tokens >= 0).nonzero().flatten()[0])
    assert int(checkpoint_slots[null]) == null

    def without_null(vector):
        return torch.where(vector == null, -1, vector).to(torch.int32)

    masked = slot_run(
        without_null(initial_slots),
        without_null(final_slots),
        without_null(checkpoint_slots),
    )
    nulled = slot_run(initial_slots, final_slots, checkpoint_slots, null_slot_id=null)
    assert masked[0] == nulled[0] == key
    assert all(map(torch.equal, nulled[1:], masked[1:])), key
    assert torch.equal(nulled[3][null], sentinel[null]), key


@requires_cake_device
@pytest.mark.parametrize(
    "form,override",
    (("cu_seqlens", None), ("derived", None), ("cu_seqlens", "three_phase")),
)
def test_pool_rows_beyond_32_bit_offsets(monkeypatch, form, override):
    """A bf16 pool of 32768 rows of 8 heads, 65544 elements apart: row 32764
    crosses ``2**31`` elements and rows 32765-32767 start past it.  Sequences
    reading and writing them in place, with their checkpoints stored to rows
    32767 and 32766 of the same pool, are bitwise the dense call, and the
    padding between the rows keeps its values."""

    free, _ = torch.cuda.mem_get_info()
    if free < 5 * 2**30:
        pytest.skip("needs about 4.3 GB for the pool")
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override or "auto")
    nheads, rows, stride = 8, 32768, 65544
    call = _Call((96, 160, 200), nheads, torch.bfloat16, form=form)
    constructor, tensors, arguments, _ = _inputs(call)
    storage = torch.empty(
        (rows - 1) * stride + nheads * _ROW, dtype=torch.bfloat16, device="cuda"
    )
    pool = storage.as_strided((rows, nheads, 64, 128), (stride, _ROW, 128, 1))
    assert 32764 * stride < 2**31 < 32765 * stride
    for row in (32763, 32764, 32765):
        pool[row].copy_(torch.randn(nheads, 64, 128, device="cuda"))
    pads = [
        storage[row * stride + nheads * _ROW : (row + 1) * stride]
        for row in (32763, 32764)
    ]
    for pad in pads:
        pad.fill_(-7.0)
    slots = torch.tensor([32765, 32764, -1], dtype=torch.int32, device="cuda")
    final_slots = torch.tensor([32765, 32764, 32763], dtype=torch.int32, device="cuda")
    # Checkpoints: one chunk-unaligned token, one at the sequence end.
    tokens = torch.tensor([40, 256, -1], dtype=torch.int32, device="cuda")
    checkpoint_slots = torch.tensor([32767, 32766, 0], dtype=torch.int32, device="cuda")
    if form != "cu_seqlens":
        # The seq_idx triple captures segment ends only: the sequence ends.
        tokens = torch.tensor([96, 256, -1], dtype=torch.int32, device="cuda")
    gathered = _gather(pool, slots)
    name, out, returned = _run(
        constructor,
        tensors,
        arguments,
        initial_states=pool,
        initial_state_slots=slots,
        final_states=pool,
        final_state_slots=final_slots,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=checkpoint_slots,
        checkpoint_states=pool,
    )
    assert name in _ALL_SLOT_KEYS and returned is pool
    dense_checkpoints = torch.zeros(
        (3, nheads, 64, 128), dtype=torch.bfloat16, device="cuda"
    )
    (twin_out, twin_final), _ = _dense_twin_run(
        _dense_twin(name),
        constructor,
        tensors,
        arguments,
        initial_states=gathered,
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=torch.tensor(
            [0, 1, -1], dtype=torch.int32, device="cuda"
        ),
        checkpoint_states=dense_checkpoints,
    )
    assert torch.equal(out, twin_out)
    for sequence, row in enumerate((32765, 32764, 32763)):
        assert torch.equal(pool[row], twin_final[sequence]), row
    for sequence, row in enumerate((32767, 32766)):
        assert torch.equal(pool[row], dense_checkpoints[sequence]), row
    for pad in pads:
        assert bool((pad == -7.0).all())
    del storage, pool
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU: CUDA-graph capture with slot vectors.


def _allocation_requests(device) -> Optional[int]:
    if torch.cuda.get_allocator_backend() != "native":
        return None
    return int(torch.cuda.memory_stats(device)["allocation.all.allocated"])


@contextmanager
def _no_allocation_requests(device):
    """Assert that the block issues no allocation request to the caching
    allocator."""

    before = _allocation_requests(device)
    yield
    after = _allocation_requests(device)
    if before is not None:
        assert after == before, f"{after - before} allocator request(s)"


def _length_mixes(lengths, seed):
    """Other length mixes of the same total and sequence count: half the
    sequences carrying every token and the rest zero-length padding (four
    sequences or more), and a random split with zero-length sequences."""

    count, total = len(lengths), sum(lengths)
    mixes = []
    if count >= 4:
        half = count // 2
        base, extra = divmod(total, half)
        mixes.append((base,) * (half - 1) + (base + extra,) + (0,) * (count - half))
    generator = torch.Generator().manual_seed(seed)
    cuts = sorted(
        torch.randint(0, total + 1, (count - 1,), generator=generator).tolist()
    )
    mixes.append(tuple(b - a for a, b in zip([0, *cuts], [*cuts, total], strict=True)))
    return mixes


_CAPTURE_CALLS = tuple(
    _Call((1,) * 1024, nheads, state_dtype, has_z=False)
    for nheads in (128, 64, 32, 16)
    for state_dtype in _STATE_DTYPES
) + (
    _Call((96, 160), 8, torch.bfloat16),
    _Call((1500, 2600), 4, torch.float32, override="three_phase"),
)


@requires_cake_device
@pytest.mark.parametrize(
    "call", (*_CAPTURE_CALLS, *_sm107_params(*_SM107_WRAP_CU_SEQLENS_CALLS)), ids=str
)
def test_captured_slot_call_replays_with_mutated_slots_and_pools(monkeypatch, call):
    """A prepared slot call (one in-place pool) captured into a CUDA graph
    replays with the contents of its slot vectors, pool and ``cu_seqlens`` at
    replay time: after each mutation (other length mixes, -1 padding, other
    slots, new pool contents) the replay is bitwise an eager call on the same
    contents, and neither allocates."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = min(num_sequences + 3, 48)
    page = _Page(rows, call.nheads, call.state_dtype, seed=41)
    slots, _ = _slot_vectors(num_sequences, rows, seed=42)
    cu_seqlens = arguments["cu_seqlens"]
    out = torch.empty_like(tensors[0])
    states = dict(
        initial_states=page.pool,
        initial_state_slots=slots,
        final_states=page.pool,
        final_state_slots=slots,
    )
    runner = SSDCombined(**constructor, backend="cake")
    x, dt, A, B, C = tensors
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **arguments, **states, out=out)
    assert prepared.program_name in _ALL_SLOT_KEYS
    assert prepared.workspace_key.state_slots
    runner.run(*tensors, **arguments, **states, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _no_allocation_requests(out.device):
        _, returned = runner.run(*tensors, **arguments, **states, out=out)
    assert returned is page.pool
    assert runner._cake_runner._allocations_in_last_run == 0

    for step, lengths in enumerate((call.lengths, *_length_mixes(call.lengths, 43))):
        metadata = _metadata(lengths, device="cuda")
        cu_seqlens.copy_(metadata["cu_seqlens"])
        new_slots, _ = _slot_vectors(num_sequences, rows, seed=50 + step)
        # Padding sequences (no tokens) carry -1 slots.
        empty = torch.tensor([length == 0 for length in lengths], device="cuda")
        slots.copy_(torch.where(empty, torch.full_like(new_slots, -1), new_slots))
        page.pool.copy_(
            torch.randn(page.pool.shape, device="cuda").to(call.state_dtype)
        )
        before = page.raw.clone()
        out.fill_(float("nan"))
        with _no_allocation_requests(out.device):
            graph.replay()
        torch.cuda.synchronize()
        replay_out, replay_raw = out.clone(), page.raw.clone()
        page.raw.copy_(before)
        out.fill_(float("nan"))
        runner.run(*tensors, **arguments, **states, out=out)
        torch.cuda.synchronize()
        assert runner._cake_runner.last_program_name == prepared.program_name
        assert torch.equal(out, replay_out), step
        assert torch.equal(page.raw, replay_raw), step


_CAPTURE_CHECKPOINT_CALLS = (
    _Call((96, 160, 40, 300), 8, torch.bfloat16),
    _Call((1,) * 128, 64, torch.float32, has_z=False),
    _Call((1500, 2600), 4, torch.float32, override="always"),
    _Call((1500, 2600), 4, torch.float16, override="three_phase"),
)


@requires_cake_device
@pytest.mark.parametrize(
    "call",
    (*_CAPTURE_CHECKPOINT_CALLS, *_sm107_params(*_SM107_CU_SEQLENS_CALLS)),
    ids=str,
)
def test_captured_slot_call_replays_with_mutated_flags_and_checkpoints(
    monkeypatch, call
):
    """A slot call prepared with flags, a checkpoint pool (the in-place state
    pool), strided slot vectors and ``null_slot_id=0``, captured into a CUDA
    graph without an allocation request: after each mutation of the length
    mix, the slot table, the flags, the checkpoint offsets and slots (some
    null) and the pool contents, the replay is bitwise an eager call on the
    same contents (output and every byte of the page)."""

    _set_override(monkeypatch, call)
    constructor, tensors, arguments, _ = _inputs(call)
    num_sequences = len(call.lengths)
    rows = 2 * num_sequences + 3
    page = _Page(rows, call.nheads, call.state_dtype, seed=91)
    cu_seqlens = arguments["cu_seqlens"]
    table = torch.zeros((num_sequences, 2), dtype=torch.int32, device="cuda")
    blocks = torch.zeros((num_sequences, 3), dtype=torch.int32, device="cuda")
    flags = torch.zeros(num_sequences, dtype=torch.bool, device="cuda")
    tokens = torch.full((num_sequences,), -1, dtype=torch.int32, device="cuda")

    def mutate(lengths, seed):
        """Write a new contract-respecting step into the captured buffers:
        state rows and checkpoint rows disjoint, row 0 the null block."""

        generator = torch.Generator().manual_seed(seed)
        cu_seqlens.copy_(_metadata(lengths, device="cuda")["cu_seqlens"])
        order = torch.randperm(rows - 1, generator=generator) + 1
        table[:, 0] = order[:num_sequences].int().cuda()
        checkpoint = order[num_sequences : 2 * num_sequences].int()
        null = torch.rand(num_sequences, generator=generator) < 0.25
        blocks[:, 1] = torch.where(null, 0, checkpoint).int().cuda()
        flags.copy_(torch.rand(num_sequences, generator=generator) < 0.5)
        offsets = [
            int(torch.randint(0, length + 1, (1,), generator=generator))
            for length in lengths
        ]
        tokens.copy_(
            _checkpoint_tokens(
                cu_seqlens, torch.tensor(offsets, dtype=torch.int32, device="cuda")
            )
        )
        page.pool.copy_(
            torch.randn(page.pool.shape, device="cuda").to(call.state_dtype)
        )

    mutate(call.lengths, 92)
    out = torch.empty_like(tensors[0])
    states = dict(
        initial_states=page.pool,
        initial_state_slots=table[:, 0],
        initial_state_flags=flags,
        final_states=page.pool,
        final_state_slots=table[:, 0],
        checkpoint_token_indices=tokens,
        checkpoint_state_slots=blocks[:, 1],
        checkpoint_states=page.pool,
        null_slot_id=0,
    )
    runner = SSDCombined(**constructor, backend="cake")
    x, dt, A, B, C = tensors
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **arguments, **states, out=out)
    assert prepared.program_name in _ALL_SLOT_KEYS
    runner.run(*tensors, **arguments, **states, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _no_allocation_requests(out.device):
        _, returned = runner.run(*tensors, **arguments, **states, out=out)
    assert returned is page.pool
    assert runner._cake_runner._allocations_in_last_run == 0

    for step, lengths in enumerate((call.lengths, *_length_mixes(call.lengths, 93))):
        mutate(lengths, 100 + step)
        before = page.raw.clone()
        out.fill_(float("nan"))
        with _no_allocation_requests(out.device):
            graph.replay()
        torch.cuda.synchronize()
        replay_out, replay_raw = out.clone(), page.raw.clone()
        page.raw.copy_(before)
        out.fill_(float("nan"))
        runner.run(*tensors, **arguments, **states, out=out)
        torch.cuda.synchronize()
        assert runner._cake_runner.last_program_name == prepared.program_name
        assert torch.equal(out, replay_out), step
        assert torch.equal(page.raw, replay_raw), step
        assert torch.equal(page.pool[0], page.view(before)[0]), step


@requires_cake_device
def test_ssd_combined_fwd_returns_the_final_pool():
    """The functional entry with an in-place pool returns ``(out, pool)``,
    bitwise the runner's call on a copy of the page."""

    functional = importlib.import_module("flashinfer.mamba.ssd_combined")
    call = _Call((96, 160), 8, torch.bfloat16)
    constructor, tensors, arguments, _ = _inputs(call)
    page = _Page(5, 8, torch.bfloat16, seed=61)
    copy = _Page(5, 8, torch.bfloat16, seed=0)
    copy.raw.copy_(page.raw)
    slots = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    out, final = functional.ssd_combined_fwd(
        *tensors,
        **arguments,
        initial_states=page.pool,
        initial_state_slots=slots,
        final_states=page.pool,
        final_state_slots=slots,
    )
    assert final is page.pool
    _, out_runner, _ = _run(
        constructor,
        tensors,
        arguments,
        initial_states=copy.pool,
        initial_state_slots=slots,
        final_states=copy.pool,
        final_state_slots=slots,
    )
    assert torch.equal(out, out_runner) and torch.equal(page.raw, copy.raw)
