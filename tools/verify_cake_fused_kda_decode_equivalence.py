# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove Cake fused KDA decode equivalence without running timing measurements."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


_SCHEMA = "cake-fused-kda-decode-equivalence-v1"
_PREDECESSOR_COMMIT = "00a9d35a9d2ec1870a2068d9f970a7c15a9fa92a"
_PREDECESSOR_ALIGNMENT_ASSERT = (
    "static_assert(alignof(CUtensorMap) == 128, "
    '"CUtensorMap CUDA ABI must be 128-byte aligned");'
)
_CURRENT_ALIGNMENT_ASSERT = (
    "static_assert(alignof(FlashInferTensorMap) == 128, "
    '"kernel tensor-map ABI must be 128-byte aligned");'
)
_EMPTY_CUOBJDUMP_TRAILERS = {
    "--dump-sass": (
        "\nFatbin elf code:\n"
        "================\n"
        "arch = sm_100a\n"
        "code version = [1,8]\n"
        "host = linux\n"
        "compile_size = 64bit\n"
        "compressed\n\n"
        "\tcode for sm_100a\n"
    ),
    "--dump-resource-usage": (
        "\nFatbin elf code:\n"
        "================\n"
        "arch = sm_100a\n"
        "code version = [1,8]\n"
        "host = linux\n"
        "compile_size = 64bit\n"
        "compressed\n\n"
        "Resource usage:\n"
        " Common:\n"
        "  GLOBAL:0\n"
    ),
}
_HEAD_DIM = 128
_FULL_DOMAIN_HEADS = (12, 24, 32, 48, 96)
_FULL_DOMAIN_TAIL_ROWS = (384, 512, 768, 1024, 1536, 2048, 4096)
_FULL_DOMAIN_SHAPES = tuple(
    (num_heads, num_rows)
    for num_heads in _FULL_DOMAIN_HEADS
    for num_rows in range(1, 257)
) + tuple(
    (num_heads, num_rows)
    for num_heads in _FULL_DOMAIN_HEADS
    for num_rows in _FULL_DOMAIN_TAIL_ROWS
)
_VARIANT_CASES = {
    "repeated_safe_f32": (12, 3, "float32", "page", "repeated"),
    "repeated_safe_bf16": (12, 3, "bfloat16", "page", "repeated"),
    "wide512_positive_f32": (12, 1, "float32", "page", "positive"),
    "wide512_f32": (12, 2, "float32", "page", "null"),
    "wide512_bf16": (12, 1, "bfloat16", "page", "positive"),
    "compact_async_pr_eval_h96_f32": (96, 4, "float32", "page", "positive"),
    "compact_async_positive_f32": (12, 25, "float32", "page", "positive"),
    "compact_async_f32": (12, 25, "float32", "page", "null"),
    "compact_async_bf16": (12, 25, "bfloat16", "page", "positive"),
    "high_work_positive_h96_pr_strides_f32": (
        96,
        13,
        "float32",
        "page",
        "positive",
    ),
    "high_work_positive_h96_f32": (96, 13, "float32", "padded", "positive"),
    "high_work_positive_pr_eval_h12_f32": (
        12,
        99,
        "float32",
        "page",
        "positive",
    ),
    "high_work_positive_pr_eval_h24_f32": (
        24,
        50,
        "float32",
        "page",
        "positive",
    ),
    "high_work_positive_pr_eval_h32_f32": (
        32,
        32,
        "float32",
        "page",
        "positive",
    ),
    "high_work_positive_pr_eval_h48_f32": (
        48,
        25,
        "float32",
        "page",
        "positive",
    ),
    "high_work_positive_f32": (12, 99, "float32", "padded", "positive"),
    "high_work_f32": (12, 99, "float32", "page", "null"),
    "high_work_bf16": (12, 99, "bfloat16", "page", "positive"),
    "pr_eval_h32_f32": (32, 5, "float32", "page", "positive"),
    "direct_positive_f32": (24, 7, "float32", "page", "positive"),
    "direct_f32": (24, 7, "float32", "page", "null"),
    "direct_bf16": (24, 7, "bfloat16", "page", "positive"),
    "wide512_vector4_positive_f32": (12, 24, "float32", "page", "positive"),
}

# These factories remain independently executable after the production
# dispatcher moved positive-unique FP32 fallback traffic to wide CTAs.
_FACTORY_ONLY_VARIANTS = frozenset(("direct_positive_f32", "pr_eval_h32_f32"))


def _canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _write_text_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(value)


def _record_progress(value):
    raw_path = os.environ.get("CODESLACK_STEP_PROGRESS_FILE")
    if not raw_path:
        return
    path = Path(raw_path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(f"{value}\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(command, *, cwd=None, env=None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _clean_commit(root, expected=None):
    if _run(("git", "status", "--porcelain"), cwd=root):
        raise RuntimeError(f"repository is not clean: {root}")
    commit = _run(("git", "rev-parse", "HEAD"), cwd=root).strip()
    if len(commit) != 40 or (expected is not None and commit != expected):
        raise RuntimeError(f"repository commit is invalid: {root}: {commit}")
    return commit


def _normalize_kernel_symbol(payload, symbol, description):
    occurrences = payload.count(symbol)
    if occurrences != 1:
        raise RuntimeError(
            f"{description} contains {occurrences} occurrences of {symbol!r}"
        )
    return payload.replace(symbol, "CAKE_FUSED_KDA_KERNEL_SYMBOL")


def _normalize_predecessor_kernel_source(payload):
    occurrences = payload.count(_PREDECESSOR_ALIGNMENT_ASSERT)
    if occurrences != 1:
        raise RuntimeError(
            "predecessor source contains "
            f"{occurrences} legacy tensor-map alignment assertions"
        )
    return payload.replace(_PREDECESSOR_ALIGNMENT_ASSERT, _CURRENT_ALIGNMENT_ASSERT, 1)


def _normalize_predecessor_binding(source):
    body_guard = (
        "#ifndef FLASHINFER_FUSED_KDA_DECODE_BODY_FILE\n"
        '#error "FLASHINFER_FUSED_KDA_DECODE_BODY_FILE must name one frozen CUDA body"\n'
        "#endif\n"
    )
    if source.count(body_guard) != 1:
        raise RuntimeError("predecessor binding body guard is invalid")
    source = source.replace(body_guard, "", 1)
    preamble_start = source.index("#include <cstdint>\n")
    preamble_end = source.index("#undef int8_t\n", preamble_start) + len(
        "#undef int8_t\n"
    )
    source = source[:preamble_start] + source[preamble_end:]
    replacements = (
        (
            "FLASHINFER_FUSED_KDA_DECODE",
            "FLASHINFER_CAKE_FUSED_KDA_DECODE",
        ),
        ("fused_kda_decode_generated", "cake_fused_kda_decode"),
        (
            "fused KDA decode generated kernel ABI changed",
            "Cake fused KDA decode kernel ABI changed",
        ),
        (
            "fused KDA decode argument-plan identity",
            "Cake fused KDA decode argument-plan identity",
        ),
        ("this fused KDA decode module", "this Cake fused KDA decode module"),
        (
            "fused KDA decode dynamic shared memory",
            "Cake fused KDA decode dynamic shared memory",
        ),
        (
            "cudaFuncSetAttribute(fused KDA decode)",
            "cudaFuncSetAttribute(Cake fused KDA decode)",
        ),
        (
            "fused KDA decode repeated-row launch",
            "Cake fused KDA decode repeated-row launch",
        ),
        ("fused KDA decode launch", "Cake fused KDA decode launch"),
    )
    for predecessor, current in replacements:
        source = source.replace(predecessor, current)
    return source.replace(
        'identity must be a full SHA-256");\n\n\n#include',
        'identity must be a full SHA-256");\n\n#include',
        1,
    )


def _canonical_compile_flags(command, *, root, workspace, uri):
    tokens = shlex.split(command)
    result = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in ("-c", "-o"):
            skip_next = True
            continue
        normalized = token.replace(str(root), "<REPO>")
        normalized = normalized.replace(str(workspace), "<WORKSPACE>")
        normalized = normalized.replace(uri, "<MODULE>")
        normalized = normalized.replace(
            "FLASHINFER_FUSED_KDA_DECODE_TARGET_MINOR",
            "FLASHINFER_CAKE_FUSED_KDA_DECODE_TARGET_MINOR",
        )
        if not result:
            normalized = Path(normalized).name
        result.append(normalized)
    if not result or result[0] != "nvcc":
        raise RuntimeError("JIT compile command does not use nvcc")
    return result


def _tool_output(command):
    return _run(tuple(command)).strip()


def _normalize_cuobjdump_record(output, option):
    trailer = _EMPTY_CUOBJDUMP_TRAILERS[option]
    while output.endswith(trailer):
        output = output[: -len(trailer)]
    return output


def _cuobjdump_record(cuobjdump, library, symbol, option):
    output = _run((cuobjdump, option, "--function", symbol, str(library)))
    if symbol not in output:
        raise RuntimeError(f"cuobjdump did not report {symbol!r} from {library}")
    normalized = (
        output.replace(str(library), "<CONTAINER>")
        .replace(library.name, "<CONTAINER>")
        .replace(symbol, "CAKE_FUSED_KDA_KERNEL_SYMBOL")
    )
    return _normalize_cuobjdump_record(normalized, option)


def _canonicalize_reused_worker_evidence(result, details_root):
    for variant in result["variants"]:
        variant_root = details_root / variant["name"]
        evidence = {}
        for label, filename, option in (
            ("object", "object.sass.txt", "--dump-sass"),
            ("library", "library.sass.txt", "--dump-sass"),
            ("object_resources", "object.resources.txt", "--dump-resource-usage"),
            ("library_resources", "library.resources.txt", "--dump-resource-usage"),
        ):
            path = variant_root / filename
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"reused worker evidence is missing: {path}")
            raw = path.read_text()
            evidence[label] = (
                hashlib.sha256(raw.encode()).hexdigest(),
                hashlib.sha256(
                    _normalize_cuobjdump_record(raw, option).encode()
                ).hexdigest(),
            )
        raw_sass_sha256 = _canonical_json_sha256(
            {
                "object": evidence["object"][0],
                "library": evidence["library"][0],
            }
        )
        raw_resource_usage_sha256 = _canonical_json_sha256(
            {
                "object": evidence["object_resources"][0],
                "library": evidence["library_resources"][0],
            }
        )
        if (
            variant["sass_sha256"] != raw_sass_sha256
            or variant["resource_usage_sha256"] != raw_resource_usage_sha256
        ):
            raise RuntimeError(
                f"reused worker evidence hashes changed for {variant['name']}"
            )
        variant["sass_sha256"] = _canonical_json_sha256(
            {
                "object": evidence["object"][1],
                "library": evidence["library"][1],
            }
        )
        variant["resource_usage_sha256"] = _canonical_json_sha256(
            {
                "object": evidence["object_resources"][1],
                "library": evidence["library_resources"][1],
            }
        )
    return result


def _ninja_object_for_source(spec, source):
    source = source.resolve()
    matches = []
    for line in spec.ninja_path.read_text().splitlines():
        if not line.startswith("build ") or ": cuda_compile " not in line:
            continue
        output, input_path = line[6:].split(": cuda_compile ", 1)
        if Path(input_path).resolve() == source:
            matches.append(Path(output).resolve())
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve Ninja object for {source}")
    return matches[0]


def _tensor_digest(torch, tensors, scalars):
    digest = hashlib.sha256()
    digest.update(json.dumps(scalars, sort_keys=True, separators=(",", ":")).encode())
    for name, tensor in tensors:
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        contiguous = tensor.detach().contiguous().view(torch.uint8).cpu()
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _page_strides(torch, num_heads, state_dtype):
    hidden = num_heads * _HEAD_DIM
    conv_bytes = 3 * hidden * 3 * torch.bfloat16.itemsize
    state_bytes = (
        num_heads
        * _HEAD_DIM
        * _HEAD_DIM
        * torch.empty((), dtype=state_dtype).element_size()
    )
    page_bytes = conv_bytes + state_bytes
    return page_bytes // torch.bfloat16.itemsize, page_bytes // torch.empty(
        (), dtype=state_dtype
    ).element_size()


def _wide_num_slots(num_rows, conv_stride, state_stride, qkv_size, num_heads):
    int32_max = 2**31 - 1
    conv_slots = math.floor((int32_max - (3 * qkv_size - 1)) / conv_stride) + 2
    state_elements = num_heads * _HEAD_DIM * _HEAD_DIM
    state_slots = math.floor((int32_max - (state_elements - 1)) / state_stride) + 2
    return max(num_rows + 1, min(conv_slots, state_slots))


def _selected_slots(torch, state_indices):
    return torch.unique(state_indices[state_indices > 0], sorted=True).to(torch.long)


def _mutable_views(torch, tensors, selected_slots):
    return (
        ("output", tensors["output"]),
        ("selected_conv_state", tensors["conv_state"].index_select(0, selected_slots)),
        ("selected_state", tensors["state"].index_select(0, selected_slots)),
    )


def _make_inputs(torch, variant_name, seed):
    wide_offsets = variant_name.endswith("_wide_slot_offsets")
    base_name = (
        variant_name[: -len("_wide_slot_offsets")] if wide_offsets else variant_name
    )
    if base_name not in _VARIANT_CASES:
        raise RuntimeError(f"missing execution case for {variant_name!r}")
    num_heads, num_rows, state_dtype_name, layout, slot_class = _VARIANT_CASES[
        base_name
    ]
    state_dtype = torch.bfloat16 if state_dtype_name == "bfloat16" else torch.float32
    hidden = num_heads * _HEAD_DIM
    qkv_size = 3 * hidden
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape, dtype=torch.float32, scale=1.0):
        value = torch.randn(shape, dtype=torch.float32, generator=generator)
        if scale != 1.0:
            value.mul_(scale)
        return value.to(dtype=dtype, device="cuda")

    if layout == "page":
        x_padding, beta_padding, output_gate_padding = 17, 1, 7
        conv_stride, state_stride = _page_strides(torch, num_heads, state_dtype)
    elif layout == "padded":
        x_padding, beta_padding, output_gate_padding = 29, 3, 11
        conv_stride = 9 * hidden + 12
        state_stride = num_heads * _HEAD_DIM * _HEAD_DIM + 8
    else:
        raise RuntimeError(f"unknown execution layout: {layout}")
    num_slots = max(num_rows + 1, 4)
    if wide_offsets:
        num_slots = _wide_num_slots(
            num_rows, conv_stride, state_stride, qkv_size, num_heads
        )

    x_storage = randn((num_rows, qkv_size + x_padding), torch.bfloat16)
    conv_state = torch.empty_strided(
        (num_slots, qkv_size, 3),
        (conv_stride, 1, qkv_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    state = torch.empty_strided(
        (num_slots, num_heads, _HEAD_DIM, _HEAD_DIM),
        (state_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        dtype=state_dtype,
        device="cuda",
    )
    beta_storage = randn((1, num_rows, num_heads + beta_padding), torch.bfloat16)
    output_gate_storage = randn(
        (num_rows, hidden + output_gate_padding), torch.bfloat16
    )
    if slot_class == "positive":
        state_indices = torch.arange(num_rows, 0, -1, dtype=torch.int32, device="cuda")
    elif slot_class == "null":
        state_indices = torch.arange(num_rows, 0, -1, dtype=torch.int32, device="cuda")
        state_indices[0] = 0
        if num_rows >= 3:
            state_indices[1] = -1
    elif slot_class == "repeated":
        state_indices = torch.tensor(
            [1 + row % 2 for row in range(num_rows)],
            dtype=torch.int32,
            device="cuda",
        )
    else:
        raise RuntimeError(f"unknown execution slot class: {slot_class}")
    if wide_offsets:
        high_slot = num_slots - 1
        if slot_class == "repeated":
            state_indices.fill_(high_slot)
        else:
            live_rows = torch.nonzero(state_indices > 0).flatten()
            state_indices[live_rows[0]] = high_slot
    selected_slots = _selected_slots(torch, state_indices)
    if selected_slots.numel() == 0:
        raise RuntimeError(f"execution case for {variant_name!r} has no live slots")
    conv_state.index_copy_(
        0,
        selected_slots,
        randn((selected_slots.numel(), qkv_size, 3), torch.bfloat16, 0.1),
    )
    state.index_copy_(
        0,
        selected_slots,
        randn(
            (selected_slots.numel(), num_heads, _HEAD_DIM, _HEAD_DIM),
            state_dtype,
            0.01,
        ),
    )

    tensors = {
        "x": x_storage[:, :qkv_size],
        "weight": randn((3, 4, hidden), scale=0.1),
        "conv_state": conv_state,
        "raw_gate": randn((1, num_rows, num_heads, _HEAD_DIM), torch.bfloat16),
        "raw_beta": beta_storage[:, :, :num_heads],
        "A_log": randn((num_heads,), scale=0.5),
        "dt_bias": randn((hidden,), scale=0.1),
        "state_indices": state_indices,
        "state": state,
        "output_gate": output_gate_storage.as_strided(
            (num_rows, num_heads, _HEAD_DIM),
            (hidden + output_gate_padding, _HEAD_DIM, 1),
        ),
        "norm_weight": randn((_HEAD_DIM,)),
        "output": torch.zeros(
            (1, num_rows, num_heads, _HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        ),
    }
    scalars = {
        "use_lower_bound": 1,
        "lower_bound": -5.0,
        "norm_eps": 1.0e-5,
        "conv_state_shape": list(conv_state.shape),
        "conv_state_stride": list(conv_state.stride()),
        "state_shape": list(state.shape),
        "state_stride": list(state.stride()),
    }
    immutable_views = [
        (name, tensor)
        for name, tensor in sorted(tensors.items())
        if name not in ("conv_state", "state", "output")
    ]
    input_sha256 = _tensor_digest(
        torch,
        (*immutable_views, *_mutable_views(torch, tensors, selected_slots)),
        scalars,
    )
    mutable_before_sha256 = _tensor_digest(
        torch, _mutable_views(torch, tensors, selected_slots), scalars
    )
    return tensors, scalars, selected_slots, input_sha256, mutable_before_sha256


def _variant_record(variant, abi):
    return {
        "name": variant.name,
        "target": variant.target,
        "body": variant.body_path.name,
        "source_sha256": variant.source_sha256,
        "kernel_symbol": variant.kernel_symbol,
        "abi_kind": variant.abi_kind,
        "abi": [list(argument) for argument in abi[variant.abi_kind]],
        "state_dtype": variant.state_dtype,
        "slot_offset_bits": variant.slot_offset_bits,
        "extra_cuda_cflags": list(variant.extra_cuda_cflags),
        "threads": variant.threads,
        "dynamic_smem_bytes": variant.dynamic_smem_bytes,
    }


def _worker_imports(side):
    if side == "predecessor":
        module = importlib.import_module("flashinfer.jit.fused_kda_decode_generated")
        return {
            "abi": module.FUSED_KDA_DECODE_GENERATED_ABIS,
            "variants": module.load_fused_kda_decode_generated_variants,
            "spec": module.gen_fused_kda_decode_generated_module,
            "select": module.select_fused_kda_decode_generated_variant,
            "program_identity": None,
        }
    module = importlib.import_module("flashinfer.jit.cake_fused_kda_decode")
    return {
        "abi": module.CAKE_FUSED_KDA_DECODE_ABIS,
        "variants": module.get_cake_fused_kda_decode_variants,
        "spec": module.gen_cake_fused_kda_decode_module,
        "select": module.select_cake_fused_kda_decode_variant,
        "program_identity": module.get_cake_fused_kda_decode_program_identity,
    }


def _select_routes(torch, side, selector, variants):
    records = []
    for index, (num_heads, num_rows) in enumerate(_FULL_DOMAIN_SHAPES):
        hidden = num_heads * _HEAD_DIM
        conv_stride, state_stride = _page_strides(torch, num_heads, torch.float32)
        kwargs = {
            "target": "sm100a",
            "num_heads": num_heads,
            "num_rows": num_rows,
            "num_slots": num_rows + 1,
            "state_dtype": "float32",
            "lower_bound": -5.0,
            "norm_eps": 1.0e-5,
            "x_row_stride": 3 * hidden + 17,
            "conv_slot_stride": conv_stride,
            "beta_row_stride": num_heads + 1,
            "state_slot_stride": state_stride,
            "output_gate_row_stride": hidden + 7,
            "variants": variants,
        }
        if side == "predecessor":
            kwargs["slot_class"] = "positive_unique"
        else:
            kwargs["state_indices_mode"] = "positive_unique"
        selected = selector(**kwargs)
        if selected is None:
            raise RuntimeError(
                f"production dispatcher has no route for H={num_heads}, rows={num_rows}"
            )
        records.append(
            {
                "shape_index": index,
                "num_heads": num_heads,
                "num_rows": num_rows,
                "variant_name": None if selected is None else selected.name,
            }
        )
    return records


def _worker(args):
    root = Path(args.root).resolve()
    workspace = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("worker must execute from its requested repository")
    sys.path.insert(0, str(root))
    torch = importlib.import_module("torch")
    importlib.import_module("tvm_ffi")
    for name in (
        "FLASHINFER_EXTRA_CUDAFLAGS",
        "FLASHINFER_EXTRA_LDFLAGS",
        "FLASHINFER_CXX_LAUNCHER",
        "FLASHINFER_NVCC_LAUNCHER",
    ):
        if os.environ.get(name):
            raise RuntimeError(f"{name} must be unset for equivalence proof")
    if not torch.cuda.is_available():
        raise RuntimeError("equivalence proof requires CUDA")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (properties.major, properties.minor) != (
        10,
        0,
    ) or "B200" not in properties.name.upper():
        raise RuntimeError("equivalence proof requires one B200")
    gpu_rows = [
        row.strip()
        for row in _tool_output(
            (
                "nvidia-smi",
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            )
        ).splitlines()
        if row.strip()
    ]
    if len(gpu_rows) != 1:
        raise RuntimeError("equivalence worker requires exactly one visible GPU")
    gpu_uuid, gpu_name = (field.strip() for field in gpu_rows[0].split(",", 1))
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required")

    api = _worker_imports(args.side)
    variants = api["variants"]()
    expected_count = 44 if args.side == "predecessor" else 2 * len(_VARIANT_CASES)
    if len(variants) != expected_count:
        raise RuntimeError(
            f"{args.side} worker requires exactly {expected_count} variants"
        )
    route_records = _select_routes(torch, args.side, api["select"], variants)
    details_root = Path(args.output).resolve().parent / f"{args.side}-details"
    details_root.mkdir(mode=0o700)
    result_variants = []
    for index, variant in enumerate(variants):
        _record_progress(
            f"{args.side}: variant {index + 1}/{expected_count} {variant.name}: starting"
        )
        print(
            f"[{args.side}] {index + 1}/{expected_count} fresh-JIT and execute {variant.name}",
            flush=True,
        )
        predecessor_source = None
        if args.side == "predecessor":
            predecessor_source = variant.body_path.read_text()
            variant.body_path.write_text(
                _normalize_predecessor_kernel_source(predecessor_source)
            )
        try:
            spec = api["spec"](variant.name, variant.target)
            if spec.is_aot:
                raise RuntimeError(
                    f"equivalence proof refuses AOT cache hit: {spec.name}"
                )
            module = spec.build_and_load()
        finally:
            if predecessor_source is not None:
                variant.body_path.write_text(predecessor_source)
        library = spec.jit_library_path.resolve()
        if not library.is_file():
            raise RuntimeError(f"JIT library is missing: {library}")
        commands = spec.get_compile_commands()
        device_command_indices = [
            command_index
            for command_index, command in enumerate(commands)
            if (
                command["file"].endswith(variant.body_path.name)
                if args.side == "current"
                else command["file"].endswith("fused_kda_decode_generated_binding.cu")
            )
        ]
        if len(device_command_indices) != 1:
            raise RuntimeError(
                f"cannot identify device compile command for {variant.name}"
            )
        device_command_index = device_command_indices[0]
        compile_flags = _canonical_compile_flags(
            commands[device_command_index]["command"],
            root=root,
            workspace=workspace,
            uri=spec.name,
        )
        all_compile_flags = [
            _canonical_compile_flags(
                command["command"],
                root=root,
                workspace=workspace,
                uri=spec.name,
            )
            for command in commands
        ]
        if any(flags != compile_flags for flags in all_compile_flags):
            raise RuntimeError(
                f"translation-unit compile flags differ for {variant.name}"
            )
        object_path = _ninja_object_for_source(
            spec, Path(commands[device_command_index]["file"])
        )
        if not object_path.is_file():
            raise RuntimeError(f"JIT device object is missing: {object_path}")
        object_sass = _cuobjdump_record(
            cuobjdump, object_path, variant.kernel_symbol, "--dump-sass"
        )
        library_sass = _cuobjdump_record(
            cuobjdump, library, variant.kernel_symbol, "--dump-sass"
        )
        object_resources = _cuobjdump_record(
            cuobjdump, object_path, variant.kernel_symbol, "--dump-resource-usage"
        )
        library_resources = _cuobjdump_record(
            cuobjdump, library, variant.kernel_symbol, "--dump-resource-usage"
        )
        variant_details = details_root / variant.name
        variant_details.mkdir(mode=0o700)
        for filename, content in (
            ("compile_flags.json", json.dumps(compile_flags, indent=2) + "\n"),
            ("object.sass.txt", object_sass),
            ("library.sass.txt", library_sass),
            ("object.resources.txt", object_resources),
            ("library.resources.txt", library_resources),
        ):
            _write_text_exclusive(variant_details / filename, content)
        torch.cuda.reset_peak_memory_stats()
        (
            tensors,
            scalars,
            selected_slots,
            input_sha256,
            mutable_before_sha256,
        ) = _make_inputs(torch, variant.name, 91000 + index)
        state_indices_mode = {
            "positive": "positive_unique",
            "null": "unique_or_null",
            "repeated": "repeated_positive",
        }[_VARIANT_CASES[variant.name.removesuffix("_wide_slot_offsets")][4]]
        selector_kwargs = {
            "target": variant.target,
            "num_heads": tensors["A_log"].numel(),
            "num_rows": tensors["x"].shape[0],
            "num_slots": tensors["conv_state"].shape[0],
            "state_dtype": (
                "bfloat16" if tensors["state"].dtype == torch.bfloat16 else "float32"
            ),
            "lower_bound": scalars["lower_bound"],
            "norm_eps": scalars["norm_eps"],
            "x_row_stride": tensors["x"].stride(0),
            "conv_slot_stride": tensors["conv_state"].stride(0),
            "beta_row_stride": tensors["raw_beta"].stride(1),
            "state_slot_stride": tensors["state"].stride(0),
            "output_gate_row_stride": tensors["output_gate"].stride(0),
            "variants": variants,
        }
        if args.side == "predecessor":
            selector_kwargs["slot_class"] = state_indices_mode
        else:
            selector_kwargs["state_indices_mode"] = state_indices_mode
        selected_variant = api["select"](**selector_kwargs)
        factory_only = (
            args.side == "current"
            and variant.name.removesuffix("_wide_slot_offsets")
            in _FACTORY_ONLY_VARIANTS
        )
        if selected_variant is None or (
            not factory_only and selected_variant.name != variant.name
        ):
            raise RuntimeError(
                f"execution fixture selected {getattr(selected_variant, 'name', None)!r}, "
                f"expected {variant.name!r}"
            )
        # `module` was freshly built for this exact named variant above.
        # Execute its real binding even if the live host dispatcher now uses
        # another schedule; the independent production route is recorded below.
        module.run(
            tensors["x"],
            tensors["weight"],
            tensors["conv_state"],
            tensors["raw_gate"],
            tensors["raw_beta"],
            tensors["A_log"],
            tensors["dt_bias"],
            tensors["state_indices"],
            tensors["state"],
            tensors["output_gate"],
            tensors["norm_weight"],
            tensors["output"],
            scalars["use_lower_bound"],
            scalars["lower_bound"],
            scalars["norm_eps"],
        )
        torch.cuda.synchronize()
        execution_sha256 = _tensor_digest(
            torch,
            _mutable_views(torch, tensors, selected_slots),
            scalars,
        )
        if (
            execution_sha256 == mutable_before_sha256
            or not torch.count_nonzero(tensors["output"]).item()
        ):
            raise RuntimeError(
                f"execution fixture was not observable for {variant.name}"
            )
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        if peak_allocated_bytes > 24 * 1024**3:
            raise RuntimeError(
                f"execution fixture exceeded 24 GiB for {variant.name}: "
                f"{peak_allocated_bytes} bytes"
            )
        result_variants.append(
            {
                **_variant_record(variant, api["abi"]),
                "compile_flags": compile_flags,
                "compile_flags_sha256": _canonical_json_sha256(compile_flags),
                "sass_sha256": _canonical_json_sha256(
                    {
                        "object": hashlib.sha256(object_sass.encode()).hexdigest(),
                        "library": hashlib.sha256(library_sass.encode()).hexdigest(),
                    }
                ),
                "resource_usage_sha256": _canonical_json_sha256(
                    {
                        "object": hashlib.sha256(object_resources.encode()).hexdigest(),
                        "library": hashlib.sha256(
                            library_resources.encode()
                        ).hexdigest(),
                    }
                ),
                "execution_input_sha256": input_sha256,
                "execution_sha256": execution_sha256,
                "peak_allocated_bytes": peak_allocated_bytes,
                "object_sha256": _file_sha256(object_path),
                "library_sha256": _file_sha256(library),
                "execution_binding": "explicit_named_factory",
                "production_variant": selected_variant.name,
                "production_selects_executed_factory": (
                    selected_variant.name == variant.name
                ),
            }
        )
        _record_progress(
            f"{args.side}: variant {index + 1}/{expected_count} {variant.name}: sealed"
        )
        del module, tensors
        torch.cuda.empty_cache()

    toolchain = {
        "cuda_version": str(torch.version.cuda),
        "nvcc_version": _tool_output((shutil.which("nvcc") or "nvcc", "--version")),
        "cuobjdump_version": _tool_output((cuobjdump, "--version")),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "tvm_ffi_version": importlib.metadata.version("apache-tvm-ffi"),
    }
    _write_json_atomic(
        Path(args.output).resolve(),
        {
            "side": args.side,
            "root": str(root),
            "commit": _clean_commit(root),
            "workspace": str(workspace),
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "compute_capability": [properties.major, properties.minor],
            "node": platform.node(),
            "toolchain": toolchain,
            "toolchain_sha256": _canonical_json_sha256(toolchain),
            "program_identity_sha256": (
                None if api["program_identity"] is None else api["program_identity"]()
            ),
            "variants": result_variants,
            "routes": route_records,
        },
    )


def _row_variant_names(rows_root):
    names = []
    for index in range(len(_FULL_DOMAIN_SHAPES)):
        receipt = json.loads((rows_root / f"row-{index:04d}.json").read_text())
        row = receipt.get("row")
        cells = (
            []
            if not isinstance(row, dict)
            else [
                cell
                for cell in row.get("measurements", [])
                if isinstance(cell, dict) and cell.get("backend") == "candidate"
            ]
        )
        variants = {cell.get("variant_name") for cell in cells}
        if len(cells) != 2 or len(variants) != 1 or None in variants:
            raise RuntimeError(f"row {index} has invalid candidate route evidence")
        names.append(variants.pop())
    return names


def _orchestrator(args):
    current_root = Path(args.current_root).resolve()
    predecessor_root = Path(args.predecessor_root).resolve()
    output = Path(args.output).resolve()
    work_root = Path(args.work_root).resolve()
    reuse_work_root = (
        None if args.reuse_work_root is None else Path(args.reuse_work_root).resolve()
    )
    checkpoint_path = Path(args.predecessor_checkpoint).resolve()
    rows_root = Path(args.predecessor_rows_root).resolve()
    manifest_path = (
        predecessor_root / "csrc/kda/fused_kda_decode_generated_manifest.json"
    )
    script_path = Path(__file__).resolve()
    current_commit = _clean_commit(current_root)
    predecessor_commit = _clean_commit(predecessor_root, _PREDECESSOR_COMMIT)
    if script_path.parent.parent != current_root:
        raise RuntimeError("verifier must be executed from the current repository")
    if output.is_relative_to(current_root) or work_root.is_relative_to(current_root):
        raise RuntimeError("proof output and work root must be outside the repository")
    if reuse_work_root is not None and reuse_work_root.is_relative_to(current_root):
        raise RuntimeError("reused proof work root must be outside the repository")
    if output.exists():
        raise RuntimeError("equivalence output already exists")
    if (
        not checkpoint_path.is_file()
        or not rows_root.is_dir()
        or not manifest_path.is_file()
    ):
        raise RuntimeError("predecessor checkpoint, rows, or manifest is unavailable")
    checkpoint = json.loads(checkpoint_path.read_text())
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("progress", {}).get("completed_rows")
        != len(_FULL_DOMAIN_SHAPES)
        or len(str(checkpoint.get("identity", {}).get("source_commit", ""))) != 40
    ):
        raise RuntimeError("predecessor checkpoint is not the complete measured run")
    predecessor_manifest_sha256 = _file_sha256(manifest_path)
    if (
        checkpoint.get("identity", {}).get("manifest_sha256")
        != predecessor_manifest_sha256
    ):
        raise RuntimeError("predecessor checkpoint does not bind the manifest")

    if work_root.exists():
        raise RuntimeError("equivalence work root must not already exist")
    work_root.mkdir(parents=True, mode=0o700)
    os.chmod(work_root, 0o700)
    side_results = {}
    for side, root in (
        ("predecessor", predecessor_root),
        ("current", current_root),
    ):
        if reuse_work_root is not None:
            result = reuse_work_root / f"{side}.json"
            details_root = reuse_work_root / f"{side}-details"
            if not result.is_file() or result.is_symlink() or not details_root.is_dir():
                raise RuntimeError(f"reused {side} worker result is unavailable")
            side_results[side] = _canonicalize_reused_worker_evidence(
                json.loads(result.read_text()), details_root
            )
            _record_progress(f"orchestrator: {side} worker evidence reused")
            continue
        _record_progress(f"orchestrator: {side} worker starting")
        workspace = work_root / f"{side}-workspace"
        result = work_root / f"{side}.json"
        workspace.mkdir(mode=0o700)
        environment = dict(os.environ)
        environment.update(
            {
                "FLASHINFER_WORKSPACE_BASE": str(workspace),
                "FLASHINFER_JIT_DEBUG": "0",
                "FLASHINFER_JIT_VERBOSE": "0",
                "PYTHONPATH": str(root),
            }
        )
        subprocess.run(
            (
                sys.executable,
                str(script_path),
                "--worker-side",
                side,
                "--worker-root",
                str(root),
                "--worker-output",
                str(result),
            ),
            cwd=root,
            env=environment,
            check=True,
        )
        side_results[side] = json.loads(result.read_text())
        _record_progress(f"orchestrator: {side} worker sealed")

    predecessor_result = side_results["predecessor"]
    current_result = side_results["current"]
    for field in (
        "gpu_uuid",
        "gpu_name",
        "compute_capability",
        "node",
        "toolchain",
        "toolchain_sha256",
    ):
        if predecessor_result[field] != current_result[field]:
            raise RuntimeError(f"old/new workers differ in {field}")
    if predecessor_result["commit"] != predecessor_commit:
        raise RuntimeError("predecessor worker commit changed")
    if current_result["commit"] != current_commit:
        changed_paths = _run(
            (
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                f"{current_result['commit']}..{current_commit}",
                "--",
            ),
            cwd=current_root,
        ).splitlines()
        verifier_path = script_path.relative_to(current_root).as_posix()
        proof_only_paths = [
            "tests/jit/test_flash_kda_frozen_idents_jit.py",
            verifier_path,
        ]
        if reuse_work_root is None or changed_paths != proof_only_paths:
            raise RuntimeError("current worker commit changed outside proof-only paths")

    manifest = json.loads(manifest_path.read_text())
    predecessor_binding = (
        predecessor_root / "csrc/kda/fused_kda_decode_generated_binding.cuh"
    ).read_text()
    current_binding = (
        current_root / "csrc/kda/cake_fused_kda_decode_binding.cuh"
    ).read_text()
    if _normalize_predecessor_binding(predecessor_binding) != current_binding:
        raise RuntimeError(
            "old/new host launch bindings are not branding-only equivalent"
        )
    manifest_variants = manifest.get("variants")
    old_variants = predecessor_result["variants"]
    new_variants = current_result["variants"]
    if not all(
        isinstance(value, list) and len(value) == 44
        for value in (manifest_variants, old_variants, new_variants)
    ):
        raise RuntimeError("old/new proof must cover exactly 44 variants")
    proof_variants = []
    current_csrc = current_root / "csrc/kda"
    for manifest_variant, old, new in zip(
        manifest_variants, old_variants, new_variants, strict=True
    ):
        name = manifest_variant.get("name")
        if old["name"] != name or new["name"] != name:
            raise RuntimeError("variant order changed")
        old_source = (
            predecessor_root / "csrc/kda" / manifest_variant["body"]
        ).read_text()
        new_source = (current_csrc / new["body"]).read_text()
        if (
            old["source_sha256"] != hashlib.sha256(old_source.encode()).hexdigest()
            or new["source_sha256"] != hashlib.sha256(new_source.encode()).hexdigest()
        ):
            raise RuntimeError(f"worker source evidence changed for {name}")
        old_normalized = _normalize_kernel_symbol(
            _normalize_predecessor_kernel_source(old_source),
            old["kernel_symbol"],
            f"predecessor {name}",
        )
        new_normalized = _normalize_kernel_symbol(
            new_source, new["kernel_symbol"], f"current {name}"
        )
        static_equal = (
            old_normalized == new_normalized
            and old["target"] == new["target"]
            and old["abi_kind"] == new["abi_kind"]
            and old["abi"] == new["abi"]
            and old["state_dtype"] == new["state_dtype"]
            and old["slot_offset_bits"] == new["slot_offset_bits"]
            and old["extra_cuda_cflags"] == new["extra_cuda_cflags"]
            and old["threads"] == new["threads"]
            and old["dynamic_smem_bytes"] == new["dynamic_smem_bytes"]
        )
        flags_equal = old["compile_flags"] == new["compile_flags"]
        sass_equal = old["sass_sha256"] == new["sass_sha256"]
        resources_equal = old["resource_usage_sha256"] == new["resource_usage_sha256"]
        inputs_equal = old["execution_input_sha256"] == new["execution_input_sha256"]
        execution_equal = old["execution_sha256"] == new["execution_sha256"]
        if not all(
            (
                static_equal,
                flags_equal,
                sass_equal,
                resources_equal,
                inputs_equal,
                execution_equal,
            )
        ):
            raise RuntimeError(f"equivalence failed for {name}")
        proof_variants.append(
            {
                "name": name,
                "target": new["target"],
                "predecessor_source_sha256": old["source_sha256"],
                "current_source_sha256": new["source_sha256"],
                "predecessor_kernel_symbol": old["kernel_symbol"],
                "current_kernel_symbol": new["kernel_symbol"],
                "normalized_source_sha256": hashlib.sha256(
                    new_normalized.encode()
                ).hexdigest(),
                "symbol_rename_occurrences": 1,
                "abi_sha256": _canonical_json_sha256(new["abi"]),
                "compile_flags_sha256": new["compile_flags_sha256"],
                "predecessor_sass_sha256": old["sass_sha256"],
                "current_sass_sha256": new["sass_sha256"],
                "predecessor_resource_usage_sha256": old["resource_usage_sha256"],
                "current_resource_usage_sha256": new["resource_usage_sha256"],
                "execution_input_sha256": new["execution_input_sha256"],
                "predecessor_execution_sha256": old["execution_sha256"],
                "current_execution_sha256": new["execution_sha256"],
                "source_transform_exact": True,
                "predecessor_alignment_assert_normalized": True,
                "abi_equal": True,
                "compile_flags_equal": True,
                "launch_equal": True,
                "sass_equal": True,
                "resource_usage_equal": True,
                "execution_bitwise_equal": True,
            }
        )

    old_routes = predecessor_result["routes"]
    new_routes = current_result["routes"]
    measured_names = _row_variant_names(rows_root)
    mismatches = []
    for index, (old, new, measured) in enumerate(
        zip(old_routes, new_routes, measured_names, strict=True)
    ):
        if old != new or old["variant_name"] != measured:
            mismatches.append(index)
    if len(old_routes) != len(_FULL_DOMAIN_SHAPES) or mismatches:
        raise RuntimeError(f"full-domain route equivalence failed: {mismatches[:8]}")
    used_variants = sorted({record["variant_name"] for record in new_routes})
    shape_inventory = [
        {"num_heads": heads, "num_rows": rows} for heads, rows in _FULL_DOMAIN_SHAPES
    ]
    receipt = {
        "schema": _SCHEMA,
        "verifier": {
            "script_sha256": _file_sha256(script_path),
            "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "")),
            "node": current_result["node"],
            "gpu_uuid": current_result["gpu_uuid"],
            "gpu_name": current_result["gpu_name"],
            "compute_capability": current_result["compute_capability"],
            **current_result["toolchain"],
            "toolchain_sha256": current_result["toolchain_sha256"],
        },
        "predecessor": {
            "commit": predecessor_commit,
            "manifest_sha256": predecessor_manifest_sha256,
        },
        "current": {
            "commit": current_commit,
            "program_identity_sha256": current_result["program_identity_sha256"],
            "registry_sha256": _file_sha256(
                current_root / "flashinfer/jit/cake_fused_kda_decode.py"
            ),
            "binding_sha256": _file_sha256(
                current_csrc / "cake_fused_kda_decode_binding.cuh"
            ),
        },
        "variants": proof_variants,
        "route_equivalence": {
            "shape_count": len(_FULL_DOMAIN_SHAPES),
            "shape_inventory_sha256": _canonical_json_sha256(shape_inventory),
            "predecessor_routes_sha256": _canonical_json_sha256(old_routes),
            "current_routes_sha256": _canonical_json_sha256(new_routes),
            "missing": sum(record["variant_name"] is None for record in new_routes),
            "mismatches": mismatches,
            "used_variants": used_variants,
        },
        "result": {
            "variant_count": len(proof_variants),
            "route_count": len(new_routes),
            "all_variants_passed": True,
            "all_routes_passed": True,
            "eligible_for_timing_inheritance": True,
        },
    }
    if not receipt["verifier"]["slurm_job_id"]:
        raise RuntimeError("equivalence verifier must run inside a Slurm job")
    _write_json_atomic(output, receipt)
    _record_progress("orchestrator: equivalence receipt sealed")
    print(f"wrote equivalence receipt: {output}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root")
    parser.add_argument("--predecessor-root")
    parser.add_argument("--predecessor-checkpoint")
    parser.add_argument("--predecessor-rows-root")
    parser.add_argument("--work-root")
    parser.add_argument("--reuse-work-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--worker-side", choices=("predecessor", "current"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-root", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_side is not None:
        if args.worker_root is None or args.worker_output is None:
            parser.error("worker mode requires --worker-root and --worker-output")
    elif any(
        getattr(args, name) is None
        for name in (
            "current_root",
            "predecessor_root",
            "predecessor_checkpoint",
            "predecessor_rows_root",
            "work_root",
            "output",
        )
    ):
        parser.error("orchestrator mode requires all proof paths")
    return args


def main():
    args = _parse_args()
    if args.worker_side is not None:
        _worker(
            argparse.Namespace(
                side=args.worker_side,
                root=args.worker_root,
                output=args.worker_output,
            )
        )
    else:
        _orchestrator(args)


if __name__ == "__main__":
    main()
