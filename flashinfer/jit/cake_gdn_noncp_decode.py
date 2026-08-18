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

"""Checksum-verified JIT loader for the source-only Cake GDN backend."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple

from filelock import FileLock
from tvm_ffi import cpp

from . import env as jit_env
from .cpp_ext import get_cuda_path, get_nvcc_parallelism_flags

CakeGDNArch = Literal["sm_100a", "sm_103a"]

_EXPORT_SCHEMA = "flashinfer-gdn-noncp-decode-standalone-export-v1"
_MANIFEST_SHA256 = "474ee84e279b2c82b42e9e8146f5b05618968005a07635c8f382fe414127a4e4"
_GENERATOR_COMMIT = "11cb68d34a2d52710c599ab1d746e261dee5ddae"
_BASELINE_REVISIONS = {
    "decode": "1bc1cd99461e61fe99a4a35aa873879ac08130b5",
    "prefill": "8044d94bf9acc5369857baf88d28906bb32bf264",
}
_ARCH_ACTIVE_CLUSTERS: dict[CakeGDNArch, int] = {
    "sm_100a": 148,
    "sm_103a": 160,
}


class CakeGDNUnsupportedError(NotImplementedError):
    """The explicit Cake backend cannot serve the requested GDN contract."""


class CakeGDNRoute(NamedTuple):
    route_id: str
    variant_name: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "gdn" / "cake" / "noncp_decode"
    if installed.exists():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "gdn"
        / "cake"
        / "noncp_decode"
    )
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "frozen Cake GDN non-CP/decode sources were not found; checked "
        f"{installed} and {checkout}"
    )


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _source_dir() / "manifest.json"
    observed_digest = _sha256(path)
    if observed_digest != _MANIFEST_SHA256:
        raise RuntimeError(
            f"Cake GDN manifest drift at {path}: "
            f"expected {_MANIFEST_SHA256}, got {observed_digest}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    observed = (
        manifest.get("schema"),
        manifest.get("generator_commit"),
        manifest.get("source_only"),
        manifest.get("binary_artifacts"),
        manifest.get("baseline_revisions"),
        manifest.get("contract_row_count"),
        manifest.get("architecture_row_count"),
        manifest.get("admitted_architecture_rows"),
        manifest.get("fail_closed_architecture_rows"),
        manifest.get("variant_count"),
        len(manifest.get("variants", [])),
        manifest.get("scope", {}).get("explicit_backend_policy"),
    )
    expected = (
        _EXPORT_SCHEMA,
        _GENERATOR_COMMIT,
        True,
        False,
        _BASELINE_REVISIONS,
        1761,
        3522,
        3462,
        60,
        77,
        77,
        "one listed Cake variant or fail closed; no external fallback",
    )
    if observed != expected:
        raise RuntimeError(
            "Cake GDN manifest does not match the frozen support contract: "
            f"expected {expected!r}, got {observed!r}"
        )
    return manifest


def _kernel_record(name: str) -> dict[str, Any]:
    records = [record for record in _manifest()["variants"] if record["name"] == name]
    if len(records) != 1:
        raise ValueError(f"unknown Cake GDN kernel: {name!r}")
    return records[0]


def _cuda_record(record: dict[str, Any], arch: CakeGDNArch) -> dict[str, Any]:
    outputs = [
        output for output in record["outputs"] if arch in output["architectures"]
    ]
    if len(outputs) != 1:
        raise ValueError(f"kernel {record['name']!r} does not support {arch}")
    return outputs[0]


def _compile_cubin(
    source: Path,
    *,
    arch: CakeGDNArch,
    digest: str,
    compile_options: tuple[str, ...],
) -> bytes:
    cache_dir = jit_env.FLASHINFER_JIT_DIR / "cake_gdn_noncp_decode" / arch
    cache_dir.mkdir(parents=True, exist_ok=True)
    cubin = cache_dir / f"{source.stem}-{digest[:16]}.cubin"
    lock = FileLock(f"{cubin}.lock", thread_local=False)
    with lock:
        if not cubin.exists():
            nvcc = Path(get_cuda_path()) / "bin" / "nvcc"
            if not nvcc.is_file():
                raise RuntimeError(f"nvcc was not found at {nvcc}")
            with tempfile.NamedTemporaryFile(
                dir=cache_dir,
                prefix=f".{cubin.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                command = [
                    str(nvcc),
                    "--cubin",
                    "--std=c++17",
                    "-O3",
                    f"--gpu-architecture={arch}",
                    *compile_options,
                    *get_nvcc_parallelism_flags(),
                    str(source),
                    "-o",
                    str(temporary),
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"failed to compile {source.name} for {arch}:\n{result.stdout}"
                    )
                os.replace(temporary, cubin)
            finally:
                temporary.unlink(missing_ok=True)
    return cubin.read_bytes()


@functools.cache
def load_cake_gdn_kernel(name: str, arch: CakeGDNArch):
    """Compile and load one checksum-verified Cake GDN host entrypoint."""

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise ValueError(f"unsupported Cake GDN architecture: {arch!r}")
    record = _kernel_record(name)
    cuda = _cuda_record(record, arch)
    host = record["host_binding"]
    root = _source_dir()
    cuda_path = root / cuda["path"]
    host_path = root / host["path"]
    headers = _manifest().get("cuda_headers", [])
    sources = [
        (cuda_path, cuda["sha256"]),
        (host_path, host["sha256"]),
        *((root / header["path"], header["sha256"]) for header in headers),
    ]
    for path, expected in sources:
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Cake GDN source drift at {path}: expected {expected}, got {observed}"
            )
    compile_options = tuple(record.get("compile_options", ()))
    unsupported_options = set(compile_options) - {"--use_fast_math"}
    if unsupported_options:
        raise RuntimeError(
            f"unsupported Cake GDN compile options: {sorted(unsupported_options)!r}"
        )
    compile_digest = hashlib.sha256(
        "\0".join(
            [
                arch,
                cuda["sha256"],
                *(header["sha256"] for header in headers),
                *compile_options,
            ]
        ).encode()
    ).hexdigest()
    cubin = _compile_cubin(
        cuda_path,
        arch=arch,
        digest=compile_digest,
        compile_options=compile_options,
    )
    module_digest = hashlib.sha256(
        f"{compile_digest}\0{host['sha256']}".encode()
    ).hexdigest()
    module = cpp.load_inline(
        f"flashinfer_cake_gdn_{name}_{arch}_{module_digest[:12]}",
        cpp_sources=host_path.read_text(encoding="utf-8"),
        embed_cubin={host["module_ident"]: cubin},
        extra_include_paths=[
            str(Path(get_cuda_path()) / "include"),
            str(root.parents[2]),
            str(root.parents[3] / "include"),
        ],
        extra_ldflags=["-lcuda"],
    )
    return module[host["entry"]]


def _power_of_two_log2(value: int) -> int:
    return value.bit_length() - 1 if value > 0 and value & (value - 1) == 0 else -1


def _prefill_schedule_attr(
    *,
    dvsplit: bool,
    use_initial_state: bool,
    io_dtype: str,
    state_dtype: str,
    single_chunk: bool,
) -> str:
    stem = (
        "flashinfer_blackwell_gdn_prefill_dvsplit"
        if dvsplit
        else "flashinfer_blackwell_gdn_prefill"
    )
    if (
        single_chunk
        and io_dtype == "bfloat16"
        and not use_initial_state
        and state_dtype == "float32"
    ):
        return f"{stem}_single_chunk"
    if state_dtype in {"float16", "float8_e4m3fn", "float8_e5m2"}:
        suffix = {
            "float16": "f16state",
            "float8_e4m3fn": "e4m3state",
            "float8_e5m2": "e5m2state",
        }[state_dtype]
        return f"{stem}_initial_{suffix}"
    pieces = [stem]
    if use_initial_state:
        pieces.append("initial")
    if io_dtype == "float16":
        pieces.append("f16io")
    if state_dtype == "bfloat16":
        pieces.append("bf16state")
    return "_".join(pieces)


def _variant_for(
    *, domain: str, schedule_attr: str, specializations: dict[str, int | float]
) -> dict[str, Any]:
    source_schedule = {
        "decode": "cake.generated.decode",
        "prefill": "cake.generated.prefill",
    }[domain]
    expected_source = f"{source_schedule}:{schedule_attr}"
    matches = [
        record
        for record in _manifest()["variants"]
        if record["domain"] == domain
        and record["source_schedule"] == expected_source
        and record["specializations"] == specializations
        and record["tma_abi"] == "pointer"
    ]
    if len(matches) != 1:
        raise CakeGDNUnsupportedError(
            "no exact frozen Cake GDN variant for "
            f"{expected_source} with {specializations!r}"
        )
    return matches[0]


@functools.cache
def select_cake_gdn_prefill_variant(
    *,
    arch: CakeGDNArch,
    io_dtype: str,
    state_dtype: str,
    num_seqs: int,
    total_seq_len: int,
    max_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    store_final_state: bool,
    checkpoint_every_n_tokens: int,
    use_state_indices: bool,
) -> CakeGDNRoute:
    """Resolve host-visible non-CP metadata to one frozen Cake variant."""

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported architecture {arch}")
    if io_dtype not in {"float16", "bfloat16"}:
        raise CakeGDNUnsupportedError(f"unsupported I/O dtype {io_dtype}")
    if state_dtype not in {
        "float32",
        "bfloat16",
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    }:
        raise CakeGDNUnsupportedError(f"unsupported state dtype {state_dtype}")
    min_heads = min(num_q_heads, num_v_heads)
    num_o_heads = max(num_q_heads, num_v_heads)
    if (
        min_heads <= 0
        or num_k_heads != min_heads
        or num_o_heads % min_heads
        or _power_of_two_log2(num_o_heads // min_heads) < 0
        or _power_of_two_log2(num_o_heads) < 0
    ):
        raise CakeGDNUnsupportedError("unsupported GDN head mapping")
    if (
        num_seqs <= 0
        or total_seq_len <= 0
        or max_seq_len <= 0
        or max_seq_len > total_seq_len
    ):
        raise CakeGDNUnsupportedError(
            "Cake GDN prefill requires positive, consistent sequence metadata"
        )
    if checkpoint_every_n_tokens < 0 or (
        checkpoint_every_n_tokens and checkpoint_every_n_tokens % 64
    ):
        raise CakeGDNUnsupportedError(
            "checkpoint interval must be zero or a positive multiple of 64"
        )
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if use_state_indices and (not use_initial_state or not store_final_state):
        raise CakeGDNUnsupportedError(
            "indexed state requires initial and final state"
        )
    low_precision_state = state_dtype in {
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    }
    if low_precision_state and (
        io_dtype != "bfloat16"
        or not use_initial_state
        or not store_final_state
        or enable_checkpoints
        or use_state_indices
    ):
        raise CakeGDNUnsupportedError(
            "low-precision state requires BF16 I/O, initial+final state, "
            "no checkpoints, and packed state"
        )
    if enable_checkpoints and (
        io_dtype != "float16"
        or state_dtype != "float32"
        or use_initial_state
        or not store_final_state
        or use_state_indices
    ):
        raise CakeGDNUnsupportedError(
            "checkpoint route requires FP16 I/O, FP32 state, no initial state, "
            "final state, and packed state"
        )
    dvsplit = (
        4 * num_seqs * num_o_heads <= _ARCH_ACTIVE_CLUSTERS[arch]
    )
    if low_precision_state and not dvsplit:
        raise CakeGDNUnsupportedError(
            "low-precision state requires the DV-split physical schedule"
        )
    single_chunk = (
        max_seq_len <= 64
        and not use_initial_state
        and not store_final_state
        and not enable_checkpoints
    )
    schedule_attr = _prefill_schedule_attr(
        dvsplit=dvsplit,
        use_initial_state=use_initial_state,
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        single_chunk=single_chunk,
    )
    specializations = {
        "ENABLE_CHECKPOINTS": int(enable_checkpoints),
        "HEAD_GROUP_LOG2": _power_of_two_log2(num_o_heads // min_heads),
        "IS_GQA": int(num_q_heads >= num_v_heads),
        "NUM_O_HEADS_LOG2": _power_of_two_log2(num_o_heads),
        "SINGLE_CHUNK_NO_STATE": int(single_chunk),
        "STORE_FINAL_STATE": int(store_final_state),
        "USE_INITIAL_STATE": int(use_initial_state),
        "USE_STATE_INDICES": int(use_state_indices),
    }
    record = _variant_for(
        domain="prefill",
        schedule_attr=schedule_attr,
        specializations=specializations,
    )
    if single_chunk:
        route = "cake.gdn_prefill.noncp.single_chunk"
    elif enable_checkpoints:
        route = "cake.gdn_prefill.noncp.checkpoints"
    else:
        route = "cake.gdn_prefill.noncp"
    return CakeGDNRoute(
        f"{route}.{'dvsplit' if dvsplit else 'full_dv'}", record["name"]
    )


@functools.cache
def select_cake_gdn_decode_variant(
    *,
    arch: CakeGDNArch,
    batch_size: int,
    io_dtype: str,
    state_dtype: str,
    head_size: int,
    layout: str,
    num_k_heads: int,
    num_q_heads: int,
    num_v_heads: int,
    scale: float,
    seq_len: int,
    use_qk_l2norm: bool,
    strided_inputs: bool = False,
    disable_state_update: bool = False,
    cache_intermediate_states: bool = False,
    cache_steps: int = 0,
) -> CakeGDNRoute:
    """Resolve one frozen FP32 T=1 or exact promoted BF16 serving row."""

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported architecture {arch}")
    if io_dtype != "bfloat16" or state_dtype not in {"float32", "bfloat16"}:
        raise CakeGDNUnsupportedError(
            "Cake decode requires BF16 I/O and FP32 or BF16 state"
        )
    if head_size != 128:
        raise CakeGDNUnsupportedError(
            "Cake decode requires K=V=128"
        )
    if not use_qk_l2norm:
        raise CakeGDNUnsupportedError(
            "Cake T=1 child contract requires in-kernel Q/K L2 normalization"
        )
    if (
        num_q_heads <= 0
        or num_k_heads != num_q_heads
        or num_v_heads <= 0
        or num_v_heads % num_q_heads
    ):
        raise CakeGDNUnsupportedError("unsupported decode head mapping")
    if batch_size <= 0:
        raise CakeGDNUnsupportedError("decode batch size must be positive")
    if state_dtype == "bfloat16":
        promoted = {
            (4, 1, 16, 32, True, False, False, 0),
            (4, 2, 16, 32, False, True, True, 4),
            (8, 3, 16, 64, True, True, True, 3),
            (8, 4, 16, 64, True, True, True, 4),
            (8, 2, 16, 64, True, False, False, 0),
            (8, 4, 16, 64, True, False, True, 5),
        }
        key = (
            batch_size,
            seq_len,
            num_q_heads,
            num_v_heads,
            strided_inputs,
            disable_state_update,
            cache_intermediate_states,
            cache_steps,
        )
        if layout != "pretranspose" or num_k_heads != num_q_heads or key not in promoted:
            raise CakeGDNUnsupportedError(
                "BF16 decode is limited to the six exact promoted indexed/verify rows"
            )
        state_heads = batch_size * num_v_heads
        tile_v = 128 if state_heads >= 1024 else 64 if state_heads >= 512 else 32
        update_state = not cache_intermediate_states
        record = _variant_for(
            domain="decode",
            schedule_attr="gdn_decode_pretranspose_mtp_t4_bf16state_wide128",
            specializations={
                "CACHE_INTERMEDIATE_STATES": int(cache_intermediate_states),
                "H": num_q_heads,
                "HV": num_v_heads,
                "INTERMEDIATE_BATCH_STRIDE": (
                    cache_steps * num_v_heads * 128 * 128
                    if cache_intermediate_states
                    else 128 * 128
                ),
                "INTERMEDIATE_TOKEN_STRIDE": (
                    num_v_heads * 128 * 128
                    if cache_intermediate_states
                    else 128 * 128
                ),
                "SCALE": scale,
                "STRIDED_INPUTS": int(strided_inputs),
                "TILE_V_WIDE": tile_v,
                "T_STEPS": seq_len,
                "UPDATE_STATE": int(update_state),
            },
        )
        if seq_len == 1:
            route = "cake.gdn_decode.indexed_bf16_t1"
        elif disable_state_update:
            route = f"cake.gdn_decode.indexed_bf16_verify_t{seq_len}"
        elif cache_intermediate_states:
            route = f"cake.gdn_decode.indexed_bf16_checkpoint_t{seq_len}"
        else:
            route = f"cake.gdn_decode.indexed_bf16_update_t{seq_len}"
        return CakeGDNRoute(f"{route}.wide{tile_v}", record["name"])

    if seq_len != 1:
        raise CakeGDNUnsupportedError(
            "FP32-state child contract requires sequence length 1"
        )
    if disable_state_update or cache_intermediate_states or cache_steps:
        raise CakeGDNUnsupportedError(
            "FP32-state T=1 child contract requires state update and no cache"
        )
    if layout == "pretranspose":
        schedule_attr = "gdn_decode_pretranspose_splitv8"
        route = "cake.gdn_decode.indexed_fp32_t1_splitv8"
    elif layout == "nontranspose":
        if batch_size < 32:
            schedule_attr = "gdn_decode_nontranspose_fp32_t1_small"
            route = "cake.gdn_decode.direct_fp32_t1_nontranspose_small"
        else:
            schedule_attr = "gdn_decode_nontranspose_fp32_t1"
            route = "cake.gdn_decode.direct_fp32_t1_nontranspose_large"
    else:
        raise CakeGDNUnsupportedError(
            f"unsupported decode state layout {layout}"
        )
    record = _variant_for(
        domain="decode",
        schedule_attr=schedule_attr,
        specializations={"H": num_q_heads, "HV": num_v_heads, "SCALE": scale},
    )
    return CakeGDNRoute(route, record["name"])


def arch_for_compute_capability(major: int, minor: int) -> CakeGDNArch:
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise CakeGDNUnsupportedError(
        f"Cake GDN supports only SM100a/SM103a, got compute capability {major}.{minor}"
    )


__all__ = [
    "CakeGDNArch",
    "CakeGDNRoute",
    "CakeGDNUnsupportedError",
    "arch_for_compute_capability",
    "load_cake_gdn_kernel",
    "select_cake_gdn_decode_variant",
    "select_cake_gdn_prefill_variant",
]
