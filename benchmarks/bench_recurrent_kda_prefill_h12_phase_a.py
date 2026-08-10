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

"""Phase-A H12 recurrent-KDA prefill correctness and CUPTI evidence harness.

The checked-in preset is the complete six-case denominator from FlashInfer
#4351.  Every FlashInfer timing sample invokes the public ``recurrent_kda``
API.  Its reportable number is the first-to-last correlated GPU activity span,
which must contain one beta-pack activity followed by the M128 recurrence.
The recurrence-only (prepared) activity is retained under a separate field and
is never substituted for the public number.

The harness also verifies an in-allocation force-build of the exact pinned
MoonshotAI/FlashKDA checkout and requires a clean, identifiable FLA checkout
whose Triton path is forced with ``FLA_FLASH_KDA=0``.  Correctness covers the
public output and complete final state against a direct BF16-state recurrence,
the pinned FlashKDA implementation, and FLA/Triton.  Promotion additionally
requires the exact changed-beta CUDA Graph test and matching SM100a+SM103a
receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from kda_h12_evidence import (
    EVIDENCE_REPORT_SCHEMA_VERSION,
    FLASH_KDA_BASELINE_REVISION,
    FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
    FLASHINFER_H12_ROUTE_REVISION,
    FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION,
    GRAPH_TEST_NODE_ID,
    GRAPH_TEST_SOURCE,
    GRAPH_TEST_SOURCE_LINE_RANGE,
    CpuBracket,
    EvidencePreset,
    GpuActivity,
    LaunchActivity,
    SUPPORTED_ARCHITECTURES,
    correlate_samples,
    load_preset,
    summarize_samples,
    verify_flash_kda_provenance,
)


BENCHMARKS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARKS_DIR.parent
DEFAULT_PRESET = BENCHMARKS_DIR / "presets" / "recurrent_kda_prefill_h12_phase_a.json"
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2


@dataclass
class CaseRuntime:
    """GPU tensors and callable paths for one deterministic preset case."""

    metadata: dict
    tensors: dict[str, Any]
    initial_state_seed: Any
    candidate_output: Any
    candidate_state_pool: Any
    candidate_run: Callable[[], object]
    candidate_reset: Callable[[], None]
    flash_kda_output: Any
    flash_kda_final_state: Any
    flash_kda_raw_run: Callable[[], object]
    flash_kda_adapted_run: Callable[[], object]
    flash_kda_adapted_reset: Callable[[], None]
    fla_run: Callable[[], object]


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git provenance query failed at {root}: {error.output.strip()}"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_provenance(stack: SimpleNamespace) -> dict:
    source_commit = _git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    worktree_changes = _git_output(
        REPOSITORY_ROOT,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if worktree_changes:
        raise RuntimeError(
            "Phase-A evidence requires a FlashInfer checkout with no tracked "
            "or nonignored untracked changes; "
            f"found:\n{worktree_changes}"
        )
    required_ancestors = {
        "phase_a_upstream_main": FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION,
        "non_aligned_h12_public_route_pr_4351": FLASHINFER_H12_ROUTE_REVISION,
    }
    for label, revision in required_ancestors.items():
        try:
            _git_output(
                REPOSITORY_ROOT,
                "merge-base",
                "--is-ancestor",
                revision,
                source_commit,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"FlashInfer source does not contain required {label} "
                f"revision {revision}"
            ) from error

    imported_module_paths = {
        name: Path(module.__file__).resolve(strict=True)
        for name, module in stack.candidate_modules.items()
    }
    for name, path in imported_module_paths.items():
        if not path.is_relative_to(REPOSITORY_ROOT):
            raise RuntimeError(
                f"{name} must be imported from the verified FlashInfer "
                f"worktree: module={path}, worktree={REPOSITORY_ROOT}"
            )
    source_paths = (
        Path("flashinfer/kda.py"),
        Path("flashinfer/kda_prefill.py"),
        Path("csrc/kda/flashkda_binding_common.cuh"),
        Path("csrc/kda/flashkda_bf16_fused_m128_binding.cu"),
        Path("csrc/kda/flashkda_bf16_fused_m128.cu"),
        Path("benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py"),
        Path("benchmarks/build_flash_kda_phase_a.py"),
        Path("benchmarks/kda_h12_evidence.py"),
        Path("benchmarks/presets/recurrent_kda_prefill_h12_phase_a.json"),
        Path("benchmarks/reduce_kda_h12_phase_a.py"),
        Path(GRAPH_TEST_SOURCE),
    )
    return {
        "repository": "https://github.com/flashinfer-ai/flashinfer.git",
        "source_dir": str(REPOSITORY_ROOT),
        "source_commit": source_commit,
        "required_ancestor_revisions": required_ancestors,
        "worktree_clean_including_untracked": True,
        "imported_module_paths": {
            name: str(path) for name, path in imported_module_paths.items()
        },
        "imported_module_sha256": {
            name: _sha256(path) for name, path in imported_module_paths.items()
        },
        "source_sha256": {
            str(path): _sha256(REPOSITORY_ROOT / path) for path in source_paths
        },
    }


def _import_gpu_stack() -> SimpleNamespace:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU evidence requires a FlashInfer development environment with torch"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-A H12 evidence requires a CUDA GPU")

    kda_module = importlib.import_module("flashinfer.kda")
    prefill_module = importlib.import_module("flashinfer.kda_prefill")
    testing_utils = importlib.import_module("flashinfer.testing.utils")
    utils_module = importlib.import_module("flashinfer.utils")

    return SimpleNamespace(
        torch=torch,
        recurrent_kda=kda_module.recurrent_kda,
        workspace_type=prefill_module.RecurrentKDAPrefillWorkspace,
        get_compute_capability=utils_module.get_compute_capability,
        get_l2_cache_size=testing_utils.get_l2_cache_size,
        candidate_modules={
            "flashinfer.kda": kda_module,
            "flashinfer.kda_prefill": prefill_module,
        },
    )


def _require_cupti():
    try:
        from cupti import cupti
    except ImportError as error:
        raise RuntimeError(
            "reportable Phase-A timing requires cupti-python >= 13"
        ) from error
    try:
        package_version = importlib.metadata.version("cupti-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "reportable Phase-A timing requires an identifiable cupti-python package"
        ) from error
    if int(package_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"reportable Phase-A timing requires cupti-python >= 13, got {package_version}"
        )
    return cupti, package_version


def _hardware_metadata(stack: SimpleNamespace) -> dict:
    torch = stack.torch
    device = torch.device("cuda")
    capability = stack.get_compute_capability(device)
    if capability not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "Phase-A H12 prefill requires exact CC 10.0 (SM100a) or "
            f"CC 10.3 (SM103a), got CC {capability[0]}.{capability[1]}"
        )
    properties = torch.cuda.get_device_properties(device)
    device_index = torch.cuda.current_device()
    device_uuid = str(getattr(properties, "uuid", "unavailable"))
    return {
        "device_name": properties.name,
        "device_index": device_index,
        "device_uuid": device_uuid,
        "compute_capability": list(capability),
        "cuda_arch": SUPPORTED_ARCHITECTURES[capability],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "l2_cache_bytes": stack.get_l2_cache_size(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _run_changed_beta_graph_test() -> dict:
    """Run and receipt the exact H6/H12 changed-beta CUDA Graph regression."""

    command = [sys.executable, "-m", "pytest", "-q", GRAPH_TEST_NODE_ID]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    source_path = REPOSITORY_ROOT / GRAPH_TEST_SOURCE
    return {
        "source": GRAPH_TEST_SOURCE,
        "source_line_range": list(GRAPH_TEST_SOURCE_LINE_RANGE),
        "source_sha256": _sha256(source_path),
        "node_id": GRAPH_TEST_NODE_ID,
        "parameterization": {"num_heads": [6, 12]},
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _load_flash_kda(
    source_dir: Path,
    build_manifest_path: Path,
) -> tuple[Any, dict]:
    source_dir = source_dir.resolve(strict=True)
    sys.path.insert(0, str(source_dir))
    try:
        flash_kda = importlib.import_module("flash_kda")
        extension = importlib.import_module("flash_kda_C")
    except ImportError as error:
        raise RuntimeError(
            "install/build MoonshotAI/FlashKDA at the pinned revision "
            f"{FLASH_KDA_BASELINE_REVISION} before running this harness"
        ) from error
    provenance = verify_flash_kda_provenance(
        package_path=Path(flash_kda.__file__),
        extension_path=Path(extension.__file__),
        source_dir=source_dir,
        build_manifest_path=build_manifest_path,
    )
    return flash_kda, provenance


def _package_git_facts(path: Path) -> dict:
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"required FLA package is not identifiable as a Git checkout: {path}"
        )
    root = Path(result.stdout.strip()).resolve()
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise RuntimeError(
            "required FLA checkout has tracked or nonignored untracked changes:\n"
            f"{status}"
        )
    revision = _git_output(root, "rev-parse", "HEAD")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError(
            f"required FLA checkout has invalid Git revision {revision!r}"
        )
    return {
        "git_source_dir": str(root),
        "git_revision": revision,
        "worktree_clean_including_untracked": True,
    }


def _load_required_fla() -> tuple[Callable, dict]:
    # FLA's FlashKDA backend is enabled by default whenever flash_kda is
    # importable.  Force zero before importing FLA so this comparison remains
    # the independent Triton implementation rather than the pinned peer again.
    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    preimported = sorted(
        name for name in sys.modules if name == "fla" or name.startswith("fla.")
    )
    if preimported:
        raise RuntimeError(
            "FLA was imported before its backend-dispatch policy could be "
            f"frozen: {preimported!r}"
        )
    try:
        fla = importlib.import_module("fla")
        kda_ops = importlib.import_module("fla.ops.kda")
        chunk_kda = kda_ops.chunk_kda
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Phase-A evidence requires an installed FLA/Triton checkout"
        ) from error

    package_path = Path(fla.__file__).resolve(strict=True)
    op_path = Path(importlib.import_module("fla.ops.kda.chunk").__file__).resolve(
        strict=True
    )
    git_facts = _package_git_facts(op_path)
    git_source_dir = Path(git_facts["git_source_dir"])
    for label, path in (("package", package_path), ("KDA op", op_path)):
        if not path.is_relative_to(git_source_dir):
            raise RuntimeError(
                f"FLA {label} must resolve inside its verified Git checkout: {path}"
            )
    distribution_version = None
    for distribution in ("flash-linear-attention", "fla"):
        try:
            distribution_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return chunk_kda, {
        "available": True,
        "implementation": "fla.ops.kda.chunk_kda (Triton forced)",
        "distribution_version": distribution_version,
        "package_path": str(package_path),
        "package_sha256": _sha256(package_path),
        "op_path": str(op_path),
        "op_sha256": _sha256(op_path),
        "forced_environment": {
            "FLA_FLASH_KDA": "0",
            "FLA_DISABLE_BACKEND_DISPATCH": "1",
        },
        **git_facts,
    }


def _offsets(seq_lens: tuple[int, ...]) -> list[int]:
    values = [0]
    for seq_len in seq_lens:
        values.append(values[-1] + seq_len)
    return values


def _make_state_pool(initial_state, rotations: int):
    return initial_state.unsqueeze(0).expand(rotations, *initial_state.shape).clone()


def _make_case_runtime(
    *,
    stack: SimpleNamespace,
    case,
    rotations: int,
    flash_kda,
    fla_chunk_kda: Callable,
) -> CaseRuntime:
    torch = stack.torch
    device = torch.device("cuda")
    num_heads = 12
    head_dim = 128
    total_tokens = case.total_tokens
    shape = (1, total_tokens, num_heads, head_dim)
    generator = torch.Generator(device=device).manual_seed(case.seed)
    q = torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
    k = torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
    g = torch.randn(shape, generator=generator, device=device).to(torch.bfloat16)
    beta = torch.randn(
        (1, total_tokens, num_heads),
        generator=generator,
        device=device,
    ).to(torch.bfloat16)
    A_log = torch.log(
        torch.empty(num_heads, dtype=torch.float32, device=device).uniform_(
            1.0,
            16.0,
            generator=generator,
        )
    )
    dt_bias = torch.randn(
        (num_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    initial_state_seed = (
        torch.randn(
            (len(case.seq_lens), num_heads, head_dim, head_dim),
            generator=generator,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    offsets = _offsets(case.seq_lens)
    packed = case.layout == "packed"
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device=device) if packed else None
    )
    cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int64) if packed else None
    seq_order = (
        torch.tensor(
            sorted(
                range(len(case.seq_lens)),
                key=case.seq_lens.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device=device,
        )
        if packed
        else None
    )
    scale = head_dim**-0.5
    candidate_output = torch.empty_like(q)
    candidate_workspace = stack.workspace_type(device)
    candidate_state_pool = _make_state_pool(initial_state_seed, rotations)
    candidate_cursor = [0]

    candidate_kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "output": candidate_output,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "cu_seqlens": cu_seqlens,
        "beta_is_logit": True,
        "seq_order": seq_order,
        "prefill_workspace": candidate_workspace,
    }

    def candidate_run():
        state_index = candidate_cursor[0]
        if state_index >= rotations:
            raise RuntimeError(
                f"candidate state rotations exhausted: {state_index} >= {rotations}"
            )
        candidate_cursor[0] += 1
        return stack.recurrent_kda(
            **candidate_kwargs,
            initial_state=candidate_state_pool[state_index],
        )

    def candidate_reset() -> None:
        candidate_state_pool.copy_(initial_state_seed.unsqueeze(0))
        candidate_cursor[0] = 0

    flash_kda_output = torch.empty_like(q)
    flash_kda_final_state = torch.empty_like(initial_state_seed)
    flash_kda_workspace = torch.empty(
        flash_kda.get_workspace_size(
            total_tokens,
            num_heads,
            len(case.seq_lens),
        ),
        dtype=torch.uint8,
        device=device,
    )

    def flash_kda_raw_run():
        return flash_kda._fwd_raw(
            q,
            k,
            v,
            g,
            beta,
            scale,
            flash_kda_output,
            flash_kda_workspace,
            A_log,
            dt_bias,
            -5.0,
            initial_state=initial_state_seed,
            final_state=flash_kda_final_state,
            cu_seqlens=cu_seqlens,
        )

    flash_kda_adapted_state_pool = _make_state_pool(initial_state_seed, rotations)
    flash_kda_adapted_cursor = [0]

    def flash_kda_adapted_run():
        state_index = flash_kda_adapted_cursor[0]
        if state_index >= rotations:
            raise RuntimeError(
                "FlashKDA adapted state rotations exhausted: "
                f"{state_index} >= {rotations}"
            )
        flash_kda_adapted_cursor[0] += 1
        state = flash_kda_adapted_state_pool[state_index]
        flash_kda._fwd_raw(
            q,
            k,
            v,
            g,
            beta,
            scale,
            flash_kda_output,
            flash_kda_workspace,
            A_log,
            dt_bias,
            -5.0,
            initial_state=state,
            final_state=flash_kda_final_state,
            cu_seqlens=cu_seqlens,
        )
        state.copy_(flash_kda_final_state)

    def flash_kda_adapted_reset() -> None:
        flash_kda_adapted_state_pool.copy_(initial_state_seed.unsqueeze(0))
        flash_kda_adapted_cursor[0] = 0

    fla_initial_state = initial_state_seed.float()

    def fla_run():
        with torch.inference_mode():
            return fla_chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=fla_initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
    }
    return CaseRuntime(
        metadata={
            "name": case.name,
            "layout": case.layout,
            "seq_lens": list(case.seq_lens),
            "total_tokens": total_tokens,
            "num_sequences": len(case.seq_lens),
            "num_heads": num_heads,
            "head_dim_qk": head_dim,
            "head_dim_vo": head_dim,
            "dtype": "bfloat16",
            "initial_state": "provided_bfloat16",
            "seed": case.seed,
            "variant": "m128",
        },
        tensors=tensors,
        initial_state_seed=initial_state_seed,
        candidate_output=candidate_output,
        candidate_state_pool=candidate_state_pool,
        candidate_run=candidate_run,
        candidate_reset=candidate_reset,
        flash_kda_output=flash_kda_output,
        flash_kda_final_state=flash_kda_final_state,
        flash_kda_raw_run=flash_kda_raw_run,
        flash_kda_adapted_run=flash_kda_adapted_run,
        flash_kda_adapted_reset=flash_kda_adapted_reset,
        fla_run=fla_run,
    )


def _independent_bf16_recurrence(torch, runtime: CaseRuntime):
    """Direct per-token recurrence with a BF16 state store after every token."""

    tensors = runtime.tensors
    q = tensors["q"]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    q_flat = torch.nn.functional.normalize(q.float(), dim=-1).reshape(
        -1,
        num_heads,
        head_dim,
    )
    k_flat = torch.nn.functional.normalize(
        tensors["k"].float(),
        dim=-1,
    ).reshape(-1, num_heads, head_dim)
    v_flat = tensors["v"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(tensors["beta"].float().reshape(-1, num_heads))
    g_flat = tensors["g"].float().reshape(-1, num_heads, head_dim)
    gate = -5.0 * torch.sigmoid(
        torch.exp(tensors["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + tensors["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    state = runtime.initial_state_seed.clone()
    out = torch.empty_like(q_flat)
    seq_lens = tuple(runtime.metadata["seq_lens"])
    offsets = _offsets(seq_lens)
    max_seq_len = max(seq_lens)
    device = q.device
    num_sequences = len(seq_lens)
    sequence_rows = []
    token_rows = []
    active_counts = []
    for token_in_sequence in range(max_seq_len):
        active_sequences = [
            sequence
            for sequence, seq_len in enumerate(seq_lens)
            if token_in_sequence < seq_len
        ]
        token_indices = [
            offsets[sequence] + token_in_sequence for sequence in active_sequences
        ]
        active_counts.append(len(active_sequences))
        sequence_rows.append(
            active_sequences + [0] * (num_sequences - len(active_sequences))
        )
        token_rows.append(token_indices + [0] * (num_sequences - len(token_indices)))
    sequence_schedule = torch.tensor(
        sequence_rows,
        dtype=torch.int64,
        device=device,
    )
    token_schedule = torch.tensor(token_rows, dtype=torch.int64, device=device)
    scale = head_dim**-0.5
    for token_in_sequence, active_count in enumerate(active_counts):
        active_sequences = sequence_schedule[token_in_sequence, :active_count]
        token_indices = token_schedule[token_in_sequence, :active_count]
        k_token = k_flat[token_indices]
        decayed = state[active_sequences].float() * decay[token_indices].unsqueeze(-2)
        predicted = torch.einsum("nhk,nhvk->nhv", k_token, decayed)
        residual = beta_flat[token_indices].unsqueeze(-1) * (
            v_flat[token_indices] - predicted
        )
        updated = decayed + residual.unsqueeze(-1) * k_token.unsqueeze(-2)
        quantized_state = updated.to(torch.bfloat16)
        state[active_sequences] = quantized_state
        projected = torch.einsum(
            "nhk,nhvk->nhv",
            q_flat[token_indices],
            quantized_state.float(),
        )
        out[token_indices] = (scale * projected).to(torch.bfloat16)
    return out.reshape_as(q), state


def _assert_bf16_close(torch, *, label: str, actual, expected) -> dict:
    if actual.dtype != torch.bfloat16:
        actual = actual.to(torch.bfloat16)
    if expected.dtype != torch.bfloat16:
        expected = expected.to(torch.bfloat16)
    max_abs = float((actual.float() - expected.float()).abs().max())
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error
    return {
        "passed": True,
        "max_abs": max_abs,
        "atol": BF16_ATOL,
        "rtol": BF16_RTOL,
        "compared_dtype": "bfloat16",
        "compared_numel": actual.numel(),
    }


def _check_correctness(torch, runtime: CaseRuntime) -> dict:
    runtime.candidate_reset()
    actual_output, actual_state = runtime.candidate_run()
    if actual_state is None or (
        actual_state.data_ptr() != runtime.candidate_state_pool[0].data_ptr()
    ):
        raise AssertionError("public recurrent_kda did not return the in-place state")
    torch.cuda.synchronize()
    candidate_output = actual_output.clone()
    candidate_state = actual_state.clone()

    reference_output, reference_state = _independent_bf16_recurrence(torch, runtime)
    torch.cuda.synchronize()
    independent = {
        "output": _assert_bf16_close(
            torch,
            label="public output vs independent recurrence",
            actual=candidate_output,
            expected=reference_output,
        ),
        "final_state": _assert_bf16_close(
            torch,
            label="public final state vs independent recurrence",
            actual=candidate_state,
            expected=reference_state,
        ),
    }

    runtime.flash_kda_raw_run()
    torch.cuda.synchronize()
    flash_kda = {
        "output": _assert_bf16_close(
            torch,
            label="public output vs pinned FlashKDA",
            actual=candidate_output,
            expected=runtime.flash_kda_output,
        ),
        "final_state": _assert_bf16_close(
            torch,
            label="public final state vs pinned FlashKDA",
            actual=candidate_state,
            expected=runtime.flash_kda_final_state,
        ),
    }

    fla_output, fla_state = runtime.fla_run()
    torch.cuda.synchronize()
    fla = {
        "output": _assert_bf16_close(
            torch,
            label="public output vs FLA/Triton",
            actual=candidate_output,
            expected=fla_output,
        ),
        "final_state": _assert_bf16_close(
            torch,
            label="public final state vs FLA/Triton",
            actual=candidate_state,
            expected=fla_state,
        ),
    }
    return {
        "passed": True,
        "public_output_and_full_final_state": True,
        "independent_bf16_recurrence": independent,
        "pinned_flash_kda": flash_kda,
        "fla_triton": fla,
    }


def _cupti_activity_name(activity, kind: str) -> str:
    if kind == "kernel":
        return str(activity.name)
    if kind == "memcpy":
        return (
            f"MEMCPY(copy_kind={int(activity.copy_kind)},bytes={int(activity.bytes)})"
        )
    if kind == "memset":
        return f"MEMSET(value={int(activity.value)},bytes={int(activity.bytes)})"
    raise AssertionError(f"unsupported GPU activity kind {kind!r}")


class _CuptiTracer:
    """Reusable CUPTI activity tracer with exactly one process finalization."""

    def __init__(self, *, stack: SimpleNamespace, cupti) -> None:
        self._stack = stack
        self._cupti = cupti
        self._closed = False
        self._launches: list[LaunchActivity] = []
        self._activities: list[GpuActivity] = []
        self._runtime_kind = int(cupti.ActivityKind.RUNTIME)
        self._driver_kind = int(cupti.ActivityKind.DRIVER)
        self._kernel_kind = int(cupti.ActivityKind.CONCURRENT_KERNEL)
        self._memcpy_kind = int(cupti.ActivityKind.MEMCPY)
        self._memset_kind = int(cupti.ActivityKind.MEMSET)
        self._kinds = (
            cupti.ActivityKind.RUNTIME,
            cupti.ActivityKind.DRIVER,
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        )
        torch = stack.torch
        device = torch.device("cuda")
        self._l2_flush = torch.empty(
            2 * stack.get_l2_cache_size(device),
            dtype=torch.int8,
            device=device,
        )
        cupti.activity_register_callbacks(
            self._buffer_requested,
            self._buffer_completed,
        )

    @staticmethod
    def _buffer_requested():
        return 8 * 1024 * 1024, 0

    def _buffer_completed(self, records) -> None:
        for record in records:
            kind = int(record.kind)
            if kind in {self._runtime_kind, self._driver_kind}:
                launch_kind = "runtime" if kind == self._runtime_kind else "driver"
                self._launches.append(
                    LaunchActivity(
                        start_ns=int(record.start),
                        end_ns=int(record.end),
                        correlation_id=int(record.correlation_id),
                        kind=launch_kind,
                        name=f"{launch_kind}:cbid={int(record.cbid)}",
                    )
                )
                continue
            gpu_kind = None
            if kind == self._kernel_kind:
                gpu_kind = "kernel"
            elif kind == self._memcpy_kind:
                gpu_kind = "memcpy"
            elif kind == self._memset_kind:
                gpu_kind = "memset"
            if gpu_kind is not None:
                self._activities.append(
                    GpuActivity(
                        start_ns=int(record.start),
                        end_ns=int(record.end),
                        correlation_id=int(record.correlation_id),
                        kind=gpu_kind,
                        name=_cupti_activity_name(record, gpu_kind),
                    )
                )

    def trace(
        self,
        *,
        run: Callable[[], object],
        warmup_iters: int,
        repeat_iters: int,
        require_h12_public_route: bool,
    ) -> list[dict]:
        """Trace one block without an event or wall-clock timing fallback."""

        if self._closed:
            raise RuntimeError("CUPTI tracer has already been finalized")
        torch = self._stack.torch
        for _ in range(warmup_iters):
            run()
        torch.cuda.synchronize()

        self._launches = []
        self._activities = []
        brackets = []
        enabled = []
        try:
            for kind in self._kinds:
                self._cupti.activity_enable(kind)
                enabled.append(kind)
            for _ in range(repeat_iters):
                # Cold-L2 work completes before the submission/e2e bracket.
                self._l2_flush.zero_()
                torch.cuda.synchronize()
                start_ns = int(self._cupti.get_timestamp())
                run()
                submitted_ns = int(self._cupti.get_timestamp())
                torch.cuda.synchronize()
                synchronized_ns = int(self._cupti.get_timestamp())
                brackets.append(CpuBracket(start_ns, submitted_ns, synchronized_ns))
            self._cupti.activity_flush_all(1)
        finally:
            for kind in enabled:
                self._cupti.activity_disable(kind)
        return correlate_samples(
            brackets=brackets,
            launches=self._launches,
            activities=self._activities,
            require_h12_public_route=require_h12_public_route,
        )

    def close(self) -> None:
        if not self._closed:
            self._cupti.finalize()
            self._closed = True


def _run_timing_blocks(
    *,
    stack: SimpleNamespace,
    tracer: _CuptiTracer,
    runtime: CaseRuntime,
    warmup_iters: int,
    repeat_iters: int,
    blocks: int,
) -> tuple[dict, list[dict]]:
    paths = {
        "flashinfer_public": (
            runtime.candidate_run,
            runtime.candidate_reset,
            True,
        ),
        "flash_kda_raw": (
            runtime.flash_kda_raw_run,
            lambda: None,
            False,
        ),
        "flash_kda_public_semantics_adapted": (
            runtime.flash_kda_adapted_run,
            runtime.flash_kda_adapted_reset,
            False,
        ),
        "fla_triton": (runtime.fla_run, lambda: None, False),
    }

    base_order = list(paths)
    sample_blocks = {name: [] for name in paths}
    measurement_order = []
    for block_index in range(blocks):
        block_order = base_order if block_index % 2 == 0 else list(reversed(base_order))
        for order_index, name in enumerate(block_order):
            run, reset, require_h12_public_route = paths[name]
            reset()
            stack.torch.cuda.synchronize()
            samples = tracer.trace(
                run=run,
                warmup_iters=warmup_iters,
                repeat_iters=repeat_iters,
                require_h12_public_route=require_h12_public_route,
            )
            for sample in samples:
                sample["block_index"] = block_index
                sample["order_index"] = order_index
            sample_blocks[name].extend(samples)
            measurement_order.append(
                {
                    "block_index": block_index,
                    "order_index": order_index,
                    "path": name,
                }
            )

    timings = {
        name: summarize_samples(
            samples,
            require_h12_public_route=name == "flashinfer_public",
        )
        for name, samples in sample_blocks.items()
    }
    call_paths = {
        "flashinfer_public": "flashinfer.kda.recurrent_kda",
        "flash_kda_raw": "flash_kda._fwd_raw",
        "flash_kda_public_semantics_adapted": (
            "flash_kda._fwd_raw followed by final-state copy-back"
        ),
        "fla_triton": "fla.ops.kda.chunk_kda (backend dispatch disabled)",
    }
    for name, timing in timings.items():
        timing["call_path"] = call_paths[name]
    fla_names = [
        name
        for sample_names in timings["fla_triton"]["kernel_activity_names_samples"]
        for name in sample_names
    ]
    if not fla_names:
        raise RuntimeError("required FLA/Triton path produced no kernel activities")
    if any("flashkda" in name.lower() for name in fla_names):
        raise RuntimeError(
            "FLA comparison routed back to FlashKDA despite FLA_FLASH_KDA=0"
        )
    return timings, measurement_order


def _case_result(
    *,
    stack: SimpleNamespace,
    tracer: _CuptiTracer,
    case,
    flash_kda,
    fla_chunk_kda: Callable,
    warmup_iters: int,
    repeat_iters: int,
    blocks: int,
) -> dict:
    rotations = warmup_iters + repeat_iters
    runtime = _make_case_runtime(
        stack=stack,
        case=case,
        rotations=rotations,
        flash_kda=flash_kda,
        fla_chunk_kda=fla_chunk_kda,
    )
    # Warm JIT modules and descriptor storage before correctness or timing.
    runtime.candidate_reset()
    runtime.candidate_run()
    runtime.flash_kda_raw_run()
    runtime.fla_run()
    stack.torch.cuda.synchronize()

    correctness = _check_correctness(stack.torch, runtime)
    timings, measurement_order = _run_timing_blocks(
        stack=stack,
        tracer=tracer,
        runtime=runtime,
        warmup_iters=warmup_iters,
        repeat_iters=repeat_iters,
        blocks=blocks,
    )
    candidate_ms = timings["flashinfer_public"]["median_gpu_span_ms"]
    speedups = {
        "vs_pinned_flash_kda_raw": (
            timings["flash_kda_raw"]["median_gpu_span_ms"] / candidate_ms
        ),
        "vs_pinned_flash_kda_public_semantics_adapted": (
            timings["flash_kda_public_semantics_adapted"]["median_gpu_span_ms"]
            / candidate_ms
        ),
    }
    speedups["vs_fla_triton"] = (
        timings["fla_triton"]["median_gpu_span_ms"] / candidate_ms
    )
    return {
        **runtime.metadata,
        "correctness": correctness,
        "timings": timings,
        "measurement_order": measurement_order,
        "per_case_speedups": speedups,
        "cross_shape_aggregate": None,
    }


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.validate_only:
        return
    if args.flash_kda_source_dir is None:
        parser.error("--flash-kda-source-dir is required for GPU evidence")
    if args.flash_kda_build_manifest is None:
        parser.error("--flash-kda-build-manifest is required for GPU evidence")
    if args.json is None:
        parser.error("--json is required so raw activity samples are preserved")
    if args.json.resolve().is_relative_to(REPOSITORY_ROOT):
        parser.error("--json must be outside the verified FlashInfer checkout")
    if args.json.resolve().is_relative_to(args.flash_kda_source_dir.resolve()):
        parser.error("--json must be outside the verified FlashKDA checkout")
    if args.warmup_iters <= 0 or args.repeat_iters <= 0:
        parser.error("--warmup-iters and --repeat-iters must be positive")
    if args.blocks < 2:
        parser.error("--blocks must be at least 2 for forward/reverse order evidence")


def _preset_summary(preset: EvidencePreset) -> dict:
    return {
        "name": preset.name,
        "path": preset.path,
        "sha256": preset.sha256,
        "common": preset.common,
        "aggregation": preset.aggregation,
        "cases": [
            {
                "name": case.name,
                "layout": case.layout,
                "seq_lens": list(case.seq_lens),
                "total_tokens": case.total_tokens,
                "seed": case.seed,
            }
            for case in preset.cases
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the checked-in preset without importing torch.",
    )
    parser.add_argument("--flash-kda-source-dir", type=Path)
    parser.add_argument("--flash-kda-build-manifest", type=Path)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    _validate_args(parser, args)

    preset = load_preset(args.preset)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "preset": _preset_summary(preset),
                    "flash_kda_required_revision": FLASH_KDA_BASELINE_REVISION,
                    "flash_kda_build_manifest": {
                        "required": True,
                        "schema_version": FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
                        "helper": "benchmarks/build_flash_kda_phase_a.py",
                        "requires_slurm_gpu_allocation": True,
                        "requires_force_rebuild": True,
                    },
                    "fla_triton": {
                        "required": True,
                        "preimported_fails_closed": True,
                        "clean_identifiable_git_checkout_required": True,
                        "forced_environment": {
                            "FLA_FLASH_KDA": "0",
                            "FLA_DISABLE_BACKEND_DISPATCH": "1",
                        },
                        "all_six_cases_require_output_final_state_and_timing": True,
                    },
                    "changed_beta_cuda_graph_test": {
                        "required": True,
                        "source": GRAPH_TEST_SOURCE,
                        "source_line_range": list(GRAPH_TEST_SOURCE_LINE_RANGE),
                        "node_id": GRAPH_TEST_NODE_ID,
                        "parameterization": {"num_heads": [6, 12]},
                        "command": ["python", "-m", "pytest", "-q", GRAPH_TEST_NODE_ID],
                    },
                    "promotion": {
                        "per_arch_flag": "complete_per_arch_denominator",
                        "required_architectures": ["sm100a", "sm103a"],
                        "reducer": "benchmarks/reduce_kda_h12_phase_a.py",
                        "dual_arch_flag": "promotion_complete_dual_arch",
                    },
                    "flashinfer_required_ancestor_revisions": {
                        "phase_a_upstream_main": (
                            FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION
                        ),
                        "non_aligned_h12_public_route_pr_4351": (
                            FLASHINFER_H12_ROUTE_REVISION
                        ),
                    },
                    "gpu_execution": "not_requested",
                },
                indent=2,
            )
        )
        return

    stack = _import_gpu_stack()
    cupti, cupti_version = _require_cupti()
    hardware = _hardware_metadata(stack)
    graph_receipt = _run_changed_beta_graph_test()
    candidate_provenance = _candidate_provenance(stack)
    assert args.flash_kda_source_dir is not None
    assert args.flash_kda_build_manifest is not None
    flash_kda, flash_kda_provenance = _load_flash_kda(
        args.flash_kda_source_dir,
        args.flash_kda_build_manifest,
    )
    fla_chunk_kda, fla_provenance = _load_required_fla()

    report = {
        "schema_version": EVIDENCE_REPORT_SCHEMA_VERSION,
        "suite": "recurrent_kda_prefill_h12_phase_a",
        "preset": _preset_summary(preset),
        "candidate_provenance": candidate_provenance,
        "baselines": {
            "flash_kda": {
                "available": True,
                "required_revision": FLASH_KDA_BASELINE_REVISION,
                **flash_kda_provenance,
            },
            "fla_triton": fla_provenance,
        },
        "hardware": hardware,
        "changed_beta_cuda_graph_test": graph_receipt,
        "measurement": {
            "timing_backend": "cupti_activity",
            "cupti_python_version": cupti_version,
            "cold_l2": True,
            "warmup_iters_per_block": args.warmup_iters,
            "repeat_iters_per_block": args.repeat_iters,
            "blocks": args.blocks,
            "public_metric": (
                "first-to-last correlated activity span including beta pack "
                "and recurrence"
            ),
            "prepared_metric": (
                "recurrence activity from the same public sample, reported separately"
            ),
            "synchronized_e2e_is_diagnostic_only": True,
            "cross_shape_geomean": False,
            "promotion_unit": "complete six-case denominator on one architecture",
        },
        "cases": [],
    }
    if not graph_receipt["passed"]:
        report["complete_per_arch_denominator"] = False
        assert args.json is not None
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(
            "changed-beta CUDA Graph promotion test failed; "
            f"receipt written to {args.json}"
        )
    tracer = _CuptiTracer(stack=stack, cupti=cupti)
    try:
        for case in preset.cases:
            result = _case_result(
                stack=stack,
                tracer=tracer,
                case=case,
                flash_kda=flash_kda,
                fla_chunk_kda=fla_chunk_kda,
                warmup_iters=args.warmup_iters,
                repeat_iters=args.repeat_iters,
                blocks=args.blocks,
            )
            report["cases"].append(result)
            print(
                f"{case.name}: public="
                f"{result['timings']['flashinfer_public']['median_gpu_span_ms'] * 1000:.3f}us "
                "prepared="
                f"{result['timings']['flashinfer_public']['prepared_recurrence']['median_gpu_span_ms'] * 1000:.3f}us "
                "correctness=pass"
            )
            del result
            stack.torch.cuda.empty_cache()
    finally:
        tracer.close()

    report["complete_per_arch_denominator"] = len(report["cases"]) == 6
    assert args.json is not None
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
