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

"""CUPTI benchmark for recurrent-KDA prefill public API shapes.

The default case set combines the original H64/H96 coverage, six H12 shapes
representing Kimi-K3's per-rank head count under TP8, and four fixed-layout
small-BH shapes. ``--case-set production`` selects the 29-shape inference
portfolio used to qualify the BT16 prepare/chain route, including fixed,
packed, irregular-tail, high-sequence-count, and long-context cases.

The FlashInfer candidate is always invoked through the public
``recurrent_kda`` API. ``--candidate-route dispatcher`` measures the natural
device/shape policy, while ``nonpersistent`` supplies the same explicit
workspace and packed sequence order used by the historical benchmark to keep
B200 on the direct schedule family. ``--backend`` selects one public API
backend per invocation; compare auto, CuTe DSL, and Cake with separate commands
over the same case set. The resolved backend, logical schedule, physical module
variants, and target are recorded during untimed warmup. With
``--flash-kda-peer``, two commit-verified MoonshotAI/FlashKDA measurements are
reported:

* the raw ``_fwd_raw`` kernel timing scope;
* a public-semantics adapter that follows ``_fwd_raw`` with the same-stream
  state copy-back required to emulate ``recurrent_kda`` on FlashKDA.

All paths use the same deterministic tensors and seeds. Preinitialized
rotating state buffers ensure every timed invocation sees the same initial
state. The FlashInfer path updates each state slot in place inside the kernel;
it has no state scratch or copy-back. Allocation, metadata, sequence ordering,
build/JIT, and state-pool reset are outside the measured region.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from flashinfer.kda import recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability

FLASH_KDA_PEER_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLASH_KDA_CUTLASS_COMMIT = "5c149f52a436782210263fb2f19b354443a61c6a"
DEFAULT_LEGACY_STATE_ROTATIONS = 1024
DEFAULT_H12_STATE_ROTATIONS = 4096
DEFAULT_PRODUCTION_STATE_BUDGET_BYTES = 8 * 1024**3
_CUPTI_ESTIMATE_CALLS_PER_BLOCK = 1 + 5
SUPPORTED_FLASH_KDA_ARCHS = {(10, 0): "sm100a", (10, 3): "sm103a"}
BENCHMARKS_DIR = Path(__file__).resolve().parent
H12_PRESET = BENCHMARKS_DIR / "presets" / "recurrent_kda_prefill_h12.json"


@dataclass(frozen=True)
class Case:
    name: str
    num_heads: int
    seq_lens: tuple[int, ...]
    packed: bool
    seed: int


@dataclass
class PreparedCase:
    candidate_run: Callable[[], tuple[torch.Tensor, Optional[torch.Tensor]]]
    peer_raw_run: Optional[Callable[[], None]]
    peer_adapted_run: Optional[Callable[[], None]]
    reset_state_pools: Callable[[], None]
    candidate_output: torch.Tensor
    candidate_state_pool: torch.Tensor
    peer_raw_output: Optional[torch.Tensor]
    peer_raw_final_state: Optional[torch.Tensor]
    peer_adapted_output: Optional[torch.Tensor]
    peer_adapted_state_pool: Optional[torch.Tensor]
    state_cursors: dict[str, list[int]]
    metadata: dict


def _load_h12_cases(path: Path = H12_PRESET) -> tuple[Case, ...]:
    """Load the small checked-in Kimi-K3 TP8 benchmark denominator."""

    payload = json.loads(path.read_text())
    expected_common = {
        "num_heads": 12,
        "head_dim_qk": 128,
        "head_dim_vo": 128,
        "dtype": "bfloat16",
        "initial_state": "provided",
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "beta_is_logit": True,
        "lower_bound": -5.0,
    }
    if payload.get("schema_version") != 1:
        raise ValueError("H12 preset schema_version must be 1")
    if payload.get("name") != "recurrent_kda_prefill_h12":
        raise ValueError("unexpected H12 preset name")
    if payload.get("common") != expected_common:
        raise ValueError("unexpected H12 preset common parameters")
    if payload.get("aggregation") != "per_case_only":
        raise ValueError("H12 benchmark reports per-case results only")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 6:
        raise ValueError("H12 preset must contain six cases")
    if any(
        not isinstance(item, dict) or item.get("layout") not in {"fixed", "packed"}
        for item in raw_cases
    ):
        raise ValueError("H12 preset cases must use fixed or packed layout")

    cases = tuple(
        Case(
            name=item["name"],
            num_heads=expected_common["num_heads"],
            seq_lens=tuple(item["seq_lens"]),
            packed=item["layout"] == "packed",
            seed=item["seed"],
        )
        for item in raw_cases
    )
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("H12 preset must contain six uniquely named cases")
    if any(not case.seq_lens or min(case.seq_lens) <= 0 for case in cases):
        raise ValueError("H12 sequence lengths must be positive")
    return cases


LEGACY_CASES = (
    Case("h96_fixed8192", 96, (8192,), False, 10000),
    Case("h96_mixed", 96, (1300, 547, 2048, 963, 271, 3063), True, 10001),
    Case("h96_uniform", 96, (1024,) * 8, True, 10002),
    Case("h64_fixed8192", 64, (8192,), False, 10003),
    Case("h64_mixed", 64, (1300, 547, 2048, 963, 271, 3063), True, 10004),
    Case("h64_uniform", 64, (1024,) * 8, True, 10005),
)
H12_CASES = _load_h12_cases()
SMALL_BH_CASES = (
    Case("h8_fixed_65536", 8, (65536,), False, 11000),
    Case("h4_fixed_65536_holdout", 4, (65536,), False, 11001),
    Case("h1_fixed_131072", 1, (131072,), False, 11002),
    Case("h1_fixed_1048576", 1, (1048576,), False, 11003),
)
CASES = LEGACY_CASES + H12_CASES + SMALL_BH_CASES

# This public, executable inventory is intentionally expressed with tuple
# multiplication for the large uniform packed cases. It preserves every shape
# without checking in megabytes of repeated JSON integers.
PRODUCTION_CASES = (
    Case("h96_fixed_8192", 96, (8192,), False, 10000),
    Case("h96_mixed_varlen", 96, (1300, 547, 2048, 963, 271, 3063), True, 10001),
    Case("h96_uniform_varlen", 96, (1024,) * 8, True, 10002),
    Case("h64_fixed_8192", 64, (8192,), False, 10003),
    Case("h64_mixed_varlen", 64, (1300, 547, 2048, 963, 271, 3063), True, 10004),
    Case("h64_uniform_varlen", 64, (1024,) * 8, True, 10005),
    Case("h32_fixed_8192", 32, (8192,), False, 11000),
    Case("h32_mixed_varlen", 32, (1300, 547, 2048, 963, 271, 3063), True, 11001),
    Case("h96_uniform_n16", 96, (1024,) * 16, True, 11002),
    Case("h96_uniform_n32_holdout", 96, (1024,) * 32, True, 11003),
    Case("h96_uniform_n64", 96, (1024,) * 64, True, 11004),
    Case("h96_uniform_n128_holdout", 96, (1024,) * 128, True, 11005),
    Case("h96_uniform_n256", 96, (1024,) * 256, True, 11006),
    Case("h96_short_varlen", 96, (64, 128, 256), True, 11007),
    Case("h96_irregular_tail_varlen", 96, (17, 33, 65), True, 11008),
    Case("h16_fixed_16384", 16, (16384,), False, 11009),
    Case("h16_fixed_32768_holdout", 16, (32768,), False, 11010),
    Case("h16_fixed_65536", 16, (65536,), False, 11011),
    Case("h8_fixed_65536", 8, (65536,), False, 11012),
    Case("h4_fixed_65536_holdout", 4, (65536,), False, 11013),
    Case("h4_tail_seq1_to_15", 4, tuple(range(1, 16)), True, 11014),
    Case("h1_fixed_1048576", 1, (1048576,), False, 11015),
    Case("h96_fixed_37", 96, (37,), False, 11016),
    Case("h96_fixed_97", 96, (97,), False, 11017),
    Case("h96_packed_n1_16", 96, (16,), True, 11018),
    Case("h96_packed_uniform_n2_t16", 96, (16, 16), True, 11019),
    Case("h1_fixed_131072", 1, (131072,), False, 11020),
    Case("h1_packed_n1_131072", 1, (131072,), True, 11021),
    Case("h1_packed_524288_524288", 1, (524288, 524288), True, 11022),
)

_BT16_PREPARE_VARIANTS = frozenset(("bt16_prepare", "bt16_prepare_beta_tma"))
_BT16_CHAIN_VARIANTS = frozenset(
    (
        "bt16_chain_m64_s7",
        "bt16_chain_m64_s8",
        "bt16_chain_m64_s9",
    )
)
_BT16_COMBINED_VARIANTS = frozenset(("bt16_prepare_chain_m64_s8",))


def _resolve_recorded_cake_route(
    routes: list[tuple[str, str]],
) -> tuple[str, str, list[str]]:
    """Normalize one-stage and BT16 two-stage Cake warmup observations."""

    if len(routes) == 1:
        variant, target = routes[0]
        if variant in _BT16_COMBINED_VARIANTS:
            return "bt16_prepare_chain_m64", target, [variant]
        if (
            variant not in _BT16_PREPARE_VARIANTS
            and variant not in _BT16_CHAIN_VARIANTS
        ):
            return variant, target, [variant]
    if len(routes) == 2:
        (prepare_variant, prepare_target), (chain_variant, chain_target) = routes
        if (
            prepare_variant in _BT16_PREPARE_VARIANTS
            and chain_variant in _BT16_CHAIN_VARIANTS
            and prepare_target == chain_target
        ):
            return (
                "bt16_prepare_chain_m64",
                prepare_target,
                [prepare_variant, chain_variant],
            )
    raise RuntimeError(
        "expected one Cake module or one ordered BT16 prepare/chain pair "
        f"during warmup, got {routes}"
    )


def _default_state_rotations(case: Case) -> int:
    base = (
        DEFAULT_H12_STATE_ROTATIONS
        if case in H12_CASES
        else DEFAULT_LEGACY_STATE_ROTATIONS
    )
    if case not in PRODUCTION_CASES:
        return base
    state_bytes = len(case.seq_lens) * case.num_heads * 128 * 128 * 2
    budget_capacity = max(8, DEFAULT_PRODUCTION_STATE_BUDGET_BYTES // state_bytes)
    return min(base, budget_capacity)


def _timing_iteration_budget(
    *,
    state_rotation_capacity: int,
    requested_dry_run_iters: int,
    requested_repeat_iters: int,
) -> tuple[int, int]:
    """Fit explicit CUPTI dry/repeat iterations into one rotating-state block."""

    available = state_rotation_capacity - _CUPTI_ESTIMATE_CALLS_PER_BLOCK
    if available < 2:
        raise ValueError(
            "state rotations must cover six CUPTI estimate calls plus at "
            "least one dry run and one measured iteration"
        )
    desired_dry = max(1, requested_dry_run_iters)
    desired_repeat = max(1, requested_repeat_iters)
    desired_total = desired_dry + desired_repeat
    if desired_total <= available:
        return desired_dry, desired_repeat
    dry_run_iters = max(1, round(available * desired_dry / desired_total))
    repeat_iters = available - dry_run_iters
    if repeat_iters < 1:
        repeat_iters = 1
        dry_run_iters = available - 1
    return dry_run_iters, repeat_iters


def _require_cupti() -> None:
    try:
        from cupti import cupti  # noqa: F401
    except ImportError as error:
        raise RuntimeError("cupti-python >= 13 is required") from error
    cupti_version = version("cupti-python")
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {cupti_version}")


def _hardware_metadata(device: torch.device) -> dict:
    compute_capability = get_compute_capability(device)
    properties = torch.cuda.get_device_properties(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return {
        "device_name": properties.name,
        "device_index": device_index,
        "compute_capability": list(compute_capability),
        "cuda_arch": SUPPORTED_FLASH_KDA_ARCHS[compute_capability],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to verify FlashKDA source provenance at {root}: "
            f"{error.output.strip()}"
        ) from error


def _verify_peer_provenance(flash_kda, source_dir: Path) -> dict:
    source_dir = source_dir.resolve(strict=True)
    package_path = Path(flash_kda.__file__).resolve(strict=True)
    if not package_path.is_relative_to(source_dir):
        raise RuntimeError(
            "flash_kda must be imported from the verified source checkout: "
            f"module={package_path}, checkout={source_dir}"
        )

    source_commit = _git_output(source_dir, "rev-parse", "HEAD")
    if source_commit != FLASH_KDA_PEER_COMMIT:
        raise RuntimeError(
            "unexpected FlashKDA source revision: "
            f"expected {FLASH_KDA_PEER_COMMIT}, got {source_commit}"
        )
    cutlass_dir = source_dir / "cutlass"
    cutlass_commit = _git_output(cutlass_dir, "rev-parse", "HEAD")
    if cutlass_commit != FLASH_KDA_CUTLASS_COMMIT:
        raise RuntimeError(
            "unexpected FlashKDA CUTLASS revision: "
            f"expected {FLASH_KDA_CUTLASS_COMMIT}, got {cutlass_commit}"
        )
    submodule_record = _git_output(source_dir, "ls-tree", "HEAD", "cutlass").split()
    if len(submodule_record) < 3 or submodule_record[2] != cutlass_commit:
        raise RuntimeError(
            "FlashKDA CUTLASS checkout does not match the pinned gitlink"
        )
    tracked_changes = _git_output(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        raise RuntimeError(
            f"verified FlashKDA checkout has tracked modifications:\n{tracked_changes}"
        )

    extension = import_module("flash_kda_C")
    extension_path = Path(extension.__file__).resolve(strict=True)
    if not extension_path.is_relative_to(source_dir):
        raise RuntimeError(
            "flash_kda_C must be loaded from the verified source checkout: "
            f"extension={extension_path}, checkout={source_dir}"
        )
    return {
        "repository": "https://github.com/MoonshotAI/FlashKDA.git",
        "source_dir": str(source_dir),
        "source_commit": source_commit,
        "cutlass_commit": cutlass_commit,
        "package_path": str(package_path),
        "extension_path": str(extension_path),
    }


def _make_state_pool(
    initial_state: torch.Tensor,
    rotations: int,
) -> torch.Tensor:
    return initial_state.unsqueeze(0).expand(rotations, *initial_state.shape).clone()


def _make_case(
    case: Case,
    *,
    state_rotations: int,
    candidate_route: str,
    candidate_backend: str,
    flash_kda=None,
) -> PreparedCase:
    total_tokens = sum(case.seq_lens)
    shape = (1, total_tokens, case.num_heads, 128)
    generator = torch.Generator(device="cuda").manual_seed(case.seed)
    q = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    g = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    beta = torch.randn(
        (1, total_tokens, case.num_heads),
        generator=generator,
        device="cuda",
    ).to(torch.bfloat16)
    A_log = torch.rand(
        (case.num_heads,),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.rand(
        (case.num_heads, 128),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    initial_state = (
        torch.randn(
            (len(case.seq_lens), case.num_heads, 128, 128),
            generator=generator,
            device="cuda",
        )
        * 0.25
    ).to(torch.bfloat16)
    candidate_state_pool = _make_state_pool(initial_state, state_rotations)
    candidate_output = torch.empty_like(q)
    candidate_workspace = (
        RecurrentKDAPrefillWorkspace(q.device)
        if candidate_route == "nonpersistent"
        else None
    )
    state_cursors = {"pr": [0], "adapted": [0]}

    offsets = [0]
    for seq_len in case.seq_lens:
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device="cuda") if case.packed else None
    )
    seq_order = (
        torch.tensor(
            sorted(
                range(len(case.seq_lens)),
                key=case.seq_lens.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        if case.packed and candidate_route == "nonpersistent"
        else None
    )
    scale = float(1.0 / np.sqrt(128.0))

    def candidate_run():
        state_index = state_cursors["pr"][0]
        if state_index >= state_rotations:
            raise RuntimeError(
                f"PR state rotations exhausted: {state_index} >= {state_rotations}"
            )
        state_cursors["pr"][0] += 1
        return recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=candidate_state_pool[state_index],
            output=candidate_output,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            cu_seqlens=cu_seqlens,
            beta_is_logit=True,
            seq_order=seq_order,
            prefill_workspace=candidate_workspace,
            backend=candidate_backend,
        )

    peer_raw_run = None
    peer_adapted_run = None
    peer_raw_output = None
    peer_raw_final_state = None
    peer_adapted_output = None
    peer_adapted_state_pool = None
    peer_raw_initial_state = None
    if flash_kda is not None:
        peer_raw_initial_state = initial_state.clone()
        peer_raw_final_state = torch.empty_like(initial_state)
        peer_raw_output = torch.empty_like(q)
        peer_adapted_state_pool = _make_state_pool(
            initial_state,
            state_rotations,
        )
        peer_adapted_final_state = torch.empty_like(initial_state)
        peer_adapted_output = torch.empty_like(q)
        workspace_size = flash_kda.get_workspace_size(
            total_tokens,
            case.num_heads,
            len(case.seq_lens),
        )
        peer_raw_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device="cuda",
        )
        peer_adapted_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device="cuda",
        )

        def peer_raw_run() -> None:
            flash_kda._fwd_raw(
                q,
                k,
                v,
                g,
                beta,
                scale,
                peer_raw_output,
                peer_raw_workspace,
                A_log,
                dt_bias,
                -5.0,
                initial_state=peer_raw_initial_state,
                final_state=peer_raw_final_state,
                cu_seqlens=cu_seqlens,
            )

        def peer_adapted_run() -> None:
            state_index = state_cursors["adapted"][0]
            if state_index >= state_rotations:
                raise RuntimeError(
                    "adapted-peer state rotations exhausted: "
                    f"{state_index} >= {state_rotations}"
                )
            state_cursors["adapted"][0] += 1
            adapted_state = peer_adapted_state_pool[state_index]
            flash_kda._fwd_raw(
                q,
                k,
                v,
                g,
                beta,
                scale,
                peer_adapted_output,
                peer_adapted_workspace,
                A_log,
                dt_bias,
                -5.0,
                initial_state=adapted_state,
                final_state=peer_adapted_final_state,
                cu_seqlens=cu_seqlens,
            )
            adapted_state.copy_(peer_adapted_final_state)

    def reset_state_pools() -> None:
        candidate_state_pool.copy_(initial_state.unsqueeze(0))
        state_cursors["pr"][0] = 0
        if peer_raw_initial_state is not None:
            peer_raw_initial_state.copy_(initial_state)
        if peer_adapted_state_pool is not None:
            peer_adapted_state_pool.copy_(initial_state.unsqueeze(0))
            state_cursors["adapted"][0] = 0

    # Observe the actual internal module selected by the public API once during
    # untimed warmup. This avoids duplicating dispatcher policy in the evidence
    # harness while keeping route logging out of every timed call.
    kda_prefill_module = import_module("flashinfer.kda_prefill")
    kda_prefill_cute_module = import_module("flashinfer.kda_prefill_cute")
    original_get_module = kda_prefill_module._get_flash_kda_prefill_module
    original_cute_run = kda_prefill_cute_module._run_cute_dsl_kda_prefill
    resolved_cake_routes = []
    resolved_backends = []

    def recording_get_module(variant, target):
        resolved_cake_routes.append((variant, target))
        return original_get_module(variant, target)

    def recording_cute_run(**kwargs):
        resolved_backends.append("cute-dsl")
        return original_cute_run(**kwargs)

    kda_prefill_module._get_flash_kda_prefill_module = recording_get_module
    kda_prefill_cute_module._run_cute_dsl_kda_prefill = recording_cute_run
    try:
        candidate_run()
        torch.cuda.synchronize()
    finally:
        kda_prefill_module._get_flash_kda_prefill_module = original_get_module
        kda_prefill_cute_module._run_cute_dsl_kda_prefill = original_cute_run
        reset_state_pools()
    if resolved_backends:
        if resolved_backends != ["cute-dsl"] or resolved_cake_routes:
            raise RuntimeError(
                "expected exactly one CuTe DSL route during warmup, got "
                f"backends={resolved_backends}, cake={resolved_cake_routes}"
            )
        resolved_backend = "cute-dsl"
        decomp_ctas = len(case.seq_lens) * case.num_heads * 2
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        resolved_variant = "decomp" if decomp_ctas <= sm_count else "engine"
        resolved_target = "bt16"
        resolved_physical_variants = [resolved_variant]
    elif resolved_cake_routes:
        resolved_backend = "cake"
        (
            resolved_variant,
            resolved_target,
            resolved_physical_variants,
        ) = _resolve_recorded_cake_route(resolved_cake_routes)
    else:
        raise RuntimeError(
            "expected one recurrent-KDA prefill route during warmup, got "
            f"backends={resolved_backends}, cake={resolved_cake_routes}"
        )

    metadata = {
        "name": case.name,
        "num_heads": case.num_heads,
        "seq_lens": list(case.seq_lens),
        "total_tokens": total_tokens,
        "layout": "packed" if case.packed else "fixed",
        "variant": resolved_variant,
        "physical_variants": resolved_physical_variants,
        "target": resolved_target,
        "candidate_route": candidate_route,
        "requested_backend": candidate_backend,
        "resolved_backend": resolved_backend,
        "seed": case.seed,
        "state_rotation_capacity": state_rotations,
    }
    return PreparedCase(
        candidate_run=candidate_run,
        peer_raw_run=peer_raw_run,
        peer_adapted_run=peer_adapted_run,
        reset_state_pools=reset_state_pools,
        candidate_output=candidate_output,
        candidate_state_pool=candidate_state_pool,
        peer_raw_output=peer_raw_output,
        peer_raw_final_state=peer_raw_final_state,
        peer_adapted_output=peer_adapted_output,
        peer_adapted_state_pool=peer_adapted_state_pool,
        state_cursors=state_cursors,
        metadata=metadata,
    )


def _check_peer(prepared: PreparedCase) -> dict[str, float]:
    assert prepared.peer_raw_run is not None
    assert prepared.peer_adapted_run is not None
    assert prepared.peer_raw_output is not None
    assert prepared.peer_raw_final_state is not None
    assert prepared.peer_adapted_output is not None
    assert prepared.peer_adapted_state_pool is not None
    prepared.reset_state_pools()
    prepared.candidate_run()
    prepared.peer_raw_run()
    prepared.peer_adapted_run()
    torch.cuda.synchronize()

    candidate_state = prepared.candidate_state_pool[0]
    adapted_state = prepared.peer_adapted_state_pool[0]
    comparisons = (
        (
            "raw_output_max_abs",
            prepared.candidate_output,
            prepared.peer_raw_output,
        ),
        (
            "raw_state_max_abs",
            candidate_state,
            prepared.peer_raw_final_state,
        ),
        (
            "adapted_output_max_abs",
            prepared.candidate_output,
            prepared.peer_adapted_output,
        ),
        ("adapted_state_max_abs", candidate_state, adapted_state),
    )
    diagnostics = {}
    for name, actual, expected in comparisons:
        diagnostics[name] = float((actual.float() - expected.float()).abs().max())
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-2,
            rtol=1e-2,
        )
    return diagnostics


def _measure(
    run: Callable[[], object],
    *,
    dry_run_iters: int,
    repeat_iters: int,
) -> tuple[float, list[float]]:
    measurements = bench_gpu_time(
        run,
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    samples_ms = [float(value) for value in measurements]
    return float(np.median(samples_ms)), samples_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument(
        "--case-set",
        choices=("all", "legacy", "h12", "small_bh", "production"),
        default="all",
        help=(
            "Run all cases, the original H64/H96 cases, the Kimi-K3 TP8 H12 "
            "cases, the fixed-layout small-BH cases, or the complete "
            "29-shape production portfolio."
        ),
    )
    parser.add_argument(
        "--state-rotations",
        type=int,
        help=(
            "Override the number of preinitialized same-input state slots per "
            "mutable path. By default legacy and small-BH cases use "
            f"{DEFAULT_LEGACY_STATE_ROTATIONS} slots and H12 cases use "
            f"{DEFAULT_H12_STATE_ROTATIONS} slots."
        ),
    )
    parser.add_argument(
        "--candidate-route",
        choices=("dispatcher", "nonpersistent"),
        default="dispatcher",
        help=(
            "Measure the natural public dispatcher or force B200 onto its "
            "non-persistent direct/M64 family with an explicit workspace."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cute-dsl", "cake"),
        default="auto",
        help=(
            "Select one backend for this invocation of the public recurrent_kda "
            "API; run separate commands to compare backends."
        ),
    )
    parser.add_argument(
        "--flash-kda-peer",
        action="store_true",
        help=(
            "Compare against FlashKDA commit "
            f"{FLASH_KDA_PEER_COMMIT} using both raw and "
            "public-semantics-adapted scopes."
        ),
    )
    parser.add_argument(
        "--flash-kda-source-dir",
        type=Path,
        help=(
            "Required with --flash-kda-peer. The imported editable FlashKDA "
            "package and extension must resolve inside this exact checkout."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optionally write the result list as JSON.",
    )
    args = parser.parse_args()

    if args.dry_run_iters <= 0 or args.repeat_iters <= 0:
        parser.error("--dry-run-iters and --repeat-iters must be positive")
    if args.state_rotations is not None and args.state_rotations <= 0:
        parser.error("--state-rotations must be positive")
    if args.flash_kda_peer != (args.flash_kda_source_dir is not None):
        parser.error(
            "--flash-kda-peer and --flash-kda-source-dir must be provided together"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if compute_capability not in SUPPORTED_FLASH_KDA_ARCHS:
        raise RuntimeError(
            "frozen recurrent-KDA prefill requires exact CC 10.0 "
            "(SM100a; B200/GB200) or CC 10.3 (SM103a; B300/GB300), "
            f"got CC {compute_capability[0]}."
            f"{compute_capability[1]}"
        )
    _require_cupti()
    hardware = _hardware_metadata(device)
    print(
        "Hardware: "
        f"{hardware['device_name']} cc "
        f"{hardware['compute_capability'][0]}."
        f"{hardware['compute_capability'][1]} ({hardware['cuda_arch']})"
    )

    flash_kda = None
    peer_provenance = None
    if args.flash_kda_peer:
        try:
            import flash_kda as imported_flash_kda
        except ImportError as error:
            raise RuntimeError(
                "install MoonshotAI/FlashKDA at "
                f"{FLASH_KDA_PEER_COMMIT} to run the peer comparison"
            ) from error
        flash_kda = imported_flash_kda
        assert args.flash_kda_source_dir is not None
        peer_provenance = _verify_peer_provenance(
            flash_kda,
            args.flash_kda_source_dir,
        )

    selected_cases = {
        "all": CASES,
        "legacy": LEGACY_CASES,
        "h12": H12_CASES,
        "small_bh": SMALL_BH_CASES,
        "production": PRODUCTION_CASES,
    }[args.case_set]
    results = []
    for case in selected_cases:
        state_rotations = args.state_rotations
        if state_rotations is None:
            state_rotations = _default_state_rotations(case)
        dry_run_iters, repeat_iters = _timing_iteration_budget(
            state_rotation_capacity=state_rotations,
            requested_dry_run_iters=args.dry_run_iters,
            requested_repeat_iters=args.repeat_iters,
        )
        timing_iteration_budget = {
            "cupti_estimate_calls": _CUPTI_ESTIMATE_CALLS_PER_BLOCK,
            "dry_run_iters": dry_run_iters,
            "repeat_iters": repeat_iters,
            "total_stateful_calls_per_block": (
                _CUPTI_ESTIMATE_CALLS_PER_BLOCK + dry_run_iters + repeat_iters
            ),
            "state_rotation_capacity": state_rotations,
            "low_sample_count": repeat_iters < 10,
        }
        prepared = _make_case(
            case,
            state_rotations=state_rotations,
            candidate_route=args.candidate_route,
            candidate_backend=args.backend,
            flash_kda=flash_kda,
        )
        result = {**prepared.metadata, "hardware": hardware}
        if prepared.peer_raw_run is None:
            prepared.reset_state_pools()
            prepared.candidate_run()
            torch.cuda.synchronize()
            prepared.reset_state_pools()
            candidate_ms, candidate_samples = _measure(
                prepared.candidate_run,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            candidate_block_medians = [candidate_ms]
            result["correctness_peer"] = "not_requested"
            stateful_calls = prepared.state_cursors["pr"][0]
            if (
                stateful_calls
                != timing_iteration_budget["total_stateful_calls_per_block"]
            ):
                raise RuntimeError(
                    "CUPTI candidate call count no longer matches the explicit "
                    f"iteration budget: {stateful_calls}"
                )
            result["state_slots_used_per_block"] = {"pr": [stateful_calls]}
        else:
            assert prepared.peer_adapted_run is not None
            correctness = _check_peer(prepared)
            result.update(
                {
                    "correctness_peer": "passed",
                    **correctness,
                }
            )
            samples = {"pr": [], "raw": [], "adapted": []}
            block_medians = {"pr": [], "raw": [], "adapted": []}
            state_slots_used = {"pr": [], "adapted": []}
            # Symmetric ABCCBA order bounds temperature/clock drift while
            # retaining two independent medians for every timing scope.
            for backend, run in (
                ("pr", prepared.candidate_run),
                ("raw", prepared.peer_raw_run),
                ("adapted", prepared.peer_adapted_run),
                ("adapted", prepared.peer_adapted_run),
                ("raw", prepared.peer_raw_run),
                ("pr", prepared.candidate_run),
            ):
                prepared.reset_state_pools()
                torch.cuda.synchronize()
                block_median, block_samples = _measure(
                    run,
                    dry_run_iters=dry_run_iters,
                    repeat_iters=repeat_iters,
                )
                block_medians[backend].append(block_median)
                samples[backend].extend(block_samples)
                if backend in state_slots_used:
                    stateful_calls = prepared.state_cursors[backend][0]
                    if (
                        stateful_calls
                        != timing_iteration_budget["total_stateful_calls_per_block"]
                    ):
                        raise RuntimeError(
                            "CUPTI stateful call count no longer matches the "
                            f"explicit iteration budget for {backend}: "
                            f"{stateful_calls}"
                        )
                    state_slots_used[backend].append(stateful_calls)
            del run

            candidate_block_medians = block_medians["pr"]
            candidate_samples = samples["pr"]
            candidate_ms = float(np.median(candidate_block_medians))
            raw_ms = float(np.median(block_medians["raw"]))
            adapted_ms = float(np.median(block_medians["adapted"]))
            result.update(
                {
                    "flash_kda_peer_raw_ms": raw_ms,
                    "flash_kda_peer_raw_samples_ms": samples["raw"],
                    "flash_kda_peer_raw_block_medians_ms": (block_medians["raw"]),
                    "speedup_vs_flash_kda_peer_raw": raw_ms / candidate_ms,
                    "flash_kda_peer_adapted_ms": adapted_ms,
                    "flash_kda_peer_adapted_samples_ms": samples["adapted"],
                    "flash_kda_peer_adapted_block_medians_ms": (
                        block_medians["adapted"]
                    ),
                    "speedup_vs_flash_kda_peer_adapted": (adapted_ms / candidate_ms),
                    "peer_raw_timing_scope": ("flash_kda_raw_fwd"),
                    "peer_adapted_timing_scope": (
                        "raw_fwd_plus_public_state_copy_back"
                    ),
                    "pair_order": "PR/raw/adapted/adapted/raw/PR",
                    "same_initial_state_per_timed_call": True,
                    "state_slots_used_per_block": state_slots_used,
                    "flash_kda_peer_provenance": peer_provenance,
                }
            )

        result.update(
            {
                "median_ms": candidate_ms,
                "median_us": candidate_ms * 1000.0,
                "samples_ms": candidate_samples,
                "block_medians_ms": candidate_block_medians,
                "timing_backend": "cupti",
                "cold_l2": True,
                "cuda_graph": False,
                "timing_scope": ("public_recurrent_kda_with_inplace_state_update"),
                "requested_dry_run_iters": args.dry_run_iters,
                "requested_repeat_iters": args.repeat_iters,
                "timing_iteration_budget": timing_iteration_budget,
            }
        )
        results.append(result)
        if prepared.peer_raw_run is None:
            print(
                f"{result['name']:<18} {result['resolved_backend']:<8} "
                f"{result['variant']:<10} "
                f"{result['median_us']:10.3f} us"
            )
        else:
            print(
                f"{result['name']:<18} {result['variant']:<4} "
                f"PR {result['median_us']:10.3f} us  "
                f"raw {result['flash_kda_peer_raw_ms'] * 1000.0:10.3f} us "
                f"{result['speedup_vs_flash_kda_peer_raw']:.4f}x  "
                f"adapted "
                f"{result['flash_kda_peer_adapted_ms'] * 1000.0:10.3f} us "
                f"{result['speedup_vs_flash_kda_peer_adapted']:.4f}x"
            )
        del prepared
        torch.cuda.empty_cache()

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
