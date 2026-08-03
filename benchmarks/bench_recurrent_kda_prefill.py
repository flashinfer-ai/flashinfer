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

"""CUPTI benchmark for the six frozen recurrent-KDA prefill contract shapes.

The FlashInfer candidate is always invoked through the public
``recurrent_kda`` API. With ``--flash-kda-peer``, two commit-verified
MoonshotAI/FlashKDA measurements are reported:

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
import hashlib
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

FLASH_KDA_PEER_COMMIT = "d2ff19a6a0c82f39f796f637ebd1c36090b1268f"
FLASH_KDA_CUTLASS_COMMIT = "5c149f52a436782210263fb2f19b354443a61c6a"
DEFAULT_STATE_ROTATIONS = 512


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


CASES = (
    Case("h96_fixed8192", 96, (8192,), False, 10000),
    Case("h96_mixed", 96, (1300, 547, 2048, 963, 271, 3063), True, 10001),
    Case("h96_uniform", 96, (1024,) * 8, True, 10002),
    Case("h64_fixed8192", 64, (8192,), False, 10003),
    Case("h64_mixed", 64, (1300, 547, 2048, 963, 271, 3063), True, 10004),
    Case("h64_uniform", 64, (1024,) * 8, True, 10005),
)


def _require_cupti() -> None:
    try:
        from cupti import cupti  # noqa: F401
    except ImportError as error:
        raise RuntimeError("cupti-python >= 13 is required") from error
    cupti_version = version("cupti-python")
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {cupti_version}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "package_sha256": _sha256(package_path),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
    }


def _make_state_pool(
    initial_state: torch.Tensor,
    rotations: int,
) -> torch.Tensor:
    return initial_state.unsqueeze(0).expand(rotations, *initial_state.shape).clone()


def _make_case(case: Case, *, state_rotations: int, flash_kda=None) -> PreparedCase:
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
    candidate_workspace = RecurrentKDAPrefillWorkspace(q.device)
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
        if case.packed
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

    metadata = {
        "name": case.name,
        "num_heads": case.num_heads,
        "seq_lens": list(case.seq_lens),
        "total_tokens": total_tokens,
        "layout": "packed" if case.packed else "fixed",
        "variant": "m64" if case.name == "h64_fixed8192" else "m128",
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
    warmup_ms: int,
    bench_ms: int,
) -> tuple[float, list[float]]:
    measurements = bench_gpu_time(
        run,
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=bench_ms,
    )
    samples_ms = [float(value) for value in measurements]
    return float(np.median(samples_ms)), samples_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument(
        "--state-rotations",
        type=int,
        default=DEFAULT_STATE_ROTATIONS,
        help="Number of preinitialized same-input state slots per mutable path.",
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

    if args.warmup_ms <= 0 or args.bench_ms <= 0:
        parser.error("--warmup-ms and --bench-ms must be positive")
    if args.state_rotations <= 0:
        parser.error("--state-rotations must be positive")
    if args.flash_kda_peer != (args.flash_kda_source_dir is not None):
        parser.error(
            "--flash-kda-peer and --flash-kda-source-dir must be provided together"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        raise RuntimeError("frozen recurrent-KDA prefill requires B200 (cc 10.0)")
    _require_cupti()

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

    results = []
    for case in CASES:
        prepared = _make_case(
            case,
            state_rotations=args.state_rotations,
            flash_kda=flash_kda,
        )
        result = dict(prepared.metadata)
        if prepared.peer_raw_run is None:
            prepared.reset_state_pools()
            prepared.candidate_run()
            torch.cuda.synchronize()
            prepared.reset_state_pools()
            candidate_ms, candidate_samples = _measure(
                prepared.candidate_run,
                warmup_ms=args.warmup_ms,
                bench_ms=args.bench_ms,
            )
            candidate_block_medians = [candidate_ms]
            result["correctness_peer"] = "not_requested"
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
                    warmup_ms=args.warmup_ms,
                    bench_ms=args.bench_ms,
                )
                block_medians[backend].append(block_median)
                samples[backend].extend(block_samples)
                if backend in state_slots_used:
                    state_slots_used[backend].append(prepared.state_cursors[backend][0])
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
                "warmup_ms": args.warmup_ms,
                "bench_ms": args.bench_ms,
            }
        )
        results.append(result)
        if prepared.peer_raw_run is None:
            print(
                f"{result['name']:<18} {result['variant']:<4} "
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
