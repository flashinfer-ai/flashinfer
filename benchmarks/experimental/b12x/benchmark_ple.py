#!/usr/bin/env python3
"""Benchmark the public Qwen3.8 Flash Next PLE runtime APIs.

The hash corpus uses the production 16-head, 20M-base table geometry because
hashing does not materialize the table.  Embedding lookup keeps the production
head count, head width, output width, and TP partition, but defaults to an
explicitly storage-scaled table so all three formats are runnable on one GPU.
Use ``--embedding-geometry production --allow-production-table`` to allocate
the real table.  CUDA-mapped host storage is a separate opt-in
(``--table-memory mapped_host``), never part of the default result.

PLE layer cases use S=4, H=2560, K=4, dilation=3, and four speculative tail
slots.  Recurrent state is restored before every timed invocation.  Restore
latency and kernel latency are retained separately for eager and CUDA-graph
measurements.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import json
import math
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from b12x.sequence import ple, ple_embedding, ple_hash
from b12x.sequence.ple.reference import (
    ple_projected_sequence_reference,
    ple_projected_u_reference,
)
from b12x.sequence.ple_embedding import reference as ple_embedding_reference
from b12x.sequence.ple_hash.reference import ple_hash_packed_reference
from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
)


VOCAB_SIZE = 248_320
EOS_TOKEN_ID = 248_044
MAX_ORDER = 3
HEADS_PER_ORDER = 8
HEAD_COUNT = 16
EMBEDDING_DIM = 2_560
TP_SIZE = 4
TP_RANK = 0
TABLE_ALIGNMENT = 128
PRODUCTION_BASE_TABLE_SIZE = 20_000_000
STORAGE_SCALED_BASE_TABLE_SIZE = 65_521

STREAMS = 4
HIDDEN_SIZE = 2_560
KERNEL_SIZE = 4
DILATION = 3
MAX_SPECULATIVE_TOKENS = 4
DTYPE = torch.bfloat16
EPS = 1.0e-6

APIS = ("hash", "embedding", "decode", "prefill", "mixed")
TIMING_MODES = ("eager", "graph")
QUANT_MODES = ("bf16", "fp8_e4m3_per_tensor", "nvfp4_group16")
TABLE_MEMORIES = ("device", "mapped_host")
SCHEMA = "b12x-qwen38-flash-next-ple-benchmark-v1"


@dataclass(frozen=True)
class PackedProfile:
    """Packed token/request geometry shared by hash and embedding lookup."""

    name: str
    phase: str
    query_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.query_lengths or any(length <= 0 for length in self.query_lengths):
            raise ValueError("query_lengths must contain positive request lengths")

    @property
    def tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def sequences(self) -> int:
        return len(self.query_lengths)


PACKED_PROFILES = (
    PackedProfile("decode-t1-bs1", "decode", (1,)),
    PackedProfile("spec-t4-bs1", "speculative", (4,)),
    PackedProfile("prefill-t128-bs1", "prefill", (128,)),
    PackedProfile("prefill-t512-bs4", "prefill", (128, 128, 128, 128)),
)


@dataclass(frozen=True)
class LayerProfile:
    """One stateful PLE layer transaction."""

    name: str
    mode: str
    query_lengths: tuple[int, ...]
    state_is_fresh: tuple[bool, ...]
    num_accepted_tokens: tuple[int, ...]
    request_is_prefill: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        sequences = len(self.query_lengths)
        if self.mode not in ("decode", "prefill", "mixed"):
            raise ValueError(f"invalid PLE layer mode {self.mode!r}")
        if sequences == 0 or any(length <= 0 for length in self.query_lengths):
            raise ValueError("query_lengths must contain positive request lengths")
        if len(self.state_is_fresh) != sequences:
            raise ValueError("state_is_fresh must match query_lengths")
        if len(self.num_accepted_tokens) != sequences:
            raise ValueError("num_accepted_tokens must match query_lengths")
        if self.mode == "mixed":
            if (
                self.request_is_prefill is None
                or len(self.request_is_prefill) != sequences
            ):
                raise ValueError("mixed profiles require one request mode per sequence")
        elif self.request_is_prefill is not None:
            raise ValueError("request modes are only valid for mixed profiles")

    @property
    def tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def sequences(self) -> int:
        return len(self.query_lengths)


LAYER_PROFILES = (
    LayerProfile("decode-t1-bs1", "decode", (1,), (False,), (1,)),
    LayerProfile("decode-spec4-bs1", "decode", (4,), (False,), (2,)),
    LayerProfile("prefill-t128-bs1", "prefill", (128,), (True,), (0,)),
    LayerProfile(
        "prefill-t512-bs4",
        "prefill",
        (128, 128, 128, 128),
        (True, False, True, False),
        (0, 0, 0, 0),
    ),
    LayerProfile(
        "mixed-t133-bs3",
        "mixed",
        (128, 4, 1),
        (True, False, False),
        (0, 2, 1),
        (True, False, False),
    ),
)


@dataclass(frozen=True)
class EmbeddingGeometry:
    name: str
    base_table_size: int
    storage_scope: str
    production_storage: bool


EMBEDDING_GEOMETRIES = {
    "storage-scaled": EmbeddingGeometry(
        name="storage-scaled",
        base_table_size=STORAGE_SCALED_BASE_TABLE_SIZE,
        storage_scope=(
            "Qwen compute/head geometry with a smaller persistent table; "
            "lookup locality and capacity are not production evidence"
        ),
        production_storage=False,
    ),
    "production": EmbeddingGeometry(
        name="production",
        base_table_size=PRODUCTION_BASE_TABLE_SIZE,
        storage_scope="exact Qwen persistent table geometry",
        production_storage=True,
    ),
}


@dataclass
class HashCase:
    profile: PackedProfile
    binding: ple_hash.Binding
    expected: torch.Tensor

    def launch(self) -> torch.Tensor:
        return ple_hash.run(self.binding)


@dataclass
class EmbeddingCase:
    profile: PackedProfile
    binding: ple_embedding.Binding
    storage: ple_embedding.TableStorage
    expected: torch.Tensor
    initialized_local_rows: int

    def launch(self) -> torch.Tensor:
        return ple_embedding.run(self.binding)

    def close(self) -> None:
        self.storage.close()


@dataclass
class LayerCase:
    profile: LayerProfile
    binding: ple.Binding
    initial_state: torch.Tensor
    expected_out: torch.Tensor
    expected_state: torch.Tensor

    def restore(self) -> None:
        self.binding.conv_state.copy_(self.initial_state)

    def launch(self) -> torch.Tensor:
        if self.profile.mode == "decode":
            return ple.run_decode(self.binding, eps=EPS)
        if self.profile.mode == "prefill":
            return ple.run_prefill(self.binding, eps=EPS)
        return ple.run_mixed(self.binding, eps=EPS)


def _parse_filter(
    value: str,
    *,
    choices: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    entries = tuple(part.strip() for part in value.split(",") if part.strip())
    if not entries:
        raise ValueError(f"{label} filter must not be empty")
    if "all" in entries:
        if entries != ("all",):
            raise ValueError(f"{label}=all cannot be combined with named values")
        return choices
    unknown = tuple(entry for entry in entries if entry not in choices)
    if unknown:
        raise ValueError(
            f"unknown {label}: {', '.join(unknown)}; choices are {', '.join(choices)}"
        )
    return tuple(dict.fromkeys(entries))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apis", default="all")
    parser.add_argument("--packed-profiles", default="all")
    parser.add_argument("--layer-profiles", default="all")
    parser.add_argument("--quant-modes", default="all")
    parser.add_argument(
        "--table-memory",
        default="device",
        help="device (default) or explicitly selected mapped_host",
    )
    parser.add_argument(
        "--embedding-geometry",
        choices=tuple(EMBEDDING_GEOMETRIES),
        default="storage-scaled",
    )
    parser.add_argument(
        "--allow-production-table",
        action="store_true",
        help="acknowledge the multi-GB production embedding-table allocation",
    )
    parser.add_argument("--modes", default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="print selected contracts without requiring CUDA or allocating tables",
    )
    return parser


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Iterable[str] | None,
) -> argparse.Namespace:
    args = parser.parse_args(argv)
    packed_names = tuple(profile.name for profile in PACKED_PROFILES)
    layer_names = tuple(profile.name for profile in LAYER_PROFILES)
    try:
        args.selected_apis = _parse_filter(args.apis, choices=APIS, label="apis")
        selected_packed = _parse_filter(
            args.packed_profiles,
            choices=packed_names,
            label="packed profiles",
        )
        selected_layers = _parse_filter(
            args.layer_profiles,
            choices=layer_names,
            label="layer profiles",
        )
        args.selected_quant_modes = _parse_filter(
            args.quant_modes,
            choices=QUANT_MODES,
            label="quant modes",
        )
        args.selected_table_memories = _parse_filter(
            args.table_memory,
            choices=TABLE_MEMORIES,
            label="table memory",
        )
        args.selected_modes = _parse_filter(
            args.modes,
            choices=TIMING_MODES,
            label="timing modes",
        )
    except ValueError as error:
        parser.error(str(error))
    args.selected_packed_profiles = tuple(
        profile for profile in PACKED_PROFILES if profile.name in selected_packed
    )
    args.selected_layer_profiles = tuple(
        profile for profile in LAYER_PROFILES if profile.name in selected_layers
    )
    if args.warmup <= 0 or args.samples <= 0:
        parser.error("warmup and samples must be positive")
    if args.l2_flush_bytes < 0:
        parser.error("--l2-flush-bytes must be non-negative")
    if (
        args.embedding_geometry == "production"
        and not args.allow_production_table
        and not args.list_profiles
        and "embedding" in args.selected_apis
    ):
        parser.error(
            "production embedding tables require --allow-production-table; "
            "use --list-profiles to inspect their byte counts without allocation"
        )
    return args


def _scratch(plan: Any) -> torch.Tensor:
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _query_start_loc(
    query_lengths: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=torch.int32, device=device)


def _packed_metadata(
    profile: PackedProfile,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_ids = torch.randint(
        0,
        VOCAB_SIZE - 1,
        (profile.tokens,),
        generator=generator,
        dtype=torch.int64,
    )
    if profile.tokens >= 8:
        token_ids[profile.tokens // 3] = EOS_TOKEN_ID
    history = torch.randint(
        0,
        VOCAB_SIZE - 1,
        (profile.sequences, MAX_ORDER - 1),
        generator=generator,
        dtype=torch.int64,
    )
    history[0, 0] = EOS_TOKEN_ID
    return {
        "token_ids": token_ids.to(device),
        "query_start_loc": _query_start_loc(profile.query_lengths, device),
        "committed_history": history.to(device),
        "num_seqs": torch.tensor([profile.sequences], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([profile.tokens], dtype=torch.int32, device=device),
    }


def build_hash_case(
    profile: PackedProfile,
    *,
    device: torch.device | str,
    seed: int,
) -> HashCase:
    """Build production-geometry hash through public Caps/plan/bind."""
    device = torch.device(device)
    plan = ple_hash.plan(
        ple_hash.Caps(
            device=device,
            max_tokens=profile.tokens,
            max_seqs=profile.sequences,
            vocab_size=VOCAB_SIZE,
            eos_token_id=EOS_TOKEN_ID,
            max_order=MAX_ORDER,
            heads_per_order=HEADS_PER_ORDER,
            dense_layer_ordinal=0,
            base_table_size=PRODUCTION_BASE_TABLE_SIZE,
            table_alignment=TABLE_ALIGNMENT,
        )
    )
    metadata = _packed_metadata(profile, device=device, seed=seed)
    binding = ple_hash.bind(
        plan,
        scratch=_scratch(plan),
        **metadata,
        out=torch.empty((profile.tokens, HEAD_COUNT), dtype=torch.int64, device=device),
    )
    expected = ple_hash_packed_reference(
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        eos_token_id=EOS_TOKEN_ID,
        multipliers=plan.multipliers,
        prime_sizes=plan.prime_sizes,
        table_offsets=plan.table_offsets,
        heads_per_order=HEADS_PER_ORDER,
    )
    return HashCase(profile=profile, binding=binding, expected=expected)


def build_embedding_plan(
    profile: PackedProfile,
    *,
    device: torch.device | str,
    quant_mode: str,
    geometry: EmbeddingGeometry,
    table_memory: str,
) -> ple_embedding.Plan:
    """Build the public PLE embedding plan without allocating table storage."""
    return ple_embedding.plan(
        ple_embedding.Caps(
            device=device,
            max_tokens=profile.tokens,
            max_seqs=profile.sequences,
            vocab_size=VOCAB_SIZE,
            eos_token_id=EOS_TOKEN_ID,
            max_order=MAX_ORDER,
            heads_per_order=HEADS_PER_ORDER,
            dense_layer_ordinal=0,
            base_table_size=geometry.base_table_size,
            embedding_dim=EMBEDDING_DIM,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            table_alignment=TABLE_ALIGNMENT,
            quant_mode=quant_mode,
            table_memory=table_memory,
        )
    )


def _initialize_table_storage(
    storage: ple_embedding.TableStorage,
    *,
    quant_mode: str,
    seed: int,
    global_rows: torch.Tensor,
    shard_start: int,
) -> int:
    """Initialize live local rows with row- and column-distinct signatures."""
    global_rows = torch.unique(global_rows.detach().to("cpu", torch.int64), sorted=True)
    load_device = storage.weight_load_view.device
    local_rows = (global_rows - int(shard_start)).to(load_device)
    logical_width = (
        storage.weight_load_view.shape[1] * 2
        if quant_mode == "nvfp4_group16"
        else storage.weight_load_view.shape[1]
    )

    rows = global_rows.to(load_device).view(-1, 1)
    columns = torch.arange(logical_width, dtype=torch.int64, device=load_device).view(
        1, -1
    )
    bit_indices = torch.remainder(columns + int(seed), 29)
    sign_bits = torch.bitwise_and(torch.bitwise_right_shift(rows, bit_indices), 1)
    magnitude_bits = torch.bitwise_and(
        torch.bitwise_right_shift(rows, torch.remainder(bit_indices + 7, 29)),
        3,
    )

    storage.weight_load_view.zero_()
    if quant_mode == "bf16":
        values = (magnitude_bits.float() + 1.0).mul_(0.25)
        values = torch.where(sign_bits.bool(), -values, values).to(torch.bfloat16)
        storage.weight_load_view.index_copy_(0, local_rows, values)
    elif quant_mode == "fp8_e4m3_per_tensor":
        values = (magnitude_bits.float() + 1.0).mul_(0.25)
        values = torch.where(sign_bits.bool(), -values, values).to(torch.float8_e4m3fn)
        storage.weight_load_view.view(torch.uint8).index_copy_(
            0, local_rows, values.view(torch.uint8)
        )
        assert storage.weight_scale_load_view is not None
        storage.weight_scale_load_view.fill_(0.25)
    else:
        codes = (magnitude_bits + 1 + sign_bits * 8).to(torch.uint8)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        storage.weight_load_view.index_copy_(0, local_rows, packed)
        assert storage.weight_scale_load_view is not None
        assert storage.weight_scale_2_load_view is not None
        storage.weight_scale_load_view.zero_()
        scales = torch.ones(
            (global_rows.numel(), storage.weight_scale_load_view.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=load_device,
        )
        storage.weight_scale_load_view.view(torch.uint8).index_copy_(
            0, local_rows, scales.view(torch.uint8)
        )
        storage.weight_scale_2_load_view.fill_(0.25)
    return int(global_rows.numel())


def build_embedding_case(
    profile: PackedProfile,
    *,
    device: torch.device | str,
    seed: int,
    quant_mode: str,
    geometry: EmbeddingGeometry,
    table_memory: str,
) -> EmbeddingCase:
    """Build lookup through public Caps/plan/allocate_storage/bind."""
    device = torch.device(device)
    plan = build_embedding_plan(
        profile,
        device=device,
        quant_mode=quant_mode,
        geometry=geometry,
        table_memory=table_memory,
    )
    storage = ple_embedding.allocate_storage(plan)
    try:
        metadata = _packed_metadata(profile, device=device, seed=seed)
        live_ids = ple_hash_packed_reference(
            metadata["token_ids"],
            metadata["query_start_loc"],
            metadata["committed_history"],
            eos_token_id=EOS_TOKEN_ID,
            multipliers=plan.multipliers,
            prime_sizes=plan.prime_sizes,
            table_offsets=plan.table_offsets,
            heads_per_order=HEADS_PER_ORDER,
        )
        local_live_ids = live_ids[
            (live_ids >= plan.shard_start) & (live_ids < plan.shard_end)
        ]
        initialized_local_rows = _initialize_table_storage(
            storage,
            quant_mode=quant_mode,
            seed=seed,
            global_rows=local_live_ids,
            shard_start=plan.shard_start,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        binding = ple_embedding.bind(
            plan,
            scratch=_scratch(plan),
            weight=storage.weight,
            weight_scale=storage.weight_scale,
            weight_scale_2=storage.weight_scale_2,
            **metadata,
            out=torch.empty(plan.output_shape, dtype=plan.output_dtype, device=device),
        )
        expected = ple_embedding_reference.fused(
            binding.weight,
            binding.weight_scale,
            binding.token_ids,
            binding.query_start_loc,
            binding.committed_history,
            quant_mode=quant_mode,
            weight_scale_2=binding.weight_scale_2,
            num_seqs=profile.sequences,
            num_tokens=profile.tokens,
            eos_token_id=EOS_TOKEN_ID,
            multipliers=plan.multipliers,
            prime_sizes=plan.prime_sizes,
            table_offsets=plan.table_offsets,
            heads_per_order=HEADS_PER_ORDER,
            shard_start=plan.shard_start,
            embedding_dim=EMBEDDING_DIM,
            output_dtype=plan.output_dtype,
        )
        return EmbeddingCase(
            profile=profile,
            binding=binding,
            storage=storage,
            expected=expected,
            initialized_local_rows=initialized_local_rows,
        )
    except Exception:
        storage.close()
        raise


def _randn_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    divisor: float,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
        .div_(divisor)
        .to(DTYPE)
        .contiguous()
    )


def _layer_reference(
    binding: ple.Binding,
    profile: LayerProfile,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_out = torch.zeros_like(binding.out)
    expected_state = initial_state.clone()
    starts = [int(value) for value in binding.query_start_loc.detach().cpu().tolist()]
    slots = [int(value) for value in binding.state_slot_ids.detach().cpu().tolist()]
    fresh = [bool(value) for value in binding.state_is_fresh.detach().cpu().tolist()]
    accepted = [
        int(value) for value in binding.num_accepted_tokens.detach().cpu().tolist()
    ]
    if profile.mode == "mixed":
        assert binding.request_is_prefill is not None
        prefill = [
            bool(value) for value in binding.request_is_prefill.detach().cpu().tolist()
        ]
    else:
        prefill = [profile.mode == "prefill"] * profile.sequences
    state_length = binding.plan.state_length
    max_speculative = binding.plan.caps.max_speculative_tokens

    for request, (start, end) in enumerate(zip(starts, starts[1:], strict=False)):
        slot = slots[request]
        if slot < 0 or start == end:
            continue
        if fresh[request]:
            prior = torch.zeros_like(initial_state[slot, :, :state_length])
        else:
            rollback = 0 if prefill[request] else accepted[request] - 1
            prior = initial_state[slot, :, rollback : rollback + state_length]
        contribution, newest = ple_projected_sequence_reference(
            binding.residual[start:end],
            binding.key[start:end],
            binding.value[start:end],
            k_norm_weight=binding.k_norm_weight,
            q_norm_weight=binding.q_norm_weight,
            u_norm_weight=binding.u_norm_weight,
            conv_weight=binding.conv_weight,
            eps=EPS,
            dilation=DILATION,
            prior_state=prior,
        )
        expected_out[start:end].copy_(contribution)
        if prefill[request]:
            committed = newest
        else:
            _, committed = ple_projected_sequence_reference(
                binding.residual[start : start + 1],
                binding.key[start : start + 1],
                binding.value[start : start + 1],
                k_norm_weight=binding.k_norm_weight,
                q_norm_weight=binding.q_norm_weight,
                u_norm_weight=binding.u_norm_weight,
                conv_weight=binding.conv_weight,
                eps=EPS,
                dilation=DILATION,
                prior_state=prior,
            )
        state = expected_state[slot]
        state[:, :state_length].copy_(committed)
        state[:, state_length:].zero_()
        if not prefill[request] and end - start > 1:
            _, normalized_u = ple_projected_u_reference(
                binding.residual[start:end],
                binding.key[start:end],
                binding.value[start:end],
                k_norm_weight=binding.k_norm_weight,
                q_norm_weight=binding.q_norm_weight,
                u_norm_weight=binding.u_norm_weight,
                eps=EPS,
            )
            candidates = min(end - start - 1, max_speculative)
            state[:, state_length : state_length + candidates].copy_(
                normalized_u[1 : candidates + 1].transpose(0, 1)
            )
    return expected_out, expected_state


def build_layer_case(
    profile: LayerProfile,
    *,
    device: torch.device | str,
    seed: int,
    compute_reference: bool = True,
) -> LayerCase:
    """Build an exact Qwen PLE layer through public Caps/plan/bind."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    plan = ple.plan(
        ple.Caps(
            device=device,
            mode=profile.mode,
            max_tokens=profile.tokens,
            max_seqs=profile.sequences,
            max_state_slots=profile.sequences,
            max_speculative_tokens=MAX_SPECULATIVE_TOKENS,
            streams=STREAMS,
            hidden_size=HIDDEN_SIZE,
            kernel_size=KERNEL_SIZE,
            dilation=DILATION,
            dtype=DTYPE,
        )
    )
    residual = _randn_bf16(
        (profile.tokens, STREAMS, HIDDEN_SIZE),
        generator=generator,
        device=device,
        divisor=4.0,
    )
    key = _randn_bf16(
        (profile.tokens, STREAMS, HIDDEN_SIZE),
        generator=generator,
        device=device,
        divisor=4.0,
    )
    value = _randn_bf16(
        (profile.tokens, HIDDEN_SIZE),
        generator=generator,
        device=device,
        divisor=4.0,
    )
    weights = tuple(
        _randn_bf16(
            (STREAMS * HIDDEN_SIZE,),
            generator=generator,
            device=device,
            divisor=32.0,
        )
        for _ in range(3)
    )
    conv_state = _randn_bf16(
        (profile.sequences, STREAMS * HIDDEN_SIZE, plan.state_capacity),
        generator=generator,
        device=device,
        divisor=8.0,
    )
    request_is_prefill = None
    if profile.request_is_prefill is not None:
        request_is_prefill = torch.tensor(
            profile.request_is_prefill, dtype=torch.bool, device=device
        )
    binding = ple.bind(
        plan,
        scratch=_scratch(plan),
        residual=residual,
        key=key,
        value=value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=_randn_bf16(
            (STREAMS * HIDDEN_SIZE, KERNEL_SIZE),
            generator=generator,
            device=device,
            divisor=32.0,
        ),
        query_start_loc=_query_start_loc(profile.query_lengths, device),
        state_slot_ids=torch.arange(
            profile.sequences, dtype=torch.int64, device=device
        ),
        state_is_fresh=torch.tensor(
            profile.state_is_fresh, dtype=torch.bool, device=device
        ),
        num_accepted_tokens=torch.tensor(
            profile.num_accepted_tokens, dtype=torch.int32, device=device
        ),
        num_seqs=torch.tensor([profile.sequences], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([profile.tokens], dtype=torch.int32, device=device),
        conv_state=conv_state,
        out=torch.empty_like(residual),
        request_is_prefill=request_is_prefill,
    )
    initial_state = conv_state.clone()
    if compute_reference:
        expected_out, expected_state = _layer_reference(binding, profile, initial_state)
    else:
        expected_out = torch.empty_like(binding.out)
        expected_state = torch.empty_like(initial_state)
    return LayerCase(profile, binding, initial_state, expected_out, expected_state)


def _finite_nonzero_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    floating = actual.float() if actual.is_floating_point() else actual
    finite = bool(torch.isfinite(floating).all().item())
    nonzero = int(torch.count_nonzero(actual).item())
    if not finite or nonzero == 0:
        raise AssertionError(f"finite/nonzero gate failed: {finite=}, {nonzero=}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    difference = actual.float() - expected.float()
    return {
        "finite": finite,
        "nonzero_elements": nonzero,
        "max_abs": float(difference.abs().max().item()),
        "rtol": rtol,
        "atol": atol,
    }


def _validate_hash(case: HashCase) -> dict[str, object]:
    return {
        "status": "passed",
        "oracle": "b12x.sequence.ple_hash.reference.ple_hash_packed_reference",
        "output": _finite_nonzero_close(
            case.binding.out, case.expected, rtol=0.0, atol=0.0
        ),
    }


def _validate_embedding(case: EmbeddingCase) -> dict[str, object]:
    return {
        "status": "passed",
        "oracle": "b12x.sequence.ple_embedding.reference.fused",
        "output": _finite_nonzero_close(
            case.binding.out, case.expected, rtol=0.0, atol=0.0
        ),
    }


def _validate_layer(case: LayerCase) -> dict[str, object]:
    return {
        "status": "passed",
        "oracle": "b12x.sequence.ple.reference",
        "output": _finite_nonzero_close(
            case.binding.out, case.expected_out, rtol=2.0e-2, atol=7.8125e-3
        ),
        "state": _finite_nonzero_close(
            case.binding.conv_state,
            case.expected_state,
            rtol=2.0e-2,
            atol=7.8125e-3,
        ),
    }


def _eager_samples(launch, *, prepare, warmup, samples, l2_flush):
    from b12x.testing.benchmark import transaction_samples
    return transaction_samples(launch, samples=samples, warmup=warmup,
                               prepare=prepare, l2_flush=l2_flush)


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(samples: list[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("timing summary requires at least one sample")
    return {
        "unit": "us",
        "count": len(samples),
        "minimum": min(samples),
        "p10": _percentile(samples, 0.10),
        "median": statistics.median(samples),
        "p90": _percentile(samples, 0.90),
        "maximum": max(samples),
        "raw_samples_us": samples,
    }


def _time_case(
    *,
    launch: Callable[[], torch.Tensor],
    validate: Callable[[], dict[str, object]],
    address_tensors: dict[str, torch.Tensor],
    prepare: Callable[[], None] | None,
    mode: str,
    warmup: int,
    samples: int,
    l2_flush: Callable[[], None] | None,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    if prepare is not None:
        prepare()
    output = launch()
    torch.cuda.synchronize(device)
    correctness = validate()
    if output.data_ptr() != address_tensors["out"].data_ptr():
        raise AssertionError("public PLE run did not return the bound output")

    if mode == "eager":
        raw = _eager_samples(
            launch,
            prepare=prepare,
            warmup=warmup,
            samples=samples,
            l2_flush=l2_flush,
        )
        correctness = validate()
        return (
            {
                "kernel": _summary(raw["kernel_us"]),
                "restore": _summary(raw["restore_us"]) if prepare else None,
                "step": _summary(raw["step_us"]),
            },
            correctness,
            None,
        )

    graph = capture_cuda_graph(launch, warmup=warmup, prepare=prepare)
    addresses = {name: tensor.data_ptr() for name, tensor in address_tensors.items()}
    if prepare is not None:
        prepare()
    if address_tensors["out"].is_floating_point():
        address_tensors["out"].fill_(float("nan"))
    else:
        address_tensors["out"].fill_(-1)
    address_tensors["scratch"].fill_(0xFF)
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    replay_allocation = allocated_after - allocated_before
    if replay_allocation != 0:
        raise AssertionError(f"CUDA graph replay allocated {replay_allocation} bytes")
    if addresses != {
        name: tensor.data_ptr() for name, tensor in address_tensors.items()
    }:
        raise AssertionError("bound PLE tensor addresses changed across graph replay")
    correctness = validate()
    graph_raw = bench_cuda_graph(
        graph,
        replays=samples,
        prepare=prepare,
        l2_flush=l2_flush,
    )
    correctness = validate()
    return (
        {
            "kernel": _summary(list(graph_raw["replay_us"])),
            "restore": (_summary(list(graph_raw["metadata_us"])) if prepare else None),
            "step": _summary(list(graph_raw["step_us"])),
        },
        correctness,
        {
            "stable_bound_addresses": True,
            "addresses": addresses,
            "replay_allocation_delta_bytes": replay_allocation,
            "replay_after_output_poison": True,
            "replay_after_scratch_poison": True,
        },
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _embedding_storage_contract(
    geometry: EmbeddingGeometry, quant_mode: str
) -> dict[str, object]:
    profile = PACKED_PROFILES[0]
    plan = build_embedding_plan(
        profile,
        device="cpu",
        quant_mode=quant_mode,
        geometry=geometry,
        table_memory="device",
    )
    weight_nbytes = math.prod(plan.weight_shape) * _dtype_nbytes(plan.weight_dtype)
    scale_nbytes = 0
    if plan.weight_scale_shape is not None:
        assert plan.weight_scale_dtype is not None
        scale_nbytes += math.prod(plan.weight_scale_shape) * _dtype_nbytes(
            plan.weight_scale_dtype
        )
    if plan.weight_scale_2_shape is not None:
        assert plan.weight_scale_2_dtype is not None
        scale_nbytes += math.prod(plan.weight_scale_2_shape) * _dtype_nbytes(
            plan.weight_scale_2_dtype
        )
    return {
        "geometry": geometry.name,
        "storage_scope": geometry.storage_scope,
        "production_storage": geometry.production_storage,
        "base_table_size": geometry.base_table_size,
        "table_vocab_size": plan.table_vocab_size,
        "padded_vocab_size": plan.padded_vocab_size,
        "tp_size": TP_SIZE,
        "tp_rank": TP_RANK,
        "shard_start": plan.shard_start,
        "shard_end": plan.shard_end,
        "head_count": plan.head_count,
        "head_dim": plan.head_dim,
        "embedding_dim": EMBEDDING_DIM,
        "weight_shape": list(plan.weight_shape),
        "weight_dtype": str(plan.weight_dtype),
        "weight_nbytes": weight_nbytes,
        "scale_nbytes": scale_nbytes,
        "persistent_nbytes": weight_nbytes + scale_nbytes,
    }


def _profile_listing(args: argparse.Namespace) -> dict[str, object]:
    geometry = EMBEDDING_GEOMETRIES[args.embedding_geometry]
    return {
        "schema": SCHEMA,
        "apis": list(args.selected_apis),
        "packed_profiles": [
            asdict(profile) for profile in args.selected_packed_profiles
        ],
        "layer_profiles": [asdict(profile) for profile in args.selected_layer_profiles],
        "qwen_contract": {
            "hash_base_table_size": PRODUCTION_BASE_TABLE_SIZE,
            "heads_per_order": HEADS_PER_ORDER,
            "head_count": HEAD_COUNT,
            "embedding_dim": EMBEDDING_DIM,
            "layer": {
                "streams": STREAMS,
                "hidden_size": HIDDEN_SIZE,
                "kernel_size": KERNEL_SIZE,
                "dilation": DILATION,
                "max_speculative_tokens": MAX_SPECULATIVE_TOKENS,
            },
        },
        "embedding_storage": [
            _embedding_storage_contract(geometry, mode)
            for mode in args.selected_quant_modes
        ],
        "table_memories": list(args.selected_table_memories),
        "timing_modes": list(args.selected_modes),
    }


def _git_provenance() -> dict[str, object]:
    root = pathlib.Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "worktree": str(root),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty_paths": git("status", "--short").splitlines(),
    }


def _device_provenance(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "logical_device": device.index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
    }


class _Emitter:
    def __init__(self, output: pathlib.Path | None) -> None:
        self.output = output
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8"):
                pass

    def emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.output is not None:
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def _address_tensors(binding: Any, *, stateful: bool) -> dict[str, torch.Tensor]:
    tensors = {"out": binding.out, "scratch": binding.scratch}
    if stateful:
        tensors["conv_state"] = binding.conv_state
    return tensors


def _emit_timed_result(
    emitter: _Emitter,
    *,
    api: str,
    profile: PackedProfile | LayerProfile,
    case_seed: int,
    mode: str,
    launch: Callable[[], torch.Tensor],
    validate: Callable[[], dict[str, object]],
    address_tensors: dict[str, torch.Tensor],
    prepare: Callable[[], None] | None,
    warmup: int,
    samples: int,
    l2_flush: Callable[[], None] | None,
    device: torch.device,
    extra: dict[str, object] | None = None,
) -> None:
    timing, correctness, graph_contract = _time_case(
        launch=launch,
        validate=validate,
        address_tensors=address_tensors,
        prepare=prepare,
        mode=mode,
        warmup=warmup,
        samples=samples,
        l2_flush=l2_flush,
        device=device,
    )
    record: dict[str, object] = {
        "type": "result",
        "schema": SCHEMA,
        "api": api,
        "profile": asdict(profile),
        "case_seed": case_seed,
        "timing_mode": mode,
        "correctness": correctness,
        "graph_contract": graph_contract,
        "timing": timing,
    }
    if extra:
        record.update(extra)
    emitter.emit(record)


def _validate_device(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise SystemExit(
            f"Qwen3.8 Flash Next PLE benchmarking requires SM120/SM121, got {capability}"
        )
    for name, module in (
        ("ple", ple),
        ("ple_hash", ple_hash),
        ("ple_embedding", ple_embedding),
    ):
        if not module.is_supported(device):
            raise SystemExit(f"b12x.sequence.{name} is unsupported on {device}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = _parse_args(parser, argv)
    if args.list_profiles:
        print(json.dumps(_profile_listing(args), indent=2, sort_keys=True))
        return

    require_sm120()
    device = torch.device("cuda", torch.cuda.current_device())
    _validate_device(device)
    geometry = EMBEDDING_GEOMETRIES[args.embedding_geometry]
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    emitter = _Emitter(args.output)
    command_argv = list(sys.argv[1:] if argv is None else argv)
    gpu_mode_before = nvidia_smi_gpu_mode_snapshot()
    emitter.emit(
        {
            "type": "provenance",
            "schema": SCHEMA,
            "command": shlex.join(
                [sys.executable, str(pathlib.Path(__file__).resolve()), *command_argv]
            ),
            "git": _git_provenance(),
            "device": _device_provenance(device),
            "gpu_mode_before": gpu_mode_before,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "runtime_requirements": {
                "ple": list(ple.META.requires),
                "ple_hash": list(ple_hash.META.requires),
                "ple_embedding": list(ple_embedding.META.requires),
            },
            "fallback_path": "none",
            "reference_timed": False,
            "raw_samples_preserved": True,
            "state_restore_in_kernel_timing": False,
            "selection": _profile_listing(args),
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "flush_l2": args.flush_l2,
            "l2_flush_bytes": args.l2_flush_bytes,
        }
    )

    with torch.inference_mode():
        if "hash" in args.selected_apis:
            for index, profile in enumerate(args.selected_packed_profiles):
                case_seed = args.seed + 10_007 * index
                case = build_hash_case(profile, device=device, seed=case_seed)
                for mode in args.selected_modes:
                    _emit_timed_result(
                        emitter,
                        api="ple_hash.run",
                        profile=profile,
                        case_seed=case_seed,
                        mode=mode,
                        launch=case.launch,
                        validate=lambda active=case: _validate_hash(active),
                        address_tensors=_address_tensors(case.binding, stateful=False),
                        prepare=None,
                        warmup=args.warmup,
                        samples=args.samples,
                        l2_flush=l2_flush,
                        device=device,
                        extra={
                            "hash_geometry": "production",
                            "base_table_size": PRODUCTION_BASE_TABLE_SIZE,
                        },
                    )

        if "embedding" in args.selected_apis:
            for memory_index, table_memory in enumerate(args.selected_table_memories):
                for quant_index, quant_mode in enumerate(args.selected_quant_modes):
                    storage_contract = _embedding_storage_contract(geometry, quant_mode)
                    for profile_index, profile in enumerate(
                        args.selected_packed_profiles
                    ):
                        case_seed = (
                            args.seed
                            + 100_003 * memory_index
                            + 10_007 * quant_index
                            + profile_index
                        )
                        case = build_embedding_case(
                            profile,
                            device=device,
                            seed=case_seed,
                            quant_mode=quant_mode,
                            geometry=geometry,
                            table_memory=table_memory,
                        )
                        try:
                            for mode in args.selected_modes:
                                _emit_timed_result(
                                    emitter,
                                    api="ple_embedding.run",
                                    profile=profile,
                                    case_seed=case_seed,
                                    mode=mode,
                                    launch=case.launch,
                                    validate=lambda active=case: _validate_embedding(
                                        active
                                    ),
                                    address_tensors=_address_tensors(
                                        case.binding, stateful=False
                                    ),
                                    prepare=None,
                                    warmup=args.warmup,
                                    samples=args.samples,
                                    l2_flush=l2_flush,
                                    device=device,
                                    extra={
                                        "quant_mode": quant_mode,
                                        "table_memory": table_memory,
                                        "mapped_host_nbytes": (
                                            case.storage.mapped_host_nbytes
                                        ),
                                        "table_fixture": (
                                            "zero_default_with_live_row_signatures"
                                        ),
                                        "initialized_local_rows": (
                                            case.initialized_local_rows
                                        ),
                                        "storage_contract": storage_contract,
                                    },
                                )
                        finally:
                            case.close()
                            del case

        selected_layer_apis = set(args.selected_apis) & {
            "decode",
            "prefill",
            "mixed",
        }
        for profile_index, profile in enumerate(args.selected_layer_profiles):
            if profile.mode not in selected_layer_apis:
                continue
            case_seed = args.seed + 1_000_003 + profile_index * 10_007
            case = build_layer_case(profile, device=device, seed=case_seed)
            for mode in args.selected_modes:
                _emit_timed_result(
                    emitter,
                    api=f"ple.run_{profile.mode}",
                    profile=profile,
                    case_seed=case_seed,
                    mode=mode,
                    launch=case.launch,
                    validate=lambda active=case: _validate_layer(active),
                    address_tensors=_address_tensors(case.binding, stateful=True),
                    prepare=case.restore,
                    warmup=args.warmup,
                    samples=args.samples,
                    l2_flush=l2_flush,
                    device=device,
                    extra={
                        "layer_geometry": {
                            "streams": STREAMS,
                            "hidden_size": HIDDEN_SIZE,
                            "kernel_size": KERNEL_SIZE,
                            "dilation": DILATION,
                            "max_speculative_tokens": MAX_SPECULATIVE_TOKENS,
                            "state_length": case.binding.plan.state_length,
                            "state_capacity": case.binding.plan.state_capacity,
                        }
                    },
                )

    emitter.emit(
        {
            "type": "completion",
            "schema": SCHEMA,
            "status": "passed",
            "gpu_mode_before": gpu_mode_before,
            "gpu_mode_after": nvidia_smi_gpu_mode_snapshot(),
        }
    )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
