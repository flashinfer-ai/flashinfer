"""Save/load pre-quantized MegaMoE weights and activations for moe_ep benchmarks.

Fixtures are written per EP rank under ``--fixture-dir``:

    meta.pt
    inputs_rank{R}.pt                      # bf16 source activations (save-fixtures only)
    activations_{backend_id}_rank{R}.pt    # kernel-ready prestaged activations
    weights_{backend_id}_rank{R}.pt

``save-fixtures`` quantizes bf16 weights and activations once offline. ``bench``
loads kernel-ready weight and activation tensors only (memcpy stage + kernel).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

from moe_ep_common import (
    BACKEND_IDS,
    BenchmarkInputs,
    BenchmarkWeights,
    FI_MEGAKERNEL_BY_BACKEND,
    WEIGHT_SEED_BASE,
    make_benchmark_routing_inputs,
    make_benchmark_weights,
)

TRANSFORMED_WEIGHTS_KEY = "transformed_weights"
VLLM_FP4_SHARDS_KEY = "fp4_shards"


@dataclass(frozen=True)
class FixtureMeta:
    hidden: int
    intermediate: int
    num_tokens: int
    num_max_tokens: int
    num_local_experts: int
    topk: int
    world_size: int
    activation_clamp: float | None
    fast_math: bool
    weight_seed_base: int = WEIGHT_SEED_BASE


def meta_path(fixture_dir: str) -> str:
    return os.path.join(fixture_dir, "meta.pt")


def inputs_path(fixture_dir: str, rank: int) -> str:
    return os.path.join(fixture_dir, f"inputs_rank{rank}.pt")


def weights_path(fixture_dir: str, backend_id: str, rank: int) -> str:
    return os.path.join(fixture_dir, f"weights_{backend_id}_rank{rank}.pt")


def activations_path(fixture_dir: str, backend_id: str, rank: int) -> str:
    return os.path.join(fixture_dir, f"activations_{backend_id}_rank{rank}.pt")


def save_meta(fixture_dir: str, meta: FixtureMeta) -> None:
    os.makedirs(fixture_dir, exist_ok=True)
    torch.save(asdict(meta), meta_path(fixture_dir))


def load_meta(fixture_dir: str) -> FixtureMeta:
    payload = torch.load(meta_path(fixture_dir), map_location="cpu", weights_only=False)
    return FixtureMeta(**payload)


def save_inputs(fixture_dir: str, rank: int, inputs: BenchmarkInputs) -> None:
    os.makedirs(fixture_dir, exist_ok=True)
    torch.save(
        {
            "hidden_states": inputs.hidden_states.cpu(),
            "topk_weights": inputs.topk_weights.cpu(),
            "topk_ids": inputs.topk_ids.cpu(),
        },
        inputs_path(fixture_dir, rank),
    )


def load_inputs(fixture_dir: str, rank: int) -> BenchmarkInputs:
    payload = torch.load(
        inputs_path(fixture_dir, rank),
        map_location="cpu",
        weights_only=False,
    )
    return BenchmarkInputs(
        hidden_states=payload["hidden_states"].cuda(),
        topk_weights=payload["topk_weights"].cuda(),
        topk_ids=payload["topk_ids"].cuda(),
    )


def _moe_ep_tensors_to_payload(tensors) -> dict[str, torch.Tensor]:
    payload = {
        "hidden_states": tensors.hidden_states.detach().cpu(),
        "topk_ids": tensors.topk_ids.detach().cpu(),
        "topk_weights": tensors.topk_weights.detach().cpu(),
    }
    for field_name in ("scales", "fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        value = getattr(tensors, field_name, None)
        if value is not None:
            payload[field_name] = value.detach().cpu()
    return payload


def _payload_to_moe_ep_tensors(
    payload: dict[str, torch.Tensor],
    *,
    device: torch.device | str = "cuda",
):
    from flashinfer.moe_ep import MoEEpTensors

    kwargs: dict[str, torch.Tensor] = {
        "hidden_states": payload["hidden_states"].to(device=device),
        "topk_ids": payload["topk_ids"].to(device=device),
        "topk_weights": payload["topk_weights"].to(device=device),
    }
    for field_name in ("scales", "fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        if field_name in payload:
            kwargs[field_name] = payload[field_name].to(device=device)
    return MoEEpTensors(**kwargs)


def save_prestaged_activations(
    fixture_dir: str,
    backend_id: str,
    rank: int,
    tensors,
) -> None:
    os.makedirs(fixture_dir, exist_ok=True)
    torch.save(_moe_ep_tensors_to_payload(tensors), activations_path(fixture_dir, backend_id, rank))


def load_prestaged_activations(
    fixture_dir: str,
    backend_id: str,
    rank: int,
    *,
    device: torch.device | str = "cuda",
):
    payload = torch.load(
        activations_path(fixture_dir, backend_id, rank),
        map_location="cpu",
        weights_only=False,
    )
    return _payload_to_moe_ep_tensors(payload, device=device)


def materialize_deep_gemm_prestaged_activations(inputs: BenchmarkInputs):
    """bf16 activations → deep_gemm fp8 + scale layout used by vLLM / fi_deep_gemm."""
    from deep_gemm.utils import per_token_cast_to_fp8
    from flashinfer.moe_ep import MoEEpTensors

    x_q, x_sf = per_token_cast_to_fp8(
        inputs.hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    return MoEEpTensors(
        hidden_states=x_q,
        topk_ids=inputs.topk_ids,
        topk_weights=inputs.topk_weights,
        scales=x_sf,
    )


def materialize_prestaged_activations_for_backend(
    backend_id: str,
    inputs: BenchmarkInputs,
    *,
    rank: int,
    world_size: int,
    meta: FixtureMeta,
    num_experts: int,
    num_local_experts: int,
    vllm_config=None,
):
    if backend_id in ("vllm_deepgemm", "fi_deep_gemm"):
        return materialize_deep_gemm_prestaged_activations(inputs)

    if backend_id in ("fi_nvfp4", "fi_mxfp8"):
        from backends import make_prestaged_fi_tensors

        return make_prestaged_fi_tensors(
            backend_id,
            inputs,
            rank=rank,
            world_size=world_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden=meta.hidden,
            intermediate=meta.intermediate,
            num_max_tokens=meta.num_max_tokens,
            topk=meta.topk,
            activation_clamp=meta.activation_clamp,
            use_vllm_ep_group=vllm_config is not None,
        )

    raise ValueError(f"unsupported backend_id {backend_id!r} for prestaged activations")


def _move_transformed_to_device(
    transformed: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str = "cuda",
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    fc1_w, fc1_sf = transformed[0]
    fc2_w, fc2_sf = transformed[1]
    return (
        (fc1_w.to(device=device), fc1_sf.to(device=device)),
        (fc2_w.to(device=device), fc2_sf.to(device=device)),
    )


def save_transformed_weights(
    fixture_dir: str,
    backend_id: str,
    rank: int,
    transformed: Any,
) -> None:
    os.makedirs(fixture_dir, exist_ok=True)
    fc1_w, fc1_sf = transformed[0]
    fc2_w, fc2_sf = transformed[1]
    torch.save(
        {
            TRANSFORMED_WEIGHTS_KEY: (
                (fc1_w.detach().cpu(), fc1_sf.detach().cpu()),
                (fc2_w.detach().cpu(), fc2_sf.detach().cpu()),
            ),
        },
        weights_path(fixture_dir, backend_id, rank),
    )


def load_transformed_weights(
    fixture_dir: str,
    backend_id: str,
    rank: int,
    *,
    device: torch.device | str = "cuda",
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    payload = torch.load(
        weights_path(fixture_dir, backend_id, rank),
        map_location="cpu",
        weights_only=False,
    )
    return _move_transformed_to_device(payload[TRANSFORMED_WEIGHTS_KEY], device=device)


def _quantize_vllm_fp4_shards(
    weights: BenchmarkWeights,
    *,
    experts_start_idx: int,
    intermediate: int,
) -> list[dict[str, Any]]:
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp4

    from backends import _float_ue8m0_scale_to_uint8

    num_local_experts = weights.w13.shape[0]
    shards: list[dict[str, Any]] = []
    for local_expert_id in range(num_local_experts):
        global_expert_id = experts_start_idx + local_expert_id
        shard_specs = (
            ("w1", weights.w13[local_expert_id, :intermediate]),
            ("w3", weights.w13[local_expert_id, intermediate:]),
            ("w2", weights.w2[local_expert_id]),
        )
        for shard_id, bf16 in shard_specs:
            packed, scale = per_token_cast_to_fp4(
                bf16,
                use_ue8m0=True,
                gran_k=32,
            )
            shards.append(
                {
                    "shard_id": shard_id,
                    "local_expert_id": local_expert_id,
                    "global_expert_id": global_expert_id,
                    "packed": packed.view(torch.uint8).detach().cpu(),
                    "scale_uint8": _float_ue8m0_scale_to_uint8(scale).detach().cpu(),
                }
            )
    return shards


def save_vllm_fp4_shards(
    fixture_dir: str,
    rank: int,
    shards: list[dict[str, Any]],
) -> None:
    os.makedirs(fixture_dir, exist_ok=True)
    torch.save({VLLM_FP4_SHARDS_KEY: shards}, weights_path(fixture_dir, "vllm_deepgemm", rank))


def load_vllm_fp4_shards(
    fixture_dir: str,
    rank: int,
) -> list[dict[str, Any]]:
    payload = torch.load(
        weights_path(fixture_dir, "vllm_deepgemm", rank),
        map_location="cpu",
        weights_only=False,
    )
    return payload[VLLM_FP4_SHARDS_KEY]


def materialize_fi_transformed_weights(
    backend_id: str,
    weights: BenchmarkWeights,
    *,
    hidden: int,
    intermediate: int,
    activation_clamp: float | None,
) -> Any:
    from flashinfer.moe_ep import MoEWeightPack

    megakernel = FI_MEGAKERNEL_BY_BACKEND[backend_id]
    pack = MoEWeightPack(w13=weights.w13, w2=weights.w2)

    if megakernel == "deep_gemm_mega":
        from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.weights import (
            preprocess_mega_weights,
        )

        return preprocess_mega_weights(
            pack,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )

    if megakernel == "nvfp4_cutedsl":
        from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
            preprocess_mega_weights,
        )

        return preprocess_mega_weights(
            pack,
            intermediate_size=intermediate,
            hidden_size=hidden,
            activation_clamp=activation_clamp,
        )

    if megakernel == "mxfp8_cutedsl":
        from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.weights import (
            preprocess_mega_weights,
        )

        return preprocess_mega_weights(
            pack,
            intermediate_size=intermediate,
            hidden_size=hidden,
            activation_clamp=activation_clamp,
        )

    raise ValueError(f"unsupported flashinfer backend {backend_id!r}")


def materialize_vllm_transformed_weights(
    weights: BenchmarkWeights,
    *,
    vllm_config,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
    activation_clamp: float | None,
    fast_math: bool,
) -> Any:
    from backends import build_vllm_mega_moe, release_vllm_experts

    experts = build_vllm_mega_moe(
        vllm_config,
        weights,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_max_tokens=num_max_tokens,
    )
    try:
        return experts._transformed_l1_weights, experts._transformed_l2_weights
    finally:
        release_vllm_experts(vllm_config, experts)


def save_all_fixtures_for_rank(
    *,
    fixture_dir: str,
    rank: int,
    world_size: int,
    meta: FixtureMeta,
    bench_weights: BenchmarkWeights,
    inputs: BenchmarkInputs,
    vllm_config=None,
    num_experts: int,
    experts_start_idx: int,
) -> None:
    save_meta(fixture_dir, meta)
    save_inputs(fixture_dir, rank, inputs)

    fp4_shards = _quantize_vllm_fp4_shards(
        bench_weights,
        experts_start_idx=experts_start_idx,
        intermediate=meta.intermediate,
    )
    save_vllm_fp4_shards(fixture_dir, rank, fp4_shards)

    from backends import ensure_fi_moe_ep_runtime

    use_vllm_ep = vllm_config is not None
    for backend_id in BACKEND_IDS:
        if backend_id == "vllm_deepgemm":
            if vllm_config is None:
                raise RuntimeError("vllm_config is required to save vllm_deepgemm fixtures")
            l1, l2 = materialize_vllm_transformed_weights(
                bench_weights,
                vllm_config=vllm_config,
                num_experts=num_experts,
                num_local_experts=meta.num_local_experts,
                experts_start_idx=experts_start_idx,
                topk=meta.topk,
                hidden=meta.hidden,
                intermediate=meta.intermediate,
                num_max_tokens=meta.num_max_tokens,
                activation_clamp=meta.activation_clamp,
                fast_math=meta.fast_math,
            )
            save_transformed_weights(
                fixture_dir,
                backend_id,
                rank,
                (l1, l2),
            )
            continue

        ensure_fi_moe_ep_runtime(
            rank,
            world_size,
            backend_id,
            use_vllm_ep_group=use_vllm_ep,
        )
        transformed = materialize_fi_transformed_weights(
            backend_id,
            bench_weights,
            hidden=meta.hidden,
            intermediate=meta.intermediate,
            activation_clamp=meta.activation_clamp,
        )
        save_transformed_weights(fixture_dir, backend_id, rank, transformed)

    for backend_id in BACKEND_IDS:
        prestaged = materialize_prestaged_activations_for_backend(
            backend_id,
            inputs,
            rank=rank,
            world_size=world_size,
            meta=meta,
            num_experts=num_experts,
            num_local_experts=meta.num_local_experts,
            vllm_config=vllm_config,
        )
        save_prestaged_activations(fixture_dir, backend_id, rank, prestaged)


def make_fixture_meta_from_args(args, *, world_size: int) -> FixtureMeta:
    from moe_ep_common import activation_clamp_from_args

    return FixtureMeta(
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_tokens=args.num_tokens,
        num_max_tokens=args.num_max_tokens,
        num_local_experts=args.num_local_experts,
        topk=args.topk,
        world_size=world_size,
        activation_clamp=activation_clamp_from_args(args),
        fast_math=args.fast_math,
    )


def make_fixture_inputs(rank: int, meta: FixtureMeta, *, num_experts: int) -> BenchmarkInputs:
    return make_benchmark_routing_inputs(
        rank,
        num_tokens=meta.num_tokens,
        hidden=meta.hidden,
        num_experts=num_experts,
        topk=meta.topk,
    )


def make_fixture_weights(rank: int, meta: FixtureMeta) -> BenchmarkWeights:
    return make_benchmark_weights(
        rank,
        num_local_experts=meta.num_local_experts,
        hidden=meta.hidden,
        intermediate=meta.intermediate,
    )


def require_fixtures_exist(fixture_dir: str, rank: int, backend_ids: list[str]) -> None:
    missing: list[str] = []
    if not os.path.isfile(meta_path(fixture_dir)):
        missing.append(meta_path(fixture_dir))
    if not os.path.isfile(inputs_path(fixture_dir, rank)):
        missing.append(inputs_path(fixture_dir, rank))
    for backend_id in backend_ids:
        weights_file = weights_path(fixture_dir, backend_id, rank)
        if not os.path.isfile(weights_file):
            missing.append(weights_file)
        activations_file = activations_path(fixture_dir, backend_id, rank)
        if not os.path.isfile(activations_file):
            missing.append(activations_file)
    if missing:
        raise FileNotFoundError(
            "Missing fixture files (run save-fixtures first):\n  "
            + "\n  ".join(missing)
        )
