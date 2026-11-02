"""NVLink one-sided dispatch/combine: model-shaped round trips, round sequences
with changing token counts and rank skew, and the workspace layout. A subset of
the cases also runs on the Cake backend, which implements the same protocol.

Each multi-rank case spawns one process per EP rank from this pytest process,
so a single ``pytest`` run covers every EP size the machine has GPUs for:

    pytest tests/moe_ep/test_nvlink_one_sided.py

Ported from TensorRT-LLM's
``tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py``, which tests
the same kernels.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import pytest
import torch

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
# Bytes one fabric.try_put moves; payload rows that are not a multiple of it
# take the fence path even when CFT is requested.
CFT_ALIGNMENT_BYTES = 16
CFT_MAX_TOKENS = 128


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    num_experts: int
    top_k: int
    # Activation format the MoE dispatches.
    dispatch_dtype: Literal["bf16", "blockwise_fp8", "mxfp8", "nvfp4"]


MODELS = {
    model.name: model
    for model in (
        ModelShape("gpt_oss", 2880, 128, 4, "mxfp8"),
        ModelShape("deepseek_v3", 7168, 256, 8, "blockwise_fp8"),
        ModelShape("deepseek_r1_nvfp4", 7168, 256, 8, "nvfp4"),
        ModelShape("qwen3_5_397b_a17b", 4096, 512, 10, "bf16"),
        ModelShape("deepseek_v4_flash", 4096, 256, 6, "mxfp8"),
        ModelShape("deepseek_v4_pro", 7168, 384, 6, "mxfp8"),
        # Kimi K3 communicates its latent MoE width, not the model's 7168.
        ModelShape("kimi_k3", 3584, 896, 16, "mxfp8"),
        # DeepSeek-V4.1-Flash's DSpark draft layers route 3 of 128 experts.
        ModelShape("deepseek_v41_flash_dspark", 4096, 128, 3, "mxfp8"),
    )
}

# QuantFormat member each dispatch dtype maps to (None: unquantized BF16 rows).
_DISPATCH_FORMAT = {
    "bf16": None,
    "blockwise_fp8": "DeepSeekFp8",
    "mxfp8": "MXFP8",
    "nvfp4": "NVFP4",
}


def _expert_bounds(num_experts: int, ep_size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(num_experts, ep_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


class Routing(Enum):
    """Synthetic routing: SPREAD covers every rank, LOCAL picks the source
    rank's own experts, HOTSPOT picks only rank 0's experts."""

    SPREAD = "spread"
    LOCAL = "local"
    HOTSPOT = "hotspot"

    def make_expert_ids(
        self,
        model: ModelShape,
        tokens: tuple[int, ...],
        rank: int,
        round_index: int,
    ) -> torch.Tensor:
        """int32 global expert ids shaped ``[tokens[rank], model.top_k]``."""
        ep_size = len(tokens)
        count = tokens[rank]
        token = torch.arange(count, device="cuda", dtype=torch.int64)[:, None]
        choice = torch.arange(model.top_k, device="cuda", dtype=torch.int64)[None, :]
        if self is Routing.LOCAL:
            owners = torch.full((count, model.top_k), rank, device="cuda")
        elif self is Routing.HOTSPOT:
            owners = torch.zeros((count, model.top_k), device="cuda", dtype=torch.int64)
        else:
            owners = (token + choice + rank + round_index) % ep_size
        bounds = [_expert_bounds(model.num_experts, ep_size, r) for r in range(ep_size)]
        starts = torch.tensor([b[0] for b in bounds], device="cuda")
        sizes = torch.tensor([b[1] - b[0] for b in bounds], device="cuda")
        return (
            starts[owners] + (token + choice + 3 * round_index) % sizes[owners]
        ).int()


@dataclass(frozen=True)
class Round:
    """One dispatch, simulated expert computation and combine.

    ``tokens`` holds the local token count of every EP rank (zero-token ranks
    still take part). ``delay_rank`` sleeps on the GPU after dispatch, before
    the expert computation, to skew that rank against its peers; -1 disables.
    """

    tokens: tuple[int, ...]
    routing: Routing
    delay_rank: int


@dataclass(frozen=True)
class Case:
    """A workload sharing one communicator across ``rounds``, with no host
    synchronization between rounds.

    ``backend`` names the moe_ep communication backend. ``mode`` maps to
    ``NVLinkOneSidedConfig.cft``: auto (None), fence (False) or cft (True),
    each with ``lamport=False``; ``lamport`` is ``lamport=True, cft=False``
    (Lamport steps up to its default token limit, fence steps above it).
    Cake has no CFT transport and only runs ``auto``. ``payload_in_workspace`` writes the expert output into the
    combine input region before combine. ``fp8_combine`` sends combine payloads
    as FP8. ``graph`` captures the whole round sequence in one CUDA graph and
    replays it.
    """

    model: ModelShape
    rounds: tuple[Round, ...]
    mode: Literal["auto", "fence", "cft", "lamport"] = "auto"
    payload_in_workspace: bool = False
    fp8_combine: bool = False
    graph: bool = False
    pdl: bool = True
    eplb: bool = False
    backend: Literal["nvlink_one_sided", "cake"] = "nvlink_one_sided"

    @property
    def ep_size(self) -> int:
        return len(self.rounds[0].tokens)

    @property
    def max_tokens_per_rank(self) -> int:
        """Allocation-time per-rank token limit covering every round."""
        return max(max(r.tokens) for r in self.rounds)


# ----------------------------------------------------------------------------
# Inputs and references
# ----------------------------------------------------------------------------


def _quantize(
    x: torch.Tensor, mode: str
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Plain PyTorch quantizers with token-major scales, as dispatch carries them."""
    num_tokens, hidden = x.shape
    if mode == "bf16":
        return x, None
    if mode == "blockwise_fp8":
        blocks = x.float().view(num_tokens, hidden // 128, 128)
        scale = blocks.abs().amax(-1).clamp(min=1e-12) / FP8_E4M3_MAX
        values = (blocks / scale[..., None]).to(torch.float8_e4m3fn)
        return values.view(num_tokens, hidden), scale.contiguous()
    if mode == "mxfp8":
        blocks = x.float().view(num_tokens, hidden // 32, 32)
        amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2.0**-126)
        exponent = torch.ceil(torch.log2(amax / FP8_E4M3_MAX)).clamp(-127, 127)
        values = (blocks / torch.exp2(exponent)).to(torch.float8_e4m3fn)
        return values.view(num_tokens, hidden), (exponent.squeeze(-1) + 127).to(
            torch.uint8
        )
    # NVFP4 with a unit global scale: E4M3 block scales over 16 elements.
    blocks = x.float().view(num_tokens, hidden // 16, 16)
    scale = (blocks.abs().amax(-1, keepdim=True) / FP4_E2M1_MAX).to(torch.float8_e4m3fn)
    scaled = blocks / scale.float().clamp(min=1e-12)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = torch.bucketize(scaled.abs().clamp(max=FP4_E2M1_MAX), midpoints)
    codes = (magnitude.to(torch.uint8) | ((scaled < 0).to(torch.uint8) << 3)).view(
        num_tokens, hidden
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scale.view(num_tokens, hidden // 16).view(torch.uint8)


def _dequantize(
    payload: torch.Tensor, sf: Optional[torch.Tensor], mode: str
) -> torch.Tensor:
    if mode == "bf16":
        return payload.float()
    if mode == "blockwise_fp8":
        values = payload.view(torch.float8_e4m3fn).float()
        blocks = values.reshape(values.shape[0], values.shape[1] // 128, 128)
        return (blocks * sf.view(torch.float32)[..., None]).flatten(1)
    if mode == "mxfp8":
        values = payload.view(torch.float8_e4m3fn).float()
        exponents = sf.view(torch.uint8).int() - 127
        return torch.ldexp(
            values.reshape(values.shape[0], values.shape[1] // 32, 32),
            exponents[..., None].float(),
        ).flatten(1)
    packed = payload.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
    # Decode E2M1 on the device, without a host lookup table during capture.
    mantissa = (codes & 1).float()
    exponent = (codes >> 1) & 3
    magnitude = torch.where(
        exponent == 0,
        mantissa * 0.5,
        torch.ldexp(1.0 + mantissa * 0.5, (exponent - 1).float()),
    )
    values = torch.where((codes & 8) == 0, magnitude, -magnitude)
    scales = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * scales


def _make_inputs(case: Case, rank: int, round_index: int) -> tuple[torch.Tensor, ...]:
    model = case.model
    count = case.rounds[round_index].tokens[rank]
    rows = max(count, 1)
    generator = torch.Generator(device="cuda").manual_seed(
        1234 + rank * 97 + round_index
    )
    # Positive values with per-row and per-block scales; the references
    # dequantize the transmitted bytes, so the quantizers need not be exact.
    x = torch.randint(
        1, 17, (rows, model.hidden_size), generator=generator, device="cuda"
    )
    row_scale = 2.0 ** ((torch.arange(rows, device="cuda") + round_index) % 3)
    block = 128 if model.dispatch_dtype == "blockwise_fp8" else 32
    block_scale = 2.0 ** ((torch.arange(model.hidden_size, device="cuda") // block) % 3)
    x = (x.float() / 16 * row_scale[:, None] * block_scale[None, :]).to(torch.bfloat16)
    payload, sf = _quantize(x, model.dispatch_dtype)
    payload = payload[:count]
    sf = None if sf is None else sf[:count]

    spec = case.rounds[round_index]
    slots = spec.routing.make_expert_ids(model, spec.tokens, rank, round_index)
    token = torch.arange(count, device="cuda", dtype=torch.int64)[:, None]
    choice = torch.arange(model.top_k, device="cuda", dtype=torch.int64)[None, :]
    # The first routing weight identifies the source token independently of
    # the kernels' send slots and receive counters.
    identity = rank * case.max_tokens_per_rank + token + 1
    weights = (identity * (choice + 1)).float() / (
        case.max_tokens_per_rank * case.ep_size * 32
    )
    return payload, sf, slots, weights


def _expert_output(
    payload: torch.Tensor,
    sf: Optional[torch.Tensor],
    slots: torch.Tensor,
    weights: torch.Tensor,
    case: Case,
    rank: int,
) -> torch.Tensor:
    start, end = _expert_bounds(case.model.num_experts, case.ep_size, rank)
    owned = (slots >= start) & (slots < end)
    gain = torch.where(owned, weights * (1.0 + (slots % 7).float() / 8), 0).sum(-1)
    values = _dequantize(payload, sf, case.model.dispatch_dtype)
    # Padding rows carry invalid expert ids and unspecified payload bytes.
    values = torch.where(owned.any(dim=-1)[:, None], values, 0)
    return (values * gain[:, None]).to(torch.bfloat16)


def _cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    # FP8 travels as raw bytes; the references reinterpret them.
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8)
    return tensor.cpu()


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


def _transport(cft, lamport) -> str:
    return "lamport" if lamport else "cft" if cft else "fence"


class _RecordingModule:
    """Forwards to the kernel module and records each launch's transport."""

    def __init__(self, module, launches: list) -> None:
        self._module = module
        self._launches = launches

    def __getattr__(self, name):
        return getattr(self._module, name)

    def moe_a2a_dispatch(self, *args):
        self._launches.append(("dispatch", _transport(args[10], args[13])))
        return self._module.moe_a2a_dispatch(*args)

    def moe_a2a_combine(self, *args):
        self._launches.append(("combine", _transport(args[11], args[17])))
        return self._module.moe_a2a_combine(*args)


def _run_case(case: Case, rank: int) -> dict:
    import torch.distributed as dist

    from flashinfer.fused_moe.api import QuantFormat
    from flashinfer.moe_ep import (
        BootstrapConfig,
        CakeAlltoAll,
        CakeAlltoAllConfig,
        MoEEpCommParams,
        NVLinkOneSidedAlltoAll,
        NVLinkOneSidedConfig,
        create_communication,
    )

    model = case.model
    one_sided = case.backend == "nvlink_one_sided"
    backend_cls = NVLinkOneSidedAlltoAll if one_sided else CakeAlltoAll
    if not backend_cls.is_platform_supported():
        local = f"{case.backend} is not supported on this GPU"
    else:
        local = None
    reasons = [None] * case.ep_size
    dist.all_gather_object(reasons, local)
    if any(reasons):
        return {"skip": next(r for r in reasons if r)}

    dispatch_format = _DISPATCH_FORMAT[model.dispatch_dtype]
    params = MoEEpCommParams(
        num_experts=model.num_experts,
        top_k=model.top_k,
        max_tokens_per_rank=case.max_tokens_per_rank,
        hidden_size=model.hidden_size,
        dtype=torch.bfloat16,
        dispatch_format=None
        if dispatch_format is None
        else getattr(QuantFormat, dispatch_format),
    )
    options = dict(
        use_low_precision_combine=case.fp8_combine,
        eplb_stats_num_experts=model.num_experts // 2 if case.eplb else 0,
        enable_pdl=case.pdl,
    )
    if one_sided:
        config = NVLinkOneSidedConfig(
            cft={"auto": None, "fence": False, "cft": True, "lamport": False}[
                case.mode
            ],
            # Lamport against the fence transport: test_nvlink_one_sided_lamport.py.
            lamport=case.mode == "lamport",
            timeout_sec=30,
            **options,
        )
    else:
        assert case.mode == "auto", "Cake has no CFT transport to select"
        config = CakeAlltoAllConfig(**options)
    comm = create_communication(
        BootstrapConfig(world_size=case.ep_size, rank=rank), params, config
    )
    try:
        cft_enabled = one_sided and comm._cft_enabled
        if case.mode == "cft" and not cft_enabled:
            return {"skip": "CFT counted writes are unavailable on this platform"}
        if case.mode == "lamport" and not comm.lamport_enabled:
            return {"skip": "the Lamport transport is unavailable here"}
        launches: list = []
        combine_offsets: list = []
        if one_sided:
            comm._module = _RecordingModule(comm._module, launches)
        inputs = [_make_inputs(case, rank, i) for i in range(len(case.rounds))]
        stats = None
        if case.eplb:
            stats = (
                torch.arange(model.num_experts // 2, dtype=torch.int32, device="cuda")
                + rank * 1000
            )

        def sequence() -> list[dict]:
            outputs = []
            for spec, (payload, sf, slots, weights) in zip(
                case.rounds, inputs, strict=True
            ):
                maximum = max(spec.tokens)
                recv = comm.dispatch(
                    payload,
                    slots,
                    weights,
                    hidden_states_scale=sf,
                    runtime_max_tokens_per_rank=maximum,
                    eplb_local_stats=stats,
                )
                combine_offsets.append(comm._round["combine_offset"])
                eplb = recv.eplb_gathered_stats.clone() if case.eplb else None
                if rank == spec.delay_rank:
                    torch.cuda._sleep(200_000)
                received = (
                    recv.hidden_states,
                    recv.hidden_states_scale,
                    recv.topk_ids,
                    recv.topk_weights,
                )
                # Snapshots only for single-round cases: even device copies can
                # perturb the overlap the round sequences exercise.
                snapshot = (
                    tuple(None if t is None else t.clone() for t in received)
                    if len(case.rounds) == 1
                    else None
                )
                expert_out = _expert_output(*received, case, rank)
                if case.payload_in_workspace:
                    buffer = comm.get_combine_input_buffer(torch.bfloat16)
                    buffer.copy_(expert_out)
                    expert_out = buffer
                combined = comm.combine(expert_out)
                outputs.append(
                    {"combined": combined.clone(), "dispatch": snapshot, "eplb": eplb}
                )
            return outputs

        dist.barrier()
        if case.graph:
            sequence()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = sequence()
            # Keep the captured launches only; the warm-up ran the same ones.
            del launches[: -2 * len(case.rounds)]
            for _ in range(3):
                graph.replay()
        else:
            outputs = sequence()
        torch.cuda.synchronize()

        # Region boundaries stay fixed across token counts and transports.
        assert len(set(combine_offsets)) == 1, combine_offsets
        result = {
            "rank": rank,
            "inputs": [tuple(_cpu(t) for t in tensors) for tensors in inputs],
            "outputs": [
                {
                    "combined": _cpu(o["combined"]),
                    "dispatch": None
                    if o["dispatch"] is None
                    else tuple(_cpu(t) for t in o["dispatch"]),
                    "eplb": _cpu(o["eplb"]),
                }
                for o in outputs
            ],
            "launches": launches,
            "cft_enabled": cft_enabled,
            "lamport_max_tokens": comm.lamport_max_tokens if one_sided else 0,
        }
        dist.barrier()
        return result
    finally:
        comm.destroy()


def _worker(rank: int, case: Case, out_dir: str) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        # File rendezvous: no TCP port to race for between back-to-back cases.
        init_method=f"file://{os.path.join(out_dir, 'rendezvous')}",
        rank=rank,
        world_size=case.ep_size,
        device_id=torch.device("cuda", rank),
    )
    try:
        torch.save(_run_case(case, rank), os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _run(case: Case) -> list[dict]:
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < case.ep_size:
        pytest.skip(f"requires {case.ep_size} GPUs")
    with tempfile.TemporaryDirectory() as out_dir:
        mp.spawn(
            _worker,
            args=(case, out_dir),
            nprocs=case.ep_size,
            join=True,
        )
        results = [
            torch.load(os.path.join(out_dir, f"rank{r}.pt"), weights_only=False)
            for r in range(case.ep_size)
        ]
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        assert len(skipped) == case.ep_size, "inconsistent support across ranks"
        pytest.skip(skipped[0])
    return results


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------


def _verify_dispatch(
    case: Case, results: list[dict], receiver: int, index: int, lamport: bool
) -> None:
    """Received rows, payload bytes, routing and padding against the inputs;
    a Lamport step's row of token ``t`` from a source is its row ``t``."""
    payload, sf, slots, weights = results[receiver]["outputs"][index]["dispatch"]
    maximum = max(case.rounds[index].tokens)
    payload = payload.view(case.ep_size, maximum, -1)
    sf = None if sf is None else sf.view(case.ep_size, maximum, -1)
    slots = slots.view(case.ep_size, maximum, -1)
    weights = weights.view(case.ep_size, maximum, -1)
    begin, end = _expert_bounds(case.model.num_experts, case.ep_size, receiver)
    for source in range(case.ep_size):
        orig, orig_sf, orig_slots, orig_weights = results[source]["inputs"][index]
        wanted = ((orig_slots >= begin) & (orig_slots < end)).any(dim=1)
        expected_ids = torch.where(wanted)[0]
        valid = (slots[source] >= 0).any(dim=1)
        observed_ids = (
            (weights[source, valid, 0] * (case.max_tokens_per_rank * case.ep_size * 32))
            .round()
            .long()
            - source * case.max_tokens_per_rank
            - 1
        )
        order = observed_ids.argsort()
        exact = dict(rtol=0, atol=0)
        torch.testing.assert_close(observed_ids[order], expected_ids, **exact)
        if lamport:
            torch.testing.assert_close(torch.where(valid)[0], observed_ids, **exact)
        torch.testing.assert_close(
            slots[source, ~valid], torch.full_like(slots[source, ~valid], -1)
        )
        torch.testing.assert_close(
            slots[source, valid][order], orig_slots[expected_ids], **exact
        )
        torch.testing.assert_close(
            weights[source, valid][order], orig_weights[expected_ids], **exact
        )
        torch.testing.assert_close(
            payload[source].view(torch.uint8)[valid][order],
            orig.view(torch.uint8)[expected_ids],
            **exact,
        )
        if sf is not None:
            torch.testing.assert_close(
                sf[source].view(torch.uint8)[valid][order],
                orig_sf.view(torch.uint8)[expected_ids],
                **exact,
            )


def _verify_combine(case: Case, results: list[dict], rank: int, index: int) -> None:
    """Combined output against the inputs, with per-rank BF16/FP8 rounding."""
    payload, sf, slots, weights = results[rank]["inputs"][index]
    spec = case.rounds[index]
    reference = torch.zeros(
        (spec.tokens[rank], case.model.hidden_size), dtype=torch.float32
    )
    for expert_rank in range(case.ep_size):
        contribution = _expert_output(payload, sf, slots, weights, case, expert_rank)
        if case.fp8_combine:
            contribution = (
                contribution.float()
                .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                .to(torch.float8_e4m3fn)
            )
        reference += contribution.float()
    output = results[rank]["outputs"][index]["combined"]
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output,
        reference.to(torch.bfloat16),
        rtol=0.016,
        atol=0.002,
        msg=lambda detail: (
            f"rank={rank}, round={index}, tokens={spec.tokens}, mode={case.mode}, "
            f"graph={case.graph}\n{detail}"
        ),
    )


def _assert_results(case: Case, results: list[dict]) -> None:
    for rank, result in enumerate(results):
        assert result["rank"] == rank
        assert len(result["outputs"]) == len(case.rounds)
        expected_launches = []
        for i, spec in enumerate(case.rounds):
            payload, sf, slots, weights = result["inputs"][i]
            lamport = (
                case.mode == "lamport"
                and not case.eplb
                and max(spec.tokens) <= result["lamport_max_tokens"]
            )
            _verify_combine(case, results, rank, i)
            if result["outputs"][i]["dispatch"] is not None:
                _verify_dispatch(case, results, rank, i, lamport)
            requested = (
                result["cft_enabled"]
                and case.mode != "fence"
                and (case.mode == "cft" or max(spec.tokens) <= CFT_MAX_TOKENS)
            )
            # Wire eligibility, computed independently of the backend.
            rows = (
                (payload, slots, weights)
                if sf is None
                else (payload, sf, slots, weights)
            )
            dispatch_cft = requested and all(
                t.shape[1] * t.element_size() % CFT_ALIGNMENT_BYTES == 0 for t in rows
            )
            combine_cft = (
                requested
                and case.model.hidden_size
                * (1 if case.fp8_combine else 2)
                % CFT_ALIGNMENT_BYTES
                == 0
            )
            expected_launches += [
                ("dispatch", _transport(dispatch_cft, lamport)),
                ("combine", _transport(combine_cft, lamport)),
            ]
            if case.eplb:
                expected = torch.stack(
                    [
                        torch.arange(case.model.num_experts // 2, dtype=torch.int32)
                        + r * 1000
                        for r in range(case.ep_size)
                    ]
                )
                torch.testing.assert_close(
                    result["outputs"][i]["eplb"], expected, rtol=0, atol=0
                )
        if case.backend == "nvlink_one_sided":
            assert result["launches"] == expected_launches


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------

# A four-rank communicator reused across changing token counts and peer
# dependencies, including zero-token and delayed ranks. 128/129 tokens straddle
# the automatic CFT threshold.
_ROUND_SEQUENCE = (
    Round((3, 0, 2, 1), Routing.LOCAL, 0),
    Round((1, 129, 0, 5), Routing.SPREAD, 1),
    Round((0, 5, 3, 1), Routing.HOTSPOT, 2),
    Round((128, 3, 1, 0), Routing.SPREAD, 3),
    Round((5, 2, 0, 3), Routing.LOCAL, 0),
    Round((1, 0, 4, 2), Routing.SPREAD, 1),
    # LOCAL routing gives rank 0 no data dependency on rank 1, yet rank 0's
    # combine must still wait until rank 1 has consumed its dispatched inputs;
    # otherwise rank 0's next dispatch would overwrite them.
    Round((1, 128, 0, 0), Routing.LOCAL, 1),
    Round((129, 1, 0, 0), Routing.SPREAD, 0),
)

_MODEL_EP_SIZES = (
    (MODELS["gpt_oss"], 4),
    (MODELS["deepseek_v3"], 8),
    (MODELS["deepseek_r1_nvfp4"], 8),
    (MODELS["qwen3_5_397b_a17b"], 8),
    (MODELS["deepseek_v4_flash"], 4),
    (MODELS["deepseek_v4_pro"], 8),
    (MODELS["kimi_k3"], 8),
    (MODELS["deepseek_v41_flash_dspark"], 4),
)

CASES = [
    # Model shapes, equal token counts on every rank.
    *[
        pytest.param(
            Case(
                model=model,
                rounds=(Round((num_tokens,) * ep_size, Routing.SPREAD, -1),),
            ),
            id=f"model-{model.name}-ep{ep_size}-tokens{num_tokens}",
        )
        for model, ep_size in _MODEL_EP_SIZES
        for num_tokens in (1, 128, 1024)
    ],
    # The Lamport transport at the model shapes it carries (hidden_size a
    # multiple of 256), in steps within its default token limit.
    *[
        pytest.param(
            Case(
                model=model,
                rounds=(Round((num_tokens,) * ep_size, Routing.SPREAD, -1),),
                mode="lamport",
            ),
            id=f"model-{model.name}-ep{ep_size}-tokens{num_tokens}-lamport",
        )
        for model, ep_size in _MODEL_EP_SIZES
        if model.hidden_size % 256 == 0
        for num_tokens in (1, 128)
    ],
    # BF16/FP8 combine with external or workspace-resident expert output (the
    # Lamport transport combines BF16 only).
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=(Round((9, 5), Routing.SPREAD, -1),),
                mode=mode,
                payload_in_workspace=payload_in_workspace,
                fp8_combine=fp8_combine,
            ),
            id=f"combine-{'fp8' if fp8_combine else 'bf16'}-"
            f"{'workspace' if payload_in_workspace else 'external'}"
            f"{'-lamport' if mode == 'lamport' else ''}",
        )
        for mode, fp8_combine in (("auto", False), ("auto", True), ("lamport", False))
        for payload_in_workspace in (False, True)
    ],
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=_ROUND_SEQUENCE,
                mode=mode,
                payload_in_workspace=True,
                graph=graph,
                pdl=pdl,
            ),
            id=f"round-sequence-ep4-{mode}-{'graph' if graph else 'eager'}"
            f"{'' if pdl else '-nopdl'}",
        )
        for mode, graph, pdl in (
            ("auto", False, True),
            ("auto", False, False),
            ("auto", True, True),
            ("cft", True, True),
            ("fence", True, True),
            ("lamport", False, True),
            ("lamport", False, False),
            ("lamport", True, True),
        )
    ],
    # Rank-distinct EPLB statistics, including a rank with no input tokens.
    pytest.param(
        Case(
            model=MODELS["deepseek_v3"],
            rounds=(Round((9, 0, 3, 1), Routing.SPREAD, -1),),
            eplb=True,
        ),
        id="eplb-statistics",
    ),
]


CASES += [
    # Cake: the round sequence (the combine region must not move with the token
    # count), both combine wire formats with workspace input, EPLB and models.
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=_ROUND_SEQUENCE,
                payload_in_workspace=True,
                graph=graph,
                backend="cake",
            ),
            id=f"cake-round-sequence-ep4-{'graph' if graph else 'eager'}",
        )
        for graph in (False, True)
    ],
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=(Round((9, 5), Routing.SPREAD, -1),),
                payload_in_workspace=True,
                fp8_combine=fp8_combine,
                backend="cake",
            ),
            id=f"cake-combine-{'fp8' if fp8_combine else 'bf16'}-workspace",
        )
        for fp8_combine in (False, True)
    ],
    pytest.param(
        Case(
            model=MODELS["deepseek_v3"],
            rounds=(Round((9, 0, 3, 1), Routing.SPREAD, -1),),
            eplb=True,
            backend="cake",
        ),
        id="cake-eplb-statistics",
    ),
    *[
        pytest.param(
            Case(
                model=MODELS[name],
                rounds=(Round((128,) * 8, Routing.SPREAD, -1),),
                backend="cake",
            ),
            id=f"cake-model-{name}-ep8-tokens128",
        )
        for name in ("deepseek_r1_nvfp4", "kimi_k3", "qwen3_5_397b_a17b")
    ],
]


@pytest.mark.parametrize("case", CASES)
def test_nvlink_one_sided(case: Case) -> None:
    _assert_results(case, _run(case))


# ----------------------------------------------------------------------------
# Workspace layout
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("cft", [False, True])
def test_workspace_layout(cft: bool) -> None:
    """Regions are disjoint, aligned and fixed by the allocation-time capacities."""
    from flashinfer.moe_ep.backends.split.comm.nvlink_one_sided.kernels import (
        get_nvlink_one_sided_alltoall_module,
        layout_constants,
    )

    index = layout_constants()
    ep_size, top_k, capacity, hidden = 8, 6, 128, 7168
    dispatch_bytes = ep_size * capacity * (hidden + 2 * top_k * 4)
    combine_input_bytes = ep_size * capacity * hidden * 2
    combine_recv_bytes = ep_size * capacity * hidden if cft else 0
    layout = torch.empty(index.NUM_METAINFO_FIELDS, dtype=torch.int64)
    get_nvlink_one_sided_alltoall_module().moe_a2a_get_workspace_layout(
        ep_size,
        capacity,
        top_k,
        dispatch_bytes,
        combine_input_bytes,
        combine_recv_bytes,
        0,
        cft,
        layout,
    )

    def field(name: str) -> int:
        return int(layout[getattr(index, name)])

    alignment = index.WORKSPACE_ALIGNMENT
    dispatch_start = field("DISPATCH_PAYLOAD_OFFSET_INDEX")
    dispatch_end = dispatch_start + field("DISPATCH_PAYLOAD_SIZE_INDEX")
    combine_start = field("COMBINE_INPUT_OFFSET_INDEX")
    combine_end = combine_start + field("COMBINE_INPUT_SIZE_INDEX")
    total = field("WORKSPACE_SIZE_INDEX")
    assert field("DISPATCH_PAYLOAD_SIZE_INDEX") == dispatch_bytes
    assert field("COMBINE_INPUT_SIZE_INDEX") == combine_input_bytes
    assert field("TOPK_TARGET_INDICES_OFFSET_INDEX") < dispatch_start
    assert (
        dispatch_end <= field("COMBINE_COMPLETION_FLAGS_OFFSET_INDEX") < combine_start
    )
    assert dispatch_start % alignment == combine_start % alignment == 0
    # Routing metadata reserves the configured top_k, not the largest supported one.
    routes = field("TOPK_TARGET_INDICES_OFFSET_INDEX") - field(
        "TOPK_TARGET_RANKS_OFFSET_INDEX"
    )
    assert capacity * top_k * 4 <= routes < capacity * (top_k + 1) * 4 + alignment
    if cft:
        recv_start = field("COMBINE_RECV_OFFSET_INDEX")
        recv_bytes = field("COMBINE_RECV_SIZE_INDEX")
        assert combine_end <= recv_start and recv_start % alignment == 0
        assert recv_bytes == combine_recv_bytes
        assert recv_start + recv_bytes <= total
        assert (
            dispatch_end
            <= field("COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX")
            < combine_start
        )
    else:
        assert field("COMBINE_RECV_SIZE_INDEX") == 0
        assert combine_end <= total
