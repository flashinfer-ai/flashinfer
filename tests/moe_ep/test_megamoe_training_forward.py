# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Training saves share rollout arithmetic across batches, plans and graphs."""

from dataclasses import replace
import os

import pytest

from ._megamoe_bi_helpers import assert_bits, assert_reference
from .test_a16_megamoe_batch_invariant import (
    _config as _a16_config,
    _decoded as _a16_decoded,
    _inputs as _a16_inputs,
)
from .test_mxfp4_megamoe_batch_invariant import _config as _mx4_config
from .test_mxfp4_megamoe_batch_invariant import _quantize as _mx4_quantize
from .test_mxfp8_megamoe_batch_invariant import _config as _mx8_config
from .test_mxfp8_megamoe_batch_invariant import _quantize as _mx8_quantize
from .test_nvfp4_megamoe_batch_invariant import (
    _device,
    _ieee_matmul as _ieee_precision_fixture,
    _layer,
    _plain_quantized_weights,
    _problem,
)

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("_ieee_matmul")]
_ieee_matmul = _ieee_precision_fixture
CASES = (
    ("bf16", "legacy", True),
    ("bf16", "legacy", False),
    ("nvfp4", "legacy", True),
    ("nvfp4", "standard", True),
    ("nvfp4", "standard", False),
    ("nvfp4", "genphase", True),
    ("mxfp4_mxfp8", "standard", True),
    ("mxfp4_mxfp8", "genphase", True),
    ("mxfp4", "standard", True),
    ("mxfp4", "genphase", True),
    ("mxfp8_e4m3", "legacy", True),
    ("mxfp8_e5m2", "legacy", True),
    ("mxfp8_e4m3", "standard", True),
    ("mxfp8_e4m3", "standard", False),
    ("mxfp8_e5m2", "standard", True),
    ("mxfp8_e4m3", "genphase", True),
    ("mxfp8_e5m2", "genphase", True),
    ("w4a16", "legacy", True),
    ("w4a16", "legacy", False),
    ("w8a16_e4m3", "legacy", True),
    ("w8a16_e5m2", "legacy", True),
)
HIDDEN = INTERMEDIATE = 512


def _setup(kind, provider, early, rank, world):
    from flashinfer import moe_ep

    # BF16/A16 use the shared legacy entry point on both architecture families.
    cc = _device("standard" if kind == "bf16" or "a16" in kind else provider)
    if kind == "mxfp4" and cc != (10, 7):
        pytest.skip("W4A4 MXFP4 requires Rubin")
    if "a16" in kind:
        x, scores, ids, weights = _a16_inputs(kind, rank, world, HIDDEN, INTERMEDIATE)
        config = replace(_a16_config(kind, early), intermediate_size=INTERMEDIATE)
        decoded_x, decoded_w = x.float(), _a16_decoded(weights, kind)[0].float()
    else:
        x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
        if kind == "bf16":
            config = moe_ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=3,
                batch_invariant=True,
                apply_routing_weights_before_fc2=early,
                knobs={},
            )
            decoded_x, decoded_w = x.float(), weights.w13.float()
        elif kind == "nvfp4":
            from .test_nvfp4_megamoe_batch_invariant import _config
            from .test_nvfp4_cutedsl_kernel_vs_reference import _dequant_nvfp4
            from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
                nvfp4_quantize_per_block_16,
            )

            config = _config(provider, early, HIDDEN, INTERMEDIATE)
            weights, _ = _plain_quantized_weights(weights, INTERMEDIATE)
            q, sf = nvfp4_quantize_per_block_16(x.float(), config.input_norm_const)
            decoded_x = _dequant_nvfp4(q, sf, logical_cols=HIDDEN)
            decoded_w = _dequant_nvfp4(
                weights.w13.reshape(-1, HIDDEN // 2),
                weights.w13_scale.reshape(-1, HIDDEN // 16),
                logical_cols=HIDDEN,
            ).reshape(-1, 2 * INTERMEDIATE, HIDDEN)
        else:
            is_mx4 = kind in ("mxfp4", "mxfp4_mxfp8")
            quantize = _mx4_quantize if is_mx4 else _mx8_quantize
            make_config = _mx4_config if is_mx4 else _mx8_config
            config = make_config(kind, provider, early, intermediate=INTERMEDIATE)
            q, sf = quantize(x, kind)
            w13, sf13 = quantize(weights.w13, kind, weight=True)
            w2, sf2 = quantize(weights.w2, kind, weight=True)
            weights = moe_ep.PrequantizedMoEWeights(
                w13=w13, w13_scale=sf13, w2=w2, w2_scale=sf2
            )

            def decode(data, scale, fp4):
                if fp4:
                    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
                        unpack_fp4_to_f32,
                    )

                    data = unpack_fp4_to_f32(data)
                return data.float() * scale.float().repeat_interleave(32, -1)

            decoded_x = decode(q, sf, kind == "mxfp4")
            decoded_w = decode(w13, sf13, is_mx4)
    ids[19].fill_(-1)
    scores[19].fill_(float("nan"))
    ids[20, 1] = -1
    scores[20, 1] = float("nan")
    return x, scores, ids, weights, config, decoded_x, decoded_w


def _saved(aux, selected, ids, config, decoded_x, decoded_w, rank, world):
    """Check actual saved values and associate native pool rows with source slots."""
    import torch

    active = (aux.expert_ids >= 0).nonzero(as_tuple=True)[0]
    experts = aux.expert_ids[active].long()
    packed = (
        aux.route_metadata["token_src_metadata"].view(torch.int64).flatten()[active]
    )
    if aux.route_metadata_format == "token16_slot8_rank8":
        tokens = packed.bitwise_and(0xFFFF)
        sources = (packed >> 24).bitwise_and(0xFF)
        slots = (packed >> 16).bitwise_and(0xFF)
    else:
        assert aux.route_metadata_format == "token32_slot16_rank16"
        tokens = packed.bitwise_and(0xFFFFFFFF)
        sources = (packed >> 48).bitwise_and(0xFFFF)
        slots = (packed >> 32).bitwise_and(0xFFFF)
    assert aux.fc1_preact.dtype is torch.bfloat16
    assert aux.fc1_preact.shape[1] == 2 * INTERMEDIATE
    assert torch.all((sources >= 0) & (sources < world))
    assert torch.all((tokens >= 0) & (tokens < selected.numel()))
    assert torch.all((slots >= 0) & (slots < ids.shape[-1]))
    assert torch.all((experts >= 0) & (experts < decoded_w.shape[0]))
    original_tokens = selected[tokens]
    assert torch.equal(
        ids[sources, original_tokens, slots], experts + rank * decoded_w.shape[0]
    )
    expected_count = (
        (ids[:, selected] >= rank * decoded_w.shape[0])
        & (ids[:, selected] < (rank + 1) * decoded_w.shape[0])
    ).sum()
    assert active.numel() == expected_count.item()
    keys = (sources * ids.shape[1] + original_tokens) * ids.shape[-1] + slots
    assert keys.unique().numel() == active.numel()
    actual = aux.fc1_preact[active]
    expected = torch.empty_like(actual, dtype=torch.float32)
    for expert in range(decoded_w.shape[0]):
        belongs = experts == expert
        operands = decoded_x[sources[belongs], original_tokens[belongs]]
        value = operands @ decoded_w[expert].T
        if getattr(config, "fc1_alpha", None) is not None:
            value = value * config.fc1_alpha[expert]
        expected[belongs] = value
    assert_reference(actual, expected.bfloat16(), 0.025, "saved preactivation oracle")
    canonical = torch.zeros(
        world * ids.shape[1] * ids.shape[-1],
        2 * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=actual.device,
    )
    canonical[keys] = actual
    return keys, canonical


def _run(kind, provider, early, rank, world, problem=None):
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import MoEEpTensors
    from ._megamoe_bi_helpers import all_gather_rows

    x, scores, ids, weights, config, decoded_x, decoded_w = (
        _setup(kind, provider, early, rank, world) if problem is None else problem
    )
    all_ids = all_gather_rows(ids).view(world, 257, -1) if world > 1 else ids[None]
    all_x = (
        all_gather_rows(decoded_x).view(world, 257, HIDDEN)
        if world > 1
        else decoded_x[None]
    )
    selected = torch.arange(65, device="cuda")
    training_config = replace(config, save_for_backward=True)
    with (
        _layer(config, weights, rank, world, 129, HIDDEN) as rollout,
        _layer(training_config, weights, rank, world, 129, HIDDEN) as training,
    ):
        anchor = rollout.forward(MoEEpTensors(x[:65], ids[:65], scores[:65])).clone()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor)
        actual = training.forward(MoEEpTensors(x[:65], ids[:65], scores[:65]))
        assert_bits(actual, anchor, "training vs rollout")
        assert (
            training.batch_invariant_fingerprint == rollout.batch_invariant_fingerprint
        )
        aux = training.get_training_outputs()
        _, saved = _saved(aux, selected, all_ids, config, all_x, decoded_w, rank, world)
        preact_address = aux.fc1_preact.data_ptr()
        knobs = (
            {
                "load_balance_mode": "atomic_counter",
                "token_back_mode": "reuse_dispatch_warps",
            }
            if provider == "legacy"
            else {}
            if provider == "genphase" and torch.cuda.get_device_capability() != (10, 7)
            else {"work_id_mode": "atomic_counter"}
        )
        alternate = training.create_workspace(65, knobs=knobs)
        actual = training.forward(
            MoEEpTensors(x[:65], ids[:65], scores[:65]), workspace=alternate
        )
        assert_bits(actual, anchor, "training alternate plan")
        alternate_aux = alternate.training_outputs
        assert alternate_aux.fc1_preact.data_ptr() != preact_address
        keys, current = _saved(
            alternate_aux, selected, all_ids, config, all_x, decoded_w, rank, world
        )
        assert_bits(current[keys], saved[keys], "saved preactivation alternate plan")
        for selected in (
            torch.arange(33, device="cuda").flip(0),
            torch.arange(1, device="cuda"),
        ):
            actual = training.forward(
                MoEEpTensors(x[selected], ids[selected], scores[selected])
            )
            assert_bits(actual, anchor[selected], "training batch and permutation")
            keys, current = _saved(
                training.get_training_outputs(),
                selected,
                all_ids,
                config,
                all_x,
                decoded_w,
                rank,
                world,
            )
            assert_bits(
                current[keys], saved[keys], "saved preactivation batch invariance"
            )

        selected = torch.arange(17, device="cuda")
        inputs = MoEEpTensors(
            x[selected].clone(),
            ids[selected].clone(),
            scores[selected].clone(),
            output=torch.empty_like(anchor[selected]),
        )
        training.forward(inputs)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                training.forward(inputs)
            for selected in (
                torch.arange(17, device="cuda").flip(0),
                torch.arange(17, 34, device="cuda"),
            ):
                inputs.hidden_states.copy_(x[selected])
                inputs.topk_ids.copy_(ids[selected])
                inputs.topk_weights.copy_(scores[selected])
                graph.replay()
                assert_bits(inputs.output, anchor[selected], "training graph replay")
                keys, current = _saved(
                    training.get_training_outputs(),
                    selected,
                    all_ids,
                    config,
                    all_x,
                    decoded_w,
                    rank,
                    world,
                )
                assert_bits(
                    current[keys], saved[keys], "saved preactivation graph replay"
                )
                assert (
                    training.get_training_outputs().fc1_preact.data_ptr()
                    == preact_address
                )
        finally:
            graph.reset()

        training.forward(MoEEpTensors(x[:0], ids[:0], scores[:0]))
        assert torch.all(training.get_training_outputs().expert_ids == -1)
        if world > 1:
            dist.barrier()


@pytest.mark.parametrize("kind,provider,early", CASES)
def test_training_forward_matches_rollout(kind, provider, early, monkeypatch):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank training forward")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    _run(kind, provider, early, 0, 1)


@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "kind,provider,early",
    (
        ("bf16", "legacy", True),
        ("nvfp4", "standard", True),
        ("mxfp8_e4m3", "standard", True),
        ("w4a16", "legacy", True),
    ),
)
def test_training_forward_matches_rollout_expert_parallel(
    kind, provider, early, monkeypatch
):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from ._megamoe_bi_helpers import fail_stop_distributed

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("requires torchrun with at least two ranks")
    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    moe_ep.ensure_moe_ep_cuda_device(bootstrap)
    problem = _setup(kind, provider, early, rank, world)
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(problem[4]).runtime_requirements(bootstrap)
    )
    try:
        _run(kind, provider, early, rank, world, problem)
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)
