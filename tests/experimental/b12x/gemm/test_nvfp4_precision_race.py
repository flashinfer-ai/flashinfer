"""Measured NVFP4 precision selection over the complete native candidate set."""

from dataclasses import replace

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16
from b12x.gemm.blockscaled._tuning import TUNING
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.device import detect_device
from b12x.testing.reference.helpers import require_b12x
from .test_blockscaled_a16 import make_weight, assert_close


def _quantized_source(source, activation_scale):
    from b12x._lib.intrinsics import fp4_quantize_values_torch

    rows, k = source.shape
    blocks = source.float().view(rows, k // 16, 16)
    scale = (
        (activation_scale * (blocks.abs().amax(-1, keepdim=True) / 6))
        .to(torch.float8_e4m3fn)
        .float()
    )
    effective = scale * activation_scale.reciprocal()
    inverse = torch.where(effective == 0, 0, effective.reciprocal())
    values = fp4_quantize_values_torch((blocks * inverse).clamp(-6, 6))
    return (values * effective).reshape(rows, k)


def test_auto_nvfp4_races_a16_and_a4_then_replays(tmp_path):
    require_b12x()
    torch.manual_seed(4196)
    source = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    weight, decoded, storage = make_weight("nvfp4", 128, 256)
    activation_scale = torch.tensor([128.0], device="cuda")
    output = torch.empty(8, 128, device="cuda", dtype=torch.bfloat16)
    query = blockscaled.query_from_call(
        source,
        weight,
        activation_mode="auto",
        activation_global_scale=activation_scale,
        out=output,
        expected_m=8,
    )
    query = replace(query, workspace_form="provided", workspace_nbytes=None)
    plan = blockscaled.plan(query)
    eligible = {
        config
        for _, config in TUNING.eligible_plan(
            query, detect_device(source.device).identity
        ).candidates
    }
    assert {config.mode for config in eligible} == {"a16", "quantized"}
    references = {
        "a16": source.float() @ decoded.T,
        "quantized": _quantized_source(source, activation_scale) @ decoded.T,
    }
    values, scales, global_scale, _ = _a16._weight_parts(weight)
    original_scales = storage.clone()
    seen = set()

    def call(state):
        trial_source = source.clone()
        trial_scale = activation_scale.clone()
        trial_output = torch.empty_like(output)
        workspace = torch.empty(
            state.required_workspace, device=source.device, dtype=torch.uint8
        )

        def run():
            return state.run(
                trial_source,
                values,
                scales,
                global_scale,
                activation_scale=trial_scale,
                out=trial_output,
                workspace=workspace,
            )

        run()
        assert torch.isfinite(trial_output).all() and trial_output.abs().sum() > 0
        assert_close(trial_output, references[state.config.mode])
        seen.add(state.config)

        def produce():
            trial_source.copy_(source)
            trial_scale.copy_(activation_scale)

        return PreparedCall(
            run=run,
            output=trial_output,
            produce=produce,
            owners=(trial_source, trial_scale, workspace),
        )

    with PreparationSession(
        device=source.device,
        autotune=True,
        compile_workers=2,
        cache_dir=tmp_path,
        race_batch=8,
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="nvfp4-precision", prepare_call=call, benchmark_call=call
                ),
            )
        )
        assert seen == eligible
        assert plan.selection.source == "tuned"
        print(
            f"NVFP4 M8 N128 K256: {len(seen)} candidates; selected {plan.selection.config}",
            flush=True,
        )
        workspace = torch.empty(
            plan.prepared.state.required_workspace,
            device=source.device,
            dtype=torch.uint8,
        )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(
                source,
                weight,
                activation_global_scale=activation_scale,
                out=output,
                workspace=workspace,
                plan=plan,
            )
        pointers = source.data_ptr(), output.data_ptr(), workspace.data_ptr()
        for _ in range(3):
            source.neg_()
            activation_scale.mul_(0.5)
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert pointers == (
                source.data_ptr(),
                output.data_ptr(),
                workspace.data_ptr(),
            )
            actual_source = (
                source.float()
                if plan.selection.config.mode == "a16"
                else _quantized_source(source, activation_scale)
            )
            assert_close(output, actual_source @ decoded.T)
        torch.testing.assert_close(
            storage.view(torch.uint8), original_scales.view(torch.uint8), rtol=0, atol=0
        )
        graph.reset()
