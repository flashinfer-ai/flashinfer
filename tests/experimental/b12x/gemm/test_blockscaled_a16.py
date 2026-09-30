"""Prepared shared-storage A16 correctness, dispatch, and graph contracts."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import pytest
import torch
import cutlass.cute as cute

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16, _quantize
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.device import detect_device
from b12x.gemm.blockscaled import _preparation
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.testing.reference.helpers import require_b12x


@contextmanager
def prepared_execution(
    source,
    weight,
    *,
    activation_mode,
    activation_global_scale=None,
    out=None,
    workspace=None,
    expected_m=None,
    config=None,
    name="blockscaled",
):
    query = blockscaled.query_from_call(
        source,
        weight,
        activation_mode=activation_mode,
        activation_global_scale=activation_global_scale,
        out=out,
        workspace=workspace,
        expected_m=expected_m,
    )
    declaration = blockscaled.plan(query, override=config)
    values, scales, global_scale, _ = _a16._weight_parts(weight)
    request = declaration.request(
        name=name,
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                source,
                values,
                scales,
                global_scale,
                activation_scale=activation_global_scale,
                out=out,
                workspace=workspace,
            )
        ),
    )
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        yield query, declaration


def make_workspace(source, weight, *, activation_mode, out=None, config=None,
                   activation_global_scale=None, expected_m=None):
    query = blockscaled.query_from_call(
        source, weight, activation_mode=activation_mode, out=out,
        activation_global_scale=activation_global_scale, expected_m=expected_m,
    )
    query = replace(query, workspace_form="provided")
    if config is None:
        config = _preparation.TUNING.default_config(query, detect_device(source.device).identity)
    return torch.empty(
        _preparation._workspace_bytes(query, config), device=source.device, dtype=torch.uint8
    )




@pytest.mark.parametrize("n,k,group", [(37, 80, 16), (129, 160, 32)])
def test_scale_storage_views_are_zero_copy(n, k, group):
    from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
    physical = torch.empty((n + 127) // 128 * 128, (k // group + 3) // 4 * 4, dtype=torch.uint8)
    view = (as_grouped_scale_view if group == 16 else as_grouped_scale_view_mx)(physical[None], n, k)
    flat = _a16.scale_storage(view, n, k, group)
    assert flat.data_ptr() == physical.data_ptr()
    with pytest.raises(ValueError, match="contiguous"):
        _a16.scale_storage(physical.T, n, k, group)


def make_weight(recipe, n, k, device="cuda"):
    if recipe == "nvfp4":
        codes = torch.arange(n * k, device=device).reshape(n, k) % 16
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales = (torch.rand(n, k // 16, device=device) * 2 + 0.0625).to(torch.float8_e4m3fn)
        storage = swizzle_block_scale(scales)
        global_scale = torch.tensor([0.125], dtype=torch.float32, device=device)
        weight = blockscaled.pack_weight(packed, storage, recipe="nvfp4", global_scale=global_scale)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=device)
        local = lut[codes] * scales.float().repeat_interleave(16, 1)
        return weight, local.to(torch.bfloat16).float() * global_scale, storage
    values = (torch.randn(n, k, device=device) * 2).to(torch.float8_e4m3fn)
    exponent = torch.randint(120, 133, (n, k // 32), device=device, dtype=torch.uint8)
    weight = blockscaled.pack_weight(values, exponent)
    local = values.float() * torch.exp2(exponent.float() - 127).repeat_interleave(32, 1)
    return weight, local.to(torch.bfloat16).float(), weight.weight.scale_mma


def assert_close(actual, expected):
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    assert float(error / denominator) < 0.004
    torch.testing.assert_close(actual.float(), expected.float(), atol=float(expected.abs().max()) * 0.008 + 1e-6, rtol=0.008)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("mode", ["auto", "a16", "quantized"])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_aot_functionalization_preserves_precision_and_output(recipe, mode, use_out, backend):
    require_b12x()
    import torch._inductor.config as inductor_config
    torch._dynamo.reset()
    weight, _, storage = make_weight(recipe, 128, 256)
    saved_scales = storage.view(torch.uint8).clone()
    options = {"activation_global_scale": torch.tensor([128.], device="cuda")} if recipe == "nvfp4" else {}

    def run(source, output, workspace, plan):
        return blockscaled.mm(
            source, weight, out=output if use_out else None,
            workspace=workspace, plan=plan, **options,
        )

    with inductor_config.patch(enable_auto_functionalized_v2=False):
        compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
        for m in (2, 8):
            source = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            output = torch.full((m, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
            workspace = make_workspace(
                source, weight, activation_mode=mode, out=output, expected_m=m, **options,
            ) if use_out else None
            with prepared_execution(
                source, weight, activation_mode=mode, out=output if use_out else None,
                workspace=workspace, expected_m=m, **options,
            ) as (_, plan):
                expected = run(source, output, workspace, plan).clone()
                actual = compiled(source, output, workspace, plan)
                if use_out:
                    assert actual is output
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(storage.view(torch.uint8), saved_scales, atol=0, rtol=0)
    torch._dynamo.reset()


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("allocated", [False, True])
@pytest.mark.parametrize("functionalize_v2", [False, True])
def test_a16_aot_compile_functionalizes_workspace(recipe, allocated, functionalize_v2):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, 128, 256)
    source = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16) if allocated else None
    config = BlockscaledConfig(mode="a16", tile_n=64, tile_k=128, split_k=2)
    scratch = make_workspace(
        source, weight, activation_mode="a16", out=out, config=config,
    )
    with prepared_execution(
        source, weight, activation_mode="a16", out=out, workspace=scratch, config=config,
    ) as (_, plan):
        def project(x):
            return blockscaled.mm(x, weight, out=out, workspace=scratch, plan=plan)
        compiled = torch.compile(project, fullgraph=True, options={
            "enable_auto_functionalized_v2": functionalize_v2,
        }).aot_compile(((source,), {}))
        for _ in range(2):
            source.normal_()
            assert_close(compiled(source), source.float() @ decoded.T)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("m,n,k", [(1, 40, 128), (4, 128, 256), (8, 136, 256), (16, 64, 128), (19, 40, 160), (33, 4096, 128)])
@pytest.mark.parametrize("split", [1, 2, 4, 8])
def test_a16_reference(recipe, m, n, k, split):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, n, k)
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    output = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    config = BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=split)
    provisional = blockscaled.query_from_call(
        source, weight, activation_mode="a16", out=output, expected_m=m
    )
    scratch = torch.full(
        (_preparation._workspace_bytes(provisional, config),),
        255,
        device="cuda",
        dtype=torch.uint8,
    )
    with prepared_execution(
        source,
        weight,
        activation_mode="a16",
        out=output,
        workspace=scratch,
        expected_m=m,
        config=config,
        name="a16-reference",
    ) as (_, plan):
        actual = blockscaled.mm(
            source, weight, out=output, workspace=scratch, plan=plan
        )
        assert actual.data_ptr() == output.data_ptr()
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        assert_close(actual, source.float() @ decoded.T)


@pytest.mark.parametrize("kind", ["multiplier", "reciprocal"])
def test_w4a16_raw_scale_identity_and_rounding(kind):
    require_b12x()
    n, k = 136, 96
    weight, _, storage = make_weight("nvfp4", n, k)
    original = storage.view(torch.uint8).clone()
    source = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    global_scale = weight.global_scale if kind == "multiplier" else weight.global_scale.reciprocal()
    raw_weight = blockscaled.pack_weight(
        weight.values, storage, recipe="nvfp4",
        global_scale=global_scale, global_scale_kind=kind,
    )
    with prepared_execution(source, weight, activation_mode="a16") as (_, plan):
        expected = blockscaled.mm(source, weight, plan=plan)
    with prepared_execution(source, raw_weight, activation_mode="a16") as (_, plan):
        actual = blockscaled.w4a16(
            source, weight.values, storage, global_scale,
            global_scale_kind=kind, plan=plan,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert _a16.scale_storage(weight.scale_mma, n, k, 16).data_ptr() == storage.data_ptr()
    torch.testing.assert_close(storage.view(torch.uint8), original, atol=0, rtol=0)


@pytest.mark.parametrize("recipe,tile_m", [("nvfp4", 16), ("nvfp4", 32), ("nvfp4", 64), ("mxfp8", 16)])
def test_a16_frozen_callable_and_graph_replay(recipe, tile_m, monkeypatch):
    require_b12x()
    weight, decoded, storage = make_weight(recipe, 136, 256)
    saved = storage.view(torch.uint8).clone()
    config = BlockscaledConfig(mode="a16", tile_m=tile_m, tile_n=64, tile_k=128, split_k=4)
    source = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(65, 136, device="cuda", dtype=torch.bfloat16)
    workspace = make_workspace(
        source, weight, activation_mode="a16", out=output, config=config
    )
    with ExitStack() as stack:
        plans = {}
        for m in (1, 2, 4, 8, 16, 19, 32, 33, 65):
            _, plans[m] = stack.enter_context(prepared_execution(
                source[:m], weight, activation_mode="a16", out=output[:m],
                workspace=workspace, config=config, expected_m=m,
            ))
        with kernel_resolution_guard("prepared exact-M graph"):
            for m in plans:
                result = blockscaled.mm(
                    source[:m], weight, out=output[:m], workspace=workspace,
                    plan=plans[m],
                )
                assert_close(result, source[:m].float() @ decoded.T)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                blockscaled.mm(source, weight, out=output, workspace=workspace,
                               plan=plans[65])
            pointers = (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
            for _ in range(3):
                source.normal_()
                workspace.fill_(255)
                output.fill_(float("nan"))
                graph.replay()
                assert_close(output, source.float() @ decoded.T)
            assert pointers == (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
            torch.testing.assert_close(storage.view(torch.uint8), saved, atol=0, rtol=0)
            with monkeypatch.context() as patch:
                patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected device allocation"))
                blockscaled.mm(source, weight, out=output, workspace=workspace,
                               plan=plans[65])
            graph.reset()


def test_quantized_nvfp4_activation_contract():
    require_b12x()
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
    from b12x._lib.dense_gemm import dense_gemm
    weight, _, _ = make_weight("nvfp4", 128, 256)
    source = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    ag = torch.tensor([256.], device="cuda")
    q, sf = quantize_grouped_nvfp4_torch(source[None], torch.tensor([8], device="cuda"), ag)
    expected = dense_gemm((q, sf),
                          (weight.values[:, :, None], weight.scale_mma),
                          ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                          c_dtype="bfloat16", sf_vec_size=16,
                          alpha=weight.global_scale / ag)[:, :, 0]
    config = BlockscaledConfig(mode="quantized")
    scratch = make_workspace(
        source, weight, activation_mode="quantized",
        activation_global_scale=ag, config=config,
    )
    scratch.fill_(255)
    with prepared_execution(
        source, weight, activation_mode="quantized", activation_global_scale=ag,
        workspace=scratch, config=config,
    ) as (_, plan):
        actual = blockscaled.mm(
            source, weight, activation_global_scale=ag,
            workspace=scratch, plan=plan,
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _, _, _, total = _a16._layout(8, 128, 256, True)
    sf_start, alpha_start, _, _ = _a16._layout(8, 128, 256, True)
    assert total <= scratch.numel()
    expected_bytes = _a16.scale_storage(sf, 8, 256, 16)
    torch.testing.assert_close(scratch[sf_start:sf_start + expected_bytes.numel()], expected_bytes, atol=0, rtol=0)
    torch.testing.assert_close(scratch[:8 * 128], q.flatten(), atol=0, rtol=0)
    torch.testing.assert_close(scratch[alpha_start:alpha_start + 4].view(torch.float32), weight.global_scale / ag)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_quantized_graph_and_dispatch(recipe):
    require_b12x()
    weight, _, _ = make_weight(recipe, 128, 256)
    source = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(9, 128, device="cuda", dtype=torch.bfloat16)
    options = {"activation_global_scale": torch.tensor([128.], device="cuda")} if recipe == "nvfp4" else {}
    workspace = make_workspace(source, weight, activation_mode="auto", out=output, **options)
    with ExitStack() as stack:
        _, quantized = stack.enter_context(prepared_execution(
            source, weight, activation_mode="quantized", **options,
        ))
        _, automatic = stack.enter_context(prepared_execution(
            source, weight, activation_mode="auto", **options,
        ))
        _, provided = stack.enter_context(prepared_execution(
            source, weight, activation_mode="auto", out=output, workspace=workspace, **options,
        ))
        expected = blockscaled.mm(source, weight, plan=quantized, **options)
        actual = blockscaled.mm(source, weight, plan=automatic, **options)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with kernel_resolution_guard("prepared quantized capture"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                blockscaled.mm(
                    source, weight, out=output, workspace=workspace, plan=provided, **options,
                )
            source.normal_()
            expected = blockscaled.mm(source, weight, plan=quantized, **options)
            workspace.fill_(255)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            graph.reset()


def test_invalid_mode_and_aliasing():
    require_b12x()
    weight, _, _ = make_weight("nvfp4", 128, 128)
    source = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        with prepared_execution(source, weight, activation_mode="typo"):
            pytest.fail("unknown activation precision was accepted")
    with pytest.raises(ValueError):
        with prepared_execution(source, weight, activation_mode="quantized"):
            pytest.fail("missing NVFP4 activation scale was accepted")
    with pytest.raises(ValueError):
        with prepared_execution(source, weight, activation_mode="a16", out=source):
            pytest.fail("source/output overlap was accepted")
    with pytest.raises(ValueError):
        with prepared_execution(
            source, weight, activation_mode="a16",
            workspace=torch.empty(1, dtype=torch.uint8, device="cuda"),
            config=BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=4),
        ):
            pytest.fail("insufficient caller workspace was accepted")


def test_mxfp8_prepared_functional_and_provided_forms_match():
    require_b12x()
    weight, _, _ = make_weight("mxfp8", 152, 384)
    source = torch.randn(7, 384, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(7, 152, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="quantized")
    functional_query = blockscaled.query_from_call(
        source, weight, activation_mode="quantized", expected_m=7
    )
    provisional_provided = blockscaled.query_from_call(
        source, weight, activation_mode="quantized", out=output, expected_m=7
    )
    workspace = torch.empty(
        _preparation._workspace_bytes(provisional_provided, config),
        device="cuda",
        dtype=torch.uint8,
    )
    provided_query = blockscaled.query_from_call(
        source,
        weight,
        activation_mode="quantized",
        out=output,
        workspace=workspace,
        expected_m=7,
    )
    values, scales, global_scale, _ = _a16._weight_parts(weight)
    functional = blockscaled.plan(functional_query, override=config)
    provided = blockscaled.plan(provided_query, override=config)
    requests = (
        functional.request(
            name="mxfp8-functional",
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, values, scales, global_scale)
            ),
        ),
        provided.request(
            name="mxfp8-provided",
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    source, values, scales, global_scale, out=output, workspace=workspace
                )
            ),
        ),
    )
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        with session.capture():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                functional_output = blockscaled.mm(
                    source, weight, plan=functional
                )
                blockscaled.mm(
                    source,
                    weight,
                    out=output,
                    workspace=workspace,
                    plan=provided,
                )
        graph.replay()
        torch.testing.assert_close(functional_output, output, atol=0, rtol=0)
        graph.reset()


def test_mxfp8_prefill_capacity_owned_and_provided_match():
    """Both scratch contracts retain short and long programs under frozen capture."""
    require_b12x()
    capacity, n, k = 2675, 6144, 2560
    weight, _, _ = make_weight("mxfp8", n, k)
    source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="quantized")
    workspace = make_workspace(source, weight, activation_mode="quantized",
                               out=output, config=config)
    with ExitStack() as stack:
        _, functional = stack.enter_context(prepared_execution(
            source, weight, activation_mode="quantized", config=config,
            name="mxfp8-capacity-functional",
        ))
        _, provided = stack.enter_context(prepared_execution(
            source, weight, activation_mode="quantized", config=config,
            out=output, workspace=workspace, name="mxfp8-capacity-provided",
        ))
        assert functional.prepared.state.short_dense is not None
        assert provided.prepared.state.short_dense is not None
        owned_bytes = functional.prepared.state.owned_nbytes
        assert provided.prepared.state.owned_nbytes == 0
        with kernel_resolution_guard("packed MXFP8 capacity scratch contracts"):
            for rows in (4, 2047, 2048, capacity):
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        actual = blockscaled.mm(source[:rows], weight, plan=functional)
                        blockscaled.mm(source[:rows], weight, out=output[:rows],
                                       workspace=workspace, plan=provided)
                    source.normal_()
                    workspace.fill_(255)
                    output.fill_(float("nan"))
                    actual.fill_(float("nan"))
                    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                    torch.testing.assert_close(actual, output[:rows], atol=0, rtol=0)
                    assert torch.isnan(output[rows:]).all()
                    assert functional.prepared.state.owned_nbytes == owned_bytes
                finally:
                    graph.reset()


@pytest.mark.parametrize("n,k", [(2560, 2560), (2560, 6144), (6144, 2560)])
def test_mxfp8_prefill_capacity_reuses_graph_program_for_shorter_rows(monkeypatch, n, k):
    """A capacity-tuned tile keeps live row masks and caller scratch intact."""
    from b12x.gemm._shared.wo_mxfp8 import (
        dequantize_mxfp8_rows_torch, quantize_mxfp8_rows_torch,
    )

    require_b12x()
    capacity = 6019
    weight, decoded, _ = make_weight("mxfp8", n, k)
    source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="quantized")
    workspace = make_workspace(
        source, weight, activation_mode="quantized", out=output, config=config,
    )
    with prepared_execution(
        source, weight, activation_mode="quantized", out=output,
        workspace=workspace, config=config,
    ) as (query, plan):
        assert query.expected_m is None
        state = plan.prepared.state
        assert state.short_dense is not None
        launched = []
        core_type = type(state.dense)
        original_run = core_type.run

        def record_core(core, *args, **kwargs):
            launched.append(core)
            return original_run(core, *args, **kwargs)

        monkeypatch.setattr(core_type, "run", record_core)
        with kernel_resolution_guard("capacity-tuned MXFP8 prefill"):
            for rows in (1, 4, 127, 128, 129, 2047, 2048, 2675, capacity):
                live_source, live_output = source[:rows], output[:rows]
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        blockscaled.mm(live_source, weight, out=live_output,
                                       workspace=workspace, plan=plan)
                    expected = state.short_dense if rows < 2048 else state.dense
                    assert launched[-1] is expected
                    source.normal_()
                    quantized = quantize_mxfp8_rows_torch(live_source)
                    reference = dequantize_mxfp8_rows_torch(
                        quantized.values, quantized.scale_rows,
                    ).float() @ decoded.T
                    workspace.fill_(255)
                    output.fill_(float("nan"))
                    pointers = (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
                    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                    assert pointers == (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
                    assert_close(live_output, reference)
                    assert torch.isnan(output[rows:]).all()
                finally:
                    graph.reset()




@pytest.mark.parametrize("fp4", [True, False])
def test_native_weight_pairs_exhaustive(fp4):
    """Every value byte and scale byte, including UE8M0 zero and NaNs."""
    require_b12x()
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.compiler import compile as compile_kernel
    from b12x._lib.intrinsics import nvfp4_pair_to_bf16x2_sm120, mxfp8_pair_to_bf16x2_sm120
    from b12x._lib.utils import make_ptr, current_cuda_stream

    class Decode:
        def __init__(self, fp4):
            self.fp4 = fp4

        @cute.jit
        def __call__(self, q: cute.Pointer, sf: cute.Pointer, out: cute.Pointer, stream):
            self.kernel(q, sf, out).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)

        @cute.kernel
        def kernel(self, q: cute.Pointer, sf: cute.Pointer, out: cute.Pointer):
            i = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
            if cutlass.const_expr(self.fp4):
                out[i] = nvfp4_pair_to_bf16x2_sm120(q[i], sf[i])
            else:
                out[i] = mxfp8_pair_to_bf16x2_sm120(q[i], sf[i])

    code = torch.arange(256, device="cuda", dtype=torch.int64).repeat(256)
    sf = torch.arange(256, device="cuda", dtype=torch.int64).repeat_interleave(256)
    if fp4:
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device="cuda")
        values = torch.stack((lut[code & 15], lut[code >> 4]), -1)
        factors = sf.to(torch.uint8).view(torch.float8_e4m3fn).double()
        packed = code
    else:
        values = torch.stack((code, 255 - code), -1).to(torch.uint8).view(torch.float8_e4m3fn).double()
        factors = torch.exp2(sf.double() - 127)
        factors[sf == 255] = float("nan")
        packed = code | ((255 - code) << 8)
    expected = (values * factors[:, None]).to(torch.bfloat16)
    packed, sf = packed.to(torch.uint32), sf.to(torch.uint32)
    out = torch.empty_like(expected)
    pointers = [make_ptr(cutlass.Uint32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                for tensor in (packed, sf, out)]
    fn = compile_kernel(Decode(fp4), *pointers, current_cuda_stream())
    fn(*pointers, current_cuda_stream())
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("k", [32, 96, 160])
def test_a16_short_scale_tile_tail(recipe, k):
    require_b12x()
    weight, decoded, _ = make_weight(recipe, 40, k)
    source = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="a16", tile_n=128, tile_k=128, split_k=4)
    with prepared_execution(
        source, weight, activation_mode="a16", config=config,
    ) as (_, plan):
        actual = blockscaled.mm(source, weight, plan=plan)
    assert_close(actual, source.float() @ decoded.T)


@pytest.mark.parametrize("tile_m", [16, 32, 64])
@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("k", [768, 800])
def test_nvfp4_k256_dynamic_graph_replay(tile_m, tile_n, split, k):
    require_b12x()
    from b12x.preparation import require_prepared

    weight, decoded, _ = make_weight("nvfp4", 136, k)
    source = torch.randn(65, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(65, 136, device="cuda", dtype=torch.bfloat16)
    config = BlockscaledConfig(mode="a16", tile_m=tile_m, tile_n=tile_n, tile_k=256, split_k=split)
    workspace = make_workspace(source, weight, activation_mode="a16", out=output, config=config)
    with prepared_execution(source, weight, activation_mode="a16", out=output,
                            workspace=workspace, config=config) as (_, plan):
        state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
        program = state.programs["gemm"]
        pointers = (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
        with kernel_resolution_guard("NVFP4 K256 capacity replay"):
            for m in (1, 3, 8, 17, 65):
                x, y = source[:m], output[:m]
                expected = x.float() @ decoded.T
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    blockscaled.mm(x, weight, out=y, workspace=workspace, plan=plan)
                x.neg_()
                y.fill_(float("nan"))
                workspace.fill_(255)
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert_close(y, -expected)
                assert state.programs["gemm"] is program
                assert pointers == (source.data_ptr(), output.data_ptr(), workspace.data_ptr())
                graph.reset()


def test_a16_rejects_tma_misalignment():
    require_b12x()
    weight, _, _ = make_weight("nvfp4", 37, 80)
    source = torch.randn(2, 80, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        with prepared_execution(source, weight, activation_mode="a16"):
            pytest.fail("misaligned A16 geometry was admitted")
