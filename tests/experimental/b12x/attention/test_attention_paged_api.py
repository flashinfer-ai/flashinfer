from __future__ import annotations

import torch
import cutlass.cute as cute
import pytest

import b12x._lib.compiler as cute_compiler
from b12x.attention.paged._forward import _get_cached_plane_tma_descs
from b12x._lib.compiler import KernelCompileSpec


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_public_compile_accepts_real_binding_without_running_attention(kv_dtype):
    from b12x.attention import paged
    from b12x.attention.paged.reference import paged_attention_reference
    from b12x.testing.reference.helpers import require_b12x
    from .test_attention_paged_forward import _make_workspace
    from .test_attention_paged_planner import _make_inputs

    require_b12x()
    q, k, v, pages, lengths, offsets = _make_inputs(
        q_seqlens=[6, 5], cache_seqlens=[64, 64], kv_dtype=kv_dtype,
    )
    workspace = _make_workspace(q, k, v, mode="extend")
    workspace.prepare(pages, lengths, offsets, disable_split_kv=True)
    output = torch.full_like(q, 123)
    descale = torch.ones(2, device=q.device) if kv_dtype == torch.float8_e4m3fn else None
    binding = workspace._scratch_plan.bind(
        scratch=workspace._scratch, q=q, k_cache=k, v_cache=v, output=output,
        page_table=pages, cache_seqlens=lengths, cu_seqlens_q=offsets,
        disable_split_kv=True,
        k_descale=descale, v_descale=descale,
    )
    paged.compile(binding=binding)
    torch.cuda.synchronize()
    assert bool((output == 123).all())
    actual, _ = paged.run(binding=binding)
    expected, _ = paged_attention_reference(
        q, k, v, pages, lengths, offsets, causal=True,
        k_descale=descale, v_descale=descale,
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)


class _WorkspaceStub:
    def __init__(self) -> None:
        self._live_plane_tma_desc_cache = {}


def _cache_key(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    plane_cols: int,
    tile_rows: int,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int, int]:
    return (
        int(k_cache.data_ptr()),
        int(v_cache.data_ptr()),
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        plane_cols,
        tile_rows,
    )


def test_plane_tma_descriptor_cache_keeps_distinct_layer_bindings() -> None:
    workspace = _WorkspaceStub()
    plane_cols = 64
    tile_rows = 32

    k_cache_1 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    v_cache_1 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    k_desc_1 = torch.empty((2, 16), dtype=torch.uint64)
    v_desc_1 = torch.empty((2, 16), dtype=torch.uint64)
    k_ptrs_1 = torch.empty((2,), dtype=torch.int64)
    v_ptrs_1 = torch.empty((2,), dtype=torch.int64)
    workspace._live_plane_tma_desc_cache[_cache_key(
        k_cache_1,
        v_cache_1,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    )] = (k_desc_1, v_desc_1, k_ptrs_1, v_ptrs_1)

    k_cache_2 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    v_cache_2 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    k_desc_2 = torch.empty((2, 16), dtype=torch.uint64)
    v_desc_2 = torch.empty((2, 16), dtype=torch.uint64)
    k_ptrs_2 = torch.empty((2,), dtype=torch.int64)
    v_ptrs_2 = torch.empty((2,), dtype=torch.int64)
    workspace._live_plane_tma_desc_cache[_cache_key(
        k_cache_2,
        v_cache_2,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    )] = (k_desc_2, v_desc_2, k_ptrs_2, v_ptrs_2)

    assert _get_cached_plane_tma_descs(
        workspace,
        k_cache=k_cache_1,
        v_cache=v_cache_1,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    ) == (k_desc_1, v_desc_1, k_ptrs_1, v_ptrs_1)
    assert _get_cached_plane_tma_descs(
        workspace,
        k_cache=k_cache_2,
        v_cache=v_cache_2,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    ) == (k_desc_2, v_desc_2, k_ptrs_2, v_ptrs_2)


def test_explicit_compile_spec_warms_compile_cache_during_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()

    class _Compiled:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *args):
            self.calls += 1
            assert args == ("arg",)

    def fake_compile(kernel, *args, **kwargs):
        kernel.compile_calls += 1
        assert args == ("compile-arg",)
        assert kwargs == {}
        return kernel.compiled

    monkeypatch.setattr(cute, "compile", fake_compile)

    class _Kernel:
        def __init__(self) -> None:
            self.compile_calls = 0
            self.compiled = _Compiled()

        def __call__(self, *args):
            assert args == ("arg",)
            return None

    kernel = _Kernel()
    compile_spec = KernelCompileSpec.from_fields(
        "test.paged.launch",
        1,
        ("shape", "shape-only"),
    )

    cute_compiler.launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=("compile-arg",),
        runtime_args=("arg",),
    )
    cute_compiler.launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=("different-compile-arg",),
        runtime_args=("arg",),
    )

    assert kernel.compile_calls == 1
    assert kernel.compiled.calls == 2
