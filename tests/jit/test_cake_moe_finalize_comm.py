"""Source-closure tests for the Cake MoE finalize JIT package."""

from pathlib import Path
from typing import Any, Literal, get_type_hints

from flashinfer.comm import trtllm_ar
from flashinfer.jit import cake_moe_finalize_comm as cake_finalize


def test_cake_moe_finalize_manifest_exposes_exact_route_matrix() -> None:
    specs = cake_finalize.get_cake_moe_finalize_module_specs()

    assert len(specs) == 48
    assert {
        (
            spec.arch,
            spec.dtype,
            spec.world_size,
            spec.output_profile,
            spec.use_pdl,
        )
        for spec in specs
    } == {
        (arch, dtype, world_size, output_profile, use_pdl)
        for arch in ("sm_100a", "sm_103a")
        for dtype in ("float16", "bfloat16")
        for world_size in (2, 4, 8)
        for output_profile in ("110", "111")
        for use_pdl in (False, True)
    }
    assert len({spec.device_path for spec in specs}) == 24
    assert len({spec.binding_path for spec in specs}) == 48
    assert get_type_hints(cake_finalize.CakeMoeFinalizeLaunchGrid) == {
        "grid_y": Literal[1],
        "grid_z": Literal[1],
    }
    expected_launch_grid = cake_finalize.CakeMoeFinalizeLaunchGrid(
        grid_y=1,
        grid_z=1,
    )
    assert all(spec.launch_grid == expected_launch_grid for spec in specs)
    assert cake_finalize._MAX_COMM_SIZE == trtllm_ar.MAX_COMM_SIZE
    assert cake_finalize._CONTRACT["max_lamport_comm_bytes"] == 2_145_386_496
    assert "max_tokens" not in cake_finalize._CONTRACT


def test_cake_moe_finalize_lamport_size_matches_trtllm_limit() -> None:
    for world_size, max_tokens in ((2, 74825), (4, 37412), (8, 18706)):
        assert (
            cake_finalize._lamport_comm_size_bytes(
                max_tokens,
                7168,
                2,
                world_size,
            )
            <= cake_finalize._MAX_COMM_SIZE
        )
        assert (
            cake_finalize._lamport_comm_size_bytes(
                max_tokens + 1,
                7168,
                2,
                world_size,
            )
            > cake_finalize._MAX_COMM_SIZE
        )


def test_cake_moe_finalize_jit_consumes_verified_compile_contract(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()
    spec = cake_finalize.CakeMoeFinalizeModuleSpec(
        arch="sm_103a",
        dtype="bfloat16",
        world_size=8,
        output_profile="111",
        use_pdl=True,
        name="cake_trtllm_moe_finalize_bfloat16_ws8_o111_pdl1",
        module_ident="cake_trtllm_moe_finalize_bfloat16_ws8_o111_pdl1",
        ffi_entry="run",
        device_path=Path("generated_device.cu"),
        binding_path=Path("cake_binding.cu"),
        closure_sha256="a" * 64,
        arg_plan=cake_finalize._ARG_PLAN,
        launch_grid=cake_finalize.CakeMoeFinalizeLaunchGrid(grid_y=1, grid_z=1),
    )

    def fake_gen_jit_spec(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        cake_finalize,
        "get_cake_moe_finalize_module_spec",
        lambda **_kwargs: spec,
    )
    monkeypatch.setattr(cake_finalize, "_source_dir", lambda: Path("csrc/package"))
    monkeypatch.setattr(cake_finalize, "gen_jit_spec", fake_gen_jit_spec)
    cake_finalize.gen_cake_moe_finalize_module.cache_clear()

    result = cake_finalize.gen_cake_moe_finalize_module(
        "sm_103a", "bfloat16", 8, "111", True
    )

    assert result is sentinel
    assert captured["sources"] == [spec.device_path, spec.binding_path]
    assert "-gencode=arch=compute_103a,code=sm_103a" in captured["extra_cuda_cflags"]
    assert "--use_fast_math" in captured["extra_cuda_cflags"]
    assert captured["extra_include_paths"] == [Path("csrc")]
    cake_finalize.gen_cake_moe_finalize_module.cache_clear()
