"""Executable layout invariants for the SM12x W4A4 MoE dispatch.

The static kernel and the branch-paired gated dynamic kernel build their block-scale (SF) layouts with
``tile_atom_to_shape_SF`` over the 128-aligned intermediate extent while the FP4 operands keep the true extent.  These
tests evaluate the real layout algebra (inside a ``@cute.jit`` host function, the only place the DSL allows it) and
compare it with the physical scale storage the dispatch produces (``_pad_intermediate_to_tile`` followed by
``convert_sf_from_mma_layout``):

* positive: for I in {80, 160, 320, 704} the kernel-side layout of the aligned extent has the same block count and the
  same batch stride as the physical storage, for w13 (branch-major, 2E batches) and for down (E batches);
* negative: the layout built from the true (unaligned) extent differs from the physical storage wherever the atom
  rounding differs (down K blocks for I=160/320/704), which proves the align128 guard is load-bearing; at I=80 the
  true and aligned geometries coincide (one 128-row block, two 64-column blocks) and the test records that fact;
* capacity plane: the static compile fakes are built from the routed-row capacity (``route_rows``), not the 32-row
  slot stride, and a pre-allocated static workspace whose route-output scratch or virtual-route scratch is sized from
  the slot stride is rejected by the pre-launch validator before any kernel runs.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

if is_cute_dsl_available():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blockscaled_layout as bl
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

HIDDEN, EXPERTS, TOPK = 256, 8, 2
SF_VEC = 16
ATOM_ROWS, ATOM_COLS = (
    128,
    4 * SF_VEC,
)  # one SF atom covers 128 rows x 64 columns of the operand


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _align128(v: int) -> int:
    return (v + 127) // 128 * 128


SF_VEC_CONST = 16


if is_cute_dsl_available():

    @cute.jit
    def _sf_layout_probe(
        out: cute.Tensor,
        m: cutlass.Constexpr[int],
        k: cutlass.Constexpr[int],
        l: cutlass.Constexpr[int],
    ):
        """Evaluate the kernels' SF layout helper on the host and store its geometry."""
        lay = bl.tile_atom_to_shape_SF((m, k, l), SF_VEC_CONST)
        out[0] = Int32(cute.size(lay, mode=[0]))
        out[1] = Int32(cute.size(lay, mode=[1]))
        out[2] = Int32(cute.size(lay, mode=[2]))
        out[3] = Int32(cute.crd2idx((0, 0, 1), lay))
        out[4] = Int32(cute.cosize(lay))


def _probe_layout(shape):
    """(rows, K blocks, batches, batch stride, cosize) of tile_atom_to_shape_SF(shape) as the kernels build it."""
    # Host-side JIT: the geometry is written into host memory (a device tensor would be written from the host).
    out = torch.zeros(5, dtype=torch.int32)
    _sf_layout_probe(from_dlpack(out), int(shape[0]), int(shape[1]), int(shape[2]))
    rows, cols, batches, batch_stride, cosize = out.tolist()
    return {
        "rows": rows,
        "cols": cols,
        "batches": batches,
        "batch_stride": batch_stride,
        "cosize": cosize,
    }


def _mirror(shape):
    """Pure-Python mirror of the atom tiling (ceil to 128-row x 64-column atoms, K atoms fastest, then rows, then L)."""
    m, k, l = shape
    m_tiles, k_tiles = -(-m // ATOM_ROWS), -(-k // ATOM_COLS)
    per_batch = m_tiles * k_tiles * ATOM_ROWS * (ATOM_COLS // SF_VEC)
    return {
        "m_tiles": m_tiles,
        "k_tiles": k_tiles,
        "batch_stride": per_batch,
        "cosize": per_batch * l,
    }


def _padded_storage(intermediate: int):
    """Physical scale storages the dispatch hands the kernels for a gated shape (w13 branch-major 2E batches, down E)."""
    from flashinfer.cute_dsl.utils import (
        convert_sf_from_mma_layout,
        convert_sf_to_mma_layout,
    )
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _pad_intermediate_to_tile,
    )

    torch.manual_seed(1)
    gs = torch.tensor([1.0], device="cuda")
    w1 = (
        torch.randn(
            EXPERTS, 2 * intermediate, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        / 10
    )
    w2 = (
        torch.randn(EXPERTS, HIDDEN, intermediate, dtype=torch.bfloat16, device="cuda")
        / 10
    )
    s1, s2 = [], []
    for e in range(EXPERTS):
        s1.append(
            fp4_quantize(
                w1[e], global_scale=gs, sf_vec_size=SF_VEC, is_sf_swizzled_layout=True
            )[1]
        )
        s2.append(
            fp4_quantize(
                w2[e], global_scale=gs, sf_vec_size=SF_VEC, is_sf_swizzled_layout=True
            )[1]
        )
    w1_sf = convert_sf_to_mma_layout(
        torch.cat(s1), m=2 * intermediate, k=HIDDEN, num_groups=EXPERTS
    )
    w2_sf = convert_sf_to_mma_layout(
        torch.cat(s2), m=HIDDEN, k=intermediate, num_groups=EXPERTS
    )
    n = _align128(intermediate)
    _, w1_sf_p, _, w2_sf_p, _, n_pad = _pad_intermediate_to_tile(
        w1,
        w1_sf,
        w2,
        w2_sf,
        None,
        intermediate,
        128,
        HIDDEN,
        EXPERTS,
        True,
        "nvfp4",
        pad_fp4=False,
    )
    assert n_pad == n
    w13_storage = convert_sf_from_mma_layout(
        w1_sf_p, m=2 * n, k=HIDDEN, num_groups=EXPERTS, sf_vec_size=SF_VEC
    ).contiguous()
    down_storage = convert_sf_from_mma_layout(
        w2_sf_p, m=HIDDEN, k=n, num_groups=EXPERTS, sf_vec_size=SF_VEC
    ).contiguous()
    return n, w13_storage, down_storage


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestScaleLayoutInvariants:
    @pytest.mark.parametrize("intermediate", [80, 160, 320, 704])
    def test_aligned_layout_matches_physical_storage(self, intermediate):
        n, w13_storage, down_storage = _padded_storage(intermediate)
        # w13: branch-major [n, K, 2E] - one 128-aligned branch per batch index.
        w13 = _probe_layout((n, HIDDEN, 2 * EXPERTS))
        assert w13 == {
            **w13,
            **{k: v for k, v in _mirror((n, HIDDEN, 2 * EXPERTS)).items() if k in w13},
        }
        assert w13["batches"] == 2 * EXPERTS
        assert w13["cosize"] == w13_storage.numel(), (w13, w13_storage.numel())
        assert (
            w13["batch_stride"]
            == w13_storage.numel() // (2 * EXPERTS)
            == n * HIDDEN // SF_VEC
        )
        assert (
            w13["rows"] // ATOM_ROWS == n // ATOM_ROWS
        )  # block count of the aligned extent
        # down: [K, n, E] - the reduction extent is the aligned intermediate size.
        down = _probe_layout((HIDDEN, n, EXPERTS))
        assert down["batches"] == EXPERTS
        assert down["cosize"] == down_storage.numel(), (down, down_storage.numel())
        assert (
            down["batch_stride"]
            == down_storage.numel() // EXPERTS
            == HIDDEN * n // SF_VEC
        )

    @pytest.mark.parametrize("intermediate", [80, 160, 320, 704])
    def test_true_extent_layout_differs_where_atoms_round(self, intermediate):
        """The layout of the true extent does not describe the physical storage: its down K-block count (and therefore
        its batch stride) is smaller whenever ceil(I/64) < ceil(align128(I)/64); w13 rounds rows to 128 in both cases,
        so its geometry differs only through the down view.  I=80 is the coincident case (1 x 2 atoms either way)."""
        n, w13_storage, down_storage = _padded_storage(intermediate)
        true_down = _probe_layout((HIDDEN, intermediate, EXPERTS))
        aligned_down = _probe_layout((HIDDEN, n, EXPERTS))
        expected_differs = -(-intermediate // ATOM_COLS) != -(-n // ATOM_COLS)
        assert expected_differs == (intermediate in (160, 320, 704))
        if expected_differs:
            assert (
                true_down["batch_stride"]
                < aligned_down["batch_stride"]
                == down_storage.numel() // EXPERTS
            )
            assert true_down["cosize"] != down_storage.numel()
        else:
            # I=80: the atom rounding makes the true and aligned SF geometries coincide, so the load-bearing invariant
            # is the TMA stride rule instead: a true-extent down view [K, 80, E] packs 40 bytes per row (not a legal
            # 16-byte TMA stride), which is why the dispatch streams the padded extent for this shape.
            assert true_down == aligned_down
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
                static_needs_256_extent,
                true_extent_views_supported,
            )

            assert intermediate // 2 % 16 != 0 and not true_extent_views_supported(
                intermediate
            )
            assert static_needs_256_extent(intermediate)
        true_w13 = _probe_layout((intermediate, HIDDEN, 2 * EXPERTS))
        assert true_w13["rows"] == _align128(
            intermediate
        )  # atom rounding hides the true extent for w13
        assert true_w13["batch_stride"] == w13_storage.numel() // (2 * EXPERTS)

    def test_mirror_matches_dsl_for_reference_shapes(self):
        for shape in (
            (384, 2560, 1024),
            (320, 2560, 1024),
            (2560, 384, 512),
            (2560, 320, 512),
            (768, 2560, 128),
        ):
            probe, mirror = _probe_layout(shape), _mirror(shape)
            assert (
                probe["batch_stride"] == mirror["batch_stride"]
                and probe["cosize"] == mirror["cosize"]
            ), (shape, probe, mirror)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestCapacityPlane:
    def test_static_compile_fakes_are_capacity_sized(self):
        """The route-output scratch fake spans route_rows (routed-row capacity) x retained groups x K and the packed
        planes span the slot stride; neither is sized from a constant slot count in place of the capacity."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        tree = ast.parse(inspect.getsource(moe_dispatch._get_static_kernel))
        fakes: dict[str, str] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and node.targets
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                if (
                    name.endswith("_fake")
                    and node.value.args
                    and len(node.value.args) > 1
                ):
                    fakes[name] = ast.unparse(node.value.args[1])
        assert "route_rows" in fakes["route_output_scratch_fake"], fakes[
            "route_output_scratch_fake"
        ]
        assert "_STATIC_SLOT_ROWS" not in fakes["route_output_scratch_fake"]
        # the chunk map (virtual-route scratch) is sized from the routed-row capacity as well
        assert "route_rows" in fakes["virt_route_scratch_fake"], fakes[
            "virt_route_scratch_fake"
        ]
        assert "_STATIC_SLOT_ROWS" not in fakes["virt_route_scratch_fake"]
        assert (
            "max_rows" in fakes["packed_a_fake"] and "state_E" in fakes["packed_a_fake"]
        ), fakes["packed_a_fake"]

    def test_slot_sized_scratch_is_rejected_before_launch(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        routed_rows = (
            3 * 32 * 8
        )  # 768 routed rows -> capacity well above the slot stride
        ws = moe_dispatch.allocate_sm120_moe_workspace(
            state_E=EXPERTS,
            weight_E=EXPERTS,
            routed_rows=routed_rows,
            k=HIDDEN,
            n=320,
            num_topk=TOPK,
            device=torch.device("cuda"),
            quant_mode="nvfp4",
            backend="static",
            activation="silu",
        )
        kwargs = dict(
            state_E=EXPERTS,
            weight_E=EXPERTS,
            routed_rows=routed_rows,
            k=HIDDEN,
            n=384,
            num_topk=TOPK,
            device=torch.device("cuda"),
            activation_precision="fp4",
            quant_mode="nvfp4",
        )
        moe_dispatch._validate_static_workspace_for_launch(
            ws, **kwargs
        )  # the capacity-sized workspace passes
        groups = moe_dispatch._static_retained_groups(384)
        good = ws.route_output_scratch
        ws.route_output_scratch = torch.empty(
            (moe_dispatch._STATIC_SLOT_ROWS, groups, HIDDEN),
            dtype=good.dtype,
            device=good.device,
        )
        with pytest.raises(ValueError, match="route_output_scratch"):
            moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
        ws.route_output_scratch = good
        good_virt = ws.virt_route_scratch
        ws.virt_route_scratch = torch.zeros(
            (EXPERTS * (1 + 1) + 8,), dtype=good_virt.dtype, device=good_virt.device
        )
        with pytest.raises(ValueError, match="virt_route_scratch"):
            moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
        ws.virt_route_scratch = good_virt
        moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
