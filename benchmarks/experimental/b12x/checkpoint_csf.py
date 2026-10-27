"""Slab-format checkpoint readers for NVFP4 baseline kernel measurements."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

_PADDED_SCALE = 0x38


@dataclass(frozen=True)
class CsfMatrix:
    weight: object
    fixed: torch.Tensor
    exceptions: torch.Tensor
    global_scale: torch.Tensor
    input_scale: torch.Tensor


def tp_extent(intermediate_size: int, tp_rank: int, tp_size: int, alignment: int):
    """Validate an equal TP partition and return its intermediate-axis bounds."""
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or intermediate_size <= 0
        or intermediate_size % (alignment * tp_size)
    ):
        raise ValueError(
            f"CSF requires a valid TP rank and extents aligned to {alignment} "
            "intermediate channels"
        )
    local = intermediate_size // tp_size
    return tp_rank * local, (tp_rank + 1) * local


class TensorView:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "U8" if self.tensor.dtype == torch.uint8 else str(self.tensor.dtype)

    def __getitem__(self, extent):
        return self.tensor[extent]


def _slice_scale_plane(
    fixed,
    exceptions,
    rows,
    columns,
    row_slice,
    column_slice,
    out_rows=None,
    out_columns=None,
):
    """Copy aligned compressed extents and rebase exception positions.

    ``out_rows``/``out_columns`` extend the slice to a padded TP shard whose
    weights are zero there: appended rows get the E4M3 scale 1.0 (a zero scale
    would be stored as a replacement word in every packed word), appended
    columns zero codes over their row's base.
    """
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("NVFP4-CSF row slices require complete 16-row slabs")
    if not (0 <= c0 < c1 <= columns and c0 % 2 == c1 % 2 == columns % 2 == 0):
        raise ValueError("NVFP4-CSF column slices require complete nibble pairs")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("NVFP4-CSF requires uint8 fixed and uint32 exception tensors")
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + columns // 2))
    bases = stream[:, :16]
    if np.any(bases > 240):
        raise ValueError("NVFP4-CSF row bases must be in 0..240")
    packed = stream[:, 16:].reshape(rows, columns // 2)
    out_rows = r1 - r0 if out_rows is None else out_rows
    out_columns = c1 - c0 if out_columns is None else out_columns
    if out_rows < r1 - r0 or out_rows % 16 or out_columns < c1 - c0 or out_columns % 2:
        raise ValueError("NVFP4-CSF padded slices must extend whole slabs and pairs")
    selected = np.zeros((out_rows, out_columns // 2), dtype=np.uint8)
    selected[: r1 - r0, : (c1 - c0) // 2] = packed[r0:r1, c0 // 2 : c1 // 2]
    slab_bases = np.full((out_rows // 16, 16), _PADDED_SCALE, dtype=np.uint8)
    slab_bases[: (r1 - r0) // 16] = bases[r0 // 16 : r1 // 16]
    result = np.concatenate((slab_bases, selected.reshape(out_rows // 16, -1)), 1)
    words = exceptions.numpy().reshape(-1)
    positions = words & np.uint32(0xFFFFFF)
    if len(words) and (
        positions[-1] >= rows * columns or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError("NVFP4-CSF exception positions must be sorted and unique")
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    local_positions = (rr[keep] - r0) * out_columns + cc[keep] - c0
    selected_words = (words[keep] & np.uint32(0xFF000000)) | local_positions
    return result.copy(), selected_words.astype(np.uint32)


def prepare_nvfp4_csf_weights(
    experts: Iterable[tuple[CsfMatrix, CsfMatrix, CsfMatrix]],
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
    local_size=None,
):
    """Slice and upload up/gate/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    ``local_size`` is the per-rank width of a TP-padded expert (GLM-5.3 at
    TP6: 2048 channels padded to 6 x 352): rank r holds checkpoint channels
    [r * local_size, (r + 1) * local_size) and zeros past the checkpoint.
    """
    from b12x.moe.fused_moe import PackedWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 128:
        raise ValueError("NVFP4-CSF requires experts and 128-aligned hidden channels")
    if local_size is None:
        first, last = tp_extent(intermediate_size, tp_rank, tp_size, 32)
        local = last - first
    else:
        local = int(local_size)
        if local <= 0 or local % 32 or local * tp_size < intermediate_size:
            raise ValueError(
                "NVFP4-CSF padded shards need 32-aligned widths covering the expert"
            )
        first = min(tp_rank * local, intermediate_size)
        last = min(first + local, intermediate_size)
    real = last - first
    if real <= 0:
        raise ValueError("NVFP4-CSF requires checkpoint channels on every TP rank")
    w13 = torch.zeros(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    # Model loaders may set BF16 as the default dtype. Calibration belongs to
    # the source FP32 contract and must not be rounded with the model weights.
    g13, g2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    a13, a2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []

    def scalar(value, role):
        if value is None or value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(f"NVFP4 {role} must be one FP32 value")
        if not bool(torch.isfinite(value).all() and (value > 0).all()):
            raise ValueError(f"NVFP4 {role} must be positive and finite")
        return value.reshape(())

    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        f13, e13, global13, input13 = [], [], [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() != "U8":
                raise ValueError(
                    f"NVFP4 nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            global_scale, input_scale = (
                scalar(projection.global_scale, "global_scale"),
                scalar(projection.input_scale, "input_scale"),
            )
            if matrix < 2:
                if real:
                    w13[expert, matrix * local : matrix * local + real].copy_(
                        view[first:last, :]
                    )
                rows, columns = intermediate_size, hidden_size // 16
                row_slice, column_slice = (first, last), (0, columns)
                out_rows, out_columns = local, columns
                global13.append(global_scale)
                input13.append(input_scale)
            else:
                if real:
                    w2[expert, :, : real // 2].copy_(view[:, first // 2 : last // 2])
                rows, columns = hidden_size, intermediate_size // 16
                row_slice, column_slice = (0, rows), (first // 16, last // 16)
                out_rows, out_columns = rows, local // 16
                g2[expert], a2[expert] = global_scale, input_scale
            fixed, exceptions = _slice_scale_plane(
                projection.fixed,
                projection.exceptions,
                rows,
                columns,
                row_slice,
                column_slice,
                out_rows,
                out_columns,
            )
            if matrix < 2:
                f13.append(fixed)
                if matrix:
                    exceptions += np.uint32(local * columns)
                e13.append(exceptions)
            else:
                fixed2.append(fixed)
                exceptions2.append(exceptions)
        if not torch.equal(global13[0], global13[1]):
            raise ValueError("NVFP4-CSF requires equal gate/up global weight scales")
        g13[expert] = global13[0]
        a13[expert] = torch.stack(input13).amax()
        fixed13.append(np.concatenate(f13))
        exceptions13.append(np.concatenate(e13))
    return SlabNvfp4Weights(
        packed=PackedWeights(
            w13=w13.to(device),
            w2=w2.to(device),
            w13_block_scales=w13_scale_scratch,
            w2_block_scales=w2_scale_scratch,
            w13_global_scales=g13.to(device),
            w2_global_scales=g2.to(device),
            input_scale=a13.to(device).reciprocal(),
            intermediate_scale=a2.to(device).reciprocal(),
            immutable_input_scales=True,
        ),
        w13_scales=SlabScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed13),
            tuple(torch.from_numpy(p) for p in exceptions13),
        ),
        w2_scales=SlabScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed2),
            tuple(torch.from_numpy(p) for p in exceptions2),
        ),
    )


@dataclass(frozen=True)
class SlabScalePlanes:
    """Slab baseline scale tensors used only by checkpoint comparisons."""

    fixed: tuple
    exceptions: tuple


@dataclass(frozen=True)
class SlabNvfp4Weights:
    """Native packed weights and slab scales for the expansion baseline."""

    packed: object
    w13_scales: object
    w2_scales: object


def prepare_reference_weights(*, plan, weights):
    """Prepare the slab baseline or a block-format comparison arm."""
    from dataclasses import replace

    from b12x.moe.fused_moe import (
        CsfBlockPlanes,
        Nvfp4CsfWeights,
        PreparedExperts,
        PackedSource,
        WeightPacking,
    )
    from b12x.moe.fused_moe.planning import ActivationMode, prepare_weights

    if not hasattr(weights, "w13_scales"):
        return prepare_weights(plan=plan, weights=weights)
    if isinstance(weights.w13_scales, CsfBlockPlanes):
        return prepare_weights(
            plan=plan,
            weights=Nvfp4CsfWeights(
                weights.packed, weights.w13_scales, weights.w2_scales
            ),
        )
    from b12x._lib.quant.csf_blocks import Nvfp4BlockPlane, Nvfp4CompactBlockPlane
    from b12x._lib.quant.nvfp4_csf import (
        Nvfp4CsfDecoder,
        make_nvfp4_csf_batch,
        repack_nvfp4_csf_batch,
    )

    if (
        not isinstance(plan.source, PackedSource)
        or plan.source.format.value != "modelopt_nvfp4"
        or plan.source.w13_layout.value != "w13"
        or plan.activation.mode not in (ActivationMode.A4, ActivationMode.A16)
        or (
            plan.activation.mode is ActivationMode.A16
            and plan.prepared_format.packing is not WeightPacking.MMA_PACKED
        )
        or (
            plan.activation.a16_max_tokens
            and plan.activation.mode is not ActivationMode.A16
        )
    ):
        raise ValueError(
            "NVFP4-CSF requires native ModelOpt NVFP4 A4/A16 in up/gate order"
        )
    geometry = plan.geometry
    planes = tuple(
        make_nvfp4_csf_batch(
            plane.fixed,
            plane.exceptions,
            rows=rows,
            columns=columns,
            device=weights.packed.w13.device,
        )
        if isinstance(plane, SlabScalePlanes)
        else plane
        for plane, rows, columns in (
            (
                weights.w13_scales,
                2 * geometry.intermediate_size,
                geometry.hidden_size // 16,
            ),
            (
                weights.w2_scales,
                geometry.hidden_size,
                geometry.intermediate_size // 16,
            ),
        )
    )
    block_scales = isinstance(planes[0], Nvfp4BlockPlane)
    compact_scales = isinstance(planes[0], Nvfp4CompactBlockPlane)
    if block_scales != isinstance(planes[1], Nvfp4BlockPlane):
        raise ValueError("NVFP4 projections must use the same CSF storage layout")
    if compact_scales != isinstance(planes[1], Nvfp4CompactBlockPlane):
        raise ValueError("NVFP4 projections must use the same exception index")
    if block_scales and plan.activation.mode is ActivationMode.A16:
        raise ValueError("Block CSF W4A16 scale preparation is not implemented")
    if plan.activation.mode is ActivationMode.A16:
        from b12x.moe._shared.kernels.w4a16.prepare import (
            _nvfp4_compute_scale_factor,
            _process_nvfp4_packed_scales,
        )

        # Standard W4A16 preparation chooses one scale normalization per
        # projection and packs weights/scales into its MMA layout. Expand
        # one layer in shared scratch to retain that exact preparation law.
        source_scales = (
            weights.packed.w13_block_scales,
            weights.packed.w2_block_scales,
        )
        staging_decoder = Nvfp4CsfDecoder.prepare(*planes, *source_scales)
        ids = torch.arange(
            plan.geometry.num_experts,
            dtype=torch.int32,
            device=weights.packed.w13.device,
        )
        staging_decoder.decode(ids, *source_scales)
        factors = tuple(
            _nvfp4_compute_scale_factor(t, plan.activation.io_dtype)
            for t in source_scales
        )
    prepared = prepare_weights(plan=plan, weights=weights.packed)
    if plan.activation.mode is ActivationMode.A16:
        representation = prepared._impl.representation
        packed = representation.value
        # The native F8_128x4 grid pads rows to 128 and k-groups to four;
        # W4A16's packed scales are unpadded, a prefix of that buffer.
        outputs = tuple(
            source.reshape(-1)[: target.numel()].view(target.shape)
            for source, target in zip(
                source_scales, (packed.w13_scale, packed.w2_scale), strict=True
            )
        )
        tables = []
        for factor in factors:
            alphabet = torch.arange(
                256, dtype=torch.uint8, device=weights.packed.w13.device
            ).view(torch.float8_e4m3fn)
            alphabet = (
                alphabet[:, None]
                .expand(256, 4)
                .contiguous()
                .to(plan.activation.io_dtype)
            )
            tables.append(
                _process_nvfp4_packed_scales(alphabet, scale_factor=factor)
                .view(torch.uint8)[:, 0]
                .contiguous()
            )
        planes = tuple(
            repack_nvfp4_csf_batch(plane, row_rotation=rotation, value_lut=table)
            for plane, rotation, table in zip(
                planes, (plan.geometry.intermediate_size, 0), tables, strict=True
            )
        )
        from b12x._lib.quant.nvfp4_csf_packed import (
            PackedCsfPlane,
            build_packed_csf_scales,
        )

        planes = tuple(
            PackedCsfPlane.of(build_packed_csf_scales(plane, inline=0))
            for plane in planes
        )
        packed = replace(packed, w13_scale=outputs[0], w2_scale=outputs[1])
        prepared = replace(
            prepared,
            _impl=replace(
                prepared._impl,
                representation=replace(representation, value=packed),
                w1_blockscale=outputs[0],
                w2_blockscale=outputs[1],
            ),
        )
    indexed = None
    if not block_scales and plan.activation.mode is ActivationMode.A4:
        from b12x._lib.quant.nvfp4_csf_indexed import prepare_indexed_scales

        indexed = tuple(prepare_indexed_scales(plane) for plane in planes)
        planes = tuple(
            replace(
                plane,
                fixed=value.storage[1024 : 1024 + plane.fixed.numel()].view_as(
                    plane.fixed
                ),
            )
            for plane, value in zip(planes, indexed, strict=True)
        )
    decoder = Nvfp4CsfDecoder.prepare(
        *planes,
        prepared._impl.w1_blockscale,
        prepared._impl.w2_blockscale,
        indexed_scales=indexed,
    )
    return PreparedExperts(
        plan=plan,
        _impl=replace(
            prepared._impl,
            plan=plan._impl,
            nvfp4_csf=decoder,
        ),
    )


def synthetic_planes(rng, experts, rows, columns, exceptions, dense_rows, device):
    """CSF scale planes and their logical E8M0 grids."""
    from b12x._lib.quant.x4t_scales import make_x4t_scale_batch

    fixed, records, grids = [], [], []
    for _ in range(experts):
        bases = rng.integers(112, 124, rows).astype(np.uint8)
        grid = bases[:, None] + rng.integers(0, 2, (rows, columns)).astype(np.uint8)
        hot = rng.random((rows, columns)) < exceptions
        grid[hot] = (bases[:, None] + rng.choice([-1, 2], (rows, columns)))[hot]
        for row in rng.choice(rows, dense_rows, replace=False):
            grid[row] = bases[row] + rng.integers(-1, 4, columns)
        selected = grid == bases[:, None] + 1
        outside = (grid != bases[:, None]) & ~selected
        selectors = np.packbits(selected, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        records.append(
            torch.from_numpy(
                position | (grid.reshape(-1)[position].astype(np.uint32) << 24)
            )
        )
        grids.append(grid)
    batch = make_x4t_scale_batch(
        fixed, records, rows=rows, columns=columns, device=device
    )
    return batch, torch.from_numpy(np.stack(grids)).to(device)
