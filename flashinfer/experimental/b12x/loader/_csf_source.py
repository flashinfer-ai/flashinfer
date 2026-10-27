"""Checkpoint components and logical views for GPU compressed-scale loading."""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class CsfSource:
    """Checkpoint planes sliced and joined by metadata only.

    Direct loaders populate CUDA tensors. Readers using CPU file mappings may
    retain compressed bytes on the host until the preparation device is known.
    """

    components: tuple[torch.Tensor, ...]
    starts: torch.Tensor
    source_rows: int
    source_columns: int
    quantization: str
    segments: int
    encoding: str = "csf_32x2_v1"
    join: int = 1
    row_range: tuple[int, int] | None = None
    column_range: tuple[int, int] | None = None
    padded_rows: int | None = None
    padded_columns: int | None = None
    padding_base: int = 0

    @property
    def rows(self):
        first, last = self.row_range or (0, self.source_rows)
        return (self.padded_rows or last - first) * self.join

    @property
    def columns(self):
        first, last = self.column_range or (0, self.source_columns)
        return self.padded_columns or last - first

    @property
    def num_experts(self):
        return self.segments // self.join

    @property
    def device(self):
        return self.components[0].device

    def select(self, **kwargs):
        result = replace(self, **kwargs)
        result.validate()
        return result

    def to(self, device):
        device = torch.device(device)
        if self.device == device:
            return self
        return replace(
            self,
            components=tuple(t.to(device) for t in self.components),
            starts=self.starts.to(device),
        )

    def validate(self):
        bits = {"nvfp4": 4, "mxfp4": 1}.get(self.quantization)
        if bits is None or self.encoding not in ("csf", "csf_32x2_v1"):
            raise ValueError("Unsupported CSF quantization or checkpoint encoding")
        r, c = self.source_rows, self.source_columns
        if r <= 0 or r % 32 or c <= 0 or self.segments <= 0:
            raise ValueError(
                "CSF source requires positive geometry and complete 32-row blocks"
            )
        if self.join <= 0 or self.segments % self.join:
            raise ValueError("CSF segments must contain whole joined experts")
        r0, r1 = self.row_range or (0, r)
        c0, c1 = self.column_range or (0, c)
        if not (0 <= r0 < r1 <= r and r0 % 32 == r1 % 32 == 0 and 0 <= c0 < c1 <= c):
            raise ValueError(
                "CSF slices require complete 32-row groups and valid columns"
            )
        if self.rows // self.join < r1 - r0 or self.rows % 32 or self.columns < c1 - c0:
            raise ValueError("CSF padding must cover the selected logical rectangle")
        if not 0 <= self.padding_base <= 256 - (1 << bits):
            raise ValueError("CSF padding base exceeds its palette")
        if self.encoding == "csf":
            if r * c > 1 << 24:
                raise ValueError(
                    "CSF slab positions require fewer than 2**24 scale elements"
                )
            shapes = ((self.segments, r // 16, 16 * (1 + (c * bits + 7) // 8)), (-1,))
            dtypes = (torch.uint8, torch.uint32)
        else:
            groups = (r + 127) // 128 * ((c + 3) // 4)
            pages = (groups + 31) // 32
            shapes = (
                (self.segments, r),
                (self.segments, r * c * bits // 8),
                (self.segments, pages),
                (self.segments, r // 32 * ((c + 1) // 2) + pages),
                (-1,),
            )
            dtypes = (
                torch.uint8,
                torch.uint8,
                torch.uint32,
                torch.uint16,
                torch.uint16,
            )
        if len(self.components) != len(shapes):
            raise ValueError("CSF checkpoint component count differs from its encoding")
        for tensor, shape, dtype in zip(self.components, shapes, dtypes, strict=True):
            if (
                tensor.dtype != dtype
                or tensor.ndim != len(shape)
                or any(
                    want != -1 and have != want
                    for have, want in zip(tensor.shape, shape)
                )
                or tensor.device.type not in ("cpu", "cuda")
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "CSF components require matching contiguous geometry, device and dtypes"
                )
        if (
            self.starts.dtype != torch.int64
            or self.starts.shape != (self.segments + 1,)
            or self.starts.device != self.device
            or not self.starts.is_contiguous()
        ):
            raise ValueError(
                "CSF exception segment starts require a contiguous int64 vector"
            )


def stage_csf_sources(
    components,
    *,
    rows,
    columns,
    quantization,
    device,
    encoding="csf_32x2_v1",
    copy=None,
    **selection,
):
    """Copy checkpoint components verbatim, optionally through a direct-load session.

    Component shapes supply allocation sizes. Scale values are never inspected
    or transformed on the host. The caller flushes queued copies before use.
    """
    components = tuple(tuple(plane) for plane in components)
    if not components:
        raise ValueError("CSF staging requires at least one source plane")
    starts = [0]
    for plane in components:
        starts.append(starts[-1] + plane[-1].numel())
    tensors = [
        torch.empty((len(components), *t.shape), dtype=t.dtype, device=device)
        for t in components[0][:-1]
    ]
    tensors.append(
        torch.empty(starts[-1], dtype=components[0][-1].dtype, device=device)
    )
    for plane in components:
        if len(plane) != len(tensors):
            raise ValueError("CSF source component counts must agree")
        for field, src in enumerate(plane):
            expected = components[0][field]
            if src.dtype != expected.dtype or (
                src.ndim != 1
                if field == len(plane) - 1
                else src.shape != expected.shape
            ):
                raise ValueError("CSF source component geometry must agree")
    if copy is not None:
        for field, dst in enumerate(tensors):
            copy.concatenate(dst, [plane[field] for plane in components])
    else:
        for index, plane in enumerate(components):
            for field, src in enumerate(plane):
                dst = (
                    tensors[field][starts[index] : starts[index + 1]]
                    if field == len(plane) - 1
                    else tensors[field][index]
                )
                dst.copy_(src)
    result = CsfSource(
        tuple(tensors),
        torch.tensor(starts, dtype=torch.int64, device=device),
        rows,
        columns,
        quantization,
        len(components),
        encoding,
        **selection,
    )
    result.validate()
    return result


def upload_csf_planes(planes, device, **selection):
    """Upload export arrays without CPU slicing, decoding, or layout conversion."""
    first = planes[0]
    if any(
        (p.rows, p.columns, p.quantization)
        != (first.rows, first.columns, first.quantization)
        for p in planes
    ):
        raise ValueError("CSF source planes must have matching geometry")
    return stage_csf_sources(
        [
            tuple(
                torch.from_numpy(getattr(p, name))
                for name in ("bases", "codes", "page_bases", "offsets", "exceptions")
            )
            for p in planes
        ],
        rows=first.rows,
        columns=first.columns,
        quantization=first.quantization,
        device=device,
        **selection,
    )


def batch_csf_sources(planes, device, **selection):
    """Batch export arrays or matching GPU checkpoint views without decoding."""
    first = planes[0]
    if not isinstance(first, CsfSource):
        return upload_csf_planes(planes, device, **selection)
    keys = (
        "source_rows",
        "source_columns",
        "quantization",
        "encoding",
        "row_range",
        "column_range",
        "padded_rows",
        "padded_columns",
        "padding_base",
    )
    if any(
        p.segments != 1
        or p.join != 1
        or any(getattr(p, key) != getattr(first, key) for key in keys)
        for p in planes
    ):
        raise ValueError("CSF batching requires matching single-plane sources")
    options = {key: getattr(first, key) for key in keys[4:]}
    options.update(selection)
    return stage_csf_sources(
        [tuple(t[0] for t in p.components[:-1]) + (p.components[-1],) for p in planes],
        rows=first.source_rows,
        columns=first.source_columns,
        quantization=first.quantization,
        encoding=first.encoding,
        device=device,
        **options,
    )
