"""Stage compressed checkpoint scales for CUDA slicing and layout conversion."""

from dataclasses import dataclass

import torch


@dataclass
class CsfTransfer:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    starts: torch.Tensor
    rows: int
    columns: int
    join: int
    row_range: tuple[int, int] | None
    column_range: tuple[int, int] | None

    def convert(self):
        """Convert after the caller flushes the owning direct-weight session."""
        from b12x._lib.quant._nvfp4_csf_prepare import convert_nvfp4_csf_storage

        return convert_nvfp4_csf_storage(
            self.fixed,
            self.exceptions,
            self.starts,
            rows=self.rows,
            columns=self.columns,
            join=self.join,
            row_range=self.row_range,
            column_range=self.column_range,
        )


def queue_nvfp4_csf(
    session,
    fixed_sources,
    exception_sources,
    *,
    rows,
    columns,
    join=1,
    row_range=None,
    column_range=None,
):
    """Queue source segments directly to CUDA without reading CPU scale values."""
    if not fixed_sources or len(fixed_sources) != len(exception_sources):
        raise ValueError("CSF source lists must be nonempty and equally sized")
    shape = (rows // 16, 16 * (1 + columns // 2))
    if any(f.dtype != torch.uint8 or f.shape != shape for f in fixed_sources):
        raise ValueError("CSF fixed plane geometry or dtype mismatch")
    if any(e.dtype != torch.uint32 or e.ndim != 1 for e in exception_sources):
        raise ValueError("CSF exceptions must be uint32 vectors")
    offsets = [0]
    for source in exception_sources:
        offsets.append(offsets[-1] + source.numel())
    device = torch.device("cuda", session.device)
    fixed = torch.empty((len(fixed_sources), *shape), dtype=torch.uint8, device=device)
    exceptions = torch.empty(offsets[-1], dtype=torch.uint32, device=device)
    for i, (f, e) in enumerate(zip(fixed_sources, exception_sources, strict=True)):
        if not session(fixed[i], f) or not session(
            exceptions[offsets[i] : offsets[i + 1]], e
        ):
            raise RuntimeError("Direct loader rejected CSF transfer")
    return CsfTransfer(
        fixed,
        exceptions.view(torch.uint8),
        torch.tensor(offsets, dtype=torch.int64, device=device),
        rows,
        columns,
        join,
        row_range,
        column_range,
    )
