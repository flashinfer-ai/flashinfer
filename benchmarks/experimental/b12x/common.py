from __future__ import annotations

from collections.abc import Callable
import csv
import io
import subprocess
import time

import torch

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x._lib.utils import get_hardware_info


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_AUTO_L2_FLUSH_MULTIPLIER = 2
_FALLBACK_L2_FLUSH_BYTES = 128 << 20
_L2_FLUSH_BUFFER_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


_NVIDIA_SMI_GPU_MODE_FIELDS = (
    "index",
    "uuid",
    "pstate",
    "persistence_mode",
    "compute_mode",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks_throttle_reasons.active",
    "power.draw",
    "power.limit",
    "temperature.gpu",
)


def nvidia_smi_gpu_mode_snapshot() -> dict[str, object]:
    """Capture the physical GPU's benchmark-relevant operating state."""
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch_uuid = str(getattr(properties, "uuid", ""))
    target_uuid = torch_uuid if torch_uuid.startswith("GPU-") else f"GPU-{torch_uuid}"
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_NVIDIA_SMI_GPU_MODE_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    captured_ns = time.time_ns()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "captured_unix_ns": captured_ns,
            "torch_uuid": torch_uuid,
            "nvidia_smi_uuid": target_uuid,
            "command": command,
            "available": False,
            "error": str(error),
        }

    rows = list(csv.reader(io.StringIO(completed.stdout), skipinitialspace=True))
    matches = [
        row
        for row in rows
        if len(row) == len(_NVIDIA_SMI_GPU_MODE_FIELDS) and row[1] == target_uuid
    ]
    if len(matches) != 1:
        return {
            "captured_unix_ns": captured_ns,
            "torch_uuid": torch_uuid,
            "nvidia_smi_uuid": target_uuid,
            "command": command,
            "available": False,
            "error": f"expected one UUID match, found {len(matches)}",
            "raw_stdout": completed.stdout,
        }
    return {
        "captured_unix_ns": captured_ns,
        "torch_uuid": torch_uuid,
        "nvidia_smi_uuid": target_uuid,
        "command": command,
        "available": True,
        "fields": dict(zip(_NVIDIA_SMI_GPU_MODE_FIELDS, matches[0], strict=True)),
    }


def require_sm120() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run b12x benchmarks")
    return torch.device("cuda")


def resolve_l2_flush_bytes(bytes_hint: int) -> int:
    if bytes_hint < 0:
        raise ValueError(f"l2 flush bytes must be non-negative, got {bytes_hint}")
    if bytes_hint > 0:
        return int(bytes_hint)
    try:
        l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    except Exception:
        l2_bytes = 0
    if l2_bytes <= 0:
        try:
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            l2_bytes = int(properties.L2_cache_size)
        except Exception:
            l2_bytes = 0
    if l2_bytes > 0:
        return _AUTO_L2_FLUSH_MULTIPLIER * l2_bytes
    return _FALLBACK_L2_FLUSH_BYTES


def make_l2_flush_fn(
    enabled: bool,
    bytes_hint: int = 0,
) -> Callable[[], None] | None:
    """Return an L2 eviction sweep without leaving bulk writeback traffic."""
    if not enabled:
        return None
    flush_bytes = resolve_l2_flush_bytes(bytes_hint)
    device_idx = torch.cuda.current_device()
    cache_key = (device_idx, flush_bytes)
    state = _L2_FLUSH_BUFFER_CACHE.get(cache_key)
    if state is None:
        buffer = torch.ones(
            (flush_bytes + 3) // 4,
            dtype=torch.float32,
            device=f"cuda:{device_idx}",
        )
        reduction = torch.empty((), dtype=torch.float32, device=buffer.device)
        state = (buffer, reduction)
        _L2_FLUSH_BUFFER_CACHE[cache_key] = state
    buffer, reduction = state

    def flush(
        buf: torch.Tensor = buffer,
        out: torch.Tensor = reduction,
    ) -> None:
        torch.sum(buf, dim=0, out=out)

    return flush


def make_sparse_pool_locs(
    *,
    active_tokens: int,
    pool_tokens: int,
    seed: int,
    device: torch.device,
    page_size: int = 64,
) -> torch.Tensor:
    if pool_tokens < active_tokens:
        raise ValueError(
            f"pool_tokens {pool_tokens} must be at least active_tokens {active_tokens}"
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if pool_tokens % page_size != 0:
        raise ValueError(
            f"pool_tokens {pool_tokens} must be divisible by page_size {page_size} "
            "for the paged benchmark contract"
        )
    if active_tokens == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    active_pages = (active_tokens + page_size - 1) // page_size
    pool_pages = pool_tokens // page_size
    if pool_pages < active_pages:
        raise ValueError(
            f"pool page capacity {pool_pages} is smaller than active page count {active_pages}"
        )
    page_ids = torch.randperm(pool_pages, generator=gen, dtype=torch.int64)[
        :active_pages
    ]
    locs = []
    remaining = active_tokens
    for page_id in page_ids.tolist():
        take = min(page_size, remaining)
        locs.append(page_id * page_size + torch.arange(take, dtype=torch.int64))
        remaining -= take
    return torch.cat(locs).to(device=device, dtype=torch.int32)


def scatter_rows_into_pool(
    rows: torch.Tensor,
    *,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> torch.Tensor:
    pool_shape = (pool_tokens, *rows.shape[1:])
    pool = torch.zeros(pool_shape, dtype=rows.dtype, device=rows.device)
    pool[pool_locs.to(torch.long)] = rows
    return pool


def make_dense_candidate_page_table(
    *,
    batch_size: int,
    token_locs: torch.Tensor,
    width: int,
    fill_value: int = 0,
) -> torch.Tensor:
    if token_locs.ndim != 1:
        raise ValueError(f"token_locs must be rank-1, got {tuple(token_locs.shape)}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if token_locs.shape[0] > width:
        raise ValueError(
            f"width {width} must be at least token_locs length {token_locs.shape[0]}"
        )
    page_table = torch.full(
        (batch_size, width),
        int(fill_value),
        dtype=torch.int32,
        device=token_locs.device,
    )
    if token_locs.numel():
        page_table[:, : token_locs.shape[0]] = token_locs.unsqueeze(0).expand(
            batch_size, -1
        )
    return page_table


def make_dense_real_page_table(
    *,
    batch_size: int,
    token_locs: torch.Tensor,
    width_blocks: int,
    page_size: int = 64,
    fill_value: int = -1,
) -> torch.Tensor:
    if token_locs.ndim != 1:
        raise ValueError(f"token_locs must be rank-1, got {tuple(token_locs.shape)}")
    if width_blocks <= 0:
        raise ValueError(f"width_blocks must be positive, got {width_blocks}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    page_ids = token_locs[::page_size] // page_size
    if page_ids.shape[0] > width_blocks:
        raise ValueError(
            f"width_blocks {width_blocks} must be at least active page count {page_ids.shape[0]}"
        )
    real_page_table = torch.full(
        (batch_size, width_blocks),
        int(fill_value),
        dtype=torch.int32,
        device=token_locs.device,
    )
    if page_ids.numel():
        real_page_table[:, : page_ids.shape[0]] = page_ids.unsqueeze(0).expand(
            batch_size, -1
        )
    return real_page_table


def capture_cuda_graph(
    fn: Callable[[], object],
    *,
    warmup: int,
    prepare: Callable[[], None] | None = None,
) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if prepare is not None:
        prepare()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    graph._b12x_benchmark_call = fn
    return graph


def bench_cuda_graph(graph, *, replays, prepare=None, l2_flush=None):
    from b12x.testing.benchmark import graph_samples

    return graph_samples(graph, replays=replays, prepare=prepare, l2_flush=l2_flush)


def bench_gpu_ms(fn, *, warmup, iters, l2_flush=None):
    from b12x.testing.benchmark import median_ms

    return median_ms(fn, warmup=warmup, iters=iters, l2_flush=l2_flush)


def compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return torch.tensor([value], dtype=torch.float32, device=x.device)


def compute_per_group_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    numerator = torch.full_like(amax, FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    return torch.where(amax > 0, numerator / amax, torch.ones_like(amax))


def make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    scale: float = 0.25,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) * scale
    row_counts = torch.full(
        (shape[0],), shape[1], dtype=torch.int32, device=source.device
    )
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale
