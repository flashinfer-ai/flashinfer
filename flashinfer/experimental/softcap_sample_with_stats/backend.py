"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host side of the SM90 CuTe-DSL softcap + sampling-statistics backend: argument checks, the
persistent compile cache, the per-stream exchange workspace and the three launches.
"""

import functools
from typing import Dict, Optional, Tuple, Union

import torch

VOCAB_SIZE = 262144
MAX_ROWS = (
    65535  # 32-bit Philox counters and the magic row // rows_per_temperature division
)
MIN_SOFTCAP = 2.0**-16
MAX_SOFTCAP = 64.0
MAX_SEED = 2**32

_MODULE = "softcap_sample_with_stats"
_CN = 8  # CTAs per row group (keep in sync with kernels.CN)
_NHELP = 4  # helper CTAs on the SMs the cooperative grid leaves idle
# Ratio of a helper row's time to one cooperative row period on H200; only balances the
# helper suffix length against the cooperative grid.
_HELPER_RHO = 10.0

_LOG2E = 1.4426950408889634
_PHILOX_KA = 0x9E3779B9
_PHILOX_KB = 0xBB67AE85


def _is_sm90(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0)


def check_inputs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    seed: Union[int, torch.Tensor],
    softcap: float,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    """Raise on an argument the kernels do not support."""
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if not _is_sm90(logits.device):
        raise NotImplementedError(
            "softcap_sample_with_stats supports SM90 (H100 / H200) only; device "
            f"{logits.device} has compute capability "
            f"{torch.cuda.get_device_capability(logits.device)}"
        )
    if logits.dtype != torch.bfloat16:
        raise ValueError(f"logits must be bfloat16, got {logits.dtype}")
    if logits.dim() != 2 or logits.shape[1] != VOCAB_SIZE:
        raise ValueError(
            f"logits must have shape [R, {VOCAB_SIZE}], got {tuple(logits.shape)}"
        )
    if not logits.is_contiguous() or logits.data_ptr() % 16:
        raise ValueError("logits must be contiguous and 16-byte aligned")
    rows = logits.shape[0]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"logits must have 1 to {MAX_ROWS} rows, got {rows}")
    if (
        not isinstance(temperature, torch.Tensor)
        or temperature.device != logits.device
        or temperature.dtype != torch.float32
        or temperature.dim() != 1
        or not temperature.is_contiguous()
    ):
        raise ValueError(
            "temperature must be a contiguous 1-D float32 tensor on the logits' device"
        )
    groups = temperature.numel()
    if groups == 0 or rows % groups:
        raise ValueError(
            f"temperature.numel() ({groups}) must divide the number of rows ({rows})"
        )
    if isinstance(seed, torch.Tensor):
        if (
            seed.device != logits.device
            or seed.dtype != torch.int64
            or seed.numel() < 1
        ):
            raise ValueError(
                "a tensor seed must be an int64 tensor with at least one element on the "
                "logits' device"
            )
    elif (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED
    ):
        raise ValueError(
            f"seed must be an int in [0, 2**32) or an int64 tensor, got {seed!r}"
        )
    softcap = float(softcap)
    if not MIN_SOFTCAP <= softcap <= MAX_SOFTCAP:
        raise ValueError(f"softcap must be in [2**-16, {MAX_SOFTCAP}], got {softcap}")
    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 4:
            raise ValueError("out must be a tuple (argmax, sample, entropy, probs)")
        expect = (
            ("argmax", torch.int64, (rows,)),
            ("sample", torch.int64, (rows,)),
            ("entropy", torch.float32, (rows,)),
            ("probs", torch.bfloat16, (rows, VOCAB_SIZE)),
        )
        for t, (name, dtype, shape) in zip(out, expect, strict=True):
            if (
                not isinstance(t, torch.Tensor)
                or t.device != logits.device
                or t.dtype != dtype
                or tuple(t.shape) != shape
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"out {name} must be a contiguous {dtype} tensor of shape {shape} on "
                    "the logits' device"
                )
        if out[3].data_ptr() % 16:
            raise ValueError("out probs must be 16-byte aligned")


def _kernel_source_files() -> Tuple[str, ...]:
    from . import kernels

    return (kernels.__file__,)


def _sym_tensor(dtype, shape, align):
    """Contiguous placeholder with a dynamic row count; 2-D rows are VOCAB_SIZE wide."""
    import cutlass.cute as cute

    stride = (VOCAB_SIZE, 1) if len(shape) == 2 else (1,)
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


@functools.cache
def _compiled_main(device_index: int, seed_from_tensor: bool):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Uint32

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import kernels

    def compile_fn():
        r, nq, nx, ns = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
        return cute.compile(
            kernels._rowstats_host,
            _sym_tensor(cutlass.BFloat16, (r, VOCAB_SIZE), 16),
            _sym_tensor(Float32, (nq,), 4),
            _sym_tensor(cutlass.Int64, (r,), 8),
            _sym_tensor(cutlass.Int64, (r,), 8),
            _sym_tensor(Float32, (r,), 4),
            _sym_tensor(cutlass.BFloat16, (r, VOCAB_SIZE), 16),
            _sym_tensor(Int32, (nx,), 16),
            _sym_tensor(cutlass.Int64, (ns,), 8),
            Int32(1),
            Uint32(0),
            Uint32(1),
            tuple(Uint32(0) for _ in range(20)),
            Float32(0.1),
            Float32(30.0 * _LOG2E),
            Int32(1),
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            seed_from_tensor,
            options="--enable-tvm-ffi",
        )

    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            _MODULE,
            f"rowstats_v{VOCAB_SIZE}_seedtensor{int(seed_from_tensor)}",
            compile_fn,
            extra_key_files=_kernel_source_files(),
        )


@functools.cache
def _compiled_helper(device_index: int, seed_from_tensor: bool):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Uint32

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import kernels

    def compile_fn():
        r, nq, ns = cute.sym_int(), cute.sym_int(), cute.sym_int()
        return cute.compile(
            kernels._helper_host,
            _sym_tensor(cutlass.BFloat16, (r, VOCAB_SIZE), 16),
            _sym_tensor(Float32, (nq,), 4),
            _sym_tensor(cutlass.Int64, (r,), 8),
            _sym_tensor(cutlass.Int64, (r,), 8),
            _sym_tensor(Float32, (r,), 4),
            _sym_tensor(cutlass.BFloat16, (r, VOCAB_SIZE), 16),
            _sym_tensor(cutlass.Int64, (ns,), 8),
            Int32(1),
            Int32(1),
            Int32(4),
            Uint32(0),
            Uint32(1),
            tuple(Uint32(0) for _ in range(20)),
            Float32(0.1),
            Float32(30.0 * _LOG2E),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            seed_from_tensor,
            options="--enable-tvm-ffi",
        )

    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            _MODULE,
            f"helper_v{VOCAB_SIZE}_seedtensor{int(seed_from_tensor)}",
            compile_fn,
            extra_key_files=_kernel_source_files(),
        )


@functools.cache
def _compiled_cold_entropy(device_index: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import kernels

    def compile_fn():
        r, nq = cute.sym_int(), cute.sym_int()
        return cute.compile(
            kernels._cold_entropy_host,
            _sym_tensor(cutlass.BFloat16, (r, VOCAB_SIZE), 16),
            _sym_tensor(Float32, (nq,), 4),
            _sym_tensor(Float32, (r,), 4),
            Int32(1),
            Int32(1),
            Float32(0.1),
            Float32(30.0 * _LOG2E),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            _MODULE,
            f"cold_entropy_v{VOCAB_SIZE}",
            compile_fn,
            extra_key_files=_kernel_source_files(),
        )


@functools.cache
def _device_layout(device_index: int) -> Tuple[int, int]:
    """(row groups, helper CTAs) for this device's SM count."""
    nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
    groups = max(1, nsm // _CN)
    helpers = _NHELP if nsm - _CN * groups >= _NHELP else 0
    return groups, helpers


def _helper_rows(rows: int, groups: int, helpers: int) -> int:
    """Rows given to the helper CTAs: a multiple of `helpers` balancing the helpers' time
    against the cooperative grid's, preferring none on a tie."""
    if helpers == 0:
        return 0
    est = int(rows / (groups * _HELPER_RHO + helpers))
    best = (float((rows + groups - 1) // groups), 0)
    for k in range(max(1, est - 3), est + 4):
        if helpers * k >= rows:
            continue
        t = max(float((rows - helpers * k + groups - 1) // groups), k * _HELPER_RHO)
        if t < best[0]:
            best = (t, helpers * k)
    return best[1]


# Exchange records, one workspace per (device, stream): launches on one stream are ordered, so
# they can share it; concurrent launches on different streams must not. Each holds
# groups x 2 parities x CN ranks x 64-byte records, then a 16-byte completion counter, and is
# zero at rest (every launch leaves it zeroed). A workspace first allocated during CUDA graph
# capture comes from the graph's memory pool and is zero-filled by a captured memset.
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}
_dummy_seed: Dict[int, torch.Tensor] = {}


def _workspace(device: torch.device, groups: int) -> Tuple[torch.Tensor, int]:
    nrec16 = groups * 2 * _CN * 4  # 16-byte chunks of records
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    ws = _workspaces.get(key)
    if ws is None:
        ws = torch.zeros(nrec16 * 4 + 4, dtype=torch.int32, device=device)
        _workspaces[key] = ws
    return ws, nrec16


def run(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    seed: Union[int, torch.Tensor],
    softcap: float,
    argmax: torch.Tensor,
    sample: torch.Tensor,
    entropy: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Launch on the current stream; arguments must have passed `check_inputs`."""
    device = logits.device
    dev = device.index
    rows = logits.shape[0]
    rows_per_temp = rows // temperature.numel()
    groups, helpers = _device_layout(dev)
    n_help = _helper_rows(rows, groups, helpers)
    n_main = rows - n_help
    n_groups = max(1, min(n_main, groups))
    ws, nrec16 = _workspace(device, groups)

    seed_from_tensor = isinstance(seed, torch.Tensor)
    if seed_from_tensor:
        seed_t = seed
        rk: Tuple[int, ...] = (0,) * 20
    else:
        seed_t = _dummy_seed.get(dev)
        if seed_t is None:
            seed_t = _dummy_seed[dev] = torch.zeros(1, dtype=torch.int64, device=device)
        k0 = seed & 0xFFFFFFFF
        rk = tuple((k0 + r * _PHILOX_KA) & 0xFFFFFFFF for r in range(10)) + tuple(
            (r * _PHILOX_KB) & 0xFFFFFFFF for r in range(10)
        )

    cap = float(softcap)
    c2 = 2.0 * _LOG2E / cap
    caple = cap * _LOG2E
    # row // rows_per_temp as mul.hi + mad with a host-built reciprocal (exact below 2**16)
    clm, clb = ((-(-(1 << 32) // rows_per_temp)), 0) if rows_per_temp > 1 else (0, 1)

    _compiled_main(dev, seed_from_tensor)(
        logits,
        temperature,
        argmax,
        sample,
        entropy,
        probs,
        ws,
        seed_t,
        n_main,
        clm,
        clb,
        rk,
        c2,
        caple,
        n_groups,
        nrec16,
    )
    if n_help:
        _compiled_helper(dev, seed_from_tensor)(
            logits,
            temperature,
            argmax,
            sample,
            entropy,
            probs,
            seed_t,
            rows,
            n_main,
            helpers,
            clm,
            clb,
            rk,
            c2,
            caple,
        )
    _compiled_cold_entropy(dev)(
        logits, temperature, entropy, rows, rows_per_temp, c2, caple
    )
