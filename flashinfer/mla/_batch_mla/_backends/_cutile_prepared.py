# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-time export and launch-only execution of the existing cuTile MLA kernels.

Explicit public AOT signatures avoid ct.launch specializing on pointers or the
KV pool size first encountered during run/capture. No native kernel is changed.
"""

import ctypes
import functools
import io
import math
import threading

import torch

from ._capabilities import _BackendPlanUnsupportedError


_COMPILE_LOCK = threading.Lock()
# CUDA graphs can outlive their wrappers: never evict loaded code. Bound both
# signatures and libraries process-wide by admitting at most 128 exports (a
# split-KV plan can need two). At capacity, cached signatures remain usable;
# new signatures raise a typed refusal so auto can try another backend.
_MAX_LOADED_LIBRARIES = 128
_LOADED_LIBRARIES: dict[tuple, tuple] = {}


def _check(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"cuTile MLA CUDA driver call failed: {result!r}")
    return values[0] if values else None


def _configuration(batch, heads, page, capacity, sms, capability):
    """Freeze the native non-autotuned heuristic during planning."""
    choices = (128, 64, 32, 16, 8) if batch >= 16 else (16, 8, 32)
    block_h = next((h for h in choices if h <= heads and heads % h == 0), heads)
    if block_h & (block_h - 1):
        block_h = heads & -heads
    # Bound the existing per-page TMA/concatenation tree. Sixteen page-one
    # loads fail compilation for D=512 on tileiras 13.3; eight compile and
    # preserve the same attention semantics without a kernel change.
    block_n = min(max(page, 16), 8 * page)
    split_len, splits = capacity, 1
    if capacity > 256 and batch * max(heads // block_h, 1) < sms:
        raw = capacity // max(sms // batch, 1)
        split_len = max(1 << (max(raw, 1) - 1).bit_length(), 128)
        splits = (capacity + split_len - 1) // split_len
    num_ctas = 2 if batch >= 16 and block_h >= 64 else None
    # The 64-head/eight-key tile produces NaNs with two CTAs on SM100,
    # including through native ct.launch. One CTA preserves the same kernel
    # and correct BF16/FP16 results; larger head/key tiles retain two CTAs.
    if capability == (10, 0) and block_h == 64 and block_n == 8:
        num_ctas = 1
    return (
        block_h,
        block_n,
        splits,
        split_len,
        1 << (splits - 1).bit_length(),
        heads < 64 and capability not in ((12, 0), (12, 1)),
        num_ctas,
    )


def _validate_launch_grids(batch, heads, block_h, splits):
    # CUDA logical grid Y/Z are 16-bit; X is limited to 2**31 - 1.
    # Check actual native grids, not a blanket upper bound on query heads.
    if batch > 2**31 - 1:
        raise _BackendPlanUnsupportedError("cutile decode grid X exceeds 2147483647.")
    if heads // block_h > 65535:
        raise _BackendPlanUnsupportedError("cutile decode grid Y exceeds 65535.")
    if splits > 65535:
        raise _BackendPlanUnsupportedError("cutile decode grid Z exceeds 65535.")
    if splits > 1 and heads > 65535:
        raise _BackendPlanUnsupportedError("cutile reduction grid Y exceeds 65535.")


def _array(compilation, ct, dtype, shape, *, aliases=(), strides=None, divisible=1):
    kwargs = dict(
        index_dtype=ct.int64,
        stride_lower_bound_incl=0,
        alias_groups=aliases,
        may_alias_internally=False,
        stride_constant=strides
        if strides is not None
        else (None,) * (len(shape) - 1) + (1,),
        stride_divisible_by=divisible,
        shape_divisible_by=1,
        base_addr_divisible_by=2 if dtype in (ct.float16, ct.bfloat16) else 4,
    )
    # Public v1 export in cuda-tile 1.4 has no shape_constant argument. Preserve
    # its dynamic-shape ABI; v2 (1.5+) adds exact plan-known shape constraints.
    if hasattr(compilation.CallingConvention, "cutile_python_v2"):
        kwargs["shape_constant"] = shape
    else:
        kwargs["shape_divisible_by"] = tuple(n or 1 for n in shape)
    return compilation.ArrayConstraint(dtype, len(shape), **kwargs)


def _load_kernel(kernel, parameters, symbol, architecture, device_index):
    # Only called under _COMPILE_LOCK during plan, never on the run path.
    key = (kernel, parameters, symbol, architecture, device_index)
    cached = _LOADED_LIBRARIES.get(key)
    if cached is not None:
        return cached[1]
    if len(_LOADED_LIBRARIES) >= _MAX_LOADED_LIBRARIES:
        raise _BackendPlanUnsupportedError(
            f"cuTile MLA loaded-library limit ({_MAX_LOADED_LIBRARIES}) reached; "
            "existing kernels remain available, but new specializations cannot be loaded."
        )
    from cuda.bindings import driver
    from cuda.tile import compilation

    convention = compilation.CallingConvention
    cconv = (
        convention.cutile_python_v2()
        if hasattr(convention, "cutile_python_v2")
        else convention.cutile_python_v1()
    )
    signature = compilation.KernelSignature(parameters, cconv, symbol=symbol)
    binary = io.BytesIO()
    compilation.export_kernel(
        kernel, [signature], binary, gpu_code=architecture, output_format="cubin"
    )
    library = _check(driver.cuLibraryLoadData(binary.getvalue(), [], [], 0, [], [], 0))
    try:
        function = _check(driver.cuLibraryGetKernel(library, symbol.encode()))
        _check(
            driver.cuKernelGetFunction(function)
        )  # resolve lazy loading before capture
    except BaseException:
        driver.cuLibraryUnload(library)
        raise
    _LOADED_LIBRARIES[key] = (library, function)
    return function


def _parameters(arrays, scalars=()):
    # Public cuTile v1/v2 ABI: pointer, every shape, then every element stride.
    # Use int64 even for small plans: pool extents/strides are only known at run,
    # and grid-valid large-head plans can exceed int32 query row strides.
    values, types = [], []
    for tensor in arrays:
        shape = tuple(tensor.shape)
        # A singleton axis cannot affect addressing. Normalize its arbitrary
        # PyTorch stride rather than assume it matches a compact allocation.
        strides = tuple(
            s if n != 1 else 1 for n, s in zip(shape, tensor.stride(), strict=True)
        )
        if any(value < 0 or value > 2**63 - 1 for value in shape + strides):
            raise ValueError(
                "cuTile array shapes and strides must fit nonnegative int64."
            )
        values.append(tensor.data_ptr())
        types.append(ctypes.c_void_p)
        values.extend(shape + strides)
        types.extend((ctypes.c_int64,) * (2 * len(shape)))
    if any(value < 0 or value > 2**63 - 1 for value in scalars):
        raise ValueError("cuTile metadata strides must fit nonnegative int64.")
    values.extend(scalars)
    types.extend((ctypes.c_int64,) * len(scalars))
    return tuple(values), tuple(types)


class _PreparedCutileMLA:
    def __init__(
        self, *, device, batch, heads, page, table_width, dtype, sm_scale, dim=512
    ):
        import cuda.tile as ct
        from cuda.bindings import driver
        from cuda.tile import compilation
        from ....attention.kernels.cutile import fmha_decode_bsr_cutile as native
        from ....cutile.cutile_common import cached_replace_hints

        self.device, self.driver = device, driver
        self.batch, self.heads, self.dim = batch, heads, dim
        self.table_width = table_width
        capability = torch.cuda.get_device_capability(device)
        architecture = f"sm_{capability[0]}{capability[1]}"
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        config = _configuration(batch, heads, page, table_width * page, sms, capability)
        block_h, block_n, splits, split_len, padded_splits, trans_qk, num_ctas = config
        _validate_launch_grids(batch, heads, block_h, splits)
        self.splits = splits
        self._configuration = config
        tile_dtype = ct.float16 if dtype == torch.float16 else ct.bfloat16
        a = functools.partial(_array, compilation, ct)
        # Shared read-only alias group also covers independent inputs that happen
        # to share an allocation; the adapter separately rejects output overlap.
        query = a(
            tile_dtype,
            (batch, heads, dim),
            aliases=("inputs",),
            divisible=(1, 64 if heads > 1 else 1, 1),
        )
        query_rope = a(
            tile_dtype,
            (batch, heads, 64),
            aliases=("inputs",),
            divisible=(1, 64 if heads > 1 else 1, 1),
        )
        cache = a(
            tile_dtype,
            (None, page, dim),
            aliases=("inputs",),
            divisible=(1, 64 if page > 1 else 1, 1),
        )
        cache_rope = a(
            tile_dtype,
            (None, page, 64),
            aliases=("inputs",),
            divisible=(1, 64 if page > 1 else 1, 1),
        )
        lengths = a(ct.int32, (batch,), strides=(1,))
        table = a(ct.int32, (batch * table_width,), strides=(1,))
        partial = a(
            tile_dtype,
            (splits, batch, heads, dim),
            strides=(
                batch * heads * dim if splits > 1 else None,
                heads * dim if batch > 1 else None,
                dim if heads > 1 else None,
                1,
            ),
        )
        lse = (
            a(
                ct.float32,
                (batch, heads, padded_splits),
                strides=(
                    heads * padded_splits if batch > 1 else None,
                    padded_splits if heads > 1 else None,
                    1,
                ),
            )
            if splits > 1
            else a(ct.float32, (1,), strides=(1,))
        )
        parameters = (
            query,
            query_rope,
            cache,
            cache,
            cache_rope,
            lengths,
            table,
            partial,
            lse,
            float(sm_scale),
            1.0,
            page,
            block_h,
            block_n,
            dim,
            64,
            heads,
            splits,
            split_len,
            splits > 1,
            compilation.ScalarConstraint(ct.int64),
            min(block_n, page),
            max(block_n // page, 1),
            trans_qk,
        )
        kernel = native._decode_mla_kv_paged_kernel
        if num_ctas:
            kernel = cached_replace_hints(kernel, num_ctas=num_ctas)
        with torch.cuda.device(device), _COMPILE_LOCK:
            self.decode = _load_kernel(
                kernel, parameters, "flashinfer_mla_decode", architecture, device.index
            )
            self.reduce = None
            if splits > 1:
                output = a(
                    tile_dtype,
                    (batch, heads, dim),
                    strides=(
                        heads * dim if batch > 1 else None,
                        dim if heads > 1 else None,
                        1,
                    ),
                )
                reduce_parameters = (
                    partial,
                    lse,
                    output,
                    lengths,
                    heads,
                    split_len,
                    splits,
                    padded_splits,
                    dim,
                )
                self.reduce = _load_kernel(
                    native._splitk_reduce_kernel,
                    reduce_parameters,
                    "flashinfer_mla_reduce",
                    architecture,
                    device.index,
                )
        # All split buffers are private, staged before the backend publishes.
        self.partial = (
            torch.zeros((splits, batch, heads, dim), device=device, dtype=dtype)
            if splits > 1
            else None
        )
        self.lse = torch.full(
            (batch, heads, padded_splits) if splits > 1 else (1,),
            -math.inf,
            device=device,
            dtype=torch.float32,
        )
        self.decode_grid = (batch, heads // block_h, splits)
        self.reduce_grid = (batch, heads, 1)

    def _launch(self, kernel, grid, arrays, scalars=()):
        config = self.driver.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid
        # The driver uses logical tile blocks; native num_ctas is encoded by the
        # exported kernel, not a guessed physical CUDA thread/block dimension.
        config.blockDimX = config.blockDimY = config.blockDimZ = 1
        config.sharedMemBytes = 0
        config.numAttrs = 0
        config.hStream = torch.cuda.current_stream(self.device).cuda_stream
        _check(
            self.driver.cuLaunchKernelEx(
                config, kernel, _parameters(arrays, scalars), 0
            )
        )

    def __call__(
        self,
        q,
        qr,
        kv,
        kr,
        lengths,
        table,
        k_scale,
        v_scale,
        *,
        max_seq_len=-1,
        outputs,
    ):
        partial = (
            self.partial
            if self.reduce is not None
            else outputs.view(1, self.batch, self.heads, self.dim)
        )
        self._launch(
            self.decode,
            self.decode_grid,
            (q, qr, kv, kv, kr, lengths.view(-1), table.view(-1), partial, self.lse),
            (self.table_width,),
        )
        if self.reduce is not None:
            self._launch(
                self.reduce,
                self.reduce_grid,
                (partial, self.lse, outputs, lengths.view(-1)),
            )
        return outputs


def prepare_cutile_mla_decode(**kwargs):
    return _PreparedCutileMLA(**kwargs)
