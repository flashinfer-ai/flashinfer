"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SSD (Structured State-Space Duality) Combined Forward Pass for Mamba2
=====================================================================

This module provides the combined forward pass for Mamba2 SSD, combining:
1. Triton kernel for cumsum computation
2. CuTe DSL kernel for the main SSD computation (Blackwell optimized)
"""

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda_drv
import torch
from cutlass import Int32
from cutlass.base_dsl.compiler import GenerateLineInfo  # profiling
from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module
from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd
from .ssd_kernel import SSDKernel


@functools.cache
def _get_seq_chunk_cumsum_module():
    """Get cached seq_chunk_cumsum JIT module."""
    return gen_seq_chunk_cumsum_module().build_and_load()


@functools.cache
def _get_compiled_ssd_kernel(
    chunk_size: int,
    headdim: int,
    dstate: int,
    has_d: bool,
    d_has_hdim: bool,
    has_init_states: bool,
    has_varlen: bool,
    has_z: bool,
    io_dtype,
    state_dtype,
    seq_idx_dtype,
    cumsum_dtype,
    acc_dtype,
    max_active_clusters: int,
):
    """Compile SSD kernel with fake tensors for shape-polymorphic caching.

    Uses CuTe DSL fake tensors with symbolic dimensions so the compiled kernel
    can be reused across different batch sizes, sequence lengths, etc.
    The compiled kernel accepts torch tensors directly via TVM-FFI.
    """
    kernel_obj = SSDKernel(
        chunk_size,
        headdim,
        dstate,
        has_d,
        d_has_hdim,
        has_init_states,
        has_varlen,
        has_z,
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        seq_idx_dtype=seq_idx_dtype,
        cumsum_delta_dtype=cumsum_dtype,
        acc_dtype=acc_dtype,
    )

    L = chunk_size
    D = headdim
    N = dstate

    # Symbolic dimensions for all runtime-variable sizes
    sym_C = cute.sym_int()  # nchunks
    sym_EH = cute.sym_int()  # nheads
    sym_B = cute.sym_int()  # batch
    sym_G = cute.sym_int()  # ngroups

    # Stride order convention for make_fake_compact_tensor:
    #   stride_order[i] = rank of dim i, where rank 0 = innermost (stride 1).
    # This differs from mark_compact_shape_dynamic which uses inv(perm).
    # Formula: make_fake_stride_order[i] = N - 1 - permute_order[i]

    # x: (D, L, C, EH, B) — D stride 1 (from permute (4,2,1,3,0) of 5D)
    x_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (D, L, sym_C, sym_EH, sym_B),
        stride_order=(0, 2, 3, 1, 4),
        assumed_align=16,
    )

    # cumsum_delta: (L, C, EH, B) — L stride 1 (from permute (3,2,1,0) of 4D)
    cumsum_delta_fake = cute.runtime.make_fake_compact_tensor(
        cumsum_dtype,
        (L, sym_C, sym_EH, sym_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    # delta: (L, C, EH, B) — L stride 1 (same layout as cumsum_delta)
    delta_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, sym_C, sym_EH, sym_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    # B: (L, N, C, G, B) — N stride 1 (from permute (2,4,1,3,0) of 5D)
    b_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, N, sym_C, sym_G, sym_B),
        stride_order=(2, 0, 3, 1, 4),
        assumed_align=16,
    )

    # C: (L, N, C, G, B) — N stride 1 (same layout as B)
    c_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, N, sym_C, sym_G, sym_B),
        stride_order=(2, 0, 3, 1, 4),
        assumed_align=16,
    )

    # y: (L, D, C, EH, B) — L stride 1 (from permute (4,2,3,1,0) of 5D)
    y_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, D, sym_C, sym_EH, sym_B),
        stride_order=(0, 2, 1, 3, 4),
        assumed_align=16,
    )

    # init_states and fstate use a separate batch sym because in varlen mode
    # init_states has num_seqs as batch dim (not the packed batch dim of x).
    sym_state_B = cute.sym_int()

    # init_states: (N, D, EH, state_B) — N stride 1 (from permute (3,2,1,0) of 4D)
    init_states_fake = None
    if has_init_states:
        init_states_fake = cute.runtime.make_fake_compact_tensor(
            state_dtype,
            (N, D, sym_EH, sym_state_B),
            stride_order=(0, 1, 2, 3),
            assumed_align=16,
        )

    # fstate: (N, D, EH, state_B) — N stride 1 (same layout as init_states)
    fstate_fake = cute.runtime.make_fake_compact_tensor(
        state_dtype,
        (N, D, sym_EH, sym_state_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    # D tensor: (D_dim, EH) — D_dim stride 1 (from permute (1,0) of 2D)
    d_fake = None
    if has_d:
        d_dim = D if d_has_hdim else 1
        d_fake = cute.runtime.make_fake_compact_tensor(
            io_dtype,
            (d_dim, sym_EH),
            stride_order=(0, 1),
            assumed_align=16,
        )

    # z: same layout as x — optional
    z_fake = None
    if has_z:
        z_fake = cute.runtime.make_fake_compact_tensor(
            io_dtype,
            (D, L, sym_C, sym_EH, sym_B),
            stride_order=(0, 2, 3, 1, 4),
            assumed_align=16,
        )

    # Varlen metadata — optional
    seq_idx_fake = None
    chunk_indices_fake = None
    chunk_offsets_fake = None
    seq_chunk_cumsum_fake = None
    if has_varlen:
        sym_batch_dim = cute.sym_int()
        sym_seqlen = cute.sym_int()
        seq_idx_fake = cute.runtime.make_fake_compact_tensor(
            seq_idx_dtype,
            (sym_batch_dim, sym_seqlen),
            stride_order=(1, 0),
            assumed_align=4,
        )
        sym_nlc = cute.sym_int()  # num_logical_chunks
        chunk_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nlc,),
            assumed_align=4,
        )
        chunk_offsets_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nlc,),
            assumed_align=4,
        )
        sym_nsp1 = cute.sym_int()  # num_seqs + 1
        seq_chunk_cumsum_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nsp1,),
            assumed_align=4,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile[GenerateLineInfo(True)](
        kernel_obj,
        x_fake,
        cumsum_delta_fake,
        delta_fake,
        b_fake,
        c_fake,
        y_fake,
        init_states_fake,
        fstate_fake,
        d_fake,
        z_fake,
        seq_idx_fake,
        chunk_indices_fake,
        chunk_offsets_fake,
        seq_chunk_cumsum_fake,
        Int32(0),  # num_logical_chunks (dummy, runtime value)
        Int32(0),  # num_seqs (dummy, runtime value)
        max_active_clusters,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled


class SSDCombined:
    """Mamba2 SSD combined forward pass with cached host-side state.

    Caches compiled kernel, hardware info, and pre-allocated buffers
    to minimize host-side overhead on repeated calls.

    Dtype expectations (no runtime conversions — assert on mismatch):
        io_dtype (bf16):  x, B, C, z, D, out, dt_processed
        state_dtype (bf16): initial_states, final_states
        fp32 always:      A, dA_cumsum
        any (bf16/fp32):  dt, dt_bias (consumed by Triton cumsum preprocessor)

    Usage::

        ssd = SSDCombined(chunk_size=128, nheads=8, headdim=64,
                          dstate=128, ngroups=8)
        out, final_states = ssd.run(x, dt, A, B, C, D=D, dt_bias=dt_bias, ...)
    """

    def __init__(
        self,
        chunk_size: int,
        nheads: int,
        headdim: int,
        dstate: int,
        ngroups: int,
        io_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.bfloat16,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_initial_states: bool = False,
        has_varlen: bool = False,
        has_z: bool = False,
        seq_idx_dtype=torch.int64,
    ):
        from ..utils import get_compute_capability

        major, minor = get_compute_capability(torch.device("cuda"))
        if major < 10:
            raise ValueError(
                f"SSDCombined requires SM100+ (Blackwell or newer). "
                f"Got SM{major}{minor}."
            )

        self.chunk_size = chunk_size
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.ngroups = ngroups
        self._has_d = has_d
        self._d_has_hdim = d_has_hdim
        self._has_init_states = has_initial_states
        self._has_varlen = has_varlen
        self._has_z = has_z
        self._state_torch_dtype = state_dtype

        # Resolve dtypes
        assert io_dtype == torch.bfloat16, f"io_dtype must be bfloat16, got {io_dtype}"
        self._io_dtype = cutlass.BFloat16
        _state_dtype_map = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            # torch.float32: cutlass.Float32, # -- not supported yet
        }
        assert state_dtype in _state_dtype_map, (
            f"state_dtype must be one of {list(_state_dtype_map.keys())}, got {state_dtype}"
        )
        self._state_dtype = _state_dtype_map[state_dtype]
        self._cumsum_dtype = cutlass.Float32
        self._acc_dtype = cutlass.Float32
        self._io_torch_dtype = cutlass_torch.dtype(self._io_dtype)

        # Resolve seq_idx dtype
        _seq_idx_dtype_map = {
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }
        self._seq_idx_cutlass_dtype = _seq_idx_dtype_map.get(
            seq_idx_dtype, cutlass.Int32
        )

        # Compile kernel (cached across instances with same compile-time params).
        # HardwareInfo uses cuda.bindings.driver which requires a driver context.
        # torch.cuda.init() alone is insufficient; allocating on GPU forces
        # creation of the full driver context.
        _, ctx = cuda_drv.cuCtxGetCurrent()
        if int(ctx) == 0:
            torch.empty(0, device="cuda")
        hardware_info = cutlass.utils.HardwareInfo()
        self._max_active_clusters = hardware_info.get_max_active_clusters(1)

        self._compiled_kernel = _get_compiled_ssd_kernel(
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_initial_states,
            has_varlen,
            has_z,
            self._io_dtype,
            self._state_dtype,
            self._seq_idx_cutlass_dtype,
            self._cumsum_dtype,
            self._acc_dtype,
            self._max_active_clusters,
        )

        # Pre-allocated buffer caches (lazily initialized on first run)
        self._fstate_shape = None
        self._fstate_torch = None

        self._seq_cumsum_size = 0
        self._seq_cumsum_buf = None

    # -- buffer cache helpers --------------------------------------------------

    def _get_or_alloc_fstate(self, fstate_batch):
        shape = (fstate_batch, self.nheads, self.headdim, self.dstate)
        if self._fstate_shape != shape:
            state_torch_dtype = cutlass_torch.dtype(self._state_dtype)
            # Allocate contiguous (B, EH, D, N); kernel receives permuted view
            self._fstate_torch = torch.empty(
                *shape, dtype=state_torch_dtype, device="cuda"
            )
            self._fstate_shape = shape
        return self._fstate_torch

    def _get_or_alloc_seq_cumsum(self, num_seqs, device):
        size = num_seqs + 1
        if self._seq_cumsum_size != size:
            self._seq_cumsum_buf = torch.zeros(
                size,
                dtype=torch.int32,
                device=device,
            )
            self._seq_cumsum_size = size
        else:
            self._seq_cumsum_buf.zero_()
        return self._seq_cumsum_buf

    # -- main entry point ------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run SSD combined forward pass.

        Parameters match ``ssd_combined_fwd`` — see its docstring for details.
        """
        chunk_size = self.chunk_size

        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        nchunks = seqlen // chunk_size

        assert seqlen % chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
        )

        # A is always fp32
        assert A.dtype == torch.float32, f"A must be float32, got {A.dtype}"

        # Validate varlen arguments
        if seq_idx is not None:
            assert seq_idx.shape == (batch, seqlen), (
                f"seq_idx shape {seq_idx.shape} doesn't match "
                f"(batch={batch}, seqlen={seqlen})"
            )
            assert seq_idx.dtype in (torch.int32, torch.int64), (
                f"seq_idx must be int32 or int64, got {seq_idx.dtype}"
            )
        if chunk_indices is not None:
            assert chunk_indices.dim() == 1, (
                f"chunk_indices must be 1D, got {chunk_indices.dim()}D"
            )
            assert chunk_indices.dtype == torch.int32, (
                f"chunk_indices must be int32, got {chunk_indices.dtype}"
            )
        if chunk_offsets is not None:
            assert chunk_offsets.dim() == 1, (
                f"chunk_offsets must be 1D, got {chunk_offsets.dim()}D"
            )
            assert chunk_offsets.dtype == torch.int32, (
                f"chunk_offsets must be int32, got {chunk_offsets.dtype}"
            )
        if chunk_indices is not None and chunk_offsets is not None:
            assert chunk_indices.shape == chunk_offsets.shape, (
                f"chunk_indices and chunk_offsets must have the same shape, "
                f"got {chunk_indices.shape} vs {chunk_offsets.shape}"
            )

        if out is not None:
            assert out.shape == (batch, nheads, headdim, nchunks, chunk_size), (
                f"out shape {out.shape} doesn't match "
                f"expected ({batch}, {nheads}, {headdim}, {nchunks}, {chunk_size})"
            )
            assert out.dtype == x.dtype, (
                f"out dtype {out.dtype} doesn't match x dtype {x.dtype}"
            )
            assert out.is_contiguous(), (
                "out must be contiguous in (B, EH, D, C, L) layout"
            )

        # Triton kernel outputs dt_processed directly in io_dtype (bf16),
        # avoiding a separate float32->bf16 copy.
        dA_cumsum, dt_processed = chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            dt_out_dtype=self._io_torch_dtype,
        )

        if out is None:
            out = torch.empty(
                batch,
                nheads,
                headdim,
                nchunks,
                chunk_size,
                dtype=x.dtype,
                device=x.device,
            )

        io_torch_dtype = self._io_torch_dtype

        # x: (B, seqlen, EH, D) -> (D, L, C, EH, B)
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        assert x.dtype == io_torch_dtype, (
            f"x dtype {x.dtype} doesn't match {io_torch_dtype}"
        )
        x_permuted = x_reshaped.permute(4, 2, 1, 3, 0)

        # z: same layout as x
        z_permuted = None
        if z is not None:
            z_reshaped = z.reshape(batch, nchunks, chunk_size, nheads, headdim)
            assert z.dtype == io_torch_dtype, (
                f"z dtype {z.dtype} doesn't match {io_torch_dtype}"
            )
            z_permuted = z_reshaped.permute(4, 2, 1, 3, 0)

        # dt: already in io_dtype from cumsum kernel
        delta_permuted = dt_processed.permute(3, 2, 1, 0)  # (L, C, EH, B)

        # dA_cumsum: (B, EH, C, L) -> (L, C, EH, B)
        assert dA_cumsum.dtype == cutlass_torch.dtype(self._cumsum_dtype), (
            f"dA_cumsum dtype {dA_cumsum.dtype} doesn't match "
            f"cumsum_dtype {self._cumsum_dtype}"
        )
        cumsum_delta_permuted = dA_cumsum.permute(3, 2, 1, 0)

        # B: (B, seqlen, G, N) -> (L, N, C, G, B)
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert B.dtype == io_torch_dtype, (
            f"B dtype {B.dtype} doesn't match {io_torch_dtype}"
        )
        b_permuted = B_reshaped.permute(2, 4, 1, 3, 0)

        # C: (B, seqlen, G, N) -> (L, N, C, G, B)
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert C.dtype == io_torch_dtype, (
            f"C dtype {C.dtype} doesn't match {io_torch_dtype}"
        )
        c_permuted = C_reshaped.permute(2, 4, 1, 3, 0)

        # D tensor: reshape to match kernel layout
        d_tensor = None
        if self._has_d and D is not None:
            assert D.dtype == io_torch_dtype, (
                f"D dtype {D.dtype} doesn't match io_dtype {io_torch_dtype}"
            )
            if self._d_has_hdim:
                if D.dim() == 1:
                    # (nheads,) -> (headdim, nheads): broadcast must be materialized
                    # (TVM-FFI enforces stride_order=(0,1), so dim 0 must be stride 1)
                    # it's a copy, but a small one.
                    # afaik no inference framework really hits this case
                    d_tensor = D.unsqueeze(1).expand(-1, headdim).contiguous().t()
                else:
                    # (nheads, headdim) -> (headdim, nheads): view, no copy
                    d_tensor = D.t()
            else:
                if D.dim() == 2:
                    d_tensor = D[:, 0].unsqueeze(0)  # (1, nheads)
                else:
                    d_tensor = D.unsqueeze(0)  # (1, nheads)

        # init_states: (B, EH, D, N) -> (N, D, EH, B)
        init_states_permuted = None
        if self._has_init_states and initial_states is not None:
            assert initial_states.dtype == self._state_torch_dtype, (
                f"init_states dtype {initial_states.dtype} doesn't match "
                f"state_dtype {self._state_torch_dtype}"
            )
            init_states_permuted = initial_states.permute(3, 2, 1, 0)

        # out: (B, EH, D, C, L) -> (L, D, C, EH, B)
        assert out.dtype == io_torch_dtype, (
            f"out dtype {out.dtype} doesn't match {io_torch_dtype}"
        )
        y_permuted = out.permute(4, 2, 3, 1, 0)

        # fstate: allocate (B, EH, D, N), pass permuted (N, D, EH, B) to kernel
        fstate_batch = initial_states.shape[0] if initial_states is not None else batch
        fstate_torch = self._get_or_alloc_fstate(fstate_batch)
        fstate_permuted = fstate_torch.permute(3, 2, 1, 0)

        # Varlen metadata: seq_chunk_cumsum is output buffer for Triton kernel, passed as input to CUTLASS kernel
        seq_chunk_cumsum = None
        num_seqs = 0

        if (
            seq_idx is not None
            and chunk_indices is not None
            and chunk_offsets is not None
        ):
            if initial_states is None:
                raise ValueError(
                    "initial_states must be provided in varlen mode (when seq_idx, "
                    "chunk_indices, and chunk_offsets are given) to determine num_seqs"
                )
            num_seqs = initial_states.shape[0]
            num_logical_chunks_local = len(chunk_indices)
            seq_chunk_cumsum = self._get_or_alloc_seq_cumsum(
                num_seqs,
                seq_idx.device,
            )
            _get_seq_chunk_cumsum_module().seq_chunk_cumsum(
                seq_idx,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                None,  # tile_state: let kernel allocate internally
                chunk_size,
                num_logical_chunks_local,
                num_seqs,
            )

        num_logical_chunks = (
            len(chunk_indices) if chunk_indices is not None else nchunks
        )

        self._compiled_kernel(
            x_permuted,
            cumsum_delta_permuted,
            delta_permuted,
            b_permuted,
            c_permuted,
            y_permuted,
            init_states_permuted,
            fstate_permuted,
            d_tensor,
            z_permuted,
            seq_idx,
            chunk_indices,
            chunk_offsets,
            seq_chunk_cumsum,
            Int32(num_logical_chunks),
            Int32(num_seqs),
        )

        out_view = out.permute(0, 3, 4, 1, 2).reshape(
            batch,
            seqlen,
            nheads,
            headdim,
        )
        # fstate_torch is (B, EH, D, N) — already in the expected return layout
        fstate_out = fstate_torch if return_final_states else None
        return out_view, fstate_out
