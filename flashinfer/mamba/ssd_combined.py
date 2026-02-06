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

import torch

from ..api_logging import flashinfer_api
from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd


def _import_cutlass_modules():
    """Import CUTLASS modules (only when needed, as they require SM100)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    from .ssd_kernel import SSDKernel

    return {
        "cuda": cuda,
        "cutlass": cutlass,
        "cute": cute,
        "cutlass_torch": cutlass_torch,
        "from_dlpack": from_dlpack,
        "SSDKernel": SSDKernel,
    }


def _create_cutlass_tensor(
    shape, permute_order, dtype, dynamic_modes, cutlass_torch, from_dlpack
):
    """
    Create a tensor using the exact logic from mamba2_ssd.py to ensure compatibility.

    Args:
        shape: Base shape of the tensor (before permutation)
        permute_order: Order to permute dimensions
        dtype: CUTLASS dtype
        dynamic_modes: List of modes to mark as dynamic
        cutlass_torch: cutlass.torch module
        from_dlpack: cutlass.cute.runtime.from_dlpack function

    Returns:
        (cute_tensor, torch_tensor): The CuTe tensor wrapper and the underlying PyTorch tensor on GPU
    """
    # Create a dummy CPU tensor with the base layout to establish the permutation pattern
    base_tensor = torch.empty(*shape, dtype=torch.float32)
    permuted_tensor = base_tensor.permute(permute_order)

    # Move to GPU with target dtype - this creates the specific layout CUTLASS expects
    torch_dtype = cutlass_torch.dtype(dtype)
    dst_tensor = permuted_tensor.to(torch_dtype).cuda()

    # Create CuTe tensor
    cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
    for mode in dynamic_modes:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=mode, stride_order=dst_tensor.dim_order()
        )

    return cute_tensor, dst_tensor


class _SSDKernel:
    """
    Internal wrapper around CuTe DSL SSD kernel.

    Handles tensor layout conversions between PyTorch and CUTLASS formats.
    """

    def __init__(
        self,
        chunk_size: int,
        headdim: int,
        dstate: int,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_init_states: bool = False,
        io_dtype=None,
        cumsum_dtype=None,
        acc_dtype=None,
    ):
        """
        Initialize the kernel.

        Args:
            chunk_size: L - size of each chunk
            headdim: D - head dimension
            dstate: N - state dimension
            has_d: Whether to fuse D scaling (Y += X*D)
            d_has_hdim: If True, D is (headdim, nheads), else (1, nheads)
            has_init_states: Whether initial states are provided
            io_dtype: Input/output dtype (default: cutlass.BFloat16)
            cumsum_dtype: Cumsum intermediate dtype (default: cutlass.Float32)
            acc_dtype: Accumulator dtype (default: cutlass.Float32)
        """
        self.modules = _import_cutlass_modules()
        cutlass = self.modules["cutlass"]

        self.chunk_size = chunk_size
        self.headdim = headdim
        self.dstate = dstate
        self.has_d = has_d
        self.d_has_hdim = d_has_hdim
        self.has_init_states = has_init_states

        self.io_dtype = io_dtype or cutlass.BFloat16
        self.cumsum_dtype = cumsum_dtype or cutlass.Float32
        self.acc_dtype = acc_dtype or cutlass.Float32

        # Create the kernel
        SSDKernel = self.modules["SSDKernel"]
        self.kernel = SSDKernel(
            self.io_dtype,
            self.cumsum_dtype,
            self.acc_dtype,
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_init_states,
        )

        self._compiled_kernel = None

    def run(
        self,
        x: torch.Tensor,
        dA_cumsum: torch.Tensor,
        dt_processed: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        init_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the SSD kernel with preprocessed inputs.

        Args:
            x: (batch, seqlen, nheads, headdim)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            dt_processed: (batch, nheads, nchunks, chunk_size)
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            D: Optional (nheads, headdim) or (nheads,)
            init_states: Optional (batch, nheads, headdim, dstate)

        Returns:
            out: (batch, seqlen, nheads, headdim)
            final_states: (batch, nheads, headdim, dstate)
        """
        cutlass = self.modules["cutlass"]
        cute = self.modules["cute"]
        cutlass_torch = self.modules["cutlass_torch"]
        from_dlpack = self.modules["from_dlpack"]

        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        chunk_size = self.chunk_size
        nchunks = seqlen // chunk_size

        # Convert tensors to CUTLASS layout
        # x: Triton (batch, seqlen, nheads, headdim) -> CUTLASS (headdim, chunk_size, nchunks, nheads, batch)
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        x_tensor, x_dst = _create_cutlass_tensor(
            [batch, nheads, headdim, nchunks, chunk_size],
            [2, 4, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
            cutlass_torch,
            from_dlpack,
        )
        x_dst.copy_(x_reshaped.permute(4, 2, 1, 3, 0).to(x_dst.dtype))

        # delta (dt_processed): (batch, nheads, nchunks, chunk_size) -> (chunk_size, nchunks, nheads, batch)
        # Need dtype conversion from float32 to bf16, but keep the non-contiguous permuted layout
        # so that mode 0 (L) has stride 1 as the kernel expects
        io_torch_dtype = cutlass_torch.dtype(self.io_dtype)
        # First convert dtype (creates contiguous copy in original layout), then permute (zero-copy view)
        dt_permuted = dt_processed.to(io_torch_dtype).permute(
            3, 2, 1, 0
        )  # (L, C, EH, B)
        delta_tensor = from_dlpack(dt_permuted, assumed_align=16)
        # Mark dynamic modes: C=1, EH=2, B=3 (L=0 is static)
        for mode in [1, 2, 3]:
            delta_tensor = delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=dt_permuted.dim_order()
            )

        assert dA_cumsum.dtype == cutlass_torch.dtype(self.cumsum_dtype), (
            f"dA_cumsum dtype {dA_cumsum.dtype} doesn't match cumsum_dtype {self.cumsum_dtype}"
        )
        # cumsum_delta (dA_cumsum): same layout as delta
        # Zero-copy: permute is just a view, from_dlpack sees modes in permuted order
        cumsum_permuted = dA_cumsum.permute(
            3, 2, 1, 0
        )  # (L, C, EH, B) - zero-copy view
        cumsum_delta_tensor = from_dlpack(cumsum_permuted, assumed_align=16)
        # Mark dynamic modes: C=1, EH=2, B=3 (L=0 is static)
        for mode in [1, 2, 3]:
            cumsum_delta_tensor = cumsum_delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=cumsum_permuted.dim_order()
            )

        # B: Triton (batch, seqlen, ngroups, dstate) -> CUTLASS (chunk_size, dstate, nchunks, ngroups, batch)
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        b_tensor, b_dst = _create_cutlass_tensor(
            [batch, ngroups, dstate, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
            cutlass_torch,
            from_dlpack,
        )
        b_dst.copy_(B_reshaped.permute(2, 4, 1, 3, 0).to(b_dst.dtype))

        # C: same layout as B
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        c_tensor, c_dst = _create_cutlass_tensor(
            [batch, ngroups, dstate, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
            cutlass_torch,
            from_dlpack,
        )
        c_dst.copy_(C_reshaped.permute(2, 4, 1, 3, 0).to(c_dst.dtype))

        # D: (nheads,) -> CUTLASS (1, nheads) or (headdim, nheads)
        if self.has_d and D is not None:
            if self.d_has_hdim:
                # D is (nheads, headdim) -> (headdim, nheads)
                if D.dim() == 1:
                    D = D.unsqueeze(1).expand(-1, headdim)
                d_tensor, d_dst = _create_cutlass_tensor(
                    [nheads, headdim],
                    [1, 0],
                    self.io_dtype,
                    [1],
                    cutlass_torch,
                    from_dlpack,
                )
                d_dst.copy_(D.t().to(d_dst.dtype))
            else:
                # D is (nheads,) -> (1, nheads)
                if D.dim() == 2:
                    D = D[:, 0]
                d_tensor, d_dst = _create_cutlass_tensor(
                    [nheads, 1],
                    [1, 0],
                    self.io_dtype,
                    [1],
                    cutlass_torch,
                    from_dlpack,
                )
                d_dst.copy_(D.unsqueeze(0).to(d_dst.dtype))
        else:
            d_tensor = None

        if self.has_init_states and init_states is not None:
            # init_states: (batch, nheads, headdim, dstate) -> (headdim, dstate, nheads, batch)
            # Kernel expects: (D, N, EH, B) = (headdim, dstate, nheads, batch)
            init_states_reshaped = init_states.permute(2, 3, 1, 0)
            init_states_tensor, init_states_dst = _create_cutlass_tensor(
                [batch, nheads, headdim, dstate],
                [2, 3, 1, 0],
                self.io_dtype,
                [2, 3],
                cutlass_torch,
                from_dlpack,
            )
            init_states_dst.copy_(init_states_reshaped.to(init_states_dst.dtype))
        else:
            init_states_tensor = None

        # Output tensors
        # y: (chunk_size, headdim, nchunks, nheads, batch)
        y_tensor, y_cutlass = _create_cutlass_tensor(
            [batch, nheads, headdim, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
            cutlass_torch,
            from_dlpack,
        )

        # fstate: (headdim, dstate, nheads, batch)
        fstate_tensor, fstate_cutlass = _create_cutlass_tensor(
            [batch, nheads, headdim, dstate],
            [2, 3, 1, 0],
            self.io_dtype,
            [2, 3],
            cutlass_torch,
            from_dlpack,
        )

        # Get max active clusters
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)

        stream = cutlass.cuda.default_stream()

        # Compile kernel if not already done
        if self._compiled_kernel is None:
            self._compiled_kernel = cute.compile(
                self.kernel,
                x_tensor,
                cumsum_delta_tensor,
                delta_tensor,
                b_tensor,
                c_tensor,
                y_tensor,
                init_states_tensor,
                fstate_tensor,
                d_tensor,
                max_active_clusters,
                stream,
            )

        # Run kernel
        self._compiled_kernel(
            x_tensor,
            cumsum_delta_tensor,
            delta_tensor,
            b_tensor,
            c_tensor,
            y_tensor,
            init_states_tensor,
            fstate_tensor,
            d_tensor,
            stream,
        )

        # Convert outputs back to Triton layout
        # y_cutlass is (L, D, C, EH, B)
        # We need to map it back to (batch, seqlen, nheads, headdim)
        # Permute (L, D, C, EH, B) -> (B, C, L, EH, D)
        y_permuted = y_cutlass.permute(4, 2, 0, 3, 1)
        y_out = y_permuted.reshape(batch, seqlen, nheads, headdim)

        # fstate_cutlass is (D, N, EH, B)
        # We need (batch, nheads, headdim, dstate)
        # Permute (D, N, EH, B) -> (B, EH, D, N)
        fstate_out = fstate_cutlass.permute(3, 2, 0, 1).contiguous()

        return y_out, fstate_out


@functools.cache
def _get_ssd_kernel(
    chunk_size: int,
    headdim: int,
    dstate: int,
    has_d: bool,
    d_has_hdim: bool,
    has_init_states: bool = False,
) -> _SSDKernel:
    """Get cached SSD kernel."""
    return _SSDKernel(
        chunk_size=chunk_size,
        headdim=headdim,
        dstate=dstate,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_init_states=has_init_states,
    )


@flashinfer_api
def ssd_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSD (Structured State-Space Duality) combined forward pass for Mamba2.

    This function combines:
    1. Triton kernel for cumsum computation (_chunk_cumsum_fwd)
    2. CuTe DSL kernel for the main SSD computation (Blackwell optimized)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (batch, seqlen, nheads, headdim)
    dt : torch.Tensor
        Delta time tensor with shape (batch, seqlen, nheads)
    A : torch.Tensor
        A matrix with shape (nheads,) - should be negative for stability
    B : torch.Tensor
        B matrix with shape (batch, seqlen, ngroups, dstate)
    C : torch.Tensor
        C matrix with shape (batch, seqlen, ngroups, dstate)
    chunk_size : int
        Size of each chunk for processing (must divide seqlen)
    D : torch.Tensor, optional
        D vector with shape (nheads,) or (nheads, headdim)
    dt_bias : torch.Tensor, optional
        Optional dt bias with shape (nheads,)
    dt_softplus : bool
        Whether to apply softplus to dt
    dt_limit : tuple
        (min, max) limits for dt values
    initial_states: torch.Tensor, optional
        Optional (batch, nheads, headdim, dstate)
    Returns
    -------
    out : torch.Tensor
        Output tensor with shape (batch, seqlen, nheads, headdim)
    final_states : torch.Tensor
        Final state tensor with shape (batch, nheads, headdim, dstate)

    Notes
    -----
    - Requires SM100+ (Blackwell) for the CuTe DSL SSD kernel.
    - The kernel is hardcoded for N=128 (dstate), L=128 (chunk_size), D=64 (headdim).
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape

    assert seqlen % chunk_size == 0, (
        f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    )

    # Step 1: Compute cumsum using Triton kernel
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    # dt_processed: (batch, nheads, nchunks, chunk_size) - after softplus/bias
    dA_cumsum, dt_processed = chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # Step 2: Run SSD kernel
    has_d = D is not None
    d_has_hdim = has_d and D.dim() == 2
    has_init_state = True if initial_states is not None else False

    kernel = _get_ssd_kernel(
        chunk_size=chunk_size,
        headdim=headdim,
        dstate=dstate,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_init_states=has_init_state,
    )

    out, final_states = kernel.run(
        x=x,
        dA_cumsum=dA_cumsum,
        dt_processed=dt_processed,
        B=B,
        C=C,
        D=D,
        init_states=initial_states,
    )

    return out, final_states
