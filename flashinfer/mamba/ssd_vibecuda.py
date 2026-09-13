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

VibeCUDA backend
================

Hand-written CUDA implementation of the Mamba2 SSD combined forward pass
(cp.async staging + mma.sync m16n8k16 bf16/fp16 with fp32 accumulation, no
cuBLAS and no CuTe-DSL).  At most two kernel launches per call:

* ``ssd_k1_kernel`` — chunk-state builder, only for the uniform (non-varlen)
  multi-chunk layout;
* ``ssd_k3_kernel`` — fused output kernel: masked-decay intra-chunk matmuls,
  inter-chunk state contribution (inline chain for varlen, global chunk-state
  chain for the uniform multi-chunk layout), D-skip, optional SiLU z gate,
  per-token output store, and inline final-state accumulation;
* ``ssd_k3l_kernel`` — lean row-split variant of the fused kernel, selected
  by shape metadata for the tiny uniform single-chunk family.

Compiled through the regular FlashInfer nvcc JIT path; see
``flashinfer/jit/mamba/vibecuda_ssd.py``.  Device code lives in
``include/flashinfer/mamba/vibecuda_ssd_combined.cuh`` with the TVM-FFI
launcher in ``csrc/vibecuda_mamba_ssd_combined.cu``.
"""

import functools
from typing import Optional, Tuple

import torch

from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module
from ..jit.mamba.vibecuda_ssd import gen_vibecuda_ssd_combined_module

_CHUNK = 128
_HEADDIM = 64
_DSTATE = 128
# Device-side table bounds (shared by the kernel header); the wrapper fails
# loudly instead of silently overflowing the varlen decode table.
_MAX_CHUNKS = 384
_MAX_SEQS = 128


@functools.cache
def _get_vibecuda_module():
    """Get the cached VibeCUDA SSD combined JIT module."""
    return gen_vibecuda_ssd_combined_module().build_and_load()


@functools.cache
def _get_seq_chunk_cumsum_module():
    """Get the cached seq_chunk_cumsum JIT module."""
    return gen_seq_chunk_cumsum_module().build_and_load()


class VibeCUDASSDCombined:
    """Mamba2 SSD combined forward pass backed by the VibeCUDA kernels.

    Mirrors :class:`CakeSSDCombined`'s constructor and ``run`` surface so it
    can be selected with ``SSDCombined(..., backend="vibecuda")``.
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
        if chunk_size != _CHUNK or headdim != _HEADDIM or dstate != _DSTATE:
            raise ValueError(
                "VibeCUDA SSDCombined requires chunk_size=128, headdim=64, "
                f"dstate=128; got chunk_size={chunk_size}, headdim={headdim}, "
                f"dstate={dstate}"
            )
        if nheads <= 0 or ngroups <= 0 or nheads % ngroups:
            raise ValueError(
                "VibeCUDA SSDCombined requires positive nheads divisible by ngroups"
            )
        if io_dtype != torch.bfloat16:
            raise ValueError(
                f"VibeCUDA SSDCombined requires io_dtype=bfloat16, got {io_dtype}"
            )
        if state_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "VibeCUDA SSDCombined requires state_dtype bfloat16 or float16, "
                f"got {state_dtype}"
            )
        if seq_idx_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "VibeCUDA SSDCombined seq_idx dtype must be int32 or int64, "
                f"got {seq_idx_dtype}"
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
        self._io_torch_dtype = io_dtype
        self._state_torch_dtype = state_dtype
        self._seq_idx_dtype = seq_idx_dtype

        self._module = _get_vibecuda_module()
        self._zero_dt_bias_cache: dict[
            tuple[int | None, torch.dtype], torch.Tensor
        ] = {}

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _contiguous(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is not None and not t.is_contiguous():
            return t.contiguous()
        return t

    def _zero_dt_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.index, dtype)
        value = self._zero_dt_bias_cache.get(key)
        if value is None:
            value = torch.zeros(self.nheads, dtype=dtype, device=device)
            self._zero_dt_bias_cache[key] = value
        return value

    @staticmethod
    def _require_same_device(
        reference: torch.Tensor,
        **tensors: Optional[torch.Tensor],
    ) -> None:
        if reference.device.type != "cuda":
            raise ValueError("VibeCUDA SSDCombined inputs must be CUDA tensors")
        for name, tensor in tensors.items():
            if tensor is not None and tensor.device != reference.device:
                raise ValueError(
                    f"{name} must be on {reference.device}, got {tensor.device}"
                )

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
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        update_seq_chunk_cumsum: bool = False,
        checkpoint_token_indices: Optional[torch.Tensor] = None,
        checkpoint_state_slots: Optional[torch.Tensor] = None,
        checkpoint_states: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run SSD combined forward pass; see ``SSDCombined.run``."""
        if x.ndim != 4:
            raise ValueError("x must be a 4D [batch, seqlen, nheads, headdim] tensor")
        batch, seqlen, nheads, headdim = x.shape
        if seqlen % _CHUNK:
            raise ValueError("seqlen must be divisible by chunk_size=128")
        if (nheads, headdim) != (self.nheads, _HEADDIM):
            raise ValueError(f"x must have shape [batch, seqlen, {self.nheads}, 64]")
        if tuple(B.shape) != (batch, seqlen, self.ngroups, _DSTATE):
            raise ValueError(f"B must have shape [batch, seqlen, {self.ngroups}, 128]")
        if C.shape != B.shape:
            raise ValueError("C must have the same shape as B")
        if x.dtype != torch.bfloat16 or B.dtype != x.dtype or C.dtype != x.dtype:
            raise ValueError("x, B, and C must be bfloat16")
        if tuple(dt.shape) != (batch, seqlen, self.nheads):
            raise ValueError(f"dt must have shape [batch, seqlen, {self.nheads}]")
        if dt.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("dt must be bfloat16 or float32")
        if tuple(A.shape) != (self.nheads,) or A.dtype != torch.float32:
            raise ValueError(f"A must have shape [{self.nheads}] and dtype float32")
        if (D is not None) != self._has_d or (z is not None) != self._has_z:
            raise ValueError("runtime D/z presence must match the constructor")
        if (initial_states is not None) != self._has_init_states:
            raise ValueError(
                "runtime initial_states presence must match the constructor"
            )

        self._require_same_device(
            x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=seq_chunk_cumsum,
            checkpoint_token_indices=checkpoint_token_indices,
            checkpoint_state_slots=checkpoint_state_slots,
            checkpoint_states=checkpoint_states,
            out=out,
        )

        has_varlen = seq_idx is not None
        if has_varlen and not self._has_varlen:
            raise ValueError(
                "seq_idx provided but VibeCUDASSDCombined was constructed with "
                "has_varlen=False"
            )
        if not has_varlen and self._has_varlen:
            raise ValueError(
                "VibeCUDASSDCombined was constructed with has_varlen=True but no "
                "seq_idx was provided"
            )
        metadata_pair = (chunk_indices, chunk_offsets)
        if any(value is not None for value in metadata_pair) and not all(
            value is not None for value in metadata_pair
        ):
            raise ValueError(
                "chunk_indices and chunk_offsets must be supplied together"
            )
        if not has_varlen and (
            any(value is not None for value in metadata_pair)
            or seq_chunk_cumsum is not None
        ):
            raise ValueError(
                "batched mode does not accept chunk metadata or seq_chunk_cumsum"
            )
        if has_varlen:
            if batch != 1:
                raise ValueError("varlen mode requires packed x with batch == 1")
            if initial_states is None:
                raise ValueError(
                    "initial_states must be provided in varlen mode to determine "
                    "num_seqs"
                )
            if seq_idx.dtype != self._seq_idx_dtype:
                raise ValueError(
                    f"seq_idx dtype {seq_idx.dtype} does not match the constructor "
                    f"dtype {self._seq_idx_dtype}"
                )
            if seq_idx.numel() != batch * seqlen:
                raise ValueError(
                    "seq_idx must have one entry per packed token "
                    f"({batch * seqlen}), got {seq_idx.numel()}"
                )
            if tuple(seq_idx.shape) != (batch, seqlen):
                raise ValueError(
                    f"seq_idx must have shape [{batch}, {seqlen}], got "
                    f"{tuple(seq_idx.shape)}"
                )
        num_seqs = (
            initial_states.shape[0]
            if has_varlen and initial_states is not None
            else batch
        )
        if initial_states is not None:
            expected_states = (num_seqs, self.nheads, _HEADDIM, _DSTATE)
            if tuple(initial_states.shape) != expected_states:
                raise ValueError(f"initial_states must have shape {expected_states}")
            if initial_states.dtype != self._state_torch_dtype:
                raise ValueError("initial_states dtype must match state_dtype")
        if num_seqs > _MAX_SEQS:
            raise ValueError(
                f"VibeCUDA SSDCombined supports at most {_MAX_SEQS} sequences, "
                f"got {num_seqs}"
            )

        nchunks = seqlen // _CHUNK
        # Safe upper bound on the total logical-chunk count: batch * nchunks for
        # the uniform layout; every varlen sequence wastes at most one partial
        # chunk, so seqlen/CS + num_seqs bounds the packed count.
        nchunk_bound = nchunks + num_seqs if has_varlen else batch * nchunks
        meta_ci = None
        meta_co = None
        nmeta = 0
        if has_varlen and chunk_indices is not None and chunk_offsets is not None:
            if (
                chunk_indices.ndim != 1
                or chunk_offsets.ndim != 1
                or chunk_indices.shape != chunk_offsets.shape
            ):
                raise ValueError(
                    "chunk_indices/chunk_offsets must be matching 1D vectors"
                )
            if chunk_indices.numel() > 0:
                if (
                    chunk_indices.dtype != torch.int32
                    or chunk_offsets.dtype != torch.int32
                ):
                    raise ValueError("chunk_indices/chunk_offsets must be int32")
                meta_ci = self._contiguous(chunk_indices)
                meta_co = self._contiguous(chunk_offsets)
                nmeta = meta_ci.numel()
                # Metadata defines the tiling exactly; the grid (and bound) is nmeta.
                nchunk_bound = nmeta
        if nchunk_bound > _MAX_CHUNKS:
            raise ValueError(
                f"VibeCUDA SSDCombined supports at most {_MAX_CHUNKS} chunk "
                f"segments, bound is {nchunk_bound}"
            )

        # Selective checkpoint capture (same contract as the Cake backend):
        # one exclusive token boundary per sequence — absolute in the packed
        # token axis for varlen, sequence-relative for the batched layout.
        # Negative token or slot disables capture for that sequence. The kernel
        # writes the part-start state at the boundary in place into the
        # caller-owned checkpoint_states rows.
        ck_args = (checkpoint_token_indices, checkpoint_state_slots, checkpoint_states)
        if any(value is not None for value in ck_args):
            if not all(value is not None for value in ck_args):
                raise ValueError(
                    "checkpoint_token_indices, checkpoint_state_slots, and "
                    "checkpoint_states must be supplied together"
                )
            if (
                checkpoint_token_indices.dtype != torch.int32
                or checkpoint_state_slots.dtype != torch.int32
            ):
                raise ValueError("checkpoint_token_indices/slots must be int32")
            if (
                checkpoint_token_indices.numel() != num_seqs
                or checkpoint_state_slots.numel() != num_seqs
            ):
                raise ValueError(
                    "checkpoint_token_indices/slots must have one entry per "
                    f"sequence ({num_seqs})"
                )
            if checkpoint_states.dim() != 4 or tuple(checkpoint_states.shape[1:]) != (
                self.nheads,
                _HEADDIM,
                _DSTATE,
            ):
                raise ValueError(
                    "checkpoint_states must have shape "
                    f"[num_checkpoints, {self.nheads}, {_HEADDIM}, {_DSTATE}]"
                )
            if checkpoint_states.dtype != self._state_torch_dtype:
                raise ValueError("checkpoint_states dtype must match state_dtype")
            if not checkpoint_states.is_contiguous():
                raise ValueError("checkpoint_states must be contiguous")
            if checkpoint_token_indices.ndim != 1 or checkpoint_state_slots.ndim != 1:
                raise ValueError("checkpoint_token_indices/slots must be 1D vectors")
            checkpoint_token_indices = self._contiguous(checkpoint_token_indices)
            checkpoint_state_slots = self._contiguous(checkpoint_state_slots)

        if dt_bias is not None:
            if tuple(dt_bias.shape) != (self.nheads,) or dt_bias.dtype != dt.dtype:
                raise ValueError(
                    f"dt_bias must have shape [{self.nheads}] and dtype "
                    f"matching dt ({dt.dtype})"
                )
        # Match the CAKE contract exactly: a provided D is (nheads,) or
        # (nheads, headdim) bfloat16. A 2D D with a per-head constructor
        # reduces to its first column (same coercion as the CAKE public runner);
        # a 1D D with a d_has_hdim constructor stays a per-head scalar because
        # no headdim dimension exists to broadcast from.
        d_c: Optional[torch.Tensor] = None
        d_mode = 0
        if D is not None:
            valid_d_shapes = ((self.nheads,), (self.nheads, _HEADDIM))
            if tuple(D.shape) not in valid_d_shapes or D.dtype != torch.bfloat16:
                raise ValueError(
                    f"D must have shape [{self.nheads}] or "
                    f"[{self.nheads}, 64] and dtype bfloat16"
                )
            if not D.is_contiguous():
                raise ValueError("D must be contiguous")
            d_c = D
            if self._d_has_hdim and D.dim() == 2:
                d_mode = 2
            else:
                # Mode 3 reads the first column with the source tensor's fixed
                # row stride. This avoids both shared cross-stream scratch and
                # a per-call allocation/copy in the timed execution path.
                d_mode = 3 if D.dim() == 2 else 1
        if z is not None and (z.shape != x.shape or z.dtype != torch.bfloat16):
            raise ValueError("z must have the same shape and dtype as x")

        x_c = self._contiguous(x)
        dt_c = self._contiguous(dt)
        B_c = self._contiguous(B)
        C_c = self._contiguous(C)
        dt_bias_c = (
            self._contiguous(dt_bias)
            if dt_bias is not None
            else self._zero_dt_bias(x.device, dt_c.dtype)
        )
        z_c = self._contiguous(z) if z is not None else None
        initial_c = self._contiguous(initial_states)
        seq_idx_c = self._contiguous(seq_idx) if has_varlen else None

        if seq_chunk_cumsum is not None and (
            tuple(seq_chunk_cumsum.shape) != (num_seqs + 1,)
            or seq_chunk_cumsum.dtype != torch.int32
        ):
            raise ValueError("seq_chunk_cumsum shape or dtype is invalid")

        # dt_limit (0.0, inf) is the "unbounded" mode: when softplus is on the
        # clamp is an identity on the (non-negative) softplus output, so the
        # kernel may skip it; with softplus off the clamp to [0, inf) is kept so
        # the semantics still match clamp(dt, dt_lo, dt_hi) exactly.
        dt_lo, dt_hi = (float(v) for v in dt_limit)
        if dt_lo == 0.0 and dt_hi == float("inf") and dt_softplus:
            unbounded = 1
        else:
            unbounded = 0

        # fp32 scratch for the uniform multi-chunk chunk-state pre-pass
        # (chunk_state then da_last, padded to at least 64 floats so the tensor
        # always has valid storage).
        need_gs = (not has_varlen) and nchunks > 1
        cs_floats = nchunk_bound * nheads * _HEADDIM * _DSTATE if need_gs else 0
        dal_floats = nchunk_bound * nheads if need_gs else 0
        workspace = torch.empty(
            cs_floats + dal_floats + 64, dtype=torch.float32, device=x.device
        )

        # y buffer: a caller-provided ``out`` uses the public SSDCombined
        # chunk-major layout (batch, nheads, headdim, nchunks, chunk); an
        # internal allocation is packed token-major (batch, seqlen, nheads,
        # headdim) and returned directly (the single kernel store covers every
        # element in both layouts).
        if out is not None:
            expected_out = (batch, self.nheads, _HEADDIM, nchunks, _CHUNK)
            if tuple(out.shape) != expected_out or out.dtype != torch.bfloat16:
                raise ValueError(
                    f"out must have shape {expected_out} and dtype bfloat16"
                )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            y_chunk_major = 1
        else:
            out = torch.empty(
                batch, seqlen, nheads, _HEADDIM, dtype=torch.bfloat16, device=x.device
            )
            y_chunk_major = 0

        final_states = torch.empty(
            num_seqs,
            nheads,
            _HEADDIM,
            _DSTATE,
            dtype=self._state_torch_dtype,
            device=x.device,
        )

        self._module.vibecuda_ssd_combined_fwd(
            x_c,
            dt_c,
            dt_bias_c,
            A,
            B_c,
            C_c,
            d_c,
            z_c,
            initial_c,
            seq_idx_c if has_varlen else None,
            meta_ci,
            meta_co,
            checkpoint_states,
            checkpoint_token_indices,
            checkpoint_state_slots,
            workspace,
            out,
            final_states,
            nchunk_bound,
            1 if dt_softplus else 0,
            float(dt_lo),
            float(dt_hi),
            unbounded,
            d_mode,
            1 if has_varlen else 0,
            y_chunk_major,
        )

        if (
            has_varlen
            and chunk_indices is not None
            and chunk_offsets is not None
            and (seq_chunk_cumsum is None or update_seq_chunk_cumsum)
        ):
            if seq_chunk_cumsum is None:
                seq_chunk_cumsum = torch.zeros(
                    num_seqs + 1, dtype=torch.int32, device=x.device
                )
            if (
                tuple(seq_chunk_cumsum.shape) != (num_seqs + 1,)
                or seq_chunk_cumsum.dtype != torch.int32
            ):
                raise ValueError("seq_chunk_cumsum shape or dtype is invalid")
            module = _get_seq_chunk_cumsum_module()
            tile_state_bytes = module.seq_chunk_cumsum_tile_state_size(num_seqs)
            tile_state = (
                torch.empty(tile_state_bytes, dtype=torch.uint8, device=x.device)
                if tile_state_bytes > 0
                else None
            )
            module.seq_chunk_cumsum(
                seq_idx_c,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                tile_state,
                _CHUNK,
                len(chunk_indices),
                num_seqs,
            )

        if y_chunk_major:
            out = out.permute(0, 3, 4, 1, 2).reshape(batch, seqlen, nheads, _HEADDIM)
        return out, final_states if return_final_states else None
