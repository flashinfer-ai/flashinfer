"""``grouped_gemm_nt_masked``'s compiled-kernel API on SM8x: the SM100 kernel's problem check
and tensor contract, the GEMM on the SM8x MXFP8 or FP4 path, then the kernel's epilogue in
PyTorch; rows from ``masked_m[e]`` on are left unwritten."""

from typing import Callable, Optional, Tuple

import cutlass
import torch

from ..cute_dsl.utils import cutlass_to_torch_dtype
from ..tllm_enums import ActivationType
from .kernels.grouped_gemm_masked_blackwell import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from .sm8x_fp4_dense import _pad
from .sm8x_fp8_dense import _OUT, _tuned_mx8, _tuned_rows_fp4

_SAT = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_NAME = "grouped_gemm_nt_masked_sm8x"


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _at(t: torch.Tensor, dtype: torch.dtype, shape: Tuple[int, ...]) -> torch.Tensor:
    """What the kernel reads of ``t``: ``shape`` contiguous ``dtype`` from its first element
    (the compiled kernel takes only the data pointers)."""
    offset = t.storage_offset() * t.element_size() // dtype.itemsize
    return torch.empty(0, dtype=dtype, device=t.device).set_(
        t.untyped_storage(), offset, shape
    )


def _unstack(sf: torch.Tensor, batches: int, rows: int, nka: int) -> torch.Tensor:
    """``_restack``'s inverse: the 128x4-swizzled scales of ``batches`` operands, each at a
    128-row atom, as those of their ``batches * rows`` rows stacked."""
    p = _pad(rows, 128)
    t = torch.arange(batches * rows, device=sf.device)
    j = torch.arange(nka * 4, device=sf.device)
    jo = j // 4 * 512 + j % 4

    def off(x: torch.Tensor) -> torch.Tensor:
        return x // 128 * nka * 512 + x % 32 * 16 + x % 128 // 32 * 4

    out = torch.zeros(
        _pad(batches * rows, 128) * nka * 4, dtype=torch.uint8, device=sf.device
    )
    out[off(t)[:, None] + jo] = sf[off(t // rows * p + t % rows)[:, None] + jo]
    return out


def _masked_store(dst: torch.Tensor, val: torch.Tensor, keep: torch.Tensor) -> None:
    if dst.element_size() == 1:
        dst, val = dst.view(torch.uint8), val.view(torch.uint8)
    dst.copy_(torch.where(keep, val, dst))


def sm8x_masked_gemm_api(
    m: int,
    n: int,
    k: int,
    l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype,
    sf_dtype,
    c_dtype,
    alpha_dtype,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    sm_version: str,
    num_ranks: int,
    enable_dst_signals: bool,
    enable_barrier_flag: bool,
    is_combine_fusion: bool,
    is_swap_ab: bool,
    activation_type: Optional[int],
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    enable_alpha_up: bool,
    enable_a_per_token_scale: bool,
) -> Callable[..., torch.Tensor]:
    """``get_cute_dsl_compiled_masked_gemm_kernel`` on SM8x: the problem rejected as the SM100
    compile rejects it, else its ``tensor_api`` on the SM8x GEMMs."""
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype, sf_dtype, sf_vec_size, c_dtype, mma_tiler_mn, cluster_shape_mn,
        m, n, k, l, a_major, b_major, c_major,
    ):  # fmt: skip
        raise TypeError(
            f"MaskedBatchedMatmulCuteDSL: Unsupported with {ab_dtype}, {sf_dtype}, "
            f"{sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, "
            f"{k}, {l}, {a_major}, {b_major}, {c_major}"
        )
    gated = activation_type is not None
    if activation_type is not None:
        if int(activation_type) != ActivationType.Swiglu.value:
            raise NotImplementedError(
                f"Unsupported fused activation {activation_type!r}; only "
                "ActivationType.Swiglu is supported"
            )
        if mma_tiler_mn != (128, 128) and mma_tiler_mn != (128, 256):
            raise ValueError(
                "Fused gated activation requires mma_tiler_mn (128, 128) or "
                f"(128, 256), got {mma_tiler_mn}"
            )
        if is_combine_fusion or is_swap_ab:
            raise ValueError(
                "Fused gated activation does not support combine fusion or swap_ab"
            )
    quantize_output = c_dtype == cutlass.Float4E2M1FN
    if quantize_output and (is_combine_fusion or is_swap_ab):
        raise ValueError("NVFP4 output does not support combine fusion or swap_ab")
    if is_combine_fusion:
        raise NotImplementedError(
            "Masked grouped GEMM combine fusion stores through multimem/multicast, "
            "which SM8x lacks"
        )
    if enable_dst_signals:
        assert l <= 8, "need to be packable into a u64"

    fp4 = ab_dtype == cutlass.Float4E2M1FN
    ab_t = torch.uint8 if fp4 else cutlass_to_torch_dtype(ab_dtype)
    c_t = cutlass_to_torch_dtype(c_dtype)
    alpha_t = (
        torch.float32 if alpha_dtype is None else cutlass_to_torch_dtype(alpha_dtype)
    )
    c_n = n // 2 if gated else n
    kb, rk = (k // 2 if fp4 else k), _cdiv(_cdiv(k, sf_vec_size), 4)
    big, small = (n, m) if is_swap_ab else (m, n)  # the rows masked (tokens), weights'
    cm, cn = cluster_shape_mn
    cta_m = mma_tiler_mn[0] // (2 if mma_tiler_mn[0] == 256 else 1)
    cta_n = mma_tiler_mn[1] // (2 if gated else 1)
    tiles = l * _cdiv(_cdiv(m, cta_m), cm) * _cdiv(_cdiv(c_n, cta_n), cn)

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        masked_m_tensor_gpu: torch.Tensor,
        dst_signals_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        topk_weights_tensor_gpu: Optional[torch.Tensor] = None,
        idx_src_info_tensor_gpu: Optional[torch.Tensor] = None,
        rank_src_info_tensor_gpu: Optional[torch.Tensor] = None,
        out_ptrs_tensor_gpu: Optional[torch.Tensor] = None,
        barrier_flag_local_tensor_gpu: Optional[torch.Tensor] = None,
        barrier_flag_multicast_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_up_tensor_gpu: Optional[torch.Tensor] = None,
        a_per_token_scale_tensor_gpu: Optional[torch.Tensor] = None,
        out_scale_tensor_gpu: Optional[torch.Tensor] = None,
        out_global_scale_tensor_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if c_tensor_gpu is None:
            if quantize_output:
                raise ValueError(
                    "NVFP4 output requires c_tensor_gpu, out_scale_tensor_gpu "
                    "and out_global_scale_tensor_gpu"
                )
            c_tensor_gpu = torch.empty((l, m, c_n), dtype=c_t, device="cuda")
        assert enable_dst_signals == (dst_signals_tensor_gpu is not None)
        assert enable_barrier_flag == (barrier_flag_local_tensor_gpu is not None)
        assert enable_alpha_up == (alpha_up_tensor_gpu is not None)
        assert enable_a_per_token_scale == (a_per_token_scale_tensor_gpu is not None)
        assert quantize_output == (out_scale_tensor_gpu is not None)
        assert quantize_output == (out_global_scale_tensor_gpu is not None)
        dev = a_tensor_gpu.device
        if l and big and small and k:
            _run(
                a_tensor_gpu, b_tensor_gpu, sfa_tensor_gpu, sfb_tensor_gpu,
                _at(masked_m_tensor_gpu, torch.int32, (l,)), c_tensor_gpu,
                alpha_tensor_gpu, alpha_up_tensor_gpu, a_per_token_scale_tensor_gpu,
                out_scale_tensor_gpu, out_global_scale_tensor_gpu, dev,
            )  # fmt: skip
        if dst_signals_tensor_gpu is not None:  # each CTA of the persistent grid adds 1
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
            ctas = min(sms // (cm * cn), sm_count, tiles) * cm * cn
            _at(dst_signals_tensor_gpu, torch.int32, (l,)).add_(ctas)
        return c_tensor_gpu

    def _run(a, b, sfa, sfb, mm, c, alpha, alpha_up, token_scale, out_scale, gsf, dev):
        A = _at(a, ab_t, (l, m, kb) if a_major == "k" else (l, kb, m))
        B = _at(b, ab_t, (l, n, kb) if b_major == "k" else (l, kb, n))
        A = A if a_major == "k" else A.transpose(1, 2).contiguous()
        B = B if b_major == "k" else B.transpose(1, 2).contiguous()
        sa = _at(sfa, torch.uint8, (l * _pad(m, 128) * rk * 4,))
        sb = _at(sfb, torch.uint8, (l * _pad(n, 128) * rk * 4,))
        (tok, s_tok), (wt, s_wt) = (
            ((B, sb), (A, sa)) if is_swap_ab else ((A, sa), (B, sb))
        )
        if quantize_output:
            out = _at(c, torch.uint8, (l, m, c_n // 2))
        else:
            out = _at(c, c_t, (l, m, c_n) if c_major == "n" else (l, c_n, m))
            out = out if (c_major == "n") != is_swap_ab else out.transpose(1, 2)
        rows = torch.arange(big, device=dev)
        keep = (rows[None, :] < mm.view(l, 1))[..., None]
        al = None if alpha is None else _at(alpha, alpha_t, (l,)).float()

        def gemm(dst: torch.Tensor, scale: Optional[torch.Tensor]) -> None:
            """``dst`` (masked rows) ``= scale (tok @ wt^T)``, ``scale`` FP4 only."""
            if not fp4:
                st = _unstack(s_tok, l, big, rk) if big % 128 and l > 1 else s_tok
                sw = _unstack(s_wt, l, small, rk) if small % 128 and l > 1 else s_wt
                _tuned_mx8(_NAME, True, [tok, wt, st, sw, mm, dst])
                return
            fmt = 0 if sf_dtype == cutlass.Float8E4M3FN else 5
            x, w = tok.reshape(l * big, kb), wt
            if fmt and sf_vec_size == 16:  # 16-element blocks as 32-element MXFP4 ones
                x = torch.cat([x.view(-1, 8), torch.zeros_like(x).view(-1, 8)], 1)
                w = torch.cat([w.reshape(-1, 8), torch.zeros_like(w).view(-1, 8)], 1)
                x, w = x.view(l * big, 2 * kb), w.view(l, small, 2 * kb)
            st = (
                _unstack(s_tok, l, big, rk)
                if not fmt and big % 128 and l > 1
                else s_tok
            )
            starts = torch.arange(l, device=dev) * big
            asf_off = torch.arange(l, dtype=torch.int32, device=dev) * (
                (_pad(big, 128) - big) if fmt else 0
            )
            o2 = dst if dst.is_contiguous() else dst.contiguous()
            _tuned_rows_fp4(
                _NAME, fmt,
                [x, w, st, s_wt, starts, starts + mm.long().clamp(0, big),
                 torch.ones(l, device=dev) if scale is None else scale, asf_off,
                 o2.view(l * big, small)],
            )  # fmt: skip
            if o2 is not dst:
                dst.copy_(o2)

        def plain(dt: torch.dtype) -> torch.Tensor:
            """The plain epilogue's ``alpha (tok @ wt^T)`` as ``dt``."""
            if fp4 or al is None:
                v = torch.empty((l, big, small), dtype=dt, device=dev)
                gemm(v, al)
                return v
            v = torch.empty((l, big, small), dtype=torch.float32, device=dev)
            gemm(v, None)
            return (v * al.view(l, 1, 1)).to(dt)

        if not gated:
            if c_t in _OUT:
                if fp4 or al is None:
                    gemm(out, al)
                else:
                    _masked_store(out, plain(c_t), keep)
                return
            v = plain(torch.bfloat16 if quantize_output else torch.float32)
        else:
            acc = torch.empty((l, m, n), dtype=torch.float32, device=dev)
            gemm(acc, None)
            blocks = acc.view(l, m, n // 128, 2, 64)
            up = blocks[:, :, :, 0].reshape(l, m, c_n)
            gate = blocks[:, :, :, 1].reshape(l, m, c_n)
            gs = 1.0 if al is None else al.view(l, 1, 1)
            us = gs if alpha_up is None else _at(alpha_up, torch.float32, (l, 1, 1))
            if token_scale is not None:
                ts = _at(token_scale, torch.float32, (l, m, 1))
                gs, us = gs * ts, us * ts
            gate = (gate * gs).clamp(max=swiglu_limit)
            up = (up * us).clamp(-swiglu_limit, swiglu_limit)
            v = gate * torch.sigmoid(swiglu_alpha * gate) * (up + swiglu_beta)
            if c_t in _OUT:
                _masked_store(out, v.to(c_t), keep)
                return
            if quantize_output:
                v = v.to(torch.bfloat16)
        if not quantize_output:  # FP8, saturating as the kernel's conversion
            _masked_store(out, v.clamp(-_SAT[c_t], _SAT[c_t]).to(c_t), keep)
            return
        from ..quantization.fp4_quantization import scaled_fp4_grouped_quantize

        q, s = scaled_fp4_grouped_quantize(
            v.contiguous(), mm, _at(gsf, torch.float32, (l,)), nvfp4_4over6=None
        )
        _masked_store(out, q.permute(2, 0, 1).view(torch.uint8), keep)
        rm, rkc = _cdiv(m, 128), _cdiv(c_n, 64)
        osf = _at(out_scale, torch.uint8, (l, rm, rkc, 32, 4, 4))
        row = (  # (rm, 32, 4): the 128x4 atom's row rmi * 128 + r4 * 32 + r32
            torch.arange(rm, device=dev).view(rm, 1, 1) * 128
            + torch.arange(32, device=dev).view(32, 1)
            + torch.arange(4, device=dev) * 32
        )
        osf.copy_(
            torch.where(
                row.view(1, rm, 1, 32, 4, 1) < mm.view(l, 1, 1, 1, 1, 1),
                s.permute(5, 2, 4, 0, 1, 3).view(torch.uint8),
                osf,
            )
        )

    return tensor_api
