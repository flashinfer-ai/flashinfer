"""All-native SM120/121 implementation of the existing cute-dsl FP4 backend."""

from functools import cache

import torch

from ....autotuner import TunableRunner
from ....utils import get_compute_capability, get_device_index
from . import policy

_COMPILED = {}


def check_inputs(a, b, sfa, sfb, alpha, out_dtype, out):
    if a.ndim != 2 or b.ndim != 2 or b.shape[0] != a.shape[1]:
        raise ValueError("SM12x cute-dsl requires matching two-dimensional operands")
    m, n, k = a.shape[0], b.shape[1], a.shape[1] * 2
    policy.check_shape(m, n, k)
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError("SM12x cute-dsl requires packed uint8 NVFP4 operands")
    if not a.is_contiguous() or not b.T.is_contiguous():
        raise ValueError("SM12x cute-dsl requires compact A and column-major B")
    if out_dtype != torch.bfloat16:
        raise ValueError("SM12x cute-dsl currently supports BF16 output only")
    if sfa.dtype not in (torch.uint8, torch.float8_e4m3fn) or sfb.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise ValueError("SM12x cute-dsl requires E4M3 scale bytes")
    if sfa.shape != (((m + 127) // 128) * 128, k // 16) or sfb.shape != (
        k // 16,
        n,
    ):
        raise ValueError("SM12x cute-dsl requires the physical 128x4 scale layout")
    if not sfa.is_contiguous() or not sfb.T.is_contiguous():
        raise ValueError("SM12x cute-dsl requires compact scale storage")
    tensors = [a, b, sfa, sfb]
    if alpha is not None:
        if alpha.dtype != torch.float32 or alpha.numel() != 1:
            raise ValueError("SM12x cute-dsl alpha must be a GPU FP32 scalar")
        tensors.append(alpha)
    if out is not None:
        if out.dtype != out_dtype or out.shape != (m, n) or not out.is_contiguous():
            raise ValueError("SM12x cute-dsl requires compact BF16 out[M,N]")
        tensors.append(out)
    if a.device.type != "cuda" or any(t.device != a.device for t in tensors):
        raise ValueError("SM12x cute-dsl requires tensors on one CUDA device")
    if get_compute_capability(a.device) not in ((12, 0), (12, 1)):
        raise ValueError("SM12x cute-dsl requires SM120 or SM121")
    if a.data_ptr() % 32 or b.data_ptr() % 32:
        raise ValueError("SM12x cute-dsl requires 32-byte aligned operands")
    if sfa.data_ptr() % 16 or sfb.data_ptr() % 16:
        raise ValueError("SM12x cute-dsl requires 16-byte aligned scales")
    if alpha is not None and alpha.data_ptr() % 4:
        raise ValueError("SM12x cute-dsl requires 4-byte aligned alpha")
    if out is not None:
        if out.data_ptr() % 16:
            raise ValueError("SM12x cute-dsl requires 16-byte aligned output")
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for tensor in tensors[:-1]:
            tlo = tensor.data_ptr()
            thi = tlo + tensor.numel() * tensor.element_size()
            if lo < thi and tlo < hi:
                raise ValueError("SM12x cute-dsl output must not overlap an input")
    return m, n, k


def check_requirement(a, b, sfa, sfb, alpha, dtype, out, block_size, nvfp4, sf8):
    if block_size != 16 or not nvfp4 or sf8:
        raise ValueError("SM12x cute-dsl requires NVFP4 with 128x4 scales")
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        raise ValueError("SM12x cute-dsl requires CUDA 13 or newer")
    check_inputs(a, b, sfa, sfb, alpha, dtype, out)
    return True


def _compile(m, n, k, tactic):
    import cutlass
    import cutlass.cute as cute

    from ....cute_dsl.utils import get_max_active_clusters
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ... import gemm_mm_fp4_cute_dsl as helpers
    from .. import dense_blockscaled_gemm_sm120_b12x as b12x
    from . import blockscaled_gemm_dispatch, narrow, raw

    mac = get_max_active_clusters(1)
    family = tactic[0]
    if family == "raw":
        _, epi_m, epi_n, swizzle, elected, raster_m = tactic
        gemm = raw.Sm120BlockScaledGemmKernel(
            cutlass.Float32,
            16,
            (128, 128, 128),
            (epi_m, epi_n),
            swizzle_size=swizzle,
            elected_release=elected,
            raster_along_m=raster_m,
        )

        class Adapter:
            def __init__(self):
                self.gemm = gemm

            @cute.jit
            def wrapper(
                self,
                a: cute.Tensor,
                b: cute.Tensor,
                out: cute.Tensor,
                sf_m: cutlass.Int64,
                sf_n: cutlass.Int64,
                sf_k: cutlass.Int64,
                batch: cutlass.Constexpr,
                sfa: cute.Pointer,
                sfb: cute.Pointer,
                alpha: cute.Tensor,
                max_active_clusters: cutlass.Constexpr,
                stream,
                swap_ab: cutlass.Constexpr = False,
            ):
                ap = cute.recast_ptr(a.iterator, dtype=cutlass.Float4E2M1FN)
                bp = cute.recast_ptr(b.iterator, dtype=cutlass.Float4E2M1FN)
                self.gemm(
                    cute.make_tensor(
                        ap, cute.make_layout((m, k, 1), stride=(k, 1, m * k))
                    ),
                    cute.make_tensor(
                        bp, cute.make_layout((n, k, 1), stride=(k, 1, n * k))
                    ),
                    cute.make_tensor(
                        sfa, cute.make_layout((((m + 127) // 128 * 128) * k // 16,))
                    ),
                    cute.make_tensor(sfb, cute.make_layout((n * k // 16,))),
                    cute.make_tensor(
                        out.iterator, cute.make_layout((m, n, 1), stride=(n, 1, m * n))
                    ),
                    alpha,
                    max_active_clusters,
                    stream,
                )

        kernel = Adapter()
        shape_name = f"m{m}_n{n}_k{k}_"
        module = raw
    else:
        _, tile_m, tile_n, tile_k = tactic
        module = narrow if family == "narrow" else b12x
        cls = module.DenseGemmKernel
        if not cls.can_implement(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            (tile_m, tile_n),
            (1, 1),
            n,
            k,
            1,
            "k",
            "k",
            "n",
            load_path="tma",
            swap_ab=False,
            tile_k=tile_k,
        ):
            raise ValueError("SM12x cute-dsl narrow kernel rejected the tactic")
        kernel = cls(
            16,
            (tile_m, tile_n),
            (1, 1),
            mma_k=64,
            tile_k=tile_k,
            single_work_tile_per_cta=False,
            use_prefetch=False,
            enable_pdl=False,
            direct_one_m_tile_scheduler=False,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=False,
            use_m1_non_tma_sfa=False,
            load_path="tma",
            swap_ab=False,
            enable_iket=False,
        )
        # The existing helper and narrow wrapper use dynamic M/N/K and SF extents.
        shape_name = "dynamic_"

    compile_fn = helpers._make_blockscaled_gemm_compile_fn(
        kernel,
        cutlass.Uint8,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        32,
        False,
        (m + 127) // 128,
        n // 128,
        k // 64,
        1,
        mac,
    )
    name = shape_name + "_".join(
        str(int(x)) if isinstance(x, bool) else str(x) for x in tactic
    )
    return build_and_load_cute_dsl_kernel(
        f"{policy.VERSION}_{family}",
        f"{name}_pdl0_mac{mac}",
        compile_fn,
        extra_key_files=(
            __file__,
            policy.__file__,
            module.__file__,
            blockscaled_gemm_dispatch.__file__,
            helpers.__file__,
        ),
    )


class Sm12xCuTeFp4GemmRunner(TunableRunner):
    def get_cache_key_extras(self, inputs):
        a, _, sfa, sfb, alpha, dtype, _, _, _, _ = inputs
        return (
            policy.VERSION,
            get_compute_capability(a.device),
            str(a.dtype),
            str(sfa.dtype),
            str(sfb.dtype),
            str(dtype),
            alpha is None,
        )

    def validate_tactic(self, inputs, tactic):
        try:
            a, b, sfa, sfb, alpha, dtype, out, block_size, nvfp4, _ = inputs
            if block_size != 16 or not nvfp4 or out is None:
                return False
            shape = check_inputs(a, b, sfa, sfb, alpha, dtype, out)
            return policy.compatible(*shape, tactic)
        except (ValueError, TypeError, AttributeError):
            return False

    def get_valid_tactics(self, inputs, profile):
        if not self.validate_tactic(inputs, -1):
            return []
        a, b = inputs[:2]
        return list(policy.valid_tactics(a.shape[0], b.shape[1], a.shape[1] * 2))

    def _get_compiled(self, inputs, tactic):
        a, b = inputs[:2]
        m, n, k = a.shape[0], b.shape[1], a.shape[1] * 2
        shape = (m, n, k) if tactic[0] == "raw" else ()
        key = (get_device_index(a.device), tactic, shape)
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SM12x cute-dsl needs eager preparation before capture"
                )
            with torch.cuda.device(a.device):
                _COMPILED[key] = _compile(m, n, k, tactic)
        return _COMPILED[key]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        from ...gemm_mm_fp4_cute_dsl import _prepare_alpha_for_launch

        if not self.validate_tactic(inputs, tactic):
            # V2 may ask the first runner to prepare -1 even for an empty profile.
            if do_preparation and (tactic is None or tactic == -1):
                return inputs[6]
            raise ValueError("Invalid SM12x cute-dsl tactic or actual input contract")
        a, b, sfa, sfb, alpha, _, out, _, _, _ = inputs
        m, n, k = a.shape[0], b.shape[1], a.shape[1] * 2
        launch_alpha = _prepare_alpha_for_launch(alpha, a.device)
        if do_preparation:
            for choice in self.get_valid_tactics(inputs, None):
                self._get_compiled(inputs, choice)
            return out
        if tactic is None or tactic == -1:
            tactic = policy.default_tactic(m, n, k)
        compiled = self._get_compiled(inputs, tactic)
        args = (
            a,
            b.T,
            out,
            (m + 127) // 128,
            n // 128,
            k // 64,
            sfa.data_ptr(),
            sfb.data_ptr(),
            launch_alpha,
        )
        # Exported b12x wrappers retain their three optional SVDQuant arguments.
        compiled(*(args if tactic[0] == "raw" else (*args, None, None, None)))
        return out


@cache
def get_runner():
    # PDL is an optional performance hint. This initial SM12x path uses only
    # ordinary stream-ordered launches, for both values of public enable_pdl.
    return Sm12xCuTeFp4GemmRunner()
