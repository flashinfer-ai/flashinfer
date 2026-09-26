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
        ((n + 127) // 128) * 128,
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


def _compile(m, n, k, tactic, *, compute_capability=None):
    import cutlass
    import cutlass.cute as cute

    from ....cute_dsl import utils as cute_dsl_utils
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ... import gemm_mm_fp4_cute_dsl as helpers
    from . import blockscaled_gemm_dispatch, raw

    family = tactic[0]
    if family in ("independent", "independent_tma"):
        from . import independent_small, independent_tma

        small = family == "independent"
        module = independent_small if small else independent_tma
        op = (
            module.IndependentSmall(m, n, k)
            if small
            else module.IndependentMedium(m, n, k)
        )
        scalar = cutlass.Int32 if small else cutlass.Uint8
        divisor = 8 if small else 2
        operands = [
            cute.runtime.make_fake_compact_tensor(
                scalar, (rows, k // divisor), stride_order=(1, 0), assumed_align=32
            )
            for rows in (m, n)
        ]
        operands += [
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (((rows + 127) // 128) * 128 * (k // 64),),
                assumed_align=16,
            )
            for rows in (m, n)
        ]
        operands += [
            cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (1,), assumed_align=4
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16
            ),
        ]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return build_and_load_cute_dsl_kernel(
            f"{policy.VERSION}_{family}",
            f"m{m}_n{n}_k{k}",
            lambda: cute.compile(
                op.launch if small else op,
                *operands,
                stream,
                options="--enable-tvm-ffi",
            ),
            extra_key_files=(__file__, module.__file__),
        )
    mac = cute_dsl_utils.get_max_active_clusters(1)
    if family == "raw":
        if len(tactic) == 6:
            _, epi_m, epi_n, swizzle, elected, raster_m = tactic
            tile_k, internal_swap = 128, False
        elif (
            len(tactic) == 8
            and compute_capability == (12, 1)
            and policy.compatible(
                m, n, k, tactic, compute_capability=compute_capability
            )
        ):
            _, epi_m, epi_n, swizzle, elected, raster_m, tile_k, internal_swap = tactic
        else:
            raise ValueError("Invalid SM12x raw tactic")
        kernel_m, kernel_n = (n, m) if internal_swap else (m, n)
        gemm = raw.Sm120BlockScaledGemmKernel(
            cutlass.Float32,
            16,
            (128, 128, tile_k),
            (epi_m, epi_n),
            swizzle_size=swizzle,
            elected_release=elected,
            raster_along_m=raster_m,
            half_stage_wait=(
                compute_capability == (12, 1)
                and m == 8192
                and (n, k) == (5120, 17408)
                and tactic == ("raw", 64, 32, 8, False, True)
            ),
            extra_mainloop_stage=(
                compute_capability == (12, 1)
                and m in (4096, 8192)
                and (n, k) == (5120, 17408)
                and tactic == ("raw", 64, 32, 8, False, True)
            ),
        )

        register_redistribution = compute_capability == (12, 1) and (
            (
                (m, n, k) == (256, 9216, 7168)
                and tactic == ("raw", 64, 32, 2, False, True, 256, True)
            )
            or (
                (m, n, k) == (512, 8192, 2048)
                and tactic == ("raw", 64, 32, 4, False, True, 256, True)
            )
            or (
                (m, n, k) == (512, 7168, 5120)
                and tactic == ("raw", 64, 32, 8, False, True, 256, False)
            )
        )
        if register_redistribution:
            gemm.load_register_requirement = 24
            gemm.mma_register_requirement = 240

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
                        bp if internal_swap else ap,
                        cute.make_layout((kernel_m, k, 1), stride=(k, 1, kernel_m * k)),
                    ),
                    cute.make_tensor(
                        ap if internal_swap else bp,
                        cute.make_layout((kernel_n, k, 1), stride=(k, 1, kernel_n * k)),
                    ),
                    cute.make_tensor(
                        sfb if internal_swap else sfa,
                        cute.make_layout((((kernel_m + 127) // 128 * 128) * k // 16,)),
                    ),
                    cute.make_tensor(
                        sfa if internal_swap else sfb,
                        cute.make_layout((kernel_n * k // 16,)),
                    ),
                    cute.make_tensor(
                        out.iterator,
                        cute.make_layout(
                            (kernel_m, kernel_n, 1),
                            stride=(1, n, m * n) if internal_swap else (n, 1, m * n),
                        ),
                    ),
                    alpha,
                    max_active_clusters,
                    stream,
                )

        kernel = Adapter()
        shape_name = (
            f"m{m}_n{n}_k{k}_"
            f"ab5{int(gemm.extra_mainloop_stage)}_half{int(gemm.half_stage_wait)}_"
        )
        if register_redistribution:
            shape_name += "regs24_240_"
        module = raw
    compile_fn = helpers._make_blockscaled_gemm_compile_fn(
        kernel,
        cutlass.Uint8,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        32,
        False,
        (m + 127) // 128,
        (n + 127) // 128,
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
            cute_dsl_utils.__file__,
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
            return policy.compatible(
                *shape, tactic, compute_capability=get_compute_capability(a.device)
            )
        except (ValueError, TypeError, AttributeError):
            return False

    def get_valid_tactics(self, inputs, profile):
        if not self.validate_tactic(inputs, -1):
            return []
        a, b = inputs[:2]
        return list(
            policy.valid_tactics(
                a.shape[0],
                b.shape[1],
                a.shape[1] * 2,
                compute_capability=get_compute_capability(a.device),
            )
        )

    def _get_compiled(self, inputs, tactic):
        a, b = inputs[:2]
        m, n, k = a.shape[0], b.shape[1], a.shape[1] * 2
        shape = (m, n, k)
        compute_capability = get_compute_capability(a.device)
        key = (get_device_index(a.device), compute_capability, tactic, shape)
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SM12x cute-dsl needs eager preparation before capture"
                )
            with torch.cuda.device(a.device):
                _COMPILED[key] = _compile(
                    m, n, k, tactic, compute_capability=compute_capability
                )
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
            tactic = policy.default_tactic(
                m, n, k, compute_capability=get_compute_capability(a.device)
            )
        compiled = self._get_compiled(inputs, tactic)
        if tactic[0] in ("independent", "independent_tma"):
            packed_a, packed_b = a, b.T
            if tactic[0] == "independent":
                packed_a, packed_b = (
                    packed_a.view(torch.int32),
                    packed_b.view(torch.int32),
                )
            compiled(
                packed_a,
                packed_b,
                sfa.view(torch.uint8).reshape(-1).view(torch.int32),
                sfb.T.view(torch.uint8).reshape(-1).view(torch.int32),
                launch_alpha.reshape(1),
                out,
            )
        else:
            compiled(
                a,
                b.T,
                out,
                (m + 127) // 128,
                (n + 127) // 128,
                k // 64,
                sfa.data_ptr(),
                sfb.data_ptr(),
                launch_alpha,
            )
        return out


@cache
def get_runner():
    return Sm12xCuTeFp4GemmRunner()
