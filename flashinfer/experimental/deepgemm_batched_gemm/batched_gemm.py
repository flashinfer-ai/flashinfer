"""Prepared per-head FP8 projections with BF16 or dynamic FP8 output on SM103a."""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name('batched_gemm_catalog.json').read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags
    record = _catalog()['programs'][name]
    spec = gen_jit_spec(name=name,
        sources=[env.FLASHINFER_CSRC_DIR / p.removeprefix('csrc/') for p in record['sources']],
        extra_cuda_cflags=[*sm103a_nvcc_flags, *record['compile_flags'],
                           '--device-entity-has-hidden-visibility=false'],
        extra_ldflags=['-lcuda'], extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False)
    return spec.build_and_load(), {**record, 'library_path': str(spec.get_library_path())}


def route_key(options):
    return ':'.join(str(options[key]) for key in
                    ('tokens', 'num_heads', 'inner', 'width', 'num_sms', 'num_stages', 'epilogue'))


def _pack_scales(scales, padded_rows):
    import torch
    heads, rows, groups = scales.shape
    exponents = (scales.contiguous().view(torch.int32).to(torch.int64) >> 23) & 255
    groups_of_four = exponents.reshape(heads, rows, groups // 4, 4)
    shifts = torch.arange(4, dtype=torch.int64, device=scales.device) * 8
    words = (groups_of_four << shifts).sum(-1).to(torch.int32)
    storage = torch.zeros((heads, groups // 4, padded_rows), dtype=torch.int32, device=scales.device)
    storage[:, :, :rows] = words.permute(0, 2, 1)
    return storage.reshape(heads * (groups // 4), padded_rows).view(torch.uint32)


class BatchedGemmPlan:
    """Prepared A[T,H,K] @ B[H,N,K] -> output[T,H,N].

    Inputs are E4M3 with positive power-of-two FP32 scales per A token/K128
    and B N128/K128 block. Scale packing and output/workspace allocation occur
    during preparation. Operand values may change between runs; prepare a new
    plan when the input scales change. Dynamic FP8 returns E4M3 values and
    packed per-32 UE8M0 scale words. BF16 optionally applies runtime alpha.
    Submit on the current stream; do not concurrently reuse one output.
    """
    def __init__(self, a, b, *, output_fp8=True, alpha=None, out=None, output_scales=None,
                 descriptor_workspace=None):
        import torch
        aq, asf = a
        bq, bsf = b
        if aq.device.type != 'cuda' or torch.cuda.get_device_capability(aq.device) != (10, 3):
            raise RuntimeError('Batched FP8 projection requires the validated SM103a target')
        if aq.ndim != 3 or bq.ndim != 3 or aq.dtype != torch.float8_e4m3fn or bq.dtype != torch.float8_e4m3fn:
            raise ValueError('Expected E4M3 A[T,H,K] and B[H,N,K]')
        tokens, heads, inner = aq.shape
        width = bq.shape[1]
        if bq.shape[0] != heads or bq.shape[2] != inner or tokens < 1 or width % 128 or inner % 512:
            raise ValueError('Expected matching H/K, T>=1, N divisible by128 and K divisible by512')
        if asf.dtype != torch.float32 or bsf.dtype != torch.float32 or tuple(asf.shape) != (tokens, heads, inner // 128) or tuple(bsf.shape) != (heads, width // 128, inner // 128):
            raise ValueError('Expected FP32 A[T,H,K/128] and B[H,N/128,K/128] scales')
        if any(t.device != aq.device for t in (bq, asf, bsf)) or not aq.is_contiguous() or not bq.is_contiguous():
            raise ValueError('Operands must be contiguous; operands and scales must share a CUDA device')
        if output_fp8 and alpha is not None:
            raise ValueError('Dynamic FP8 output and runtime alpha are separate epilogues')
        if not output_fp8 and output_scales is not None:
            raise ValueError('BF16 output has no output scales')
        epilogue = 'fp8' if output_fp8 else 'alpha' if alpha is not None else 'bf16'
        sms = torch.cuda.get_device_properties(aq.device).multi_processor_count
        self.options = dict(tokens=tokens, num_heads=heads, inner=inner, width=width,
                            num_sms=sms, num_stages=5, epilogue=epilogue)
        try:
            route = _catalog()['routes'][route_key(self.options)]
        except KeyError as error:
            raise NotImplementedError(f'No exported batched projection schedule for {self.options}') from error
        cfg = route['config']
        dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
        if out is None:
            out = torch.empty((tokens, heads, width), dtype=dtype, device=aq.device)
        if out.dtype != dtype or tuple(out.shape) != (tokens, heads, width) or not out.is_contiguous() or out.device != aq.device:
            raise ValueError('Output must have the selected dtype and contiguous [T,H,N] layout on the input device')
        if output_fp8:
            shape = (tokens, heads * width // 128)
            stride = (1, (tokens + 3) // 4 * 4)
            if output_scales is None:
                output_scales = torch.empty_strided(shape, stride, dtype=torch.int32, device=aq.device)
            if output_scales.dtype != torch.int32 or output_scales.device != aq.device or tuple(output_scales.shape) != shape or tuple(output_scales.stride()) != stride:
                raise ValueError('Output scales must be int32 [T,H*N/128], column-major with T padded to4')
            words = output_scales.untyped_storage().nbytes() // output_scales.element_size() - output_scales.storage_offset()
            sf_storage = output_scales.as_strided((words,), (1,)).view(torch.uint32)
            sf_stride = output_scales.stride(1)
        else:
            sf_storage = torch.empty((1,), dtype=torch.uint32, device=aq.device)
            sf_stride = 0
        packed_a = _pack_scales(asf.permute(1, 0, 2), (tokens + 127) // 128 * 128)
        packed_b = _pack_scales(bsf.repeat_interleave(128, dim=1), width)
        self.bindings = dict(A=aq.view(torch.uint8), B=bq.view(torch.uint8), SFA=packed_a, SFB=packed_b,
            D=out.view(torch.uint8) if output_fp8 else out, SFD=sf_storage, M=tokens,
            grid_m=cfg['grid_m'], sfd_stride=sf_stride, alpha=1.0 if alpha is None else float(alpha),
            grid_x=cfg['grid'][0], grid_y=cfg['grid'][1], grid_z=cfg['grid'][2])
        module, record = load_program(route['program'])
        workspace_bytes = record['tma_workspace_bytes']
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=aq.device)
            if descriptor_workspace.dtype != torch.uint8 or descriptor_workspace.device != aq.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel() < workspace_bytes or descriptor_workspace.data_ptr() % 128:
                raise ValueError('Descriptor workspace must be aligned contiguous CUDA uint8 storage')
        args = tuple(descriptor_workspace if kind == 'workspace' else self.bindings[name] for kind, name in record['arg_plan'])
        self._submission = (module[record['ffi_entry']], args)
        self._retained = (module, record, a, b, out, output_scales, sf_storage, self.bindings, descriptor_workspace)
        self.values, self.scales = out, output_scales
        self.output = (out, output_scales) if output_fp8 else out
        self.descriptor_workspace = descriptor_workspace

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.output
