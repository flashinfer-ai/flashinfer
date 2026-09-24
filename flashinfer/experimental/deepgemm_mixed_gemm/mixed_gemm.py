"""Prepared mixed FP8 E4M3 × packed FP4 E2M1 GEMM with BF16 output on SM103a."""

from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("mixed_gemm_catalog.json").read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags

    record = _catalog()["programs"][name]
    spec = gen_jit_spec(
        name=name,
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[
            *sm103a_nvcc_flags,
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_key(options):
    return ":".join(
        str(options.get(key, default))
        for key, default in (
            ("M", None),
            ("N", None),
            ("K", None),
            ("num_sms", None),
            ("variant", None),
            ("block_n", 128),
            ("gran_k_a", 32),
        )
    )


class MixedGemmPlan:
    """Prepared FP8 E4M3 A @ packed FP4 E2M1 B.T with BF16 output.

    Scales use catalog-declared packed UE8M0 storage. Model routes use per-32
    A/B scales. Explicit variants use per-128 A scales, with the original
    BK256 route's broadcast packed layout. No alpha epilogue is implemented.
    Call run() on the current stream; do not concurrently reuse one output.
    """

    def __init__(
        self,
        a,
        b,
        a_scales,
        b_scales,
        *,
        m,
        out=None,
        block_n=128,
        gran_k_a=32,
        variant=None,
        descriptor_workspace=None,
    ):
        import torch

        if a.device.type != "cuda" or torch.cuda.get_device_capability(a.device) != (
            10,
            3,
        ):
            raise RuntimeError(
                "Mixed FP8×FP4 GEMM requires the validated SM103a target"
            )
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.dtype not in (torch.float8_e4m3fn, torch.uint8)
            or b.dtype not in (torch.int8, torch.uint8)
        ):
            raise ValueError(
                "A must be E4M3 FP8 (or raw uint8); B must be packed E2M1 bytes"
            )
        n, k = b.shape[0], b.shape[1] * 2
        if a.shape[1] != k:
            raise ValueError("A/B logical K dimensions must agree")
        sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        self.options = dict(
            M=m,
            N=n,
            K=k,
            num_sms=sms,
            block_n=block_n,
            gran_k_a=gran_k_a,
            variant=variant,
        )
        try:
            route = _catalog()["routes"][route_key(self.options)]
        except KeyError as error:
            raise NotImplementedError(
                f"No exported mixed GEMM schedule for {self.options}"
            ) from error
        cfg = route["config"]
        if a.shape[0] != cfg["input_m"]:
            raise ValueError(
                f"A must have {cfg['input_m']} rows for this production route"
            )
        for tensor, shape in (
            (a_scales, (cfg["sfa_words"], cfg["sfa_mn"])),
            (b_scales, (cfg["sfb_words"], cfg["sfb_mn"])),
        ):
            if (
                tensor.dtype not in (torch.int32, torch.uint32)
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(
                    f"Packed scale shape must be {shape} in int32/uint32 words"
                )
        if out is None:
            out = torch.empty(
                (cfg["output_m"], n), dtype=torch.bfloat16, device=a.device
            )
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (cfg["output_m"], n):
            raise ValueError("Output must be BF16 with the production physical M and N")
        tensors = (a, b, a_scales, b_scales, out)
        if any(t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError(
                "Operands, packed scales and output must be contiguous on one CUDA device"
            )
        self.bindings = dict(
            A=a.view(torch.uint8),
            B=b.view(torch.uint8),
            SFA=a_scales.view(torch.uint32),
            SFB=b_scales.view(torch.uint32),
            C_tma=out,
            M=cfg["output_m"],
            N=n,
            K=k,
            grid_m=cfg["grid_m"],
            grid_n=cfg["grid_n"],
            K_tiles=cfg["K_tiles"],
            grid_x=cfg["grid"][0],
            grid_y=cfg["grid"][1],
            grid_z=cfg["grid"][2],
        )
        module, record = load_program(route["program"])
        workspace_bytes = record["tma_workspace_bytes"]
        own_descriptors = bool(workspace_bytes and descriptor_workspace is None)
        caller_descriptor_workspace = descriptor_workspace
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=a.device
                )
            if (
                descriptor_workspace.dtype != torch.uint8
                or descriptor_workspace.device != a.device
                or not descriptor_workspace.is_contiguous()
                or descriptor_workspace.numel() < workspace_bytes
                or descriptor_workspace.data_ptr() % 128
            ):
                raise ValueError(
                    "Descriptor workspace must be aligned contiguous CUDA uint8 storage"
                )
        args = tuple(
            descriptor_workspace if kind == "workspace" else self.bindings[name]
            for kind, name in record["arg_plan"]
        )
        descriptor_state = fallback_workspace = None
        if own_descriptors:
            import tvm_ffi

            descriptor_state = torch.empty(
                workspace_bytes, dtype=torch.uint8, device="cpu"
            )
            fallback_workspace = torch.empty_like(descriptor_workspace)
            args += (descriptor_state, fallback_workspace)
            with tvm_ffi.use_torch_stream():
                module["initialize_cached"](*args)
            self._submission = (module["run_cached"], args)
        else:
            self._submission = (module[record["ffi_entry"]], args)
        self._retained = (
            module,
            record,
            tensors,
            descriptor_workspace,
            self.bindings,
            descriptor_state,
            fallback_workspace,
        )
        self.storage, self.output = out, out[:m]
        self.descriptor_workspace = caller_descriptor_workspace

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.output
