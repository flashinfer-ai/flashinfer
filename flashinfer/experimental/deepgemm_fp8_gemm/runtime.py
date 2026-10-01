"""Prepared FP8 1D1D GEMM launch for the exported per-architecture PTX specializations."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from .ptx_builder import build_from_ptx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


@functools.cache
def _catalog():
    return json.loads((HERE / "catalog.json").read_text())


def device_arch(device):
    """Exact generated-program architecture for ``device`` (raises when none is catalogued)."""
    import torch

    device = torch.device(device)
    catalogued = sorted(_catalog()["arches"])
    if device.type != "cuda":
        raise RuntimeError("FP8 1D1D GEMM requires a CUDA device")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = _ARCHES.get(capability)
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"FP8 1D1D GEMM has no exported programs for compute capability "
            f"{capability}; catalogued architectures: {catalogued}"
        )
    return arch


def supported_num_sms(arch):
    """SM counts with catalogued routes for ``arch``."""
    routes = _catalog()["arches"][arch]["routes"].values()
    return sorted({route["config"]["num_sms"] for route in routes})


@functools.cache
def load_program(arch, name, cache_dir):
    """Assemble the exported ``arch`` PTX program ``name`` and load its binding."""
    record = dict(_catalog()["arches"][arch]["programs"][name])
    out = Path(cache_dir) / arch / name
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    module = build_from_ptx(
        ptx_path=ROOT / record["sources"][0],
        binding_path=ROOT / record["sources"][1],
        module_ident=record["module_ident"],
        arch=arch,
        ptxas_options=record["compile_flags"],
        workdir=out / "build",
        receipt_path=out / "build.json",
        include_paths=[ROOT / "csrc", ROOT / "include"],
    )
    receipt = json.loads((out / "build.json").read_text())
    record.update(
        library_path=receipt["artifact_path"],
        build_receipt=str(out / "build.json"),
        assembled_cubin_sha256=receipt["assembled_cubin_sha256"],
    )
    return module, record


class Fp8GemmPlan:
    """One prepared launch; ``run()`` submits it on the current PyTorch stream."""

    def __init__(self, entry, args, bindings, record):
        self._submission = (entry, args)
        self.bindings = bindings
        self.record = record

    def run(self):
        import tvm_ffi

        entry, args = self._submission
        with tvm_ffi.use_torch_stream():
            entry(*args)


def prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, *, accumulate=False, cache_dir):
    """Prepare the exported M4096/N7168/K4096 route with prepacked MN-major UE8M0 scale words.

    ``a`` ``[M, K]`` and ``b`` ``[N, K]`` contain FP8 E4M3 bytes (``uint8``).
    ``sfa`` ``[K/512, M]`` and ``sfb`` ``[K/512, N]`` are ``uint32`` words packing
    four adjacent K128 UE8M0 scale bytes, stored MN-major. Forward writes BF16
    ``out`` ``[M, N]``; accumulation reads and updates FP32 ``out`` in place.
    Restore the initializer before each independent accumulated evaluation.
    The device architecture and SM count select the exact exported program.
    Packing and allocations precede ``run()``, which allocates nothing.
    """
    import torch

    case = "wgrad" if accumulate else "forward"
    arch = device_arch(out.device)
    section = _catalog()["arches"][arch]
    route = section["routes"].get(case)
    if route is None:
        raise RuntimeError(
            f"No exported {case} program for {arch}; catalogued routes: "
            f"{sorted(section['routes'])}"
        )
    config = route["config"]
    M, N, K = config["M"], config["N"], config["K"]
    if (
        tuple(a.shape) != (M, K)
        or tuple(b.shape) != (N, K)
        or tuple(out.shape) != (M, N)
    ):
        raise ValueError(f"This prepared specialization requires M{M}/N{N}/K{K}")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise TypeError("a and b must carry FP8 E4M3 bytes as uint8 tensors")
    if (
        sfa.dtype != torch.uint32
        or sfb.dtype != torch.uint32
        or tuple(sfa.shape) != (K // 512, M)
        or tuple(sfb.shape) != (K // 512, N)
    ):
        raise TypeError(
            "sfa/sfb must be uint32 MN-major packed UE8M0 words of shape [K/512, rows]"
        )
    if out.dtype != (torch.float32 if accumulate else torch.bfloat16):
        raise TypeError(
            "Accumulator/output dtype does not match the selected specialization"
        )
    if any(t.device != out.device for t in (a, b, sfa, sfb)):
        raise ValueError("All operands must live on the output device")
    num_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
    if num_sms != config["num_sms"]:
        raise ValueError(
            f"The exported {arch} {case} program is specialized for "
            f"{config['num_sms']} SMs; this device has {num_sms}"
        )
    module, record = load_program(
        arch, route["program"], str(Path(cache_dir).resolve())
    )
    bindings = dict(
        A=a,
        B=b,
        SFA=sfa,
        SFB=sfb,
        C_tma=out,
        M=M,
        N=N,
        K=K,
        grid_m=config["grid_m"],
        grid_n=config["grid_n"],
        K_tiles=config["K_tiles"],
        grid_x=config["grid"][0],
        grid_y=config["grid"][1],
        grid_z=config["grid"][2],
    )
    args = tuple(bindings[key] for _, key in record["arg_plan"])
    return Fp8GemmPlan(module[record["ffi_entry"]], args, bindings, record)
