"""Prepared fused projection, Sinkhorn, residual mixing and RMSNorm on SM103a."""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("mega_mhc_catalog.json").read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags
    record = _catalog()["programs"][name]
    spec = gen_jit_spec(name=name,
        sources=[env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]],
        extra_cuda_cflags=[*sm103a_nvcc_flags, *record["compile_flags"],
                           "--device-entity-has-hidden-visibility=false"],
        extra_ldflags=["-lcuda"], extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False)
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_key(config):
    return ":".join(str(config[key]) for key in (
        "num_tokens", "hidden", "num_sms", "shifted", "store_fp8", "shared_sf_block_m", "deterministic"))


def _new_outputs(x, residual, post_mix, comb_res_mix, shifted_prev_mix, sf_layout, shared_sf_block_m):
    import torch
    tokens, hidden = x.shape
    outputs = dict(new_residual=torch.empty_like(residual), new_post_mix=torch.empty_like(post_mix),
                   new_comb_res_mix=torch.empty_like(comb_res_mix), y_bf16=torch.empty_like(x))
    if shifted_prev_mix is not None:
        outputs["new_prev_mix"] = torch.empty_like(shifted_prev_mix)
    if sf_layout != "bf16":
        outputs["y_fp8"] = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device)
        if sf_layout == "col":
            outputs["y_gemm_sf"] = torch.empty_strided((tokens, hidden // 128),
                (1, (tokens + 3) // 4 * 4), dtype=torch.int32, device=x.device)
        else:
            outputs["y_routed_sf"] = torch.empty((tokens, hidden // 128), dtype=torch.int32, device=x.device)
            rows = (tokens + shared_sf_block_m - 1) // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128)
            storage = torch.empty_strided((rows, hidden // 128), (1, rows), dtype=torch.int32, device=x.device)
            outputs.update(y_shared_sf_storage=storage, y_shared_sf=storage[:tokens],
                           shared_sf_block_m=shared_sf_block_m)
    return outputs


class MegaMHCPlan:
    """Bind a fused mHC operation and reuse its private invocation state.

    x/residual/rmsnorm_weight are BF16; projection and mixing coefficients are
    FP32. The four residual routes produce BF16 normalized output plus FP8
    E4M3 output and packed per-32 UE8M0 scales. ``sf_layout='col'`` stores
    column-major scale words; ``'extra'`` also stores the shared-expert padded
    row permutation. Only shapes present in the exported catalog are accepted.

    Scratch, split_barriers and launch_epochs may be caller-owned. Fresh
    barriers/epochs must be zeroed once before first use and retained across
    ordinary calls and CUDA Graph replay. Epoch storage always has num_sms
    entries, including CTAs omitted by a selected launch grid. One plan must
    not execute concurrently on multiple streams. run() uses the current
    PyTorch stream and performs no device allocation or workspace reset.
    """
    def __init__(self, *, x, residual, post_mix, comb_res_mix, shifted_prev_mix,
                 fn, mix_scales, mix_bases, rmsnorm_weight, hc_mult=4,
                 hc_norm_eps=2e-5, hc_pre_eps=3e-4, hc_post_scale=1.25,
                 sinkhorn_eps=2e-6, num_sinkhorn_iters=20, rmsnorm_eps=7e-6,
                 rmsnorm_scale=1.25, sf_layout="col", shared_sf_block_m=224,
                 out=None, deterministic=False, scratch=None, split_barriers=None,
                 launch_epochs=None, descriptor_workspace=None):
        import torch
        if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 3):
            raise RuntimeError("Mega mHC requires the validated SM103a target")
        if x.ndim != 2 or hc_mult != 4 or sf_layout not in ("bf16", "col", "extra"):
            raise ValueError("Expected x[T,H], four residual routes and col/extra/bf16 scale layout")
        tokens, hidden = x.shape
        if hidden % 1024 or not 1 <= tokens <= 1 << 20 or num_sinkhorn_iters < 1:
            raise ValueError("Expected H divisible by 1024, 1 <= T <= 2**20 and positive Sinkhorn iterations")
        if sf_layout == "extra" and shared_sf_block_m <= 0:
            raise ValueError("shared_sf_block_m must be positive")
        shifted = shifted_prev_mix is not None
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        self.options = dict(num_tokens=tokens, hidden=hidden, num_sms=sms, shifted=shifted,
            store_fp8=sf_layout != "bf16", shared_sf_block_m=shared_sf_block_m if sf_layout == "extra" else 0,
            deterministic=bool(deterministic))
        try:
            route = _catalog()["routes"][route_key(self.options)]
        except KeyError as error:
            raise NotImplementedError(f"No exported mHC schedule for {self.options}") from error
        cfg = self.config = route["config"]

        def require(name, tensor, dtype, shape, *, contiguous=True):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape or tensor.device != x.device:
                raise ValueError(f"{name} must be {dtype}{shape} on the input CUDA device")
            if contiguous and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        for name, tensor, dtype, shape in (
            ("x", x, torch.bfloat16, (tokens, hidden)),
            ("residual", residual, torch.bfloat16, (tokens, 4, hidden)),
            ("post_mix", post_mix, torch.float32, (tokens, 4, 1)),
            ("comb_res_mix", comb_res_mix, torch.float32, (tokens, 4, 4)),
            ("fn", fn, torch.float32, (24, 4 * hidden)),
            ("mix_scales", mix_scales, torch.float32, (3,)),
            ("mix_bases", mix_bases, torch.float32, (24,)),
            ("rmsnorm_weight", rmsnorm_weight, torch.bfloat16, (hidden,))):
            require(name, tensor, dtype, shape)
        if shifted:
            require("shifted_prev_mix", shifted_prev_mix, torch.float32, (tokens, 4, 1))
        if out is None:
            out = _new_outputs(x, residual, post_mix, comb_res_mix, shifted_prev_mix, sf_layout, shared_sf_block_m)
        for name, dtype, shape in (
            ("new_residual", torch.bfloat16, (tokens, 4, hidden)),
            ("new_post_mix", torch.float32, (tokens, 4, 1)),
            ("new_comb_res_mix", torch.float32, (tokens, 4, 4)),
            ("y_bf16", torch.bfloat16, (tokens, hidden))):
            require(name, out[name], dtype, shape)
        if shifted:
            require("new_prev_mix", out["new_prev_mix"], torch.float32, (tokens, 4, 1))
        if sf_layout != "bf16":
            require("y_fp8", out["y_fp8"], torch.float8_e4m3fn, (tokens, hidden))
            primary = out["y_gemm_sf" if sf_layout == "col" else "y_routed_sf"]
            require("primary scale words", primary, torch.int32, (tokens, hidden // 128), contiguous=False)
            expected_stride = (1, (tokens + 3) // 4 * 4) if sf_layout == "col" else (hidden // 128, 1)
            if primary.stride() != expected_stride:
                raise ValueError(f"Primary scale strides must be {expected_stride}")
            if sf_layout == "extra":
                rows = (tokens + shared_sf_block_m - 1) // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128)
                storage, shared = out["y_shared_sf_storage"], out["y_shared_sf"]
                require("shared scale storage", storage, torch.int32, (rows, hidden // 128), contiguous=False)
                require("shared scale view", shared, torch.int32, (tokens, hidden // 128), contiguous=False)
                if storage.stride() != (1, rows) or shared.stride() != (1, rows) or storage.data_ptr() != shared.data_ptr() or out["shared_sf_block_m"] != shared_sf_block_m:
                    raise ValueError("Shared scales require the padded column-major storage and its first-T-row view")
        scratch_shape = ((tokens + 63) // 64 * cfg["num_splits"] * (1536 + 128),)
        barriers_shape = (2 * ((1 << 20) // 64) * 16,)
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=x.device)
        if split_barriers is None:
            split_barriers = torch.zeros(barriers_shape, dtype=torch.uint64, device=x.device)
        if launch_epochs is None:
            launch_epochs = torch.zeros((sms,), dtype=torch.uint64, device=x.device)
        require("scratch", scratch, torch.float32, scratch_shape)
        require("split_barriers", split_barriers, torch.uint64, barriers_shape)
        require("launch_epochs", launch_epochs, torch.uint64, (sms,))
        dummy_u32 = torch.empty(1, dtype=torch.uint32, device=x.device)
        dummy_u8 = torch.empty(1, dtype=torch.uint8, device=x.device)
        dummy_float = torch.empty(1, dtype=torch.float32, device=x.device)
        primary = out.get("y_gemm_sf", out.get("y_routed_sf", dummy_u32))
        shared = out.get("y_shared_sf", dummy_u32)

        def raw_storage(tensor):
            words = tensor.untyped_storage().nbytes() // tensor.element_size() - tensor.storage_offset()
            return tensor.as_strided((words,), (1,)).view(torch.uint32)

        self.bindings = dict(residual_map=residual, x_map=x, fn_map=fn,
            post_map=post_mix.view(tokens, 4), comb_map=comb_res_mix.view(tokens, 16),
            prev_map=(shifted_prev_mix if shifted else post_mix).view(tokens, 4),
            new_residual_map=out["new_residual"], y_map=out["y_bf16"] if shifted else x,
            mix_scales=mix_scales, mix_bases=mix_bases, new_prev_mix=out.get("new_prev_mix", dummy_float),
            new_post_mix=out["new_post_mix"], new_comb_res_mix=out["new_comb_res_mix"],
            rmsnorm_weight=rmsnorm_weight, new_residual=out["new_residual"], y_bf16=out["y_bf16"],
            y_fp8=out["y_fp8"].view(torch.uint8) if "y_fp8" in out else dummy_u8,
            y_primary_sf=raw_storage(primary), y_shared_sf=raw_storage(shared), scratch=scratch,
            split_barriers=split_barriers, launch_epochs=launch_epochs, num_tokens=tokens,
            hc_norm_eps=hc_norm_eps, hc_pre_eps=hc_pre_eps, hc_post_scale=hc_post_scale,
            sinkhorn_eps=sinkhorn_eps, num_sinkhorn_iters=num_sinkhorn_iters,
            rmsnorm_eps=rmsnorm_eps, rmsnorm_scale=rmsnorm_scale,
            primary_sf_stride_token=primary.stride(0), primary_sf_stride_word=primary.stride(1) if primary.ndim == 2 else 0,
            shared_sf_stride_word=shared.stride(1) if shared.ndim == 2 else 0,
            grid_x=cfg["num_launch_sms"], grid_y=1, grid_z=1)
        module, record = load_program(route["program"])
        workspace_bytes = record["tma_workspace_bytes"]
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=x.device)
            if descriptor_workspace.dtype != torch.uint8 or descriptor_workspace.device != x.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel() < workspace_bytes or descriptor_workspace.data_ptr() % 128:
                raise ValueError("Descriptor workspace must be aligned contiguous CUDA uint8 storage")
        args = tuple(descriptor_workspace if kind == "workspace" else self.bindings[name]
                     for kind, name in record["arg_plan"])
        self._submission = (module[record["ffi_entry"]], args)
        self._retained = (module, record, x, residual, post_mix, comb_res_mix, shifted_prev_mix,
                          fn, mix_scales, mix_bases, rmsnorm_weight, dummy_u32, dummy_u8, dummy_float,
                          descriptor_workspace)
        self.outputs, self.scratch, self.split_barriers, self.launch_epochs = out, scratch, split_barriers, launch_epochs

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.outputs
