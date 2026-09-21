"""Prepared fused BF16 routing GEMM and normalized top-k weights on SM103a."""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("mega_gate_catalog.json").read_text())


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
        "M", "K", "E", "num_topk", "num_sms", "scoring_func", "has_bias", "has_image_bias",
        "has_mask", "has_physical_map", "has_fixed", "has_random", "has_unmapped", "deterministic"))


class MegaGatePlan:
    """Bind a fused routing operation; run() returns (expert_indices, weights).

    x is BF16[M,K], weight is BF16[E,K]. Biases are FP32[E]. Physical mapping
    uses int32[E+shared,width] and logical_count int32[E+shared]. Optional logical top-k
    output is int64[M,topk], with unit column stride. The selected rows emit
    int64 indices and FP32 normalized unbiased scores scaled by routed_scale.
    Deterministic mode preserves the source split-K selection and replay rule.

    Optional scratch and score_barriers are caller-owned; fresh barriers must
    be zeroed once before first use. A plan retains all buffers and submits on
    the current PyTorch stream. Do not use one plan concurrently across streams.
    Only configurations present in the exported physical catalog are accepted.
    """
    def __init__(self, x, weight, num_topk=6, *, scoring_func="sqrtsoftplus", bias=None,
                 image_bias=None, image_token_mask=None, mask=None, to_physical_map=None,
                 logical_count=None, fix_routing_mask=None, force_random=None,
                 unmapped_topk_idx=None, use_shared_as_routed=False, num_shared_experts=1,
                 routed_scaling_factor=1.5, ep_rank=0, out=None, deterministic=False,
                 scratch=None, score_barriers=None, descriptor_workspace=None):
        import torch
        if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 3):
            raise RuntimeError("Mega Gate requires the validated SM103a target")
        if x.ndim != 2 or weight.ndim != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("x and weight must be contiguous BF16 matrices")
        M, K = x.shape
        E, wk = weight.shape
        if wk != K or not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("x and weight must be contiguous and share K")
        if (image_bias is None) != (image_token_mask is None) or (to_physical_map is None) != (logical_count is None):
            raise ValueError("image and physical mapping tensors must be supplied in pairs")
        if fix_routing_mask is not None and unmapped_topk_idx is None:
            raise ValueError("fixed routing requires unmapped_topk_idx")
        if not 0 <= ep_rank <= 0x7fffffff:
            raise ValueError("ep_rank must fit nonnegative int32")
        shared = num_shared_experts if use_shared_as_routed else 0
        if shared and (shared not in (1, 2) or num_topk % shared or E % (num_topk // shared)):
            raise ValueError("shared experts require the shared/top-k divisibility")
        if num_topk + shared > 32:
            raise ValueError("routed and shared output slots must fit one warp")
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        self.config = dict(M=M, K=K, E=E, num_topk=num_topk, num_sms=sms, scoring_func=scoring_func,
            has_bias=bias is not None, has_image_bias=image_bias is not None, has_mask=mask is not None,
            has_physical_map=to_physical_map is not None, has_fixed=fix_routing_mask is not None,
            has_random=force_random is not None, has_unmapped=unmapped_topk_idx is not None,
            deterministic=bool(deterministic))
        try:
            route = _catalog()["routes"][route_key(self.config)]
        except KeyError as error:
            raise NotImplementedError(f"No exported routing schedule for {self.config}") from error
        cfg = route["config"]
        blocks, aligned_e = (M + cfg["block_tokens"] - 1) // cfg["block_tokens"], (E + 127) // 128 * 128
        if out is None:
            out = (torch.empty((M, num_topk + shared), dtype=torch.int64, device=x.device),
                   torch.empty((M, num_topk + shared), dtype=torch.float32, device=x.device))
        if len(out) != 2 or any(tuple(t.shape) != (M, num_topk + shared) for t in out) or out[0].dtype != torch.int64 or out[1].dtype != torch.float32:
            raise ValueError("out must be int64/FP32 tensors of shape [M,topk+shared]")
        scratch_shape = (blocks, cfg["num_split_k"], cfg["block_tokens"], aligned_e)
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=x.device)
        if score_barriers is None:
            score_barriers = torch.zeros((blocks, 16), dtype=torch.uint64, device=x.device)
        if tuple(scratch.shape) != scratch_shape or scratch.dtype != torch.float32:
            raise ValueError(f"scratch must be FP32{scratch_shape}")
        if tuple(score_barriers.shape) != (blocks, 16) or score_barriers.dtype != torch.uint64:
            raise ValueError("score_barriers must be uint64[ceil(M/block_tokens),16]")
        for name, tensor, dtype, shape in (
            ("bias", bias, torch.float32, (E,)), ("image_bias", image_bias, torch.float32, (E,)),
            ("logical_count", logical_count, torch.int32, (E + shared,))):
            if tensor is not None and (tensor.dtype != dtype or tuple(tensor.shape) != shape):
                raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
        for name, tensor in (("image_token_mask", image_token_mask), ("mask", mask),
                             ("fix_routing_mask", fix_routing_mask), ("force_random", force_random)):
            if tensor is not None and (tensor.dtype not in (torch.bool, torch.uint8) or tuple(tensor.shape) != (M,)):
                raise ValueError(f"{name} must be bool/uint8[M]")
        if to_physical_map is not None and (to_physical_map.dtype != torch.int32 or to_physical_map.ndim != 2 or to_physical_map.shape[0] != E + shared):
            raise ValueError("to_physical_map must be int32[E+shared,width]")
        if unmapped_topk_idx is not None and (unmapped_topk_idx.dtype != torch.int64 or tuple(unmapped_topk_idx.shape) != (M, num_topk) or unmapped_topk_idx.stride(1) != 1):
            raise ValueError("unmapped_topk_idx must be int64[M,topk] with unit column stride")
        tensors = (x, weight, bias, image_bias, image_token_mask, mask, to_physical_map, logical_count,
                   fix_routing_mask, force_random, *out, scratch, score_barriers)
        if any(t.device != x.device or not t.is_contiguous() for t in tensors if t is not None):
            raise ValueError("inputs and workspace must be contiguous on one CUDA device")
        if unmapped_topk_idx is not None and unmapped_topk_idx.device != x.device:
            raise ValueError("unmapped_topk_idx must be on the input device")
        dummy_f32 = torch.empty(1, dtype=torch.float32, device=x.device)
        dummy_i32 = torch.empty(1, dtype=torch.int32, device=x.device)
        dummy_i64 = torch.empty(1, dtype=torch.int64, device=x.device)
        dummy_u8 = torch.empty(1, dtype=torch.uint8, device=x.device)
        byte = lambda t: t.view(torch.uint8) if t is not None else dummy_u8
        self.bindings = dict(X=x, W=weight, bias=bias if bias is not None else dummy_f32,
            image_bias=image_bias if image_bias is not None else dummy_f32,
            image_mask=byte(image_token_mask), mask=byte(mask),
            physical_map=to_physical_map if to_physical_map is not None else dummy_i32,
            logical_count=logical_count if logical_count is not None else dummy_i32,
            topk_idx=out[0], topk_weights=out[1],
            unmapped_idx=unmapped_topk_idx if unmapped_topk_idx is not None else dummy_i64,
            scratch=scratch, score_barriers=score_barriers, fixed_mask=byte(fix_routing_mask),
            random_mask=byte(force_random), num_tokens=M, num_shared=shared,
            map_width=to_physical_map.shape[1] if to_physical_map is not None else 1,
            ep_rank=ep_rank, routed_scale=routed_scaling_factor,
            unmapped_stride=unmapped_topk_idx.stride(0) if unmapped_topk_idx is not None else num_topk,
            grid_x=cfg["num_launch_sms"], grid_y=1, grid_z=1)
        module, record = load_program(route["program"])
        workspace_bytes = record["tma_workspace_bytes"]
        own_descriptors = bool(workspace_bytes and descriptor_workspace is None)
        caller_descriptor_workspace = descriptor_workspace
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=x.device)
            if descriptor_workspace.dtype != torch.uint8 or descriptor_workspace.device != x.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel() < workspace_bytes or descriptor_workspace.data_ptr() % 128:
                raise ValueError("descriptor workspace must be aligned contiguous CUDA uint8 storage")
        args = tuple(descriptor_workspace if kind == "workspace" else self.bindings[name]
                     for kind, name in record["arg_plan"])
        descriptor_state = fallback_workspace = None
        if own_descriptors:
            import tvm_ffi
            descriptor_state = torch.empty(workspace_bytes, dtype=torch.uint8, device="cpu")
            fallback_workspace = torch.empty_like(descriptor_workspace)
            args += (descriptor_state, fallback_workspace)
            # Source pointer descriptors use synchronous HtoD initialization.
            # Immutable storage is ready when construction returns, including
            # for a subsequent run on another current stream.
            with tvm_ffi.use_torch_stream():
                module["initialize_cached"](*args)
            self._submission = (module["run_cached"], args)
        else:
            self._submission = (module[record["ffi_entry"]], args)
        self._retained = (module, record, tensors, unmapped_topk_idx, descriptor_workspace,
                          dummy_f32, dummy_i32, dummy_i64, dummy_u8,
                          descriptor_state, fallback_workspace)
        self.outputs, self.scratch, self.score_barriers = out, scratch, score_barriers
        self.descriptor_workspace = caller_descriptor_workspace

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.outputs
