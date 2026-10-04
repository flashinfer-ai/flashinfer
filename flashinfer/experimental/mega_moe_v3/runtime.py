"""Prepared v3 full pipeline and grouped FP4 compute surfaces on SM100a/SM103a."""

import json
from pathlib import Path

from ..deepgemm_common import load_catalog
from .preparation import prepare_pipeline_bindings, build_tile_lists

LABEL = "Generated MegaMoE v3"


def catalog():
    """The family catalog (schema ``mega_moe_v3.v3``): one program per schedule,
    compiled for every listed architecture; the pipeline programs take the SM
    count from the compile line (``-DNUM_CTAS=<physical SMs>``)."""
    return load_catalog(Path(__file__).with_name("catalog.json"), label=LABEL)


def device_arch(device):
    """Exact generated-program architecture for ``device`` (raises when none is catalogued)."""
    return catalog().device_arch(device)


def supported_num_sms(arch):
    """SM counts with catalogued routes for ``arch``."""
    return list(catalog().supported_num_sms(arch))


def stage_program(stage, arch):
    """Program name of a route stage on ``arch``: shared, or one of a per-architecture pair."""
    program = stage["program"]
    return program if isinstance(program, str) else program[arch]


def load_program(arch, name, num_sms):
    """Build and load program ``name`` for ``arch``; ``NUM_CTAS`` is the route's SM count."""
    definitions = catalog().definitions(arch, name, {"NUM_CTAS": int(num_sms)})
    return catalog().load_program(arch, name, definitions)


def key(surface, args):
    names = (
        (
            "num_tokens",
            "num_experts",
            "top_k",
            "hidden",
            "intermediate",
            "routed_weight_dtype",
            "activation_clamp",
        )
        if surface == "pipeline"
        else (
            ("num_experts", "per_expert_M", "N", "K")
            if surface == "grouped_l1"
            else ("num_experts", "per_expert_M", "hidden", "intermediate")
        )
    )
    return json.dumps(
        [surface, {k: args[k] for k in names}], sort_keys=True, separators=(",", ":")
    )


def _route(arch, surface, args):
    """Catalogued route for ``surface``/``args``; unknown routes raise."""
    try:
        return catalog().routes(arch)[key(surface, args)]
    except KeyError as error:
        raise RuntimeError(
            f"No exported {arch} {surface} schedule for {args}"
        ) from error


def _route_arch(route, device):
    """Architecture of ``device``; it must carry the SM count ``route`` was generated for."""
    arch = device_arch(device)
    sms = catalog().device_num_sms(device, arch)
    if sms != route["num_sms"][arch]:
        raise RuntimeError(
            f"The exported {arch} {route['metadata']['surface']} schedule was generated "
            f"for {route['num_sms'][arch]} physical SMs; this device has {sms}"
        )
    return arch


class _Stage:
    def __init__(self, arch, program, num_sms, bindings):
        import torch

        module, record = load_program(arch, program, num_sms)
        self.bindings = dict(bindings)
        grid = self.bindings.pop("grid")
        self.bindings.update(grid_x=grid[0], grid_y=grid[1], grid_z=grid[2])
        tensor = next(v for v in bindings.values() if isinstance(v, torch.Tensor))
        self.workspace = (
            torch.empty(
                record["tma_workspace_bytes"], dtype=torch.uint8, device=tensor.device
            )
            if record["tma_workspace_bytes"]
            else None
        )
        self.args = tuple(
            self.workspace if kind == "workspace" else self.bindings[name]
            for kind, name in record["arg_plan"]
        )
        self.module, self.entry = module, module[record["ffi_entry"]]
        prepare_entry = record.get("ffi_prepare_entry")
        if prepare_entry is not None:
            import tvm_ffi

            with tvm_ffi.use_torch_stream():
                self.entry = module[prepare_entry](*self.args)
            # Prepared descriptor bytes and Tensor owners are fixed for this stage.
            # Complete initialization before another stream captures or launches it.
            torch.cuda.current_stream(tensor.device).synchronize()
            self.args = ()

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self.entry(*self.args)


class V3Plan:
    """Retain operands/workspaces; submit the compute scope of one catalog route.

    Every exported pipeline route is self-cleaning: it launches exactly one
    kernel per run(), the kernel zeroes its per-launch workspace words before it
    exits and its grid gates are phase-toggling words that no host code resets.
    Such a plan zeroes the caller's counter workspace once, when it is bound
    (never inside run(), so a captured run() holds only the kernel node), and
    rejects reset(). ``self_cleaning`` reports which contract applies.

    Grouped fused keeps its host reset inside run(): it zeroes the L1 arrival
    words before the kernel. Grouped L2 repacks both scale tensors into the
    group-folded layout once at preparation (and again on ``update_scales()``),
    so run() launches only the clustered GEMM. run() owns no CUDA graph;
    callers may capture it after binding. Never time only
    launch_without_reset for a fused route.
    """

    def __init__(
        self,
        route,
        stage_bindings,
        outputs,
        *,
        reset_storage=None,
        reset_buffers=(),
        preparation=(),
        owners=(),
    ):
        import torch

        t = next(
            v
            for b in stage_bindings.values()
            for v in b.values()
            if isinstance(v, torch.Tensor)
        )
        self.arch = _route_arch(route, t.device)
        self.num_sms = route["num_sms"][self.arch]
        self.route, self.outputs = route, outputs
        self.reset_storage, self.reset_buffers = reset_storage, tuple(reset_buffers)
        # Declared per architecture by the exported catalog route; absent means
        # the original host-reset lifecycle.
        self.self_cleaning = bool(route["metadata"].get("self_cleaning", False))
        stages = {
            s["name"]: _Stage(
                self.arch,
                stage_program(s, self.arch),
                self.num_sms,
                stage_bindings[s["name"]],
            )
            for s in route["stages"]
        }
        self.preparation_stages = tuple(stages[name] for name in preparation)
        self.stages = tuple(
            stage for name, stage in stages.items() if name not in preparation
        )
        self.owners = owners
        if torch.cuda.is_current_stream_capturing() and (
            self.self_cleaning or self.preparation_stages
        ):
            # One-time work (counter zero, scale repack) must not become a
            # graph node: the kernel keeps the words clean afterwards and the
            # repacked scales are reused by every later launch.
            raise RuntimeError(
                "Binding a MegaMoE plan during CUDA stream capture is "
                "unsupported; bind (and warm up) before capturing run()"
            )
        if self.self_cleaning:
            self._zero_workspace()
        self.update_scales()

    def _zero_workspace(self):
        if self.reset_storage is not None:
            self.reset_storage.zero_()
        else:
            for tensor in self.reset_buffers:
                tensor.zero_()

    def reset(self):
        if self.self_cleaning:
            raise RuntimeError(
                "This route is self-cleaning: the kernel owns its workspace "
                "lifecycle and the plan zeroed the counters once when bound; "
                "do not reset them on the host"
            )
        self._zero_workspace()

    def update_scales(self):
        """Re-run the route's preparation stages (grouped L2: the scale repack)
        after the caller changed the natural-layout scale words in place."""
        for stage in self.preparation_stages:
            stage.run()

    def launch_without_reset(self):
        for stage in self.stages:
            stage.run()
        return self.outputs

    def run(self):
        if not self.self_cleaning:
            self.reset()
        return self.launch_without_reset()


def bind_prepared(
    surface,
    args,
    stage_bindings,
    outputs,
    *,
    reset_storage=None,
    reset_buffers=(),
    preparation=(),
    owners=(),
):
    """Bind explicit packed tensors to a catalog route with its original lifecycle.

    ``reset_storage``/``reset_buffers`` are the route's counter words. A
    self-cleaning route zeroes them once here (outside any stream capture) and
    never again; the workspace may already have been used by another plan of
    the same route as long as no launch is in flight on it. ``preparation``
    names the route stages that run at bind time (and on ``update_scales()``)
    instead of inside run().
    """
    import torch

    t = next(
        v
        for b in stage_bindings.values()
        for v in b.values()
        if isinstance(v, torch.Tensor)
    )
    arch = device_arch(t.device)
    route = _route(arch, surface, args)
    main = stage_bindings[route["stages"][-1]["name"]]
    if tuple(main["grid"]) != tuple(route["grid"][arch]):
        raise ValueError("No exported scheduling specialization for this prepared grid")
    return V3Plan(
        route,
        stage_bindings,
        outputs,
        reset_storage=reset_storage,
        reset_buffers=reset_buffers,
        preparation=preparation,
        owners=owners,
    )


def prepare_pipeline(inputs):
    """Accept packed E4M3 activations, FP4/FP8 weights, float UE8M0 scales and routes.

    Required dictionary fields: num_experts/top_k/num_tokens/hidden/intermediate,
    routed_weight_dtype/activation_clamp, x_fp8_packed/x_sf_packed,
    topk_idx(int64)/topk_weights(FP32), w1_fp4/w2_fp4 (or w1_fp8/w2_fp8),
    and w1_sf/w2_sf (FP32 powers-of-two, granularity32). Gate/up weights are
    logical halves; preparation applies the selected 8-way interleave and
    source-schedule scale permutation. No upstream library is imported. The
    counter workspace is allocated zeroed and the self-cleaning kernel never
    needs it zeroed again.
    """
    arch = device_arch(inputs["x_fp8_packed"].device)
    route = _route(arch, "pipeline", inputs)
    # The persistent grid derives from the SM count the route was generated for.
    data = prepare_pipeline_bindings(inputs, route["num_sms"][arch])
    return bind_prepared(
        "pipeline",
        inputs,
        {"pipeline": data["bindings"]},
        data["output"],
        reset_storage=data["reset_storage"],
        reset_buffers=data["reset_buffers"],
        owners=(inputs, data),
    )


def prepare_grouped_l2(
    A, B, SFA, SFB, per_expert_M, *, out=None, packed_a=None, packed_b=None
):
    """Grouped gran32 E4M3 × packed E2M1 -> BF16; scales are repacked at preparation."""
    import torch

    e, n, _ = B.shape
    m, k = A.shape
    q = k // 128
    te, tm, offsets, total = build_tile_lists(per_expert_M, e)
    if total != m:
        raise ValueError("per_expert_M must cover A rows")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    if packed_a is None:
        packed_a = torch.empty((q, m), dtype=torch.uint32, device=A.device)
    if packed_b is None:
        packed_b = torch.empty((e * q, n), dtype=torch.uint32, device=A.device)
    args = dict(
        num_experts=e, per_expert_M=list(per_expert_M), hidden=n, intermediate=k
    )
    arch = device_arch(A.device)
    route = _route(arch, "grouped_l2", args)
    tiles = te.numel()
    main = dict(
        grid=tuple(route["grid"][arch]),
        A=A.view(torch.uint8),
        B=B.view(torch.uint8),
        SFA=packed_a,
        SFB=packed_b,
        Out=out,
        tile_expert=te,
        tile_m_local=tm,
        expert_row_offsets=offsets,
        M_total=m,
        N=n,
        K=k,
        grid_n=n // 128,
        K_tiles=k // 256,
        total_m_tiles=tiles,
    )
    repack = dict(
        grid=((m * q + 255) // 256 + (e * n * q + 255) // 256, 1, 1),
        src_a=SFA.view(torch.uint32),
        src_b=SFB.view(torch.uint32),
        dst_a=packed_a,
        dst_b=packed_b,
        a_rows=m,
        b_rows=e * n,
        b_group_rows=n,
        words=q,
    )
    return bind_prepared(
        "grouped_l2",
        args,
        {"scale_layout": repack, "main": main},
        out,
        preparation=("scale_layout",),
        owners=(A, B, SFA, SFB, packed_a, packed_b),
    )


def prepare_grouped_fused(bindings, per_expert_M):
    """Bind grouped L1/SwiGLU/L2 tensors; natural L1 byte scales, packed L2 word scales."""
    args = dict(
        num_experts=len(per_expert_M),
        per_expert_M=list(per_expert_M),
        hidden=bindings["K1"],
        intermediate=bindings["K2"],
    )
    return bind_prepared(
        "grouped_fused",
        args,
        {"main": bindings},
        bindings["Out"],
        reset_buffers=(bindings["l1_arrival"],),
        owners=(bindings,),
    )


def prepare_grouped_l1(bindings, per_expert_M):
    """Bind grouped L1/SwiGLU; outputs are E4M3 bytes and per32 UE8M0 scale bytes."""
    args = dict(
        num_experts=len(per_expert_M),
        per_expert_M=list(per_expert_M),
        N=bindings["N"],
        K=bindings["K"],
    )
    return bind_prepared(
        "grouped_l1",
        args,
        {"main": bindings},
        (bindings["C_fp8"], bindings["SF_out"]),
        owners=(bindings,),
    )
