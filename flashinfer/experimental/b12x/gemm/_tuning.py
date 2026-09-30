"""Native three-dimensional dense GEMM host launch contract."""

from dataclasses import asdict, dataclass, field, fields

from b12x.preparation._efficiency import capture_exhaustive_search

from b12x.preparation import FrozenMapping
from b12x.preparation.tuning import Knob, ParameterBinding, ParameterSpace, TuningContract


RECIPES = (
    "nvfp4",
    "mxfp4",
    "mxfp8",
    "tensor_fp8",
    "block_fp8",
    "mxfp6_e2m3",
    "mxfp6_e3m2",
    "w6a8_e2m3",
    "w6a8_e3m2",
)


@dataclass(frozen=True, kw_only=True)
class DenseGemmQuery:
    recipe: str
    entry_point: str
    weight_storage: str
    output_dtype: str
    batch: int
    max_rows: int
    in_features: int
    out_features: int
    output_mode: str = "functional"
    alpha_mode: str = "tensor"
    expected_m: int | None = None
    overrides: FrozenMapping = FrozenMapping()
    codegen: FrozenMapping | None = None
    workspace_nbytes: int | None = None
    sm_count: int | None = None
    workspace_form: str = "owned"
    sfb_k_replicated: bool = False
    exhaustive: bool = field(default_factory=capture_exhaustive_search)

    def __post_init__(self):
        object.__setattr__(self, "overrides", FrozenMapping(self.overrides))
        object.__setattr__(self, "codegen", _codegen_snapshot() if self.codegen is None else FrozenMapping(self.codegen))


def _codegen_snapshot():
    from b12x._lib import dense_gemm as dense
    return FrozenMapping({
        "split_k_atomic": dense._B12X_DENSE_SPLITK_TURBO,
        "fused_fp6_quant": dense._DENSE_FUSED_QUANT,
    })


@dataclass(frozen=True, kw_only=True)
class DenseGemmConfig:
    backend: str
    tile_m: int
    tile_n: int
    tile_k: int
    load_path: str
    swap_ab: bool
    split_k_slices: int | None
    large_m_unroll: bool | None
    target_occupancy: int | None


def validate_query(query):
    if query.recipe not in RECIPES:
        raise ValueError("unknown dense GEMM recipe")
    if query.entry_point not in ("gemm.mm", "gemm.blockscaled.mm"):
        raise ValueError("unknown serialized dense GEMM entry point")
    if query.output_dtype not in ("bfloat16", "float16"):
        raise ValueError("dense GEMM search requires BF16 or FP16 output")
    if query.output_mode not in ("functional", "provided"):
        raise ValueError("dense GEMM output mode must be functional or provided")
    if query.alpha_mode not in ("unit", "tensor"):
        raise ValueError("dense GEMM alpha mode must be unit or tensor")
    counts = (query.batch, query.max_rows, query.in_features, query.out_features)
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("dense GEMM shape coordinates must be positive integers")
    alignment = 256 if query.recipe == "mxfp4" else 128
    if query.in_features % alignment:
        raise ValueError("dense GEMM requires recipe-aligned K")
    fp6 = query.recipe.startswith(("mxfp6_", "w6a8_"))
    if query.weight_storage not in (("packed", "expanded") if fp6 else ("native",)):
        raise ValueError("weight storage does not match the dense GEMM recipe")
    if query.recipe == "block_fp8" and (query.batch != 1 or query.out_features % 128):
        raise ValueError("block-FP8 requires one matrix and N divisible by 128")
    if query.expected_m is not None and (type(query.expected_m) is not int or query.expected_m <= 0):
        raise ValueError("expected_m must be a positive integer or None")
    if query.sm_count is not None and (type(query.sm_count) is not int or query.sm_count <= 0):
        raise ValueError("SM capacity must be a positive integer or None")
    if type(query.sfb_k_replicated) is not bool:
        raise TypeError("weight-scale replication must be a boolean")
    if query.sfb_k_replicated and query.recipe != "mxfp8":
        raise ValueError("weight-scale replication requires MXFP8")
    if query.codegen != _codegen_snapshot():
        raise ValueError("dense declaration code-generation controls differ from the loaded kernels")
    allowed = {
        "mma_tiler_mn", "load_path", "swap_ab", "_tile_k_override",
        "_split_k_slices_override", "_large_m_unroll_override", "_target_occupancy_override",
    }
    if set(query.overrides) - allowed:
        raise ValueError("unknown dense launch constraint")
    if query.workspace_nbytes is not None and (
        type(query.workspace_nbytes) is not int or query.workspace_nbytes < 0 or query.workspace_nbytes % 4
    ):
        raise ValueError("dense workspace capacity must be a nonnegative FP32 byte count")
    if query.workspace_form not in ("owned", "provided"):
        raise ValueError("unknown dense workspace form")
    if query.workspace_form == "owned" and query.workspace_nbytes is not None:
        raise ValueError("owned dense workspace cannot carry a caller capacity limit")


def operand_options(query):
    recipe = query.recipe
    if recipe.startswith(("mxfp6_", "w6a8_")):
        fmt = recipe.rsplit("_", 1)[1]
        return dict(
            ab_dtype=f"float6_{fmt}fn",
            sf_dtype="float8_e8m0fnu",
            sf_vec_size=32,
            a_preexpanded=True,
            b_preexpanded=query.weight_storage == "expanded",
            b_packed=query.weight_storage == "packed",
            b_fmt=fmt,
            a_fmt="e4m3" if recipe.startswith("w6a8_") else fmt,
        )
    return dict(
        ab_dtype="float4_e2m1fn" if recipe in ("nvfp4", "mxfp4") else "float8_e4m3fn",
        sf_dtype={"nvfp4": "float8_e4m3fn", "block_fp8": "float32"}.get(
            recipe, "float8_e8m0fnu"
        ),
        sf_vec_size={"nvfp4": 16, "block_fp8": 128}.get(recipe, 32),
        plain_fp8=recipe == "tensor_fp8",
        block_fp8=recipe == "block_fp8",
        sfb_k_replicated=query.sfb_k_replicated,
    )


def query_from_call(lhs, rhs, out=None, *, entry_point, options):
    """Normalize the native-call geometry and immutable launch constraints."""
    import torch

    a, sfa = lhs
    b, sfb = rhs
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(
            "dense search requires native three-dimensional operand values"
        )
    m, packed_k, batch = a.shape
    n, rhs_k, rhs_batch = b.shape
    if (
        batch != rhs_batch
        or a.device != b.device
        or a.stride(1) != 1
        or b.stride(1) != 1
    ):
        raise ValueError(
            "dense search requires matching batches/devices and contiguous K storage"
        )
    expected_m = options.get("expected_m")
    if expected_m is not None and (type(expected_m) is not int or expected_m <= 0):
        raise ValueError("expected_m must be a positive integer or None")
    if options.get("cluster_shape_mn", (1, 1)) != (1, 1):
        raise ValueError("dense search requires a single-CTA cluster")
    unsupported = tuple(
        name
        for name in (
            "rhs_values_tiled",
            "_quantized_c",
            "x_bf16",
            "w_gscale",
            "row_scale",
        )
        if options.get(name) is not None
    )
    if unsupported:
        raise ValueError(f"dense search does not model auxiliary operands: {unsupported}")
    ab, sf, vec = (
        options.get(name) for name in ("ab_dtype", "sf_dtype", "sf_vec_size")
    )
    storage = "native"
    if ab == "float4_e2m1fn":
        recipe = {("float8_e4m3fn", 16): "nvfp4", ("float8_e8m0fnu", 32): "mxfp4"}.get(
            (sf, vec)
        )
        k = packed_k * 2
        expected_rhs_k = packed_k
        value_dtype = torch.uint8
    elif ab == "float8_e4m3fn":
        if options.get("block_fp8", False):
            recipe = "block_fp8" if (sf, vec) == ("float32", 128) else None
        elif (sf, vec) == ("float8_e8m0fnu", 32):
            recipe = "tensor_fp8" if options.get("plain_fp8", False) else "mxfp8"
        else:
            recipe = None
        k = expected_rhs_k = packed_k
        value_dtype = torch.float8_e4m3fn
    elif ab in ("float6_e2m3fn", "float6_e3m2fn"):
        fmt = "e2m3" if ab == "float6_e2m3fn" else "e3m2"
        a_fmt = options.get("a_fmt") or fmt
        b_fmt = options.get("b_fmt") or fmt
        expanded, packed = (
            options.get("b_preexpanded", False),
            options.get("b_packed", False),
        )
        if (
            not options.get("a_preexpanded", False)
            or expanded == packed
            or b_fmt != fmt
            or a_fmt not in (fmt, "e4m3")
            or (sf, vec) != ("float8_e8m0fnu", 32)
        ):
            raise ValueError(
                "dense FP6 search requires preexpanded A and declared packed or expanded B"
            )
        recipe = ("w6a8_" if a_fmt == "e4m3" else "mxfp6_") + fmt
        storage = "expanded" if expanded else "packed"
        k = packed_k
        expected_rhs_k = k if expanded else 3 * k // 4
        value_dtype = torch.uint8
        if a_fmt == "e4m3" and a.dtype != torch.float8_e4m3fn:
            raise ValueError("W6A8 search requires E4M3 activation values")
    else:
        recipe = None
    if recipe is None:
        raise ValueError("dense operand recipe has no declared search contract")
    expected = operand_options(
        DenseGemmQuery(
            recipe=recipe,
            entry_point=entry_point,
            weight_storage=storage,
            output_dtype=options["c_dtype"],
            batch=batch,
            max_rows=m,
            in_features=k,
            out_features=n,
        )
    )
    for name in (
        "plain_fp8",
        "block_fp8",
        "a_preexpanded",
        "b_preexpanded",
        "b_packed",
    ):
        if recipe == "block_fp8" and name == "plain_fp8":
            continue
        if options.get(name, False) != expected.get(name, False):
            raise ValueError(f"dense {name} does not match the declared recipe")
    if not recipe.startswith(("mxfp6_", "w6a8_")) and any(
        options.get(name) is not None for name in ("a_fmt", "b_fmt")
    ):
        raise ValueError("dense format overrides require an FP6 recipe")
    if (
        rhs_k != expected_rhs_k
        or b.dtype != value_dtype
        or (not recipe.startswith("w6a8_") and a.dtype != value_dtype)
    ):
        raise ValueError("dense operand storage differs from the declared recipe")
    for rows, scales, weight in ((m, sfa, False), (n, sfb, True)):
        if recipe == "block_fp8":
            shape = (rows // 128 if weight else rows, k // 128)
            strides = (shape[1], 1)
        else:
            row_blocks, k_blocks = (rows + 127) // 128, (k // vec + 3) // 4
            shape = (32, 4, row_blocks, 4, k_blocks, batch)
            strides = (16, 4, k_blocks * 512, 1, 512, row_blocks * k_blocks * 512)
        if (
            tuple(scales.shape) != shape
            or scales.stride() != strides
            or scales.dtype != getattr(torch, sf)
            or scales.device != a.device
        ):
            raise ValueError(
                "dense scale storage differs from the measured grouped or block layout"
            )
    alpha = options.get("alpha")
    alpha_dtype = options.get("alpha_dtype") or (
        "float32" if alpha is None else str(alpha.dtype).split(".")[-1]
    )
    if alpha_dtype != "float32" or (
        alpha is not None
        and (
            alpha.dtype != torch.float32
            or alpha.numel() != 1
            or alpha.device != a.device
        )
    ):
        raise ValueError("dense search requires unit alpha or a scalar FP32 tensor")
    workspace = options.get("_split_k_workspace")
    if workspace is not None and (
        workspace.dtype != torch.float32 or workspace.device != a.device
        or not workspace.is_contiguous() or (workspace.numel() and workspace.data_ptr() % 16)
    ):
        raise ValueError("dense workspace must be aligned contiguous FP32 storage on the operand device")
    query = DenseGemmQuery(
        recipe=recipe,
        entry_point=entry_point,
        weight_storage=storage,
        output_dtype=options["c_dtype"],
        batch=batch,
        max_rows=m,
        in_features=k,
        out_features=n,
        output_mode="functional" if out is None else "provided",
        alpha_mode="unit" if alpha is None else "tensor",
        expected_m=expected_m,
        sm_count=options.get("sm_count"),
        sfb_k_replicated=options.get("sfb_k_replicated", False),
        workspace_nbytes=None if workspace is None else workspace.numel() * workspace.element_size(),
        workspace_form="owned" if workspace is None else "provided",
        overrides=FrozenMapping({
            name: options[name] for name in (
                "mma_tiler_mn", "load_path", "swap_ab", "_tile_k_override",
                "_split_k_slices_override", "_large_m_unroll_override", "_target_occupancy_override",
            ) if options.get(name) is not None
        }),
    )
    validate_query(query)
    if out is not None and (
        out.shape != (m, n, batch)
        or out.dtype != getattr(torch, query.output_dtype)
        or out.device != a.device
        or out.stride(1) != 1
    ):
        raise ValueError(
            "dense output storage differs from the declared search contract"
        )
    return query


def launch_options(query, config):
    """Translate decisions to existing host overrides without changing specialization."""
    options = dict(
        mma_tiler_mn=(config.tile_m, config.tile_n),
        load_path=config.load_path,
        swap_ab=config.swap_ab,
    )
    if query.recipe in ("nvfp4", "mxfp4", "mxfp8"):
        options["_tile_k_override"] = config.tile_k
    for name in ("split_k_slices", "large_m_unroll", "target_occupancy"):
        value = getattr(config, name)
        if value is not None:
            options[f"_{name}_override"] = value
    return options


def knob_values(query):
    """Recipe-specific host knob choices; validate_config applies interdependent launch constraints."""
    fp4 = query.recipe in ("nvfp4", "mxfp4")
    fp8 = query.recipe in ("mxfp8", "tensor_fp8", "block_fp8")
    return dict(
        backend=("cutedsl",),
        tile_m=(16, 32, 64, 128),
        tile_n=(16, 32, 64, 128),
        tile_k=(128, 256, 512)
        if fp4
        else (64, 128)
        if query.recipe in ("mxfp8", "tensor_fp8")
        else (128,),
        load_path=("tma", "cpasync") if query.recipe == "nvfp4" else ("tma",),
        swap_ab=(False, True)
        if query.recipe in ("nvfp4", "mxfp8", "tensor_fp8")
        else (False,),
        split_k_slices=(1, 2, 4) if query.recipe in ("mxfp8", "block_fp8") else (None,),
        large_m_unroll=(False, True) if fp8 and query.batch == 1 else (None,),
        target_occupancy=(1, 2, 3, 4)
        if fp4 and query.output_mode == "provided"
        else (None,),
    )


def validate_config(query, config, device):
    from b12x._lib import dense_gemm as dense

    if not isinstance(config, DenseGemmConfig):
        raise ValueError("dense GEMM requires a typed launch config")
    for name, values in knob_values(query).items():
        value = getattr(config, name)
        if not any(
            type(value) is type(allowed) and value == allowed for allowed in values
        ):
            raise ValueError(f"dense GEMM {name} is outside its recipe/caller domain")
    selected_options = launch_options(query, config)
    for name, value in query.overrides.items():
        if selected_options.get(name) != value:
            raise ValueError(f"dense configuration conflicts with caller constraint {name}")
    options = operand_options(query)
    if not dense.DenseGemmKernel.can_implement(
        dense.get_cutlass_dtype(options["ab_dtype"]),
        dense.get_cutlass_dtype(options["sf_dtype"]),
        options["sf_vec_size"],
        dense.get_cutlass_dtype(query.output_dtype),
        (config.tile_m, config.tile_n),
        (1, 1),
        query.out_features,
        query.in_features,
        query.batch,
        "k",
        "k",
        "n",
        load_path=config.load_path,
        swap_ab=config.swap_ab,
        block_fp8=query.recipe == "block_fp8",
    ):
        raise ValueError("dense GEMM tile/layout violates the production MMA contract")
    if query.in_features % config.tile_k:
        raise ValueError("dense GEMM K must divide into complete staged tiles")
    if query.recipe in ("nvfp4", "mxfp4"):
        # FP4 allocates at least one A/B/scale stage and one complete C tile.
        sizes = (
            config.tile_m * config.tile_k // 2,
            config.tile_n * config.tile_k // 2,
            max(128, config.tile_m) * config.tile_k // options["sf_vec_size"],
            max(128, config.tile_n) * config.tile_k // options["sf_vec_size"],
            config.tile_m * config.tile_n * 2,
        )
        minimum_smem = 1024 + sum((size + 1023) // 1024 * 1024 for size in sizes)
        if minimum_smem > dense.utils.get_smem_capacity_in_bytes("sm_120"):
            raise ValueError(
                f"dense GEMM requires at least {minimum_smem} shared-memory bytes"
            )
    if query.recipe == "tensor_fp8":
        if device is None:
            raise ValueError("tensor-FP8 tile K requires the device SM count")
        tile_k = (
            128
            if config.swap_ab
            else dense._select_mxfp8_tile_k(
                query.max_rows,
                query.out_features,
                query.in_features,
                query.expected_m,
                query.sm_count if query.sm_count is not None else device.sm_count,
            )
        )
        if config.tile_k != tile_k:
            raise ValueError(
                "tensor-FP8 tile K must match the production-derived value"
            )
    if query.recipe in ("mxfp8", "tensor_fp8"):
        dense._validate_mxfp8_bk64_plan(
            config.tile_k, (config.tile_m, config.tile_n), config.swap_ab
        )
    if config.split_k_slices is not None and config.split_k_slices > 1:
        if (
            query.max_rows > min(8, config.tile_m)
            or query.batch != 1
            or config.swap_ab
            or query.in_features % (config.tile_k * config.split_k_slices)
        ):
            raise ValueError(
                "dense GEMM split-K requires unswapped single-batch decode and divisible K"
            )
        if config.split_k_slices == 4 and not dense._B12X_DENSE_SPLITK_TURBO:
            raise ValueError(
                "four-way split-K requires the production atomic reduction path"
            )
        if query.output_dtype != "bfloat16" and dense._B12X_DENSE_SPLITK_TURBO:
            raise ValueError("atomic BF16 split-K requires BF16 output")
    from ._preparation import _configured_lowering
    p = _configured_lowering(query, config, device)
    if query.workspace_nbytes is not None and p.policy.split_k_slices > 1 and not p.policy.split_k_atomic_bf16:
        required = p.policy.split_k_slices * query.max_rows * query.out_features * 4
        if query.workspace_nbytes < required:
            raise ValueError("dense caller workspace is smaller than this configuration requires")


def default_config(query, device):
    from b12x._lib import dense_gemm as dense
    from ._preparation import _default_lowering
    p = _default_lowering(query, device)
    fp4 = query.recipe in ("nvfp4", "mxfp4")
    fp8 = query.recipe in ("mxfp8", "tensor_fp8", "block_fp8")
    occupancy = None
    if fp4 and query.output_mode == "provided":
        occupancy = p.target_occupancy_override
        if occupancy is None:
            occupancy = dense._dense_gemm_target_occupancy(
                n=p.n, k=p.k, l=p.l, ab_dtype=dense.get_cutlass_dtype(p.ab_dtype),
                c_dtype=dense.get_cutlass_dtype(p.c_dtype), tile_k=p.tile_k,
                mma_tiler_mn=p.mma_tiler_mn, cluster_shape_mn=p.cluster_shape_mn,
                sm_count=p.sm_count, load_path=p.load_path, swap_ab=p.swap_ab,
                b_tile_major=p.b_tile_major,
            )
    return DenseGemmConfig(
        backend="cutedsl", tile_m=p.mma_tiler_mn[0], tile_n=p.mma_tiler_mn[1],
        tile_k=p.tile_k, load_path=p.load_path, swap_ab=p.swap_ab,
        split_k_slices=p.policy.split_k_slices if query.recipe in ("mxfp8", "block_fp8") else None,
        large_m_unroll=p.policy.large_m_unroll if fp8 and query.batch == 1 else None,
        target_occupancy=occupancy,
    )


def _parameters(query, device):
    values = knob_values(query)
    mapping = {
        "load_path": "load_path", "swap_ab": "swap_ab",
        "_tile_k_override": "tile_k", "_split_k_slices_override": "split_k_slices",
        "_large_m_unroll_override": "large_m_unroll", "_target_occupancy_override": "target_occupancy",
    }
    for name, value in query.overrides.items():
        if name == "mma_tiler_mn":
            values["tile_m"], values["tile_n"] = (value[0],), (value[1],)
        else:
            values[mapping[name]] = (value,)

    def short_unswapped_async(p):
        return (query.recipe != "nvfp4" or p["load_path"] != "cpasync"
                or (not p["swap_ab"] and query.in_features <= 256))

    def bounded_row_padding(p):
        row_tile = p["tile_n"] if p["swap_ab"] else p["tile_m"]
        if query.recipe == "nvfp4":
            return row_tile < 128 or query.max_rows >= 64 or query.in_features <= 512
        if query.recipe in ("mxfp8", "block_fp8", "tensor_fp8"):
            return row_tile < 128 or query.max_rows >= 64 or p["tile_k"] == 64
        return True

    def narrow_output_swap(p):
        return (query.recipe != "mxfp8" or not p["swap_ab"]
                or query.max_rows > 16 or 2 * query.out_features <= query.in_features)

    def bounded_prefill_k_tile(p):
        return (query.recipe != "nvfp4" or p["tile_k"] != 512
                or query.max_rows <= 256 or query.in_features <= 1024
                or (query.out_features <= 1024 and p["tile_m"] == 64))

    def reuse_rows(p):
        row_tile = p["tile_n"] if p["swap_ab"] else p["tile_m"]
        return query.recipe != "mxfp8" or query.max_rows <= 128 or row_tile >= 32

    return ParameterSpace.create(
        TUNING.knobs, values=values, exhaustive=query.exhaustive,
        efficiency_predicates=() if query.overrides or query.batch != 1 else (
            short_unswapped_async, bounded_row_padding, narrow_output_swap,
            bounded_prefill_k_tile, reuse_rows,
        ),
    )


def _validate_query(query, device):
    if not isinstance(query, DenseGemmQuery):
        raise TypeError("query must be DenseGemmQuery")
    validate_query(query)


TUNING = TuningContract(
    component_id="gemm.mm",
    query_schema_version=7,
    config_schema_version=2,
    query_fields=frozenset(field.name for field in fields(DenseGemmQuery)),
    config_fields=frozenset(field.name for field in fields(DenseGemmConfig)),
    encode_query=lambda query: {field.name: getattr(query, field.name) for field in fields(query)},
    encode_config=asdict,
    decode_config=lambda payload: DenseGemmConfig(**dict(payload)),
    default_config=default_config,
    validate_query=_validate_query,
    validate_config=validate_config,
    candidate_contract_version=4,
    knobs=(
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(name="tile_m", values=(16, 32, 64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_n", values=(16, 32, 64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_k", values=(64, 128, 256, 512), binding=ParameterBinding.COMPILE),
        Knob(name="load_path", values=("tma", "cpasync"), binding=ParameterBinding.COMPILE),
        Knob(name="swap_ab", values=(False, True), binding=ParameterBinding.COMPILE),
        Knob(name="split_k_slices", values=(None, 1, 2, 4), binding=ParameterBinding.COMPILE),
        Knob(name="large_m_unroll", values=(None, False, True), binding=ParameterBinding.COMPILE),
        Knob(name="target_occupancy", values=(None, 1, 2, 3, 4), binding=ParameterBinding.COMPILE),
    ),
    parameters=_parameters,
)
