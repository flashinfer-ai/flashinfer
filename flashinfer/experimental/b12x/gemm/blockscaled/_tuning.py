"""Complete packed-call precision decisions and fixed wrapper contracts."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec

from dataclasses import asdict, dataclass, replace

from b12x.preparation import BackendConfig, FrozenMapping, make_fixed_contract
from b12x.preparation.tuning import (
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.gemm._tuning import _codegen_snapshot


@dataclass(frozen=True, kw_only=True)
class BlockscaledQuery:
    recipe: str
    num_tokens: int
    in_features: int
    padded_in_features: int
    out_features: int
    activation_mode: str = "auto"
    activation_scale_available: bool = False
    global_scale_kind: str | None = None
    source_contiguous: bool = True
    source_aligned: bool = True
    output_mode: str = "functional"
    workspace_form: str = "owned"
    workspace_nbytes: int | None = None
    expected_m: int | None = None
    codegen: FrozenMapping | None = None

    def __post_init__(self):
        if self.global_scale_kind is None:
            object.__setattr__(
                self,
                "global_scale_kind",
                "multiplier" if self.recipe == "nvfp4" else "none",
            )
        object.__setattr__(
            self,
            "codegen",
            _codegen_snapshot()
            if self.codegen is None
            else FrozenMapping(self.codegen),
        )


@dataclass(frozen=True, kw_only=True)
class BlockscaledConfig:
    mode: str
    tile_m: int | None = None
    tile_n: int | None = None
    tile_k: int | None = None
    split_k: int | None = None

    @classmethod
    def from_config(cls, payload):
        if set(payload) != {"mode", "tile_m", "tile_n", "tile_k", "split_k"}:
            raise ValueError(
                "packed precision config requires mode and four nullable A16 knobs"
            )
        return cls(**dict(payload))

    def to_dict(self):
        return asdict(self)


def _validate_query(query, device):
    if not isinstance(query, BlockscaledQuery) or query.recipe not in (
        "nvfp4",
        "mxfp8",
        "iq2_xs",
        "iq2_xxs",
        "q8_0",
    ):
        raise ValueError(
            "packed BF16 execution requires NVFP4, MXFP8, or IQ2_XS weights"
        )
    if any(
        type(value) is not int or value <= 0
        for value in (
            query.num_tokens,
            query.in_features,
            query.padded_in_features,
            query.out_features,
        )
    ):
        raise ValueError("packed call dimensions must be positive integers")
    if query.in_features > query.padded_in_features or query.in_features % 8:
        raise ValueError("packed logical K must be aligned and fit its stored K")
    if query.padded_in_features % (16 if query.recipe == "nvfp4" else 32):
        raise ValueError("stored K must match the weight scale group")
    if query.recipe in BLOCK_CODECS:
        if (
            query.in_features != query.padded_in_features
            or query.in_features % block_codec(query.recipe).block_weights
        ):
            raise ValueError("IQ2_XS requires unpadded K divisible by 256")
        if query.activation_mode == "quantized" or query.activation_scale_available:
            raise ValueError(
                "IQ2_XS requires BF16 activations without activation scaling"
            )
    if query.activation_mode not in ("auto", "a16", "quantized"):
        raise ValueError("activation mode must be auto, a16, or quantized")
    if query.output_mode not in (
        "functional",
        "provided",
    ) or query.workspace_form not in ("provided", "owned"):
        raise ValueError("unknown packed output or workspace form")
    kinds = ("multiplier", "reciprocal") if query.recipe == "nvfp4" else ("none",)
    if query.global_scale_kind not in kinds:
        raise ValueError("weight global-scale semantics do not match the recipe")
    if any(
        type(value) is not bool
        for value in (
            query.activation_scale_available,
            query.source_contiguous,
            query.source_aligned,
        )
    ):
        raise TypeError("packed availability and layout coordinates must be boolean")
    if query.expected_m is not None and (
        type(query.expected_m) is not int or query.expected_m <= 0
    ):
        raise ValueError("expected_m must be positive or None")
    if query.workspace_nbytes is not None and (
        type(query.workspace_nbytes) is not int or query.workspace_nbytes < 0
    ):
        raise ValueError("workspace limit must be nonnegative or None")
    if query.codegen != _codegen_snapshot():
        raise ValueError("packed declaration code-generation snapshot changed")


def _automatic_a16(query, device):
    return (
        device is not None
        and device.compute_capability in ((12, 0), (12, 1))
        and query.in_features == query.padded_in_features
        and query.in_features % 32 == 0
        and query.out_features % 8 == 0
        and query.source_contiguous
        and query.source_aligned
        and query.num_tokens <= 8
    )


def _default_config(query, device):
    if query.activation_mode == "quantized":
        return BlockscaledConfig(mode="quantized")
    if _automatic_a16(query, device):
        return _a16_default_config(query, device)
    if query.activation_mode == "a16" or query.recipe in BLOCK_CODECS:
        return _a16_default_config(query, device)
    return BlockscaledConfig(mode="quantized")


def _a16_default_config(query, device):
    if query.recipe == "mxfp8" or device is None:
        if _automatic_a16(query, device):
            return BlockscaledConfig(mode="a16", tile_n=128, tile_k=64, split_k=4)
        return BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=1)
    m, n, k = query.num_tokens, query.out_features, query.in_features
    sms = device.sm_count
    iq2 = query.recipe in ("iq2_xs", "iq2_xxs")
    if query.recipe in BLOCK_CODECS and m <= 8:
        # SIMT avoids tile staging for N8 packing and low-reuse GEMVs.
        if n % 128 or (m <= 2 and ((iq2 and n >= k) or k <= 1024)):
            rows = max(1, m // 4) if n % 128 else 1 << (m - 1).bit_length()
            return BlockscaledConfig(
                mode="a16",
                tile_m=rows,
                tile_n=4,
                tile_k=256,
                split_k=1,
            )
        if n >= 128 * sms:
            return BlockscaledConfig(
                mode="a16",
                tile_m=8,
                tile_n=64,
                tile_k=256,
                split_k=1,
            )
    tile_m = 16 if iq2 else max(16, min(64, 1 << (m.bit_length() - 1)))
    if query.recipe == "nvfp4" and m > 16 and (n + 127) // 128 >= sms / 2:
        tile_m = 32
    rows = (m + tile_m - 1) // tile_m
    wide = rows * ((n + 127) // 128) >= sms / 2
    tile_n = (
        128
        if (
            query.recipe == "nvfp4"
            or (m > 16 and (iq2 or n % 128 or wide))
            or (query.recipe == "q8_0" and n >= k)
        )
        else 64
    )
    tile_k = (128 if m <= 16 else 256) if iq2 else 64
    grid = rows * ((n + tile_n - 1) // tile_n)
    splits = [1]
    if grid < sms:
        for split in (2, 4, 8):
            workspace = split * m * n * 4
            if k // split >= 512 and (
                query.workspace_nbytes is None or workspace <= query.workspace_nbytes
            ):
                splits.append(split)
    # Extra partitions must offset their partial-output and reduction overhead.
    reduction_penalty = 0.005 if iq2 else 0.025
    split_k = min(
        splits,
        key=lambda split: (
            ((grid * split + sms - 1) // sms) / split + reduction_penalty * (split - 1)
        ),
    )
    if query.recipe == "nvfp4" and m <= 16 and split_k == 1:
        tile_k = 128
    return BlockscaledConfig(
        mode="a16",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
    )


def functional_mxfp8_quantization(query, config):
    return (
        query.recipe == "mxfp8"
        and config.mode == "quantized"
        and query.output_mode == "functional"
        and query.workspace_form == "owned"
    )


def effective_a16_config(query, config):
    return (
        config.tile_n,
        config.tile_k,
        min(config.split_k, (query.in_features + config.tile_k - 1) // config.tile_k),
    )


def _validate_config(query, config, device):
    if not isinstance(config, BlockscaledConfig) or config.mode not in (
        "a16",
        "quantized",
    ):
        raise ValueError("invalid packed precision configuration")
    if query.activation_mode != "auto" and config.mode != query.activation_mode:
        raise ValueError("configuration conflicts with caller activation precision")
    if query.recipe in BLOCK_CODECS and config.mode != "a16":
        raise ValueError("IQ2_XS supports only A16 execution")
    if config.mode == "a16":
        simt = query.recipe in BLOCK_CODECS and config.tile_n == 4
        row_tiles = (
            (1, 2, 4, 8)
            if simt
            else (8, 16, 32, 64)
            if query.recipe in BLOCK_CODECS
            else (16, 32, 64)
        )
        if (
            (simt and config.tile_m is None)
            or config.tile_m is not None
            and (type(config.tile_m) is not int or config.tile_m not in row_tiles)
        ):
            raise ValueError("invalid A16 row tile")
        if query.recipe == "mxfp8" and config.tile_m not in (None, 16):
            raise ValueError("MXFP8 A16 requires a 16-row tile")
        if not simt and config.tile_m == 8 and config.split_k != 1:
            raise ValueError("transposed IQ2_XS does not support split-K")
        if (
            type(config.tile_n) is not int
            or config.tile_n not in ((4,) if simt else (64, 128))
            or type(config.tile_k) is not int
            or config.tile_k
            not in (
                (256,)
                if simt
                else (64, 128, 256)
                if query.recipe in ("iq2_xs", "iq2_xxs", "q8_0", "nvfp4")
                else (64, 128)
            )
            or type(config.split_k) is not int
            or config.split_k not in ((1,) if simt else (1, 2, 4, 8))
        ):
            raise ValueError("invalid A16 launch geometry")
        if query.padded_in_features % 32 or query.out_features % 8:
            raise ValueError("A16 requires stored K32 and N8")
        if device is None or device.compute_capability not in ((12, 0), (12, 1)):
            raise ValueError("A16 requires SM120/SM121")
        if not query.source_contiguous or not query.source_aligned:
            raise ValueError("A16 requires contiguous 16-byte-aligned source storage")
        if query.workspace_nbytes is not None:
            split = effective_a16_config(query, config)[2]
            needed = (
                split * query.num_tokens * query.out_features * 4 if split > 1 else 0
            )
            if query.workspace_nbytes < needed:
                raise ValueError("caller workspace is too small for A16 split-K")
    else:
        if any(
            value is not None
            for value in (config.tile_m, config.tile_n, config.tile_k, config.split_k)
        ):
            raise ValueError("quantized execution has no A16 launch knobs")
        if query.padded_in_features % 128:
            raise ValueError("quantized execution requires stored K128")
        if query.workspace_nbytes is not None and not functional_mxfp8_quantization(
            query, config
        ):
            from ._a16 import _layout

            needed = _layout(
                query.num_tokens,
                query.out_features,
                query.padded_in_features,
                query.recipe == "nvfp4",
                (64, 64, 1),
            )[-1]
            if query.workspace_nbytes < needed:
                raise ValueError(
                    "caller workspace is too small for quantized execution"
                )
        if not functional_mxfp8_quantization(query, config):
            if query.out_features % 8:
                raise ValueError("workspace quantization requires N8")
            if not query.source_contiguous or not query.source_aligned:
                raise ValueError(
                    "workspace quantization requires contiguous aligned source storage"
                )
        if query.recipe == "nvfp4" and not query.activation_scale_available:
            raise ValueError("quantized NVFP4 requires the caller's activation scale")
        if query.recipe == "mxfp8" and query.activation_scale_available:
            raise ValueError("MXFP8 does not use an activation global scale")


def _tuning_parameters(query, device):
    del device

    def eligible(parameters):
        if parameters["mode"] != "a16":
            return True
        if parameters["tile_n"] == 4:
            return (
                query.recipe in BLOCK_CODECS
                and query.num_tokens <= 8
                and parameters["tile_m"] in (1, 2, 4, 8)
                and parameters["tile_k"] == 256
                and parameters["split_k"] == 1
            )
        if parameters["tile_m"] == 8:
            return (
                query.recipe in BLOCK_CODECS
                and parameters["split_k"] == 1
                and (parameters["tile_n"] == 64 or query.out_features > 64)
            )
        return (
            parameters["tile_m"] in (16, 32, 64)
            and parameters["tile_k"]
            in (
                (64, 128, 256)
                if query.recipe in ("iq2_xs", "iq2_xxs", "q8_0", "nvfp4")
                else (64, 128)
            )
            and (
                parameters["tile_m"] == 16
                or query.recipe != "mxfp8"
                and parameters["tile_m"] <= query.num_tokens
            )
            and (parameters["tile_n"] == 64 or query.out_features > 64)
        )

    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "mode": ("a16",)
            if query.recipe in BLOCK_CODECS
            else ("a16", "quantized")
            if query.activation_mode == "auto"
            else (query.activation_mode,)
        },
        predicates=(eligible,),
    )


def _equivalence(query, device, config):
    if config.mode == "quantized":
        return {"mode": "quantized"}
    n, k, split = effective_a16_config(query, config)
    return {
        "mode": "a16",
        "tile_m": config.tile_m or 16,
        "tile_n": n,
        "tile_k": k,
        "split_k": split,
    }


TUNING = TuningContract(
    component_id="gemm.blockscaled_precision",
    query_schema_version=22,
    config_schema_version=22,
    query_fields=frozenset(BlockscaledQuery.__dataclass_fields__),
    config_fields=frozenset(BlockscaledConfig.__dataclass_fields__),
    encode_query=lambda query: {
        name: getattr(query, name) for name in query.__dataclass_fields__
    },
    encode_config=BlockscaledConfig.to_dict,
    decode_config=BlockscaledConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    candidate_contract_version=19,
    knobs=(
        Knob(
            name="mode", values=("a16", "quantized"), binding=ParameterBinding.COMPILE
        ),
        Knob(
            name="tile_m",
            values=(1, 2, 4, 8, 16, 32, 64),
            when=FrozenMapping({"mode": "a16"}),
        ),
        Knob(name="tile_n", values=(4, 64, 128), when=FrozenMapping({"mode": "a16"})),
        Knob(name="tile_k", values=(64, 128, 256), when=FrozenMapping({"mode": "a16"})),
        Knob(name="split_k", values=(1, 2, 4, 8), when=FrozenMapping({"mode": "a16"})),
    ),
    parameters=_tuning_parameters,
    equivalence_key=_equivalence,
)


@dataclass(frozen=True, kw_only=True)
class FixedBlockscaledQuery:
    """Fixed kernel recipe with bounded dynamic M when expected_m is None."""

    recipe: str
    call_kind: str
    max_rows: int
    in_features: int
    padded_in_features: int
    out_features: int
    input_dtype: str
    output_dtype: str
    expected_m: int | None
    alpha_mode: str | None = None
    source_scale_form: str | None = None
    codegen: FrozenMapping | None = None

    def __post_init__(self):
        if self.alpha_mode is None:
            object.__setattr__(
                self, "alpha_mode", "tensor" if self.recipe == "tensor_fp8" else "unit"
            )
        if self.source_scale_form is None:
            form = (
                "none"
                if self.call_kind == "packed"
                and (self.recipe == "tensor_fp8" or self.input_dtype == "float16")
                else "block"
                if self.recipe == "block_fp8"
                else "swizzled"
            )
            object.__setattr__(self, "source_scale_form", form)
        object.__setattr__(
            self,
            "codegen",
            _codegen_snapshot()
            if self.codegen is None
            else FrozenMapping(self.codegen),
        )


def _validate_fixed_query(query, device):
    if not isinstance(query, FixedBlockscaledQuery):
        raise TypeError("query must be FixedBlockscaledQuery")
    if query.codegen != _codegen_snapshot():
        raise ValueError("fixed packed code-generation snapshot changed")
    if (
        min(query.max_rows, query.in_features, query.out_features) <= 0
        or (query.expected_m is not None and query.expected_m <= 0)
        or query.padded_in_features < query.in_features
        or query.output_dtype not in ("bfloat16", "float16")
        or query.alpha_mode not in ("unit", "tensor")
    ):
        raise ValueError("invalid fixed blockscaled metadata")
    if query.call_kind == "serialized":
        if (
            query.recipe not in ("nvfp4", "mxfp4", "block_fp8")
            or query.in_features != query.padded_in_features
            or query.in_features % (256 if query.recipe == "mxfp4" else 128)
            or query.out_features % (128 if query.recipe == "block_fp8" else 8)
        ):
            raise ValueError("unsupported serialized blockscaled recipe or geometry")
        dtype = "float8_e4m3fn" if query.recipe == "block_fp8" else "uint8"
        if query.input_dtype != dtype:
            raise ValueError("serialized values do not match the operand recipe")
    elif query.call_kind == "packed":
        if query.padded_in_features % 128:
            raise ValueError("packed weights require stored K128")
        if query.recipe == "tensor_fp8":
            if query.input_dtype != "float8_e4m3fn":
                raise ValueError("tensor-FP8 requires prequantized E4M3 activations")
            if query.alpha_mode != "tensor":
                raise ValueError("tensor-FP8 requires its combined scale tensor")
        elif query.recipe == "mxfp8":
            if query.input_dtype not in ("float16", "float8_e4m3fn"):
                raise ValueError("BF16 packed MXFP8 belongs to the precision contract")
            if query.input_dtype == "float16" and query.output_dtype != "float16":
                raise ValueError("plain packed MXFP8 preserves source dtype")
            if query.alpha_mode != "unit":
                raise ValueError("packed MXFP8 has no global output scale")
        else:
            raise ValueError("packed NVFP4 belongs to the BF16 precision contract")
    else:
        raise ValueError("native 3D operands belong to the native GEMM contract")
    if query.source_scale_form not in ("none", "swizzled", "mma", "compact", "block"):
        raise ValueError("unknown activation scale storage form")


FIXED_TUNING = replace(
    make_fixed_contract(
        component_id="gemm.blockscaled.fixed",
        query_type=FixedBlockscaledQuery,
        backend="cutedsl",
    ),
    query_schema_version=22,
    validate_query=_validate_fixed_query,
)
