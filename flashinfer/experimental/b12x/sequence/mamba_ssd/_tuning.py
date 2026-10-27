"""Immutable geometry and capacity for packed Mamba-2 scans."""

from dataclasses import asdict, dataclass

from b12x.preparation import Knob, TuningContract


@dataclass(frozen=True, kw_only=True)
class MambaQuery:
    heads: int
    groups: int
    head_dim: int
    dstate: int
    max_tokens: int
    max_seqs: int
    max_slots: int
    x_stride: int
    b_stride: int
    c_stride: int
    dt_stride: int
    state_stride: int
    state_dtype: str
    dt_dtype: str
    d_dtype: str
    bias_dtype: str
    index_dtype: str
    dt_softplus: bool = True

    def __post_init__(self):
        for name in ("heads", "groups", "max_tokens", "max_seqs", "max_slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Mamba {name} must be a positive integer")
        if (
            self.heads % self.groups
            or self.head_dim not in (64, 128)
            or self.dstate not in (64, 128)
        ):
            raise ValueError(
                "Mamba requires grouped heads and head/state dimensions of 64 or 128"
            )
        for name, minimum in (
            ("x_stride", self.heads * self.head_dim),
            ("b_stride", self.groups * self.dstate),
            ("c_stride", self.groups * self.dstate),
            ("dt_stride", self.heads),
            ("state_stride", self.heads * self.head_dim * self.dstate),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"Mamba {name} must cover its complete row")
        for name in ("state_dtype", "dt_dtype", "d_dtype", "bias_dtype"):
            if getattr(self, name) not in ("bfloat16", "float32"):
                raise TypeError(f"Mamba {name} must be bfloat16 or float32")
        if (
            self.index_dtype not in ("int32", "int64")
            or type(self.dt_softplus) is not bool
        ):
            raise TypeError(
                "Mamba requires integer state indices and a boolean dt_softplus"
            )
        if max(self.max_tokens, self.max_seqs, self.heads) >= 2**31:
            raise ValueError("Mamba launch capacities exceed Int32")
        if (
            max(
                self.max_slots * self.state_stride,
                self.max_tokens
                * max(self.x_stride, self.b_stride, self.c_stride, self.dt_stride),
            )
            >= 2**63
        ):
            raise ValueError("Mamba storage offsets exceed Int64")


@dataclass(frozen=True)
class MambaConfig:
    prepare_warps: int
    carry_elements: int
    output_rows: int


def _parameters(q, device):
    if q.max_tokens < 128:
        return dict(prepare_warps=(4,), carry_elements=(1,), output_rows=(16,))
    output_rows = (64, 128)
    return dict(
        prepare_warps=(4, 8, 16), carry_elements=(4, 8, 16), output_rows=output_rows
    )


def _validate_query(q, device):
    if not isinstance(q, MambaQuery):
        raise TypeError("Mamba scans require MambaQuery")


def _validate_config(q, c, device):
    if not isinstance(c, MambaConfig):
        raise TypeError("Mamba scans require MambaConfig")
    for name, values in _parameters(q, device).items():
        if getattr(c, name) not in values:
            raise ValueError(f"invalid Mamba {name}: {getattr(c, name)}")


TUNING = TuningContract(
    component_id="sequence.mamba_ssd",
    query_schema_version=1,
    config_schema_version=1,
    semantic_version=2,
    query_fields=frozenset(MambaQuery.__dataclass_fields__),
    config_fields=frozenset(MambaConfig.__dataclass_fields__),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda p: MambaConfig(**p),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda q, d: MambaConfig(16, 8, 64)
    if q.max_tokens >= 128
    else MambaConfig(4, 1, 16),
    knobs=tuple(
        Knob(name=name, values=None) for name in MambaConfig.__dataclass_fields__
    ),
    parameters=_parameters,
)
