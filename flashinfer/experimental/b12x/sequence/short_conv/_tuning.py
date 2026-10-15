"""Fixed packed-request convolution geometry and launch layouts."""

from dataclasses import dataclass

from b12x.preparation import make_fixed_contract

OPERANDS = (
    "x",
    "conv_weight",
    "query_start_loc",
    "state_slot_ids",
    "state_is_fresh",
    "num_accepted_tokens",
    "request_is_prefill",
    "num_seqs",
    "num_tokens",
    "conv_state",
    "out",
    "gathered_state",
    "request_ids",
)


@dataclass(frozen=True, kw_only=True)
class ShortConvQuery:
    mode: str
    channels: int
    kernel_size: int
    dilation: int
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    max_speculative_tokens: int
    weight_dtype: str = "bfloat16"
    state_strides: tuple[int, int, int] | None = None
    input_row_stride: int | None = None
    input_alignments: tuple[int, ...] = (16,) * len(OPERANDS)

    def __post_init__(self):
        if self.weight_dtype not in ("bfloat16", "float32"):
            raise ValueError("short convolution weights require BF16 or FP32")
        if self.mode not in ("decode", "prefill", "mixed"):
            raise ValueError("short convolution mode must be decode, prefill, or mixed")
        if any(
            type(value) is not int or not 0 < value < 2**31
            for value in (
                self.channels,
                self.kernel_size,
                self.dilation,
                self.max_tokens,
                self.max_seqs,
                self.max_state_slots,
            )
        ):
            raise ValueError(
                "short convolution dimensions must be positive Int32 integers"
            )
        if self.kernel_size < 2:
            raise ValueError("short convolution requires at least two taps")
        if (
            type(self.max_speculative_tokens) is not int
            or self.max_speculative_tokens < 0
        ):
            raise ValueError(
                "short convolution speculative capacity must be nonnegative"
            )
        input_stride = (
            self.channels if self.input_row_stride is None else self.input_row_stride
        )
        if type(input_stride) is not int or input_stride < self.channels:
            raise ValueError(
                "short convolution input rows must have a nonoverlapping positive stride"
            )
        if input_stride * self.max_tokens >= 2**63:
            raise ValueError(
                "short convolution input exceeds the Int64 indexing domain"
            )
        object.__setattr__(self, "input_row_stride", input_stride)
        capacity = self.state_capacity
        if capacity >= 2**31:
            raise ValueError(
                "short convolution history exceeds the Int32 indexing domain"
            )
        strides = (
            (self.channels * capacity, capacity, 1)
            if self.state_strides is None
            else tuple(self.state_strides)
        )
        if len(strides) != 3 or any(type(s) is not int or s <= 0 for s in strides):
            raise ValueError(
                "short convolution history requires three positive strides"
            )
        if strides[1:] != (capacity, 1) or strides[0] < self.channels * capacity:
            raise ValueError(
                "short convolution history must be dense within nonoverlapping slots"
            )
        if strides[0] * self.max_state_slots >= 2**63:
            raise ValueError("short convolution pool exceeds the Int64 indexing domain")
        alignments = tuple(self.input_alignments)
        if len(alignments) != len(OPERANDS) or any(
            type(a) is not int or a not in (1, 2, 4, 8, 16) for a in alignments
        ):
            raise ValueError("short convolution requires every operand's alignment")
        object.__setattr__(self, "state_strides", strides)
        object.__setattr__(self, "input_alignments", alignments)

    @property
    def state_length(self):
        return self.dilation * (self.kernel_size - 1)

    @property
    def state_capacity(self):
        return self.state_length + self.max_speculative_tokens


TUNING = make_fixed_contract(
    component_id="sequence.short_conv",
    query_type=ShortConvQuery,
    backend="triton",
)
