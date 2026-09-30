"""Immutable shape and layout contract for exact embedding lookup."""
from dataclasses import dataclass, replace

from b12x.preparation import make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class EmbeddingQuery:
    max_rows: int
    table_rows: int
    width: int
    row_stride: int
    weight_dtype: str
    id_dtype: str
    device_count: bool = False

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.max_rows, self.table_rows)):
            raise ValueError("embedding row bounds must be nonnegative integers")
        if type(self.width) is not int or self.width <= 0 or type(self.row_stride) is not int or self.row_stride < self.width:
            raise ValueError("embedding requires a positive width and nonoverlapping rows")
        if self.max_rows >= 2**31 or self.row_stride * self.table_rows >= 2**63:
            raise ValueError("embedding bounds exceed the native indexing domain")
        if self.weight_dtype not in ("bfloat16", "float32", "uint8") or self.id_dtype not in ("int32", "int64"):
            raise TypeError("embedding requires BF16/FP32 weights and Int32/Int64 IDs")
        if self.weight_dtype == "uint8" and (self.width % 32 or self.row_stride < self.width // 32 * 34):
            raise ValueError("Q8_0 embedding requires K32 and complete 34-byte blocks")
        if type(self.device_count) is not bool or (self.device_count and self.max_rows == 0):
            raise ValueError("device-count embedding requires a positive row capacity")


TUNING = make_fixed_contract(
    component_id="sequence.embedding", query_type=EmbeddingQuery, backend="cute",
)

TUNING = replace(TUNING, query_schema_version=2)
