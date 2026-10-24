"""Geometry contract for vectorized paged K/V writes."""

from dataclasses import dataclass

from b12x.preparation import make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class PagedKVWriteQuery:
    num_kv_heads: int
    head_dim_k: int
    head_dim_v: int
    page_size: int
    kv_dtype: str

    def __post_init__(self):
        if any(
            type(value) is not int or not 0 < value < 2**31
            for value in (
                self.num_kv_heads,
                self.head_dim_k,
                self.head_dim_v,
                self.page_size,
            )
        ):
            raise ValueError("paged KV dimensions must be positive Int32 integers")
        if self.kv_dtype not in ("bfloat16", "float8_e4m3fn"):
            raise ValueError("paged KV writes require BF16 or FP8 E4M3 caches")
        vector = 16 if self.kv_dtype == "float8_e4m3fn" else 8
        if self.head_dim_k % vector or self.head_dim_v % vector:
            raise ValueError(f"paged KV head dimensions must be multiples of {vector}")
        if self.num_kv_heads * (self.head_dim_k + self.head_dim_v) >= 2**31:
            raise ValueError("paged KV row width exceeds the Int32 indexing domain")


TUNING = make_fixed_contract(
    component_id="attention.paged_kv_write",
    query_type=PagedKVWriteQuery,
    backend="cute",
)
