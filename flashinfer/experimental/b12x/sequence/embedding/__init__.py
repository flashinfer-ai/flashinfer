"""BF16/FP32 and Q8_0 token-row lookup with caller-owned output and runtime counts.

Invalid live IDs raise a CUDA device error (also during graph replay); they
never select a safe zero row. Q8_0 rows are decoded directly to BF16.
"""
from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="embedding", group="sequence", api_style="oneshot",
    entry_points=("plan", "query_from_call", "run", "is_supported", "clear_caches", "EmbeddingQuery"),
    dtypes=("bfloat16", "float32", "uint8", "int32", "int64"),
    provenance=Provenance(
        repo="https://github.com/phaelon74/b12x",
        commit="75ffee6375b0577ce2c8d6931ffacefda3ecbdd6",
        paths=("b12x/_lib/compiler.py", "b12x/_lib/utils.py")),
    notes="Prepared plain/Q8_0 gather with runtime row IDs and optional device live count.",
    test_path="tests/experimental/b12x/sequence/test_embedding.py",
)

if TYPE_CHECKING:
    from .api import EmbeddingQuery, clear_caches, is_supported, plan, query_from_call, run

install_lazy_api(globals(), META)
