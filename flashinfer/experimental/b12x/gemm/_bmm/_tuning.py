"""Host execution contract for native rowwise-MXFP8 BMM."""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class BmmQuery:
    batch: int
    max_rows: int
    in_features: int
    out_features: int
    b_major: str
    sf_axis: str


def validate_query(query):
    from b12x.gemm._shared.mxfp8_bmm import can_implement

    if not can_implement(
        batch=query.batch,
        max_m=query.max_rows,
        n=query.out_features,
        k=query.in_features,
        b_major=query.b_major,
        sf_axis=query.sf_axis,
    ):
        raise ValueError("BMM plan requires a supported BF16 x MXFP8 geometry")


TUNING = make_fixed_contract(
    component_id="gemm.bmm",
    query_type=BmmQuery,
    backend="cutedsl",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query, device) -> None:
    _validate_fixed_query(query, device)
    validate_query(query)


TUNING = replace(TUNING, validate_query=_validate_query)
