try:
    from .gdn import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None  # type: ignore

__all__ = ["chunk_gated_delta_rule"]
