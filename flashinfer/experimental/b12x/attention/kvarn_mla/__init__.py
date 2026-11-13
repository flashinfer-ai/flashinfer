"""KVarN5 MLA staging for the native SparkInfer GLM sparse-attention path."""

from .api import stage_k5_as_fp8_records

__all__ = ["stage_k5_as_fp8_records"]
