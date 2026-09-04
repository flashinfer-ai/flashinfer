"""Mock experimental feature (b), used only to verify the experimental-track CI path.

Throwaway. Not part of any release, not registered for AOT, not exported from
the top-level package.
"""

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def mock_scale_b(x, factor=2):
    """Scale ``x`` by ``factor``. Exists only to be called by a test."""
    return x * factor
