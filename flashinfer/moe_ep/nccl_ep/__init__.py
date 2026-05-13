"""NCCL-EP backend stub. Real implementation lands in Part B (B3-B4).

Importing this module succeeds even when the native libs are absent; calling
into the (yet-to-be-implemented) Fleet/Handle factory functions will raise
:class:`flashinfer.moe_ep.MoEEpNotBuiltError`.
"""

from __future__ import annotations
