"""CuTeDSL NVFP4 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.cutedsl_megamoe.nvfp4_mega_moe``.

    Expert weights must be NVFP4 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    NVFP4 weights with ``w13_scale`` / ``w2_scale``. ``activation="relu2"``
    accepts a semantic I-wide W1 plane. ``relu2_kernel="padded"`` preserves
    the compatibility adapter that expands FC1 to 2*I; ``"single_plane"``
    selects the native I-wide kernel.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    apply_topk_in_fc1: bool = True
    # In-flight top-k combine: cross-rank REDG atomic-add collapses the combine
    # as peer data arrives (no per-topk staging / explicit tail reduce).
    # ~1-2% faster and removes the multi-GB combine staging from the symmetric
    # workspace.  Requires apply_topk_in_fc1=True and combine_dtype="bf16";
    # accumulation order is nondeterministic (tolerance-compare outputs).
    in_kernel_fc2_reduce: bool = False
    # Cross-rank combine wire format: "bf16" (exact), "mxfp8" (2x less combine
    # traffic), "nvfp4" (4x less).  Quantized wires trade accuracy for NVLink
    # bandwidth and require in_kernel_fc2_reduce=False.
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"
    input_norm_const: float = 1.0
    fc1_alpha: Optional["torch.Tensor"] = None
    fc2_alpha: Optional["torch.Tensor"] = None
    fc1_norm_const: Optional["torch.Tensor"] = None
    # Kernel tuning knobs (see kernel_src.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(num_max_tokens).
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None
    # Appended for positional-constructor compatibility with the original
    # dataclass field order. New integrations should always pass it by name.
    activation: Literal["swiglu", "relu2"] = "swiglu"
    # Appended after activation for the same positional-compatibility reason.
    # Padded remains the default until the native kernel is fully validated.
    relu2_kernel: Literal["padded", "single_plane"] = "padded"

    def __post_init__(self) -> None:
        if self.activation not in ("swiglu", "relu2"):
            raise ValueError(
                f"activation must be 'swiglu' or 'relu2', got {self.activation!r}."
            )
        if self.relu2_kernel not in ("padded", "single_plane"):
            raise ValueError(
                "relu2_kernel must be 'padded' or 'single_plane', got "
                f"{self.relu2_kernel!r}."
            )
        if self.relu2_kernel == "single_plane" and self.activation != "relu2":
            raise ValueError("relu2_kernel='single_plane' requires activation='relu2'.")
        if self.activation == "relu2" and (
            self.gate_up_clamp is not None or self.activation_clamp is not None
        ):
            raise ValueError(
                "ReLU2 MegaMoE does not support gate_up_clamp or activation_clamp."
            )

    @property
    def layout_identity(self) -> str:
        """Compile/workspace/cache identity for the physical FC1 layout."""
        if self.activation == "swiglu":
            return "swiglu"
        return f"relu2_{self.relu2_kernel}"

    @property
    def physical_fc1_size(self) -> int:
        """Physical FC1 projection width consumed by the selected kernel."""
        if self.layout_identity == "relu2_single_plane":
            return self.intermediate_size
        return 2 * self.intermediate_size
