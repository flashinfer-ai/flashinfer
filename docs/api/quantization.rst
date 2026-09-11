.. _apiquantization:

flashinfer.quantization
=======================

Quantization-related kernels for FP4, FP8, and packbits utilities.

.. currentmodule:: flashinfer.quantization

Types and Enums
---------------

.. autosummary::
    :toctree: ../generated

    SfLayout

NVFP4 4over6 recipe types. These are defined in
``flashinfer.quantization.nvfp4_quantization_utils`` and re-exported here;
only the canonical ``flashinfer.quantization`` spelling is documented, because
``sphinx -W`` treats a second description of the same object as an error.

.. autosummary::
    :toctree: ../generated

    NVFP4Recipe
    NVFP44Over6Config
    NVFP44Over6ErrMode

Packbits Utilities
------------------

.. autosummary::
    :toctree: ../generated

    packbits
    segment_packbits

NVFP4 4over6 Recipes
--------------------

Standard NVFP4 gives each 16-element block a single E4M3 block scale derived
from the block's absolute maximum: ``amax / 6``, because ``6`` is the largest
magnitude an E2M1 element can hold. "4over6" is a two-candidate block-scale
search — the quantizer also forms the 1.5x tighter ``amax / 4`` scale,
quantizes the block with both, dequantizes both, and keeps whichever
reconstructs the block with less error. The ``amax / 4`` candidate buys
resolution for the bulk of the block by clipping its largest magnitudes, so it
wins on blocks with no dominant outlier. The extra work is paid entirely inside
the quantizer: the result is ordinary NVFP4 and needs no change downstream.

4over6 requires fp16/bf16 input and E4M3 (not UE8M0) block scales.

Two knobs matter in practice:

``e4m3_max=256``
    Treat ``256``, not the full-range ``448``, as the top of the E4M3
    block-scale range. This is what keeps the ``amax / 4`` candidate
    representable: ``256 * 1.5 = 384`` still fits E4M3, whereas a 448-based
    global scale would need ``672`` and saturate for blocks near the tensor
    amax. It also changes the global scale — see the warning below.

``err_mode="MSE"``
    Score the two candidates by summed squared error instead of summed
    absolute error. MSE punishes the few large residuals that ``amax / 4``
    clipping creates much harder than MAE does, so it picks ``amax / 4`` less
    often. Prefer it when a handful of outlier activations dominate downstream
    accuracy.

``err_use_fast_math=True`` evaluates that error in fp16 rather than exactly in
fp32. It is cheaper, and on near-ties it can select the other candidate, so
treat it as a distinct recipe rather than a pure speed knob.

.. warning::

    The global scale must be built from the **same** recipe as the quantizer
    call. ``e4m3_max`` appears both in
    :func:`~flashinfer.quantization.make_nvfp4_global_scale` and in the
    kernel's candidate search; a mismatch silently rescales the whole tensor
    instead of raising.

.. code-block:: python

    import torch
    from flashinfer.quantization import (
        NVFP44Over6Config,
        make_nvfp4_global_scale,
        nvfp4_quantize,
    )

    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    recipe = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")

    # Same recipe object on both calls - see the warning above.
    global_sf = make_nvfp4_global_scale(
        x, per_token_activation=False, nvfp4_4over6=recipe
    )
    x_q, sf = nvfp4_quantize(x, global_sf, nvfp4_4over6=recipe)

Precedence
~~~~~~~~~~

Every NVFP4 entry point that used to read the environment now takes an
``nvfp4_4over6=`` keyword, typed ``NVFP44Over6Setting``, with three states:

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - ``nvfp4_4over6=``
      - Meaning
    * - ``NVFP4Recipe.FROM_ENV`` (default; ``None`` is an alias)
      - Read ``FLASHINFER_NVFP4_4OVER6``; when it is ``"1"``, the other three
        ``FLASHINFER_NVFP4_4OVER6_*`` variables supply the recipe. Read on
        every call. Byte-for-byte the behaviour that predates this parameter.
    * - ``NVFP4Recipe.STANDARD``
      - 4over6 off. ``FLASHINFER_NVFP4_4OVER6=1`` cannot turn it back on.
    * - ``NVFP44Over6Config(...)``
      - On with exactly this recipe. The environment is ignored, and there is
        **no** per-field merge: a field left at its dataclass default keeps
        that default rather than picking up the environment's value.

The four ``FLASHINFER_NVFP4_4OVER6*`` variables are documented in the
repository's ``CLAUDE.md``. They remain supported, but they are legacy: being
process-wide, they cannot express two models served in one process under
different recipes.

.. note::

    :func:`~flashinfer.quantization.make_nvfp4_global_scale` and
    :func:`~flashinfer.quantization.nvfp4_e4m3_max` default to
    ``NVFP4Recipe.STANDARD``, not ``FROM_ENV``. Their pre-existing contract is
    that passing nothing means "no 4over6", and promoting that to an
    environment read would change the scale returned to every existing caller.

Resolution happens in exactly one place,
:func:`~flashinfer.quantization.resolve_nvfp4_4over6`. Below it, every kernel
driver sees the two-state ``Optional[NVFP44Over6Config]`` in which ``None``
means off, and :func:`~flashinfer.quantization.nvfp4_4over6_code` packs that
resolved value into the ``int64`` the custom ops take.
:func:`~flashinfer.quantization.nvfp4_4over6_from_code` inverts the packing,
which is how logs and repro tooling recover the recipe a captured call actually
ran with.

.. autosummary::
    :toctree: ../generated

    resolve_nvfp4_4over6
    nvfp4_4over6_code
    nvfp4_4over6_from_code
    make_nvfp4_global_scale
    nvfp4_e4m3_max

FP4 Quantization
----------------

Core kernels for NVFP4 / MXFP4 (de)quantization and the scale-factor
layout helpers used by the FP4 GEMM/MoE pipelines.

.. autosummary::
    :toctree: ../generated

    fp4_quantize
    nvfp4_quantize
    nvfp4_batched_quantize
    mxfp4_quantize
    mxfp4_dequantize
    mxfp4_dequantize_host
    block_scale_interleave
    e2m1_and_ufp8sf_scale_to_float
    scaled_fp4_grouped_quantize
    silu_and_mul_nvfp4_quantize
    shuffle_matrix_a
    shuffle_matrix_sf_a

.. note::

    ``flashinfer.quantization.nvfp4_block_scale_interleave`` is an alias
    for :func:`block_scale_interleave` (same Python object). Use either
    name; we document the canonical ``block_scale_interleave`` to avoid
    Sphinx ``duplicate object description`` warnings under ``-W``.

FP4 KV Cache Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

GPU-accelerated quantization / dequantization for KV-cache data using the
linear (non-swizzled) block-scale layout.

- :func:`nvfp4_kv_dequantize`: SM80+ (Ampere and later)
- :func:`nvfp4_kv_dequantize_paged`: SM80+ (Ampere and later)
- :func:`nvfp4_kv_quantize`: SM100+ (Blackwell and later)
- :func:`nvfp4_quantize_paged_kv_cache`

.. autosummary::
    :toctree: ../generated

    nvfp4_kv_quantize
    nvfp4_kv_dequantize
    nvfp4_kv_dequantize_paged
    nvfp4_quantize_paged_kv_cache

FP8 Quantization
----------------

.. autosummary::
    :toctree: ../generated

    mxfp8_quantize
    mxfp8_grouped_quantize
    mxfp8_dequantize_host
    per_token_group_quant_8bit

.. note::

    ``mxfp8_grouped_quantize`` uses a cuTile backend and requires SM100+ and
    ``cuda.tile`` (a ``requirements.txt`` dependency). ``K`` must be divisible
    by 32 and is padded internally to 128-column tiles.

CuTe-DSL Quantization Kernels (experimental)
--------------------------------------------

The CuTe-DSL backends are conditionally available when the
``nvidia-cutlass-dsl`` package is installed. At runtime they are also
re-exported as ``flashinfer.quantization.{nvfp4,mxfp4,mxfp8}_quantize_cute_dsl``
when available; documenting them here via their canonical submodule
path keeps the docs build from depending on the CuTe-DSL stack being
importable.

.. currentmodule:: flashinfer.quantization.kernels.nvfp4_quantize

.. autosummary::
    :toctree: ../generated

    nvfp4_quantize_cute_dsl
    nvfp4_quantize_per_token_cute_dsl

.. currentmodule:: flashinfer.quantization.kernels.mxfp4_quantize

.. autosummary::
    :toctree: ../generated

    mxfp4_quantize_cute_dsl

.. currentmodule:: flashinfer.quantization.kernels.mxfp8_quantize

.. autosummary::
    :toctree: ../generated

    mxfp8_quantize_cute_dsl
