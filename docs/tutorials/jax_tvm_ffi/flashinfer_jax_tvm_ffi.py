# This file is the canonical Sphinx-Gallery source for the tutorial.
# The Sphinx docs build generates HTML, a downloadable .py file, and a
# downloadable .ipynb notebook from this single source.

"""
Enabling FlashInfer GPU Kernels on JAX with the JAX TVM FFI Bridge
==================================================================


Overview
--------


JAX's XLA compiler is excellent for training and general tensor computation, but LLM inference has a distinct performance profile: every decode step attends over an ever-growing KV-cache, placing the bottleneck squarely on memory bandwidth rather than raw compute. FlashInfer is a library of hand-tuned CUDA kernels built precisely for this regime.

FlashInfer ships every CUDA kernel as a shared library - an ``.so``-file compiled with a cross-language binary interface defined by `Apache TVM's Foreign Function Interface <https://github.com/apache/tvm-ffi>`_. Any language with a TVM FFI binding can load these ``.so``-files and call the functions inside. This tutorial shows how to do it from JAX via `jax-tvm-ffi <https://github.com/NVIDIA/jax-tvm-ffi>`_, a bridge library that adapts TVM FFI functions to XLA custom calls.


What you'll build
-----------------


Three FlashInfer kernels:


.. list-table::
   :header-rows: 1

   * - Kernel
     - What it computes
     - New concept
   * - ``silu_and_mul``
     - Gated FFN activation: ``silu(gate) x up``
     - The minimal bridge: load -> register -> call
   * - ``apply_rope``
     - Rotary positional embeddings on packed batches
     - Multiple outputs; argument reordering
   * - ``single_decode``
     - Attention over a KV-cache (single request, GQA)
     - Type-specialized JIT; scratch buffers; optional-argument sentinels


At the end, all three run together inside a single ``@jax.jit`` region - exactly as they would in a real LLM decode loop.


Preliminaries
-------------


Hardware and software requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 0

   * - GPU
     - NVIDIA, SM 7.5+ (Turing or later)
   * - Python packages
     - ``jax``, ``jax-tvm-ffi``, ``flashinfer``
   * - CUDA
     - 12.6+
"""

# %%
# Setting the environment
# ~~~~~~~~~~~~~~~~~~~~~~~
#
#
# The easiest way to get a working JAX environment is the `NVIDIA NGC JAX container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax>`_. To install it manually:
#
# Recommended (CUDA 13):
#
#
# .. code-block:: bash
#
#    pip install 'jax[cuda13]'
#
#
# For CUDA 12.x:
#
#
# .. code-block:: bash
#
#    pip install 'jax[cuda12]'
#
#
# Three packages are required beyond a standard JAX environment:
#
#
# .. list-table::
#    :header-rows: 1
#
#    * - Package
#      - Role
#    * - ``flashinfer-python``
#      - FlashInfer CUDA kernels + JIT compilation system
#    * - ``jax-tvm-ffi``
#      - Bridge: adapts TVM FFI functions to XLA custom calls
#
#
# ``flashinfer-python`` ships pre-built wheels for each CUDA/Python combination. The ``--extra-index-url`` below selects the CUDA 13.0 wheel; replace ``cu130`` with a corresponding mapping for your `CUDA Toolkit release <https://developer.nvidia.com/cuda-toolkit-archive>`_:
#
# - CUDA 13 -> cu130
# - CUDA 12.x -> cu12x (e.g., cu126)
#
# Run the cell below only once in your environment.

# %%
# Install the tutorial dependencies before running the notebook or script::
#
#    pip install flashinfer-python -U jax-tvm-ffi \
#        --no-build-isolation  \
#        --extra-index-url https://flashinfer.ai/whl/cu130/  \
#

# %%
# Loading dependencies
# ~~~~~~~~~~~~~~~~~~~~

# %%
# Run the cell below to load the dependencies.

# %%
import os
import time
import math
import jinja2
import numpy as np
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF/XLA info & warnings

if "CUDA_HOME" not in os.environ:
    try:
        nvcc = subprocess.check_output(["which", "nvcc"], text=True).strip()
        os.environ["CUDA_HOME"] = str(os.path.dirname(os.path.dirname(nvcc)))
    except subprocess.CalledProcessError:
        os.environ["CUDA_HOME"] = "/usr/local/cuda"

if "--xla_gpu_cuda_data_dir=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')} "
        f"--xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"
    ).strip()

import jax
import jax.numpy as jnp

import jax_tvm_ffi  # Bridge adapter: TVM FFI -> XLA custom call
from flashinfer.jit import gen_act_and_mul_module, gen_jit_spec, env as jit_env
from flashinfer.jit.rope import gen_rope_module
from flashinfer.jit.attention.utils import generate_additional_params
from flashinfer.jit.utils import write_if_different

print(f"JAX:        {jax.__version__}")
print(f"Devices:    {jax.devices()}")
print(f"CUDA home:  {os.environ.get('CUDA_HOME')}")
print(f"JIT cache:  {jit_env.FLASHINFER_GEN_SRC_DIR.parent}")

# %%
# The JAX TVM FFI bridge
# ----------------------
#
#
# Every FlashInfer kernel lives in a compiled ``.so``-file. Getting that kernel into a ``@jax.jit`` computation graph takes three steps - the same three steps for every kernel in this tutorial:
#
#
# .. code-block:: text
#
#      Step 1  BUILD & LOAD   jit_spec.build_and_load()  ->  tvm_ffi.Module
#      Step 2  REGISTER       jax_tvm_ffi.register_ffi_target(name, wrapper, arg_spec)
#      Step 3  CALL           jax.ffi.ffi_call(name, output_shapes)(*inputs, **scalar_attrs)
#
#
# Step 1 - Compile and load the TVM FFI module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# FlashInfer compiles kernels on demand using its JIT system. ``flashinfer.jit`` contains framework-agnostic helpers - one per kernel family - that generate CUDA source code, compile it with nvcc + ninja, and produce a shared library (``.so``-file). The result is cached in ``~/.cache/flashinfer/`` so compilation only happens once per configuration. You'll see the specific helper for each kernel when we get to the examples.
#
# Each helper returns a **JitSpec** - a recipe describing what to compile. Calling ``.build_and_load()`` on a JitSpec runs the compilation pipeline and returns a ``tvm_ffi.Module``:
#
#
# .. code-block:: python
#
#    module = some_jit_spec.build_and_load()
#    module.my_kernel_function          # tvm_ffi.Function, callable from Python
#    module.my_kernel_function(a, b, c) # call it directly
#
#
# Under the hood, ``.build_and_load()`` writes the generated ``.so`` to the cache directory and calls ``tvm_ffi.load_module()`` to open it. If the ``.so`` already exists (cache hit), the compilation step is skipped and ``tvm_ffi.load_module()`` loads the cached binary directly.
#
# In other words: ``tvm_ffi.load_module(path)`` is the low-level primitive that opens any ``.so``-file, while ``.build_and_load()`` is the high-level entry point that handles source generation, compilation, caching, *and* loading in one call. In this tutorial we always use ``.build_and_load()`` because FlashInfer's JIT system manages the full pipeline for us.
#
#
# Step 2 - Register as a JAX FFI target
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# ``jax_tvm_ffi.register_ffi_target(name, wrapper, arg_spec)`` teaches XLA about the kernel. It wraps a Python callable into an XLA custom call node. Once registered, XLA can schedule the kernel inside ``@jax.jit`` computations.
#
# The ``wrapper`` is a Python function whose job is to call the TVM function. It may need to reorder arguments: JAX delivers output tensors first and input tensors second, but TVM functions are compiled with their own parameter order that you must match.
#
#
# .. code-block:: python
#
#    def _wrapper(out, x, scalar):                    # positional order set by arg_spec
#        module.my_kernel_function(x, out, scalar)    # reorder here to match TVM signature
#
#    jax_tvm_ffi.register_ffi_target(
#        'registered_name', _wrapper,
#        arg_spec=['rets', 'args', 'attrs.scalar'],
#        platform='gpu', allow_cuda_graph=True, pass_owned_tensor=True,
#    )
#
#
# .. note:: Concept: ``arg_spec`` routing
#
#
#    ``arg_spec`` is a list that tells the bridge how to route JAX's three categories of call-time data into the wrapper's positional arguments:
#
#    .. list-table::
#       :header-rows: 1
#
#       * - ``arg_spec`` value
#         - What the wrapper receives
#       * - ``'rets'``
#         - All output tensors, pre-allocated by XLA
#       * - ``'args'``
#         - All input tensors, in call order
#       * - ``'attrs.KEY'``
#         - A scalar keyword argument named ``KEY``
#
#    For example, ``arg_spec=['rets', 'args', 'attrs.top_k']`` means the wrapper is called as ``wrapper(*outputs, *inputs, top_k)``.
#
#    With ``pass_owned_tensor=True`` (required for GPU kernels), the tensors inside the wrapper are ``tvm_ffi.Tensor`` objects - they have a ``.shape`` attribute and are passed directly to TVM functions.
#
#
# Step 3 - Call as a regular JAX expression
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. code-block:: python
#
#    result = jax.ffi.ffi_call(
#        'registered_name',
#        jax.ShapeDtypeStruct(output_shape, output_dtype),   # tells XLA what to pre-allocate
#        vmap_method='broadcast_all',                        # how jax.vmap should batch this call
#    )(*input_tensors, scalar=value)
#
#
# The name ``'registered_name'`` must match what was passed to ``register_ffi_target`` in Step 2, and ``scalar=value`` must match the ``'attrs.scalar'`` entry in ``arg_spec``.
#
# Output shapes must be statically known - XLA needs them at trace time to pre-allocate the output buffers that get passed as ``'rets'`` in the wrapper. The call site is a regular JAX expression and composes naturally with ``@jax.jit``.
#
#
# .. note:: Concept: ``vmap_method``
#
#
#    ``vmap_method`` tells JAX how to handle ``jax.vmap`` over this call:
#
#    .. list-table::
#       :header-rows: 1
#
#       * - Value
#         - Behaviour
#       * - ``'broadcast_all'``
#         - Treat all inputs as sharing the same batch axis; run the kernel once per batch element. Use this when the operation is independent across a batch dimension, such as an element-wise activation or a per-token positional embedding.
#       * - omitted
#         - The call raises ``NotImplementedError`` inside ``jax.vmap``. Use this when the kernel's internals, such as scratch buffers or cross-head reduction, do not decompose cleanly over an added batch axis.
#
#    Please refer to `this documentation <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#auto-vectorization-with-jax-vmap>`_ to learn more about ``vmap``.
#
#
# The full execution path
# ~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. code-block:: text
#
#      User JAX code
#          |  jax.ffi.ffi_call(name, output_shapes)(*inputs, **attrs)
#          v
#      XLA Compiler
#          - emits a custom call node; records output shapes and scalar attrs at trace time
#          |
#      XLA Runtime
#          - looks up the registered target; pre-allocates output buffers
#          |
#      JAX-TVM-FFI Bridge
#          - unpacks call frame -> positional args according to arg_spec
#          |
#      Python wrapper  (user-defined)
#          - reorders arguments to match the TVM function signature
#          |
#      tvm_ffi.Function  ->  CUDA kernel

# %%
# Example 1: Gated SiLU (a.k.a. SwiGLU-style)
# -------------------------------------------
#
#
# In this example, you should learn how minimal the JAX-TVM bridge can be when the function signature and ``arg_spec`` are aligned: we have one kernel and one thin wrapper, and JAX handles the rest.
#
# Modern LLM feed-forward layers use a gated activation instead of a plain nonlinearity. A common form is:
#
#
# .. code-block:: text
#
#    FFN(x) = SiLU(W1 * x) * (W2 * x)
#
# where * is elementwise multiply.
#
# In practice, implementations often compute both linear projections in one matmul by using a weight with twice the hidden width, producing a tensor of shape ``[..., 2H]``. Here, hidden size ``H`` is just the width of the model's internal feature vectors. And in gated FFNs, we temporarily double it (``2H``) to compute two parallel projections before combining them back to size ``H``:
#
#
# .. code-block:: text
#
#    a = input[..., :H] (the "gate" half)
#    b = input[..., H:] (the "value / up-projection" half)
#
#
# The TVM function signature is:
#
#
# .. code-block:: python
#
#    silu_and_mul(out, input, enable_pdl)
#
#
# - ``out``: a pre-allocated output buffer that the kernel writes into.
# - ``input``: the fused ``[..., 2H]`` activation tensor (contains both halves concatenated).
# - ``enable_pdl``: a boolean that toggles Programmatic Dependent Launch (PDL) - an SM90+ feature that can help chain GPU work with less host involvement. For this tutorial, we keep it ``False``.
#
# The same compiled binary supports ``float16`` and ``bfloat16``, selecting the right path via runtime dispatch based on the input dtype.
#
# This is the simplest bridge case: **one input tensor -> one output tensor -> one scalar attribute**. And because the TVM function's parameter order (``out, input, enable_pdl``) matches the order JAX delivers them (``rets``, then ``args``, then ``attrs``), the wrapper needs no reordering.
#
#
# Compile and load
# ~~~~~~~~~~~~~~~~
#
#
# ``gen_act_and_mul_module('silu')`` returns a JitSpec - a recipe describing what to compile.
#
# ``.build_and_load()`` runs nvcc + ninja, writes the .so to the FlashInfer cache directory, and returns a ``tvm_ffi.Module``.

# %%
print("Compiling silu_and_mul (first run may take ~30 s)...")
silu_module = gen_act_and_mul_module("silu").build_and_load()
print(f"  Module type: {type(silu_module).__name__}")
print(f"  Function:    {silu_module.silu_and_mul}")

# %%
# Register as a JAX FFI target and validate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# In this section, we connect the compiled TVM kernel to JAX by registering it as an FFI target and exposing a clean Python wrapper. The key idea is that JAX groups parameters as outputs **(rets) -> inputs (args) -> scalar attributes (attrs)**, and because the TVM function already expects ``(out, input, enable_pdl)`` in that exact order, the wrapper simply forwards the call with no argument reshuffling. The ``silu_and_mul`` function then uses ``jax.ffi.ffi_call`` to allocate the output buffer, pass the fused ``[..., 2H]`` tensor to the kernel, and return the computed ``[..., H]`` result, while remaining fully batch-compatible via ``vmap_method="broadcast_all"``.
#
# The validation step should confirm correctness against a pure JAX reference.

# %%
# -- Register as a JAX FFI target --------------------------------------

# TVM function:  silu_and_mul(out, input, enable_pdl)
# arg_spec:      ['rets', 'args', 'attrs.enable_pdl']

# JAX delivers (out, input, enable_pdl) in the same order as the TVM function
# expects, so no reordering is needed in this wrapper.


def _silu_and_mul_wrapper(out, x, enable_pdl):
    silu_module.silu_and_mul(out, x, enable_pdl)


jax_tvm_ffi.register_ffi_target(
    "flashinfer.silu_and_mul",
    _silu_and_mul_wrapper,
    arg_spec=["rets", "args", "attrs.enable_pdl"],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)

# -- JAX-facing function -----------------------------------------------


def silu_and_mul(x: jax.Array) -> jax.Array:
    """Fused silu(gate) * up.  Input: [..., 2H]  Output: [..., H]"""
    out_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    return jax.ffi.ffi_call(
        "flashinfer.silu_and_mul",
        jax.ShapeDtypeStruct(out_shape, x.dtype),
        vmap_method="broadcast_all",  # element-wise op: independent across any batch dim
    )(x, enable_pdl=False)


# -- Validate ------------------------------------------------------------------

TOKENS, HIDDEN = 32, 256
gate_up = jax.random.normal(jax.random.key(0), (TOKENS, 2 * HIDDEN), dtype=jnp.float16)

out = silu_and_mul(gate_up)

# Compute reference to test our function
gate_ref = gate_up[..., :HIDDEN].astype(jnp.float32)
up_ref = gate_up[..., HIDDEN:].astype(jnp.float32)
ref = (jax.nn.silu(gate_ref) * up_ref).astype(jnp.float16)

np.testing.assert_allclose(
    np.array(out.astype(jnp.float32)),
    np.array(ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)
print("silu_and_mul: PASSED")
print(f"  {gate_up.shape} -> {out.shape}")
print(f"  max error: {float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref))):.5f}")

# %%
# Example 2: Rotary Positional Embeddings
# ---------------------------------------
#
#
# In this example, you should learn how to handle two new complications that were absent in Example 1: a TVM function that **produces two outputs** and whose **argument order does not match JAX's convention**. Both are resolved entirely inside the Python wrapper - registration and the call site otherwise follow the same pattern.
#
# RoPE encodes each token's position by rotating its query and key vectors in pairs. For position ``p`` and dimension index ``i``:
#
#
# .. code-block:: text
#
#    theta_i = rope_theta^(-2i / head_dim)
#    q_rot[2i]   = q[2i]  * cos(p*theta_i) - q[2i+1] * sin(p*theta_i)
#    q_rot[2i+1] = q[2i]  * sin(p*theta_i) + q[2i+1] * cos(p*theta_i)
#
#
# The same rotation is applied to both ``q`` and ``k``, so the kernel produces two rotated outputs.
#
#
# Ragged (packed) batches
# ~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Rather than a padded ``[batch, seq_len, heads, dim]`` tensor, ``apply_rope`` takes a flat ``[total_tokens, heads, dim]`` tensor with two auxiliary arrays:
#
# - ``indptr`` - a CSR-style pointer array where ``indptr[i]:indptr[i+1]`` gives the token range of sequence ``i``
# - ``offsets`` - the KV-cache position of the first token of each sequence
#
# This lets the kernel handle variable-length sequences without padding overhead.
#
#
# The TVM function signature
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. code-block:: python
#
#    apply_rope(q, k, q_rope, k_rope, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta)
#
#
# - ``q``, ``k``: input query and key tensors, shape ``[total_tokens, num_heads, head_dim]``
# - ``q_rope``, ``k_rope``: pre-allocated output buffers - they appear **in the middle** of the argument list, after the inputs but before the index arrays
# - ``indptr``, ``offsets``: ragged-batch descriptors
# - ``rotary_dim``, ``interleave``, ``rope_scale``, ``rope_theta``: scalar parameters
#
# Because the outputs are interleaved with the inputs, the JAX convention (all outputs first) does not match the TVM signature. The wrapper must swap them back.
#
#
# .. note:: Concept: argument reordering
#
#
#    TVM functions are compiled with whatever parameter order the kernel author chose. JAX's FFI convention always delivers ``(outputs, inputs, scalars)``. The wrapper bridges the two:
#
#    .. code-block:: python
#
#       # JAX delivers (via arg_spec): q_rope, k_rope, q, k, indptr, offsets, *scalars
#       # TVM function expects:        q, k, q_rope, k_rope, indptr, offsets, *scalars
#       def _wrapper(q_rope, k_rope, q, k, indptr, offsets, *scalars):
#           tvm_fn(q, k, q_rope, k_rope, indptr, offsets, *scalars)  # reorder here
#
#
# Compile and load
# ~~~~~~~~~~~~~~~~
#
#
# ``gen_rope_module()`` returns a JitSpec for the RoPE kernel. Unlike ``gen_act_and_mul_module``, this kernel dispatches over dtypes at runtime from a single binary - no dtype specialisation is needed at compile time.

# %%
print("Compiling apply_rope...")
rope_module = gen_rope_module().build_and_load()
print(f"  Function: {rope_module.apply_rope}")

# %%
# Register as a JAX FFI target and validate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# The ``apply_rope`` wrapper introduces the argument-reordering pattern described above. Because the TVM function places ``q_rope`` and ``k_rope`` after the input tensors - rather than at the beginning - the wrapper receives them first (via ``'rets'``) and must swap them back before calling the TVM function. The kernel also produces two output tensors, so ``jax.ffi.ffi_call`` receives a tuple of two ``ShapeDtypeStruct`` descriptors and returns a tuple of two JAX arrays. Because RoPE rotates each token independently, the operation decomposes cleanly over any outer batch dimension: ``vmap_method='broadcast_all'`` enables ``jax.vmap`` to map over it without any changes to the kernel.
#
# The validation uses two sequences with different starting positions - sequence 0 begins at position 0 and sequence 1 at position 100 - simulating a prefill where the second sequence already has 100 tokens in the KV-cache. Each token is rotated by angles matching its absolute position, so the test exercises both the CSR indexing and the offset arithmetic.

# %%
# -- Register as a JAX FFI target ----------------------------------------------
#
# TVM function: apply_rope(q, k, q_rope, k_rope, indptr, offsets,
#                          rotary_dim, interleave, rope_scale, rope_theta)
#
# JAX delivers: q_rope, k_rope first (rets), then q, k, indptr, offsets (args).
# TVM expects:  q, k, q_rope, k_rope, indptr, offsets, ... - the wrapper swaps them.


def _rope_wrapper(
    q_rope,
    k_rope,
    q,
    k,
    indptr,
    offsets,
    rotary_dim,
    interleave,
    rope_scale,
    rope_theta,
):
    rope_module.apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


jax_tvm_ffi.register_ffi_target(
    "flashinfer.apply_rope",
    _rope_wrapper,
    arg_spec=[
        "rets",
        "args",
        "attrs.rotary_dim",
        "attrs.interleave",
        "attrs.rope_scale",
        "attrs.rope_theta",
    ],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)

# -- JAX-facing function -------------------------------------------------------


def apply_rope(q, k, indptr, offsets, *, rope_theta=1e4):
    """Apply rotary positional embeddings to packed query and key tensors.

    q, k:     [total_tokens, num_heads, head_dim]
    indptr:   [num_seqs + 1]  CSR-style token range per sequence
    offsets:  [num_seqs]      absolute position of the first token of each sequence
    Returns:  (q_rotated, k_rotated), same shapes as inputs
    """
    head_dim = q.shape[-1]
    return jax.ffi.ffi_call(
        "flashinfer.apply_rope",
        (
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
        ),
        vmap_method="broadcast_all",  # each packed batch is independent; safe to vmap
    )(
        q,
        k,
        indptr,
        offsets,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=float(rope_theta),
    )


# -- Validate ------------------------------------------------------------------


def _reference_rope(x, positions, theta=1e4):
    """Non-interleaved RoPE reference.  x: [tokens, heads, dim]"""
    x32 = x.astype(jnp.float32)
    d = x32.shape[-1] // 2
    freqs = 1.0 / (theta ** (2.0 * jnp.arange(d, dtype=jnp.float32) / x32.shape[-1]))
    angles = positions[:, None].astype(jnp.float32) * freqs[None, :]  # [T, d]
    cos_a = jnp.cos(angles)[:, None, :]  # [T, 1, d]
    sin_a = jnp.sin(angles)[:, None, :]
    x1, x2 = x32[..., :d], x32[..., d:]
    return jnp.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    ).astype(x.dtype)


# Two sequences of 8 tokens: first starts at position 0, second at position 100
NUM_HEADS, HEAD_DIM, SEQ_LEN, NUM_SEQ = 8, 64, 8, 2
ROPE_THETA = 1e4

q_in = jax.random.normal(
    jax.random.key(1), (NUM_SEQ * SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
)
k_in = jax.random.normal(
    jax.random.key(2), (NUM_SEQ * SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
)

indptr = jnp.array([0, SEQ_LEN, 2 * SEQ_LEN], dtype=jnp.int32)
offsets = jnp.array([0, 100], dtype=jnp.int32)

q_rot, k_rot = apply_rope(q_in, k_in, indptr, offsets, rope_theta=ROPE_THETA)

positions = jnp.concatenate(
    [jnp.arange(SEQ_LEN, dtype=jnp.int32) + off for off in [0, 100]]
)
q_ref = _reference_rope(q_in, positions, theta=ROPE_THETA)
k_ref = _reference_rope(k_in, positions, theta=ROPE_THETA)

for name, got, ref in [("q", q_rot, q_ref), ("k", k_rot, k_ref)]:
    np.testing.assert_allclose(
        np.array(got.astype(jnp.float32)),
        np.array(ref.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )
    max_err = float(jnp.max(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"apply_rope {name}: PASSED  max_err={max_err:.5f}")
print(
    f"  Input: {q_in.shape}  ({NUM_SEQ} seqs x {SEQ_LEN} tokens, offsets {offsets.tolist()})"
)

# %%
# Example 3: Single-request decode attention
# ------------------------------------------
#
#
# In this example, you should learn about three patterns that appear together for the first time: a kernel **compiled separately per dtype and head dimension**, a **scratch buffer** the caller must allocate as an output, and **optional arguments** signalled as absent with empty tensors.
#
# In autoregressive generation, each new token attends over all previously generated tokens stored in the KV-cache:
#
#
# .. code-block:: text
#
#    Attention(q, K, V) = softmax(q KT / sqrtd) V
#
#
# where ``q`` is a single query (or a small group in Grouped Query Attention), and ``K``, ``V`` are the full cached sequences. FlashInfer's single-request kernel is tuned for exactly this shape: one query token, many keys, memory-bound.
#
#
# Type-specialized compilation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Unlike ``silu_and_mul`` and ``apply_rope`` - which dispatch over dtypes at runtime from a single binary - the decode attention kernel is **compiled separately for each ``(dtype, head_dim)`` combination**. This lets the compiler choose tile sizes and memory access patterns for the exact configuration, with no runtime branching.
#
# FlashInfer's JIT system renders Jinja2 templates with concrete type names to produce configuration-specific CUDA code. A separate ``.so`` is compiled for each combination and identified by a URI that doubles as the on-disk cache key. ``gen_decode_jit_spec``, defined in the cell below, builds that URI and invokes the JIT system.
#
#
# The TVM function signature
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. code-block:: python
#
#    run(q, k, v,
#        tmp, out,            # outputs: scratch buffer first, result second
#        maybe_lse,           # optional output: log-sum-exp values
#        kv_layout_code, window_left,
#        maybe_alibi_slopes,  # optional input: ALiBi position biases
#        logits_soft_cap, sm_scale, rope_rcp_scale, rope_rcp_theta)
#
#
# - ``q``, ``k``, ``v``: query and KV-cache tensors
# - ``tmp``: a **scratch buffer** for split-K partial sums - must be provided by the caller as a pre-allocated output
# - ``out``: the attention result
# - ``maybe_lse``: optional log-sum-exp output; ``None`` to skip
# - ``kv_layout_code``: ``0`` for NHD layout, ``1`` for HND
# - ``window_left``: sliding-window cutoff; ``-1`` for full attention
# - ``maybe_alibi_slopes``: optional ALiBi position biases; ``None`` to skip
# - ``logits_soft_cap``, ``sm_scale``, ``rope_rcp_scale``, ``rope_rcp_theta``: scalar parameters
#
# Two patterns are new here:
#
#
# .. note:: Concept: scratch buffer
#
#
#    The decode kernel uses an internal buffer for split-K partial results. We declare it as an output tensor so XLA pre-allocates it, then discard it after the call:
#
#    .. code-block:: python
#
#       out, _, _ = jax.ffi.ffi_call(target, (result_struct, tmp_struct, lse_struct))(...)
#       #                                                    ^^^          ^^^
#       #                                           32 MB scratch     LSE sentinel
#
#
# .. note:: Concept: empty-tensor sentinel
#
#
#    Optional arguments are signalled as absent with a tensor whose first dimension is zero. Inside the wrapper, ``tensor.shape[0] == 0`` maps to ``None``:
#
#    .. code-block:: python
#
#       # As an output (not computed):
#       jax.ShapeDtypeStruct((0,), jnp.float32)
#
#       # As an input (not provided):
#       jnp.empty((0,), dtype=jnp.float32)
#
#
# Compile and load
# ~~~~~~~~~~~~~~~~
#
#
# ``gen_decode_jit_spec`` assembles the build recipe: it renders two Jinja2 templates with the concrete dtype and head-dimension values, copies the ``.cu`` source files into the generated directory, and returns a ``JitSpec`` that ``.build_and_load()`` compiles and caches.

# %%
# -- Type mappings for Jinja template rendering --------------------------------

DTYPE_CPP = {"float16": "half", "bfloat16": "nv_bfloat16", "float32": "float"}
DTYPE_SAFE = {"float16": "f16", "bfloat16": "bf16", "float32": "f32"}
POS_ENC = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}


def gen_decode_jit_spec(dtype: str = "float16", head_dim: int = 64):
    """Return a JitSpec for type-specialized single-request decode attention."""
    s = DTYPE_SAFE[dtype]
    uri = (
        f"single_decode_with_kv_cache_dtype_q_{s}_dtype_kv_{s}_dtype_o_{s}_"
        f"head_dim_qk_{head_dim}_head_dim_vo_{head_dim}_"
        f"posenc_0_use_swa_False_use_logits_cap_False"
    )
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_dir, exist_ok=True)

    # generate_additional_params produces the C++ boilerplate strings
    # for the optional alibi-slopes tensor and four scalar parameters.
    params_decl, func_params, params_setter = generate_additional_params(
        additional_tensor_names=["maybe_alibi_slopes"],
        additional_tensor_dtypes=["float"],
        additional_scalar_names=[
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],
        additional_scalar_dtypes=["double", "double", "double", "double"],
    )

    kwargs = dict(
        additional_func_params=func_params,
        additional_params_decl=params_decl,
        additional_params_setter=params_setter,
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        variant_name="DefaultAttention<false, false, false, false>",
        dtype_q=DTYPE_CPP[dtype],
        dtype_kv=DTYPE_CPP[dtype],
        dtype_o=DTYPE_CPP[dtype],
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        pos_encoding_mode=POS_ENC[0],
        use_sliding_window="false",
        use_logits_soft_cap="false",
    )

    csrc = jit_env.FLASHINFER_CSRC_DIR

    # Render Jinja2 templates with the type-specific values
    for tmpl, out in [
        ("single_decode_customize_config.jinja", "single_decode_config.inc"),
        ("single_decode_kernel_inst.jinja", "single_decode_kernel.cu"),
    ]:
        rendered = jinja2.Template((csrc / tmpl).read_text()).render(**kwargs)
        write_if_different(gen_dir / out, rendered)

    # Copy the .cu source files that #include the rendered headers
    sources = [gen_dir / "single_decode_kernel.cu"]
    for fname in ["single_decode.cu", "single_decode_jit_binding.cu"]:
        dest = gen_dir / fname
        write_if_different(dest, (csrc / fname).read_text())
        sources.append(dest)

    return gen_jit_spec(uri, sources)


# -- Compile and load ----------------------------------------------------------

DTYPE, HEAD_DIM = "float16", 64
print(f"Compiling decode attention ({DTYPE}, head_dim={HEAD_DIM})...")
decode_module = gen_decode_jit_spec(DTYPE, HEAD_DIM).build_and_load()
print(f"  run function: {decode_module.run}")

# %%
# Register as a JAX FFI target
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# The ``decode_attention`` wrapper handles all three patterns from this example. The optional ``lse`` and ``alibi_slopes`` arguments use the empty-tensor sentinel: when ``tensor.shape[0] == 0``, the wrapper passes ``None`` to the TVM function, telling the kernel to skip that output or ignore that input. The scratch buffer ``tmp`` is declared as a pre-allocated output - XLA reserves 32 MB for it - and the caller discards it after the call. Finally, ``vmap_method`` is intentionally omitted: the scratch buffer is a flat array with no batch dimension, and GQA head-grouping does not decompose cleanly over an added outer batch axis. Callers that need batching should loop explicitly or use a batch-aware kernel variant.

# %%
# -- Register ------------------------------------------------------------------
#
# TVM function signature:
#   run(q, k, v, tmp, out, maybe_lse, kv_layout_code, window_left,
#       maybe_alibi_slopes, logits_soft_cap, sm_scale, rope_rcp_scale, rope_rcp_theta)
#
# Sentinel rule: tensor.shape[0] == 0  =>  pass None  (argument is absent)

_run = decode_module.run


def _decode_wrapper(
    out,
    tmp,
    lse_or_empty,  # <- rets
    q,
    k,
    v,
    alibi_or_empty,  # <- args
    kv_layout_code,
    window_left,  # <- attrs
    logits_soft_cap,
    sm_scale,
    rope_scale,
    rope_theta,
):
    lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
    alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
    # Reorder to match TVM function signature:
    # kv_layout_code and window_left come before maybe_alibi_slopes
    _run(
        q,
        k,
        v,
        tmp,
        out,
        lse,
        kv_layout_code,
        window_left,
        alibi,
        logits_soft_cap,
        sm_scale,
        rope_scale,
        rope_theta,
    )


# Embed dtype + head_dim in the name: each compiled binary is a distinct target
DECODE_TARGET = f"flashinfer.single_decode_{DTYPE}_h{HEAD_DIM}"

jax_tvm_ffi.register_ffi_target(
    DECODE_TARGET,
    _decode_wrapper,
    arg_spec=[
        "rets",
        "args",
        "attrs.kv_layout_code",
        "attrs.window_left",
        "attrs.logits_soft_cap",
        "attrs.sm_scale",
        "attrs.rope_scale",
        "attrs.rope_theta",
    ],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)

# -- JAX-facing function -------------------------------------------------------


def decode_attention(q, k, v):
    """Single-request GQA decode attention.

    q: [num_qo_heads, head_dim]          float16   (query for one new token)
    k: [kv_len, num_kv_heads, head_dim]  float16   (full KV-cache, NHD layout)
    v: [kv_len, num_kv_heads, head_dim]  float16
    Returns: [num_qo_heads, head_dim]
    """
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    tmp_elems = 32 * 1024 * 1024 // np.dtype(q.dtype).itemsize

    out, _, _ = jax.ffi.ffi_call(
        DECODE_TARGET,
        (
            jax.ShapeDtypeStruct(q.shape, q.dtype),  # out
            jax.ShapeDtypeStruct((tmp_elems,), q.dtype),  # tmp   (scratch, discarded)
            jax.ShapeDtypeStruct((0,), jnp.float32),  # lse   (sentinel: not computed)
        ),
    )(
        q,
        k,
        v,
        jnp.empty((0,), dtype=jnp.float32),  # alibi slopes: sentinel (not used)
        kv_layout_code=0,  # NHD=0, HND=1
        window_left=-1,  # full attention window
        logits_soft_cap=0.0,
        sm_scale=sm_scale,
        rope_scale=1.0,
        rope_theta=1e4,
    )
    return out


print(f"Registered '{DECODE_TARGET}'.")

# %%
# The validation uses a GQA configuration: 16 query heads and 4 KV heads (4x grouping) attending over a 512-token KV-cache. A pure-JAX reference computes the grouped softmax attention in float32 for comparison.

# %%
# -- Validate ------------------------------------------------------------------


def _reference_gqa_decode(q, k, v):
    """Reference GQA decode.  q: [H_q, D]  k, v: [S, H_kv, D]  (NHD)"""
    H_q, H_kv = q.shape[0], k.shape[1]
    scale = q.shape[-1] ** -0.5
    q32 = q.astype(jnp.float32).reshape(H_kv, H_q // H_kv, -1)  # [H_kv, group, D]
    scores = jnp.einsum("hgd,shd->hgs", q32, k.astype(jnp.float32)) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hgs,shd->hgd", weights, v.astype(jnp.float32))
    return out.reshape(H_q, -1)


# GQA: 16 query heads, 4 KV heads (4x grouping), 512-token KV-cache
NUM_QO, NUM_KV, KV_LEN = 16, 4, 512

q = jax.random.normal(jax.random.key(10), (NUM_QO, HEAD_DIM), dtype=jnp.float16)
k = jax.random.normal(jax.random.key(11), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)
v = jax.random.normal(jax.random.key(12), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)

out_raw = decode_attention(q, k, v)
out_ref = _reference_gqa_decode(q, k, v)

np.testing.assert_allclose(
    np.array(out_raw.astype(jnp.float32)),
    np.array(out_ref),
    rtol=1e-2,
    atol=1e-2,
)
print("decode_attention: PASSED")
print(
    f"  GQA: {NUM_QO} query / {NUM_KV} KV heads ({NUM_QO // NUM_KV}x groups), kv_len={KV_LEN}"
)
print(f"  Output: {out_raw.shape}")
print(
    f"  Max error: {float(jnp.max(jnp.abs(out_raw.astype(jnp.float32) - out_ref))):.4f}"
)

# %%
# Composing kernels in ``@jax.jit``
# ---------------------------------
#
#
# In this final section, you should learn that registered TVM FFI targets are plain XLA custom call nodes and compose naturally inside a single ``@jax.jit``-decorated function alongside any other JAX operations - no special handling is needed.
#
# A realistic LLM decode step brings all three kernels together:
#
# 1. ``silu_and_mul`` - gated FFN activation
# 2. ``apply_rope`` - rotary embeddings applied to the new query and key tokens
# 3. ``decode_attention`` - cross-attention over the full KV-cache with the RoPE'd query
#
# XLA compiles the entire function once. All three kernels become custom call nodes in the same HLO computation graph.
#
# .. note:: Where to put ``@jax.jit``
#
#    ``jax.ffi.ffi_call`` works with or without ``@jax.jit``. Two patterns are common:
#
#    **Per-kernel JIT** - decorate each helper individually:
#
#    .. code-block:: python
#
#       @jax.jit
#       def silu_and_mul(x): ...
#
#       @jax.jit
#       def decode_attention(q, k, v): ...
#
#    Each kernel compiles independently and runs fast when called in isolation. Good for standalone use, exploratory work, and benchmarking individual kernels.
#
#    **Outer JIT only** - helpers are plain functions; only the composition function is decorated:
#
#    .. code-block:: python
#
#       def silu_and_mul(x): ...       # plain Python
#       def decode_attention(q, k, v): ...
#
#       @jax.jit
#       def decode_step(...):
#           ffn_out = silu_and_mul(...)
#           attn_out = decode_attention(...)
#           return ffn_out, attn_out
#
#    XLA traces the entire decode step as a single computation graph, seeing all custom call nodes at once. This is the right choice for production LLM inference, where all kernels run together every step anyway.
#
#    JAX handles nested ``@jax.jit`` correctly - inner jits are inlined during tracing - so there is no penalty to mixing patterns later. This tutorial uses **outer JIT only** to keep the focus on the composition.

# %%
# -- Inputs for the decode step -----------------------------------------------

# FFN input: 4 tokens, each with a 2H fused gate+up projection
gate_up = jax.random.normal(jax.random.key(20), (4, 2 * HEAD_DIM), dtype=jnp.float16)

# GQA dimensions and KV-cache (same config as the validate cell)
NUM_QO, NUM_KV, KV_LEN = 16, 4, 512

q = jax.random.normal(jax.random.key(10), (NUM_QO, HEAD_DIM), dtype=jnp.float16)
k = jax.random.normal(jax.random.key(11), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)
v = jax.random.normal(jax.random.key(12), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)

# New-token Q and K as packed batches for apply_rope: [tokens, heads, dim]
q_new = q.reshape(1, NUM_QO, HEAD_DIM)  # [1, 16, 64]  <- new query token
k_new = k[:1]  # [1,  4, 64]  <- new key token
indptr = jnp.array([0, 1], dtype=jnp.int32)  # one sequence of length 1
offsets = jnp.array([KV_LEN], dtype=jnp.int32)  # new token sits at position KV_LEN


# -- @jax.jit composition ------------------------------------------------------


@jax.jit
def decode_step(gate_up, q_new, k_new, k_cache, v_cache, indptr, offsets):
    """One LLM decode step compiled into a single XLA computation."""
    # 1. Gated FFN activation
    ffn_out = silu_and_mul(gate_up)

    # 2. Rotary embeddings for the new query and key tokens
    q_r, k_r = apply_rope(q_new, k_new, indptr, offsets)

    # 3. Decode attention over the full KV-cache with the RoPE'd query
    attn_out = decode_attention(q_r.reshape(NUM_QO, HEAD_DIM), k_cache, v_cache)

    return ffn_out, attn_out


ffn_out, attn_out = decode_step(gate_up, q_new, k_new, k, v, indptr, offsets)

# Validate against calling each kernel individually (outside @jax.jit)
ffn_ref = silu_and_mul(gate_up)
q_r, k_r = apply_rope(q_new, k_new, indptr, offsets)
attn_ref = decode_attention(q_r.reshape(NUM_QO, HEAD_DIM), k, v)

np.testing.assert_allclose(
    np.array(ffn_out.astype(jnp.float32)),
    np.array(ffn_ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)
np.testing.assert_allclose(
    np.array(attn_out.astype(jnp.float32)),
    np.array(attn_ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)

print("@jax.jit composition: PASSED")
print(f"  gate_up {gate_up.shape} -> ffn_out  {ffn_out.shape}")
print(f"  q_new   {q_new.shape}  -> attn_out {attn_out.shape}")

# -- Latency benchmark ----------------------------------------------------------

_ = decode_attention(q, k, v).block_until_ready()  # warm-up (triggers XLA compilation)
N = 100
t0 = time.perf_counter()
for _ in range(N):
    decode_attention(q, k, v).block_until_ready()
us = (time.perf_counter() - t0) / N * 1e6
print(
    f"\ndecode_attention  kv_len={KV_LEN}, {NUM_QO}/{NUM_KV} GQA heads  ->  {us:.1f} us"
)

# %%
# Summary
# -------
#
#
# You have applied the JAX-TVM FFI bridge to three real LLM inference kernels, each revealing a new layer of the pattern.
#
#
# .. code-block:: text
#
#      Step 1  BUILD & LOAD   jit_spec.build_and_load()  ->  tvm_ffi.Module
#      Step 2  REGISTER       jax_tvm_ffi.register_ffi_target(name, wrapper, arg_spec)
#      Step 3  CALL           jax.ffi.ffi_call(name, output_shapes)(*inputs, **scalar_attrs)
#
#
# FlashInfer provides the same bridge for batch decode with paged KV-cache, variable-length prefill attention, fused mixture-of-experts, quantized GEMM, and more. Every kernel uses the same three-step recipe summarized above. The main variables are:
#
# - How many output tensors to declare, and whether any are scratch buffers to discard after the call
# - Whether any inputs or outputs are optional (use the empty-tensor sentinel)
# - Whether the kernel needs type-specialized compilation (Jinja template rendering) or dispatches at runtime over dtypes
#
#
# Beyond the examples in this tutorial
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# This tutorial demonstrated ``silu_and_mul``, ``apply_rope``, and ``single_decode`` as representative examples, but FlashInfer's strength lies in its broader library of high-performance kernels - including Multi-head Latent Attention (MLA), sparse attention, TensorRT-LLM generative batch attention, and fused Mixture-of-Experts (MoE). The same three-step ``jax-tvm-ffi`` pattern shown here applies directly to all of them: compile the kernel, register the wrapper, and call it from JAX. No changes to the bridge are needed - only the TVM function signature and ``arg_spec`` vary per kernel.
#
# For more details on ``jax-tvm-ffi`` itself - including how to wrap your own C++ or Python TVM functions - see the `jax-tvm-ffi documentation and examples <https://github.com/NVIDIA/jax-tvm-ffi>`_.
