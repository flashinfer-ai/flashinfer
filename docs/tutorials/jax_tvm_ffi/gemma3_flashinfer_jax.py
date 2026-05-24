# This file is the canonical Sphinx-Gallery source for the tutorial.
# The Sphinx docs build generates HTML, a downloadable .py file, and a
# downloadable .ipynb notebook from this single source.

"""
Gemma 3 on JAX with FlashInfer and the JAX TVM FFI Bridge
=========================================================


Overview
--------


:doc:`flashinfer_jax_tvm_ffi` built three FlashInfer kernels from scratch and wired them into JAX as XLA custom calls. This tutorial connects those same primitives to a real language model: **Gemma 3 1B Instruct**, Google's open-weight instruction-tuned LLM.

Every Gemma 3 transformer layer uses the following kernels:


.. list-table::
   :header-rows: 1

   * - Part 1 kernel
     - Role in Gemma 3
   * - ``gelu_tanh_and_mul``
     - Gated FFN activation (GeGLU variant of SiLU-GLU)
   * - ``apply_rope``
     - Query and key positional embeddings - with two different theta values
   * - ``decode_attention``
     - Attention over the growing KV-cache - two compiled variants


Three things are new compared to Part 1:

- **GeGLU instead of SiLU-GLU** - Gemma 3 uses ``gelu_tanh`` for its gated FFN; FlashInfer ships this as a one-word change from ``silu``.
- **QK-norm** - per-head RMSNorm applied to Q and K before computing dot products, replacing the logit soft-capping that Gemma 2 used.
- **Dual RoPE theta** - local-attention layers use theta = 10 000; global-attention layers use theta = 1 000 000. We select the right value per layer and pass it to ``apply_rope``.


Preliminaries
-------------


Hardware and software requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 0

   * - GPU
     - NVIDIA, SM 7.5+ (Turing or later)
   * - Python packages
     - ``jax``, ``jax-tvm-ffi``, ``flashinfer``, ``torch``, ``safetensors``, ``huggingface_hub``, ``transformers``
   * - HuggingFace
     - Account with Gemma 3 licence accepted - `request access <https://huggingface.co/google/gemma-3-1b-it>`_
"""

# %%
# Setting the environment
# ~~~~~~~~~~~~~~~~~~~~~~~
#
#
# If you haven't gone through :doc:`flashinfer_jax_tvm_ffi`, refer to it for the JAX and FlashInfer installation instructions.
#
# Four additional packages are required:
#
#
# .. list-table::
#    :header-rows: 1
#
#    * - Package
#      - Role
#    * - ``torch`` (CPU)
#      - Provides ``torch.dtype`` literals used by FlashInfer's JIT API
#    * - ``safetensors``
#      - Efficient loading of model weights from the HuggingFace format
#    * - ``huggingface_hub``
#      - Model download from the HuggingFace Hub
#    * - ``transformers``
#      - Tokenizer and chat-template formatting
#
#
# Run the cell below only once in your environment.

# %%
# Install the tutorial dependencies before running the notebook or script::
#
#    pip install torch --index-url https://download.pytorch.org/whl/cpu
#    pip install safetensors huggingface_hub transformers

# %%
# Loading dependencies
# ~~~~~~~~~~~~~~~~~~~~

# %%
# Run the cell below to load the dependencies.

# %%
import json
import math
import os
import time
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF/XLA info & warnings

import importlib.util

IN_COLAB = importlib.util.find_spec("google.colab") is not None

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

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from safetensors import safe_open
import jax_tvm_ffi

print(f"JAX:        {jax.__version__}")
print(f"Devices:    {jax.devices()}")
print(f"CUDA home:  {os.environ['CUDA_HOME']}")

# %%
# HuggingFace Authentication
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Gemma 3 is a gated model. Before downloading the weights, you need to accept the licence on the HuggingFace model page - visit `google/gemma-3-1b-it <https://huggingface.co/google/gemma-3-1b-it>`_ and click *Request access*.
#
# The cell below reads your token from the ``HF_TOKEN`` environment variable, falls back to the Colab Secrets API if running on Colab, or prompts interactively.

# %%
# -- HuggingFace Authentication --------------------------------------------------
# Accept the Gemma 3 license at: https://huggingface.co/google/gemma-3-1b

if IN_COLAB:
    from google.colab import userdata

    HF_TOKEN = userdata.get("HF_TOKEN")
else:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    from getpass import getpass

    HF_TOKEN = getpass(
        "Hugging Face token not found in environment. Please enter it here: "
    )

if not HF_TOKEN:
    raise RuntimeError("Authentication failed: Hugging Face token is not set.")

# Ensure token is set in this process
os.environ["HF_TOKEN"] = HF_TOKEN

# Verify identity
api = HfApi()
user_info = api.whoami(token=HF_TOKEN)
username = user_info.get("name") or "Unknown user"
print(f"Authenticated with Hugging Face successfully as: {username}")

# %%
# Downloading the model weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# The cell below downloads all model shards (~2 GB on first run) from the HuggingFace Hub, loads them as ``bfloat16`` JAX arrays, and instantiates the tokenizer. Weights are cached locally; subsequent runs skip the download.

# %%
MODEL_ID = "google/gemma-3-1b-it"
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# -- Tokenizer ------------------------------------------------------------------
print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, cache_dir=HF_CACHE)

# -- Weights --------------------------------------------------------------------
print("Downloading model weights (~2 GB on first run)...")
model_dir = snapshot_download(MODEL_ID, token=HF_TOKEN, cache_dir=f"{HF_CACHE}/hub")

# Weights are split across shards - discover the full list from the index file
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        shard_files = sorted(set(json.load(f)["weight_map"].values()))
else:
    shard_files = ["model.safetensors"]

print(f"Loading {len(shard_files)} shard(s) as JAX bfloat16 arrays...")
weights = {}
for shard in shard_files:
    with safe_open(os.path.join(model_dir, shard), framework="numpy") as f:
        for key in f.keys():
            # jnp.array handles any numpy dtype (float32, bfloat16, ...) -> bfloat16
            weights[key] = jnp.array(f.get_tensor(key), dtype=jnp.bfloat16)

n_params = sum(int(w.size) for w in weights.values())
print(f"Loaded {len(weights)} tensors  ({n_params / 1e9:.2f} B parameters)")
print(f"  embed_tokens:        {weights['model.embed_tokens.weight'].shape}")
print(
    f"  layer 0  q_proj:     {weights['model.layers.0.self_attn.q_proj.weight'].shape}"
)
print(
    f"  layer 0  q_norm:     {weights['model.layers.0.self_attn.q_norm.weight'].shape}"
)
print(f"  layer 0  gate_proj:  {weights['model.layers.0.mlp.gate_proj.weight'].shape}")

# %%
# Gemma 3 Transformer Layer
# -------------------------
#
#
# Each Gemma 3 1B layer has a **sandwich-norm** structure: RMSNorm before *and* after each sub-layer.
#
#
# .. code-block:: text
#
#    -- Prefill (prompt, T tokens in parallel) ------------------------------------
#    x --+-- RMSNorm (input_layernorm) ------------------------------------------+
#        |   Q, K, V  <-  linear projections                                      |
#        |   Q, K     <-  QK-norm  (per-head RMSNorm, new in Gemma 3)            |
#        |   Q, K     <-  `apply_rope`  (theta = local  /  global, per layer)        |
#        |   out      <-  `prefill_attention`  (causal, local or global)          |
#        |   RMSNorm (post_attention_layernorm)                                   |
#        +-- + -------------------------------------------------------------------+
#
#    -- Decode (one new token at a time) ------------------------------------------
#    x --+-- RMSNorm (input_layernorm) ------------------------------------------+
#        |   Q, K, V  <-  linear projections                                      |
#        |   Q, K     <-  QK-norm  (per-head RMSNorm, new in Gemma 3)            |
#        |   Q, K     <-  `apply_rope`  (theta = local  /  global, per layer)        |
#        |   K, V     ->  KV-cache append                                         |
#        |   out      <-  `decode_attention`  (local sliding-window or global)    |
#        |   RMSNorm (post_attention_layernorm)                                   |
#        +-- + -------------------------------------------------------------------+
#
#    -- Shared FFN (same code for prefill and decode) -----------------------------
#    x --+-- RMSNorm (pre_feedforward_layernorm) ----------------------------------+
#        |   gate, up  <-  separate linear projections                              |
#        |   hidden    <-  `gelu_tanh_and_mul`( concat(gate, up) )                 |
#        |   out       <-  down_proj(hidden)                                        |
#        |   RMSNorm (post_feedforward_layernorm)                                  |
#        +-- + --------------------------------------------------------------------+
#
#
# Local vs global attention - the 5:1 pattern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Gemma 3 alternates attention span in a repeating 5:1 pattern:
#
#
# .. list-table::
#    :header-rows: 1
#
#    * - Layer type
#      - Frequency
#      - Attends to
#      - RoPE theta
#    * - **Local**
#      - 5 of every 6
#      - Last ``sliding_window`` tokens
#      - ``rope_local_base_freq``
#    * - **Global**
#      - 1 of every 6
#      - Full KV-cache
#      - ``rope_theta``
#
#
# For Gemma 3 1B (26 layers): layers **5, 11, 17, 23** are global; the remaining 22 are local. Exact values for window size and theta are read from ``config.json`` in the next cell.

# %%
with open(os.path.join(model_dir, "config.json")) as _f:
    _raw = json.load(_f)

# Gemma 3 wraps the language model config under "text_config" in its multimodal JSON
cfg = _raw.get("text_config", _raw)

HIDDEN = cfg["hidden_size"]
INTERMEDIATE = cfg["intermediate_size"]
N_LAYERS = cfg["num_hidden_layers"]
N_Q = cfg["num_attention_heads"]
N_KV = cfg["num_key_value_heads"]
HEAD_DIM = cfg.get("head_dim", HIDDEN // N_Q)
VOCAB = cfg["vocab_size"]
RMS_EPS = cfg.get("rms_norm_eps", 1e-6)
SLIDING_WINDOW = cfg.get("sliding_window", 1024)
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Dual RoPE theta: local layers use a small base, global layers use a large base.
# Gemma 3 stores these as rope_local_base_freq (local) and rope_theta (global).
ROPE_THETA_LOCAL = int(cfg.get("rope_local_base_freq", 10_000))
ROPE_THETA_GLOBAL = int(cfg.get("rope_theta", 1_000_000))


def is_global(layer_idx: int) -> bool:
    """True for global (full-attention) layers (Gemma 3 1B: 5, 11, 17, 23)."""
    return (layer_idx + 1) % 6 == 0


print("Architecture loaded from config.json:")
print(f"  hidden={HIDDEN}, intermediate={INTERMEDIATE}, layers={N_LAYERS}")
print(f"  N_Q={N_Q}, N_KV={N_KV}, head_dim={HEAD_DIM}  (GQA ratio {N_Q // N_KV}x)")
print(f"  vocab={VOCAB}, rms_eps={RMS_EPS}")
print(f"  sliding_window={SLIDING_WINDOW}")
print(
    f"  rope_theta_local={ROPE_THETA_LOCAL:,}, rope_theta_global={ROPE_THETA_GLOBAL:,}"
)
print()
print(f"{'Layer':>5}  {'Type':>8}  {'RoPE theta':>12}  {'Window':>8}")
print("-" * 42)
for i in range(N_LAYERS):
    kind = "global" if is_global(i) else "local"
    theta = ROPE_THETA_GLOBAL if is_global(i) else ROPE_THETA_LOCAL
    window = "full" if is_global(i) else f"{SLIDING_WINDOW:,}"
    print(f"{i:>5}  {kind:>8}  {theta:>12,}  {window:>8}")

# %%
# Concept 1: GeGLU - ``gelu_tanh`` replaces ``silu``
# --------------------------------------------------
#
#
# Part 1 used FlashInfer's ``silu_and_mul`` kernel. Gemma 3 swaps the gate activation:
#
#
# .. code-block:: text
#
#    SiLU-GLU (Llama, Gemma 2):  out = silu(gate) * up
#    GeGLU    (Gemma 3):         out = gelu_tanh(gate) * up
#
#
# where ``gelu_tanh`` is the tanh-approximated GELU, matching ``torch.nn.functional.gelu(x, approximate="tanh")``.
#
# FlashInfer ships all three variants - ``silu``, ``gelu``, ``gelu_tanh`` - through the same ``gen_act_and_mul_module`` interface. Switching from Part 1 is a one-word change:
#
#
# .. code-block:: python
#
#    # Part 1
#    silu_module = gen_act_and_mul_module('silu').build_and_load()
#
#    # Part 2 - Gemma 3
#    gelu_module = gen_act_and_mul_module('gelu_tanh').build_and_load()
#
#
# Everything else - the three-step bridge pattern, the wrapper, the ``ffi_call`` shape declaration - is identical.

# %%
# Compile and register all kernels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# All four kernel pairs - ``gelu_tanh_and_mul``, ``apply_rope``, and local/global variants of both decode and prefill attention - are compiled and registered in a single cell below. Concepts 2 (QK-norm) and 3 (dual RoPE theta) are explained in the sections that follow; they require no additional kernels beyond what Part 1 introduced.

# %%
import torch as _torch  # used only for dtype spec in gen_single_*_module
from flashinfer.jit import (
    gen_act_and_mul_module,
    gen_single_decode_module,
    gen_single_prefill_module,
)
from flashinfer.jit.rope import gen_rope_module

# -- 1. gelu_tanh_and_mul ------------------------------------------------------
print("Compiling gelu_tanh_and_mul...")
_gelu_mod = gen_act_and_mul_module("gelu_tanh").build_and_load()


def _gelu_wrapper(out, x, enable_pdl):
    _gelu_mod.gelu_tanh_and_mul(out, x, enable_pdl)


jax_tvm_ffi.register_ffi_target(
    "flashinfer.gelu_tanh_and_mul",
    _gelu_wrapper,
    arg_spec=["rets", "args", "attrs.enable_pdl"],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)


def gelu_and_mul(x: jax.Array) -> jax.Array:
    """Fused gelu_tanh(gate) * up.  Input: [..., 2H]  Output: [..., H]"""
    out_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    return jax.ffi.ffi_call(
        "flashinfer.gelu_tanh_and_mul",
        jax.ShapeDtypeStruct(out_shape, x.dtype),
        vmap_method="broadcast_all",  # element-wise op: independent across any batch dim
    )(x, enable_pdl=False)


# -- 2. apply_rope -------------------------------------------------------------
print("Compiling apply_rope...")
_rope_mod = gen_rope_module().build_and_load()


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
    _rope_mod.apply_rope(
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


def apply_rope(q, k, indptr, offsets, rope_theta=1e4):
    """Apply RoPE to packed batches.  Returns (q_rope, k_rope)."""
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
        rotary_dim=q.shape[-1],
        interleave=False,
        rope_scale=1.0,
        rope_theta=float(rope_theta),
    )


# -- 3. decode_attention: local + global variants -------------------------------
_TMP_ELEMS = 32 * 1024 * 1024 // 2  # 32 MB scratch buffer in bfloat16 elements

print(f"Compiling decode attention (local, sliding-window={SLIDING_WINDOW})...")
_local_dec_mod = gen_single_decode_module(
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    True,
    False,
).build_and_load()
print("Compiling decode attention (global, full attention)...")
_global_dec_mod = gen_single_decode_module(
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    False,
    False,
).build_and_load()


def _make_decode_wrapper(run_fn):
    def _w(
        out,
        tmp,
        lse_or_empty,
        q,
        k,
        v,
        alibi_or_empty,
        layout,
        window_left,
        logits_soft_cap,
        sm_scale,
        rope_rcp_scale,
        rope_rcp_theta,
    ):
        lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
        alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
        run_fn(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            layout,
            window_left,
            alibi,
            logits_soft_cap,
            sm_scale,
            rope_rcp_scale,
            rope_rcp_theta,
        )

    return _w


_DEC_ARG_SPEC = [
    "rets",
    "args",
    "attrs.layout",
    "attrs.window_left",
    "attrs.logits_soft_cap",
    "attrs.sm_scale",
    "attrs.rope_rcp_scale",
    "attrs.rope_rcp_theta",
]
_KW = dict(platform="gpu", allow_cuda_graph=True, pass_owned_tensor=True)

jax_tvm_ffi.register_ffi_target(
    "flashinfer.decode_local",
    _make_decode_wrapper(_local_dec_mod.run),
    _DEC_ARG_SPEC,
    **_KW,
)
jax_tvm_ffi.register_ffi_target(
    "flashinfer.decode_global",
    _make_decode_wrapper(_global_dec_mod.run),
    _DEC_ARG_SPEC,
    **_KW,
)


def decode_attention(q, k_cache, v_cache, global_layer=False):
    """Single-request GQA decode attention.

    q:       [N_Q, HEAD_DIM]              bfloat16
    k_cache: [seq_len, N_KV, HEAD_DIM]   bfloat16
    v_cache: [seq_len, N_KV, HEAD_DIM]   bfloat16
    Returns: [N_Q, HEAD_DIM]
    """
    target = "flashinfer.decode_global" if global_layer else "flashinfer.decode_local"
    window = -1 if global_layer else SLIDING_WINDOW
    out, _, _ = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((_TMP_ELEMS,), jnp.bfloat16),
            jax.ShapeDtypeStruct((0,), jnp.float32),
        ),
        # vmap_method intentionally omitted: the scratch buffer has no batch
        # dimension, and GQA head-grouping does not decompose over an outer batch axis.
    )(
        q,
        k_cache,
        v_cache,
        jnp.empty((0,), dtype=jnp.float32),
        layout=0,
        window_left=window,
        logits_soft_cap=0.0,
        sm_scale=SM_SCALE,
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0,
    )
    return out


# -- 4. prefill_attention: local + global variants ------------------------------
print(f"Compiling prefill attention (local, sliding-window={SLIDING_WINDOW})...")
_local_pre_mod = gen_single_prefill_module(
    "fa2",
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    True,
    False,
    False,
).build_and_load()
print("Compiling prefill attention (global, full attention)...")
_global_pre_mod = gen_single_prefill_module(
    "fa2",
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    False,
    False,
    False,
).build_and_load()


def _make_prefill_wrapper(run_fn):
    def _w(
        out,
        tmp,
        lse_or_empty,
        q,
        k,
        v,
        alibi_or_empty,
        mask_mode_code,
        layout,
        window_left,
        logits_soft_cap,
        sm_scale,
        rope_rcp_scale,
        rope_rcp_theta,
    ):
        lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
        alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
        run_fn(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            mask_mode_code,
            layout,
            window_left,
            None,
            alibi,
            logits_soft_cap,
            sm_scale,
            rope_rcp_scale,
            rope_rcp_theta,
        )

    return _w


_PRE_ARG_SPEC = [
    "rets",
    "args",
    "attrs.mask_mode_code",
    "attrs.layout",
    "attrs.window_left",
    "attrs.logits_soft_cap",
    "attrs.sm_scale",
    "attrs.rope_rcp_scale",
    "attrs.rope_rcp_theta",
]

jax_tvm_ffi.register_ffi_target(
    "flashinfer.prefill_local",
    _make_prefill_wrapper(_local_pre_mod.run),
    _PRE_ARG_SPEC,
    **_KW,
)
jax_tvm_ffi.register_ffi_target(
    "flashinfer.prefill_global",
    _make_prefill_wrapper(_global_pre_mod.run),
    _PRE_ARG_SPEC,
    **_KW,
)


def prefill_attention(q, k, v, layer_i):
    """FlashInfer causal GQA attention for processing a multi-token prompt.

    Uses the same kernel-bridge pattern as decode_attention: mask_mode_code=1
    selects causal masking; window_left controls the sliding-window cut-off.

    q, k, v: [T, n_heads, HEAD_DIM]  bfloat16
    Returns:  [T, N_Q,    HEAD_DIM]  bfloat16
    """
    glob = is_global(layer_i)
    target = "flashinfer.prefill_global" if glob else "flashinfer.prefill_local"
    window = -1 if glob else SLIDING_WINDOW
    out, _, _ = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),  # output
            jax.ShapeDtypeStruct((_TMP_ELEMS,), jnp.bfloat16),  # tmp scratch
            jax.ShapeDtypeStruct((0,), jnp.float32),
        ),  # lse (discard)
        # vmap_method intentionally omitted: scratch buffer + GQA head-grouping
        # do not decompose over an outer batch axis.
    )(
        q,
        k,
        v,
        jnp.empty((0,), dtype=jnp.float32),  # alibi = empty sentinel
        mask_mode_code=1,
        layout=0,
        window_left=window,
        logits_soft_cap=0.0,
        sm_scale=SM_SCALE,
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0,
    )
    return out


print("All kernels compiled and registered.")

# %%
# Concept 2: QK-norm - per-head normalization
# -------------------------------------------
#
#
# Gemma 2 bounded attention score magnitudes with logit soft-capping: ``scores = tanh(scores / 50) x 50``. Gemma 3 replaces this with **QK-norm**: an RMSNorm applied independently to each query and key head *after* the linear projection and *before* the dot product.
#
#
# .. code-block:: python
#
#    # Gemma 2 (inside the attention kernel, via logits_soft_cap parameter)
#    scores = tanh(q @ k.T / sqrt(d) / 50) * 50
#
#    # Gemma 3 (in JAX, before calling decode_attention)
#    q = rms_norm_per_head(q, q_norm_weight)   # [N_Q,  head_dim]
#    k = rms_norm_per_head(k, k_norm_weight)   # [N_KV, head_dim]
#    scores = q @ k.T / sqrt(d)               # bounded by weight norms
#
#
# The norm weights ``q_norm.weight`` and ``k_norm.weight`` have shape ``[head_dim]`` - the same weight is shared across all heads. In the model state dict they are ``model.layers.{i}.self_attn.q_norm.weight``.
#
#
# Concept 3: Dual RoPE theta - one theta per attention scope
# ----------------------------------------------------------
#
#
# Standard RoPE uses a single base frequency theta. Gemma 3 uses two:
#
#
# .. list-table::
#    :header-rows: 1
#
#    * - Layer type
#      - theta
#      - Why
#    * - Local (5/6 of layers)
#      - 10 000
#      - Standard positional bias for the 1 024-token window
#    * - Global (1/6 of layers)
#      - 1 000 000
#      - Slower-decaying frequencies for long-range context
#
#
# In code this is a single ``if`` in the layer function - we select the right theta and pass it to ``apply_rope`` as a scalar attribute. The kernel is compiled once; the theta value is a runtime parameter.

# %%
# -- Pure-JAX building blocks ---------------------------------------------------


@jax.jit
def rms_norm(x, weight, eps=RMS_EPS):
    """Gemma-style RMSNorm: normalise then scale by (1 + weight)."""
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * (1.0 + weight)


@jax.jit
def qk_norm(x, weight):
    """Per-head RMSNorm for Q or K vectors.  x: [..., head_dim]."""
    return rms_norm(x, weight)


def embed(token_ids):
    """Embedding lookup.  Gemma multiplies by sqrt(hidden_size) to keep
    hidden-state norms stable through the first RMSNorm.

    token_ids: [T]  ->  [T, HIDDEN]
    """
    return weights["model.embed_tokens.weight"][token_ids] * math.sqrt(HIDDEN)


def lm_head(h):
    """Project hidden state to vocabulary logits.  h: [HIDDEN]  ->  [VOCAB] float32."""
    # Gemma 3 ties the LM head to the embedding matrix
    lm_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return h.astype(jnp.float32) @ lm_w.astype(jnp.float32).T


def ffn(h, layer_i):
    """GeGLU feed-forward block.  h: [..., HIDDEN]  ->  [..., HIDDEN].

    Handles both single-token decode (h: [HIDDEN]) and
    full-sequence prefill (h: [T, HIDDEN]) with the same code.
    """
    pre = rms_norm(
        h, weights[f"model.layers.{layer_i}.pre_feedforward_layernorm.weight"]
    )
    gate = (
        pre @ weights[f"model.layers.{layer_i}.mlp.gate_proj.weight"].T
    )  # [..., INTER]
    up = pre @ weights[f"model.layers.{layer_i}.mlp.up_proj.weight"].T  # [..., INTER]

    # Concatenate along the last axis: gelu_and_mul splits it back in two
    gate_up = jnp.concatenate([gate, up], axis=-1)  # [..., 2*INTER]
    hidden = gelu_and_mul(gate_up)  # [..., INTER]  <- FlashInfer kernel

    out = hidden @ weights[f"model.layers.{layer_i}.mlp.down_proj.weight"].T
    out = rms_norm(
        out, weights[f"model.layers.{layer_i}.post_feedforward_layernorm.weight"]
    )
    return out


# Quick sanity check on the FFN
_x_test = jax.random.normal(jax.random.key(0), (HIDDEN,), dtype=jnp.bfloat16)
_out = ffn(_x_test, 0)
print(
    f"FFN layer 0: {_x_test.shape} -> {_out.shape}  dtype={_out.dtype}  ok={not jnp.any(jnp.isnan(_out))}"
)

# %%
# Prefill layer and full forward pass
# -----------------------------------
#
#
# ``prefill_layer`` processes all prompt tokens in parallel through one transformer layer and builds the initial KV-cache. ``prefill`` chains it across all 26 layers, then applies ``rms_norm`` to the last token's hidden state and returns the per-layer KV-caches that the decode loop will update.

# %%
# -- Prefill layer (processes all T prompt tokens in parallel) ------------------


def prefill_layer(h, layer_i):
    """Run one transformer layer over the full prompt.

    h: [T, HIDDEN]  bfloat16
    Returns: (h: [T, HIDDEN], kv_cache: (k: [T, N_KV, D], v: [T, N_KV, D]))
    """
    T = h.shape[0]
    glob = is_global(layer_i)
    rope_theta = ROPE_THETA_GLOBAL if glob else ROPE_THETA_LOCAL

    # -- Attention -------------------------------------------------------------
    ln = rms_norm(h, weights[f"model.layers.{layer_i}.input_layernorm.weight"])

    q = (ln @ weights[f"model.layers.{layer_i}.self_attn.q_proj.weight"].T).reshape(
        T, N_Q, HEAD_DIM
    )
    k = (ln @ weights[f"model.layers.{layer_i}.self_attn.k_proj.weight"].T).reshape(
        T, N_KV, HEAD_DIM
    )
    v = (ln @ weights[f"model.layers.{layer_i}.self_attn.v_proj.weight"].T).reshape(
        T, N_KV, HEAD_DIM
    )

    # QK-norm (per head, same weight across all token positions)
    q = qk_norm(q, weights[f"model.layers.{layer_i}.self_attn.q_norm.weight"])
    k = qk_norm(k, weights[f"model.layers.{layer_i}.self_attn.k_norm.weight"])

    # RoPE over all T tokens at once: one sequence starting at offset 0
    indptr = jnp.array([0, T], dtype=jnp.int32)
    offsets = jnp.array([0], dtype=jnp.int32)
    q, k = apply_rope(q, k, indptr, offsets, rope_theta=rope_theta)

    # FlashInfer causal attention
    attn_out = prefill_attention(q, k, v, layer_i)

    attn_out = attn_out.reshape(T, N_Q * HEAD_DIM)
    attn_out = attn_out @ weights[f"model.layers.{layer_i}.self_attn.o_proj.weight"].T
    attn_out = rms_norm(
        attn_out, weights[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
    )
    h = h + attn_out

    # -- FFN (works naturally for [T, HIDDEN]) --------------------------------
    h = h + ffn(h, layer_i)

    # KV-cache: store the RoPE-applied K and raw V for all prompt positions
    return h, (k, v)


# -- Full prefill pass ---------------------------------------------------------


def prefill(prompt_ids):
    """Process the full prompt.  Returns (h_last: [HIDDEN], kv_caches)."""
    h = embed(jnp.array(prompt_ids))  # [T, HIDDEN]

    kv_caches = []
    for i in range(N_LAYERS):
        h, kv_cache = prefill_layer(h, i)
        kv_caches.append(kv_cache)

    # Final norm applied to the last token's hidden state
    h_last = rms_norm(h[-1], weights["model.norm.weight"])  # [HIDDEN]
    return h_last, kv_caches


# %%
# Decode attention layer
# ----------------------
#
#
# ``decode_layer`` processes one newly generated token through a full transformer layer. It applies QK-norm, selects the layer's RoPE theta, calls ``apply_rope`` and ``decode_attention``, appends to the KV-cache, and returns the updated hidden state.

# %%
# -- Decode attention layer (one new token, growing KV-cache) -------------------


def decode_layer(h, layer_i, kv_cache, pos):
    """Process a single new token through one transformer layer.

    h:        [HIDDEN]                           bfloat16
    kv_cache: (k: [pos, N_KV, D], v: [pos, N_KV, D])
    pos:      current token's position in the full sequence (Python int)
    Returns:  (h: [HIDDEN], updated_kv_cache)
    """
    glob = is_global(layer_i)
    rope_theta = ROPE_THETA_GLOBAL if glob else ROPE_THETA_LOCAL

    # -- Attention -------------------------------------------------------------
    ln = rms_norm(h, weights[f"model.layers.{layer_i}.input_layernorm.weight"])

    q = (ln @ weights[f"model.layers.{layer_i}.self_attn.q_proj.weight"].T).reshape(
        N_Q, HEAD_DIM
    )
    k = (ln @ weights[f"model.layers.{layer_i}.self_attn.k_proj.weight"].T).reshape(
        N_KV, HEAD_DIM
    )
    v = (ln @ weights[f"model.layers.{layer_i}.self_attn.v_proj.weight"].T).reshape(
        N_KV, HEAD_DIM
    )

    # QK-norm (Concept 2: Gemma 3 replaces soft-capping with per-head RMSNorm)
    q = qk_norm(q, weights[f"model.layers.{layer_i}.self_attn.q_norm.weight"])
    k = qk_norm(k, weights[f"model.layers.{layer_i}.self_attn.k_norm.weight"])

    # apply_rope with the layer's theta (Concept 3: different theta for local vs global)
    q_pack, k_pack = q[None], k[None]  # [1, heads, D]  packed batch of 1 token
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    offsets = jnp.array([pos], dtype=jnp.int32)
    q_r, k_r = apply_rope(q_pack, k_pack, indptr, offsets, rope_theta=rope_theta)
    q_r = q_r.squeeze(0)  # [N_Q,  D]
    k_r = k_r.squeeze(0)  # [N_KV, D]

    # Append RoPE'd K and raw V to the KV-cache
    # NOTE: Using jnp.concatenate to grow KV cache is intentional.
    # In standard JAX this is inefficient (O(N^2)) and you'd normally preallocate
    # and use lax.dynamic_update_slice. However, FlashInfer's single-request
    # decode kernel infers sequence length from k_cache/v_cache.shape.
    # Therefore we must keep the cache length equal to the actual number of tokens.
    # Switching to a fixed-size buffer would require a different FlashInfer API
    # (e.g. paged KV cache) or an explicit length/mask.
    k_cache, v_cache = kv_cache
    k_cache = jnp.concatenate([k_cache, k_r[None]], axis=0)  # [pos+1, N_KV, D]
    v_cache = jnp.concatenate([v_cache, v[None]], axis=0)  # [pos+1, N_KV, D]

    # Decode attention over the full KV-cache (FlashInfer kernel)
    attn_out = decode_attention(q_r, k_cache, v_cache, global_layer=glob)  # [N_Q, D]

    attn_out = attn_out.reshape(N_Q * HEAD_DIM)
    attn_out = attn_out @ weights[f"model.layers.{layer_i}.self_attn.o_proj.weight"].T
    attn_out = rms_norm(
        attn_out, weights[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
    )
    h = h + attn_out

    # -- FFN -------------------------------------------------------------------
    h = h + ffn(h, layer_i)

    return h, (k_cache, v_cache)


# %%
# Decode step
# -----------
#
#
#    **Why there is no ``@jax.jit`` here**
#
#    The FlashInfer kernels (``decode_attention``, ``apply_rope``, ...) are fully ``@jax.jit``-compatible XLA custom calls. The obstacle is the KV-cache. Each decode step appends one new row:
#
#    ```python
#    k_cache = jnp.concatenate([k_cache, k_r[None]], axis=0)  # shape grows every step
#    ```
#
#    ``@jax.jit`` requires statically known output shapes. Because ``k_cache.shape[0]`` increments at every step, XLA would have to recompile ``decode_step`` on each call - far more expensive than running eagerly.
#
#    A production system fixes this by pre-allocating a maximum-length cache and writing into it with ``jax.lax.dynamic_update_slice``, which keeps shapes static and allows the entire decode loop to be compiled with ``jax.lax.scan``. That is the paged KV-cache direction described in the Summary.

# %%
# -- One decode step -----------------------------------------------------------


def decode_step(token_id, kv_caches, pos):
    """Process one newly generated token and predict the next.

    token_id:  int   the most recently produced token
    kv_caches: list  one (k, v) tuple per layer
    pos:       int   this token's position in the full sequence
    Returns:   (logits: [VOCAB] float32, updated_kv_caches)
    """
    h = embed(jnp.array([token_id])).squeeze(0)  # [HIDDEN]

    new_kv = []
    for i in range(N_LAYERS):
        h, kv = decode_layer(h, i, kv_caches[i], pos)
        new_kv.append(kv)

    h = rms_norm(h, weights["model.norm.weight"])
    logits = lm_head(h)
    return logits, new_kv


# -- Stop tokens ---------------------------------------------------------------
# Gemma instruct ends its turn with <end_of_turn>, not the generic <eos>.
# Collect all token IDs that should halt generation.
_STOP_IDS = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
for _tok in ["<end_of_turn>", "<eos>"]:
    _id = tokenizer.convert_tokens_to_ids(_tok)
    if _id is not None and _id != tokenizer.unk_token_id:
        _STOP_IDS.add(_id)


# -- Text generation -----------------------------------------------------------


def generate(prompt, max_new_tokens=200, temperature=0.7, seed=0):
    """Autoregressive generation with the Gemma 3 instruct chat template."""
    messages = [{"role": "user", "content": prompt}]

    # Render chat template to plain text first.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Then tokenize explicitly and extract only input_ids.
    enc = tokenizer(rendered, add_special_tokens=False)
    prompt_ids = enc["input_ids"]

    # Flatten batch dimension if present.
    if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    T = len(prompt_ids)
    key = jax.random.key(seed)

    print(f"Prompt ({T} tokens): {prompt!r}")
    print(f"Rendered prompt preview: {rendered[:120]!r}")

    print("Prefilling...", end=" ", flush=True)
    t0 = time.perf_counter()
    h_last, kv_caches = prefill(prompt_ids)
    jax.block_until_ready(h_last)
    print(f"{time.perf_counter() - t0:.1f}s")

    def _sample(logits, key):
        if temperature == 0.0:
            return int(jnp.argmax(logits)), key
        key, subkey = jax.random.split(key)
        return int(jax.random.categorical(subkey, logits / temperature)), key

    print("Response: ", end="", flush=True)

    generated = []
    for step in range(max_new_tokens):
        if step == 0:
            logits = lm_head(h_last)
        else:
            logits, kv_caches = decode_step(generated[-1], kv_caches, T + step - 1)
        next_tok, key = _sample(logits, key)
        generated.append(next_tok)
        if next_tok in _STOP_IDS:
            break
        print(tokenizer.decode([next_tok]), end="", flush=True)

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)


# %%
# Running inference
# -----------------
#
#
# The cell below runs the model on three sample questions using the Gemma 3 instruct chat template. XLA compiles the kernels on the first call; subsequent prompts reuse the cached compilation.

# %%
questions = [
    "What is the capital of Germany",
    "How does rotary positional embedding differ from learned positional embedding",
    "What is grouped-query attention and why is it useful",
]

for q in questions:
    generate(q, max_new_tokens=150, temperature=0.7, seed=0)
    print()

# %%
# Summary
# -------
#
#
# We have implemented end-to-end autoregressive inference for Gemma 3 1B Instruct using four FlashInfer kernels as the computational backbone - covering both the prompt (prefill) and generation (decode) phases.
#
#
# The complete inference recipe
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. code-block:: python
#
#    # 1. Compile kernels (once)
#    gelu_module          = gen_act_and_mul_module('gelu_tanh').build_and_load()
#    rope_module          = gen_rope_module().build_and_load()
#    decode_local_module  = gen_single_decode_module(..., use_sliding_window=True, ...).build_and_load()
#    decode_global_module = gen_single_decode_module(..., use_sliding_window=False, ...).build_and_load()
#    prefill_local        = gen_single_prefill_module('fa2', ..., use_sliding_window=True).build_and_load()
#    prefill_global       = gen_single_prefill_module('fa2', ..., use_sliding_window=False).build_and_load()
#
#    # 2. Prefill: FlashInfer causal attention over all prompt tokens -> KV-cache
#    h_last, kv_caches = prefill(prompt_ids)
#
#    # 3. Decode: FlashInfer decode_attention, one token at a time
#    for step in range(max_new_tokens):
#        if step == 0:
#            logits = lm_head(h_last)
#        else:
#            logits, kv_caches = decode_step(generated[-1], kv_caches, T + step - 1)
#
#
# Going further
# ~~~~~~~~~~~~~
#
#
# - **Paged KV-cache**: Replace the growing list with a fixed-size paged cache and use FlashInfer's ``BatchDecodeWithPagedKVCacheWrapper`` for batch inference with mixed sequence lengths.
# - **Sampling**: Extend the sampler with top-p nucleus sampling or top-k filtering on the logits.
# - **Continuous batching**: Process multiple requests simultaneously, filling the decode kernel's batch dimension.
