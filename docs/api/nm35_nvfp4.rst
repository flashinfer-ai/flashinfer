.. _apinm35_nvfp4:

flashinfer.nm35_nvfp4
=====================

``nm35-nvfp4`` is an explicit attention backend for Nemotron 3.5 Super
with an NVFP4 KV cache on SM100 GPUs (``sm_100a``); it was validated on GB200.
It is selected with ``backend="nm35-nvfp4"`` on
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` and
:func:`flashinfer.prefill.trtllm_batch_context_with_kv_cache`, takes the same
arguments as ``backend="trtllm-gen"``, and never participates in automatic
backend selection.

A call that runs on ``trtllm-gen`` returns exactly what ``backend="trtllm-gen"``
returns. A call that runs on one of the backend's kernels returns the same
attention output up to FP8 rounding, not the same bits: see `Numerics`_.

Supported geometry
------------------

The backend adds three kernels, compiled from source for ``sm_100a``. They
serve one attention geometry:

* FP8 E4M3 query and output;
* NVFP4 K/V cache with FP8 E4M3 block scales (``kv_cache_sf``);
* 32 query heads, 2 KV heads, head dimension 128;
* 128-token pages with the shared page-index layout. The kernels read ``HND``
  pages; ``NHD`` input is first copied to ``HND``, as for ``trtllm-gen``;
* causal dense attention: no sliding window, sinks, LSE output, skip-softmax
  or block-sparse attention.

A call outside this geometry runs on ``trtllm-gen``, unchanged.

The kernels build with the CUDA 13.0 toolkit or newer. From CUDA 13.4 on, the
decode kernels dequantize NVFP4 with one PTX instruction per four values; older
assemblers reject it and get a bit-identical sequence through FP16 instead.
Built that way, decode was measured about 1.25 to 1.4 times slower than a 13.4
build at 16 or more requests, so such a build routes fewer decode shapes to its kernels
(see `Routing`_); prefill is unaffected. A module built that way says so once at
``INFO`` level. CUDA 13.4 or newer is recommended.
Where the module is neither prebuilt nor buildable, the backend logs one warning
and runs every call on ``trtllm-gen``.

Routing
-------

Inside the geometry the kernel is chosen from the call's shape, from whether it
passes ``cum_seq_lens_q`` and, for decode, from whether the module was built
with the fused NVFP4 multiply (CUDA 13.4 or newer). The thresholds were measured
against ``trtllm-gen`` on GB200 (152 SMs) and are applied to every ``sm_100``
device.

**Decode** (:func:`select_decode_kernel`) with the fused multiply, first match
wins:

.. list-table::
   :header-rows: 1

   * - Condition
     - Kernel
   * - longest KV of at most 8192 tokens
     - ``trtllm-gen``
   * - more than 8 query tokens per request, or more than 256 requests
     - ``trtllm-gen``
   * - 2, 4, 6 or 8 query tokens and ``cum_seq_lens_q``
     - ``trtllm-gen``
   * - 6 query tokens, at most 2 requests
     - ``mtp``
   * - 6 or 7 query tokens
     - ``trtllm-gen``
   * - 2, 4 or 8 query tokens
     - ``mtp``
   * - 1, 3 or 5 query tokens
     - ``base``

**Decode** without the fused multiply (CUDA 13.0 to 13.3): the same first three
rows apply, then a call runs on a kernel only in these cases, all others on
``trtllm-gen``. They are the shapes where the kernels were measured clearly
faster than ``trtllm-gen`` with a CUDA 13.0 build.

.. list-table::
   :header-rows: 1

   * - Query tokens
     - Requests
     - Longest KV
     - Kernel
   * - 2
     - any
     - any
     - ``mtp``
   * - 1, 3 or 5
     - 1
     - any
     - ``base``
   * - 4 or 8
     - 1
     - any
     - ``mtp``
   * - 5
     - 2
     - any
     - ``base``
   * - 8
     - 2
     - any
     - ``mtp``
   * - 8
     - 16 or more
     - any
     - ``mtp``
   * - 5
     - 16 or more
     - more than 32768 tokens
     - ``base``
   * - 1
     - 16 or more
     - more than 65536 tokens
     - ``base``

Requests differ in query length only in a call that passes
``cum_seq_lens_q``; both tables then read the query tokens of the longest
request. ``base`` runs such a call one query token at a time, and ``mtp`` does
not serve it. This holds even when the offsets give every request the same
length, because a CUDA graph captured from such a call replays whatever
offsets it holds then. A call whose requests always have the same length runs
on ``mtp``, or on ``base`` with all its query tokens at once, only when it
passes ``q_len_per_req`` instead.

A kernel also declines a call it cannot serve, such as a workspace that is too
small or tensors that are not contiguous; that call runs on ``trtllm-gen``. The
reason is logged at ``DEBUG`` level on the ``flashinfer.nm35_nvfp4``
logger.

**Prefill** (:func:`select_prefill_kernel`): a call with one request of at most
32768 query tokens and a KV longer than 8192 tokens runs on the prefill kernel,
provided ``workspace_buffer`` holds at least :func:`prefill_workspace_bytes`.
Every other call runs on ``trtllm-gen``.

``max_kv_len`` must be at least the true KV length of the request, as for
``trtllm-gen``: the prefill kernel dequantizes exactly that many tokens.

Workspace
---------

The prefill kernel first dequantizes every NVFP4 page the request can see into
``workspace_buffer`` and runs FP8 attention on that copy. This needs 64 KiB per
page plus a fixed 72 MiB: 584 MiB for a 1M-token KV. Serving one request per
call is what bounds it. Size the buffer once, for the longest KV the server
will see:

.. code-block:: python

    from flashinfer.nm35_nvfp4 import prefill_workspace_bytes

    workspace_buffer = torch.zeros(
        prefill_workspace_bytes(max_model_len), dtype=torch.uint8, device="cuda"
    )

The decode kernels need less: at most 42.8 MiB on 152 SMs, for 256 requests of 8
query tokens. The need grows with the requests, their query tokens and the SM
count.

Numerics
--------

The kernels take every K and V value as its NVFP4 value times its block scale,
rounded to FP8 E4M3 with saturation at ±448, and round the output to FP8 E4M3.
``trtllm-gen`` was measured to agree with this, also where values saturate.
Intermediate rounding differs, so the output of a kernel is close to that of
``trtllm-gen`` but not bit-identical.

Measured on GB200 over uniform and sharply peaked attention, 1 to 1M KV tokens
and 1 to 256 requests, as the relative L2 difference of one output row (one head
of one query token):

.. list-table::
   :header-rows: 1

   * - Compared
     - Mean
     - Largest row
   * - kernels and ``trtllm-gen``
     - 0.3% to 4.9%
     - 8.9%
   * - kernels and an FP32 reference
     - 1.5% to 4.1%
     - 8.9%
   * - ``trtllm-gen`` and an FP32 reference
     - 1.1% to 4.1%
     - 7.2%

The FP32 reference applies the same E4M3 rounding to K and V and computes the
rest in FP32. Under sharply peaked attention the decode kernels are further from
it than ``trtllm-gen`` on average: 1.5% to 3.5% against 1.1% to 2.1%.

:func:`route_counts` reports how many calls each kernel served, which is the
quickest way to confirm what a deployment runs.

API
---

.. currentmodule:: flashinfer.nm35_nvfp4

.. autosummary::
    :toctree: ../generated

    select_decode_kernel
    select_prefill_kernel
    prefill_workspace_bytes
    is_supported_device
    route_counts
    reset_route_counts
