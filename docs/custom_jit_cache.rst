.. _custom-jit-cache:

Custom JIT-Cache Wheels
=======================

The published ``flashinfer-jit-cache`` package installs one provider wheel per
CUDA architecture (``flashinfer-jit-cache-sm90a``, ``flashinfer-jit-cache-sm100a``,
...). Each provider holds every ahead-of-time (AOT) module FlashInfer builds for
that architecture: hundreds of attention, GEMM, MoE and communication variants.

A deployment that serves one model on one GPU type loads a small fraction of
them. ``scripts/custom_jit_cache.py`` builds provider wheels that hold only the
modules a deployment needs, for only the architectures it runs on, and a
matching ``flashinfer-jit-cache`` shim. Modules left out still work: FlashInfer
JIT-compiles them on first use, provided the CUDA toolkit is installed.

.. note::
   This is a proof of concept. The recipe format and the command line may change.

Overview
--------

A *recipe* names the modules to ship and the architectures to build for. There
are two ways to fill it in, and they combine:

* **Capture** (recommended): run a representative workload with
  ``FLASHINFER_JIT_USAGE_LOG`` set. FlashInfer appends one JSON line for each
  module the process loads, whether it came from an installed jit-cache or was
  compiled at runtime. The log lists exact module names, so the wheel matches
  the dtypes, head dimensions, page sizes and backends the workload really uses.
* **Hand-written selection**: list module names or glob patterns over them.
  ``list`` prints the names available for an architecture.

Module names encode the kernel's specialization, for example::

    batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False_sm90

so a pattern such as ``batch_prefill_with_kv_cache_dtype_q_bf16_*_head_dim_qk_128_*``
selects one family.

Requirements
------------

Building needs the same environment as a regular jit-cache build: a FlashInfer
source checkout at the release the deployment runs, installed with
``pip install --no-build-isolation -e .``, the CUDA toolkit, and the ``build``
package. ``list`` and ``resolve`` need the CUDA toolkit but no GPU. ``capture``
needs a GPU and the workload.

.. important::
   Build from the same FlashInfer version as the ``flashinfer-python`` the
   deployment runs. The provider and shim versions must match it, and module
   names can change between releases, so capture again after an upgrade.

Step 1: Capture the modules a workload uses
-------------------------------------------

.. code-block:: bash

    python scripts/custom_jit_cache.py capture --log usage.jsonl -- \
        python my_server.py --model meta-llama/Llama-3.1-8B-Instruct

``capture`` only sets ``FLASHINFER_JIT_USAGE_LOG`` and runs the command, so
setting the variable yourself works the same way, including for every rank of a
multi-process job (records are appended atomically, one line each):

.. code-block:: bash

    export FLASHINFER_JIT_USAGE_LOG=$PWD/usage.jsonl
    torchrun --nproc-per-node 8 my_server.py ...

Capture runs fastest with the full ``flashinfer-jit-cache`` for the GPU
installed, because nothing has to compile. Without it, every module the
workload touches is JIT-compiled once during capture.

A capture sees only the code paths the run exercises. Cover everything the
deployment will do: prefill and decode, every batch-size bucket and CUDA-graph
capture size, speculative decoding, LoRA, and any autotuning, which loads
candidate kernels it may later choose. Append several runs to the same log or
pass several logs to the recipe.

Step 2: Write a recipe
----------------------

Generate a recipe from the log:

.. code-block:: bash

    python scripts/custom_jit_cache.py init --name llama8b-h100 --from-usage usage.jsonl

which writes ``llama8b-h100.json``:

.. code-block:: json

    {
      "name": "llama8b-h100",
      "cuda_archs": ["9.0a"],
      "usage_logs": ["usage.jsonl"],
      "modules": [],
      "include": [],
      "exclude": [],
      "aot_config": {}
    }

Recipe keys:

``name``
    Identifies the build; recorded in each provider's manifest as ``selection``.
``cuda_archs``
    Architectures to build one provider wheel each for, in any spelling the
    regular build accepts (``9.0a``, ``sm100a``, ``12.0f``). ``init`` fills in
    the architectures seen while capturing; edit them to build for other GPUs.
``usage_logs``
    ``FLASHINFER_JIT_USAGE_LOG`` files, relative to the recipe. Their nvcc
    modules are added to ``modules``.
``modules``
    Exact module names to ship.
``include`` / ``exclude``
    Shell-style glob patterns (``*``, ``?``, ``[...]``) matched against full
    module names. ``exclude`` wins over ``modules`` and ``include``.
``aot_config``
    Overrides for the AOT candidate set before selection, with the keys of
    ``flashinfer.aot.get_default_config()``: ``fa2_head_dim`` and
    ``fa3_head_dim`` as ``[[qk, vo], ...]``; ``f16_dtype`` and ``f8_dtype`` as
    torch dtype names; ``use_sliding_window`` and ``use_logits_soft_cap`` as
    lists of booleans; and the ``add_*`` booleans. It can add variants the
    default build leaves out, such as ``"fa2_head_dim": [[512, 512]]``, or make
    resolution faster by skipping families.

TOML recipes (``.toml``) with the same keys are accepted on Python 3.11 and later.
``scripts/custom_jit_cache_recipes/`` has a hand-written example.

Step 3: Check the selection
---------------------------

.. code-block:: bash

    python scripts/custom_jit_cache.py resolve llama8b-h100.json -v

prints, for each architecture, how many of the available AOT modules the recipe
selects (with ``-v``, their names) and flags:

* requested modules that are not in that architecture's AOT set: modules for
  another architecture, or modules FlashInfer never builds ahead of time. These
  are JIT-compiled at runtime.
* ``include`` patterns that match nothing.
* captured modules from other toolchains (CuTe-DSL, NVRTC, ...). Jit-cache wheels
  only carry nvcc modules; the others always compile at runtime and keep their
  own on-disk caches.

``--strict`` makes ``resolve`` fail on any of the first two, which is useful in
CI. To browse the available names:

.. code-block:: bash

    python scripts/custom_jit_cache.py list --arch 9.0a --pattern '^batch_decode' 'norm'

Step 4: Build and install
-------------------------

.. code-block:: bash

    python scripts/custom_jit_cache.py build llama8b-h100.json --local-version cu130

builds one provider wheel per architecture in ``cuda_archs`` and a
``flashinfer-jit-cache`` shim that depends on exactly those providers. Wheels go
to ``dist/custom-jit-cache/<name>/`` (``--out-dir`` to change it) with a
``<name>.report.json`` listing each wheel's size and module count.

.. code-block:: bash

    pip install dist/custom-jit-cache/llama8b-h100/*.whl

The custom providers keep the official distribution names
(``flashinfer-jit-cache-sm90a``) so the shim, the runtime discovery and
``pip`` treat them as drop-in replacements. Do not install them next to the
official jit-cache for the same version.

``--local-version`` sets the PEP 440 local label of both wheels (default:
``$FLASHINFER_LOCAL_VERSION``). The runtime accepts a jit-cache whose version is
the ``flashinfer-python`` version plus any local label, so ``cu130.llama8b``
is also valid. ``--no-shim`` builds only the providers.

Compiled objects are kept in ``build/aot-providers/<arch>/`` and reused by later
builds, including full ones, so iterating on a recipe only compiles new modules.

Verifying a wheel
-----------------

Run the workload with the custom wheels installed and JIT disabled. Any module
the wheel lacks raises ``MissingJITCacheError`` naming it, instead of compiling
silently:

.. code-block:: bash

    FLASHINFER_DISABLE_JIT=1 python my_server.py ...

Add the named module to the recipe's ``modules`` and rebuild. Runtime-only
modules (CuTe-DSL and other non-nvcc toolchains) are not in any jit-cache, so
under ``FLASHINFER_DISABLE_JIT`` they load only from their own on-disk caches;
``resolve`` lists the ones the capture recorded.

Building without the script
---------------------------

The script drives the regular provider build with one extra variable,
``FLASHINFER_JIT_CACHE_PROVIDER_SELECTION``, which points at a JSON *selection*
(the recipe without ``cuda_archs`` and ``usage_logs``):

.. code-block:: bash

    export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
    export FLASHINFER_JIT_CACHE_PROVIDER_SELECTION=$PWD/selection.json
    python -m build --no-isolation --skip-dependency-check --wheel flashinfer-jit-cache-provider

    export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
    python -m build --no-isolation --skip-dependency-check --wheel flashinfer-jit-cache

``--skip-dependency-check`` matters once ``FLASHINFER_LOCAL_VERSION`` names a
CUDA version (``cu130``): the build requirements then pin that CUDA release's
wheels for an isolated build, which the current environment need not match.

Remove ``flashinfer-jit-cache-provider/build/`` first when the same checkout
built a larger provider for the same architecture: setuptools reuses that
staging directory and would copy the earlier modules into the new wheel.
With ``FLASHINFER_CUDA_ARCH_LIST`` set,
``python -m flashinfer.aot --selection selection.json --out-dir <dir>`` builds
the same subset into a plain directory.

Limitations
-----------

* Only nvcc modules registered in ``flashinfer/aot.py`` can be shipped.
  Modules FlashInfer never builds ahead of time JIT-compile at runtime however
  the wheel is built.
* ``flashinfer-cubin`` (downloaded TensorRT-LLM kernels) is separate and
  unchanged.
* The usage log sees modules loaded through ``JitSpec.build_and_load()``: every
  jit-cache module, and CuTe-DSL kernels cached through ``JitSpecCuteDsl``.
  Kernels compiled by direct ``cute.compile()`` calls, Triton or cuTile are not
  recorded. On SM80, for example, ``rmsnorm`` defaults to such a CuTe-DSL kernel.
  None of these can ship in a jit-cache, so the wheel is unaffected, but the
  runtime-only list ``resolve`` prints is incomplete, and
  ``FLASHINFER_DISABLE_JIT`` does not stop these kernels from compiling.
* Selection is by module name. There is no static analysis of model code yet:
  which modules a call loads depends on runtime values (dtypes, head dimensions,
  page size, backend choice and autotuning), which capture observes directly.
