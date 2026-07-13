.. _apimagi_ffa:

flashinfer.magi_ffa
~~~~~~~~~~~~~~~~~~~

MagiAttention Flex Flash Attention adapter (experimental). MagiAttention
(SandAI, Apache-2.0) is an **optional dependency** that is never installed
automatically.

Installation
------------

MagiAttention and FlashInfer declare conflicting ``nvidia-cutlass-dsl``
constraints (MagiAttention pins ``==4.3.5``/``==4.4.2``; FlashInfer requires
``>=4.5.0``), so the install order matters — install MagiAttention *after*
FlashInfer, then restore cutlass-dsl:

.. code-block:: bash

    pip install flashinfer-python
    pip install magi_attention==1.1.0.post10
    pip install "nvidia-cutlass-dsl>=4.5.0"  # restore; magi's install downgrades it

``1.1.0.post10`` is the FlashInfer-validated MagiAttention version; other
versions import with a warning.

Support matrix
--------------

- **Hopper (SM90)** — FlashInfer-validated (opt-in CI + verified on H20).
- **Blackwell (FFA_FA4) / Ampere** — supported upstream by MagiAttention but
  *not* validated by FlashInfer CI.

API
---

.. currentmodule:: flashinfer.magi_ffa

.. autosummary::
    :toctree: ../generated

    flex_flash_attn
