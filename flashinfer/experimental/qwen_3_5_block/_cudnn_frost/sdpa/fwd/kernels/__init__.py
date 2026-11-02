# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-forward kernel templates, grouped by target architecture.

Layout
------
One package per arch line — ``sm80/``, ``sm90/`` (Hopper), ``sm100/``
(SM100/SM103), ``sm107/`` (Rubin), ``sm120/`` (SM120/SM121) — holding the
flavor templates that arch owns.  Within a package the filename encodes the
rest of the coverage matrix: ``<phase>_d<dim>_<dtype-family>.py``, e.g.
``sm100/prefill_d512_f16.py`` (``f16`` covers fp16 and bf16, picked by
``TemplateParams``).

A file omits the dimension when one implementation covers a range of head
dims: ``sm120/prefill_f16.py`` and ``sm120/prefill_fp8.py`` do, while
``sm120/prefill_d256_f16.py`` is the d256 flavor (head dims that tile at 256 on
both sides). ``sm120/prefill_d512_f16.py`` serves both head dims in (256, 512]
at multiples of eight, with two warps per Q slab splitting the head dimension;
``sm120/prefill_d512_fp8.py`` and ``sm120/prefill_d512_mxfp8.py`` are its FP8
and MXFP8 siblings (multiples of 16).
``sm90/prefill_d512_f16.py`` is that line's only template: one D512 tile over
three warpgroups, for every head dim up to it at multiples of eight.
A ``decode_`` phase is a second template for the SAME flavor and contract,
selected by the adapter when the graph is decode-shaped:
``sm100/decode_d256_f16.py`` takes the f16/bf16 d256 graphs whose S_q x packed
heads fit 16 rows (``TemplateParams.decode_q_tile``; its 32-row tile compiles but
is not routed on the Blackwell line, ``config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS``),
the prefill template the rest.  ``sm107/decode_d256_f16.py`` is that tile's Rubin
sibling -- the same body under ``config_sm107.make_cfg_d256_decode`` (the Rubin SMEM
and descriptor-window validators, the module's ``DESC_VERSION`` / ``SPIN_RING_WAITS``
constants), selected by the same record field on cc 10.7 under the Rubin route
(``config_sm107.decode_d256_q_tile``): the 32-row tile IS routed there, in up to two
TOKEN UNITS of the body (``Q_TOKEN_UNITS``: a unit covers one Q box of tokens and
streams the KV range once) for a packed MTP step -- S_q = 4 at 24/2 = two units of
two tokens x 12 heads.

Modules shared across arch lines stay at THIS level rather than inside one
arch's package, so the directory a file lives in always names its only owner:

- ``_common_blackwell.py`` — the tcgen05 pipeline helpers shared by ``sm100/``
  and ``sm107/`` (cc 100-119; ``engines._BLACKWELL`` spans the same range).
- ``thd_helpers.py`` — the THD/varlen metadata + per-batch O-descriptor setup,
  used by ``sm90/``, ``sm100/``, ``sm107/`` and ``sm120/``.

Loading
-------
Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to the cached compiler entry or the architecture's prepared host. The adapter's
``_SM*_KERNEL_FILES`` maps hold paths RELATIVE to this directory
(``"sm107/prefill_d128_fp8.py"``), which the loader joins onto it. Import a
template directly only for its all-defaults standalone path.

The SM80 templates were vendored in July 2026 from a retired internal
repository and are maintained in-tree. They now use the same TemplateParams
loader. Forward launch lowering lives in ``sm80/prepared_host.py``; the two
forward templates contain the device implementation, with no tensor compiler
or fake-operand construction. The public wrappers live in the parent SDPA API.
"""
