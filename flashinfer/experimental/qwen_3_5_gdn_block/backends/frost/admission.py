# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Admission of the FROST backend: which device, geometry and step shapes the FROST GDN kernels serve.

Imports torch and FlashInfer's CuTe DSL presence probe only -- never the kernels -- so the support probe stays cheap.
The step-shape rules mirror the kernel wrappers in ``gdn.py`` / ``gdn_verify.py`` (cited per rule); ``runner.py``
re-checks them against the wrappers at plan time.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...support import Qwen35GdnDecline, Qwen35GdnGeometry, device_cc

SUPPORTED_CC = (10, 0)
#: the one geometry the kernels are written for (Qwen3.8-27B at TP 1); every index in them is a literal of it
GEOMETRY = {
    "d_model": 5120,
    "num_v_heads": 48,
    "num_k_heads": 16,
    "head_k_dim": 128,
    "head_v_dim": 128,
    "conv_kernel": 4,
}
#: gdn_verify.THREADS
THREADS = 256
#: FrostGdnVerify.fuses_conv admits more requests only while 48 x CTAs x requests fits one wave
VERIFY_MAX_REQUESTS_ANY_DEVICE = 4


def verify_ctas(requests: int) -> int:
    """CTAs per V head of the verify launch (FrostGdnVerify.ctas)."""
    return 2 if requests == 1 else 1


def verify_max_tokens(requests: int) -> int:
    """The fused conv's per-thread unit budget (FrostGdnVerify.fuses_conv): ``T * (32 + 128 // ctas // 4) <= 2 x THREADS``
    -- 10 tokens per request at one request, 8 from two requests on."""
    return 2 * THREADS // (32 + 128 // verify_ctas(requests) // 4)


def _sm_count(device) -> int:
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def support(
    device,
    geometry: Qwen35GdnGeometry,
    form: str,
    requests: Optional[int],
    tokens_per_request: Optional[int],
) -> Optional[Qwen35GdnDecline]:
    """The FROST backend's support function (the registry's ``support_fn``)."""
    off = {
        k: getattr(geometry, k)
        for k, v in GEOMETRY.items()
        if getattr(geometry, k) != v
    }
    if off:
        return Qwen35GdnDecline(
            "geometry",
            "the FROST GDN kernels are written for d_model 5120, 48 V heads, 16 K heads, head dims 128 and conv "
            f"kernel 4 (Qwen3.8-27B at TP 1); got {off}",
        )
    cc = device_cc(device)
    if cc != SUPPORTED_CC:
        found = "no CUDA device" if cc is None else f"SM{cc[0]}{cc[1]}"
        return Qwen35GdnDecline(
            "device",
            f"the FROST GDN kernels target SM{SUPPORTED_CC[0]}{SUPPORTED_CC[1]} (cc 10.0) only; found {found}",
        )
    from flashinfer.cute_dsl.availability import is_cute_dsl_available

    if not is_cute_dsl_available():
        return Qwen35GdnDecline(
            "dsl",
            "nvidia-cutlass-dsl is not installed; the FROST GDN kernels are CuTe DSL",
        )
    if form == "verify":
        if tokens_per_request is not None and tokens_per_request < 2:
            return Qwen35GdnDecline(
                "shape",
                "the verify form carries at least one draft token per request (tokens_per_request >= 2); one token "
                "per request is the decode form",
            )
        if (
            requests is None
            and tokens_per_request is not None
            and tokens_per_request > verify_max_tokens(1)
        ):
            return Qwen35GdnDecline(
                "shape",
                f"the FROST verify's fused conv takes at most {verify_max_tokens(1)} tokens per request; got "
                f"tokens_per_request={tokens_per_request}",
            )
        if requests is not None:
            ctas = verify_ctas(requests)
            if not (
                requests <= VERIFY_MAX_REQUESTS_ANY_DEVICE
                or 48 * ctas * requests <= _sm_count(device)
            ):
                return Qwen35GdnDecline(
                    "shape",
                    f"the FROST verify fuses the speculative conv for at most {VERIFY_MAX_REQUESTS_ANY_DEVICE} "
                    f"requests (or a grid within one wave); got requests={requests}",
                )
            if (
                tokens_per_request is not None
                and tokens_per_request > verify_max_tokens(requests)
            ):
                return Qwen35GdnDecline(
                    "shape",
                    f"the FROST verify's fused conv takes at most {verify_max_tokens(requests)} tokens per request "
                    f"at {requests} request(s); got tokens_per_request={tokens_per_request}",
                )
    return None
