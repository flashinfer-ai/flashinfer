# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/tuning/decode.max_active_clusters.py @ 3f7ff225 (2026-06-30) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Generated MoE decode MAX_ACTIVE_CLUSTERS tuning data."""

from .registry import register_max_active_clusters_policy

# micro:   routed_rows < 64 (direct-micro selection cutover)
# dynamic: otherwise

register_max_active_clusters_policy(
    regime="decode",
    backend="micro",
    ladder=(
        (2, 84),
        (4, 127),
        (8, 107),
        (10, 84),
        (16, 63),
        (20, 84),
    ),
)

register_max_active_clusters_policy(
    regime="decode",
    backend="dynamic",
    ladder=(
        (640, 188),
        (1024, 147),
    ),
)

# Prepared MXFP4 decode has two resident CTAs per SM.  A logical cap of 64 is
# doubled by the launcher to a 128-CTA worker grid.  M=1 sizes that grid for the
# wider of its fixed FC1 and FC2 domains; M=2 exactly fills it, and M=4/8 use a
# short grid-stride tail without making every barrier span the full machine.
register_max_active_clusters_policy(
    regime="decode",
    backend="dynamic_w4a8_decode",
    ladder=((64, 64),),
)
