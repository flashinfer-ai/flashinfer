# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.


def _overlaps(a, b):
    def span(x):
        return x.data_ptr(), x.data_ptr() + (
            1 + sum((n - 1) * s for n, s in zip(x.shape, x.stride(), strict=True))
        ) * x.element_size()

    al, ah = span(a)
    bl, bh = span(b)
    return al < bh and bl < ah
