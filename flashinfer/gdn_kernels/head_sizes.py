"""Which (head_size, head_size_v) pairs each GDN path implements.

One check, one exception type, so every entry point rejects an unsupported
pair the same way: ``NotImplementedError``, which survives ``python -O``.
"""

# SM90 and SM100 chunked prefill; head_size_v may be narrower than head_size.
CHUNKED_PREFILL_HEAD_SIZES = frozenset({(128, 128), (128, 64), (64, 64)})
# SM120 prefill derives one head dim from q.
SM120_PREFILL_HEAD_SIZES = frozenset({(128, 128)})
# MTP decode; head_size_v > head_size works too (checked on SM90).
MTP_DECODE_HEAD_SIZES = frozenset({(128, 128), (128, 64), (64, 64), (64, 128)})


def check_head_sizes(
    what: str, head_size: int, head_size_v: int, supported: frozenset
) -> None:
    """Raise ``NotImplementedError`` unless ``(head_size, head_size_v)`` is supported."""
    if (head_size, head_size_v) not in supported:
        raise NotImplementedError(
            f"{what} supports (head_size, head_size_v) in "
            f"{sorted(supported, reverse=True)}, got ({head_size}, {head_size_v})"
        )
