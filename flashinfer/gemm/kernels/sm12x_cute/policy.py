"""Shape and tactic admission for the SM12x CuTe NVFP4 runner."""

# Family and every code-generation parameter are part of the persisted tactic.
NARROW_TACTICS = (("narrow", 32, 128, 256), ("b12x", 64, 128, 256))
RAW_TACTICS = (
    ("raw", 64, 32, 8, False, True),
    ("raw", 32, 64, 13, True, True),
)
VERSION = "sm12x_cute_nvfp4_v2"


def check_shape(m, n, k):
    if min(m, n, k) <= 0 or n % 64 or k % 64:
        raise ValueError("SM12x cute-dsl requires M > 0, N % 64 = 0, K % 64 = 0")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("SM12x cute-dsl requires element offsets below 2**31")


def _sm121_tactic(m, n, k, compute_capability):
    if compute_capability == (12, 1) and (m, n, k) == (512, 8192, 2048):
        return ("raw", 64, 32, 4, False, True, 256, True)
    if compute_capability == (12, 1) and (m, n, k) == (256, 7168, 4608):
        return ("b12x", 64, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) == (256, 9216, 7168):
        return ("raw", 64, 32, 2, False, True, 256, True)
    if compute_capability == (12, 1) and (m, n, k) == (512, 7168, 5120):
        return ("raw", 64, 32, 8, False, True)
    if compute_capability == (12, 1) and (m, n, k) == (1024, 896, 1024):
        return ("raw", 32, 64, 13, True, True)
    if compute_capability == (12, 1) and (m, n, k) == (512, 5120, 640):
        return ("cooperative", 128, 128, 128)
    if compute_capability == (12, 1) and (m, n, k) == (512, 5120, 2560):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) in (
        (256, 7168, 256),
        (256, 7168, 512),
    ):
        return ("raw", 32, 64, 13, True, True, 256, False)
    if compute_capability == (12, 1) and (m, n, k) in (
        (128, 2688, 1856),
        (128, 3712, 2688),
        (128, 2688, 3712),
    ):
        return ("b12x", 128, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) in (
        (512, 1792, 5120),
        (512, 5120, 1024),
        (512, 5120, 1280),
        (512, 5120, 2048),
    ):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) in (
        (512, 1280, 8192),
        (512, 896, 5120),
    ):
        return ("cooperative", 128, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) in (
        (512, 5120, 8192),
        (512, 8192, 3584),
        (512, 3584, 5120),
    ):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) == (512, 2560, 8192):
        return ("cooperative", 128, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) == (512, 5120, 5120):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) == (1024, 512, 7168):
        return ("b12x", 128, 128, 128)
    if compute_capability == (12, 1) and (m, n, k) == (8192, 34816, 5120):
        return ("raw", 32, 64, 13, True, True, 256, False)
    if compute_capability == (12, 1) and (m, n, k) in (
        (2000, 2688, 3712),
        (2000, 3712, 2688),
        (2000, 1856, 2688),
        (2000, 2688, 1856),
    ):
        return ("cooperative", 128, 128, 128)
    if compute_capability == (12, 1) and (m, n, k) == (512, 8192, 4096):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) == (1024, 7168, 4608):
        return ("raw", 64, 32, 8, False, True, 256, True)
    if compute_capability == (12, 1) and (m, n, k) in (
        (512, 8192, 8192),
        (512, 8192, 7168),
    ):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) == (1024, 9216, 7168):
        return ("raw", 64, 32, 8, False, True, 256, True)
    if compute_capability == (12, 1) and (m, n, k) == (512, 10240, 8192):
        return ("cooperative", 128, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) == (512, 8192, 14336):
        return ("cooperative", 128, 64, 256)
    if compute_capability == (12, 1) and (m, n, k) == (512, 5120, 16384):
        return ("cooperative", 128, 128, 256)
    if compute_capability == (12, 1) and (m, n, k) == (1024, 4608, 7168):
        return ("raw", 64, 32, 8, False, True, 256, True)
    if compute_capability == (12, 1) and (m, n, k) == (512, 8192, 28672):
        return ("cooperative", 256, 128, 128)
    if compute_capability == (12, 1) and (m, n, k) == (512, 5120, 4096):
        return ("cooperative", 128, 64, 256)
    if compute_capability != (12, 1) or (n, k) not in (
        (34816, 5120),
        (5120, 17408),
    ):
        return None
    if 1 <= m <= 16 or m == 32:
        return ("narrow", 32, 128, 512)
    if m == 64:
        return ("b12x", 64, 128, 512)
    if m == 128:
        return ("b12x_single_cta", 128, 128, 256)
    if (m, n, k) in ((1024, 5120, 17408), (2048, 5120, 17408)):
        return ("raw", 64, 32, 8, False, True, 256, True)
    return _sm121_cooperative_tactic(m, n, k, compute_capability)


def _sm121_cooperative_tactic(m, n, k, compute_capability):
    if compute_capability != (12, 1):
        return None
    if (m, n, k) in (
        (256, 34816, 5120),
        (256, 5120, 17408),
        (512, 34816, 5120),
    ):
        return ("cooperative", 128, 128, 256)
    if (m, n, k) == (512, 5120, 17408):
        return ("cooperative", 128, 64, 256)
    if (m, n, k) in ((1024, 34816, 5120), (2048, 34816, 5120)):
        return ("cooperative", 256, 128, 128)
    return None


def valid_tactics(m, n, k, *, compute_capability=None):
    check_shape(m, n, k)
    if n % 128 or k % 256:
        return (_sm121_tactic(m, n, k, compute_capability) or NARROW_TACTICS[1],)
    choices = NARROW_TACTICS + (RAW_TACTICS if m % 128 == 0 else ())
    preferred = _sm121_tactic(m, n, k, compute_capability)
    if preferred is None or preferred in choices:
        return choices
    previous_default = default_tactic(m, n, k)
    return tuple(
        preferred if choice == previous_default else choice for choice in choices
    )


def compatible(m, n, k, tactic, *, compute_capability=None):
    try:
        choices = valid_tactics(m, n, k, compute_capability=compute_capability)
        legacy_choices = valid_tactics(m, n, k)
    except ValueError:
        return False
    # Replacing a profiling offer must not invalidate an existing cached tactic.
    return (
        tactic is None
        or tactic == -1
        or tactic in choices
        or tactic in legacy_choices
        or (
            tactic == ("cooperative", 128, 128, 256)
            and m == 2000
            and _sm121_tactic(m, n, k, compute_capability)
            == ("cooperative", 128, 128, 128)
        )
        or (
            compute_capability == (12, 1)
            and (m, n, k) == (512, 8192, 28672)
            and tactic == ("cooperative", 128, 128, 256)
        )
        or (
            compute_capability == (12, 1)
            and (m, n, k) == (1024, 512, 7168)
            and tactic == ("cooperative", 128, 64, 256)
        )
    )


def default_tactic(m, n, k, *, compute_capability=None):
    check_shape(m, n, k)
    if n % 128 or k % 256:
        return _sm121_tactic(m, n, k, compute_capability) or NARROW_TACTICS[1]
    preferred = _sm121_tactic(m, n, k, compute_capability)
    if preferred is not None:
        return preferred
    if m % 128 == 0:
        return RAW_TACTICS[int(n >= k)]
    return NARROW_TACTICS[int(m > 32)]
