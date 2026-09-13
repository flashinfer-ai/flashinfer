"""Shape and tactic admission for the SM12x CuTe NVFP4 runner."""

# Family and every code-generation parameter are part of the persisted tactic.
NARROW_TACTICS = (("narrow", 32, 128, 256), ("b12x", 64, 128, 256))
RAW_TACTICS = (
    ("raw", 64, 32, 8, False, True),
    ("raw", 32, 64, 13, True, True),
)
VERSION = "sm12x_cute_nvfp4_v1"


def check_shape(m, n, k):
    if min(m, n, k) <= 0 or n % 128 or k % 256:
        raise ValueError("SM12x cute-dsl requires M > 0, N % 128 = 0, K % 256 = 0")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("SM12x cute-dsl requires element offsets below 2**31")


def valid_tactics(m, n, k):
    check_shape(m, n, k)
    return NARROW_TACTICS + (RAW_TACTICS if m % 128 == 0 else ())


def compatible(m, n, k, tactic):
    try:
        choices = valid_tactics(m, n, k)
    except ValueError:
        return False
    return tactic is None or tactic == -1 or tactic in choices


def default_tactic(m, n, k):
    check_shape(m, n, k)
    if m % 128 == 0:
        return RAW_TACTICS[int(n >= k)]
    return NARROW_TACTICS[int(m > 32)]
