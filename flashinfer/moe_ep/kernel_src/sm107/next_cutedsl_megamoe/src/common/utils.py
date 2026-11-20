"""Integer arithmetic shared by host and DSL callers."""


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def round_up(value, alignment):
    return ceil_div(value, alignment) * alignment
