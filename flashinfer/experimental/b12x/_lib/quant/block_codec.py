"""Static storage contracts for packed weight-only codecs.

These properties are consumed while preparing weights and constructing CuTe
kernels. Codec selection is never a device runtime argument.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockCodec:
    name: str
    block_weights: int
    payload_bytes: int
    base_bytes: int
    subscale_bytes: int
    grid_entries: int = 0

    @property
    def block_bytes(self) -> int:
        return self.payload_bytes + self.base_bytes + self.subscale_bytes

    @property
    def metadata_bytes(self) -> int:
        return self.base_bytes + self.subscale_bytes

    @property
    def pack_factor(self) -> int:
        return self.block_weights // self.payload_bytes

    def lut_bytes(self, *, selectors: bool = False) -> int:
        return self.grid_entries * (8 if selectors else 16)


IQ2_CODECS = ("iq2_xs", "iq2_xxs")
BLOCK_CODECS = (*IQ2_CODECS, "q8_0")
_CODECS = {
    "iq2_xs": BlockCodec("iq2_xs", 256, 64, 2, 8, 512),
    "iq2_xxs": BlockCodec("iq2_xxs", 256, 64, 2, 0, 256),
    "q8_0": BlockCodec("q8_0", 32, 32, 2, 0),
}


def block_codec(name: str) -> BlockCodec:
    try:
        return _CODECS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported packed block codec {name!r}") from exc
