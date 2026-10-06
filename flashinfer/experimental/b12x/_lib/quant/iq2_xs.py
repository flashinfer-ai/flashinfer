"""Shared IQ2 codebooks and weight-only execution tables.

IQ2_XS encodes each group of eight weights as a 16-bit descriptor.  Bits
0..8 select one of 512 magnitude vectors and bits 9..15 select one of 128
permitted sign masks. The device execution table holds 512 magnitude vectors;
the kernel derives the sign mask, including its parity bit, from the descriptor.
Checkpoint weights remain descriptor-coded.
"""

from __future__ import annotations

import functools

import torch

from .block_codec import IQ2_CODECS, block_codec


IQ2_XS_BLOCK_SIZE = 256
IQ2_XS_DESCRIPTOR_WEIGHTS = 8
IQ2_XS_NATIVE_TILE = 16
IQ2_XS_EXEC_LUT_ENTRIES = 1 << 16
IQ2_XS_MAGNITUDE_LUT_BYTES = 512 * 8 * 2
IQ2_XS_SELECTOR_LUT_BYTES = 512 * 4 * 2

# Public llama.cpp iq2xs_grid, encoded at two bits per magnitude with
# 0 -> 8, 1 -> 25, and 2 -> 43.
_IQ2_XS_GRID_2BIT = bytes.fromhex(
    "00000200050008000a0011001400160019002000220025002800410044004600"
    "49005000520055005800610064008000820085008800910094009900a0000101"
    "04010601090110011201150118011a0121012401400142014501480151015401"
    "6001680181018401900100020202050208021102140220024102440250025502"
    "80028a0201040404060409041004120415041804210424044004420445044804"
    "5104540456046004810484049004000502050505080511051405200541054405"
    "500561058005010604061006260640064206840600080208050808080a081108"
    "14082008250841084408500858088008a008aa08010904091009400981098909"
    "000a200a280a960aa00a01100410061009101010121015101810211024104010"
    "4210451048105110541060106a10811084109010001102110511081111111411"
    "2011411144115011801194119611011204120612101240126012001402140514"
    "0814111414142014411444144914501464148014011504151015401500161416"
    "49160118041810181218401854188618001905196619511aa91a002002200520"
    "08200a201120142020204120442050208020a020012104211021402148216521"
    "002222228022a82201240424102429244024002541255225992501261a26a626"
    "002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
    "0640094010401240154018402140244040404240454048404a40514054406040"
    "6540814084409040004102410541084111411441204141414441504180418541"
    "a241014204421042124229424042004402440544084411441444194420444144"
    "4444504480449444014504451045244540459a4500460a464446504601480448"
    "1048404845485448624800491149444950496949044a00500250055008501150"
    "145020502850415044505050805001510451105115514051425100524452aa52"
    "0154045410542154405460548154a154005508558055885521566856a1560058"
    "14584158505899581a5940594259855a0160046010604060546062608660a960"
    "006124624a62926200641664106540654565a46501686a682569066a546a626a"
    "00800280058008801180148020802a8041804480508080808280a880aa800181"
    "0481068110814081518159810082208280828282a082a8820184048410841284"
    "158440846084898400854485a58518866a860088088825885a8880888288a888"
    "0689228a808a888a968aa88a0190049010904090569084900091229164915692"
    "89920094059444945094589429959095929541965198a6984999159a609a00a0"
    "02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
    "95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
)

# GGML 64302f42eb959556633c89ae651915d8157dcd63, iq2xxs_grid (MIT).
_IQ2_XXS_GRID_2BIT = bytes.fromhex(
    "00000200050008000a00110014002000220028002a0041004400500058006100"
    "6400800082008a00a20001010401100115014001840198010002020222028202"
    "010404041004210424044004420448046004810484049004a404000502050805"
    "200546056905800591050906100640068406a406000805080808140828084108"
    "440850085208880804094009020a140a01100410101021104010601084109010"
    "951000110811201150115a118011241245120014081420142514491480141815"
    "6215001616160118041810184018811800190519a019511a002002200a204420"
    "6120802082202921482100220222012404241024402456240025412564259026"
    "082820289428442a014004401040184021402440404048405640604081408440"
    "9040004120416141804185410142104248425642684200440844204480449944"
    "124524450046014804481048404845480049584961498249454a904a00500850"
    "1150195020508050885004514251a4519152905492540a550156545600581158"
    "195864584059085a046010604060686000615561186260620064056410651265"
    "84654268008002800a8041808280048118814081118201840484108415844084"
    "608400854685948509864086608602880489118a0490109024904090a1901691"
    "8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)

_IQ2_XS_SIGN_MASKS = bytes.fromhex(
    "008182038405068788090a8b0c8d8e0f"
    "901112931495961718999a1b9c1d1e9f"
    "a02122a324a5a62728a9aa2bac2d2eaf"
    "30b1b233b43536b7b8393abb3cbdbe3f"
    "c04142c344c5c64748c9ca4bcc4d4ecf"
    "50d1d253d45556d7d8595adb5cddde5f"
    "60e1e263e46566e7e8696aeb6cedee6f"
    "f07172f374f5f67778f9fa7bfc7d7eff"
)


@functools.cache
def iq2_xs_grid_cpu(codec: str = "iq2_xs") -> torch.Tensor:
    """Return the codec's exact public magnitude grid as uint8."""

    if codec not in IQ2_CODECS:
        raise ValueError(f"{codec!r} has no IQ2 magnitude grid")
    spec = block_codec(codec)
    packed = torch.tensor(list(_IQ2_XS_GRID_2BIT if codec == "iq2_xs" else _IQ2_XXS_GRID_2BIT), dtype=torch.uint8)
    codes = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), 1)
    magnitudes = torch.tensor((8, 25, 43, 0), dtype=torch.uint8)
    return magnitudes[codes.long()].reshape(spec.grid_entries, 8).contiguous()


@functools.cache
def iq2_xs_execution_lut_cpu() -> torch.Tensor:
    """Return all 65,536 signed descriptor vectors as contiguous int8."""

    descriptors = torch.arange(IQ2_XS_EXEC_LUT_ENTRIES, dtype=torch.int32)
    grid = iq2_xs_grid_cpu()[(descriptors & 0x1FF).long()].to(torch.int16)
    masks = torch.tensor(list(_IQ2_XS_SIGN_MASKS), dtype=torch.int16)
    signs = 1 - 2 * ((masks[(descriptors >> 9).long(), None] >> torch.arange(8)) & 1)
    return (grid * signs).to(torch.int8).contiguous()


@functools.cache
def iq2_xs_pair_selectors_cpu(codec: str = "iq2_xs") -> torch.Tensor:
    """Return byte selectors for pairs from three scaled BF16 magnitudes."""

    grid = iq2_xs_grid_cpu(codec)
    codes = (grid == 25).to(torch.int32) + 2 * (grid == 43).to(torch.int32)
    lo, hi = 2 * codes[:, 0::2], 2 * codes[:, 1::2]
    return (lo | ((lo + 1) << 4) | (hi << 8) | ((hi + 1) << 12)).to(torch.int16).contiguous()


_DEVICE_TABLES: dict[tuple[str, int | None, bool, str], torch.Tensor] = {}


def _iq2_xs_execution_lut_device(
    device_type: str, device_index: int | None, selectors: bool, codec: str
) -> torch.Tensor:
    table = (
        iq2_xs_pair_selectors_cpu(codec)
        if selectors
        else iq2_xs_grid_cpu(codec).to(dtype=torch.bfloat16)
    )
    return table.to(device=torch.device(device_type, device_index)).contiguous()


def iq2_xs_execution_lut(
    device: torch.device | str, *, prepare: bool = False, selectors: bool = False,
    codec: str = "iq2_xs"
) -> torch.Tensor:
    """Return the codec-specific magnitude or byte-selector table."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    block_codec(codec)
    key = (resolved.type, index, bool(selectors), codec)
    if key not in _DEVICE_TABLES:
        if not prepare:
            raise RuntimeError(
                "IQ2_XS execution table must be prepared before binding or replay"
            )
        _DEVICE_TABLES[key] = _iq2_xs_execution_lut_device(*key)
    return _DEVICE_TABLES[key]


__all__ = [
    "IQ2_XS_BLOCK_SIZE",
    "IQ2_XS_DESCRIPTOR_WEIGHTS",
    "IQ2_XS_EXEC_LUT_ENTRIES",
    "IQ2_XS_NATIVE_TILE",
    "iq2_xs_execution_lut",
    "iq2_xs_execution_lut_cpu",
    "iq2_xs_grid_cpu",
]
