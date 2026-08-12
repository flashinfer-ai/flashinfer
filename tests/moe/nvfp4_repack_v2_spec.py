from __future__ import annotations

import numpy as np


LAYOUT_VERSION = 2
TILE_N = 64
TILE_K = 16
THREADS = 128
BYTES_PER_THREAD = 4


def fragment_element_coordinate(
    thread: int, register: int, element: int
) -> tuple[int, int]:
    if not 0 <= thread < THREADS:
        raise ValueError("thread must be in [0,128)")
    if not 0 <= register < BYTES_PER_THREAD:
        raise ValueError("register must be in [0,4)")
    if element not in (0, 1):
        raise ValueError("element must be 0 or 1")
    warp, lane = divmod(thread, 32)
    lane_row, lane_col = divmod(lane, 4)
    row = 16 * warp + lane_row + 8 * (register % 2)
    k = 2 * lane_col + 8 * (register // 2) + element
    return row, k


def fragment_byte_coordinate(thread: int, register: int) -> tuple[int, int]:
    row, k = fragment_element_coordinate(thread, register, 0)
    return row, k // 2


def thread_scale_rows(thread: int) -> tuple[int, int]:
    if not 0 <= thread < THREADS:
        raise ValueError("thread must be in [0,128)")
    warp, lane = divmod(thread, 32)
    row = 16 * warp + lane // 4
    return row, row + 8


def repack_payload(payload: np.ndarray) -> np.ndarray:
    if payload.dtype != np.uint8 or payload.ndim != 3:
        raise ValueError("payload must be uint8 [E,N,K/2]")
    experts, rows, packed_k = payload.shape
    logical_k = packed_k * 2
    if rows % TILE_N or logical_k % TILE_K:
        raise ValueError("N and K violate tile alignment")
    n_tiles = rows // TILE_N
    k_tiles = logical_k // TILE_K
    result = np.empty(
        (experts, n_tiles, k_tiles, THREADS, BYTES_PER_THREAD),
        dtype=np.uint8,
    )
    for n_tile in range(n_tiles):
        for k_tile in range(k_tiles):
            for thread in range(THREADS):
                for register in range(BYTES_PER_THREAD):
                    row, k_byte = fragment_byte_coordinate(thread, register)
                    result[:, n_tile, k_tile, thread, register] = payload[
                        :, n_tile * TILE_N + row, k_tile * (TILE_K // 2) + k_byte
                    ]
    return result


def unpack_payload(payload: np.ndarray) -> np.ndarray:
    if payload.dtype != np.uint8 or payload.ndim != 5:
        raise ValueError("payload must be uint8 [E,Nt,Kt,128,4]")
    if payload.shape[-2:] != (THREADS, BYTES_PER_THREAD):
        raise ValueError("fragment shape is invalid")
    experts, n_tiles, k_tiles = payload.shape[:3]
    result = np.empty(
        (experts, n_tiles * TILE_N, k_tiles * TILE_K // 2),
        dtype=np.uint8,
    )
    for n_tile in range(n_tiles):
        for k_tile in range(k_tiles):
            for thread in range(THREADS):
                for register in range(BYTES_PER_THREAD):
                    row, k_byte = fragment_byte_coordinate(thread, register)
                    result[
                        :, n_tile * TILE_N + row, k_tile * (TILE_K // 2) + k_byte
                    ] = payload[:, n_tile, k_tile, thread, register]
    return result


def repack_scales(scales: np.ndarray) -> np.ndarray:
    if scales.dtype != np.uint8 or scales.ndim != 3:
        raise ValueError("scales must contain raw E4M3 bytes")
    experts, rows, k_tiles = scales.shape
    if rows % TILE_N:
        raise ValueError("scale rows must be divisible by 64")
    return (
        scales.reshape(experts, rows // TILE_N, TILE_N, k_tiles)
        .transpose(0, 1, 3, 2)
        .copy()
    )


def unpack_scales(scales: np.ndarray) -> np.ndarray:
    if scales.dtype != np.uint8 or scales.ndim != 4:
        raise ValueError("scales must be uint8 [E,Nt,Kt,64]")
    if scales.shape[-1] != TILE_N:
        raise ValueError("scale tile width must be 64")
    experts, n_tiles, k_tiles, _ = scales.shape
    return (
        scales.transpose(0, 1, 3, 2).copy().reshape(experts, n_tiles * TILE_N, k_tiles)
    )
