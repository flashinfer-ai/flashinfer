# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MXFP8-activation x MXFP4-weight NVLinkOneSided MegaMoE: Sentinel (data-as-flag) and Fence protocols.

The engine (``engine/``), its host frontend (``frontend/``) and the protocols'
PTX primitives (``comm/``); ``VENDOR.md`` records where the engine started.
Imports are lazy: importing this package does not import the CuTe DSL.
"""
