# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The SM 10.x (tcgen05) tree of the FROST GEMM engine: ``compiler``,
``epilogue_codegen`` and the ``kernel_templates/`` of the sm100 and sm103
pipeline families. Reached through the ``cudnn.gemm.frost.compiler`` /
``epilogue_codegen`` facades when this is the active family (see
``cudnn.gemm.frost.arch_family``); import the modules here by name to pin the
tree. Nothing is imported eagerly -- the compiler pulls in the CuTe DSL, and
``import cudnn`` has to stay cheap."""
