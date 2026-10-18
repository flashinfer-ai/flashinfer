"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated Kimi-K3 AttnRes programs.
# ``ARG_PLANS``: the three argument orders the generated host bindings take, as
# ``[kind, name]`` pairs (``buffer`` tensors, ``parameter`` scalars, the
# ``launch`` switch and the ``grid`` dims; ``cake_backend._bind`` orders a call
# by name).  ``_PROGRAMS``: one row per generated program, keyed by the 20-hex
# digest of its module name ``cake_kimi_k3_attn_res_<digest>``:
# ``(closure, arches, arg_plan, source, compile_flags)`` -- the first 20 hex
# digits of the exporter's closure SHA-256 (the JIT library is named
# ``<module name>_<closure>``), the architectures the one source is compiled
# for, the ``ARG_PLANS`` key, the digest of the program whose two translation
# units the record builds (``None``: its own ``<digest>_kernel.cu`` /
# ``<digest>_binding.cu`` under ``csrc/cake_kimi_k3_attn_res``) and the extra
# nvcc flags (``--use_fast_math``; a record built from another program's units
# also passes ``-DNUM_BLOCKS`` where the two programs differ in it, and the
# ``-DCAKE_PROGRAM_KERNEL`` / ``-DCAKE_PROGRAM_SHIM`` pair that selects its own
# kernel symbol and host-shim namespace).  ``MODULES`` expands the rows into
# one record per module name (``sources``, ``compile_flags``, ``arg_plan``,
# ``arches``, ``closure``).  ``KERNELS``: per architecture, the logical kernel
# key the host planner resolves (``cake_backend.plan_route(...).kernel_key``)
# -> module name.  The rows, ``ARG_PLANS`` and ``KERNELS`` are written by the
# generated-program export; do not edit by hand.  The shared sources and the
# tabular form are a hand fold of the exported literals pending their
# regeneration by the export.
ARG_PLANS: dict[str, list[list[str]]] = {
    "common": [
        ["buffer", "out"],
        ["buffer", "prefix"],
        ["buffer", "delta"],
        ["buffer", "blocks"],
        ["buffer", "norm_weight"],
        ["buffer", "qk_weight"],
        ["buffer", "output_norm_weight"],
        ["parameter", "blocks_m_stride"],
        ["parameter", "blocks_k_stride"],
        ["parameter", "eps"],
        ["parameter", "output_norm_eps"],
        ["parameter", "M"],
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "direct": [
        ["buffer", "out"],
        ["buffer", "prefix"],
        ["buffer", "delta"],
        ["buffer", "blocks"],
        ["buffer", "norm_weight"],
        ["buffer", "qk_weight"],
        ["buffer", "output_norm_weight"],
        ["parameter", "out_stride"],
        ["parameter", "prefix_stride"],
        ["parameter", "delta_stride"],
        ["parameter", "blocks_m_stride"],
        ["parameter", "blocks_k_stride"],
        ["parameter", "eps"],
        ["parameter", "output_norm_eps"],
        ["parameter", "block_write_idx"],
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "k0_tma": [
        ["buffer", "out"],
        ["buffer", "prefix"],
        ["buffer", "delta"],
        ["buffer", "output_norm_weight"],
        ["parameter", "output_norm_eps"],
        ["parameter", "M"],
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
# Row vocabulary: the architecture lists a program is compiled for and the
# fast-math build (the ``_fm`` keys of ``KERNELS``).
_SM100 = ("sm_100a",)
_SM103 = ("sm_103a",)
_BOTH = ("sm_100a", "sm_103a")
_FM = ("--use_fast_math",)
_PROGRAMS: dict[str, tuple[Any, ...]] = {
    "0d77890adbb0f7175149": ("5a43e184adb76b0da3ca", _SM100, "common", None, ()),
    "110f5de77158b1bcfe57": ("55c85867c640ccad6e30", _SM100, "common", None, ()),
    "12101c0c5b4336b8c00e": ("771d716944531c2c796b", _BOTH, "common", None, _FM),
    "12cd385f9c0656783118": ("726e3af7a113abf343e9", _SM103, "common", None, _FM),
    "13cf584a1ad91f2c2137": ("eaef3bea6c923f006928", _SM103, "common", None, _FM),
    "14855b6501f28a8af20c": ("1fb886d3d81ab6b6d818", _BOTH, "common", None, _FM),
    "1974170f2264826fd01b": ("e7c21196db2feeeac094", _BOTH, "common", None, _FM),
    "207e4fd6704c831c149b": ("ee48852a28a52f415bf9", _SM100, "common", None, _FM),
    "27ecb918713280215e50": ("bce9905f406699b1f413", _BOTH, "common", None, ()),
    "291d118f0d9137e82482": ("4905c46adae2c2383ee2", _BOTH, "direct", None, ()),
    "2d95afccc67d7e5696b6": ("07e0f4ceef86dcb21e16", _SM100, "common", None, _FM),
    "3055514f3a17ac73dc8d": (
        "0d5cc6943607b73bdf77",
        _SM103,
        "common",
        "14855b6501f28a8af20c",
        (
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_3055514f3a17ac73dc8d",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2560d48b6b112705",
        ),
    ),
    "3249f042c00c3ee119ef": ("8d02f022e60c37171290", _BOTH, "common", None, _FM),
    "34865263e0a3756e1d33": (
        "42e289c18c94ee05d78f",
        _BOTH,
        "common",
        "3249f042c00c3ee119ef",
        (
            "--use_fast_math",
            "-DNUM_BLOCKS=3",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_34865263e0a3756e1d33",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_6436058c860e6aeb",
        ),
    ),
    "34ad0cffc89ae1e7aa29": ("1d7e3db4486cd212a821", _SM103, "common", None, ()),
    "3912cd6ce83f35c2504d": ("7bbdcd8ffcb9b32293fb", _BOTH, "common", None, ()),
    "3ccd8bb156fa4071f08c": ("4089658a78bbb30654bb", _BOTH, "common", None, ()),
    "4857e02681f352274a12": ("480df253b6e1d37ccf67", _BOTH, "common", None, _FM),
    "4a9681983424fd0a1293": ("d9ef0b36bf8091e58b00", _SM100, "common", None, _FM),
    "4f0cabee0d8f305aace2": ("eee667bf8ab7f21f1587", _BOTH, "common", None, ()),
    "532cf65fac66f79cbbd7": ("e4235b2d948de087c8da", _BOTH, "common", None, _FM),
    "53790424af406f2edf30": ("dc22be013a47d0c73078", _SM103, "common", None, ()),
    "55b1ddf9602fd45a1404": ("07a4b7072b823dac0162", _SM103, "common", None, ()),
    "58c36fe827b800606c5e": (
        "cb39925ada01811a9451",
        _BOTH,
        "common",
        "3912cd6ce83f35c2504d",
        (
            "-DNUM_BLOCKS=6",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_58c36fe827b800606c5e",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2178fba40d078f7d",
        ),
    ),
    "5e7c661caa058a3ed916": ("6547e15f3c139128b4da", _BOTH, "common", None, _FM),
    "63d5baa9d0eb595ac0ac": ("0b528d3a7bf3e950b7a5", _BOTH, "common", None, _FM),
    "6468e67b15dab2e973eb": (
        "405658221cf96e19cda7",
        _BOTH,
        "common",
        "3912cd6ce83f35c2504d",
        (
            "--use_fast_math",
            "-DNUM_BLOCKS=7",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_6468e67b15dab2e973eb",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_5abf574f56fd3b70",
        ),
    ),
    "682069efbbc5e016c716": ("55f187b4523bab143538", _SM103, "common", None, _FM),
    "709f669fe8647d61aaf0": ("9a6597bb79598a6f75ce", _BOTH, "common", None, _FM),
    "73915c216dbaf8da3bbd": ("030138726958b4decf24", _SM100, "common", None, _FM),
    "74f8cbf06549fb15441f": ("92082c20bcb313db01be", _BOTH, "common", None, ()),
    "767d23dd29a86915ae01": (
        "3fc9ed0beb29b07161e8",
        _SM100,
        "common",
        "110f5de77158b1bcfe57",
        (
            "--use_fast_math",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_767d23dd29a86915ae01",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2b549692c339cfd3",
        ),
    ),
    "788cbcf6993668e37d68": ("15be229f6da6af11cf12", _BOTH, "common", None, _FM),
    "7c8168585b950bb93dc9": ("438a0c5b001e566a7082", _BOTH, "common", None, ()),
    "7d4a21fb3d795a6640b3": ("ed52f6b5f8879d5080d2", _BOTH, "common", None, _FM),
    "7d9facbe77d92ebf65f1": ("5642c41f50e71b03fc78", _BOTH, "common", None, _FM),
    "80715310f7aa0cb02593": ("054763d13ceabe598f35", _BOTH, "k0_tma", None, ()),
    "8149765ce4eb797a1263": ("4510c2d800a93d4e93a2", _BOTH, "common", None, _FM),
    "87822e6dd352c14dcd2b": ("26a8dbcc658e47284f1e", _BOTH, "common", None, _FM),
    "8a0a9252345c6b4769de": ("3131ec465f2abd9d885e", _BOTH, "common", None, _FM),
    "8e61459b47e062ea2e03": ("378c8eb619be67b1de32", _SM103, "common", None, _FM),
    "923c80c79dc7e5f529c3": ("00739f3681fd1a6e2e7e", _SM103, "common", None, ()),
    "935cbafc006862bdf628": (
        "47f3b906a254a71629e3",
        _SM103,
        "common",
        "34ad0cffc89ae1e7aa29",
        (
            "-DNUM_BLOCKS=3",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_935cbafc006862bdf628",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2260b3dcc0f6db4a",
        ),
    ),
    "959261ede770f517ff44": ("e5f95070bbc86fc69df6", _SM103, "common", None, ()),
    "970f5ee5b6e600f64d47": ("f3a753e9e533b8cb82f4", _BOTH, "common", None, ()),
    "99cdfcf0127721240020": ("37fb3aa8045a1c4e3e7c", _SM100, "common", None, ()),
    "99f148a06e530e457f4a": ("5f6452a41c1ad3cc1892", _BOTH, "common", None, _FM),
    "9bbfc3ccb47bb2370fbc": ("d3ccf2594819db9aecc1", _BOTH, "common", None, _FM),
    "9f31bc7a05f97159bb96": ("8f174c587ad843d169da", _SM100, "common", None, _FM),
    "9f53a8193923490ea432": ("dbe1351a53f39151b5b5", _BOTH, "common", None, _FM),
    "9f673b47d6e8c220ecdd": ("a436c095bc34d7ee0c28", _SM103, "common", None, ()),
    "a15c9b10e0c06764bb50": ("54924432be39a8a24c02", _SM103, "common", None, ()),
    "a2129211efec0ffa911f": ("79caa5f873af5b489734", _SM100, "common", None, ()),
    "a3e722cd107123bd4726": (
        "1d909a7e9e9abac962d2",
        _SM103,
        "common",
        "13cf584a1ad91f2c2137",
        (
            "--use_fast_math",
            "-DNUM_BLOCKS=4",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_a3e722cd107123bd4726",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_7a0a75df3725e291",
        ),
    ),
    "ab1c5516198a5b0fed97": (
        "aa816ae9189d26348a8a",
        _BOTH,
        "common",
        "87822e6dd352c14dcd2b",
        (
            "--use_fast_math",
            "-DNUM_BLOCKS=6",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_ab1c5516198a5b0fed97",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e6a156180d908faa",
        ),
    ),
    "ab92afb21c782d62a0ad": ("0942c1fa36cfa1a315b1", _BOTH, "common", None, _FM),
    "b0f25d64586277608a9b": (
        "7267b49decbd8d55c685",
        _SM100,
        "common",
        "959261ede770f517ff44",
        (
            "--use_fast_math",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_b0f25d64586277608a9b",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_773d9a3b48b36231",
        ),
    ),
    "b41d37766b2613218b29": ("fab0b418287650934e4c", _BOTH, "common", None, _FM),
    "b4c932378bcb431ee27a": ("1578e9b0ad901caf08e9", _SM100, "common", None, _FM),
    "b545bf557abbe26776c0": ("621c3eb83203abad23e3", _SM103, "common", None, _FM),
    "b71ef632ff11c3a6abca": ("50d44d3920cc30d28db0", _SM100, "common", None, _FM),
    "b885dd8a2d8b94643f50": ("c6293f2b85047d388446", _BOTH, "common", None, _FM),
    "ba11780fad8db3117e12": ("35680253cc3986457800", _SM103, "common", None, ()),
    "bc41069b56c74214c587": ("0ab18941551dbcbe21b4", _BOTH, "common", None, ()),
    "be7bfd8261eeb4e6d8b5": ("c1ae37b32fc11c26fbb7", _BOTH, "common", None, ()),
    "c0ab67a53d992f2fdf17": ("cda00a5d1aacb4f1888a", _BOTH, "common", None, _FM),
    "c15281d36724f27d10e3": ("575510279d4ecbdaffa9", _SM100, "common", None, _FM),
    "c33474069e53fb4759dc": (
        "0998b2c98d4a8cb879d2",
        _BOTH,
        "common",
        "3912cd6ce83f35c2504d",
        (
            "-DNUM_BLOCKS=5",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_c33474069e53fb4759dc",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4a0f5c6e4b41867c",
        ),
    ),
    "c4a61daa7501667c9886": ("c1f70f055019a218cdb5", _SM103, "common", None, _FM),
    "caf1e4d34ba9c3debadd": ("def96affacaf4eabc37e", _BOTH, "common", None, _FM),
    "ce52de03a90dc1af2f74": (
        "33de681b622e8890baf2",
        _SM103,
        "common",
        "9f673b47d6e8c220ecdd",
        (
            "--use_fast_math",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_ce52de03a90dc1af2f74",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_7e9d778cfcea4671",
        ),
    ),
    "d0dc3859ec7be0407f60": (
        "2c5f0bcd18c348154589",
        _SM103,
        "common",
        "4f0cabee0d8f305aace2",
        (
            "--use_fast_math",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_d0dc3859ec7be0407f60",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_57b11a8eb6be1e9a",
        ),
    ),
    "d20108254c08ce643df7": ("a1bce3edea5f183f9925", _BOTH, "common", None, _FM),
    "d510b7826e8e4cf50bed": (
        "6ce0b8c727bd5f62ff87",
        _BOTH,
        "common",
        "3912cd6ce83f35c2504d",
        (
            "-DNUM_BLOCKS=7",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_d510b7826e8e4cf50bed",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3d5f0b1e8a40b2cb",
        ),
    ),
    "d534d40d8b49f19bd050": (
        "201af56ac635b55f3b66",
        _SM100,
        "common",
        "99cdfcf0127721240020",
        (
            "--use_fast_math",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_d534d40d8b49f19bd050",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_095e112c65779b13",
        ),
    ),
    "d57d39706cbed5379055": (
        "dfce39117e7844bba039",
        _SM100,
        "common",
        "8e61459b47e062ea2e03",
        (
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_d57d39706cbed5379055",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3499ea8cbe2be749",
        ),
    ),
    "d6c48b9566d909d3e0b8": ("52ffd760c25ca7f0729b", _BOTH, "direct", None, ()),
    "d8a00f7ef7bdcde4db62": ("2cbe2ee2721e861f6ebc", _BOTH, "common", None, _FM),
    "d959656094922bdab26f": ("17e2df7b8de748a33b71", _SM100, "common", None, _FM),
    "d99fda249fb6c79c8622": ("53f4903d0ca8da40d539", _SM103, "common", None, ()),
    "db8d149a6c5d0a8317fd": ("499986be30b2e52dc0b4", _BOTH, "common", None, _FM),
    "dbd88df98cdb78e41a4c": ("74a9f9ba0e740de4933d", _SM100, "common", None, ()),
    "dd089fe2c1492e3bb21d": ("426ac6ac6971c05c8dd5", _BOTH, "common", None, _FM),
    "decfe1cc69007dcc218a": ("a7ed53e777d7d49aa616", _BOTH, "common", None, _FM),
    "e682c53fdd5475bf33f5": ("df3112ebf02b86000d21", _SM103, "common", None, ()),
    "e95cf8d5c1116af31c5d": ("898befbe7085418eb8e7", _SM103, "common", None, ()),
    "ece2ac302cbe5d39ff8c": (
        "cc87d605bf0433f89fb8",
        _BOTH,
        "common",
        "1974170f2264826fd01b",
        (
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_attn_res_ece2ac302cbe5d39ff8c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_53941faf776b1286",
        ),
    ),
    "f4adf17b2ed959069d30": ("b98e55943fcdf175ddfb", _BOTH, "common", None, _FM),
    "fb0a96ae8e0c2345c090": ("bbf30fbf772d83940937", _BOTH, "common", None, _FM),
    "fb6f7d32a38cf4bbc808": ("d957de68925f80f801f3", _BOTH, "common", None, _FM),
    "fc09e868f74b8c6fdf0e": ("b30201f75737b0b17a90", _BOTH, "direct", None, ()),
}


def _module(digest: str, row: tuple[Any, ...]) -> dict[str, Any]:
    """The ``MODULES`` record of program ``digest`` from its ``_PROGRAMS`` row."""
    closure, arches, arg_plan, source, compile_flags = row
    units = source or digest
    return {
        "sources": [
            f"cake_kimi_k3_attn_res/cake_kimi_k3_attn_res_{units}_kernel.cu",
            f"cake_kimi_k3_attn_res/cake_kimi_k3_attn_res_{units}_binding.cu",
        ],
        "compile_flags": list(compile_flags),
        "arg_plan": arg_plan,
        "arches": list(arches),
        "closure": closure,
    }


MODULES: dict[str, dict[str, Any]] = {
    f"cake_kimi_k3_attn_res_{digest}": _module(digest, row)
    for digest, row in _PROGRAMS.items()
}
KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "direct:k0_d0_w1_n1": "cake_kimi_k3_attn_res_fc09e868f74b8c6fdf0e",
        "direct:k4_d0_w0_n1": "cake_kimi_k3_attn_res_291d118f0d9137e82482",
        "direct:k8_d1_w0_n0": "cake_kimi_k3_attn_res_d6c48b9566d909d3e0b8",
        "k0_tma": "cake_kimi_k3_attn_res_80715310f7aa0cb02593",
        "native_k8": "cake_kimi_k3_attn_res_99f148a06e530e457f4a",
        "native_m128": "cake_kimi_k3_attn_res_709f669fe8647d61aaf0",
        "persistent:k1_nc2_d2_f110000000000000": "cake_kimi_k3_attn_res_99cdfcf0127721240020",
        "persistent:k1_nc2_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_d534d40d8b49f19bd050",
        "persistent:k1_nc2_d2_f110000000000000_write": "cake_kimi_k3_attn_res_dbd88df98cdb78e41a4c",
        "persistent:k1_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_ece2ac302cbe5d39ff8c",
        "persistent:k1_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_1974170f2264826fd01b",
        "persistent:k1_nc4_d2_f110000000000000_write": "cake_kimi_k3_attn_res_7c8168585b950bb93dc9",
        "persistent:k2_nc2_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_9f31bc7a05f97159bb96",
        "persistent:k2_nc3_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_d8a00f7ef7bdcde4db62",
        "persistent:k2_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_3249f042c00c3ee119ef",
        "persistent:k2_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_3912cd6ce83f35c2504d",
        "persistent:k3_nc2_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_fb6f7d32a38cf4bbc808",
        "persistent:k3_nc3_d3_f110000000000000": "cake_kimi_k3_attn_res_0d77890adbb0f7175149",
        "persistent:k3_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_34865263e0a3756e1d33",
        "persistent:k3_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_3ccd8bb156fa4071f08c",
        "persistent:k3_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_ab92afb21c782d62a0ad",
        "persistent:k4_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_63d5baa9d0eb595ac0ac",
        "persistent:k4_nc3_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_db8d149a6c5d0a8317fd",
        "persistent:k4_nc3_d2_f110000000000000_fm_write": "cake_kimi_k3_attn_res_b885dd8a2d8b94643f50",
        "persistent:k4_nc3_d3_f011000000000000_write": "cake_kimi_k3_attn_res_970f5ee5b6e600f64d47",
        "persistent:k4_nc3_d3_f110000000000000": "cake_kimi_k3_attn_res_27ecb918713280215e50",
        "persistent:k4_nc3_d3_f110000000000000_fm": "cake_kimi_k3_attn_res_caf1e4d34ba9c3debadd",
        "persistent:k5_nc3_d3_f010000000000000": "cake_kimi_k3_attn_res_a2129211efec0ffa911f",
        "persistent:k5_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_8149765ce4eb797a1263",
        "persistent:k5_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_c33474069e53fb4759dc",
        "persistent:k5_nc4_d3_f110000000000000": "cake_kimi_k3_attn_res_be7bfd8261eeb4e6d8b5",
        "persistent:k6_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_ab1c5516198a5b0fed97",
        "persistent:k6_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_58c36fe827b800606c5e",
        "persistent:k6_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_5e7c661caa058a3ed916",
        "persistent:k7_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_87822e6dd352c14dcd2b",
        "persistent:k7_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_d510b7826e8e4cf50bed",
        "persistent:k7_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_6468e67b15dab2e973eb",
        "persistent:k7_nc4_d2_f110000000000000_write": "cake_kimi_k3_attn_res_bc41069b56c74214c587",
        "persistent:k8_nc3_d2_f011100000000000_fm": "cake_kimi_k3_attn_res_7d9facbe77d92ebf65f1",
        "persistent:k8_nc3_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_2d95afccc67d7e5696b6",
        "persistent:k8_nc5_d2_f110000000000000": "cake_kimi_k3_attn_res_74f8cbf06549fb15441f",
        "small_m_cluster2:k4_fm": "cake_kimi_k3_attn_res_dd089fe2c1492e3bb21d",
        "small_m_cluster2:k4_fm_write": "cake_kimi_k3_attn_res_c0ab67a53d992f2fdf17",
        "small_m_cluster2:k5_fm": "cake_kimi_k3_attn_res_b0f25d64586277608a9b",
        "small_m_cluster2:k6_fm": "cake_kimi_k3_attn_res_9bbfc3ccb47bb2370fbc",
        "small_m_cluster2:k6_nc4_fm": "cake_kimi_k3_attn_res_7d4a21fb3d795a6640b3",
        "small_m_cluster2:k7_fm": "cake_kimi_k3_attn_res_8a0a9252345c6b4769de",
        "small_m_cluster2:k7_fm_write": "cake_kimi_k3_attn_res_4857e02681f352274a12",
        "small_m_cluster2:k7_nc4_fm": "cake_kimi_k3_attn_res_12101c0c5b4336b8c00e",
        "small_m_cluster2:k7_nc4_fm_write": "cake_kimi_k3_attn_res_4a9681983424fd0a1293",
        "small_m_cluster2:k8": "cake_kimi_k3_attn_res_4f0cabee0d8f305aace2",
        "small_m_cluster4:k4_fm": "cake_kimi_k3_attn_res_b4c932378bcb431ee27a",
        "small_m_cluster4:k4_fm_write": "cake_kimi_k3_attn_res_73915c216dbaf8da3bbd",
        "small_m_cluster4:k5_fm": "cake_kimi_k3_attn_res_b71ef632ff11c3a6abca",
        "small_m_cluster4:k6": "cake_kimi_k3_attn_res_110f5de77158b1bcfe57",
        "small_m_cluster4:k6_fm": "cake_kimi_k3_attn_res_767d23dd29a86915ae01",
        "small_m_cluster4:k7_fm": "cake_kimi_k3_attn_res_207e4fd6704c831c149b",
        "small_m_cluster4:k7_fm_write": "cake_kimi_k3_attn_res_c15281d36724f27d10e3",
        "small_m_cluster4:k8_fm": "cake_kimi_k3_attn_res_d959656094922bdab26f",
        "small_m_direct:k0": "cake_kimi_k3_attn_res_d57d39706cbed5379055",
        "small_m_direct:k1_fm": "cake_kimi_k3_attn_res_9f53a8193923490ea432",
        "small_m_direct:k1_fm_write": "cake_kimi_k3_attn_res_532cf65fac66f79cbbd7",
        "small_m_direct:k2_fm": "cake_kimi_k3_attn_res_decfe1cc69007dcc218a",
        "small_m_direct:k3_fm": "cake_kimi_k3_attn_res_14855b6501f28a8af20c",
        "small_m_direct:k4_fm": "cake_kimi_k3_attn_res_f4adf17b2ed959069d30",
        "small_m_direct:k4_fm_write": "cake_kimi_k3_attn_res_d20108254c08ce643df7",
        "small_m_direct:k5_nc4_fm": "cake_kimi_k3_attn_res_b41d37766b2613218b29",
        "small_m_direct:k6_nc4_fm": "cake_kimi_k3_attn_res_788cbcf6993668e37d68",
        "small_m_direct:k7_nc4_fm": "cake_kimi_k3_attn_res_fb0a96ae8e0c2345c090",
    },
    "sm_103a": {
        "direct:k0_d0_w1_n1": "cake_kimi_k3_attn_res_fc09e868f74b8c6fdf0e",
        "direct:k4_d0_w0_n1": "cake_kimi_k3_attn_res_291d118f0d9137e82482",
        "direct:k8_d1_w0_n0": "cake_kimi_k3_attn_res_d6c48b9566d909d3e0b8",
        "k0_tma": "cake_kimi_k3_attn_res_80715310f7aa0cb02593",
        "native_k8": "cake_kimi_k3_attn_res_99f148a06e530e457f4a",
        "native_m128": "cake_kimi_k3_attn_res_709f669fe8647d61aaf0",
        "persistent:k1_nc4_d2_f011000000000000": "cake_kimi_k3_attn_res_d99fda249fb6c79c8622",
        "persistent:k1_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_8149765ce4eb797a1263",
        "persistent:k1_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_ece2ac302cbe5d39ff8c",
        "persistent:k1_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_1974170f2264826fd01b",
        "persistent:k1_nc4_d2_f110000000000010": "cake_kimi_k3_attn_res_53790424af406f2edf30",
        "persistent:k1_nc4_d2_f110000000000010_fm": "cake_kimi_k3_attn_res_5e7c661caa058a3ed916",
        "persistent:k1_nc4_d2_f110000000000010_write": "cake_kimi_k3_attn_res_7c8168585b950bb93dc9",
        "persistent:k1_nc4_d2_f110000000100000": "cake_kimi_k3_attn_res_a15c9b10e0c06764bb50",
        "persistent:k1_nc4_d2_f110000000100000_fm": "cake_kimi_k3_attn_res_ab92afb21c782d62a0ad",
        "persistent:k1_nc4_d2_f110000000100000_write": "cake_kimi_k3_attn_res_ba11780fad8db3117e12",
        "persistent:k2_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_87822e6dd352c14dcd2b",
        "persistent:k2_nc3_d2_f111000000000000_fm": "cake_kimi_k3_attn_res_7d9facbe77d92ebf65f1",
        "persistent:k2_nc4_d2_f011000000000000": "cake_kimi_k3_attn_res_34ad0cffc89ae1e7aa29",
        "persistent:k2_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_3249f042c00c3ee119ef",
        "persistent:k2_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_3912cd6ce83f35c2504d",
        "persistent:k3_nc2_d2_f111000000000000_fm": "cake_kimi_k3_attn_res_fb6f7d32a38cf4bbc808",
        "persistent:k3_nc4_d2_f011000000000000": "cake_kimi_k3_attn_res_935cbafc006862bdf628",
        "persistent:k3_nc4_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_34865263e0a3756e1d33",
        "persistent:k4_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_63d5baa9d0eb595ac0ac",
        "persistent:k4_nc3_d2_f011000000000000_fm_write": "cake_kimi_k3_attn_res_b885dd8a2d8b94643f50",
        "persistent:k4_nc3_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_db8d149a6c5d0a8317fd",
        "persistent:k4_nc3_d3_f011000000000000_write": "cake_kimi_k3_attn_res_970f5ee5b6e600f64d47",
        "persistent:k4_nc3_d3_f110000000000000": "cake_kimi_k3_attn_res_27ecb918713280215e50",
        "persistent:k4_nc5_d2_f010001000000000": "cake_kimi_k3_attn_res_923c80c79dc7e5f529c3",
        "persistent:k4_nc5_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_a3e722cd107123bd4726",
        "persistent:k5_nc3_d3_f010000000000000_fm": "cake_kimi_k3_attn_res_caf1e4d34ba9c3debadd",
        "persistent:k5_nc4_d2_f010000000000000": "cake_kimi_k3_attn_res_3ccd8bb156fa4071f08c",
        "persistent:k5_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_c33474069e53fb4759dc",
        "persistent:k5_nc4_d3_f100000000000000": "cake_kimi_k3_attn_res_9f673b47d6e8c220ecdd",
        "persistent:k5_nc4_d3_f100000000000000_fm": "cake_kimi_k3_attn_res_ce52de03a90dc1af2f74",
        "persistent:k5_nc4_d3_f110000000000000": "cake_kimi_k3_attn_res_be7bfd8261eeb4e6d8b5",
        "persistent:k6_nc3_d2_f011000000000000_fm": "cake_kimi_k3_attn_res_ab1c5516198a5b0fed97",
        "persistent:k6_nc4_d2_f010000000000000": "cake_kimi_k3_attn_res_55b1ddf9602fd45a1404",
        "persistent:k6_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_58c36fe827b800606c5e",
        "persistent:k7_nc3_d2_f010000000000000_fm": "cake_kimi_k3_attn_res_d8a00f7ef7bdcde4db62",
        "persistent:k7_nc4_d2_f110000000000000": "cake_kimi_k3_attn_res_d510b7826e8e4cf50bed",
        "persistent:k7_nc4_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_6468e67b15dab2e973eb",
        "persistent:k7_nc4_d2_f110000000000000_write": "cake_kimi_k3_attn_res_bc41069b56c74214c587",
        "persistent:k8_nc3_d2_f011100000000000": "cake_kimi_k3_attn_res_e682c53fdd5475bf33f5",
        "persistent:k8_nc3_d2_f011100000000001": "cake_kimi_k3_attn_res_e95cf8d5c1116af31c5d",
        "persistent:k8_nc5_d2_f110000000000000": "cake_kimi_k3_attn_res_74f8cbf06549fb15441f",
        "persistent:k8_nc5_d2_f110000000000000_fm": "cake_kimi_k3_attn_res_13cf584a1ad91f2c2137",
        "small_m_cluster2:k2_fm": "cake_kimi_k3_attn_res_c4a61daa7501667c9886",
        "small_m_cluster2:k4_fm": "cake_kimi_k3_attn_res_dd089fe2c1492e3bb21d",
        "small_m_cluster2:k4_fm_write": "cake_kimi_k3_attn_res_c0ab67a53d992f2fdf17",
        "small_m_cluster2:k5": "cake_kimi_k3_attn_res_959261ede770f517ff44",
        "small_m_cluster2:k5_nc4_fm": "cake_kimi_k3_attn_res_682069efbbc5e016c716",
        "small_m_cluster2:k6_fm": "cake_kimi_k3_attn_res_9bbfc3ccb47bb2370fbc",
        "small_m_cluster2:k6_nc4_fm": "cake_kimi_k3_attn_res_7d4a21fb3d795a6640b3",
        "small_m_cluster2:k7_fm": "cake_kimi_k3_attn_res_8a0a9252345c6b4769de",
        "small_m_cluster2:k7_fm_write": "cake_kimi_k3_attn_res_4857e02681f352274a12",
        "small_m_cluster2:k7_nc4_fm": "cake_kimi_k3_attn_res_12101c0c5b4336b8c00e",
        "small_m_cluster2:k8": "cake_kimi_k3_attn_res_4f0cabee0d8f305aace2",
        "small_m_cluster2:k8_fm": "cake_kimi_k3_attn_res_d0dc3859ec7be0407f60",
        "small_m_direct:k0_fm": "cake_kimi_k3_attn_res_8e61459b47e062ea2e03",
        "small_m_direct:k1_fm": "cake_kimi_k3_attn_res_9f53a8193923490ea432",
        "small_m_direct:k1_fm_write": "cake_kimi_k3_attn_res_532cf65fac66f79cbbd7",
        "small_m_direct:k2_fm": "cake_kimi_k3_attn_res_decfe1cc69007dcc218a",
        "small_m_direct:k3": "cake_kimi_k3_attn_res_3055514f3a17ac73dc8d",
        "small_m_direct:k3_fm": "cake_kimi_k3_attn_res_14855b6501f28a8af20c",
        "small_m_direct:k4_fm": "cake_kimi_k3_attn_res_f4adf17b2ed959069d30",
        "small_m_direct:k4_fm_write": "cake_kimi_k3_attn_res_d20108254c08ce643df7",
        "small_m_direct:k5_nc4_fm": "cake_kimi_k3_attn_res_b41d37766b2613218b29",
        "small_m_direct:k5_nc6_fm": "cake_kimi_k3_attn_res_12cd385f9c0656783118",
        "small_m_direct:k6_nc4_fm": "cake_kimi_k3_attn_res_788cbcf6993668e37d68",
        "small_m_direct:k7_nc4_fm": "cake_kimi_k3_attn_res_fb0a96ae8e0c2345c090",
        "small_m_direct:k7_nc4_fm_write": "cake_kimi_k3_attn_res_b545bf557abbe26776c0",
    },
}

# The programs use tcgen05 TMEM (the persistent and native ports) and are
# exact-architecture payloads: one source, compiled once per listed target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def _nvcc_flags(arches: list[str]) -> list[str]:
    """The target code-generation flags of ``arches`` followed by the common flags, each once."""
    flags: list[str] = []
    for arch in arches:
        for flag in ARCH_NVCC_FLAGS[arch]:
            if flag not in flags:
                flags.append(flag)
    return flags


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    table = KERNELS.get(arch)
    if not table:
        return False
    return all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """The registered module of logical kernel ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if not table:
        raise NotImplementedError(
            f"No generated Kimi-K3 AttnRes programs are registered for {arch} in this checkout"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 AttnRes kernel {key!r} for {arch} is not registered in this "
            "checkout (plan_route substitutes a registered variant of the same family; see the package README)"
        )
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"registered module {name!r} is not built for {arch} (arches: {record['arches']})"
        )
    return name


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_kimi_k3_attn_res_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_{record['closure']}",
        sources=sources,
        extra_cuda_cflags=[*_nvcc_flags(record["arches"]), *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_attn_res_module(name: str):
    return gen_cake_kimi_k3_attn_res_module(name).build_and_load()
