# v0.2.5+rocm

The initial technical preview release of a ROCm port of FlashInfer. The release
has the port of the decode kernels and some infrastructure changes.

## Added

- A port of the decode kernels to HIP. (#38, #34) @Madduri, Rishi 
- Add norm, page and rope to jit build infra (#46) @Madduri, Rishi
- Initial gpu interoperability interface for HIP/CUDA. (#22) @diptorupd
- Initial CDMA3 MFMA asbtractions (#62, #64, #68) @Madduri, Rishi
- Port build system to scikit-build-core. (#14) @diptorupd
