from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="flashinfer",
    version="0.1",
    url="https://github.com/yzh119/flashinfer",
    package_dir={'flashinfer': 'flashinfer'},
    packages=['flashinfer'],
    ext_modules=[
        CUDAExtension('flashinfer', [
            'csrc/pybind.cu',
        ], include_dirs=['../include']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires=">=3.8"
)