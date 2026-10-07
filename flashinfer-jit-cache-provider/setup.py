"""Setuptools configuration for one architecture-specific provider wheel."""

from setuptools import setup

from package_config import get_provider_build_config


config = get_provider_build_config()

setup(
    name=config.distribution,
    version=config.version,
    description=f"FlashInfer pre-compiled JIT cache provider for {config.provider_tag}",
    license="Apache-2.0",
    author="FlashInfer team",
    url="https://github.com/flashinfer-ai/flashinfer",
    python_requires=">=3.9",
    packages=[config.package],
    package_dir={config.package: "flashinfer_jit_cache_provider"},
    package_data={config.package: ["manifest.json", "jit_cache/**/*.so"]},
    include_package_data=False,
    zip_safe=False,
    entry_points={
        "flashinfer.jit_cache.providers": [
            f"{config.provider_tag}={config.package}:get_provider"
        ]
    },
)
