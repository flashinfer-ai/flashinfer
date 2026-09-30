"""Explicit activation bridges for the vLLM general-plugin registry."""

import os


def _enabled(name):
    return name in {
        item.strip() for item in os.environ.get("VLLM_PLUGINS", "").split(",")
    }


def register_loader():
    if _enabled("b12x_loader"):
        from .integration.vllm.loader import register_b12x_loader

        register_b12x_loader()
