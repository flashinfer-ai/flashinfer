#!/usr/bin/env python3
"""
Verify that all FlashInfer modules are compiled.
This script is used by task_test_jit_cache_package_build_import.sh
"""

import sys
from flashinfer.jit import jit_spec_registry
from flashinfer.aot import register_default_modules


def main():
    # Register modules if not already registered
    if not jit_spec_registry.get_all_statuses():
        register_default_modules()

    stats = jit_spec_registry.get_stats()
    print(f"Total modules: {stats['total']}")
    print(f"Compiled: {stats['compiled']}")
    print(f"Not compiled: {stats['not_compiled']}")

    if stats["not_compiled"] > 0:
        print("\nERROR: Not all modules are compiled!")
        print("\nModules not compiled:")
        all_statuses = jit_spec_registry.get_all_statuses()
        for status in all_statuses:
            if not status.is_compiled:
                print(f"  - {status.name}")
                print(f"    Sources: {[str(s) for s in status.sources]}")
                print(f"    Needs device linking: {status.needs_device_linking}")
        return 1
    else:
        print("SUCCESS: All modules are compiled!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
