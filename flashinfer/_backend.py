"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Private contracts shared by backend-dispatched FlashInfer APIs.
"""


class _BackendPlanUnsupportedError(ValueError):
    """Raised when a backend cannot plan the requested configuration.

    Automatic backend selection catches this narrow internal signal to try the
    next candidate. Invalid user input and unexpected planning failures should
    use their normal exceptions so they are not mistaken for fallback.
    """
