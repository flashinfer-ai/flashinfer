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

# Experimental exact-SM110 GQA decode implementation (a Cake package).  The host
# side lives in ``.backend`` (``sm110_gqa_decode``, ``prepare_for_launch``,
# ``launch_prepared``) and the registry in ``.cake_jit``; the public entry points
# are ``flashinfer.sm110_gqa_decode``, ``flashinfer.prepare_sm110_gqa_decode`` and
# ``flashinfer.launch_sm110_gqa_decode_prepared`` in ``flashinfer/decode.py``.
