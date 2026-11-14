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

Backend packages of the Cake training kernels (issue #8928).  A subpackage
named ``cake_<op>_<det|nondet>_train[_kcp]`` is the backend of the top-level
module of the same name, ``flashinfer.cake_<op>_<det|nondet>_train[_kcp]``;
both modes live here.  For these packages the mode is part of the module name
and no entry point takes a ``deterministic=`` keyword.  A ``det`` package
returns run-to-run bitwise-identical results.  A ``nondet`` package makes no
run-to-run bitwise promise; under ``torch.use_deterministic_algorithms(True)``
it raises PyTorch's alert at its non-deterministic step (with
``warn_only=True`` it warns and runs ``nondet``) and never falls back to
``det``.  A subpackage whose family is not renamed yet keeps its current entry
points until that rename.  Each subpackage is standalone (import it by its
full name); this module imports nothing.
"""
