# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute original DSL function bodies in host-only audits."""

import ast
import __future__
from pathlib import Path
import sys


def python_body(function):
    """Recover an undecorated function from source, even after DSL compilation.

    CuTe can replace ``__wrapped__`` with transformed DSL code after JIT. Read
    the defining module instead so host audits do not depend on test order.
    Keep that module's globals live so the audits' monkeypatches still apply.
    """
    module = sys.modules[function.__module__]
    path = Path(module.__file__)
    node = ast.parse(path.read_text(), filename=str(path))
    for name in function.__qualname__.split("."):
        node = next(
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == name
        )
    node.decorator_list = []
    code = compile(
        ast.Module(body=[node], type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    namespace = {}
    exec(code, vars(module), namespace)
    return namespace[node.name]
