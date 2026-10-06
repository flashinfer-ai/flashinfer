"""Custom ops carry tensors and primitives only; prepared state stays in Python.

Every ``torch.library.custom_op`` function in the package must be callable from
a compiled graph without Dynamo ever seeing a Python object: its parameters are
tensors, optional tensors, tensor lists, ints, floats, bools, strings, or
``torch.dtype``. Plans are referenced through their integer handle. No b12x type
is registered as a torch opaque type.
"""

from pathlib import Path
import ast
import re

import pytest

PACKAGE = Path(__file__).resolve().parents[4] / "flashinfer/experimental/b12x"

ALLOWED = {
    "torch.Tensor", "Tensor", "torch.Tensor | None", "Tensor | None",
    "list[torch.Tensor]", "list[Tensor]", "Sequence[torch.Tensor]",
    "int", "int | None", "float", "float | None", "bool", "bool | None",
    "str", "str | None", "torch.dtype", "torch.dtype | None",
    "list[int]", "tuple[int, ...]", "list[int] | None", "list[float]",
}


def _annotation(node):
    if node is None:
        return None
    text = ast.unparse(node).replace("typing.", "")
    match = re.fullmatch(r"Optional\[(.+)\]", text)
    return f"{match.group(1)} | None" if match else text


def _custom_op_functions(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = ast.unparse(target)
            if name.endswith("custom_op") or name.endswith("register_fake"):
                yield node
                break


def op_boundary_violations(root=PACKAGE):
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in _custom_op_functions(tree):
            for argument in (*function.args.args, *function.args.kwonlyargs):
                annotation = _annotation(argument.annotation)
                if annotation not in ALLOWED:
                    violations.append(
                        f"{path.relative_to(root.parent)}:{function.lineno} "
                        f"{function.name}({argument.arg}: {annotation})"
                    )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("register_opaque_type"):
                violations.append(f"{path.relative_to(root.parent)}:{node.lineno} register_opaque_type")
    return violations


def test_custom_ops_take_only_tensors_and_primitives():
    violations = op_boundary_violations()
    assert not violations, "\n".join(violations)


if __name__ == "__main__":
    for line in op_boundary_violations():
        print(line)
