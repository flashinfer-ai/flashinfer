#!/usr/bin/env python3
"""
XFails Tracker - Report Generator for pytest.mark.xfail markers

This script scans the test suite for xfail markers and generates a report
showing the total number of xfails and their reasons.

Usage:
    python scripts/xfails_tracker.py
"""

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class XFailInfo:
    """Information about a single xfail marker."""

    file_path: str
    line_number: int
    test_name: str
    reason: str
    condition: Optional[str]
    strict: Optional[bool]
    xfail_type: str  # 'decorator', 'parameter', or 'runtime'

    def __str__(self):
        return f"{self.file_path}:{self.line_number} - {self.test_name}"


class XFailCollector(ast.NodeVisitor):
    """AST visitor to collect xfail markers from test files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.xfails: List[XFailInfo] = []
        self.current_function_name: Optional[str] = None
        self.in_test_function: bool = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions to check for xfail decorators."""
        # Save the current function name for context
        old_function_name = self.current_function_name
        old_in_test_function = self.in_test_function

        self.current_function_name = node.name
        # Check if this is a test function (starts with "test_" or has @pytest marks)
        self.in_test_function = node.name.startswith("test_") or any(
            self._has_pytest_mark(dec) for dec in node.decorator_list
        )

        # Check decorators for xfail markers
        for decorator in node.decorator_list:
            self._check_decorator(decorator, node.name, node.lineno)

        # Continue visiting child nodes (to find parametrize xfails and runtime calls)
        self.generic_visit(node)

        # Restore previous function name and test status
        self.current_function_name = old_function_name
        self.in_test_function = old_in_test_function

    def visit_Call(self, node: ast.Call):
        """Visit function calls to find pytest.param with xfail marks and runtime xfails."""
        # Check if this is a pytest.param call
        if self._is_pytest_param_call(node):
            # Look for marks keyword argument
            for keyword in node.keywords:
                if keyword.arg == "marks":
                    self._check_marks_argument(keyword.value, node.lineno)

        # Check if this is a runtime pytest.xfail() call
        # Note: We check for xfails even outside test functions, as they can be in helper functions
        elif self._is_pytest_xfail_call(node):
            self._extract_runtime_xfail(node)

        self.generic_visit(node)

    def _is_pytest_param_call(self, node: ast.Call) -> bool:
        """Check if a call is pytest.param."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "param" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "pytest":
                    return True
        return False

    def _is_pytest_xfail_call(self, node: ast.Call) -> bool:
        """Check if a call is pytest.xfail()."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "xfail" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "pytest":
                    return True
        return False

    def _has_pytest_mark(self, node) -> bool:
        """Check if a decorator is a pytest mark."""
        if isinstance(node, ast.Attribute):
            if node.attr in ["parametrize", "xfail", "skip"] and isinstance(
                node.value, ast.Attribute
            ):
                if node.value.attr == "mark" and isinstance(node.value.value, ast.Name):
                    if node.value.value.id == "pytest":
                        return True
        elif isinstance(node, ast.Call):
            return self._has_pytest_mark(node.func)
        return False

    def _check_decorator(self, decorator, test_name: str, lineno: int):
        """Check if a decorator is an xfail marker."""
        # Handle @pytest.mark.xfail(...)
        if isinstance(decorator, ast.Call):
            if self._is_xfail_marker(decorator.func):
                self._extract_xfail_info(decorator, test_name, lineno, "decorator")
        # Handle @pytest.mark.xfail (without parentheses)
        elif self._is_xfail_marker(decorator):
            self.xfails.append(
                XFailInfo(
                    file_path=self.file_path,
                    line_number=lineno,
                    test_name=test_name,
                    reason="No reason provided",
                    condition=None,
                    strict=None,
                    xfail_type="decorator",
                )
            )

    def _check_marks_argument(self, marks_node, lineno: int):
        """Check the marks argument in pytest.param for xfail markers."""
        # marks can be a single marker or a list of markers
        if isinstance(marks_node, ast.Call) and self._is_xfail_marker(marks_node.func):
            test_name = self.current_function_name or "unknown"
            self._extract_xfail_info(marks_node, test_name, lineno, "parameter")
        elif isinstance(marks_node, (ast.List, ast.Tuple)):
            for mark in marks_node.elts:
                if isinstance(mark, ast.Call) and self._is_xfail_marker(mark.func):
                    test_name = self.current_function_name or "unknown"
                    self._extract_xfail_info(mark, test_name, lineno, "parameter")

    def _is_xfail_marker(self, node) -> bool:
        """Check if a node represents pytest.mark.xfail."""
        if isinstance(node, ast.Attribute):
            if node.attr == "xfail" and isinstance(node.value, ast.Attribute):
                if (
                    node.value.attr == "mark"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "pytest"
                ):
                    return True
        return False

    def _extract_xfail_info(
        self, call_node: ast.Call, test_name: str, lineno: int, xfail_type: str
    ):
        """Extract information from an xfail marker call."""
        reason = "No reason provided"
        condition = None
        strict = None

        # Check positional arguments (first arg might be condition)
        if call_node.args:
            condition = self._ast_to_string(call_node.args[0])

        # Check keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == "reason":
                reason = self._get_string_value(keyword.value)
            elif keyword.arg == "strict":
                strict = self._get_bool_value(keyword.value)
            elif keyword.arg is None:  # **kwargs
                pass

        self.xfails.append(
            XFailInfo(
                file_path=self.file_path,
                line_number=lineno,
                test_name=test_name,
                reason=reason,
                condition=condition,
                strict=strict,
                xfail_type=xfail_type,
            )
        )

    def _get_string_value(self, node) -> str:
        """Extract string value from an AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.JoinedStr):  # f-string
            return self._ast_to_string(node)
        else:
            return self._ast_to_string(node)

    def _get_bool_value(self, node) -> Optional[bool]:
        """Extract boolean value from an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return node.value
        return None

    def _ast_to_string(self, node) -> str:
        """Convert an AST node to a string representation."""
        try:
            return ast.unparse(node)
        except AttributeError:
            # Python < 3.9 doesn't have ast.unparse
            return ast.dump(node)

    def _extract_runtime_xfail(self, call_node: ast.Call):
        """Extract information from a runtime pytest.xfail() call."""
        test_name = self.current_function_name or "unknown"
        reason = "No reason provided"
        condition = (
            None  # Runtime xfails are already conditional (inside if statements)
        )

        # The first positional argument is typically the reason
        if call_node.args:
            reason = self._get_string_value(call_node.args[0])

        # Check keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == "reason":
                reason = self._get_string_value(keyword.value)

        self.xfails.append(
            XFailInfo(
                file_path=self.file_path,
                line_number=call_node.lineno,
                test_name=test_name,
                reason=reason,
                condition=condition,
                strict=None,
                xfail_type="runtime",
            )
        )


def find_test_files(tests_dir: Path) -> List[Path]:
    """Find all Python test files in the tests directory."""
    test_files = []
    # Find test files
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend(tests_dir.rglob(pattern))
    # Also find utility files that might contain xfail calls (like utils.py, conftest.py)
    for pattern in ["utils.py", "conftest.py", "helpers.py"]:
        test_files.extend(tests_dir.rglob(pattern))
    return sorted(set(test_files))  # Remove duplicates


def collect_xfails(test_files: List[Path]) -> List[XFailInfo]:
    """Collect all xfail markers from test files."""
    all_xfails = []

    for test_file in test_files:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(test_file))
            collector = XFailCollector(str(test_file))
            collector.visit(tree)
            all_xfails.extend(collector.xfails)

        except SyntaxError as e:
            print(f"Warning: Syntax error in {test_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error processing {test_file}: {e}", file=sys.stderr)

    return all_xfails


def group_xfails_by_reason(xfails: List[XFailInfo]) -> Dict[str, List[XFailInfo]]:
    """Group xfails by their reason."""
    grouped = defaultdict(list)
    for xfail in xfails:
        grouped[xfail.reason].append(xfail)
    return dict(grouped)


def format_table(xfails: List[XFailInfo], workspace_root: Path) -> str:
    """Format xfails as a table."""
    if not xfails:
        return "No xfails found in the test suite! 🎉"

    # Group by reason
    grouped = group_xfails_by_reason(xfails)

    # Build the report
    lines = []
    lines.append("=" * 100)
    lines.append("XFAILS REPORT")
    lines.append("=" * 100)
    lines.append(f"\nTotal xfails: {len(xfails)}")
    lines.append(f"Unique reasons: {len(grouped)}")
    lines.append("\n")

    # Summary table
    lines.append("-" * 100)
    lines.append(f"{'Reason':<60} {'Count':>10} {'Type':>15}")
    lines.append("-" * 100)

    # Sort by count (descending)
    sorted_reasons = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    for reason, xfails_list in sorted_reasons:
        count = len(xfails_list)
        # Get the most common type for this reason
        types = [x.xfail_type for x in xfails_list]
        type_str = max(set(types), key=types.count)
        # Truncate long reasons
        display_reason = reason if len(reason) <= 58 else reason[:55] + "..."
        lines.append(f"{display_reason:<60} {count:>10} {type_str:>15}")

    lines.append("-" * 100)
    lines.append("\n")

    # Detailed breakdown
    lines.append("=" * 100)
    lines.append("DETAILED BREAKDOWN BY REASON")
    lines.append("=" * 100)

    for reason, xfails_list in sorted_reasons:
        lines.append(f"\n[{len(xfails_list)} xfails] {reason}")
        lines.append("-" * 100)

        for xfail in xfails_list:
            # Make path relative to workspace root
            try:
                rel_path = Path(xfail.file_path).relative_to(workspace_root)
            except ValueError:
                rel_path = Path(xfail.file_path)

            location = f"{rel_path}:{xfail.line_number}"
            lines.append(f"  • {location}")
            lines.append(f"    Test: {xfail.test_name}")
            lines.append(f"    Type: {xfail.xfail_type}")
            if xfail.condition:
                lines.append(f"    Condition: {xfail.condition}")
            if xfail.strict is not None:
                lines.append(f"    Strict: {xfail.strict}")
            lines.append("")

    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    # Determine workspace root and tests directory
    workspace_root = Path(__file__).parent.parent
    tests_dir = workspace_root / "tests"

    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    # Find and process test files
    print(f"Scanning for test files in {tests_dir}...", file=sys.stderr)
    test_files = find_test_files(tests_dir)
    print(f"Found {len(test_files)} test files", file=sys.stderr)

    print("Collecting xfail markers...", file=sys.stderr)
    xfails = collect_xfails(test_files)

    # Format and print output as table
    output = format_table(xfails, workspace_root)
    print(output)


if __name__ == "__main__":
    main()
