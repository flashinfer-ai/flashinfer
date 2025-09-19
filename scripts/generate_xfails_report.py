#!/usr/bin/env python3
"""
FlashInfer Xfails Report Generator

Automatically generates a comprehensive report of all test skips and expected failures
in the FlashInfer test suite. This script can be run periodically to track progress
in fixing test issues.

Usage:
    python scripts/generate_xfails_report.py [--output OUTPUT_FILE] [--format FORMAT]
    
Arguments:
    --output: Output file path (default: XFAILS_REPORT.md)
    --format: Output format: markdown, json, or csv (default: markdown)
    --test-dir: Test directory to analyze (default: tests/)
    --verbose: Enable verbose output
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import csv


class XfailsAnalyzer:
    """Analyzes Python test files to extract pytest skips and xfails."""
    
    def __init__(self, test_dir: str, verbose: bool = False):
        self.test_dir = Path(test_dir)
        self.verbose = verbose
        self.marks = []
        
    def analyze_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract pytest marks from a single file."""
        marks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function decorators
                    for decorator in node.decorator_list:
                        mark_info = self._parse_decorator(decorator, str(file_path), node.lineno)
                        if mark_info:
                            mark_info['function'] = node.name
                            marks.append(mark_info)
                
                elif isinstance(node, ast.Call):
                    # Check for pytest.skip(), pytest.xfail() calls
                    mark_info = self._parse_call(node, str(file_path))
                    if mark_info:
                        marks.append(mark_info)
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error parsing {file_path}: {e}")
        
        return marks
    
    def _parse_decorator(self, decorator, file_path: str, lineno: int) -> Dict[str, Any]:
        """Parse pytest decorator (@pytest.mark.skipif, @pytest.mark.xfail)."""
        if isinstance(decorator, ast.Attribute):
            if self._is_pytest_mark(decorator):
                mark_type = decorator.attr
                if mark_type in ['skipif', 'xfail']:
                    return {
                        'type': mark_type,
                        'file': file_path,
                        'line': lineno,
                        'reason': 'decorator without args',
                        'condition': None
                    }
        
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute) and self._is_pytest_mark(decorator.func):
                mark_type = decorator.func.attr
                if mark_type in ['skipif', 'xfail']:
                    reason = self._extract_reason(decorator.args, decorator.keywords)
                    condition = self._extract_condition(decorator.args) if mark_type == 'skipif' else None
                    
                    return {
                        'type': mark_type,
                        'file': file_path,
                        'line': lineno,
                        'reason': reason,
                        'condition': condition
                    }
        
        return None
    
    def _parse_call(self, node, file_path: str) -> Dict[str, Any]:
        """Parse pytest.skip() or pytest.xfail() calls."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id == 'pytest' and 
                node.func.attr in ['skip', 'xfail']):
                
                reason = self._extract_reason(node.args, node.keywords)
                return {
                    'type': node.func.attr,
                    'file': file_path,
                    'line': node.lineno,
                    'reason': reason,
                    'condition': None,
                    'function': 'inline_call'
                }
        
        return None
    
    def _is_pytest_mark(self, node) -> bool:
        """Check if node represents pytest.mark."""
        return (isinstance(node.value, ast.Attribute) and
                isinstance(node.value.value, ast.Name) and
                node.value.value.id == 'pytest' and
                node.value.attr == 'mark')
    
    def _extract_reason(self, args: List[ast.AST], keywords: List[ast.keyword]) -> str:
        """Extract reason from function arguments."""
        for kw in keywords:
            if kw.arg == 'reason' and isinstance(kw.value, ast.Constant):
                return kw.value.value
        
        if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            return args[0].value
        
        return "No reason provided"
    
    def _extract_condition(self, args: List[ast.AST]) -> str:
        """Extract condition from skipif arguments."""
        if args:
            try:
                return ast.unparse(args[0])
            except:
                return "Complex condition"
        return None
    
    def analyze_all(self) -> List[Dict[str, Any]]:
        """Analyze all test files in the test directory."""
        all_marks = []
        
        for py_file in self.test_dir.glob('**/*.py'):
            if py_file.name.startswith('test_'):
                if self.verbose:
                    print(f"Analyzing {py_file}")
                file_marks = self.analyze_file(py_file)
                all_marks.extend(file_marks)
        
        self.marks = all_marks
        return all_marks


class ReportGenerator:
    """Generates reports from xfails analysis results."""
    
    def __init__(self, marks: List[Dict[str, Any]]):
        self.marks = marks
        self.categories = self._categorize_marks()
    
    def _categorize_marks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize marks by type and reason patterns."""
        categories = {
            'hardware_requirements': [],
            'feature_unsupported': [], 
            'parameter_validation': [],
            'environment_issues': [],
            'backend_limitations': [],
            'other': []
        }
        
        for mark in self.marks:
            reason = mark.get('reason', '').lower()
            condition = mark.get('condition', '') or ''
            
            if any(keyword in reason + condition for keyword in 
                   ['sm90', 'sm100', 'sm110', 'sm120', 'hopper', 'blackwell', 'compute capability']):
                categories['hardware_requirements'].append(mark)
            elif any(keyword in reason for keyword in ['not supported', 'unsupported', 'support']):
                categories['feature_unsupported'].append(mark)
            elif any(keyword in reason for keyword in ['invalid combination', 'parameter', 'must be', 'should be']):
                categories['parameter_validation'].append(mark)
            elif any(keyword in reason for keyword in ['backend', 'cutlass', 'cudnn', 'trtllm']):
                categories['backend_limitations'].append(mark)
            elif any(keyword in reason for keyword in ['memory', 'oom', 'device']):
                categories['environment_issues'].append(mark)
            else:
                categories['other'].append(mark)
        
        return categories
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        report = []
        now = datetime.now().strftime("%Y-%m-%d")
        total_marks = len(self.marks)
        
        # Header
        report.extend([
            "# FlashInfer Test Skips and Expected Failures Report",
            "",
            f"**Generated:** {now}",
            f"**Total Issues:** {total_marks}",
            "",
            "## Executive Summary",
            "",
            f"This report identifies **{total_marks} test skips and expected failures** across the FlashInfer test suite.",
            ""
        ])
        
        # Statistics table
        report.extend([
            "## Issue Breakdown",
            "",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|"
        ])
        
        for category, marks in self.categories.items():
            count = len(marks)
            percentage = (count / total_marks * 100) if total_marks > 0 else 0
            category_name = category.replace('_', ' ').title()
            report.append(f"| **{category_name}** | {count} | {percentage:.1f}% |")
        
        report.extend(["", "## Detailed Analysis", ""])
        
        # Detailed breakdown
        for category, marks in self.categories.items():
            if not marks:
                continue
                
            category_name = category.replace('_', ' ').title()
            report.extend([
                f"### {category_name} ({len(marks)} issues)",
                "",
            ])
            
            # Group by reason
            reason_groups = {}
            for mark in marks:
                reason = mark.get('reason', 'No reason provided')
                if reason not in reason_groups:
                    reason_groups[reason] = []
                reason_groups[reason].append(mark)
            
            for reason, reason_marks in sorted(reason_groups.items()):
                report.extend([
                    f"#### {reason}",
                    f"**Count:** {len(reason_marks)}",
                    "",
                    "**Affected files:**"
                ])
                
                # Group by file
                file_groups = {}
                for mark in reason_marks:
                    file_path = mark['file']
                    if file_path not in file_groups:
                        file_groups[file_path] = []
                    file_groups[file_path].append(mark)
                
                for file_path in sorted(file_groups.keys()):
                    file_marks = file_groups[file_path]
                    lines = [str(m['line']) for m in file_marks]
                    relative_path = file_path.replace('/home/runner/work/flashinfer/flashinfer/', '')
                    report.append(f"- `{relative_path}` (lines: {', '.join(lines)})")
                
                report.append("")
        
        # Recommendations
        report.extend([
            "## Recommendations",
            "",
            "### Immediate Actions",
            ""
        ])
        
        if self.categories['hardware_requirements']:
            report.extend([
                "1. **Hardware Compatibility Audit**",
                "   - Review GPU compute capability requirements",
                "   - Implement fallbacks for older hardware",
                "   - Create hardware compatibility matrix",
                ""
            ])
        
        if self.categories['feature_unsupported']:
            report.extend([
                "2. **Feature Implementation Priority**",
                "   - Prioritize implementing missing features",
                "   - Add feature availability documentation",
                "   - Implement graceful degradation",
                ""
            ])
        
        if self.categories['parameter_validation']:
            report.extend([
                "3. **Parameter Validation Enhancement**",
                "   - Improve parameter validation logic",
                "   - Add better error messages",
                "   - Create parameter validation helpers",
                ""
            ])
        
        report.extend([
            "### Long-term Goals",
            "",
            "- Reduce total xfails/skips by 50% within 6 months",
            "- Achieve 95% test pass rate on supported hardware",
            "- Implement comprehensive parameter validation framework",
            "- Create automated xfails tracking in CI/CD",
            "",
            "---",
            f"*Report generated on {now} by FlashInfer xfails analysis tool*"
        ])
        
        return "\n".join(report)
    
    def generate_json_report(self) -> str:
        """Generate JSON report."""
        report_data = {
            'generated': datetime.now().isoformat(),
            'total_issues': len(self.marks),
            'categories': {},
            'all_issues': self.marks
        }
        
        for category, marks in self.categories.items():
            report_data['categories'][category] = {
                'count': len(marks),
                'issues': marks
            }
        
        return json.dumps(report_data, indent=2)
    
    def generate_csv_report(self) -> str:
        """Generate CSV report."""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Type', 'File', 'Line', 'Function', 'Reason', 'Condition', 'Category'])
        
        # Data
        for mark in self.marks:
            category = 'other'
            for cat, cat_marks in self.categories.items():
                if mark in cat_marks:
                    category = cat
                    break
            
            writer.writerow([
                mark.get('type', ''),
                mark.get('file', ''),
                mark.get('line', ''),
                mark.get('function', ''),
                mark.get('reason', ''),
                mark.get('condition', ''),
                category
            ])
        
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Generate FlashInfer xfails report')
    parser.add_argument('--output', default='XFAILS_REPORT.md', 
                       help='Output file path (default: XFAILS_REPORT.md)')
    parser.add_argument('--format', choices=['markdown', 'json', 'csv'], default='markdown',
                       help='Output format (default: markdown)')
    parser.add_argument('--test-dir', default='tests/',
                       help='Test directory to analyze (default: tests/)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory {args.test_dir} does not exist")
        sys.exit(1)
    
    # Analyze test files
    analyzer = XfailsAnalyzer(args.test_dir, args.verbose)
    marks = analyzer.analyze_all()
    
    print(f"Found {len(marks)} xfails/skips across test files")
    
    # Generate report
    generator = ReportGenerator(marks)
    
    if args.format == 'markdown':
        report_content = generator.generate_markdown_report()
    elif args.format == 'json':
        report_content = generator.generate_json_report()
    elif args.format == 'csv':
        report_content = generator.generate_csv_report()
    
    # Write report
    with open(args.output, 'w') as f:
        f.write(report_content)
    
    print(f"Report written to {args.output}")
    
    # Print summary
    categories = generator.categories
    print("\nSummary:")
    for category, marks in categories.items():
        if marks:
            print(f"  {category.replace('_', ' ').title()}: {len(marks)}")


if __name__ == "__main__":
    main()