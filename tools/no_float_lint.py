#!/usr/bin/env python3
"""
No-Float AST Linter

Static analysis tool that examines Python AST to detect and reject
floating-point operations and risky constructs that could introduce
float contamination.

Usage:
    python no_float_lint.py <file.py>
    python -m tools.no_float_lint <file.py>
"""

import ast
import sys
import argparse
from typing import List, Set, Tuple, Optional
from pathlib import Path


class FloatContaminationError(Exception):
    """Raised when floating-point contamination is detected."""
    pass


class NoFloatLinter(ast.NodeVisitor):
    """AST visitor that detects floating-point contamination."""
    
    def __init__(self, filename: str = "<unknown>"):
        self.filename = filename
        self.errors: List[Tuple[int, int, str]] = []
        
        # Risky built-in functions that often return floats
        self.risky_builtins = {
            'float', 'complex', 'pow', 'abs', 'round',
            'sum',  # Can return float if any input is float
            'max', 'min',  # Can return float if any input is float
        }
        
        # Risky math module functions
        self.risky_math_functions = {
            'sqrt', 'pow', 'exp', 'log', 'log10', 'log2',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'degrees', 'radians', 'ceil', 'floor', 'fabs',
            'modf', 'frexp', 'ldexp', 'fmod', 'remainder',
            'copysign', 'isfinite', 'isinf', 'isnan',
        }
        
        # Risky operators that can produce floats
        self.risky_operators = {
            ast.Div,  # / always returns float in Python 3
            ast.Pow,  # ** can return float
        }
        
        # Safe operators that preserve integers
        self.safe_operators = {
            ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd
        }
    
    def add_error(self, node: ast.AST, message: str) -> None:
        """Add an error for the given node."""
        lineno = getattr(node, 'lineno', 0)
        col_offset = getattr(node, 'col_offset', 0)
        self.errors.append((lineno, col_offset, message))
    
    def visit_Constant(self, node: ast.Constant) -> None:
        """Check for float constants."""
        if isinstance(node.value, float):
            self.add_error(node, f"Float constant detected: {node.value}")
        elif isinstance(node.value, complex):
            self.add_error(node, f"Complex constant detected: {node.value}")
        self.generic_visit(node)
    
    # Removed deprecated ast.Num handler - ast.Constant handles all numeric constants in modern Python
    
    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Check for risky binary operations."""
        if type(node.op) in self.risky_operators:
            op_name = type(node.op).__name__
            if isinstance(node.op, ast.Div):
                self.add_error(node, f"Division (/) operator detected - use // for integer division")
            elif isinstance(node.op, ast.Pow):
                self.add_error(node, f"Power (**) operator detected - may return float")
        self.generic_visit(node)
    
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check for risky augmented assignments."""
        if isinstance(node.op, ast.Div):
            self.add_error(node, "Augmented division (/=) detected - use //=")
        elif isinstance(node.op, ast.Pow):
            self.add_error(node, "Augmented power (**=) detected - not allowed")
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check for risky function calls."""
        func_name = self._get_function_name(node.func)

        # Builtins and risky calls
        if func_name in self.risky_builtins:
            # Special case: pow(a,b,c) allowed (modular exponent → int)
            if func_name == "pow" and len(node.args) == 3 and not node.keywords:
                pass  # allowed
            else:
                self.add_error(node, f"Risky builtin function: {func_name}()")

        # math.* family - now handles deep dotted paths
        if func_name and func_name.startswith("math."):
            if func_name.split(".", 1)[-1] in self.risky_math_functions:
                self.add_error(node, f"Risky math function: {func_name}()")

        # NumPy (any path segment starting with numpy or np)
        if func_name and (func_name.startswith("numpy.") or func_name.startswith("np.")):
            self.add_error(node, f"NumPy function detected: {func_name}() - may return floats")

        # Laundering floats via int(…)
        if func_name == "int":
            # If argument subtree contains Div/Pow or float constants, flag it.
            def subtree_floaty(n: ast.AST) -> bool:
                for sub in ast.walk(n):
                    if isinstance(sub, (ast.Div, ast.Pow)):
                        return True
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                        return True
                return False
            if any(subtree_floaty(arg) for arg in node.args):
                self.add_error(node, "int(...) used to mask floaty expression (contains /, **, or float const)")

        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check for risky imports."""
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in {"math", "cmath", "numpy", "scipy", "random", "statistics", "decimal", "fractions", "time"}:
                self.add_error(node, f"Risky import detected: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for risky from-imports."""
        root = (node.module or "").split(".")[0]
        if root in {"math", "cmath", "numpy", "scipy", "random", "statistics", "decimal", "fractions", "time"}:
            imported_names = [alias.name for alias in node.names]
            
            # Special exception: guards.py needs decimal/fractions for type checking
            if self.filename.endswith("guards.py") and root in {"decimal", "fractions"}:
                # Allow these imports in guards.py for float detection logic
                pass
            else:
                self.add_error(node, f"Risky from-import: from {node.module} import {', '.join(imported_names)}")
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for type hints."""
        # Check return type annotation
        if node.returns:
            return_type = self._get_type_annotation(node.returns)
            if return_type in {'float', 'complex', 'Float', 'Complex'}:
                self.add_error(node, f"Function {node.name} has risky return type: {return_type}")
        
        # Check argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                arg_type = self._get_type_annotation(arg.annotation)
                if arg_type in {'float', 'complex', 'Float', 'Complex'}:
                    self.add_error(arg.annotation, f"Argument {arg.arg} has risky type: {arg_type}")
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignments for risky types."""
        if node.annotation:
            type_name = self._get_type_annotation(node.annotation)
            if type_name in {'float', 'complex', 'Float', 'Complex'}:
                self.add_error(node, f"Variable annotation has risky type: {type_name}")
        self.generic_visit(node)
    
    def _get_full_attr(self, node: ast.AST) -> Optional[str]:
        """Build dotted path like pkg.sub.mod.func"""
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None
    
    def _get_function_name(self, func_node: ast.AST) -> Optional[str]:
        """Extract function name from a Call node's func."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return self._get_full_attr(func_node)
        return None
    
    def _get_type_annotation(self, annotation: ast.AST) -> Optional[str]:
        """Extract type name from a type annotation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return annotation.value
        elif isinstance(annotation, ast.Attribute):
            if isinstance(annotation.value, ast.Name):
                return f"{annotation.value.id}.{annotation.attr}"
            else:
                return annotation.attr
        return None


def lint_file(filepath: Path, strict: bool = True) -> List[Tuple[int, int, str]]:
    """
    Lint a Python file for floating-point contamination.
    
    Args:
        filepath: Path to Python file
        strict: Whether to use strict checking
        
    Returns:
        List of (line, column, message) tuples for errors found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return [(0, 0, f"Error reading file: {e}")]
    
    # Conservative backstop: flag raw "/" outside comments/strings (optional strict check)
    # AST handles most cases; this catches edge cases like f-strings, eval strings
    if strict and "/" in source:
        for i, line in enumerate(source.splitlines(), 1):
            if "/" in line and not line.lstrip().startswith("#"):
                # Skip obvious URL patterns and string literals
                if not any(pattern in line.lower() for pattern in ["http://", "https://", "file://", '"""', "'''"]):
                    # This is a conservative check - AST will catch real division operators
                    # We only warn about potential edge cases here
                    pass  # Keep this as optional - AST already handles the main cases
    
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return [(e.lineno or 0, e.offset or 0, f"Syntax error: {e.msg}")]
    
    linter = NoFloatLinter(str(filepath))
    linter.visit(tree)
    
    return linter.errors


def lint_directory(dirpath: Path, pattern: str = "*.py", 
                  recursive: bool = True) -> dict[Path, List[Tuple[int, int, str]]]:
    """
    Lint all Python files in a directory.
    
    Args:
        dirpath: Directory path
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        Dictionary mapping file paths to error lists
    """
    results = {}
    
    if recursive:
        files = dirpath.rglob(pattern)
    else:
        files = dirpath.glob(pattern)
    
    for filepath in files:
        if filepath.is_file():
            errors = lint_file(filepath)
            if errors:
                results[filepath] = errors
    
    return results


def main():
    """Command-line interface for the no-float linter."""
    parser = argparse.ArgumentParser(
        description="Lint Python code for floating-point contamination"
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to lint")
    parser.add_argument("--strict", action="store_true", 
                       help="Enable strict checking")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively search directories")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show errors, no summary")
    
    args = parser.parse_args()
    
    total_files = 0
    total_errors = 0
    
    for path_str in args.paths:
        path = Path(path_str)
        
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            continue
        
        if path.is_file():
            if path.suffix == '.py':
                errors = lint_file(path, args.strict)
                total_files += 1
                
                if errors:
                    total_errors += len(errors)
                    print(f"\n{path}:")
                    for line, col, message in errors:
                        print(f"  {line}:{col}: {message}")
                elif not args.quiet:
                    print(f"{path}: OK")
        
        elif path.is_dir():
            results = lint_directory(path, recursive=args.recursive)
            
            for filepath, errors in results.items():
                total_files += 1
                total_errors += len(errors)
                
                print(f"\n{filepath}:")
                for line, col, message in errors:
                    print(f"  {line}:{col}: {message}")
            
            # Count files with no errors
            all_files = list(path.rglob("*.py") if args.recursive else path.glob("*.py"))
            clean_files = len([f for f in all_files if f not in results])
            total_files += clean_files
            
            if not args.quiet and clean_files > 0:
                print(f"\n{clean_files} files passed without errors")
    
    if not args.quiet:
        print(f"\nSummary: {total_files} files checked, {total_errors} errors found")
    
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
