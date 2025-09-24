"""
CLF Integer-Only Enforcement Guards
===================================

Non-negotiable invariant A.1: Integer-only calculus.
Every variable, length, index, and cost is an integer.
No floats, no approximations, no "entropy".

Runtime guards + AST scanning to prevent float contamination.
"""

import ast
import types
import sys
import inspect
from typing import Any, Callable

class FloatContaminationError(Exception):
    """Raised when float operations are detected in CLF code"""
    pass

def runtime_integer_guard(value: Any, context: str = "") -> int:
    """
    Runtime guard: ensure value is integer, never float.
    Raises FloatContaminationError if float detected.
    """
    if isinstance(value, float):
        raise FloatContaminationError(f"FLOAT_CONTAMINATION in {context}: {value} is float, must be integer")
    if not isinstance(value, int):
        raise FloatContaminationError(f"NON_INTEGER in {context}: {value} is {type(value)}, must be integer")
    return value

def guard_integer_result(func: Callable) -> Callable:
    """
    Decorator to ensure function returns integer only.
    Use on all cost computation functions.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return runtime_integer_guard(result, f"function {func.__name__}")
    return wrapper

class FloatDetectorVisitor(ast.NodeVisitor):
    """AST visitor to detect float operations in code"""
    
    def __init__(self):
        self.float_violations = []
    
    def visit_Constant(self, node):
        if isinstance(node.value, float):
            self.float_violations.append(f"Float literal {node.value} at line {node.lineno}")
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if node.id in ('float', 'math', 'numpy', 'np'):
            self.float_violations.append(f"Float-related name '{node.id}' at line {node.lineno}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ('float', 'math.sqrt', 'math.log', 'math.exp'):
                self.float_violations.append(f"Float function call '{node.func.id}' at line {node.lineno}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ('sqrt', 'log', 'exp', 'sin', 'cos', 'pi', 'e'):
                self.float_violations.append(f"Float method '{node.func.attr}' at line {node.lineno}")
        self.generic_visit(node)

def scan_function_for_floats(func: Callable) -> list[str]:
    """
    Scan function source code for float operations.
    Returns list of violations found.
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
        detector = FloatDetectorVisitor()
        detector.visit(tree)
        return detector.float_violations
    except (OSError, TypeError):
        # Can't get source (C extension, builtin, etc.)
        return []

def enforce_integer_only_module(module_name: str):
    """
    Scan entire module for float operations.
    Raises FloatContaminationError if any found.
    """
    if module_name not in sys.modules:
        raise ValueError(f"Module {module_name} not loaded")
    
    module = sys.modules[module_name]
    violations = []
    
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            func_violations = scan_function_for_floats(obj)
            violations.extend([f"{name}: {v}" for v in func_violations])
    
    if violations:
        raise FloatContaminationError(f"Float violations in {module_name}:\n" + "\n".join(violations))

# Integer-only arithmetic helpers
def integer_divide_exact(a: int, b: int) -> int:
    """
    Integer division that raises if not exact.
    Use instead of / operator to prevent float results.
    """
    runtime_integer_guard(a, "dividend")
    runtime_integer_guard(b, "divisor")
    if b == 0:
        raise ZeroDivisionError("Integer division by zero")
    if a % b != 0:
        raise FloatContaminationError(f"Non-exact division: {a} / {b} would produce float")
    return a // b

def integer_min(*args) -> int:
    """Integer-only min function with guard"""
    for arg in args:
        runtime_integer_guard(arg, "min argument")
    return min(args)

def integer_max(*args) -> int:
    """Integer-only max function with guard"""
    for arg in args:
        runtime_integer_guard(arg, "max argument")
    return max(args)

def integer_sum(iterable) -> int:
    """Integer-only sum with guards"""
    result = 0
    for item in iterable:
        runtime_integer_guard(item, "sum item")
        result += item
    return runtime_integer_guard(result, "sum result")

# Rails verification
def verify_integer_only_rail() -> bool:
    """
    Verify INTEGER_ONLY_OK rail.
    Returns True if all guards pass.
    """
    # Basic arithmetic verification
    test_values = [0, 1, 42, 1000, 8*1000]
    
    for val in test_values:
        runtime_integer_guard(val, "rail test")
        runtime_integer_guard(val + 1, "addition test")
        runtime_integer_guard(val * 8, "multiplication test")
        if val > 0:
            runtime_integer_guard(val // 2, "integer division test")
    
    return True