# tests/test_builder_audit.py
import ast
import pathlib
import re
from typing import List, Tuple

# Forbidden patterns that must not appear outside clf_fb.py
FORBIDDEN_PATTERNS = [
    r"\('CBD_LOGICAL'", r"\('CBD_BOUND'",
    r"OP_CONST[,)]", r"OP_STEP[,)]", r"OP_MATCH[,)]", r"OP_CBD256[,)]",
    r"C_CAUS", r"C_stream", r"C_op", r"C_params", r"C_L", r"C_END",
    r"emit_cbd_param_leb7", r"expand_cbd256\(",
    r"compose_cover\(", r"encode_CLF\(",
    r"compression", r"entropy", r"pattern.*detect", r"random.*pattern", 
    r"savings", r"ratio.*L", r"optimization"
]

# Files that are allowed to have forbidden patterns (exemptions)
EXEMPT_FILES = {
    "clf_fb.py",  # The sealed module itself
    "clf_canonical.py",  # Legacy module (will be cleaned up)
    "test_builder_audit.py",  # This test file
}

def find_forbidden_usage() -> List[Tuple[pathlib.Path, str, str]]:
    """Scan for forbidden patterns in Python files."""
    root = pathlib.Path("/Users/Admin/Teleport")
    violations = []
    
    for py_file in root.rglob("*.py"):
        if py_file.name in EXEMPT_FILES:
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Get line number
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content[line_start:content.find('\n', match.start())].strip()
                    violations.append((py_file, pattern, f"Line {line_num}: {line_content}"))
        except Exception as e:
            print(f"Warning: Could not scan {py_file}: {e}")
    
    return violations

def audit_direct_token_construction() -> List[Tuple[pathlib.Path, str]]:
    """Use AST to find direct tuple construction that looks like tokens."""
    root = pathlib.Path("/Users/Admin/Teleport")
    violations = []
    
    for py_file in root.rglob("*.py"):
        if py_file.name in EXEMPT_FILES:
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=str(py_file))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Tuple) and len(node.elts) == 5:
                    # Check if it looks like a token: (op, params, L, cost_info, pos)
                    if (isinstance(node.elts[0], (ast.Constant, ast.Name)) and
                        isinstance(node.elts[2], (ast.Constant, ast.Name))):  # L should be int
                        violations.append((py_file, f"Line {node.lineno}: Suspicious 5-tuple"))
                        
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
    
    return violations

def test_no_forbidden_patterns():
    """Test that no forbidden patterns exist outside exempt files."""
    violations = find_forbidden_usage()
    if violations:
        error_msg = "Forbidden CLF patterns found:\n"
        for file_path, pattern, context in violations:
            error_msg += f"  {file_path.relative_to(pathlib.Path('/Users/Admin/Teleport'))}: {pattern}\n    {context}\n"
        assert False, error_msg

def test_no_direct_token_construction():
    """Test that no direct token tuples are constructed outside exempt files."""
    violations = audit_direct_token_construction()
    if violations:
        error_msg = "Direct token construction found:\n"
        for file_path, context in violations:
            error_msg += f"  {file_path.relative_to(pathlib.Path('/Users/Admin/Teleport'))}: {context}\n"
        assert False, error_msg

def test_builder_import_usage():
    """Test that files using CLF functionality import from clf_fb."""
    root = pathlib.Path("/Users/Admin/Teleport")
    violations = []
    
    for py_file in root.rglob("*.py"):
        if py_file.name in EXEMPT_FILES or py_file.name.startswith("test_"):
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            
            # If file uses CLF functions but doesn't import from clf_fb
            if (re.search(r"encode_|compose_|Builder|Token", content) and
                not re.search(r"from teleport\.clf_fb import", content)):
                violations.append(py_file)
                
        except Exception as e:
            print(f"Warning: Could not scan {py_file}: {e}")
    
    if violations:
        error_msg = "Files using CLF without importing clf_fb:\n"
        for file_path in violations:
            error_msg += f"  {file_path.relative_to(pathlib.Path('/Users/Admin/Teleport'))}\n"
        assert False, error_msg

if __name__ == "__main__":
    print("=== CLF Builder Audit Report ===")
    
    print("\n1. Scanning for forbidden patterns...")
    violations = find_forbidden_usage()
    if violations:
        print(f"❌ Found {len(violations)} forbidden pattern violations:")
        for file_path, pattern, context in violations[:10]:  # Show first 10
            print(f"  {file_path.name}: {pattern}")
            print(f"    {context}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
    else:
        print("✅ No forbidden patterns found")
    
    print("\n2. Scanning for direct token construction...")
    token_violations = audit_direct_token_construction()
    if token_violations:
        print(f"❌ Found {len(token_violations)} direct token construction violations:")
        for file_path, context in token_violations[:5]:
            print(f"  {file_path.name}: {context}")
    else:
        print("✅ No direct token construction found")
    
    print("\n3. Summary:")
    total_violations = len(violations) + len(token_violations)
    if total_violations == 0:
        print("✅ Repository is CLF Function-Builder compliant")
    else:
        print(f"❌ Found {total_violations} total violations requiring migration")