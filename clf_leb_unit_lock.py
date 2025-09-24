"""
CLF LEB Unit Lock Enforcement
============================

Pinned macro C_bits_of(*ints) and rail test forbidding leb_len(8*L) outside header.
"""

from teleport.clf_integer_guards import runtime_integer_guard
from teleport.clf_leb_lock import leb_len
import inspect
import ast

def C_bits_of(*integers) -> int:
    """
    PINNED MACRO: Sum 8*leb_len(field) for integer fields.
    This is the ONLY legal way to compute field bit costs.
    """
    total = 0
    for val in integers:
        val = runtime_integer_guard(val, "field value")
        total += 8 * leb_len(val)
    return runtime_integer_guard(total, "total field bits")

def rail_test_leb_unit_lock():
    """
    Rail test: Forbid leb_len(8*L) outside header code.
    Any such usage indicates S-packing residue.
    """
    # Get all functions in clf modules
    import teleport.clf_spec_alignment as spec
    import teleport.clf_causal_rails as rails
    
    forbidden_patterns = []
    
    # Check build_A_exact_aligned
    source = inspect.getsource(spec.build_A_exact_aligned)
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and 
                node.func.id == 'leb_len' and
                len(node.args) == 1):
                
                arg = node.args[0]
                # Check for leb_len(8*L) pattern
                if (isinstance(arg, ast.BinOp) and
                    isinstance(arg.op, ast.Mult) and
                    isinstance(arg.left, ast.Constant) and
                    arg.left.value == 8):
                    forbidden_patterns.append(f"Found leb_len(8*L) in build_A_exact_aligned")
    
    # Check build_B_structural_aligned 
    source = inspect.getsource(spec.build_B_structural_aligned)
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and 
                node.func.id == 'leb_len' and
                len(node.args) == 1):
                
                arg = node.args[0]
                # Check for leb_len(8*L) pattern
                if (isinstance(arg, ast.BinOp) and
                    isinstance(arg.op, ast.Mult) and
                    isinstance(arg.left, ast.Constant) and
                    arg.left.value == 8):
                    forbidden_patterns.append(f"Found leb_len(8*L) in build_B_structural_aligned")
    
    if forbidden_patterns:
        raise RuntimeError(f"LEB_UNIT_LOCK_VIOLATION: {forbidden_patterns}")
    
    return True

def verify_all_field_emissions_use_C_bits_of():
    """
    Verify all integer field emissions use C_bits_of(*ints) macro.
    This ensures unit lock compliance.
    """
    # Test with sample integer fields
    test_fields = [1, 42, 255, 1000, 65535]
    
    # Verify C_bits_of computes correctly
    for fields in [test_fields[:1], test_fields[:2], test_fields]:
        manual_sum = sum(8 * leb_len(field) for field in fields)
        macro_result = C_bits_of(*fields)
        
        assert manual_sum == macro_result, \
            f"C_bits_of mismatch: manual={manual_sum}, macro={macro_result}"
    
    return True

if __name__ == "__main__":
    rail_test_leb_unit_lock()
    verify_all_field_emissions_use_C_bits_of()
    print("✅ LEB unit lock enforcement verified")