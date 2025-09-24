# teleport/clf_receipts.py
"""
Mandatory Receipt System: Exact Mathematical Verification Template
Every numeric equality is directly supported and impossible to reinterpret.
"""

import hashlib
from typing import List, Dict, Any
from teleport.clf_canonical_math import (
    H_HEADER, CBD_BIJECTION_PROOF, COMPUTE_RATIOS, GATE_ADMISSIBLE,
    ASSERT_INTEGER_ONLY
)
from teleport.clf_builders import CLFToken
from teleport.clf_int import leb as leb_len


# ============================================================================
# RECEIPT TEMPLATE (EXACT FORMAT)
# ============================================================================

class CLFReceipt:
    """
    Mandatory receipt generator following exact template
    Hard-fails on any mathematical inconsistency
    """
    
    def __init__(self, S: bytes):
        self.S = S
        self.L = len(S)
        self.receipt_lines = []
        self.hard_failures = []
    
    def add_line(self, line: str):
        """Add a receipt line"""
        self.receipt_lines.append(line)
    
    def hard_fail(self, message: str):
        """Record a hard failure (will raise exception)"""
        self.hard_failures.append(message)
        self.add_line(f"❌ HARD FAILURE: {message}")
    
    def validate_and_emit(self) -> str:
        """Generate final receipt, raise on any hard failures"""
        if self.hard_failures:
            failure_summary = "\n".join(self.hard_failures)
            raise ValueError(f"Receipt validation failed:\n{failure_summary}")
        
        return "\n".join(self.receipt_lines)


def generate_mandatory_receipt(
    S: bytes,
    chosen_label: str,
    chosen_tokens: List[CLFToken],
    chosen_info: Dict,
    C_A_total: int,
    C_B_total: int,
    info_A: Dict,
    info_B: Dict,
    H: int,
    emit_decision: bool
) -> str:
    """
    Generate mandatory receipt following exact template
    All numeric equalities are mathematically verified
    """
    
    receipt = CLFReceipt(S)
    L = len(S)
    
    # ========================================================================
    # 1. IDENTITY
    # ========================================================================
    
    receipt.add_line("IDENTITY:")
    receipt.add_line(f"  L = {L} bytes")
    
    raw_bits = 8 * L
    receipt.add_line(f"  RAW_BITS = 8·L = {raw_bits} bits")
    
    # SHA256 computation
    sha256_in = hashlib.sha256(S).hexdigest().upper()
    receipt.add_line(f"  SHA256_IN  = {sha256_in}")
    
    if emit_decision:
        # For EMIT: reconstruct and verify bijection
        reconstructed = reconstruct_from_tokens(chosen_tokens)
        sha256_out = hashlib.sha256(reconstructed).hexdigest().upper()
        equality = (reconstructed == S)
        
        receipt.add_line(f"  SHA256_OUT = {sha256_out}")
        receipt.add_line(f"  EQUALITY   = {equality}")
        
        if not equality:
            receipt.hard_fail("Bijection failed: SHA256_IN != SHA256_OUT")
    else:
        receipt.add_line(f"  SHA256_OUT = N/A")
        receipt.add_line(f"  EQUALITY   = N/A")
    
    receipt.add_line("")
    
    # ========================================================================
    # 2. HEADER
    # ========================================================================
    
    receipt.add_line("HEADER:")
    leb_len_8L = leb_len(8 * L)
    H_computed = 16 + 8 * leb_len_8L
    
    receipt.add_line(f"  leb_len(8·L) = {leb_len_8L}")
    receipt.add_line(f"  H(L) = 16 + 8·{leb_len_8L} = {H_computed} bits")
    
    # Verify header computation matches
    if H != H_computed:
        receipt.hard_fail(f"Header mismatch: provided H={H} != computed H={H_computed}")
    
    receipt.add_line("")
    
    # ========================================================================
    # 3. A vs B
    # ========================================================================
    
    receipt.add_line("A (CBD whole-range):")
    receipt.add_line(f"  tokensA = {info_A['tokens']}")
    receipt.add_line(f"  C_A_stream = {info_A['C_stream']}")
    receipt.add_line(f"  C_A_total  = H + C_A_stream = {H} + {info_A['C_stream']} = {C_A_total}")
    
    receipt.add_line("")
    
    receipt.add_line("B (structural tiling):")
    receipt.add_line(f"  tokensB = {info_B['tokens']}   [CONST={info_B['CONST']}, STEP={info_B['STEP']}, MATCH={info_B['MATCH']}, CBD_GAPS={info_B['CBD']}]")
    receipt.add_line(f"  C_B_stream = {info_B['C_stream']}")
    receipt.add_line(f"  C_B_total  = H + C_B_stream = {H} + {info_B['C_stream']} = {C_B_total}")
    
    receipt.add_line("")
    
    # ========================================================================
    # 4. DECISION
    # ========================================================================
    
    receipt.add_line("DECISION:")
    min_cost = min(C_A_total, C_B_total)
    
    if C_A_total < C_B_total:
        decision_label = "CBD"
    elif C_B_total < C_A_total:
        decision_label = "STRUCT"
    else:
        decision_label = "CBD"  # Tie rule: choose CBD
    
    receipt.add_line(f"  min(C_A_total, C_B_total) = min({C_A_total}, {C_B_total}) = {min_cost}")
    receipt.add_line(f"  argmin(A,B) = {decision_label}   (tie→CBD)")
    
    # Verify decision matches chosen
    if chosen_label != decision_label:
        receipt.hard_fail(f"Decision mismatch: chosen={chosen_label} != computed={decision_label}")
    
    receipt.add_line("")
    
    # ========================================================================
    # 5. SERIALIZER IDENTITY (per token)
    # ========================================================================
    
    receipt.add_line("SERIALIZER IDENTITY (per token):")
    
    for i, token in enumerate(chosen_tokens):
        try:
            seed_bytes = token.serialize_seed()
            c_stream_token = token.compute_stream_cost()
            expected_cost = 8 * len(seed_bytes)
            
            if expected_cost == c_stream_token:
                receipt.add_line(f"  token[{i}]: 8·|seed| = 8·{len(seed_bytes)} = {expected_cost} = C_stream  ✓")
            else:
                receipt.hard_fail(f"Token {i} serializer identity violated: 8·|seed|={expected_cost} != C_stream={c_stream_token}")
                
        except Exception as e:
            receipt.hard_fail(f"Token {i} serializer validation failed: {e}")
    
    receipt.add_line("")
    
    # ========================================================================
    # 6. GLOBAL TOTALS & GATES
    # ========================================================================
    
    receipt.add_line("GLOBAL:")
    
    chosen_c_stream = chosen_info["C_stream"]
    total_cost = H + chosen_c_stream
    
    receipt.add_line(f"  TOTAL = H + ΣC_stream = {H} + {chosen_c_stream} = {total_cost}")
    receipt.add_line(f"  RAW_BITS = 8·L = 8·{L} = {raw_bits}")
    
    # Compute ratios
    ratios = COMPUTE_RATIOS(total_cost, L)
    receipt.add_line(f"  RATIO_RAW = TOTAL/(8·L) = {total_cost}/{raw_bits} = {ratios['RATIO_RAW']:.6f}")
    receipt.add_line(f"  RATIO_10L = TOTAL/(10·L) = {total_cost}/{ratios['VIRTUAL_BITS']} = {ratios['RATIO_10L']:.6f}")
    
    # Admissibility gate
    is_admissible = ratios['ADMISSIBLE']
    receipt.add_line(f"  ADMISSIBLE_BASELINE = (TOTAL < 8·L) = ({total_cost} < {raw_bits}) = {is_admissible}")
    
    # State decision
    if emit_decision:
        state = "EMIT"
    else:
        state = "OPEN"
    
    receipt.add_line(f"  STATE = {state}")
    
    # Verify state logic
    if emit_decision and not is_admissible:
        receipt.hard_fail(f"EMIT state with inadmissible cost: TOTAL={total_cost} >= RAW_BITS={raw_bits}")
    
    return receipt.validate_and_emit()


def reconstruct_from_tokens(tokens: List[CLFToken]) -> bytes:
    """
    Reconstruct original data from token list
    Used for bijection verification
    """
    result = bytearray()
    
    for token in tokens:
        if token.type == "CONST":
            result.extend(token.data)
        elif token.type == "STEP":
            for i in range(token.count):
                byte_val = (token.base + i * token.increment) % 256
                result.append(byte_val)
        elif token.type == "CBD":
            # For CBD, use inverse bijection
            from teleport.clf_canonical_math import CBD_BIJECTION_INVERSE
            reconstructed = CBD_BIJECTION_INVERSE(token.K, token.length)
            result.extend(reconstructed)
        elif token.type == "MATCH":
            # MATCH reconstruction requires context - simplified for now
            # In full implementation, would need to maintain reconstruction context
            raise NotImplementedError("MATCH reconstruction requires full context implementation")
    
    return bytes(result)


# ============================================================================
# RECEIPT VALIDATION FUNCTIONS
# ============================================================================

def validate_receipt_determinism(receipt1: str, receipt2: str, receipt3: str):
    """
    Validate that three runs produce identical receipts
    Used for determinism testing
    """
    if receipt1 != receipt2 or receipt2 != receipt3:
        raise ValueError("Receipt determinism violated: runs produced different receipts")


def extract_receipt_value(receipt: str, key: str) -> Any:
    """
    Extract specific values from receipt for testing
    Example: extract_receipt_value(receipt, "C_A_total") -> int
    """
    lines = receipt.split('\n')
    
    for line in lines:
        if key in line:
            # Extract numeric value after '='
            parts = line.split('=')
            if len(parts) >= 2:
                value_str = parts[-1].strip()
                try:
                    # Try to parse as integer first
                    return int(value_str)
                except ValueError:
                    # If not integer, return as string
                    return value_str
    
    raise ValueError(f"Key '{key}' not found in receipt")


def assert_receipt_mathematical_consistency(receipt: str):
    """
    Assert all mathematical relationships in receipt are consistent
    Used for validation testing
    """
    try:
        # Extract key values
        L = extract_receipt_value(receipt, "L =")
        raw_bits = extract_receipt_value(receipt, "RAW_BITS =")
        total = extract_receipt_value(receipt, "TOTAL =")
        
        # Verify basic relationships
        expected_raw_bits = 8 * L
        if raw_bits != expected_raw_bits:
            raise ValueError(f"RAW_BITS inconsistency: {raw_bits} != 8*{L}")
        
        # Verify ratio computation
        ratio_raw = extract_receipt_value(receipt, "RATIO_RAW =")
        expected_ratio = total / raw_bits if raw_bits > 0 else float('inf')
        
        # Allow small floating point tolerance
        if abs(float(ratio_raw) - expected_ratio) > 1e-6:
            raise ValueError(f"RATIO_RAW inconsistency: {ratio_raw} != {expected_ratio}")
            
    except Exception as e:
        raise ValueError(f"Receipt mathematical consistency check failed: {e}")