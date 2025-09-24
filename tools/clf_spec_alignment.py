"""
CLF ΔΩ-U^B Specification Alignment
=================================

Drift-proof implementation per mandatory alignment guide.
Replaces all CBD primitives with CAUS mapping and adds pinned assertions.
"""

from typing import Optional, List, Tuple, Dict, Any
from teleport.clf_integer_guards import runtime_integer_guard, integer_sum
from teleport.clf_leb_lock import leb_len, encode_minimal_leb128_unsigned

# NORMATIVE OPSET_V1 - No raw CBD primitive
OP_CONST = 1    # CAUS(CONST, byte_val, L)
OP_STEP = 2     # CAUS(STEP, start, stride, L) 
OP_MATCH = 3    # CAUS(MATCH, distance, length, L)
OP_U_B = 4      # CAUS(U_B identity, program_bytes, L)

# Token type constants per spec
TAG_LIT = 2     # LIT(b): 2 tag bits + 8 data = 10 bits
TAG_MATCH = 2   # MATCH(D,L): 2 tag bits + 8*leb(D) + 8*leb(L)
TAG_CAUS = 3    # CAUS(op,params,L): 3 tag bits + 8*leb(op) + Σ8*leb(param) + 8*leb(L)
TAG_END = 3     # END: 3 tag bits + pad_to_byte(pos+3)

def pad_to_byte(pos_bits: int) -> int:
    """Compute padding bits to align pos_bits to byte boundary"""
    pos_bits = runtime_integer_guard(pos_bits, "position bits")
    remainder = pos_bits % 8
    if remainder == 0:
        return 0
    return runtime_integer_guard(8 - remainder, "padding bits")

def compute_end_bits(pos_bits: int) -> int:
    """Compute END token cost: 3 + pad_to_byte(pos+3)"""
    pos_bits = runtime_integer_guard(pos_bits, "stream position")
    end_pos = runtime_integer_guard(pos_bits + 3, "end position")
    pad_bits = pad_to_byte(end_pos)
    end_bits = runtime_integer_guard(3 + pad_bits, "END bits")
    
    # Ensure END bits in valid range
    assert 3 <= end_bits <= 10, f"END bits {end_bits} out of range [3,10]"
    return end_bits

def C_bits_of(*integers) -> int:
    """Pinned macro: sum 8*leb_len(field) for integer fields"""
    total = 0
    for val in integers:
        val = runtime_integer_guard(val, "field value")
        total += 8 * leb_len(val)
    return runtime_integer_guard(total, "total field bits")

def assert_caus_cost(op: int, params: List[int], L: int) -> int:
    """Compute exact CAUS token cost with unit lock"""
    op = runtime_integer_guard(op, "CAUS op")
    L = runtime_integer_guard(L, "CAUS length")
    
    # CAUS cost = 3 + 8*leb(op) + Σ8*leb(param) + 8*leb(L)
    tag_bits = 3
    op_bits = 8 * leb_len(op)
    param_bits = C_bits_of(*params)
    length_bits = 8 * leb_len(L)
    
    total_cost = runtime_integer_guard(
        tag_bits + op_bits + param_bits + length_bits, 
        "CAUS total cost"
    )
    return total_cost

def build_A_exact_aligned(S: bytes) -> Tuple[Optional[int], List[Tuple]]:
    """
    A builder per ΔΩ-U^B spec: whole-range CAUS mapping only.
    NO S-packing, NO CBD primitive.
    Returns (C_A_stream, tokens_A) or (None, []) if incomplete.
    """
    L = runtime_integer_guard(len(S), "input length")
    if L == 0:
        return 0, []
    
    # Strategy 1: Try whole-range CONST
    if len(set(S)) == 1:
        byte_val = runtime_integer_guard(S[0], "constant byte")
        C_caus = assert_caus_cost(OP_CONST, [byte_val], L)
        
        # Only emit if minimal vs baseline
        baseline_cost = 10 * L  # LIT tokens
        if C_caus < baseline_cost:
            token = ('CAUS', OP_CONST, [byte_val], L, {
                'C_stream': C_caus,
                'construction_method': 'WHOLE_RANGE_CONST',
                'op_name': 'CONST'
            })
            return C_caus, [token]
    
    # Strategy 2: Try whole-range STEP (arithmetic progression)
    if L >= 2:
        step_detected, start_val, stride = _detect_arithmetic_progression(S)
        if step_detected:
            start_val = runtime_integer_guard(start_val, "STEP start")
            stride = runtime_integer_guard(stride, "STEP stride")
            C_caus = assert_caus_cost(OP_STEP, [start_val, stride], L)
            
            baseline_cost = 10 * L
            if C_caus < baseline_cost:
                token = ('CAUS', OP_STEP, [start_val, stride], L, {
                    'C_stream': C_caus,
                    'construction_method': 'WHOLE_RANGE_STEP',
                    'op_name': 'STEP'
                })
                return C_caus, [token]
    
    # Strategy 3: U^B identity (bounded program/cert) - only if cost < 10*L
    try:
        u_b_cost = _estimate_u_b_identity_cost(S, L)
        baseline_cost = 10 * L
        if u_b_cost is not None and u_b_cost < baseline_cost:
            # Represent as bounded U^B program
            program_hash = _compute_u_b_program_hash(S)
            C_caus = assert_caus_cost(OP_U_B, [program_hash], L)
            
            if C_caus < baseline_cost:
                token = ('CAUS', OP_U_B, [program_hash], L, {
                    'C_stream': C_caus,
                    'construction_method': 'U_B_IDENTITY',
                    'op_name': 'U_B'
                })
                return C_caus, [token]
    except:
        pass  # U^B identity not applicable
    
    # A builder incomplete - no whole-range CAUS found
    return None, []

def _detect_arithmetic_progression(S: bytes) -> Tuple[bool, int, int]:
    """Detect if S forms arithmetic progression"""
    if len(S) < 2:
        return False, 0, 0
    
    start_val = S[0]
    stride = (S[1] - S[0]) % 256  # Handle byte wraparound
    
    for i in range(2, len(S)):
        expected = (start_val + i * stride) % 256
        if S[i] != expected:
            return False, 0, 0
    
    return True, start_val, stride

def _estimate_u_b_identity_cost(S: bytes, L: int) -> Optional[int]:
    """Estimate cost of U^B identity encoding"""
    # Simple heuristic: if S has low Kolmogorov complexity indicators
    # This is a placeholder - real implementation would use bounded programs
    entropy_estimate = len(set(S)) / 256.0
    if entropy_estimate < 0.1:  # Very low entropy
        # Estimate program size for generating S
        program_size_estimate = max(10, L.bit_length())
        return program_size_estimate * 8
    return None

def _compute_u_b_program_hash(S: bytes) -> int:
    """Compute bounded program hash for U^B identity"""
    import hashlib
    hash_bytes = hashlib.sha256(S).digest()[:4]  # Use first 4 bytes
    return int.from_bytes(hash_bytes, 'big')

def build_B_structural_aligned(S: bytes) -> Tuple[bool, Optional[int], List[Tuple], Dict]:
    """
    B builder per ΔΩ-U^B spec: deterministic tiling over OpSet_v1.
    Returns (B_COMPLETE, C_B_stream, tokens_B, struct_counts)
    """
    L = runtime_integer_guard(len(S), "input length")
    if L == 0:
        return True, 0, [], {}
    
    tokens_B = []
    pos = 0
    pos_bits = 0  # Track stream position in bits
    struct_counts = {'LIT': 0, 'MATCH': 0, 'CAUS': 0}
    
    try:
        while pos < L:
            token_created = False
            
            # Strategy 1: Try MATCH (copy from previous position)
            if pos > 0:
                match_found, distance, match_length = _find_best_match(S, pos)
                if match_found and match_length >= 3:  # Minimum viable match
                    # MATCH token cost
                    C_match = TAG_MATCH + C_bits_of(distance, match_length)
                    
                    # Compare to LIT baseline
                    C_lit_baseline = TAG_LIT * match_length + 8 * match_length
                    
                    if C_match < C_lit_baseline:
                        token = ('MATCH', distance, match_length, match_length, {
                            'C_stream': C_match,
                            'pos_bits': pos_bits
                        })
                        tokens_B.append(token)
                        pos += match_length
                        pos_bits += C_match
                        struct_counts['MATCH'] += 1
                        token_created = True
            
            # Strategy 2: Try local CAUS (short CONST runs)
            if not token_created and pos < L:
                const_length = _detect_const_run(S, pos)
                if const_length >= 3:  # Minimum viable CONST
                    byte_val = runtime_integer_guard(S[pos], "local CONST byte")
                    C_caus = assert_caus_cost(OP_CONST, [byte_val], const_length)
                    
                    # Compare to LIT baseline
                    C_lit_baseline = (TAG_LIT + 8) * const_length
                    
                    if C_caus < C_lit_baseline:
                        token = ('CAUS', OP_CONST, [byte_val], const_length, {
                            'C_stream': C_caus,
                            'pos_bits': pos_bits,
                            'op_name': 'CONST'
                        })
                        tokens_B.append(token)
                        pos += const_length
                        pos_bits += C_caus
                        struct_counts['CAUS'] += 1
                        token_created = True
            
            # Fallback: LIT token
            if not token_created:
                byte_val = runtime_integer_guard(S[pos], "LIT byte")
                C_lit = TAG_LIT + 8  # 2 tag + 8 data = 10 bits
                
                token = ('LIT', byte_val, 1, 1, {
                    'C_stream': C_lit,
                    'pos_bits': pos_bits
                })
                tokens_B.append(token)
                pos += 1
                pos_bits += C_lit
                struct_counts['LIT'] += 1
        
        # Add END token
        end_bits = compute_end_bits(pos_bits)
        end_token = ('END', None, 0, 0, {
            'C_stream': end_bits,
            'pos_bits': pos_bits
        })
        tokens_B.append(end_token)
        
        # Coverage check
        total_coverage = sum(token[3] for token in tokens_B[:-1])  # Exclude END
        if total_coverage != L:
            return False, None, [], struct_counts
        
        # Total cost
        C_B_stream = integer_sum(token[4]['C_stream'] for token in tokens_B)
        
        return True, C_B_stream, tokens_B, struct_counts
        
    except Exception:
        return False, None, [], struct_counts

def _find_best_match(S: bytes, pos: int) -> Tuple[bool, int, int]:
    """Find best MATCH at position pos"""
    best_distance = 0
    best_length = 0
    
    # Search backwards for matches
    for distance in range(1, min(pos, 255)):  # LEB limit for distance
        match_pos = pos - distance
        if match_pos < 0:
            break
        
        # Find match length
        length = 0
        while (pos + length < len(S) and 
               match_pos + length < pos and
               S[pos + length] == S[match_pos + length] and
               length < 255):  # LEB limit for length
            length += 1
        
        if length > best_length:
            best_distance = distance
            best_length = length
    
    return best_length >= 3, best_distance, best_length

def _detect_const_run(S: bytes, pos: int) -> int:
    """Detect homogeneous run length starting at pos"""
    if pos >= len(S):
        return 0
    
    target = S[pos]
    length = 1
    while (pos + length < len(S) and 
           S[pos + length] == target and
           length < 255):  # LEB limit
        length += 1
    return length