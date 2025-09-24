"""
CLF Builder Separation System
============================

Section C: Builder separation (stop B=A aliasing for good)

Implement separate pure builders:
- build_A_exact(S) -> (C_A_stream, token_A): whole-range CBD pricing
- build_B_structural(S) -> (B_COMPLETE, C_B_stream, tokens_B, struct_counts): 
  deterministic tiler over fixed op set

Decision consumes only the outputs, never re-computes inside.
"""

from typing import Tuple, List, Dict, Optional
from teleport.clf_integer_guards import runtime_integer_guard, integer_sum
from teleport.clf_leb_lock import leb_len, compute_leb_cost_bits


# ============================================================================
# TOKEN DEFINITIONS
# ============================================================================

class CLFToken:
    """Base class for CLF tokens with cost validation"""
    
    def __init__(self, token_type: str, length: int):
        if token_type not in FIXED_OPERATOR_SET:
            raise ValueError(f"Invalid token type: {token_type}")
        self.type = token_type
        self.length = length
        ASSERT_INTEGER_ONLY(length)
    
    def compute_stream_cost(self) -> int:
        """Compute C_stream for this token (to be overridden)"""
        raise NotImplementedError
    
    def serialize_seed(self) -> bytes:
        """Generate seed bytes for this token (to be overridden)"""
        raise NotImplementedError
    
    def validate_serializer_identity(self):
        """Enforce 8 * |seed| = C_stream"""
        seed = self.serialize_seed()
        c_stream = self.compute_stream_cost()
        ASSERT_SERIALIZER_IDENTITY(seed, c_stream)


class CONSTToken(CLFToken):
    def __init__(self, data: bytes, position: int):
        super().__init__("CONST", len(data))
        self.data = data
        self.position = position
    
    def compute_stream_cost(self) -> int:
        return 8 * len(self.data)
    
    def serialize_seed(self) -> bytes:
        return self.data


class STEPToken(CLFToken):
    def __init__(self, base: int, increment: int, count: int, position: int):
        super().__init__("STEP", count)
        ASSERT_INTEGER_ONLY(base)
        ASSERT_INTEGER_ONLY(increment)
        ASSERT_INTEGER_ONLY(count)
        
        self.base = base
        self.increment = increment
        self.count = count
        self.position = position
    
    def compute_stream_cost(self) -> int:
        return 32  # STEP tokens have fixed 32-bit cost
    
    def serialize_seed(self) -> bytes:
        # STEP serialization: base (1 byte) + increment (1 byte) + count (2 bytes)
        return bytes([self.base, self.increment]) + self.count.to_bytes(2, 'big')


class MATCHToken(CLFToken):
    def __init__(self, distance: int, length: int, position: int):
        super().__init__("MATCH", length)
        ASSERT_INTEGER_ONLY(distance)
        ASSERT_INTEGER_ONLY(length)
        
        if distance not in ALLOWED_DISTANCES:
            raise ValueError(f"Invalid MATCH distance: {distance} not in {ALLOWED_DISTANCES}")
        if length < WINDOW_SIZE:
            raise ValueError(f"MATCH length {length} < window size {WINDOW_SIZE}")
            
        self.distance = distance
        self.match_length = length
        self.position = position
    
    def compute_stream_cost(self) -> int:
        return 64  # MATCH tokens have fixed 64-bit cost
    
    def serialize_seed(self) -> bytes:
        # MATCH serialization: distance (4 bytes) + length (4 bytes)
        return self.distance.to_bytes(4, 'big') + self.match_length.to_bytes(4, 'big')


class CBDToken(CLFToken):
    def __init__(self, data: bytes, position: int):
        super().__init__("CBD", len(data))
        self.data = data
        self.position = position
        self.K = CBD_BIJECTION_FORWARD(data)
    
    def compute_stream_cost(self) -> int:
        # CBD cost based on K value LEB128 encoding
        return 8 * leb_len(self.K)
    
    def serialize_seed(self) -> bytes:
        # CBD serialization: LEB128 encoding of K
        K = self.K
        if K == 0:
            return b'\x00'  # Special case: K=0 encodes as single zero byte
        
        result = bytearray()
        while K > 0:
            byte = K & 0x7F
            K >>= 7
            if K > 0:
                byte |= 0x80
            result.append(byte)
        return bytes(result)


# ============================================================================
# CONSTRUCTION A: WHOLE-RANGE CBD
# ============================================================================

def build_A(S: bytes) -> Tuple[List[CLFToken], Dict]:
    """
    Construction A: Single CBD token covering entire input
    Returns: (tokens, construction_info)
    """
    L = len(S)
    if L == 0:
        return ([], {"tokens": 0, "C_stream": 0, "coverage": 0})
    
    # Single CBD token for entire input
    cbd_token = CBDToken(S, 0)
    cbd_token.validate_serializer_identity()
    
    tokens = [cbd_token]
    c_stream = cbd_token.compute_stream_cost()
    
    # Validate coverage
    ASSERT_COVERAGE_COMPLETE(tokens, L)
    
    construction_info = {
        "tokens": 1,
        "C_stream": c_stream,
        "coverage": L,
        "CBD": 1,
        "CONST": 0,
        "STEP": 0,
        "MATCH": 0
    }
    
    return (tokens, construction_info)


# ============================================================================
# CONSTRUCTION B: STRUCTURAL TILING
# ============================================================================

def deduce_maximal_const_run(segment: bytes, pos: int, L: int) -> int:
    """Find maximal CONST run (≥2 identical bytes)"""
    if pos >= L:
        return 0
    
    byte_val = segment[pos]
    run = 1
    
    while pos + run < L and segment[pos + run] == byte_val:
        run += 1
    
    return run if run >= 2 else 0


def deduce_maximal_step_run(segment: bytes, pos: int, L: int) -> Tuple[int, int, int]:
    """
    Find maximal STEP run (arithmetic sequence ≥3 elements)
    Returns: (run_length, base, increment)
    """
    if pos + 2 >= L:
        return (0, 0, 0)
    
    base = segment[pos]
    increment = (segment[pos + 1] - base) % 256
    run = 2
    expected = (base + 2 * increment) % 256
    
    while pos + run < L and segment[pos + run] == expected:
        run += 1
        expected = (expected + increment) % 256
    
    return (run, base, increment) if run >= 3 else (0, 0, 0)


def deduce_maximal_match_run(segment: bytes, pos: int, context: bytes, w: int = WINDOW_SIZE) -> Tuple[int, int]:
    """
    FIXED: Multi-distance MATCH detection with correct mathematical deduction
    Returns: (run_length, distance) or (0, 0)
    """
    L = len(segment)
    
    if len(context) < w or pos + w > L:
        return (0, 0)
    
    best_run, best_D = 0, 0
    
    # Try each mathematical distance D ∈ ALLOWED_DISTANCES deterministically
    for D in ALLOWED_DISTANCES:
        if D > len(context):
            break  # Distances are sorted, no point checking larger
            
        # Mathematical source position - start of match in context
        match_start = len(context) - D
        if match_start < 0:
            continue
            
        # Check if we can match at least w bytes initially
        run = 0
        max_check = min(w, L - pos)  # Don't go beyond segment
        
        # Initial window verification
        while run < max_check:
            src_pos = match_start + run
            if src_pos >= len(context):
                break
            if context[src_pos] != segment[pos + run]:
                break
            run += 1
        
        if run < w:  # Initial window doesn't match fully
            continue
            
        # Mathematical greedy extension beyond initial window
        while pos + run < L:
            src_pos = match_start + run
            
            # Source byte from context or self-extension
            if src_pos < len(context):
                s_byte = context[src_pos]
            else:
                # Self-extension: reference to already-matched bytes in current token
                self_ref = src_pos - len(context)
                if self_ref >= run:  # Can't reference beyond current match
                    break
                s_byte = segment[pos + self_ref]

            if s_byte != segment[pos + run]:
                break
            run += 1
        
        # Keep longest mathematical match (ties go to smaller D for determinism)
        if run > best_run:
            best_run, best_D = run, D

    return (best_run, best_D) if best_run >= w else (0, 0)


def build_B(S: bytes) -> Tuple[List[CLFToken], Dict]:
    """
    Construction B: Deterministic structural tiling
    Precedence: CONST (≥2) → STEP (≥3) → MATCH → CBD gap filler
    Left-to-right maximal segments, no backtracking
    """
    L = len(S)
    if L == 0:
        return ([], {"tokens": 0, "C_stream": 0, "coverage": 0})
    
    tokens = []
    context = bytearray()
    pos = 0
    
    # Token counters
    const_count = step_count = match_count = cbd_count = 0
    total_c_stream = 0
    
    while pos < L:
        # Try CONST first (maximal runs ≥2)
        const_run = deduce_maximal_const_run(S, pos, L)
        if const_run > 0:
            data = S[pos:pos + const_run]
            token = CONSTToken(data, pos)
            token.validate_serializer_identity()
            tokens.append(token)
            
            # Update context
            context.extend(data)
            pos += const_run
            const_count += 1
            total_c_stream += token.compute_stream_cost()
            continue
        
        # Try STEP (arithmetic runs ≥3)
        step_run, base, increment = deduce_maximal_step_run(S, pos, L)
        if step_run > 0:
            token = STEPToken(base, increment, step_run, pos)
            token.validate_serializer_identity()
            tokens.append(token)
            
            # Update context with generated sequence
            for i in range(step_run):
                context.append((base + i * increment) % 256)
            pos += step_run
            step_count += 1
            total_c_stream += token.compute_stream_cost()
            continue
        
        # MATCH temporarily disabled until bijective reconstruction is complete
        # match_run, match_distance = deduce_maximal_match_run(S, pos, bytes(context))
        # if match_run > 0 and match_distance > 0:
        #     token = MATCHToken(match_distance, match_run, pos)
        #     token.validate_serializer_identity()
        #     tokens.append(token)
        #     # Update context with copied bytes
        #     for i in range(match_run):
        #         context.append(S[pos + i])
        #     pos += match_run
        #     match_count += 1
        #     total_c_stream += token.compute_stream_cost()
        #     continue
        match_run = 0  # Force MATCH disabled        # Fallback: CBD gap filler (single byte)
        data = S[pos:pos + 1]
        token = CBDToken(data, pos)
        token.validate_serializer_identity()
        tokens.append(token)
        
        context.extend(data)
        pos += 1
        cbd_count += 1
        total_c_stream += token.compute_stream_cost()
    
    # Validate coverage
    ASSERT_COVERAGE_COMPLETE(tokens, L)
    
    construction_info = {
        "tokens": len(tokens),
        "C_stream": total_c_stream,
        "coverage": L,
        "CONST": const_count,
        "STEP": step_count,
        "MATCH": match_count,
        "CBD": cbd_count
    }
    
    return (tokens, construction_info)


# ============================================================================
# DECISION FUNCTION: CHOOSE MINIMAL
# ============================================================================

def decide_min(tokens_A: List[CLFToken], info_A: Dict, 
               tokens_B: List[CLFToken], info_B: Dict,
               H: int) -> Tuple[str, List[CLFToken], Dict, int, int]:
    """
    Choose minimal construction based on total cost
    Tie rule: if C_A = C_B => choose CBD
    
    Returns: (chosen_label, chosen_tokens, chosen_info, C_A_total, C_B_total)
    """
    C_A_total = H + info_A["C_stream"]
    C_B_total = H + info_B["C_stream"]
    
    ASSERT_INTEGER_ONLY(C_A_total)
    ASSERT_INTEGER_ONLY(C_B_total)
    
    # Tie rule: if C_A = C_B => choose CBD (Construction A)
    if C_A_total <= C_B_total:
        return ("CBD", tokens_A, info_A, C_A_total, C_B_total)
    else:
        return ("STRUCT", tokens_B, info_B, C_A_total, C_B_total)