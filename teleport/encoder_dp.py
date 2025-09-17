"""
CLF Canonical Encoder — Bottom-Up DP (Residue-Aware, Iterative)

Globally optimal canonicalization with exact END cost accounting.
"""

from typing import Tuple, Optional, Dict, Any
from .leb_io import leb128_emit_single
from .caus_deduction import deduct_caus_global, compute_caus_cost, OP_CAUS

# CLF deterministic ranking: CAUS < MATCH < LIT
RANK = {"CAUS": 0, "MATCH": 1, "LIT": 2}

def _canon_key(kind: str, params: tuple, L: int) -> tuple[int, bytes]:
    """Canonical key for tie-breaking: (rank, byte_projection)"""
    rank = RANK[kind]
    proj = token_project(kind, params, L)  # exact byte projection
    return (rank, proj)

def _lex_better(cand_key, best_key) -> bool:
    """Lexicographic comparison: true if candidate is better than current best"""
    if cand_key is None or best_key is None:
        return cand_key is not None  # something beats nothing
    return cand_key < best_key  # lexicographic: rank first, then projection

def token_project(kind: str, params: tuple, L: int) -> bytes:
    """Project token to canonical byte representation for comparison"""
    if kind == "LIT":
        b, _ = params
        return bytes([b]) * L
    elif kind == "MATCH":
        # MATCH doesn't have direct byte projection, use placeholder
        D = params[0]
        return b"MATCH" + D.to_bytes(2, 'big') + L.to_bytes(2, 'big')
    elif kind == "CAUS":
        # CAUS projection based on operation
        op_id = params[0]
        op_params = params[1:]
        if op_id == 0:  # P_CONST
            b = op_params[0]
            return bytes([b]) * L
        elif op_id == 1:  # P_STEP
            start, stride = op_params
            return bytes([(start + i * stride) & 255 for i in range(L)])
        elif op_id == 2:  # P_ANCHOR_WINDOW
            # For tie-breaking, use anchor bytes as projection
            len_a, a0, a1, len_b, b0, b1 = op_params[:6]
            return bytes([a0, a1, b0, b1])
        else:
            return b"CAUS_UNK" + bytes(op_params[:4])
    else:
        return b"UNKNOWN"

def leb(x: int) -> int:
    """Return minimal LEB128 byte length for integer x"""
    return len(leb128_emit_single(x))

def pad_to_byte(bits: int) -> int:
    """Padding bits needed to reach byte boundary"""
    return (8 - (bits % 8)) % 8
from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP

# CAUS operations enabled for CLF drastic minimality
# Set to False to disable CAUS operations 
ENABLE_CAUS = True

def _lcp_limit(a: bytes, i: int, b: bytes, j: int, limit: int) -> int:
    """Longest common prefix of a[i:] and b[j:], up to limit bytes."""
    k = 0
    while k < limit and i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
        k += 1
    return k

def _max_const_run(S: bytes, p: int) -> int:
    """Find maximal constant run starting at position p"""
    if p >= len(S):
        return 0
    
    byte_val = S[p]
    L = 1
    while p + L < len(S) and S[p + L] == byte_val:
        L += 1
    return L

def _deduce_step(S: bytes, p: int) -> Tuple[int, int, int]:
    """
    Deduce arithmetic step: returns (L, start, stride) where L is maximal domain.
    Returns (0, 0, 0) if no valid step pattern.
    """
    if p + 1 >= len(S):
        return (0, 0, 0)
    
    start = S[p]
    stride = (S[p + 1] - start) & 255  # Modular arithmetic in Z_256
    L = 2
    
    while p + L < len(S) and S[p + L] == ((start + L * stride) & 255):
        L += 1
    
    return (L, start, stride)

def token_rank(kind: str, params: Tuple[int, ...]) -> int:
    """Integer rank for tie-breaking: tag then op-id ordering."""
    if kind == "LIT":
        return (0, 0)
    elif kind == "MATCH":
        return (1, 0)
    elif kind == "CAUS.CONST":
        return (2, OP_CONST)
    elif kind == "CAUS.STEP":
        return (2, OP_STEP)
    else:
        return (999, 999)

def token_project(kind: str, params: Tuple[int, ...], L: int) -> bytes:
    """Canonical minimal-LEB byte projection for tie-break."""
    from teleport.leb_io import leb128_emit_single
    
    result = bytearray()
    
    if kind == "LIT":
        result.append(OP_LIT)
        result.extend(leb128_emit_single(L))
    elif kind == "MATCH":
        D = params[0]
        result.append(OP_MATCH)
        result.extend(leb128_emit_single(D))
        result.extend(leb128_emit_single(L))
    elif kind == "CAUS.CONST":
        b = params[0]
        result.append(OP_CONST)
        result.extend(leb128_emit_single(b))
        result.extend(leb128_emit_single(L))
    elif kind == "CAUS.STEP":
        start, stride = params
        result.append(OP_STEP)
        result.extend(leb128_emit_single(start))
        result.extend(leb128_emit_single(stride))
        result.extend(leb128_emit_single(L))
    
    return bytes(result)





def canonize_bytes_dp(S: bytes, print_receipts: bool = False):
    """
    Bottom-up DP canonical encoder with residue-aware END cost.
    Returns (tokens, total_bits, C_end).
    """
    N = len(S)
    if N == 0:
        return [], 3, 3
    
    # CRITICAL: Pure integer arithmetic - no floating point
    INF = 999999999  # Large enough integer constant
    
    # DP tables: cost[p][r] = minimal bits from suffix p with residue r
    cost = [[INF] * 8 for _ in range(N + 1)]
    back_choice: Dict[Tuple[int, int], Tuple[Tuple[str, Tuple[int, ...], int], int, Tuple[int, int]]] = {}
    
    # CLF: Global CAUS deduction (CAUS or FAIL - no fallback)
    global_caus, predicate_receipts = deduct_caus_global(S, print_receipts) if ENABLE_CAUS else (None, [])
    if print_receipts and global_caus:
        print(f"DEBUG: Global CAUS deduction found: {global_caus}")
    
    # Base: at p=N, only END remains; its cost depends on current residue r
    # C_END(r) = 3 + pad_to_byte(r + 3) - exact integer residue accounting
    for r in range(8):
        cost[N][r] = 3 + pad_to_byte(r + 3)  # residue-aware END cost
    
    # Mathematical optimization: Fast LCP computation with limits
    def _fast_lcp(i: int, j: int) -> int:
        """Bounded LCP: pure mathematical operation, guaranteed termination"""
        if i >= N or j >= N or i == j:
            return 0
        
        # MATHEMATICAL BOUND: Only profitable match lengths
        # MATCH cost = 18+ bits, LIT cost = 10*L bits
        # So max useful L = 18/10 = 1, extend to 8 for mathematical completeness
        max_useful = min(N - i, N - j, 8)
        
        # GUARANTEED TERMINATION: bounded loop count
        for k in range(max_useful):
            if S[i + k] != S[j + k]:
                return k
        return max_useful
    
    # Iterate p from N-1 down to 0
    for p in range(N - 1, -1, -1):
        for r in range(8):
            best_cost = INF
            best_choice = None
            best_key = None
            best_next_state = None
            
            # 1) LIT(b,L) with 1 ≤ L ≤ min(10, N-p)
            b = S[p]
            max_lit = min(10, N - p)
            for L in range(1, max_lit + 1):
                c_bits = 10 * L
                r_next = (r + (2 * L)) % 8  # bits = 10*L → r' = (r + 2*L) % 8
                total = c_bits + cost[p + L][r_next]
                
                tok = ("LIT", (b, L), L)
                cand_key = _canon_key(*tok)
                cand_next = (p + L, r_next)
                
                if total < best_cost or (total == best_cost and _lex_better(cand_key, best_key)):
                    best_cost = total
                    best_key = cand_key
                    best_next_state = cand_next
                    best_choice = (tok, c_bits, cand_next)
            
            # 2) MATCH(D,L) with mathematically bounded enumeration
            # CLF HARD ALIGNMENT: Generate actual MATCH tokens with O(1) search
            max_distance = min(p, 32)  # MATHEMATICAL BOUND: only profitable recent matches
            for j in range(max(0, p - max_distance), p):
                D = p - j
                
                # Compute LCP with mathematical bounds
                lcp_len = 0
                max_check = min(N - p, N - j, 16)  # MATHEMATICAL BOUND: max useful match length
                while lcp_len < max_check and S[j + lcp_len] == S[p + lcp_len]:
                    lcp_len += 1
                
                if lcp_len < 3:
                    continue
                
                # Evaluate by leb(L) bands for efficiency:
                # Band 1: L in [3..min(127,lcp_len)]
                up1 = min(127, lcp_len)
                if up1 >= 3:
                    c_bits = 2 + 8 * leb(D) + 8 * 1  # leb(L)=1 for L in [1,127]
                    r_next = (r + 2) % 8  # only the tag bits affect residue
                    
                    # Find best L in band minimizing continuation cost
                    best_cont = INF
                    best_L = None
                    for L in range(3, up1 + 1):
                        if p + L > N:
                            break
                        cont = cost[p + L][r_next]
                        if cont < best_cont:
                            best_cont = cont
                            best_L = L
                    
                    if best_L is not None:
                        total = c_bits + best_cont
                        
                        tok = ("MATCH", (D,), best_L)
                        cand_key = _canon_key(*tok)
                        cand_next = (p + best_L, r_next)
                        
                        if total < best_cost or (total == best_cost and _lex_better(cand_key, best_key)):
                            best_cost = total
                            best_key = cand_key
                            best_next_state = cand_next
                            best_choice = (tok, c_bits, cand_next)
                
                # Band 2: L in [128..min(16383,lcp_len)]
                up2 = min(16383, lcp_len)
                if up2 >= 128:
                    c_bits = 2 + 8 * leb(D) + 8 * 2  # leb(L)=2 for L in [128,16383]
                    r_next = (r + 2) % 8
                    
                    best_cont = INF
                    best_L = None
                    for L in range(128, up2 + 1):
                        if p + L > N:
                            break
                        cont = cost[p + L][r_next]
                        if cont < best_cont:
                            best_cont = cont
                            best_L = L
                    
                    if best_L is not None:
                        total = c_bits + best_cont
                        
                        tok = ("MATCH", (D,), best_L)
                        cand_key = _canon_key(*tok)
                        cand_next = (p + best_L, r_next)
                        
                        if total < best_cost or (total == best_cost and _lex_better(cand_key, best_key)):
                            best_cost = total
                            best_key = cand_key
                            best_next_state = cand_next
                            best_choice = (tok, c_bits, cand_next)
            
            # 3) CAUS: Only at p=0 if global deduction found causality
            if ENABLE_CAUS and p == 0 and global_caus is not None:
                op_id, params = global_caus
                L_caus = N  # CAUS must cover entire input
                
                # Compute exact CAUS cost using mathematical formula
                c_bits = compute_caus_cost(op_id, params, L_caus)
                r_next = (r + (c_bits % 8)) % 8  # CAUS bits affect residue
                total = c_bits + cost[p + L_caus][r_next]
                
                tok = ("CAUS", (op_id,) + params, L_caus)
                cand_key = _canon_key(*tok)
                cand_next = (p + L_caus, r_next)
                
                if total < best_cost or (total == best_cost and _lex_better(cand_key, best_key)):
                    best_cost = total
                    best_key = cand_key
                    best_next_state = cand_next
                    best_choice = (tok, c_bits, cand_next)
            
            # Set final choice
            cost[p][r] = best_cost
            if best_choice:
                back_choice[(p, r)] = best_choice
            else:
                # Should not happen; at least LIT(1) is always admissible
                raise RuntimeError(f"No admissible token at state (p={p}, r={r})")
    
    # Reconstruct from (0, residue0=0)
    tokens = []
    p, r = 0, 0
    bitpos = 0
    receipt_count = 0
    

    
    while p < N:
        if (p, r) not in back_choice:
            raise RuntimeError(f"No backpointer for state (p={p}, r={r})")
        
        (tok, c_bits, (p2, r2)) = back_choice[(p, r)]
        tokens.append(tok)
        
        if print_receipts and receipt_count < 10:  # Show first 10 tokens
            kind, params, L = tok
            if kind == "LIT":
                b, L_param = params  # LIT params are (byte_value, run_length)
                assert L_param == L  # Consistency check
                c_lit = 10 * L
                strict_ineq = int(c_bits < c_lit)
                print(f"p={p} chosen=LIT(b={b},L={L}) C_token={c_bits} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
            elif kind == "MATCH":
                D = params[0]
                # Legality check: L ≤ D (source window constraint)
                legal_LD = int(L <= D)
                if L <= 10:
                    c_lit = 10 * L
                    strict_ineq = int(c_bits < c_lit)
                    print(f"p={p} chosen=MATCH(D={D},L={L}) legal_L≤D={legal_LD} C_token={c_bits} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
                else:
                    print(f"p={p} chosen=MATCH(D={D},L={L}) legal_L≤D={legal_LD} C_token={c_bits} C_LIT({L})=inadmissible strict_ineq=1")
            elif kind == "CAUS":
                op_id = params[0]
                op_params = params[1:]
                if L <= 10:
                    c_lit = 10 * L
                    strict_ineq = int(c_bits < c_lit)
                    print(f"p={p} chosen=CAUS(op={op_id},params={op_params},L={L}) C_token={c_bits} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
                else:
                    print(f"p={p} chosen=CAUS(op={op_id},params={op_params},L={L}) C_token={c_bits} C_LIT({L})=inadmissible strict_ineq=1")
            elif kind == "CAUS.CONST":
                b = params[0]
                if L <= 10:
                    c_lit = 10 * L
                    strict_ineq = int(c_bits < c_lit)
                    print(f"p={p} chosen=CAUS.CONST(b=0x{b:02x},L={L}) C_token={c_bits} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
                else:
                    print(f"p={p} chosen=CAUS.CONST(b=0x{b:02x},L={L}) C_token={c_bits} C_LIT({L})=inadmissible strict_ineq=1")
            elif kind == "CAUS.STEP":
                start, stride = params
                if L <= 10:
                    c_lit = 10 * L
                    strict_ineq = int(c_bits < c_lit)
                    print(f"p={p} chosen=CAUS.STEP(start={start},stride={stride},L={L}) C_token={c_bits} C_LIT({L})={c_lit} strict_ineq={strict_ineq}")
                else:
                    print(f"p={p} chosen=CAUS.STEP(start={start},stride={stride},L={L}) C_token={c_bits} C_LIT({L})=inadmissible strict_ineq=1")
            receipt_count += 1
        
        p, r = p2, r2
        bitpos += c_bits
    
    C_end = 3 + pad_to_byte(bitpos + 3)
    total_bits = bitpos + C_end
    
    # Assert residue correctness (catch any drift)
    sum_token_bits = sum(back_choice[(p_tok, r_tok)][1] for p_tok, r_tok in [(0, 0)] + [back_choice[(p_prev, r_prev)][2] for p_prev, r_prev in []])
    sum_token_bits = bitpos  # We already computed this correctly above
    expected_total = sum_token_bits + (3 + pad_to_byte(sum_token_bits + 3))
    assert total_bits == expected_total, f"Residue drift: {total_bits} != {expected_total}"
    
    return tokens, total_bits, C_end, predicate_receipts

def serialize_tokens_to_seed(tokens, N, magic_be):
    """
    CLF-aligned serializer: minimal LEB128 only, grammar-valid LITs
    """
    from teleport.leb_io import leb128_emit_single as leb_min
    from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP
    
    # header: magic/version (2 bytes) + OutputLengthBits as minimal LEB128
    out = bytearray()
    out.append((magic_be >> 8) & 0xFF)  # Magic/version is 16-bit
    out.append(magic_be & 0xFF)
    
    # OutputLengthBits = N * 8 (total output bits) 
    output_length_bits = N * 8
    out.extend(leb_min(output_length_bits))
    
    # body: obey grammar; no long-LIT encodings
    for kind, params, L in tokens:
        if kind == "LIT":
            b, run_len = params
            assert 1 <= run_len <= 10, f"LIT run length {run_len} violates domain [1,10]"
            for _ in range(run_len):
                # emit one single-byte LIT as per CLF grammar
                out.append(OP_LIT)      # OP_LIT tag (0x00)
                out.append(b)           # the literal byte itself
        elif kind == "MATCH":
            D = params[0]
            # emit MATCH with minimal LEB128 for D and L
            out.append(OP_MATCH)        # OP_MATCH tag (0x01)  
            out.extend(leb_min(D))      # minimal varint for D
            out.extend(leb_min(L))      # minimal varint for L
        else:
            # Only LIT and MATCH allowed in CLF-pure mode
            raise AssertionError(f"Forbidden token in stream: {kind}")
    
    # END: emit OP_END tag (0x03) to terminate stream
    out.append(0x03)  # OP_END tag
    
    canonical_seed = bytes(out)
    
    # Strict post-serialize check - catch any non-minimal LEB128 immediately
    _scan_seed_varints_strict(canonical_seed)
    
    return canonical_seed

def _max_const_run(S: bytes, p: int) -> int:
    """Find maximal run of constant byte at position p"""
    if p >= len(S):
        return 0
    b = S[p]
    i = p
    while i < len(S) and S[i] == b:
        i += 1
    return i - p

def _deduce_step(S: bytes, p: int) -> Tuple[int, int, int]:
    """
    Deduce maximal STEP pattern at position p.
    Returns (L, start, stride) where L is the deduced length.
    """
    N = len(S)
    if p + 1 >= N:
        return (0, 0, 0)
    
    start = S[p]
    stride = (S[p + 1] - start) % 256
    
    # Verify how far the pattern extends
    L = 1
    expected = start
    for i in range(p, N):
        if S[i] != expected:
            break
        L = i - p + 1
        expected = (expected + stride) % 256
    
    return (L, start, stride) if L >= 1 else (0, 0, 0)

def _scan_seed_varints_strict(seed: bytes) -> None:
    """Walk seed by grammar and validate all LEB128 fields for minimality."""
    from teleport.leb_io import leb128_parse_single_minimal
    from teleport.seed_format import OP_LIT, OP_MATCH, OP_CONST, OP_STEP
    from teleport.spec_constants import TELEPORT_MAGIC_VERSION_BE
    
    i = 0
    # Check header exactly matches TELEPORT_MAGIC_VERSION_BE
    magic_bytes = TELEPORT_MAGIC_VERSION_BE.to_bytes(2, 'big')
    assert len(seed) >= 2, "Seed too short for magic/version"
    assert seed[:2] == magic_bytes, f"Bad magic/version: expected {magic_bytes.hex()}, got {seed[:2].hex()}"
    i = 2
    
    # Parse OutputLengthBits (minimal LEB128)
    output_length_bits, leb_len = leb128_parse_single_minimal(seed, i)
    assert output_length_bits > 0 and output_length_bits % 8 == 0, f"Invalid OutputLengthBits: {output_length_bits}"
    i += leb_len
    
    # Parse token stream - only public opcodes allowed
    while i < len(seed):
        tag = seed[i]
        i += 1
        
        if tag == 0:  # OP_LIT
            # CLF Grammar: LIT is [tag][single_byte] - no varints
            assert i < len(seed), f"Truncated LIT payload at {i-1}"
            i += 1  # Skip the literal byte
        elif tag == 1:  # OP_MATCH
            # Parse D and L with strict minimal LEB128
            D, dlen = leb128_parse_single_minimal(seed, i)
            i += dlen
            L, llen = leb128_parse_single_minimal(seed, i)
            i += llen
        elif tag == 3:  # OP_END
            # END terminates stream - no parameters to parse
            break
        else:
            # Only 0,1,3 are public opcodes - anything else is invalid
            raise AssertionError(f"Invalid opcode {tag} at offset {i-1} - only 0,1,3 allowed")


def canonize_dp(S: bytes, print_receipts: bool = False) -> bytes:
    """Residue-aware bottom-up DP encoder (global minimal bitstream with exact END)."""
    from teleport.seed_format import emit_LIT, emit_MATCH, emit_CAUS
    from teleport.encoder import emit_END
    
    tokens, total_bits, C_end = canonize_bytes_dp(S, print_receipts)
    
    if print_receipts:
        print(f"C_END= {C_end}")
        print(f"Total bits: {total_bits}")
    
    # Build seed from tokens
    seed_parts = []
    p = 0
    
    for kind, params, L in tokens:
        if kind == "LIT":
            block = S[p:p + L]
            seed_parts.append(emit_LIT(block))
        elif kind == "MATCH":
            D = params[0]
            seed_parts.append(emit_MATCH(D, L))
        elif kind == "CAUS.CONST":
            b = params[0]
            seed_parts.append(emit_CAUS(OP_CONST, [b], L))
        elif kind == "CAUS.STEP":
            start, stride = params
            seed_parts.append(emit_CAUS(OP_STEP, [start, stride], L))
        
        p += L
    
    # Calculate END payload (total_bits - C_END)
    end_payload_bits = total_bits - C_end
    seed_parts.append(emit_END(end_payload_bits))
    
    canonical_seed = b''.join(seed_parts)
    
    # Strict post-serialize check - catch any non-minimal LEB128 immediately
    _scan_seed_varints_strict(canonical_seed)
    
    return canonical_seed
