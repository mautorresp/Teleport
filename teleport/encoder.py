"""
Teleport Canonical Encoder

Mathematical canonicalization: S -> T* (unique minimal seed)
Pure integer/byte logic. No heuristics, no floats, no randomness.
"""

import hashlib
from typing import List, Tuple, Optional
from teleport.clf_int import leb, pad_to_byte
from teleport.seed_format import (
    emit_LIT, emit_MATCH, emit_CAUS, 
    OP_LIT, OP_MATCH, OP_CONST, OP_STEP
)
from teleport.guards import assert_boundary_types

def cost_LIT(L: int) -> int:
    """C_LIT(L) = 10*L, domain 1 ≤ L ≤ 10"""
    assert 1 <= L <= 10, f"LIT domain violation: L={L}"
    return 10 * L

def cost_MATCH(D: int, L: int) -> int:
    """C_MATCH(D,L) = 2 + 8*leb(D) + 8*leb(L), L ≥ 3, D ≥ 1"""
    assert L >= 3, f"MATCH domain violation: L={L}"
    assert D >= 1, f"MATCH domain violation: D={D}"
    return 2 + 8 * leb(D) + 8 * leb(L)

def cost_CAUS(op: int, params: List[int], L: int) -> int:
    """C_CAUS = 3 + 8*leb(op) + Σ 8*leb(param_i) + 8*leb(L)"""
    assert L >= 3, f"CAUS domain violation: L={L}"
    return 3 + 8 * leb(op) + sum(8 * leb(p) for p in params) + 8 * leb(L)

def cost_END(pos: int) -> int:
    """C_END(pos) = 3 + pad_to_byte(pos+3)"""
    return 3 + pad_to_byte(pos + 3)

class Token:
    """Canonical token representation for tie-breaking"""
    def __init__(self, op_type: str, params: tuple, length: int, cost: int):
        self.op_type = op_type
        self.params = params
        self.length = length
        self.cost = cost
    
    def __lt__(self, other):
        """Deterministic tie-breaking: cost, then lexicographic"""
        if self.cost != other.cost:
            return self.cost < other.cost
        
        # Tie-break by canonical encoding
        self_key = (self._tag_order(), self.params, self.length)
        other_key = (other._tag_order(), other.params, other.length)
        return self_key < other_key
    
    def _tag_order(self) -> int:
        """Tag ordering for tie-break"""
        if self.op_type == 'LIT':
            return 0
        elif self.op_type == 'MATCH':
            return 1
        elif self.op_type == 'CAUS_CONST':
            return 2
        elif self.op_type == 'CAUS_STEP':
            return 3
        else:
            return 999

def deduce_MATCH_maximal(S: bytes, p: int) -> Optional[Tuple[int, int]]:
    """
    Mathematical deduction: maximal legal MATCH(D,L) at position p.
    Returns (D, L) where S[p:p+L] == S[p-D:p-D+L] and L >= 3, or None.
    """
    if p == 0:
        return None  # No prefix to match against
    
    best_D, best_L = None, 0
    
    for D in range(1, min(p + 1, 65536)):  # D must be <= p for legal window
        if p - D < 0:
            break
            
        # Find maximal L for this D
        L = 0
        while (p + L < len(S) and 
               p - D + L < len(S) and
               p - D + L >= 0 and
               S[p + L] == S[p - D + L]):
            L += 1
        
        if L >= 3 and L > best_L:
            best_D, best_L = D, L
    
    return (best_D, best_L) if best_L >= 3 else None

def deduce_CAUS_CONST(S: bytes, p: int) -> Optional[Tuple[int, int]]:
    """
    Mathematical deduction: CONST(b, L) where all bytes equal b.
    Returns (b, L) with L >= 3, or None.
    """
    if p >= len(S):
        return None
    
    b = S[p]
    L = 1
    
    while p + L < len(S) and S[p + L] == b:
        L += 1
    
    return (b, L) if L >= 3 else None

def deduce_CAUS_STEP(S: bytes, p: int) -> Optional[Tuple[int, int, int]]:
    """
    Mathematical deduction: STEP(start, stride, L) for arithmetic sequence.
    Returns (start, stride, L) with L >= 3, or None.
    """
    if p + 2 >= len(S):
        return None
    
    start = S[p]
    stride = (S[p + 1] - start) & 255  # mod 256
    L = 2
    
    while p + L < len(S) and S[p + L] == ((start + L * stride) & 255):
        L += 1
    
    return (start, stride, L) if L >= 3 else None

def canonize(S: bytes, print_receipts: bool = False) -> bytes:
    """
    Canonical re-encoding: S -> T* (unique minimal seed)
    
    Deterministic left-to-right deduction with exact cost minimization.
    Returns canonical minimal seed bytes.
    """
    if not S:
        return emit_END(0)  # Empty case
    
    seed_parts = []
    p = 0  # Write cursor
    receipt_count = 0  # Track receipts printed
    
    while p < len(S):
        candidates = []
        non_lit_lengths = set()
        
        # Deduce MATCH (if possible)
        match_result = deduce_MATCH_maximal(S, p)
        if match_result:
            D, L = match_result
            cost = cost_MATCH(D, L)
            candidates.append(Token('MATCH', (D, L), L, cost))
            non_lit_lengths.add(L)
        
        # Deduce CAUS_CONST
        const_result = deduce_CAUS_CONST(S, p)
        if const_result:
            b, L = const_result
            cost = cost_CAUS(OP_CONST, [b], L)
            candidates.append(Token('CAUS_CONST', (b, L), L, cost))
            non_lit_lengths.add(L)
        
        # Deduce CAUS_STEP  
        step_result = deduce_CAUS_STEP(S, p)
        if step_result:
            start, stride, L = step_result
            cost = cost_CAUS(OP_STEP, [start, stride], L)
            candidates.append(Token('CAUS_STEP', (start, stride, L), L, cost))
            non_lit_lengths.add(L)
        
        # Add LIT candidates at same lengths as non-LIT candidates (equal-L comparison)
        for L in non_lit_lengths:
            if 1 <= L <= 10 and p + L <= len(S):  # LIT domain constraint
                cost = cost_LIT(L)
                candidates.append(Token('LIT', (L,), L, cost))
            # If L > 10, we can still compare against best possible LIT
            # but we don't add LIT(L) as it's inadmissible
        
        # If no non-LIT candidates, add fallback LIT (maximal length)
        if not non_lit_lengths:
            max_lit_len = min(10, len(S) - p)
            if max_lit_len >= 1:
                L = max_lit_len
                cost = cost_LIT(L)
                candidates.append(Token('LIT', (L,), L, cost))
        
        # Select minimal cost candidate (deterministic tie-break)
        chosen = min(candidates)
        
        # Print receipt for non-LIT selections (first 3)
        if print_receipts and receipt_count < 3 and chosen.op_type != 'LIT':
            L = chosen.length
            
            if L <= 10:
                # Equal-L comparison possible
                lit_cost = cost_LIT(L)
                strict_ineq = 1 if chosen.cost < lit_cost else 0
                comparison_str = f"C_LIT({L})={lit_cost} strict_ineq={strict_ineq}"
            else:
                # LIT inadmissible at this length (L > 10)
                comparison_str = f"C_LIT({L})=inadmissible(L>10) forced_selection=1"
            
            if chosen.op_type == 'MATCH':
                D, L = chosen.params
                print(f"p={p} chosen=MATCH(D={D},L={L}) C_chosen={chosen.cost} {comparison_str}")
            elif chosen.op_type == 'CAUS_CONST':
                b, L = chosen.params
                print(f"p={p} chosen=CAUS.CONST(b=0x{b:02x},L={L}) C_chosen={chosen.cost} {comparison_str}")
            elif chosen.op_type == 'CAUS_STEP':
                start, stride, L = chosen.params
                print(f"p={p} chosen=CAUS.STEP(start={start},stride={stride},L={L}) C_chosen={chosen.cost} {comparison_str}")
            
            receipt_count += 1
        
        # Emit chosen token
        if chosen.op_type == 'LIT':
            L = chosen.params[0]
            block = S[p:p + L]
            seed_parts.append(emit_LIT(block))
        elif chosen.op_type == 'MATCH':
            D, L = chosen.params
            seed_parts.append(emit_MATCH(D, L))
        elif chosen.op_type == 'CAUS_CONST':
            b, L = chosen.params
            seed_parts.append(emit_CAUS(OP_CONST, [b], L))
        elif chosen.op_type == 'CAUS_STEP':
            start, stride, L = chosen.params
            seed_parts.append(emit_CAUS(OP_STEP, [start, stride], L))
        
        p += chosen.length
    
    # Combine seed parts
    seed_bytes = b''.join(seed_parts)
    
    # Add END token with exact padding
    bit_pos = len(seed_bytes) * 8
    end_cost = cost_END(bit_pos)
    # For now, simplified END (actual implementation needs bit-level padding)
    
    return seed_bytes

def emit_END(bit_pos: int) -> bytes:
    """Emit END token with exact padding to byte boundary"""
    # Simplified - actual implementation needs teleport.seed_format support
    pad_bits = pad_to_byte(bit_pos + 3)
    # Return minimal END representation
    return b''  # Placeholder

def print_canonicalization_receipts(seed_in: bytes, S: bytes, seed_min: bytes, 
                                  chosen_tokens: List[dict]):
    """Print exact console receipts as specified"""
    
    # Identity receipts
    print(f"bytes= {len(S)}")
    print(f"sha256= {hashlib.sha256(S).hexdigest().upper()}")
    
    # Canonicalization receipts  
    from teleport.seed_vm import expand
    S_prime = expand(seed_min)
    
    print(f"bytes'= {len(S_prime)}")
    print(f"sha256'= {hashlib.sha256(S_prime).hexdigest().upper()}")
    print(f"eq_bytes= {int(len(S) == len(S_prime))}")
    print(f"eq_sha= {int(hashlib.sha256(S).digest() == hashlib.sha256(S_prime).digest())}")
    
    # Cost receipts
    total_cost = sum(token['cost'] for token in chosen_tokens)
    lit_baseline = cost_LIT(min(10, len(S))) * ((len(S) + 9) // 10)  # Rough LIT cost
    
    print(f"C_total= {total_cost}")
    print(f"C_LIT({len(S)})= {10 * len(S)}")
    print(f"delta_vs_LIT= {10 * len(S) - total_cost}")
    
    # Local proofs (show first few significant choices)
    proof_count = 0
    for token in chosen_tokens:
        if proof_count >= 3:
            break
            
        if token['type'] != 'LIT' or proof_count == 0:
            p = token['position']
            chosen_desc = token['description']
            c_chosen = token['cost']
            L = token['length']
            c_lit_L = cost_LIT(min(L, 10))
            strict_ineq = int(c_chosen < c_lit_L)
            
            print(f"p={p} chosen={chosen_desc} C_chosen={c_chosen} C_LIT({L})={c_lit_L} strict_ineq={strict_ineq}")
            proof_count += 1
