#!/usr/bin/env python3
"""
CLF Causal Encoder - Deductive Mathematical Implementation

Pure deductive encoding using domain predicates:
- CAUS_CONST: b repeated L times (L ≥ 1)
- CAUS_STEP: arithmetic sequence start + i*stride (L ≥ 3) 
- MATCH: forward-copy equality S[p+L] == S[p+L-D] (L ≥ 3)
- LIT: fallback with strict domain 1 ≤ L ≤ 10
- Choice by strict cost inequalities against C_LIT = 10*L
"""

import argparse
import hashlib
import pathlib
from teleport.seed_format import emit_LIT, emit_MATCH, emit_CAUS, OP_CONST, OP_STEP
from teleport.costs import cost_caus
from teleport.clf_int import leb

def deduce_const(S: bytes, p: int):
    """
    Universal constant deduction: computes maximal domain of f(i) = b.
    Always returns mathematical result - never None.
    Returns ('CAUS_CONST', (b,), L_domain)
    """
    if p >= len(S):
        return ('CAUS_CONST', (0,), 0)  # Mathematical result: empty domain
    
    b = S[p]
    L = 1
    while p + L < len(S) and S[p + L] == b:
        L += 1
    
    return ('CAUS_CONST', (b,), L)  # Always return mathematical domain

def deduce_step(S: bytes, p: int):
    """
    Universal arithmetic step deduction: computes maximal domain of f(i) = (start + i*stride) mod 256.
    Always returns mathematical result - never None.
    Returns ('CAUS_STEP', (start, stride), L_domain)
    """
    if p + 1 >= len(S):
        # Mathematical result: domain too small for arithmetic sequence
        return ('CAUS_STEP', (S[p] if p < len(S) else 0, 0), 1 if p < len(S) else 0)
    
    start = S[p]
    stride = (S[p + 1] - start) & 255  # Modular arithmetic in Z_256
    L = 2
    
    while p + L < len(S) and S[p + L] == ((start + L * stride) & 255):
        L += 1
    
    return ('CAUS_STEP', (start, stride), L)  # Always return mathematical domain

def deduce_all_matches(S: bytes, p: int, W: int = 65536):
    """
    Universal MATCH deduction: computes mathematical domain for all D values.
    Returns complete list of ('MATCH', (D,), L_domain) - never filters by existence.
    Mathematical principle: Every D has a computable maximal domain L(D).
    """
    matches = []
    if p < 1:
        return matches  # Mathematical constraint: no backward reference possible
    
    Dmax = min(p, W)
    
    for D in range(1, Dmax + 1):
        # Universal computation: maximal domain L for distance D
        L = 0
        while (p + L < len(S) and 
               p + L - D >= 0 and
               S[p + L] == S[p + L - D]):
            L += 1
        
        # Always include mathematical result - domain L for distance D
        matches.append(('MATCH', (D,), L))
    
    return matches

def encode_causal_with_receipts(S: bytes) -> tuple[bytes, dict]:
    """
    Encode payload S using deductive causal operators.
    
    Algorithm:
    1. At each position p, deduce all legal patterns (CAUS_CONST, CAUS_STEP, MATCH)
    2. Price each candidate with exact CLF formulas
    3. Filter by strict inequalities against C_LIT = 10*L
    4. Choose minimal cost with deterministic tie-breaking
    5. Always include LIT(1..10) as fallback
    
    Returns (seed_bytes, receipts_dict).
    """
    tokens = []
    receipts = {
        'payload_bytes': len(S),
        'num_tokens': 0,
        'C_TOTAL': 0,
        'C_LIT': 10 * len(S),
        'sha256_payload': hashlib.sha256(S).hexdigest()
    }
    
    p = 0
    while p < len(S):
        candidates = []
        
        # Universal deduction: all operators computed mathematically
        candidates.append(deduce_const(S, p))
        candidates.append(deduce_step(S, p))
        
        # Universal MATCH deduction: all D values computed
        candidates.extend(deduce_all_matches(S, p, W=65536))
        
        # Always include LIT fallback with domain constraint 1..10
        L_lit = min(10, len(S) - p)
        candidates.append(('LIT', (), L_lit))
        
        # Mathematical pricing with domain filtering
        priced_candidates = []
        for op_class, params, L in candidates:
            # Skip degenerate domains (L=0) except for edge cases
            if L == 0 and op_class != 'LIT':
                continue
                
            if op_class == 'LIT':
                cost = 10 * L
            elif op_class == 'MATCH':
                # Only include MATCH with substantial domain L >= 3
                if L < 3:
                    continue
                D = params[0]
                cost = 2 + 8 * leb(D) + 8 * leb(L)
            elif op_class == 'CAUS_CONST':
                # Only include CONST with substantial domain L >= 3  
                if L < 3:
                    continue
                b = params[0]
                cost = cost_caus(0, [b], L)  # OP_CONST = 0
            elif op_class == 'CAUS_STEP':
                # Only include STEP with substantial domain L >= 3
                if L < 3:
                    continue
                start, stride = params
                cost = cost_caus(1, [start, stride], L)  # OP_STEP = 1
            else:
                continue  # Unknown operation
            
            priced_candidates.append((cost, L, op_class, params))
        
        # Mathematical filtering: strict inequality cost < 10*L for non-LIT
        filtered = []
        for cost, L, op_class, params in priced_candidates:
            if op_class == 'LIT':
                filtered.append((cost, L, op_class, params))
            else:
                # Mathematical constraint: cost < 10*L (strict inequality)
                C_LIT_for_L = 10 * L
                if cost < C_LIT_for_L:  # Strict inequality admission criterion
                    filtered.append((cost, L, op_class, params))
        
        # Choose by: minimal cost → largest L → lexicographic (op_class, params)
        if not filtered:
            filtered = [(C_LIT_ref, L_lit, 'LIT', ())]  # Emergency fallback
        
        def sort_key(item):
            cost, L, op_class, params = item
            # Tie-breaking: cost, -L (larger L first), lexicographic ordering
            if op_class == 'MATCH':
                return (cost, -L, 0, params[0])  # Smallest D for MATCH
            elif op_class == 'LIT':
                return (cost, -L, 1, 0)
            elif op_class == 'CAUS_CONST':
                return (cost, -L, 2, params[0])
            elif op_class == 'CAUS_STEP':
                return (cost, -L, 3, params[0], params[1])
            else:
                return (cost, -L, 999, 0)
        
        best = min(filtered, key=sort_key)
        cost, L, op_class, params = best
        
        # Emit chosen operation
        if op_class == 'LIT':
            chunk = S[p:p + L]
            token = emit_LIT(chunk)
            print(f"emit LIT(L={L}) cost={cost}")
        elif op_class == 'MATCH':
            D = params[0]
            token = emit_MATCH(D, L)
            print(f"emit MATCH(D={D}, L={L}) cost={cost}  <  C_LIT={10*L}")
        elif op_class == 'CAUS_CONST':
            b = params[0]
            token = emit_CAUS(OP_CONST, [b], L)
            print(f"emit CAUS_CONST(b={b:02x}, L={L}) cost={cost}  <  C_LIT={10*L}")
        elif op_class == 'CAUS_STEP':
            start, stride = params
            token = emit_CAUS(OP_STEP, [start, stride], L)
            print(f"emit CAUS_STEP(start={start:02x}, stride={stride:02x}, L={L}) cost={cost}  <  C_LIT={10*L}")
        
        tokens.append(token)
        receipts['C_TOTAL'] += cost
        receipts['num_tokens'] += 1
        p += L
    
    seed = b''.join(tokens)
    receipts['seed_bytes'] = len(seed)
    
    return seed, receipts

def main():
    parser = argparse.ArgumentParser(description="CLF Causal Encoder")
    parser.add_argument("--payload", required=True, help="Path to payload file")
    args = parser.parse_args()
    
    # Read payload
    payload_path = pathlib.Path(args.payload)
    if not payload_path.exists():
        print(f"Payload file not found: {payload_path}")
        return 1
    
    payload = payload_path.read_bytes()
    
    # Encode with mathematical receipts
    seed, receipts = encode_causal_with_receipts(payload)
    
    # Write seed to test_artifacts
    pathlib.Path("test_artifacts").mkdir(exist_ok=True)
    seed_path = pathlib.Path("test_artifacts/seed_causal.bin")
    seed_path.write_bytes(seed)
    
    # Print mathematical receipts
    print(f"payload_bytes={receipts['payload_bytes']}")
    print(f"num_tokens={receipts['num_tokens']}")
    print(f"C_TOTAL={receipts['C_TOTAL']}")
    print(f"C_LIT={receipts['C_LIT']}")
    print(f"sha256_payload={receipts['sha256_payload']}")
    print(f"seed_bytes={receipts['seed_bytes']}")
    
    # Verify mathematical constraint: C_TOTAL <= C_LIT
    if receipts['C_TOTAL'] < receipts['C_LIT']:
        print("causal_advantage=1")
    else:
        print("causal_advantage=0")
    
    return 0

if __name__ == "__main__":
    exit(main())
