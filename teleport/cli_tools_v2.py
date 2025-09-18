"""
Teleport OpSet_v2 - CLI Tools
Four CLI tools for real-file testing: scan, price, expand-verify, canonical
"""

import sys
import os
from pathlib import Path

# Add teleport package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teleport.opset_v2 import (
    OP_CONST, OP_STEP, OP_LCG8, OP_LFSR8, OP_REPEAT1, OP_ANCHOR, OP_CBD,
    compute_caus_cost_v2, is_admissible_v2
)
from teleport.predicates_v2 import PREDICATE_REGISTRY_V2
from teleport.leb_io import leb128_emit_single, leb128_parse_single

def teleport_scan(filename: str) -> tuple[bool, int, tuple, str]:
    """
    CLI Tool 1: teleport-scan filename
    
    Output format:
    SUCCESS op_id param1 param2 ... OR FAILURE reason
    
    Returns: (success, op_id, params, message)
    """
    try:
        with open(filename, 'rb') as f:
            F = f.read()
    except Exception as e:
        return False, -1, (), f"file_read_error {e}"
    
    # Try predicates in registry order (CONST through CBD)
    for op_id, predicate_func in PREDICATE_REGISTRY_V2:
        try:
            success, params, reason = predicate_func(F)
            if success:
                return True, op_id, params, f"op_{op_id} {reason}"
        except Exception as e:
            # Predicate failed - continue to next
            continue
    
    # Should never reach here since CBD always succeeds
    return False, -1, (), "no_predicate_succeeded_impossible"

def teleport_price(filename: str) -> tuple[bool, int, str]:
    """
    CLI Tool 2: teleport-price filename
    
    Output format:
    SUCCESS cost_integer OR FAILURE reason
    
    Returns: (success, cost, message)
    """
    success, op_id, params, scan_msg = teleport_scan(filename)
    if not success:
        return False, -1, f"scan_failed {scan_msg}"
    
    try:
        with open(filename, 'rb') as f:
            F = f.read()
        L = len(F)
        
        cost = compute_caus_cost_v2(op_id, params, L)
        return True, cost, f"cost={cost} for_op_{op_id}_L={L}"
        
    except Exception as e:
        return False, -1, f"cost_computation_error {e}"

def serialize_generator_v2(op_id: int, params: tuple, L: int) -> bytes:
    """
    Serialize generator to seed bytes using OpSet_v2 format:
    [LEB128(op_id)][LEB128(param1)]...[LEB128(paramN)][LEB128(L)]
    """
    seed = bytearray()
    
    # Encode op_id
    seed.extend(leb128_emit_single(op_id))
    
    # Encode parameters
    for param in params:
        seed.extend(leb128_emit_single(param))
    
    # Encode L
    seed.extend(leb128_emit_single(L))
    
    return bytes(seed)

def deserialize_generator_v2(seed_bytes: bytes) -> tuple[int, tuple, int]:
    """
    Deserialize generator from seed bytes.
    Returns: (op_id, params, L)
    """
    offset = 0
    
    # Decode op_id
    op_id, consumed = leb128_parse_single(seed_bytes, offset)
    offset += consumed
    
    # Determine number of parameters based on op_id
    param_counts = {
        OP_CONST: 1,      # (b)
        OP_STEP: 2,       # (start, stride)
        OP_LCG8: 3,       # (x0, a, c)
        OP_LFSR8: 2,      # (taps, seed)
        OP_REPEAT1: -1,   # Variable: (D, *motif)
        OP_ANCHOR: -1,    # Variable: (len_A, A..., len_B, B..., inner_op, inner_params...)
        OP_CBD: -1        # Variable: (N, *bytes)
    }
    
    params = []
    
    if op_id in [OP_CONST, OP_STEP, OP_LCG8, OP_LFSR8]:
        # Fixed parameter count
        param_count = param_counts[op_id]
        for _ in range(param_count):
            param, consumed = leb128_parse_single(seed_bytes, offset)
            params.append(param)
            offset += consumed
    else:
        # Variable parameter count - decode until L
        # This requires special handling per operator
        if op_id == OP_REPEAT1:
            # First param is D (motif length)
            D, consumed = leb128_parse_single(seed_bytes, offset)
            params.append(D)
            offset += consumed
            
            # Next D params are motif bytes
            for _ in range(D):
                byte_val, consumed = leb128_parse_single(seed_bytes, offset)
                params.append(byte_val)
                offset += consumed
                
        elif op_id == OP_CBD:
            # First param is N (byte count)
            N, consumed = leb128_parse_single(seed_bytes, offset)
            params.append(N)
            offset += consumed
            
            # Next N params are literal bytes
            for _ in range(N):
                byte_val, consumed = leb128_parse_single(seed_bytes, offset)
                params.append(byte_val)
                offset += consumed
                
        elif op_id == OP_ANCHOR:
            # Complex nested structure - simplified for now
            # This would need full parser for inner generators
            raise NotImplementedError("ANCHOR deserialization needs full parser")
    
    # Decode L
    L, consumed = leb128_parse_single(seed_bytes, offset)
    offset += consumed
    
    return op_id, tuple(params), L

def expand_generator_v2(op_id: int, params: tuple, L: int) -> bytes:
    """
    Expand generator to produce exactly L bytes.
    """
    if op_id == OP_CONST:
        b = params[0]
        return bytes([b] * L)
        
    elif op_id == OP_STEP:
        start, stride = params
        result = bytearray()
        for i in range(L):
            byte_val = (start + i * stride) % 256
            result.append(byte_val)
        return bytes(result)
        
    elif op_id == OP_LCG8:
        x0, a, c = params
        result = bytearray()
        x = x0
        for i in range(L):
            result.append(x)
            if i < L - 1:  # Don't advance after last byte
                x = (a * x + c) % 256
        return bytes(result)
        
    elif op_id == OP_LFSR8:
        taps, seed = params
        result = bytearray()
        state = seed
        for i in range(L):
            result.append(state)
            if i < L - 1:  # Don't advance after last byte
                state = lfsr8_step_v2(state, taps)
        return bytes(result)
        
    elif op_id == OP_REPEAT1:
        D = params[0]
        motif = params[1:D+1]
        result = bytearray()
        for i in range(L):
            result.append(motif[i % D])
        return bytes(result)
        
    elif op_id == OP_CBD:
        N = params[0]
        literal_bytes = params[1:N+1]
        return bytes(literal_bytes)
        
    elif op_id == OP_ANCHOR:
        # Simplified ANCHOR expansion
        len_A = params[0]
        A = params[1:1+len_A]
        len_B_offset = 1 + len_A
        len_B = params[len_B_offset]
        B = params[len_B_offset+1:len_B_offset+1+len_B]
        
        # Inner generator starts after B bytes
        inner_op_offset = len_B_offset + 1 + len_B
        inner_op = params[inner_op_offset]
        inner_params = params[inner_op_offset+1:]
        
        L_interior = L - len_A - len_B
        interior = expand_generator_v2(inner_op, inner_params, L_interior)
        
        return bytes(A) + interior + bytes(B)
    
    else:
        raise ValueError(f"unknown_op_id {op_id}")

def lfsr8_step_v2(state: int, taps: int) -> int:
    """LFSR8 step function matching predicates_v2.py"""
    feedback = 0
    temp_state = state
    temp_taps = taps
    
    while temp_taps > 0:
        if temp_taps & 1:
            feedback ^= (temp_state & 1)
        temp_state >>= 1
        temp_taps >>= 1
    
    return ((state >> 1) | (feedback << 7)) & 0xFF

def teleport_expand_verify(filename: str) -> tuple[bool, bool, str]:
    """
    CLI Tool 3: teleport-expand-verify filename
    
    Tests: scan → serialize → deserialize → expand → byte-equality check
    
    Output format:
    SUCCESS verification_passed OR FAILURE reason
    
    Returns: (success, verification_passed, message)
    """
    try:
        with open(filename, 'rb') as f:
            original_F = f.read()
    except Exception as e:
        return False, False, f"file_read_error {e}"
    
    try:
        # Step 1: Scan
        success, op_id, params, scan_msg = teleport_scan(filename)
        if not success:
            return False, False, f"scan_failed {scan_msg}"
        
        L = len(original_F)
        
        # Step 2: Serialize to seed
        seed_bytes = serialize_generator_v2(op_id, params, L)
        
        # Step 3: Deserialize from seed
        recovered_op_id, recovered_params, recovered_L = deserialize_generator_v2(seed_bytes)
        
        # Step 4: Verify deserialization correctness
        if recovered_op_id != op_id or recovered_params != params or recovered_L != L:
            return True, False, f"deserialization_mismatch op={op_id}→{recovered_op_id} params={params}→{recovered_params} L={L}→{recovered_L}"
        
        # Step 5: Expand generator
        expanded_F = expand_generator_v2(op_id, params, L)
        
        # Step 6: Byte equality verification
        if expanded_F == original_F:
            return True, True, f"verification_passed op_{op_id} L={L} seed_len={len(seed_bytes)}"
        else:
            # Find first mismatch for debugging
            first_mismatch = -1
            for i in range(min(len(original_F), len(expanded_F))):
                if original_F[i] != expanded_F[i]:
                    first_mismatch = i
                    break
            
            return True, False, f"byte_mismatch at_{first_mismatch} exp={original_F[first_mismatch] if first_mismatch < len(original_F) else 'EOF'} got={expanded_F[first_mismatch] if first_mismatch < len(expanded_F) else 'EOF'}"
    
    except Exception as e:
        return False, False, f"verification_error {e}"

def teleport_canonical(filename: str) -> tuple[bool, str, str]:
    """
    CLI Tool 4: teleport-canonical filename
    
    Verifies cost formula: 8×len(seed_bytes) == C_stream
    
    Output format:
    SUCCESS cost_verification_details OR FAILURE reason
    
    Returns: (success, cost_details, message)
    """
    try:
        # Get pricing info
        price_success, cost, price_msg = teleport_price(filename)
        if not price_success:
            return False, "", f"price_failed {price_msg}"
        
        # Get generator info for serialization
        scan_success, op_id, params, scan_msg = teleport_scan(filename)
        if not scan_success:
            return False, "", f"scan_failed {scan_msg}"
        
        with open(filename, 'rb') as f:
            F = f.read()
        L = len(F)
        
        # Serialize to get seed length
        seed_bytes = serialize_generator_v2(op_id, params, L)
        seed_len = len(seed_bytes)
        
        # Verify canonical cost formula: 8×len(seed_bytes) == C_stream
        expected_stream_cost = 8 * seed_len
        
        cost_details = f"C_caus={cost} seed_bytes={seed_len} expected_C_stream={expected_stream_cost}"
        
        if expected_stream_cost == cost:
            return True, cost_details, f"cost_verification_passed {cost_details}"
        else:
            return True, cost_details, f"cost_mismatch {cost_details}"
    
    except Exception as e:
        return False, "", f"canonical_error {e}"

def main():
    """Main CLI dispatcher"""
    if len(sys.argv) < 3:
        print("Usage: python cli_tools_v2.py {scan|price|expand-verify|canonical} filename")
        sys.exit(1)
    
    command = sys.argv[1]
    filename = sys.argv[2]
    
    if not os.path.exists(filename):
        print(f"FAILURE file_not_found {filename}")
        sys.exit(1)
    
    if command == "scan":
        success, op_id, params, msg = teleport_scan(filename)
        if success:
            params_str = " ".join(map(str, params))
            print(f"SUCCESS {op_id} {params_str}")
        else:
            print(f"FAILURE {msg}")
            
    elif command == "price":
        success, cost, msg = teleport_price(filename)
        if success:
            print(f"SUCCESS {cost}")
        else:
            print(f"FAILURE {msg}")
            
    elif command == "expand-verify":
        success, verified, msg = teleport_expand_verify(filename)
        if success and verified:
            print(f"SUCCESS verification_passed")
        elif success and not verified:
            print(f"SUCCESS verification_failed {msg}")
        else:
            print(f"FAILURE {msg}")
            
    elif command == "canonical":
        success, details, msg = teleport_canonical(filename)
        if success:
            print(f"SUCCESS {details}")
        else:
            print(f"FAILURE {msg}")
            
    else:
        print(f"FAILURE unknown_command {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
