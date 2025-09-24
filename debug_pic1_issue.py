# debug_pic1_issue.py
"""
Debug the serializer identity issue with pic1.jpg
"""

import sys
sys.path.insert(0, '/Users/Admin/Teleport')

from teleport.clf_builders import build_B, CBDToken, MATCHToken, STEPToken, CONSTToken


def debug_pic1_tokens():
    """Debug the first few tokens of pic1.jpg to find the serializer issue"""
    
    # Load pic1.jpg
    with open('/Users/Admin/Teleport/pic1.jpg', 'rb') as f:
        data = f.read()
    
    print(f"Data length: {len(data)}")
    print(f"First 20 bytes: {data[:20].hex()}")
    
    # Try to build the first few tokens manually to see where it fails
    try:
        tokens, info = build_B(data)
        print(f"Successfully built {len(tokens)} tokens")
    except Exception as e:
        print(f"Error: {e}")
        
        # Debug step by step
        print("\nDebugging step by step...")
        context = []
        pos = 0
        L = len(data)
        
        for step in range(10):  # Just first 10 positions
            if pos >= L:
                break
                
            print(f"\nStep {step}, pos={pos}")
            print(f"Current byte: {data[pos]:02x}")
            print(f"Context length: {len(context)}")
            
            # Try each token type
            from teleport.clf_builders import deduce_maximal_const_run, deduce_maximal_step_run, deduce_maximal_match_run
            
            # CONST
            const_run = deduce_maximal_const_run(data, pos, L)
            print(f"CONST run: {const_run}")
            
            if const_run > 0:
                try:
                    test_data = data[pos:pos + const_run]
                    token = CONSTToken(test_data, pos)
                    seed = token.serialize_seed()
                    c_stream = token.compute_stream_cost()
                    print(f"CONST token: seed_len={len(seed)}, c_stream={c_stream}")
                    print(f"Serializer identity: 8*{len(seed)} = {8*len(seed)}, c_stream = {c_stream}")
                    token.validate_serializer_identity()
                    print("✓ CONST token valid")
                    context.extend(test_data)
                    pos += const_run
                    continue
                except Exception as e:
                    print(f"❌ CONST token failed: {e}")
            
            # STEP
            step_run, base, increment = deduce_maximal_step_run(data, pos, L)
            print(f"STEP run: {step_run}, base={base}, inc={increment}")
            
            if step_run > 0:
                try:
                    token = STEPToken(base, increment, step_run, pos)
                    seed = token.serialize_seed()
                    c_stream = token.compute_stream_cost()
                    print(f"STEP token: seed_len={len(seed)}, c_stream={c_stream}")
                    print(f"Serializer identity: 8*{len(seed)} = {8*len(seed)}, c_stream = {c_stream}")
                    token.validate_serializer_identity()
                    print("✓ STEP token valid")
                    for i in range(step_run):
                        context.append((base + i * increment) % 256)
                    pos += step_run
                    continue
                except Exception as e:
                    print(f"❌ STEP token failed: {e}")
            
            # MATCH
            if len(context) >= 4:  # Need some context for MATCH
                match_run, match_distance = deduce_maximal_match_run(data, pos, bytes(context))
                print(f"MATCH run: {match_run}, distance={match_distance}")
                
                if match_run > 0 and match_distance > 0:
                    try:
                        token = MATCHToken(match_distance, match_run, pos)
                        seed = token.serialize_seed()
                        c_stream = token.compute_stream_cost()
                        print(f"MATCH token: seed_len={len(seed)}, c_stream={c_stream}")
                        print(f"Serializer identity: 8*{len(seed)} = {8*len(seed)}, c_stream = {c_stream}")
                        token.validate_serializer_identity()
                        print("✓ MATCH token valid")
                        for i in range(match_run):
                            context.append(data[pos + i])
                        pos += match_run
                        continue
                    except Exception as e:
                        print(f"❌ MATCH token failed: {e}")
                        # Let's debug the MATCH parameters more
                        print(f"  DEBUG: distance={match_distance}, length={match_run}")
                        if match_distance == 0:
                            print("  ❌ MATCH distance is 0 - this is the bug!")
                        break
            
            # CBD fallback
            try:
                test_data = data[pos:pos + 1]
                token = CBDToken(test_data, pos)
                seed = token.serialize_seed()
                c_stream = token.compute_stream_cost()
                print(f"CBD token: seed_len={len(seed)}, c_stream={c_stream}")
                print(f"Serializer identity: 8*{len(seed)} = {8*len(seed)}, c_stream = {c_stream}")
                token.validate_serializer_identity()
                print("✓ CBD token valid")
                context.extend(test_data)
                pos += 1
                continue
            except Exception as e:
                print(f"❌ CBD token failed: {e}")
                break


if __name__ == "__main__":
    debug_pic1_tokens()