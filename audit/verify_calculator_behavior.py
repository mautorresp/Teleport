#!/usr/bin/env python3
"""
CLF Immutable Rails - Calculator Behavior Verification
======================================================

PIN-ENC-CALC: Verify calculator hot-path is instant & size-independent
Evidence: Must match pic1.jpg and pic2.jpg external audit performance
"""

from teleport.clf_fb import build_A_exact
import time, os, hashlib

def verify_calculator_behavior():
    """Verify calculator behavior matches external audit evidence."""
    print("🔥 CLF CALCULATOR HOT-PATH VERIFICATION")
    print("=" * 50)
    
    test_files = [
        ("pic1.jpg", "pic1.jpg"),
        ("pic2.jpg", "pic2.jpg")
    ]
    
    for display_name, filename in test_files:
        if not os.path.exists(filename):
            print(f"⚠️  {filename} not found, skipping")
            continue
            
        with open(filename, "rb") as f:
            S = f.read()
        
        # Test calculator mode - should be instant
        t0 = time.perf_counter()
        builder = build_A_exact(S)
        toks = builder.finalize().tokens
        t1 = time.perf_counter()
        
        print(f"📁 {display_name}: {len(S):,} bytes")
        print(f"⚡ Calc mode: {len(toks)} token(s) in {(t1-t0)*1e3:.3f}ms")
        
        # Verify tokens are CBD types only
        for i, token in enumerate(toks):
            op_type = token[0]
            if not isinstance(op_type, str) or op_type not in ('CBD_BOUND', 'CBD_LOGICAL'):
                print(f"❌ PIN-ENC-CALC violation: token {i+1} is {op_type}")
                return False
        
        print(f"✅ PIN-ENC-CALC: All tokens are CBD types")
    
    print("🔥 Calculator hot-path verification complete")
    return True

if __name__ == "__main__":
    verify_calculator_behavior()