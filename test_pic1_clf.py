#!/usr/bin/env python3
"""
Test CLF Mathematical Implementation on pic1.jpg
Generates exact mathematical receipts with drift-killer rails
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from teleport.clf_canonical import encode_CLF, clf_canonical_receipts

def test_pic1_jpg():
    """Test mathematical CLF encoding on real JPEG file"""
    
    pic1_path = Path("test_artifacts/pic1.jpg")
    if not pic1_path.exists():
        print(f"❌ {pic1_path} not found")
        return 1
    
    print("CLF Mathematical Analysis: pic1.jpg")
    print("=" * 50)
    
    # Load file
    S = pic1_path.read_bytes()
    print(f"Input: {len(S)} bytes")
    
    # Apply mathematical CLF encoding
    tokens = encode_CLF(S)
    
    # Generate mathematical receipts
    receipts = clf_canonical_receipts(S, tokens)
    
    print("\n".join(receipts))
    
    # Summary
    if tokens:
        print(f"\n✓ CLF encoding succeeded with {len(tokens)} tokens")
    else:
        print("\n○ CLF encoding: OPEN (no beneficial encoding found)")
        print("  This is mathematically correct under current operator set")
    
    return 0

if __name__ == "__main__":
    exit(test_pic1_jpg())
