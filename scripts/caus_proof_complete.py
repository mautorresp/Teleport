#!/usr/bin/env python3
"""
Teleport Complete Mathematical Proof System

Implements the "Prove Everything" CLF pipeline:
1. Either proves causality with exact generators and parameters
2. Or provides formal refutation with quantified mathematical witnesses

No middle ground, no heuristics, no formats - pure mathematical generators only.
"""

import sys
from pathlib import Path
from teleport.caus_deduction_complete import formal_caus_test

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 caus_proof_complete.py <input_file>")
        print()
        print("Complete mathematical proof system for Teleport CLF.")
        print("Either proves causality with exact generators or provides formal refutation.")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    # Read input bytes
    try:
        input_data = input_path.read_bytes()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Run complete mathematical analysis
    print(f"Teleport Complete Mathematical Proof System")
    print(f"==========================================")
    print()
    
    exit_code = formal_caus_test(input_data, str(input_path))
    
    print()
    print(f"Mathematical analysis complete.")
    print(f"Generator family: G = {{CONST, STEP, LCG8, LFSR8}}")
    print(f"All results based on exact integer arithmetic and formal verification.")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
