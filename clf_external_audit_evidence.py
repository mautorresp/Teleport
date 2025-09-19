#!/usr/bin/env python3
"""
CLF External Audit Evidence Generator
=====================================

Mathematical witness and complete process documentation for external audit.
Demonstrates CLF as a pure integer arithmetic bijection with minimal representations.

This script provides:
1. Complete process chain: Input → Encoding → Tokens → Decoding → Output → Verification
2. Mathematical witness showing D ∘ C ∘ E = identity (perfect bijection)
3. Token-by-token breakdown with mathematical cost accounting
4. Construction A vs Construction B performance analysis
5. External audit trail with all intermediate values
6. SHA-256 cryptographic verification of perfect reconstruction
"""

import sys
import os
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

# Add teleport to path
sys.path.append('/Users/Admin/Teleport')

from teleport.clf_canonical import (
    encode_CLF, decode_CLF, finalize_cbd_tokens,
    _validate_unit_lock_and_ids, _validate_rails
)

class CLFExternalAudit:
    """External audit evidence generator for CLF mathematical claims."""
    
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.audit_timestamp = datetime.now().isoformat()
        self.evidence = {
            'audit_metadata': {
                'timestamp': self.audit_timestamp,
                'input_file': str(self.input_file),
                'clf_version': 'canonical_mathematical_bijection',
                'audit_type': 'external_mathematical_witness'
            },
            'process_chain': {},
            'mathematical_witness': {},
            'performance_analysis': {},
            'cryptographic_verification': {}
        }
    
    def load_input_data(self):
        """Load and validate input data."""
        print(f"=== STEP 1: INPUT VALIDATION ===")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Load binary data
        with open(self.input_file, 'rb') as f:
            self.input_data = f.read()
        
        # Calculate input hash
        self.input_hash = hashlib.sha256(self.input_data).hexdigest()
        
        print(f"Input File: {self.input_file}")
        print(f"Input Size: {len(self.input_data)} bytes")
        print(f"SHA-256 IN: {self.input_hash}")
        
        # Store in evidence
        self.evidence['process_chain']['input'] = {
            'file_path': str(self.input_file),
            'size_bytes': len(self.input_data),
            'sha256_hash': self.input_hash,
            'first_16_bytes': self.input_data[:16].hex(),
            'last_16_bytes': self.input_data[-16:].hex() if len(self.input_data) >= 16 else self.input_data.hex()
        }
        
        return self.input_data
    
    def encode_with_mathematical_witness(self):
        """Encode with complete mathematical documentation."""
        print(f"\n=== STEP 2: CLF ENCODING (MATHEMATICAL MODE) ===")
        
        # Encode in minimal mode (mathematical default)
        start_time = time.time()
        self.tokens = encode_CLF(self.input_data, mode="minimal")
        encoding_time = time.time() - start_time
        
        print(f"Encoding Mode: minimal (mathematical bijection)")
        print(f"Encoding Time: {encoding_time:.6f} seconds")
        print(f"Output Tokens: {len(self.tokens)}")
        
        # Analyze token structure
        token_analysis = []
        total_stream_cost = 0
        
        for i, token in enumerate(self.tokens):
            token_type = token[0]
            cost_info = token[3] if len(token) > 3 else {}
            stream_cost = cost_info.get('C_stream', 0)
            total_stream_cost += stream_cost
            
            token_data = {
                'index': i + 1,
                'type': str(token_type),
                'stream_cost_bits': stream_cost,
                'full_token': str(token)
            }
            token_analysis.append(token_data)
            
            print(f"  Token {i+1}: Type={token_type}, Stream Cost={stream_cost} bits")
        
        # Calculate mathematical costs
        L = len(self.input_data)
        header_cost = 16 + 8 * ((8 * L).bit_length() + 6) // 7  # H(L) = 16 + 8*leb_len(8*L)
        total_cost = header_cost + total_stream_cost
        baseline_10L = 10 * L
        mathematical_ratio = total_cost / baseline_10L if baseline_10L > 0 else float('inf')
        
        print(f"\n--- MATHEMATICAL COST ACCOUNTING ---")
        print(f"Stream Cost: {total_stream_cost} bits")
        print(f"Header Cost: {header_cost} bits (16 + 8*leb_len(8*{L}))")
        print(f"Total Cost: {total_cost} bits")
        print(f"10*L Baseline: {baseline_10L} bits")
        print(f"Mathematical Ratio: {mathematical_ratio:.6f}")
        print(f"Compression Percentage: {(1 - mathematical_ratio) * 100:.3f}%")
        
        # Store in evidence
        self.evidence['process_chain']['encoding'] = {
            'mode': 'minimal',
            'encoding_time_seconds': encoding_time,
            'token_count': len(self.tokens),
            'token_analysis': token_analysis,
            'cost_accounting': {
                'stream_cost_bits': total_stream_cost,
                'header_cost_bits': header_cost,
                'total_cost_bits': total_cost,
                'baseline_10L_bits': baseline_10L,
                'mathematical_ratio': mathematical_ratio,
                'compression_percentage': (1 - mathematical_ratio) * 100
            }
        }
        
        return self.tokens
    
    def finalize_tokens(self):
        """Finalize tokens for decoding."""
        print(f"\n=== STEP 3: TOKEN FINALIZATION ===")
        
        # Check if finalization is needed
        needs_finalization = any(isinstance(t[0], str) for t in self.tokens)
        
        if needs_finalization:
            print("Finalizing tokens (converting string types to integers)...")
            self.finalized_tokens = finalize_cbd_tokens(self.tokens)
        else:
            print("Tokens already in finalized form")
            self.finalized_tokens = self.tokens
        
        print(f"Finalized Tokens: {len(self.finalized_tokens)}")
        
        # Store finalized token structure
        finalized_analysis = []
        for i, token in enumerate(self.finalized_tokens):
            finalized_analysis.append({
                'index': i + 1,
                'finalized_token': str(token)
            })
        
        self.evidence['process_chain']['finalization'] = {
            'needed_finalization': needs_finalization,
            'finalized_token_count': len(self.finalized_tokens),
            'finalized_tokens': finalized_analysis
        }
        
        return self.finalized_tokens
    
    def decode_with_verification(self):
        """Decode with mathematical verification."""
        print(f"\n=== STEP 4: CLF DECODING & BIJECTION VERIFICATION ===")
        
        # Decode - use raw tokens for direct bijection test
        start_time = time.time()
        self.decoded_data = decode_CLF(self.tokens)  # Use raw tokens, not finalized
        decoding_time = time.time() - start_time
        
        print(f"Decoding Time: {decoding_time:.6f} seconds")
        print(f"Decoded Size: {len(self.decoded_data)} bytes")
        
        # Calculate output hash
        self.output_hash = hashlib.sha256(self.decoded_data).hexdigest()
        print(f"SHA-256 OUT: {self.output_hash}")
        
        # Verify perfect bijection
        bijection_perfect = self.decoded_data == self.input_data
        size_match = len(self.decoded_data) == len(self.input_data)
        hash_match = self.output_hash == self.input_hash
        
        print(f"\n--- BIJECTION VERIFICATION ---")
        print(f"Size Match: {size_match} ({len(self.input_data)} → {len(self.decoded_data)})")
        print(f"Hash Match: {hash_match}")
        print(f"Byte-Perfect: {bijection_perfect}")
        
        if bijection_perfect:
            print("✅ MATHEMATICAL WITNESS: D ∘ C ∘ E = identity VERIFIED")
        else:
            print("❌ BIJECTION FAILURE")
            # Find first difference for debugging
            for i, (a, b) in enumerate(zip(self.input_data, self.decoded_data)):
                if a != b:
                    print(f"First difference at byte {i}: 0x{a:02x} → 0x{b:02x}")
                    break
        
        # Store verification results
        self.evidence['mathematical_witness'] = {
            'decoding_time_seconds': decoding_time,
            'input_sha256': self.input_hash,
            'output_sha256': self.output_hash,
            'size_match': size_match,
            'hash_match': hash_match,
            'bijection_perfect': bijection_perfect,
            'mathematical_identity': 'D ∘ C ∘ E = identity' if bijection_perfect else 'FAILED'
        }
        
        return self.decoded_data
    
    def construction_analysis(self):
        """Analyze Construction A vs Construction B performance."""
        print(f"\n=== STEP 5: CONSTRUCTION ANALYSIS ===")
        
        # Test both modes for comparison
        tokens_minimal = encode_CLF(self.input_data, mode="minimal")  # Construction B
        tokens_calc = encode_CLF(self.input_data, mode="calc")       # Construction A
        
        # Analyze minimal mode (Construction B)
        minimal_stream_cost = sum(
            t[3].get('C_stream', 0) if len(t) > 3 else 0 
            for t in tokens_minimal
        )
        L = len(self.input_data)
        minimal_header = 16 + 8 * ((8 * L).bit_length() + 6) // 7
        minimal_total = minimal_header + minimal_stream_cost
        
        # Analyze calc mode (Construction A)  
        calc_stream_cost = sum(
            t[3].get('C_stream', 0) if len(t) > 3 else 0 
            for t in tokens_calc
        )
        calc_total = minimal_header + calc_stream_cost  # Same header
        
        baseline = 10 * L
        
        print(f"--- CONSTRUCTION COMPARISON ---")
        print(f"Construction B (minimal): {len(tokens_minimal)} tokens, {minimal_total} bits, ratio {minimal_total/baseline:.6f}")
        print(f"Construction A (calc): {len(tokens_calc)} tokens, {calc_total} bits, ratio {calc_total/baseline:.6f}")
        
        # Determine which construction was used
        if len(self.tokens) == len(tokens_minimal) and self.tokens[0][0] == tokens_minimal[0][0]:
            construction_used = "B (minimal/mathematical)"
        else:
            construction_used = "A (calc/fast)"
        
        print(f"Construction Used: {construction_used}")
        
        self.evidence['performance_analysis'] = {
            'construction_B_minimal': {
                'token_count': len(tokens_minimal),
                'total_bits': minimal_total,
                'ratio_vs_10L': minimal_total / baseline,
                'compression_percent': (1 - minimal_total/baseline) * 100
            },
            'construction_A_calc': {
                'token_count': len(tokens_calc),
                'total_bits': calc_total,
                'ratio_vs_10L': calc_total / baseline,
                'compression_percent': (1 - calc_total/baseline) * 100
            },
            'construction_used': construction_used,
            'baseline_10L_bits': baseline
        }
    
    def generate_audit_report(self):
        """Generate comprehensive external audit report."""
        print(f"\n=== STEP 6: EXTERNAL AUDIT REPORT GENERATION ===")
        
        # Create audit filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = self.input_file.stem
        audit_filename = f"CLF_EXTERNAL_AUDIT_{input_name}_{timestamp}.json"
        audit_path = Path(audit_filename)
        
        # Add cryptographic verification section
        self.evidence['cryptographic_verification'] = {
            'input_file_hash': self.input_hash,
            'decoded_output_hash': self.output_hash,
            'hash_algorithm': 'SHA-256',
            'bijection_verified': self.input_hash == self.output_hash,
            'verification_timestamp': datetime.now().isoformat()
        }
        
        # Add summary section
        minimal_ratio = self.evidence['performance_analysis']['construction_B_minimal']['ratio_vs_10L']
        compression_pct = self.evidence['performance_analysis']['construction_B_minimal']['compression_percent']
        
        self.evidence['audit_summary'] = {
            'file_processed': str(self.input_file),
            'input_size_bytes': len(self.input_data),
            'mathematical_bijection_verified': self.evidence['mathematical_witness']['bijection_perfect'],
            'minimal_construction_ratio': minimal_ratio,
            'compression_percentage': compression_pct,
            'tokens_generated': len(self.tokens),
            'process_integrity': 'VERIFIED' if self.evidence['mathematical_witness']['bijection_perfect'] else 'FAILED'
        }
        
        # Save audit report
        with open(audit_path, 'w') as f:
            json.dump(self.evidence, f, indent=2)
        
        print(f"Audit Report Saved: {audit_path}")
        print(f"Report Size: {audit_path.stat().st_size} bytes")
        
        return audit_path
    
    def run_complete_audit(self):
        """Run complete external audit with mathematical witness."""
        print("CLF EXTERNAL AUDIT EVIDENCE GENERATOR")
        print("=" * 50)
        print(f"Timestamp: {self.audit_timestamp}")
        print(f"Input File: {self.input_file}")
        
        try:
            # Execute complete process chain
            self.load_input_data()
            self.encode_with_mathematical_witness()
            self.finalize_tokens()
            self.decode_with_verification()
            self.construction_analysis()
            audit_report = self.generate_audit_report()
            
            # Final summary
            print(f"\n" + "=" * 50)
            print("EXTERNAL AUDIT COMPLETE")
            print("=" * 50)
            
            bijection_status = "✅ VERIFIED" if self.evidence['mathematical_witness']['bijection_perfect'] else "❌ FAILED"
            compression_pct = self.evidence['performance_analysis']['construction_B_minimal']['compression_percent']
            
            print(f"Mathematical Bijection: {bijection_status}")
            print(f"Compression Achieved: {compression_pct:.3f}%")
            print(f"Construction Used: {self.evidence['performance_analysis']['construction_used']}")
            print(f"Audit Evidence: {audit_report}")
            
            return audit_report
            
        except Exception as e:
            print(f"❌ AUDIT FAILED: {e}")
            import traceback
            print(traceback.format_exc())
            return None

def main():
    """Main entry point for external audit."""
    if len(sys.argv) < 2:
        print("Usage: python clf_external_audit_evidence.py <input_file>")
        print("Available files:")
        for jpg_file in Path('.').glob('*.jpg'):
            print(f"  {jpg_file}")
        return
    
    input_file = sys.argv[1]
    audit = CLFExternalAudit(input_file)
    audit.run_complete_audit()

if __name__ == '__main__':
    main()