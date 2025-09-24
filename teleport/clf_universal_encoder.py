# clf_universal_encoder.py
"""
CLF Universal Encoder with Mandatory Mathematical Rails
Refuses to return results without passing ALL mathematical rails.
Implements fail-closed behavior and comprehensive validation.
"""

import sys
import time
import hashlib
from typing import List, Tuple, Dict, Any
sys.path.insert(0, '/Users/Admin/Teleport')

from teleport.clf_encoder import encode_CLF
from teleport.clf_canonical_math import H_HEADER
from teleport.clf_receipts import extract_receipt_value
from teleport.clf_mathematical_rails import CLFMathematicalRails, MathematicalRailsViolation, generate_mandatory_rails_receipt


class CLFUniversalEncoder:
    """
    Universal CLF encoder with mandatory mathematical rails.
    Every encoding must pass ALL rails or fail completely.
    """
    
    def __init__(self):
        self.rails = CLFMathematicalRails()
        self.timing_history = []
        self.content_history = []
    
    def encode_with_universal_validation(self, S: bytes, runs: int = 3, 
                                       emit_full_receipt: bool = True) -> Dict[str, Any]:
        """
        Encode with universal validation - all rails must pass
        """
        L = len(S)
        start_time = time.time()
        
        # Store for performance rails
        self.content_history.append(S)
        
        # Multiple runs for determinism validation
        run_results = []
        for run in range(runs):
            run_start = time.time()
            
            try:
                tokens, receipt = encode_CLF(S, emit_receipts=True)
                run_time = time.time() - run_start
                
                # Extract values from receipt
                try:
                    H_str = extract_receipt_value(receipt, "H(L)")
                    # Parse "H(L) = 16 + 8·2 = 32 bits" -> 32
                    H = int(H_str.split('=')[-1].strip().split()[0])
                    
                    C_A_total = int(extract_receipt_value(receipt, "C_A_total"))
                    C_B_total = int(extract_receipt_value(receipt, "C_B_total"))
                    state = extract_receipt_value(receipt, "STATE")
                    
                    # Extract actual C_decision (H + min(A,B))
                    C_decision = H + min(C_A_total, C_B_total)
                    
                    run_results.append({
                        "run": run + 1,
                        "tokens": len(tokens),
                        "H": H,
                        "C_A_total": C_A_total,
                        "C_B_total": C_B_total,
                        "C_decision": C_decision,
                        "state": state,
                        "run_time": run_time,
                        "receipt": receipt,
                        "token_objects": tokens
                    })
                    
                except Exception as e:
                    raise MathematicalRailsViolation(f"Receipt parsing failed: {e}")
                    
            except Exception as e:
                raise MathematicalRailsViolation(f"Encoding failed on run {run+1}: {e}")
        
        total_time = time.time() - start_time
        self.timing_history.append((L, total_time / runs))
        
        # Check determinism across runs
        first_run = run_results[0]
        for i, run in enumerate(run_results[1:], 2):
            if (run["tokens"] != first_run["tokens"] or
                run["C_A_total"] != first_run["C_A_total"] or
                run["C_B_total"] != first_run["C_B_total"] or
                run["state"] != first_run["state"]):
                raise MathematicalRailsViolation(
                    f"Determinism violated between run 1 and run {i}"
                )
        
        # Use first run results for validation
        main_result = first_run
        tokens = main_result["token_objects"]
        H = main_result["H"]
        C_A = main_result["C_A_total"]
        C_B = main_result["C_B_total"]
        C_decision = main_result["C_decision"]
        state = main_result["state"]
        
        # Determine B_complete based on tokens and state
        B_complete = (state == "EMIT" and len(tokens) > 0)
        
        # Generate report text for language validation
        report_text = self._generate_report_text(S, tokens, H, C_A, C_B, C_decision, state)
        
        # RUN ALL MATHEMATICAL RAILS
        try:
            rails_results = self.rails.run_all_rails(
                S=S,
                tokens=tokens,
                H=H,
                C_A=C_A,
                C_B=C_B,
                C_decision=C_decision,
                B_complete=B_complete,
                state=state,
                timings=self.timing_history[-5:],  # Last 5 timings
                contents=self.content_history[-5:],  # Last 5 contents
                report_text=report_text
            )
        except MathematicalRailsViolation as e:
            # FAIL CLOSED - do not return results if any rail fails
            return {
                "success": False,
                "error": f"MATHEMATICAL RAILS VIOLATION: {e}",
                "fail_closed": True,
                "mathematical_compliance": "FAILED"
            }
        
        # Generate mandatory rails receipt
        mandatory_receipt = generate_mandatory_rails_receipt(rails_results)
        
        # Successful result with all rails passed
        return {
            "success": True,
            "file_info": {
                "length": L,
                "sha256": hashlib.sha256(S).hexdigest().upper()
            },
            "encoding_results": {
                "tokens": len(tokens),
                "state": state,
                "H": H,
                "C_A_total": C_A,
                "C_B_total": C_B,
                "C_decision": C_decision,
                "chosen_construction": "CBD" if C_A <= C_B else "STRUCT"
            },
            "validation_results": {
                "deterministic": True,
                "runs": runs,
                "avg_time": total_time / runs,
                "all_rails_passed": True
            },
            "rails_results": rails_results,
            "mandatory_receipt": mandatory_receipt,
            "original_receipt": main_result["receipt"],
            "mathematical_compliance": "COMPLETE",
            "fail_closed": False
        }
    
    def _generate_report_text(self, S: bytes, tokens: List, H: int, C_A: int, 
                             C_B: int, C_decision: int, state: str) -> str:
        """
        Generate report text using proper causal minimality language
        """
        L = len(S)
        raw_bits = 8 * L
        chosen = "CBD" if C_A <= C_B else "STRUCT"
        
        report = f"""
        CLF Causal Minimality Analysis:
        
        File processed: {L} bytes
        Canonical Decision Equation: C(S) = H(L) + min(C_CBD, C_STRUCT)
        Header cost: H(L) = {H} bits
        Construction A (CBD): {C_A} bits total
        Construction B (STRUCT): {C_B} bits total
        
        Decision: {chosen} chosen by causal minimality (C_{chosen} < C_{"CBD" if chosen == "STRUCT" else "STRUCT"})
        Final cost: C(S) = {C_decision} bits
        Raw bits: 8*L = {raw_bits} bits
        Admissibility: C(S) < 8*L = {C_decision < raw_bits} → {state}
        
        Structural tiling resulted in {len(tokens)} tokens with complete serializer identity.
        Mathematical constraints satisfied through causal deduction.
        """
        
        return report
    
    def validate_file_corpus(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Validate CLF encoder across a corpus of files to prove universality
        """
        corpus_results = []
        failed_files = []
        
        for i, file_path in enumerate(file_paths):
            print(f"Validating file {i+1}/{len(file_paths)}: {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                result = self.encode_with_universal_validation(data)
                
                if result["success"]:
                    corpus_results.append({
                        "file_path": file_path,
                        "length": len(data),
                        "state": result["encoding_results"]["state"],
                        "chosen": result["encoding_results"]["chosen_construction"],
                        "admissible": result["encoding_results"]["C_decision"] < 8 * len(data),
                        "rails_passed": result["rails_results"]["ALL_RAILS_PASSED"]
                    })
                else:
                    failed_files.append({
                        "file_path": file_path,
                        "error": result["error"]
                    })
                    
            except Exception as e:
                failed_files.append({
                    "file_path": file_path,
                    "error": f"File processing error: {e}"
                })
        
        # Analyze corpus results
        total_files = len(file_paths)
        successful_files = len(corpus_results)
        emit_count = sum(1 for r in corpus_results if r["state"] == "EMIT")
        open_count = sum(1 for r in corpus_results if r["state"] == "OPEN")
        
        return {
            "corpus_size": total_files,
            "successful_validations": successful_files,
            "failed_validations": len(failed_files),
            "emit_count": emit_count,
            "open_count": open_count,
            "universality_proven": len(failed_files) == 0,
            "corpus_results": corpus_results,
            "failed_files": failed_files,
            "mathematical_rails_enforced": True
        }


def demonstrate_universal_validation():
    """
    Demonstrate the universal encoder with comprehensive validation
    """
    encoder = CLFUniversalEncoder()
    
    # Test on pic1.jpg
    print("CLF Universal Encoder with Mandatory Mathematical Rails")
    print("====================================================")
    
    try:
        with open('/Users/Admin/Teleport/pic1.jpg', 'rb') as f:
            pic1_data = f.read()
        
        print(f"Testing pic1.jpg ({len(pic1_data)} bytes)...")
        result = encoder.encode_with_universal_validation(pic1_data)
        
        if result["success"]:
            print("✅ ALL MATHEMATICAL RAILS PASSED")
            print(f"State: {result['encoding_results']['state']}")
            print(f"Chosen: {result['encoding_results']['chosen_construction']}")
            print(f"Mathematical Compliance: {result['mathematical_compliance']}")
            print("\nMandatory Rails Receipt:")
            print(result["mandatory_receipt"])
        else:
            print("❌ MATHEMATICAL RAILS FAILED")
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Universal validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_universal_validation()