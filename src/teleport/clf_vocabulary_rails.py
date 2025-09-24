"""
CLF Vocabulary and Logic Rails - Hard Bans
==========================================

Enforces mathematical language and prevents compression-style thinking.
All CLF systems must use causal deduction vocabulary only.
"""

from typing import List, Dict, Any
import re

# BANNED VOCABULARY - any appearance triggers RAIL_BANNED_VOCAB_HIT
BANNED_WORDS = {
    "arbitrary", "random", "entropy", "compress", "compressed", 
    "compression", "incompressible", "pattern", "patterns",
    "high-entropy", "low-entropy", "compressible", "uncompressible"
}

# REQUIRED VOCABULARY - CLF mathematical terms only
REQUIRED_TERMS = {
    "causal deduction", "structural tiling", "minimal seed", 
    "integer-exact", "bijection", "deterministic", "mathematical"
}

def rail_vocabulary_check(text: str, context: str = "unknown") -> None:
    """
    Rail R_VOCAB: Ban compression vocabulary, require causal deduction terms
    """
    text_lower = text.lower()
    
    # Check for banned words
    for banned in BANNED_WORDS:
        if banned in text_lower:
            raise RuntimeError(f"RAIL_BANNED_VOCAB_HIT: '{banned}' in {context}")
    
    # Pattern check for compression-style phrases
    compression_patterns = [
        r"can't\s+compress", r"not\s+compressible", r"too\s+random",
        r"no\s+pattern", r"high\s+entropy", r"arbitrary\s+data"
    ]
    
    for pattern in compression_patterns:
        if re.search(pattern, text_lower):
            raise RuntimeError(f"RAIL_BANNED_PATTERN_HIT: '{pattern}' in {context}")

def rail_causefail_wording(reason: str) -> None:
    """
    Rail R_CAUSEFAIL: Only mathematical failure reasons allowed
    """
    allowed_reasons = {
        "BUILDER_INCOMPLETENESS", "PROOF_INCOMPLETE", "MINIMALITY_NOT_ACHIEVED",
        "U_B_NOT_IMPLEMENTED", "SEED_DERIVATION_INCOMPLETE", "COVERAGE_INCOMPLETE"
    }
    
    if reason not in allowed_reasons:
        raise RuntimeError(f"RAIL_CAUSEFAIL_WORDING: '{reason}' not mathematical")
    
    # Ban data-blaming phrases  
    banned_phrases = ["data is", "string is", "file is", "input is"]
    reason_lower = reason.lower()
    for phrase in banned_phrases:
        if phrase in reason_lower:
            raise RuntimeError(f"RAIL_DATA_BLAMING: '{phrase}' blames input")

def rail_cbd_seed_provenance(seed_origin: str) -> None:
    """
    Rail R_SEED: Only causal seed origins allowed
    """
    allowed_origins = {
        'DERIVED_FROM_A_EXACT', 'DERIVED_FROM_B', 'EXTERNAL_PROOF'
    }
    
    banned_origins = {
        'PACKED_RAW', 'FROM_BYTES', 'FROM_MEMORYVIEW', 'ARBITRARY_DATA'
    }
    
    if seed_origin not in allowed_origins:
        raise RuntimeError(f"RAIL_SEED_PROVENANCE: '{seed_origin}' not causal")
    
    if seed_origin in banned_origins:
        raise RuntimeError(f"RAIL_SEED_BANNED: '{seed_origin}' uses S-packing")

def rail_detect_s_packing(leb_k_len: int, L: int) -> None:
    """
    Rail R_PACK: Detect ceil(8L/7) pattern indicating S-packing
    """
    ceil_8L_7 = (8 * L + 6) // 7
    
    if leb_k_len >= ceil_8L_7:
        raise RuntimeError(f"RAIL_S_PACKING_DETECTED: leb_k={leb_k_len} >= ceil(8L/7)={ceil_8L_7}")

def rail_builder_independence(A_result: Any, B_result: Any) -> None:
    """
    Rail R_INDEPENDENCE: Builders must be completely separate
    """
    if A_result is None or B_result is None:
        return  # One unavailable is fine
    
    # Check for identical results (aliasing bug)
    if (hasattr(A_result, 'get') and hasattr(B_result, 'get') and 
        A_result.get('C_stream') == B_result.get('C_stream') and
        A_result.get('tokens') == B_result.get('tokens')):
        raise RuntimeError("RAIL_BUILDER_ALIASING: A and B produced identical results")

def rail_universal_deduction_requirement(A_result: Any, B_result: Any, S: bytes) -> None:
    """
    Rail R_UNIVERSAL: Every string must admit causal deduction
    """
    L = len(S)
    
    # Both builders failed - this is BUILDER_INCOMPLETENESS, not data property
    if A_result is None and (B_result is None or not B_result.get('complete', False)):
        raise RuntimeError(f"RAIL_UNIVERSAL_VIOLATED: BUILDER_INCOMPLETENESS for L={L}")
    
    # Ban any logs that suggest data is the problem
    # This would be called after processing to check log content

def validate_mathematical_language(receipt_lines: List[str]) -> None:
    """
    Validate all receipt text uses only mathematical CLF vocabulary
    """
    full_text = "\n".join(receipt_lines)
    rail_vocabulary_check(full_text, "receipt")
    
    # Ensure mathematical framing is present
    has_mathematical_terms = any(term in full_text.lower() for term in [
        "causal", "deduction", "structural", "tiling", "minimal", "seed"
    ])
    
    if not has_mathematical_terms:
        raise RuntimeError("RAIL_MATHEMATICAL_FRAMING: Receipt lacks CLF mathematical terms")