# NOTE TO AI ASSISTANTS:
# Teleport is NOT a compression algorithm and NOT a format-aware decoder.
# It is a binary calculator that searches for provable generators (CAUS).
# Formats like JPEG, PNG, MP3 are irrelevant here; if a header matches,
# it is treated as an anchor predicate ONLY IF it can be expressed as 
# integers in the CLF predicate suite. Otherwise, it is ignored.
#
# RULES:
# 1. Minimality is mandatory. Seeds must be drastically smaller when CAUS exists.
# 2. No fallback compression. If no CAUS is deduced, output MUST be CAUSE_NOT_DEDUCED.
# 3. No floating point, no heuristics. Exact integer arithmetic only.
# 4. Formats are consequences of predicates, not primitives. Never implement "JPEG decoder".
#
# TELEPORT PREDICATES ARE MATHEMATICAL GENERATORS:
# - P_CONST(b): ∀i: S[i] = b
# - P_STEP(a,d): ∀i: S[i] = (a + i×d) & 255  
# - P_REPEAT1(D): ∀i≥D: S[i] = S[i-D]
# - P_ANCHOR(A,B,...): Find byte patterns A,B and prove generator reproduces entire S
#
# If bytes happen to contain FF D8 (JPEG start), that's just an anchor pattern.
# If bytes happen to contain PNG header, that's just another anchor pattern.
# Teleport doesn't care what the "file format" is - only whether mathematical
# generators with small parameter sets can prove they generate the exact byte string.
#
# COMPRESSION THINKING (FORBIDDEN):
# - "This is a JPEG, so we need a JPEG decoder"
# - "Parse the format structure to understand the file"  
# - "Fallback to LZ77 when format detection fails"
#
# TELEPORT THINKING (CORRECT):
# - "Do any mathematical generators prove they generate this exact byte string?"
# - "What is the minimal parameter set that deterministically produces these bytes?"
# - "If no proven generator exists, exit with CAUSE_NOT_DEDUCED"
