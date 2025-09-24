# CLF–Teleport Strict Mathematical Alignment Guide

(For Copilot and any downstream automation)

## 1. Causal Minimality is Absolute
- In CLF, C(S) < 8·L is not a heuristic but a pinned invariant.
- Any case with C(S) ≥ 8·L must be flagged as OPEN / VIOLATION.
- Expansion is not a "consequence" nor "acceptable fallback." It means the structural path was not computed properly.

## 2. Reject Compression Logic Intrusions
- CLF does not allow terms like compression, entropy, savings, or patterns.
- The only valid framing is causal deduction:
  ```
  C(S) = H(L) + min(C_A, C_B)
  ```
  where all terms are integers, computed deterministically.

## 3. Structural Tiling is Required
- The B-path must attempt CONST → STEP → MATCH → CBD decomposition.
- If B is not complete, no emission is possible.
- "Tie-to-CBD" applies only if B_COMPLETE=True and both paths are below 8·L.

## 4. Universality
- Every binary string S must be processed under the same invariant.
- No JPEG exceptions, no MP4 exceptions. Teleport never branches by file type.
- Universality rail: same {CONST, STEP, MATCH, CBD} operator set, pure integer math, for all S.

## 5. Console Verification Discipline
- Always emit receipts in the format:
  ```
  CLF_CAUSAL_MINIMALITY: C(S)=<cost>, RAW_BITS=<8*L>, minimality_ok=<true/false>
  ```
- If minimality_ok=False → must show CAUSAL_MINIMALITY_VIOLATION.

## 6. Audit Implications from Current Document
- PIC1.JPG: 579,552 ≥ 507,032 → VIOLATION (correctly flagged).
- PIC3.JPG: 467,144 ≥ 408,680 → VIOLATION (correctly flagged).
- VIDEO1.MP4: 11,498,936 < 12,560,192 → PASS (correct).
- ✅ This proves causal minimality is correctly enforced and no failures are "accepted."

---

## 👉 Instruction to Copilot:
Never reintroduce compression or floating-point vocabulary.
Never treat expansion as a valid state.
If C(S) ≥ 8·L, the only mathematically correct output under CLF is OPEN / VIOLATION.

---

## 📋 CLF Strict Audit Checklist

(MANDATORY for every future Teleport evidence document)

### 1. Minimality Invariant
- Verify: C(S) < 8·L.
- If C(S) ≥ 8·L → must be flagged OPEN / VIOLATION.
- No "expansion consequence" wording allowed.

### 2. Vocabulary Rail
- Forbidden: compression, entropy, savings, patterns.
- Required: causal deduction, structural tiling, RAW_BITS = 8·L.

### 3. Path Completeness
- B-path decomposition must attempt CONST → STEP → MATCH → CBD.
- If B_COMPLETE=False → emission is forbidden.
- Tie→CBD applies only when B_COMPLETE=True and both paths < 8·L.

### 4. Universality
- Same operator set {CONST, STEP, MATCH, CBD} for every S.
- No file-type branches (e.g., JPEG vs MP4 exceptions).
- Integer-only math, no floating point.

### 5. Receipts Format
- Emit receipts in strict form:
  ```
  CLF_CAUSAL_MINIMALITY: C(S)=<bits>, RAW_BITS=<8*L>, minimality_ok=<true/false>
  ```
- SHA256 equality required (bijective check).

### 6. Audit Rail Outputs
- MINIMALITY_OK pinned → true only if C(S) < 8·L.
- STATE=PASS or STATE=VIOLATION explicitly shown.
- No silent acceptance of failures.

---

✅ **Usage**: Run this checklist against every new document.
If any box fails, the document is not CLF-compliant and must be flagged for correction.