# CLF COMPLIANCE AUDIT GUARDRAILS

## CLF Operating Order - STRICT ENFORCEMENT

**This is a binary math calculator, not compression.**

### Mandatory Constraints (NEVER Violate):

1. **No floats, no entropy, no formats, no heuristics**
   - All operations must use pure integer arithmetic
   - All predicates must provide exact mathematical witnesses
   - No file format knowledge or MIME detection allowed
   - No statistical inference or probabilistic reasoning

2. **Deductions are proofs**
   - Predicate is either TRUE with parameters + byte-exact replay OR FALSE with quantified mismatch
   - First counterexample must be provided: `mismatch_at=i exp=v1 got=v2`
   - No search, guessing, or approximation permitted

3. **No fallback success**
   - If no generator proves S, emit GENERATOR_MISSING (code incomplete)
   - NEVER certify success without passing verifier and cost proof
   - CBD/literal storage is NOT causality - removed from success path

4. **Reports must mirror receipts**
   - If receipts show GENERATOR_MISSING, summaries must say exactly that
   - No claims of "causality established" when generators failed
   - Trust only console receipts and implemented math

5. **Every generator = three artifacts**
   - Predicate (mathematical proof or quantified refutation)
   - Exact cost (CLF formula with integers only)
   - Verifier (byte-exact replay from parameters)

### CLF Success Criteria Checklist

A run is mathematically valid ONLY if ALL criteria satisfied:

#### A) Purity & Grammar ✅
- [x] Integers only - no floats anywhere
- [x] Format-blind - bytes in, bytes out
- [x] Minimal LEB128 - all varints minimal encoding
- [x] Token domains respected (LIT: 1≤L≤10, MATCH: D≥1, L≥3, etc.)
- [x] END cost with residue: C_END(pos) = 3 + pad_to_byte(pos + 3)

#### B) Deduction (Proof of Causality) ✅
- [x] Predicate TRUE with concrete params + verifier OR FALSE with quantified refutation
- [x] No search/guessing - predicates compute equalities directly
- [x] ANCHOR rule: positive interior + inner generator proof required

#### C) Costs & Selection ✅
- [x] Exact token costs: C_CAUS = 3 + 8·leb(op) + 8·Σ leb(params) + 8·leb(N)
- [x] Residue-aware global optimality via dynamic programming
- [x] Strict inequality vs LIT when applicable
- [x] Deterministic tie-breaking (same S → same seed)

#### D) Serialization Invariants ✅
- [x] Bit-exact identity: 8·len(seed_bytes) == C_stream
- [x] Seed ↔ VM agreement - same bytecode costed and serialized

#### E) Expansion & Identity Proof ✅
- [x] Constructive replay: expand(seed) == S with eq_bytes=1, eq_sha=1
- [x] Idempotence: expand(canonize(S)) == S

#### F) Universality Obligation ✅
- [x] No fallback success - refuse certification without mathematical proof
- [x] GENERATOR_MISSING with actionable implementation receipts

#### G) Prohibitions ✅
- [x] No transport modifications - process bytes exactly as provided
- [x] No heuristics/thresholds - all claims backed by integer receipts
- [x] No format semantics - pure mathematical byte analysis only

### Test Matrix (Must Pass)

1. **CONST 30 A's**: Expect C_CAUS=27, C_END=5, C_stream=32, 8·len(seed)=32
2. **STEP sequence**: Integer costs printed, full replay proof
3. **LCG8 synthetic**: CAUS(LCG8) with replay verification  
4. **XOR_MASK8**: base ⊕ mask == S verification
5. **ANCHOR canonical**: Fixed bytes with inner generator proof
6. **pic1.jpg**: GENERATOR_MISSING with quantified receipts (until new generator implemented)

### Failure Patterns (Fix Immediately)

❌ **Summary contradicts receipts** - claiming success when GENERATOR_MISSING  
❌ **Floating point contamination** - any non-integer arithmetic  
❌ **Format knowledge** - JPEG/PNG semantics in analysis  
❌ **Fallback certification** - CBD success without mathematical proof  
❌ **Approximate calculations** - "about", "roughly", estimates  
❌ **Heuristic reasoning** - statistical inference or probabilistic claims

### Implementation Rules

1. **Each new generator must provide**:
   - Mathematical predicate with exact proof or quantified failure
   - Cost function using CLF formula: 3 + 8·leb(op) + 8·Σ leb(params) + 8·leb(N)  
   - Verifier that replay-constructs all bytes exactly

2. **GENERATOR_MISSING handling**:
   - Structured fault with invariant signatures
   - Candidate schema for missing generator
   - Constructive requirements for implementation
   - Clear indication this is code gap, not mathematical impossibility

3. **Canonical determinism**:
   - Same input always produces same result
   - No arbitrary choices or random elements
   - Lexicographic tie-breaking when costs equal

### Audit Validation

**Before claiming success, verify**:
- Predicate receipts show TRUE with mathematical parameters
- Cost equation printed with integer expansions  
- Identity verification: eq_bytes=1, eq_sha=1
- Debug invariant: 8·len(seed) == C_stream
- No floating point operations anywhere in call stack

**If GENERATOR_MISSING**:
- All predicate failures quantified with exact witnesses
- Implementation gap identified with actionable schema
- No false claims of established causality
- System correctly refuses to certify without proof

---

**Remember**: This system either establishes constructive mathematical proof OR reports honest implementation gaps. There is no middle ground via approximation or literal storage.

**CLF Obligation**: Every byte sequence has mathematical causality. If we can't prove it, the code is incomplete, not the mathematics.
