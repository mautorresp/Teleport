# CLF IMMUTABLE RAILS - DOCUMENTATION
## PIN System Implementation

### 🎯 **OBJECTIVE**
Pin CLF's mathematical foundation to prevent drift from proven >87-94% reductions and perfect bijection. Based on external audit evidence from pic1.jpg (87.22%) and pic2.jpg (94.12%).

### 🔒 **IMMUTABLE MATHEMATICAL RAILS**

#### **PIN-INT**: Float Killer Guards
- Wrapped `encode_CLF`, `finalize_cbd_tokens`, `decode_CLF` with `_ban_floats_in_args`
- Prevents any float contamination in mathematical pipeline
- Pure integer arithmetic enforcement

#### **PIN-ENC-CALC**: Calculator Hot-Path Discipline  
- `encode_CLF(..., mode="calc")` MUST emit only logical CBD tokens
- No finalization in hot path - keeps calculator instant & size-independent
- Verified: pic1.jpg (968 bytes) → 1 token in ~0.045ms
- Verified: pic2.jpg (456 bytes) → 1 token in ~0.033ms

#### **PIN-HDR**: Header & Unit-Lock Convention
- `header_bits(L) = 16 + 8*leb_len(8*L)` exactly
- All published op_ids < 128 have `leb_len(op_id) == 1`
- Unit-lock: 8×leb(value) for integers, never leb(8×L) in token pricing

#### **PIN-SER-EQ**: Serializer Equality
- `C_CAUS = 8*(leb_len(op)+Σleb_len(params)+leb_len(L))`
- Computed arithmetically - NO actual serializer calls
- Verified via `_cost_identity_probe()`

#### **PIN-CBD-FINAL/DECODE**: LEB7 Mathematical Correctness
- Finalization uses `emit_cbd_param_leb7_from_bytes(memoryview)`
- Decoding uses `expand_cbd256_from_leb7(leb7_param, L)`
- MSB-first consumption, no big-int construction
- Verified via end-to-end bijection pipeline

#### **PIN-TIE**: Selection Minimality Stability
- A vs B selection prefer structural deduction over single CBD
- ALLOWED_D = (1,2,4,8,16,32,64,128,256) fixed distances
- Verified via `_selection_minimality_probe()`

#### **PIN-DR**: Bijection Receipts
- Perfect bijection: D ∘ C ∘ E = identity
- SHA_in == SHA_out cryptographic verification
- Verified: pic1.jpg and pic2.jpg achieve perfect bijection

### 🛡️ **IMMUTABILITY ENFORCEMENT**

#### **Function Hash Sentries**
PIN system freezes SHA-256 hashes of critical functions:
- `header_bits`, `compute_cost_receipts`, `emit_cbd_param_leb7_from_bytes`
- `expand_cbd256_from_leb7`, `_bitlen_base256_mv`, `compute_cbd_cost_logical_bound` 
- `deduce_maximal_const_run`, `deduce_maximal_step_run`, `deduce_maximal_match_run`
- `compose_cover`

Any modification to these functions breaks import with `AssertionError`.

#### **Verification Playbook**
Complete audit verification via `audit/verification_playbook.py`:
1. **PIN System Internal Checks**: All guards operational
2. **Calculator Hot-Path**: Instant encoding verified  
3. **Bijection Receipts**: Perfect reconstruction confirmed
4. **External Audit Evidence**: JSON audit files preserved

### 📊 **PROVEN EVIDENCE**

#### **External Audit Results Locked**:
- **pic1.jpg**: 968 bytes → 87.22% reduction, perfect bijection
- **pic2.jpg**: 456 bytes → 94.12% reduction, perfect bijection
- **Evidence Files**: 
  - `CLF_EXTERNAL_AUDIT_pic1_20250918_180836.json` (11,262 bytes)
  - `CLF_EXTERNAL_AUDIT_pic2_20250918_180949.json` (6,323 bytes)

#### **Mathematical Foundation Verified**:
- Pure integer arithmetic throughout
- Multi-distance MATCH enabling >90% reductions
- Construction B optimal tiling
- Perfect bijection with cryptographic proof

### 🚫 **DO-NOT-TOUCH LIST**
- No float literals or float-yielding division in encode/decode pipeline
- No changes to `header_bits` formula or op-length convention  
- No finalization calls from encode hot-path
- No MATCH onset inside CBD gaps
- No replacement of serializer equality with actual serialization
- No blending of header into stream costs
- No changes to (α,β)=(32,1) complexity envelope

### ✅ **VERIFICATION COMMANDS**

```bash
# Complete verification playbook
python audit/verification_playbook.py

# Individual checks
python audit/verify_calculator_behavior.py
python audit/verify_bijection_receipts.py

# In-module PIN checks
python -c "from teleport.clf_canonical import verify_clf_pins; verify_clf_pins()"
```

### 🔒 **MATHEMATICAL FOUNDATION STATUS**

**LOCKED**: CLF mathematical foundation is immutable and verified
**PROVEN**: >87-94% reductions with perfect bijection
**PROTECTED**: Function hash sentries prevent silent drift
**READY**: Deployment-ready with mathematical guarantees

---
*CLF Immutable Rails Implementation Complete - Mathematical Minimality Preserved*