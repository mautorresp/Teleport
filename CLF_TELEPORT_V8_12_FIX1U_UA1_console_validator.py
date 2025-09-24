# === CLF V8.12 FIX1U — UNIVERSAL, CONSOLE-DRIVEN, THEOREM-LOCKED ===
# BAN: "compress", "entropy", "random", FP. Use only CLF/Teleport math.
# SEQUENCE: after each STEP, print the block; if numbers mismatch, FIX BEFORE STEP+1.

RUN_TAG = "V8_12_FIX1U_UA1"

# ---------- PINS (must not change) ----------
def leb_len_u(n:int)->int:
    assert n>=0
    if n==0: return 1
    c=0
    while n>0:
        n >>= 7
        c += 1
    return c

def header_bits(L:int)->int: return 16 + 8*leb_len_u(8*L)

def end_bits(bitpos:int)->int: return 3 + ((8 - ((bitpos+3)%8))%8)

def caus_bits(op:int, params:list[int], L_tok:int)->int:
    return 3 + 8*leb_len_u(op) + sum(8*leb_len_u(p) for p in params) + 8*leb_len_u(L_tok)

# ---------- UNIVERSAL OBJECTS ----------
OBJECTS = [
  ("pic1.jpg", "PATH_OR_FAIL"),        # real-world; do not special-case
  ("S_const_50.bin", "SYNTH_CONST_50"),
  ("S_step_256.bin", "SYNTH_STEP_256")
]

# If synthetic files don't exist, create them deterministically
def ensure_synthetic():
    import os
    if not os.path.exists("S_const_50.bin"):
        open("S_const_50.bin","wb").write(bytes([0x42])*50)
    if not os.path.exists("S_step_256.bin"):
        a0,d = 7,3
        open("S_step_256.bin","wb").write(bytes(((a0 + i*d)&255) for i in range(256)))

# ---------- DEDUCT/EXPAND (A/B) ----------
# A: exactly one token over the whole range or N/A.
# B: structural tiling; multiple tokens allowed; coverage must be exact and disjoint.
# Both must be self-contained (no RAW read-back), integer-only.

def deduct_A(S:bytes):
    # Return either None (N/A) or [(op, params[], L)] with LEN==1 and L==len(S)
    # Implement CONST whole-range; STEP whole-range. No heuristics beyond literal tests.
    L=len(S)
    if L==0: return [(2, [], 0)]  # ZERO whole-range allowed
    # CONST whole-range
    if all(b==S[0] for b in S):
        return [(10, [S[0]], L)]   # op=10 CONST-RUN whole-range
    # STEP whole-range a_i = a0 + i*d (mod 256)
    a0=S[0]
    if L>=2:
        d = (S[1]-S[0])&255
        ok = all(S[i]==((a0 + i*d)&255) for i in range(L))
        if ok: return [(11, [a0,d], L)]
    return None

def expand_token(op, params, L_tok):
    # CONST-RUN via STEP-RUN with d=0 is allowed if op uses that convention.
    if op==10:  # CONST-RUN
        assert len(params)==1
        return bytes([params[0]])*L_tok
    if op==11:  # STEP-RUN
        assert len(params)==2
        a0,d = params
        return bytes(((a0 + i*d)&255) for i in range(L_tok))
    raise AssertionError("UNKNOWN_OP")

def deduct_B(S:bytes):
    # Deterministic tiling with maximal runs of STEP-RUN; fallback to CONST-RUN.
    # MUST be self-contained (params deduced from S segments), exact coverage, no overlaps.
    L=len(S); i=0; toks=[]
    while i<L:
        # Try maximal STEP-RUN from i
        if i+1<L:
            a0=S[i]; d=(S[i+1]-S[i])&255
            j=i+2
            while j<L and S[j]==((a0 + (j-i)*d)&255): j+=1
            step_len = j-i
        else:
            step_len=1
        if step_len>=2:
            toks.append((11,[S[i], (S[i+1]-S[i])&255], step_len))
            i=j
            continue
        # Fallback: maximal CONST-RUN
        b=S[i]; j=i+1
        while j<L and S[j]==b: j+=1
        const_len = j-i
        toks.append((10,[b], const_len))
        i=j
    return toks

def check_legality_and_prices(S, toks, who):
    # Print full receipts for EVERY token: offset, L, params, sha(segment), sha(expansion), equality
    import hashlib
    L=len(S); off=0; caus=0
    for idx,(op,params,Lt) in enumerate(toks):
        seg = S[off:off+Lt]
        seg_sha = hashlib.sha256(seg).hexdigest()
        exp = expand_token(op, params, Lt)
        exp_sha = hashlib.sha256(exp).hexdigest()
        eq = (seg_sha==exp_sha)
        print(f"[{who}_{idx}] off={off} L={Lt} op={op} params={params} "
              f"segSHA={seg_sha} expSHA={exp_sha} ok={eq}")
        assert eq, f"{who}[{idx}] expansion mismatch"
        c = caus_bits(op, params, Lt)
        print(f"[{who}_{idx}] C_CAUS={c}")
        caus += c
        off += Lt
    assert off==L, f"{who} coverage mismatch"
    end = end_bits(caus)
    print(f"{who}_caus={caus}  {who}_end={end}  {who}_stream={caus+end}")
    return caus, end, caus+end

def run_one(path):
    import hashlib, os
    assert os.path.exists(path), f"Missing {path}"
    S=open(path,"rb").read(); L=len(S); RAW=8*L
    H=header_bits(L)
    print(f"\n=== OBJ {path}  L={L} RAW={RAW} H={H} ===")

    # A-path
    A_toks = deduct_A(S)
    if A_toks is None:
        print("A_kind=N/A"); A_stream=None
    else:
        assert len(A_toks)==1 and A_toks[0][2]==L, "A must be single whole-range"
        Ac, Ae, A_stream = check_legality_and_prices(S, A_toks, "A")
        print(f"Pi_A={A_stream}  Pi_A_eq={True}")

    # B-path
    B_toks = deduct_B(S)
    Bc, Be, B_stream = check_legality_and_prices(S, B_toks, "B")
    print(f"Pi_B={B_stream}  Pi_B_eq={True}")

    # Algebra (complete paths only)
    candidates=[]
    if A_stream is not None: candidates.append(H + A_stream)
    candidates.append(H + B_stream)
    C_min_total = min(candidates)
    C_min_via_streams = H + min((A_stream if A_stream is not None else 10**18), B_stream)
    print(f"ALG_EQ={(C_min_total==C_min_via_streams)}  ({C_min_total} vs {C_min_via_streams})")

    # Decision gate
    decision = ("EMIT" if C_min_total < RAW else "CAUSEFAIL")
    print(f"DECISION={decision}  C_total={C_min_total} RAW={RAW}")

# ---------- MAIN ----------
if __name__=="__main__":
    ensure_synthetic()
    for p,_ in OBJECTS:
        run_one(p)