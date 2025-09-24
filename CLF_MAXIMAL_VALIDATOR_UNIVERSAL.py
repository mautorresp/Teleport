# === CLF CALCULATOR-GRADE UNIVERSAL VALIDATOR (pinned math, DP, no drift) ===
# Stance: pure CLF math/logic, integer-only, NO FP, NO compression vocabulary.
# Operators:
#   10 = CONST-RUN  (encoded as STEP with step=0 in admissibility)
#   11 = STEP-RUN   (arithmetic progression)
#
# Unified admissibility law for a token over S[off:off+L):
# (i)   start == S[off]
# (ii)  if L>=2, step == (S[off+1] - S[off]) mod 256; if L=1 then step==0
# (iii) ∀i<L: S[off+i] == (start + i*step) mod 256
# (iv)  Maximality: left-max AND right-max
#       left-max:  off==0 OR S[off-1] != (start - step) mod 256
#       right-max: off+L==total_L OR S[off+L] != (start + L*step) mod 256
#
# Roles:
#   A = one whole-range token (CONST or STEP) or N/A
#   B = multi-token tiling that exactly covers the stream (Σ L_tok == L)
#
# Pricing (bits):
#   leb_len_u(n): 7-bit groups; leb_len_u(0)=1
#   H(L) = 16 + 8*leb_len_u(8*L)
#   C_CAUS(op, params[], L_tok) = 3 + 8*leb_len_u(op) + Σ 8*leb_len_u(param_i) + 8*leb_len_u(L_tok)
#   C_END(bitpos) = 3 + ((8 - ((bitpos + 3) % 8)) % 8)
#
# Decision gate:
#   C_min_total = H + min(complete_streams) ; EMIT iff C_min_total < 8*L  (strict inequality)
#
# Determinism:
#   - Hash the source of critical functions + BUILD_ID to produce a "deductor signature"
#   - Token list hashed (SHA256) and verified across two in-process runs
#
# DP Canonicalization (drift-killer):
#   - Enumerate at each offset the maximal lawful candidates (STEP if step!=0, CONST)
#   - Build suffix-feasible DP from the end
#   - Pick canonical token at each i using the order:
#       STEP (11) preferred over CONST (10),
#       then lexicographic params (start, step) ascending,
#       then longer length (desc)
#   - Guarantees a complete, canonical tiling (no stranding, no greediness traps)
#
# Receipts:
#   - Always print first 10 tokens and all tokens with L >= 32 (unless --quiet)
#   - Clause-by-clause re-check for those tokens
#
# CLI:
#   python3 CLF_MAXIMAL_VALIDATOR_UNIVERSAL.py <files...> [--quiet] [--export-prefix PREFIX]
#   Works for any binary: jpg, mp4, etc. Time/space are format-agnostic.

import argparse
import hashlib
import inspect
import os
import sys

BUILD_ID = "CLF_UNIVERSAL_DP_LOCK_20250923"

# ---------- Pinned math (bits) ----------
def leb_len_u(n: int) -> int:
    assert n >= 0
    if n == 0:
        return 1
    c = 0
    while n > 0:
        n >>= 7
        c += 1
    return c

def header_bits(L: int) -> int:
    return 16 + 8 * leb_len_u(8 * L)

def end_bits(bitpos: int) -> int:
    return 3 + ((8 - ((bitpos + 3) % 8)) % 8)

def caus_bits(op: int, params: list[int], L_tok: int) -> int:
    return 3 + 8 * leb_len_u(op) + sum(8 * leb_len_u(p) for p in params) + 8 * leb_len_u(L_tok)

# ---------- Unified admissibility ----------
def step_run_is_lawful(S: bytes, off: int, L: int, start: int, step: int, total_L: int):
    # (i) start matches
    if start != S[off]:
        return False, "clause_i_failed"

    # (ii) step deduction
    if L >= 2:
        expected_step = (S[off + 1] - S[off]) % 256
        if step != expected_step:
            return False, "clause_ii_failed"
    else:
        if step != 0:
            return False, "clause_ii_failed_L1"

    # (iii) progression holds
    for i in range(L):
        if S[off + i] != ((start + i * step) % 256):
            return False, f"clause_iii_failed_at_{i}"

    # (iv) strict maximality
    left_max  = (off == 0) or (S[off - 1] != ((start - step) % 256))
    right_max = (off + L == total_L) or (S[off + L] != ((start + L * step) % 256))
    if not left_max:
        return False, "left_maximality_failed"
    if not right_max:
        return False, "right_maximality_failed"

    return True, "lawful"

# ---------- Candidate enumeration (maximal only) ----------
def enumerate_maximal_candidates_at(S: bytes, off: int) -> list[tuple[int, list[int], int]]:
    """Return up to two candidates at 'off': STEP (if step!=0, L>=2) and CONST,
       both forced to maximal L and checked for strict admissibility."""
    L_total = len(S)
    if off >= L_total:
        return []

    candidates = []

    # STEP candidate (only if we have at least 2 bytes and non-zero step)
    if off + 1 < L_total:
        start = S[off]
        step = (S[off + 1] - S[off]) % 256
        if step != 0:
            j = off + 2
            # extend while arithmetic progression matches
            while j < L_total and S[j] == ((start + (j - off) * step) % 256):
                j += 1
            Ltok = j - off
            # L must be >= 2 for STEP
            ok, _ = step_run_is_lawful(S, off, Ltok, start, step, L_total)
            if ok:
                candidates.append((11, [start, step], Ltok))

    # CONST candidate (step=0)
    start = S[off]
    j = off + 1
    while j < L_total and S[j] == start:
        j += 1
    Ltok = j - off
    ok, _ = step_run_is_lawful(S, off, Ltok, start, 0, L_total)
    if ok:
        candidates.append((10, [start], Ltok))

    return candidates

# ---------- Canonical ordering ----------
def candidate_key(op: int, params: list[int], Ltok: int):
    # Canonical order: STEP (11) over CONST (10), then lexicographic params (ascending), then longer length (desc)
    # We invert op by (-op) so 11 ranks before 10; longer length by -Ltok.
    return (-op, tuple(params), -Ltok)

# ---------- DP deduction with suffix feasibility ----------
def deduce_tokens_dp(S: bytes) -> list[tuple[int, list[int], int]]:
    L = len(S)
    feasible = [False] * (L + 1)
    feasible[L] = True

    # For reconstruction
    best_choice: list[tuple[int, list[int], int] | None] = [None] * (L + 1)

    # Pass 1: feasibility only
    for i in range(L - 1, -1, -1):
        cands = enumerate_maximal_candidates_at(S, i)
        # Try in canonical order and mark feasible if any leads to feasible suffix
        for op, params, Lt in sorted(cands, key=lambda x: candidate_key(*x)):
            if feasible[i + Lt]:
                feasible[i] = True
                break

    if not feasible[0]:
        # Fatal: no complete tiling exists under strict admissibility/maximality
        # Show local context to make the failure falsifiable.
        ctx = list(S[max(0, 0 - 8):min(L, 0 + 8)])
        print("ABORT: suffix infeasible from offset 0")
        print(f"Context S[0:8]: {ctx}")
        sys.exit(1)

    # Pass 2: choose canonical tokens using DP feasibility
    i = 0
    tokens: list[tuple[int, list[int], int]] = []
    while i < L:
        cands = enumerate_maximal_candidates_at(S, i)
        chosen = None
        for op, params, Lt in sorted(cands, key=lambda x: candidate_key(*x)):
            if feasible[i + Lt]:
                chosen = (op, params, Lt)
                break
        if chosen is None:
            # Should not happen because feasible[0] true; but guard anyway
            print(f"FATAL: no canonical candidate at {i} despite feasibility")
            print(f"Context: S[{max(0,i-8)}:{min(L,i+8)}] = {list(S[max(0,i-8):min(L,i+8)])}")
            sys.exit(1)
        tokens.append(chosen)
        i += chosen[2]

    # Coverage assertion
    total_cov = sum(Lt for _, _, Lt in tokens)
    if total_cov != L:
        print("ABORT: COVERAGE MISMATCH")
        print(f"Sum L_tok = {total_cov}, L = {L}")
        sys.exit(1)

    return tokens

# ---------- Expansion (locality-only) ----------
def expand_token(op: int, params: list[int], L_tok: int) -> bytes:
    if op == 10:  # CONST-RUN
        start = params[0]
        return bytes([start]) * L_tok
    if op == 11:  # STEP-RUN
        start, step = params
        return bytes(((start + i * step) % 256) for i in range(L_tok))
    raise ValueError(f"Unknown op {op}")

# ---------- Determinism guard ----------
def deductor_signature() -> str:
    parts = [
        inspect.getsource(leb_len_u),
        inspect.getsource(header_bits),
        inspect.getsource(end_bits),
        inspect.getsource(caus_bits),
        inspect.getsource(step_run_is_lawful),
        inspect.getsource(enumerate_maximal_candidates_at),
        inspect.getsource(candidate_key),
        inspect.getsource(deduce_tokens_dp),
        inspect.getsource(expand_token),
        BUILD_ID,
    ]
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()

def tokens_sha(tokens: list[tuple[int, list[int], int]]) -> str:
    # Stable textual form (no floats, no pretty). Params are ints only.
    s = repr(tokens)
    return hashlib.sha256(s.encode()).hexdigest()

# ---------- Receipts ----------
def print_admissibility_check(S: bytes, off: int, Lt: int, op: int, params: list[int]):
    start = params[0]
    step = params[1] if op == 11 else 0
    L = len(S)

    print(f"  Clause (i): start({start}) == S[{off}]({S[off]}) → {start == S[off]}")
    if Lt >= 2:
        expected_step = (S[off + 1] - S[off]) % 256
        print(f"  Clause (ii): L≥2, step({step}) == expected({expected_step}) → {step == expected_step}")
    else:
        print(f"  Clause (ii): L=1, step({step}) == 0 → {step == 0}")

    ok_iii = True
    for i in range(min(Lt, 5)):
        expected = (start + i * step) % 256
        ok_iii &= (S[off + i] == expected)
    print(f"  Clause (iii): First {min(Lt,5)} positions match progression → {ok_iii}")

    left_max  = (off == 0) or (S[off - 1] != ((start - step) % 256))
    right_max = (off + Lt == L) or (S[off + Lt] != ((start + Lt * step) % 256))
    if off > 0:
        print(f"  Left max: S[{off-1}]({S[off-1]}) != {(start - step)%256} → {left_max}")
    else:
        print(f"  Left max: off==0 → {left_max}")

    if off + Lt < L:
        print(f"  Right max: S[{off+Lt}]({S[off+Lt]}) != {(start + Lt*step)%256} → {right_max}")
    else:
        print(f"  Right max: off+L==total_L → {right_max}")

    print(f"  Clause (iv): Maximal ({left_max} AND {right_max}) → {left_max and right_max}")

def print_receipts(S: bytes, tokens: list[tuple[int, list[int], int]], quiet: bool):
    if quiet:
        return

    print("RECEIPTS:")
    off = 0
    for idx, (op, params, Lt) in enumerate(tokens):
        if idx < 10 or Lt >= 32:
            print(f"\nTOKEN B_{idx}: off={off} L={Lt} op={op} params={params}")
            if Lt >= 32:
                block = list(S[off:off + min(32, Lt)])
                print("  Raw bytes (first 32):", block)
                print("  Distinct values:", len(set(block)))
            print_admissibility_check(S, off, Lt, op, params)
        off += Lt

# ---------- Bijection & algebra ----------
def verify_bijection(S: bytes, tokens: list[tuple[int, list[int], int]]):
    recon = b''.join(expand_token(op, params, Lt) for op, params, Lt in tokens)
    s1 = hashlib.sha256(S).hexdigest()
    s2 = hashlib.sha256(recon).hexdigest()
    print("BIJECTION IDENTITY:")
    print(f"  Original SHA:      {s1}")
    print(f"  Reconstructed SHA: {s2}")
    if s1 != s2:
        print("ABORT: DEDUCTION→EXPANSION IDENTITY FAILED")
        sys.exit(1)

    # Re-deduce and compare token SHAs
    tokens2 = deduce_tokens_dp(recon)
    t1 = tokens_sha(tokens)
    t2 = tokens_sha(tokens2)
    print(f"  Original tokens SHA:   {t1}")
    print(f"  Re-deduced tokens SHA: {t2}")
    if t1 != t2:
        print("ABORT: RE-DEDUCTION INCONSISTENCY")
        sys.exit(1)
    print("  BIJECTION: PASS")

def roles_and_algebra(S: bytes, tokens: list[tuple[int, list[int], int]]):
    L = len(S)
    RAW = 8 * L
    H = header_bits(L)

    # A-path (whole range single op) or N/A
    A_stream = None
    if L == 0:
        A_caus = 0
        A_stream = A_caus + end_bits(A_caus)
        print(f"A-role: ZERO whole-range, A_stream={A_stream}")
    elif all(b == S[0] for b in S):
        A_caus = caus_bits(10, [S[0]], L)
        A_stream = A_caus + end_bits(A_caus)
        print(f"A-role: CONST whole-range, A_stream={A_stream}")
    elif L >= 2:
        start = S[0]
        step = (S[1] - S[0]) % 256
        if all(S[i] == ((start + i * step) % 256) for i in range(L)):
            A_caus = caus_bits(11, [start, step], L)
            A_stream = A_caus + end_bits(A_caus)
            print(f"A-role: STEP whole-range, A_stream={A_stream}")
        else:
            print("A-role: N/A")
    else:
        print("A-role: N/A")

    # B-path (multi-token)
    B_caus = sum(caus_bits(op, params, Lt) for op, params, Lt in tokens)
    B_end  = end_bits(B_caus)
    B_stream = B_caus + B_end

    candidates = []
    if A_stream is not None:
        candidates.append(H + A_stream)
    candidates.append(H + B_stream)
    C_min_total = min(candidates)

    print("COSTS & GATE:")
    print(f"  H = {H}")
    print(f"  B_caus = {B_caus}")
    print(f"  B_end = {B_end}")
    print(f"  B_stream = {B_stream}")
    print(f"  Candidates: {candidates}")
    print(f"  RAW = {RAW}")
    EMIT = C_min_total < RAW
    print(f"  EMIT: {C_min_total} < {RAW} → {EMIT}")
    return H, B_caus, B_end, B_stream, C_min_total, RAW, EMIT

# ---------- Determinism check ----------
def verify_determinism(S: bytes):
    sig = deductor_signature()
    t1 = tokens_sha(deduce_tokens_dp(S))
    t2 = tokens_sha(deduce_tokens_dp(S))
    if t1 != t2:
        print("ABORT: NON-DETERMINISTIC TOKENIZATION (same process)")
        sys.exit(1)
    return sig, t1

# ---------- Export helpers (optional) ----------
def maybe_write_exports(prefix: str | None, file_path: str, summary: dict):
    if not prefix:
        return
    base = os.path.basename(file_path)
    stem = os.path.splitext(base)[0]
    # 4 concise exports: FULL_EXPLANATION, BIJECTION_EXPORT, PREDICTION_EXPORT, RAILS_AUDIT (compact text blobs)
    # We keep it minimal here; the math is already proven by the main run.
    out_full = f"{prefix}_FULL_EXPLANATION.txt"
    out_bij  = f"{prefix}_BIJECTION_EXPORT.txt"
    out_pred = f"{prefix}_PREDICTION_EXPORT.txt"
    out_rail = f"{prefix}_RAILS_AUDIT.txt"

    # Append summaries per file (one-liners)
    with open(out_full, "a") as f:
        f.write(f"{summary['name']}: L={summary['L']} tokens={summary['tokens']} "
                f"H={summary['H']} B_caus={summary['B_caus']} B_end={summary['B_end']} "
                f"B_stream={summary['B_stream']} C_total={summary['C_min_total']} RAW={summary['RAW']} "
                f"EMIT={summary['EMIT']} tokenSHA={summary['token_sha']} sig={summary['signature']}\n")
    with open(out_bij, "a") as f:
        f.write(f"{summary['name']}: tokenSHA={summary['token_sha']} bijection=PASS\n")
    with open(out_pred, "a") as f:
        f.write(f"{summary['name']}: ALG_EQ (pricing math executed) ; C_total={summary['C_min_total']}\n")
    with open(out_rail, "a") as f:
        f.write(f"{summary['name']}: roles=A/B enforced; admissibility clauses (i)-(iv) enforced; DP canonicalization; determinism guard\n")

# ---------- Per-file validation ----------
def validate_one_file(path: str, quiet: bool, export_prefix: str | None):
    if not os.path.exists(path):
        print(f"ABORT: FILE NOT FOUND: {path}")
        sys.exit(1)

    print("=" * 60)
    print(f"VALIDATING: {os.path.basename(path)}")
    print("=" * 60)

    S = open(path, "rb").read()
    L = len(S)
    print(f"Input: L={L} bytes")

    print("DETERMINISM GUARD:")
    sig, toksha = verify_determinism(S)
    print(f"  DEDUCTOR_SIGNATURE: {sig}")
    print(f"  Token SHA: {toksha}")

    tokens = deduce_tokens_dp(S)
    if not quiet:
        print(f"B-tokens: {len(tokens)}")

    print_receipts(S, tokens, quiet)
    verify_bijection(S, tokens)
    H, B_caus, B_end, B_stream, C_min_total, RAW, EMIT = roles_and_algebra(S, tokens)

    # Compact one-line summary
    print(f"SUMMARY: name={os.path.basename(path)} L={L} tokens={len(tokens)} "
          f"H={H} B_caus={B_caus} B_end={B_end} B_stream={B_stream} "
          f"C_total={C_min_total} RAW={RAW} EMIT={EMIT} tokenSHA={toksha}")

    # Optional exports
    summary = dict(
        name=os.path.basename(path),
        L=L,
        tokens=len(tokens),
        H=H,
        B_caus=B_caus,
        B_end=B_end,
        B_stream=B_stream,
        C_min_total=C_min_total,
        RAW=RAW,
        EMIT=EMIT,
        token_sha=toksha,
        signature=sig,
    )
    maybe_write_exports(export_prefix, path, summary)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="CLF Universal Maximal Validator (calculator-grade, DP, no drift)")
    ap.add_argument("files", nargs="+", help="Input files (any binary: jpg, mp4, ...)")
    ap.add_argument("--quiet", action="store_true", help="Suppress long receipts; still shows hashes & algebra")
    ap.add_argument("--export-prefix", default=None, help="Write compact 4-artifact summaries with given prefix")
    args = ap.parse_args()

    for p in args.files:
        validate_one_file(p, args.quiet, args.export_prefix)

if __name__ == "__main__":
    main()