# teleport/clf_fb.py

from typing import List, Tuple, Optional, Literal, Dict
from teleport.clf_int import leb as leb_len, assert_integer_only
from teleport.guards import assert_boundary_types
from teleport.seed_format import OP_CONST, OP_STEP, OP_MATCH, OP_CBD256
from teleport.clf_canonical import (
    header_bits,
    emit_cbd_param_leb7_from_bytes,
    expand_cbd256_from_leb7,
    compute_cost_receipts,
    compute_cost_receipts_logical_cbd,
)

Mode = Literal["calc", "minimal"]

class CLFViolation(RuntimeError): ...
class PinViolation(RuntimeError): ...

# ---------- PINS & CONSTANTS ----------
CLF_ALPHA, CLF_BETA = 32, 1           # complexity envelope (value-independent)
RESIDUAL_PASSES_MAX = 1               # pinned residual budget
WINDOW_W = 32                         # pinned rolling hash window

# op registry must remain 1-byte leb_len
_OP_REGISTRY = (OP_CONST, OP_STEP, OP_MATCH, OP_CBD256)

def _pin_unit_lock():
    for op in _OP_REGISTRY:
        if leb_len(op) != 1:
            raise PinViolation(f"unit-lock drift: leb_len({op}) != 1")
    return True

_pin_unit_lock()  # run at import

# ---------- RECEIPTS ----------
def receipt_complexity(L:int, ops:int) -> Dict:
    max_ops = CLF_ALPHA + CLF_BETA*L
    return {
        "INPUT_LENGTH": L, "ACTUAL_OPS": ops,
        "MAX_ALLOWED_OPS": max_ops,
        "ENVELOPE_SATISFIED": ops <= max_ops,
        "ALPHA": CLF_ALPHA, "BETA": CLF_BETA,
    }

def receipt_bijection_ok(tokens, S:bytes) -> bool:
    # finalize CBD_LOGICAL → OP_CBD256(leb7), then decode and compare
    from teleport.clf_canonical import finalize_cbd_tokens, decode_CLF
    fin = finalize_cbd_tokens(tokens)
    return decode_CLF(fin) == S

# ---------- BUILDING BLOCKS ----------
def _c_cost(op_id:int, params:tuple, L:int) -> Dict:
    # pure arithmetic CAUS/END identity (no emission)
    return compute_cost_receipts(op_id, params, L)

def _c_cost_cbd_logical(seg_mv:memoryview, L:int) -> Dict:
    return compute_cost_receipts_logical_cbd(seg_mv, L)

# Deterministic interval representation used by builders
Token = Tuple[object, object, int, Dict, int]  # (op|label, params|mv, L, cost_info, pos)

class Builder:
    """
    Sealed, value-independent CLF function builder.
    Use this to assemble A or B constructions safely.
    """
    __slots__ = ("_S", "_L", "_tokens", "_ops_count")

    def __init__(self, S:bytes):
        assert_boundary_types(S)
        self._S = S
        self._L = len(S)
        self._tokens: List[Token] = []
        self._ops_count = 0

    # ---- safe emitters (validate semantics, compute costs, append) ----
    def add_CONST(self, pos:int, length:int, byte_val:int):
        assert 0 <= pos < self._L and length >= 2 and pos+length <= self._L
        assert 0 <= byte_val <= 255
        info = _c_cost(OP_CONST, (byte_val,), length)
        self._tokens.append((OP_CONST, (byte_val,), length, info, pos))
        self._ops_count += 1

    def add_STEP(self, pos:int, length:int, a0:int, d:int):
        assert 0 <= pos < self._L and length >= 3 and pos+length <= self._L
        assert 0 <= a0 <= 255 and 0 <= d <= 255
        info = _c_cost(OP_STEP, (a0, d), length)
        self._tokens.append((OP_STEP, (a0, d), length, info, pos))
        self._ops_count += 1

    def add_MATCH(self, pos:int, length:int, D:int):
        # left-context-only; params=(D,length)
        assert 0 <= pos < self._L and length > 0 and pos+length <= self._L
        assert D >= 1
        info = _c_cost(OP_MATCH, (D, length), length)
        self._tokens.append((OP_MATCH, (D, length), length, info, pos))
        self._ops_count += 1

    def add_CBD_LOGICAL(self, pos:int, length:int):
        assert 0 <= pos < self._L and length > 0 and pos+length <= self._L
        mv = memoryview(self._S)[pos:pos+length]
        info = _c_cost_cbd_logical(mv, length)
        self._tokens.append(("CBD_LOGICAL", mv, length, info, pos))
        self._ops_count += 1

    # ---- coalesce and fixpoint (O(N), value-independent) ----
    def coalesce(self):
        from teleport.clf_canonical import coalesce_tokens
        self._tokens = coalesce_tokens(self._tokens, memoryview(self._S))
        return self

    # ---- finalize CBD logical to OP_CBD256(leb7) ----
    def finalize(self):
        from teleport.clf_canonical import finalize_cbd_tokens
        self._tokens = finalize_cbd_tokens(self._tokens)
        return self

    # ---- receipts & totals ----
    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    def stream_bits(self) -> int:
        return sum(t[3]["C_stream"] for t in self._tokens)

    def totals(self) -> Dict:
        H = header_bits(self._L)
        C_stream = self.stream_bits()
        return {"H": H, "C_stream": C_stream, "TOTAL": H + C_stream, "RAW": 8*self._L}

    def receipts(self) -> Dict:
        t = self.totals()
        return {
            "TOTAL_BITS": t["TOTAL"],
            "RAW_BITS": t["RAW"],
            "MINIMALITY_OK": t["TOTAL"] < t["RAW"],
            "COVERAGE_OK": sum(tok[2] for tok in self._tokens) == self._L,
            "COMPLEXITY": receipt_complexity(self._L, self._ops_count),
        }

# ---------- A/B CONSTRUCTION HELPERS ----------
def build_A_exact(S:bytes) -> Builder:
    b = Builder(S)
    b.add_CBD_LOGICAL(0, len(S))   # exact logical CBD
    return b.coalesce()

def build_B_structural(S:bytes,
                       window:int = WINDOW_W,
                       residual_passes:int = RESIDUAL_PASSES_MAX) -> Builder:
    """
    Deterministic structural tiling:
      pass1: CONST/STEP/MATCH tiling + CBD gaps
      residual: at most 1 pass over largest gap (pinned)
    The caller supplies intervals; here we show canonical hooks.
    """
    from teleport.clf_canonical import _build_maximal_intervals, _materialize_intervals
    L = len(S)
    ctx_index = {}
    struct, gaps = _build_maximal_intervals(S, L, ctx_index, window)

    b = Builder(S)
    for s,e,t,prm in _materialize_intervals(struct, gaps):
        n = e-s
        if t == "CONST":
            b.add_CONST(s, n, prm)
        elif t == "STEP":
            a0,d = prm; b.add_STEP(s, n, a0, d)
        elif t == "MATCH":
            D = prm; b.add_MATCH(s, n, D)
        elif t == "CBD_GAP":
            b.add_CBD_LOGICAL(s, n)

    # optional residual (pinned budget 0/1)
    if residual_passes >= 1:
        gaps2 = [(s, s+n) for (op, prm, n, _ci, s) in b.tokens if isinstance(op, str) and op=="CBD_LOGICAL"]
        if gaps2:
            s,e = max(gaps2, key=lambda g: g[1]-g[0])
            # rebuild only if structural stream < gap CBD stream
            mv = memoryview(S)[s:e]
            exact_gap = compute_cost_receipts_logical_cbd(mv, e-s)["C_stream"]
            # (hook) Try a local structural rebuild here if you have a local pass helper
            # If not strictly better, keep CBD gap (consequence rail)
            _ = exact_gap

    return b.coalesce()

# ---------- UNIVERSAL DECISION ----------
def encode_minimal(S:bytes, *, emit_on_minimal:bool=True) -> List[Token]:
    """
    Universal CLF encoder using the function-builder:
      1) Build A & B independently
      2) Enforce superadditivity if B is CBD-only
      3) Choose minimal, apply single gate H+min < 8L
      4) Finalize CBD & validate bijection
    """
    assert_boundary_types(S)
    L = len(S)
    if L == 0:
        return []

    A = build_A_exact(S)
    B = build_B_structural(S)

    # CBD-only B guard (superadditivity)
    only_cbd_B = all(isinstance(t[0], str) and t[0] == "CBD_LOGICAL" for t in B.tokens)
    if only_cbd_B:
        A_stream = A.stream_bits()
        B_stream = B.stream_bits()
        if B_stream + 0 < A_stream:  # no tolerance; arithmetic is exact
            raise CLFViolation(f"CBD superadditivity violated: B={B_stream} < A={A_stream}")

    # choose minimal (tie → CBD/A)
    tA, tB = A.totals(), B.totals()
    chosen = A if tA["TOTAL"] <= tB["TOTAL"] else B
    C_min = min(tA["TOTAL"], tB["TOTAL"])
    RAW = 8*L

    if C_min >= RAW:
        # Uniform outcome; pick OPEN (or raise) globally and do not deviate elsewhere
        return []

    # finalize and validate bijection
    chosen.finalize()
    if not receipt_bijection_ok(chosen.tokens, S):
        raise CLFViolation("Bijection receipt failed after finalize/decode")

    # optional receipts (can be printed by caller)
    return chosen.tokens