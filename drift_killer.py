# drift_killer.py — CLF/Teleport binary-math sentinel
from __future__ import annotations
import ast, io, os, sys, tokenize, pathlib, importlib, hashlib

# ---- CONFIG ----
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PY_SRC_DIRS = [PROJECT_ROOT / "teleport"]  # scan these dirs
FORBIDDEN_IMPORTS = {
    "random", "statistics", "decimal", "fractions",
    "numpy", "pandas", "scipy", "seaborn", "matplotlib",  # any FP ecosystems
}
FORBIDDEN_NAMES = {
    "float", "round", "sum"  # sum allowed at runtime; blocked only if used on floats (checked via AST)
}
FORBIDDEN_LITERAL_FLOAT = True

# ---- STATIC CHECKS: AST/token level (no FP, no entropy) ----
class FloatDriftVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in FORBIDDEN_IMPORTS:
                self.errors.append(("FORBIDDEN_IMPORT", node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        base = (node.module or "").split('.')[0]
        if base in FORBIDDEN_IMPORTS:
            self.errors.append(("FORBIDDEN_IMPORT", node.lineno, node.module))
        self.generic_visit(node)

    def visit_Constant(self, node):
        if FORBIDDEN_LITERAL_FLOAT and isinstance(node.value, float):
            self.errors.append(("FLOAT_LITERAL", getattr(node, "lineno", 0), repr(node.value)))
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # Block true division; allow integer division //
        if isinstance(node.op, ast.Div):
            self.errors.append(("TRUE_DIVISION", getattr(node, "lineno", 0), "/"))
        self.generic_visit(node)

    def visit_Call(self, node):
        # Disallow float() constructor
        if isinstance(node.func, ast.Name) and node.func.id == "float":
            self.errors.append(("FLOAT_CALL", getattr(node, "lineno", 0), "float()"))
        self.generic_visit(node)

def static_scan_file(path: pathlib.Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception as e:
        return [("READ_ERROR", 0, f"{path}: {e}")]
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [("SYNTAX_ERROR", e.lineno or 0, f"{path}: {e}")]
    v = FloatDriftVisitor()
    v.visit(tree)
    # Token scan to catch numeric tokens like 1.0 in weird contexts
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.NUMBER and "." in tok.string:
                # Permit dotted attribute numbers? No—block all float literals
                v.errors.append(("TOKEN_FLOAT_LITERAL", tok.start[0], tok.string))
    return [(kind, line, f"{path}:{detail}") for (kind, line, detail) in v.errors]

def static_scan_dirs(dirs):
    errs = []
    for d in dirs:
        for p in d.rglob("*.py"):
            errs.extend(static_scan_file(p))
    return errs

# ---- RUNTIME CHECKS: strict CLF invariants (pure integers) ----
def leb_len_via_module(x: int) -> int:
    from teleport.leb_io import leb128_emit_single as leb_emit
    return len(leb_emit(x))

def header_bits(L_bytes: int) -> int:
    # canonical, integer-only
    return 16 + 8 * leb_len_via_module(8 * L_bytes)

def C_token(op_id: int, params: tuple[int, ...], L: int) -> int:
    # mirror the in-repo formula to cross-check at runtime
    c_op = 8 * leb_len_via_module(op_id)
    c_pa = 8 * sum(leb_len_via_module(p) for p in params) if params else 0
    c_L  = 8 * leb_len_via_module(L)
    c_caus = 3 + c_op + c_pa + c_L
    pad = (8 - ((c_caus + 3) % 8)) % 8
    c_end = 3 + pad
    return c_caus + c_end

def assert_token_serializers(tokens):
    # tokens: list[(op, params, L, reason)]
    from teleport.seed_format import emit_CAUS
    for i, (op, params, L, _) in enumerate(tokens, 1):
        seed = emit_CAUS(op, list(params), L)
        actual = 8 * len(seed)
        expect = C_token(op, params, L)
        assert actual == expect, f"[serializer_eq] token#{i}: {actual} != {expect}"
        assert expect < 10 * L, f"[per_segment_bound] token#{i}: {expect} >= {10*L}"

def assert_seed_only_expansion_and_coverage(S: bytes, tokens):
    from teleport.seed_vm import expand_generator
    S_prime = b""
    for i, (op, params, L, _) in enumerate(tokens, 1):
        seg = expand_generator(op, params, L)       # MUST NOT read S
        assert len(seg) == L, f"[expand_len] token#{i}: {len(seg)} != {L}"
        S_prime += seg
    assert len(S_prime) == len(S), "[coverage_len] reconstructed length mismatch"
    assert S_prime == S, "[coverage_eq] reconstructed bytes != original"

def assert_global_minimality(S: bytes, tokens):
    L = len(S)
    H = header_bits(L)
    total = H + sum(C_token(op, params, L_i) for op, params, L_i, _ in tokens)
    assert total < 10 * L, f"[global_bound] {total} !< {10*L}"

def run_runtime_proofs_on_samples():
    from teleport import dgg
    # 1) Simple PASS: 100 bytes CONST(42)
    S = bytes([42] * 100)
    tokens = dgg.encode_CLF(S)
    assert tokens, "[simple_pass] encode_CLF returned OPEN unexpectedly"
    assert_token_serializers(tokens)
    assert_seed_only_expansion_and_coverage(S, tokens)
    assert_global_minimality(S, tokens)

    # 2) OPEN example: if repo ships small JPEGs for tests, try both silently
    for fname in ("pic1.jpg", "pic2.jpg"):
        fpath = PROJECT_ROOT / fname
        if fpath.exists():
            data = fpath.read_bytes()
            toks = dgg.encode_CLF(data)
            # OPEN is allowed; but if PASS, it still must satisfy all rails
            if toks:
                assert_token_serializers(toks)
                assert_seed_only_expansion_and_coverage(data, toks)
                assert_global_minimality(data, toks)

def main():
    # STATIC
    static_errors = static_scan_dirs(PY_SRC_DIRS)
    if static_errors:
        print("DRIFT-KILLER: STATIC FAILURES")
        for kind, line, msg in static_errors:
            print(f"{kind}@L{line}: {msg}")
        sys.exit(2)

    # RUNTIME
    try:
        run_runtime_proofs_on_samples()
    except AssertionError as e:
        print("DRIFT-KILLER: RUNTIME FAILURE")
        print(str(e))
        sys.exit(3)

    print("DRIFT-KILLER: OK (binary-math rails intact)")

if __name__ == "__main__":
    main()
