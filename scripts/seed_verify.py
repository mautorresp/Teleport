# scripts/seed_verify.py
import argparse, hashlib, os
from teleport.seed_vm import expand, seed_cost
from teleport.costs import cost_lit

ARTIFACT_DIR = "test_artifacts"

def ensure_artifact_dir():
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

def sha256_hex_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--seed", required=True, help="Path to seed file (bytes)")
    ap.add_argument("--outname", required=True, help="Artifact filename inside test_artifacts/ (no auto extension)")
    ap.add_argument("--target-file", required=False, help="Optional target file for comparison")
    ap.add_argument("--target-sha", required=False, help="Optional reference sha256 (64 hex)")
    ap.add_argument("--hex-prefix", type=int, default=0, help="Print first N bytes of artifact in hex (0 = skip)")
    ap.add_argument("--alias", required=False, help="Optional second filename inside test_artifacts/ (no auto extension)")
    ap.add_argument("--window-start-hex", required=False, help="Arithmetic window start hex anchor (format-agnostic)")
    ap.add_argument("--window-end-hex", required=False, help="Arithmetic window end hex anchor (format-agnostic)")
    ap.add_argument("--window-header-len-offset", type=int, default=2, help="Offset after start anchor for big-endian 16-bit length (-1 to disable)")
    ap.add_argument("--require-positive-window", action="store_true", help="Exit code 2 if window_len <= 0")
    ap.add_argument("--lint-seed", action="store_true", help="Run seed domain linting before expansion")
    args = ap.parse_args()

    ensure_artifact_dir()

    # read seed
    with open(args.seed, "rb") as f:
        seed = f.read()

    # optional seed domain linting
    if args.lint_seed:
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "scripts/seed_lint.py", "--seed", args.seed], 
                              capture_output=True)
        if result.returncode != 0:
            print("seed_domain_violation=1")
            raise SystemExit(4)

    # expand
    out = expand(seed)
    N = len(out)
    c_seed = seed_cost(seed)
    c_lit = cost_lit(N)

    # compute receipts on 'out' in-memory first
    out_path = os.path.join(ARTIFACT_DIR, args.outname)
    out_sha = hashlib.sha256(out).hexdigest()
    out_size = len(out)

    print("bytes=", N)
    print("C_SEED=", c_seed)
    print("C_LIT=", c_lit)
    print("file_bytes=", out_size)
    print("sha256=", out_sha)

    # Optional arithmetic window check (format-agnostic, integers only)
    if args.window_start_hex and args.window_end_hex:
        b = out  # expanded bytes (not re-read)
        def off(hx):
            try: return b.find(bytes.fromhex(hx))
            except ValueError: return -1
        off_start = off(args.window_start_hex)
        off_end   = off(args.window_end_hex)
        hdr_end = -1
        length = -1
        if off_start >= 0:
            if args.window_header_len_offset is not None and args.window_header_len_offset >= 0:
                k = off_start + args.window_header_len_offset
                if k+1 < len(b):
                    length = (b[k] << 8) | b[k+1]  # big-endian 16-bit
                    if length >= 2:
                        hdr_end = off_start + args.window_header_len_offset + 2 + (length - 2)
                    else:
                        hdr_end = -1
            else:
                hdr_end = off_start + 2  # just after marker
        window_len = (off_end - hdr_end) if (off_end>=0 and hdr_end>=0) else -1
        print("off_start=", off_start)
        print("len_field=", length)
        print("hdr_end=", hdr_end)
        print("off_end=", off_end)
        print("window_len=", window_len)
        if args.require_positive_window and not (window_len > 0):
            print("window_violation=1")
            raise SystemExit(2)

    # Only after all optional checks pass (or not requested), write artifact
    with open(out_path, "wb") as g:
        g.write(out)

    # Optional hex prefix (evidence-only, no labels) - use in-memory data
    if args.hex_prefix and args.hex_prefix > 0:
        prefix = out[:args.hex_prefix]
        print("hex0=", prefix.hex())

    # Optional alias (no mutation: link or copy)
    if args.alias:
        alias_path = os.path.join(ARTIFACT_DIR, args.alias)
        try:
            if os.path.exists(alias_path):
                os.remove(alias_path)
            os.link(out_path, alias_path)  # hardlink: zero-copy, same bytes
        except Exception:
            # fallback: exact byte copy
            with open(out_path, "rb") as src, open(alias_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
        # Read back to assert identity numerically (still just bytes)
        alias_sha = sha256_hex_file(alias_path)
        alias_size = os.path.getsize(alias_path)
        print("alias_path=", os.path.join(ARTIFACT_DIR, args.alias))
        print("alias_sha=", alias_sha)
        print("alias_bytes=", alias_size)
        print("eq_alias_sha=", int(alias_sha == out_sha))
        print("eq_alias_bytes=", int(alias_size == out_size))

    if args.target_file:
        tgt_size = os.path.getsize(args.target_file)
        tgt_sha = sha256_hex_file(args.target_file)
        print("sha256_target=", tgt_sha)
        print("target_bytes=", tgt_size)
        print("eq_sha=", int(out_sha == tgt_sha))
        print("eq_bytes=", int(N == tgt_size))

    if args.target_sha:
        ref = args.target_sha.strip().lower()
        print("sha256_target=", ref)
        print("eq_sha=", int(out_sha == ref))

if __name__ == "__main__":
    main()
