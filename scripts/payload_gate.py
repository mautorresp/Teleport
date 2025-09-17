# scripts/payload_gate.py
import sys, pathlib

def be16(b, i): return (b[i] << 8) | b[i+1]

def find(b, hx):
    x = b.find(bytes.fromhex(hx))
    return x

def main(path):
    b = pathlib.Path(path).read_bytes()
    n = len(b)
    off_sos = find(b, "ffda")
    off_eoi = find(b, "ffd9")
    print("len=", n)
    print("off_SOS=", off_sos)
    print("off_EOI=", off_eoi)
    if off_sos is None or off_sos < 0 or off_sos+3 >= n or off_eoi is None or off_eoi < 0:
        print("entropy_len=", -1)
        sys.exit(1)
    L = be16(b, off_sos+2)
    hdr_end = off_sos + 2 + L
    ent = off_eoi - hdr_end
    print("sos_len=", L, "hdr_end=", hdr_end, "entropy_len=", ent)
    # Gate: require at least 1 byte of entropy
    if ent <= 0:
        sys.exit(2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/payload_gate.py <file>")
        sys.exit(99)
    main(sys.argv[1])
