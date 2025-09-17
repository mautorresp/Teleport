# scripts/inspect_markers.py
import sys, struct

MARKERS = {
    0xD8: "SOI", 0xD9: "EOI", 0xE0: "APP0",
    0xDB: "DQT", 0xC0: "SOF0", 0xC4: "DHT", 0xDA: "SOS"
}

def be16(b, i): return (b[i] << 8) | b[i+1]

def main(path):
    b = open(path, "rb").read()
    n = len(b)
    print("len=", n)
    i = 0
    while i < n:
        if i+1 >= n: break
        if b[i] != 0xFF:
            i += 1
            continue
        # skip fill bytes FF FF ...
        j = i
        while j < n and b[j] == 0xFF: j += 1
        if j >= n: break
        m = b[j]
        name = MARKERS.get(m, f"FF{m:02X}")
        off = i
        i = j + 1
        # standalone markers (SOI/EOI) have no length
        if m in (0xD8, 0xD9):  # SOI, EOI
            print(f"{name} off={off}")
            continue
        if i+1 >= n:
            print(f"{name} off={off}  [truncated length]")
            break
        L = be16(b, i)
        seg_start = i + 2
        seg_end = seg_start + L - 2  # length includes itself
        print(f"{name} off={off} len={L} seg=[{seg_start}:{seg_end})")
        if seg_end > n:
            print(f"  !!! segment overruns file (ends at {seg_end} > {n})")
            break
        if m == 0xDA:  # SOS: after header, entropy starts; next marker must be EOI or RST/etc.
            print(f"  entropy starts at {seg_end}")
        i = seg_end

if __name__ == "__main__":
    main(sys.argv[1])
