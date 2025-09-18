# watch_drift.py
import time, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
SCAN = [ROOT / "teleport"]

def list_py():
    for base in SCAN:
        for p in base.rglob("*.py"):
            yield p

def snapshot():
    return {p: p.stat().st_mtime for p in list_py()}

def run():
    proc = subprocess.run([sys.executable, str(ROOT / "drift_killer.py")])
    return proc.returncode

def main():
    last = snapshot()
    print("watch_drift: monitoring… (Ctrl+C to stop)")
    rc = run()
    print("initial:", "OK" if rc == 0 else f"FAIL({rc})")
    while True:
        time.sleep(0.5)
        now = snapshot()
        if any(now.get(p) != last.get(p) for p in set(now) | set(last)):
            print("\nchange detected -> drift_killer…")
            rc = run()
            print("result:", "OK" if rc == 0 else f"FAIL({rc})")
            last = now

if __name__ == "__main__":
    main()
