#!/usr/bin/env bash
set -euo pipefail
FILE="test_data/pic2.jpg"
if [ ! -f "$FILE" ]; then
  echo "ERROR: $FILE not found" >&2; exit 2
fi

OUT=$(python3 CLF_MAXIMAL_VALIDATOR_FINAL.py "$FILE")
# Parse outputs (portable): L, leb, C, RAW, EMIT, receipt
L=$(stat -f%z "$FILE" 2>/dev/null || wc -c < "$FILE")
# integer leb: 1 if L==0 else ceil((bit_length(L))/7)
python3 - <<PY
import math,sys
L=$L
bl = 1 if L==0 else L.bit_length()
leb = 1 if L==0 else (bl + 6)//7
C = 88 + 8*leb
RAW = 10*L
emit = C < RAW
print(f"EXPECTED -> L={L}, leb={leb}, C={C}, RAW={RAW}, EMIT={emit}")
PY

echo "CALCULATOR -> $OUT" | sed 's/^[[:space:]]*//'
# Hard fail if EMIT is not True or constants drift:
python3 -c "
import re
txt='''$OUT'''
m=re.search(r'L=([\d,]+), leb=(\d+), C=(\d+) bits, RAW=([\d,]+) bits, EMIT=(True|False)', txt)
if not m: 
    print('ERROR: Could not parse calculator output'); exit(3)
L=int(m.group(1).replace(',','')); leb=int(m.group(2)); C=int(m.group(3)); RAW=int(m.group(4).replace(',','')); EMIT=(m.group(5)=='True')
bl=1 if L==0 else L.bit_length()
leb_exp=1 if L==0 else (bl+6)//7
C_exp=88+8*leb_exp
RAW_exp=10*L
assert leb==leb_exp, f'leb drift: got {leb} exp {leb_exp}'
assert C==C_exp, f'C drift: got {C} exp {C_exp}'
assert RAW==RAW_exp, f'RAW drift: got {RAW} exp {RAW_exp}'
assert EMIT==True, 'EMIT must be True for practical L'
print('✓ pic2 gate passed (leb, C, RAW, EMIT)')
"