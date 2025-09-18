#!/usr/bin/env python3
# Test with actual file data to isolate the hanging issue

import sys
sys.path.insert(0, '.')

from teleport.dgg import deduce_dynamic

# Load the test file
with open('test_artifacts/pic1.jpg', 'rb') as f:
    data = f.read()

print(f"File size: {len(data)} bytes")
print(f"First 50 bytes: {data[:50]}")

print("Calling deduce_dynamic...")
result = deduce_dynamic(data)
print(f"Result: {result}")
print("Success!")
