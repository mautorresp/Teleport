#!/usr/bin/env python3
# Minimal test to isolate the hanging issue

import sys
sys.path.insert(0, '.')

# Test the exact chain that CLI uses
from teleport.dgg import deduce_dynamic

# Start with very small data
test_data = b'\x00'
print(f"Testing with data: {test_data}")

print("Calling deduce_dynamic...")
result = deduce_dynamic(test_data)
print(f"Result: {result}")
print("Success!")
