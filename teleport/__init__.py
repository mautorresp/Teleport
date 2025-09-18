"""
Teleport - Integer-Only Mathematical Causality Detection Library

A strict integer-only library for mathematical causality detection, encoding, and data processing
that enforces no-float constraints through guards and linting.
"""

__version__ = "0.1.0"
__author__ = "Teleport Project"

# Import core modules
from .guards import no_float_guard
from .clf_int import *
from .costs import *
from .leb_io import *

__all__ = [
    "no_float_guard",
    # clf_int exports will be added
    # costs exports will be added 
    # leb_io exports will be added
]
