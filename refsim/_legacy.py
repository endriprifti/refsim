"""Import the checked code of Sections IV-V (legacy/) without copying it.

The legacy folders hold the code archives exactly as they passed the independent checks
(section4_code_B1.zip, section_V_part1_code_B2.zip, section_V_part2_code_B2.zip, cc2_recompute.py).
Wrappers in refsim import their library modules through `load()`, so the simulator and the book
compute with the same lines of code.
"""
from __future__ import annotations

import importlib
import os
import sys

from .params import ROOT

LEGACY = os.path.join(ROOT, "legacy")
FOLDERS = {"s4": "section4_B1", "v1": "section5_part1_B2", "v2": "section5_part2_B2", "cc2": "consistency_check_2"}


def load(folder: str, module: str):
    path = os.path.join(LEGACY, FOLDERS.get(folder, folder))
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module)
