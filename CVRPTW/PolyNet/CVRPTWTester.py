"""CVRPTW PolyNet tester compatibility wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_LINC_DIR = Path(__file__).resolve().parents[1] / "LINC"
if str(_LINC_DIR) not in sys.path:
    sys.path.insert(0, str(_LINC_DIR))

_SPEC = importlib.util.spec_from_file_location("_linc_cvrptw_tester", _LINC_DIR / "CVRPTWTester.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load LINC CVRPTWTester from {_LINC_DIR}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CVRPTWTester = _MODULE.CVRPTWTester

__all__ = ["CVRPTWTester"]
