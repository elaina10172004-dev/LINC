"""CVRPTW PolyNet model compatibility wrapper.

The converted CVRPTW PolyNet checkpoint was produced with the current shared
CVRPTW architecture knobs enabled/disabled through checkpoint metadata.  Reuse
that implementation here so strict checkpoint loading keeps catching real
architecture drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_LINC_DIR = Path(__file__).resolve().parents[1] / "LINC"
if str(_LINC_DIR) not in sys.path:
    sys.path.insert(0, str(_LINC_DIR))

_SPEC = importlib.util.spec_from_file_location("_linc_cvrptw_model", _LINC_DIR / "CVRPTWModel.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load LINC CVRPTWModel from {_LINC_DIR}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CVRPTWModel = _MODULE.CVRPTWModel
CANDIDATE_FEATURE_INDEX = _MODULE.CANDIDATE_FEATURE_INDEX
_get_encoding = _MODULE._get_encoding

__all__ = ["CVRPTWModel", "CANDIDATE_FEATURE_INDEX", "_get_encoding"]
