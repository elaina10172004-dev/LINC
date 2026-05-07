"""CVRPTW PolyNet environment compatibility wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_LINC_DIR = Path(__file__).resolve().parents[1] / "LINC"
if str(_LINC_DIR) not in sys.path:
    sys.path.insert(0, str(_LINC_DIR))

_SPEC = importlib.util.spec_from_file_location("_linc_cvrptw_env", _LINC_DIR / "CVRPTWEnv.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load LINC CVRPTWEnv from {_LINC_DIR}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Reset_State = _MODULE.Reset_State
Step_State = _MODULE.Step_State


class CVRPTWEnv(_MODULE.CVRPTWEnv):
    def __init__(self, **env_params):
        env_params.setdefault("enable_candidate_features", False)
        super().__init__(**env_params)


__all__ = ["CVRPTWEnv", "Reset_State", "Step_State"]
