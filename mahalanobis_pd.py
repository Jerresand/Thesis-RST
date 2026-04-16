from __future__ import annotations

# Convenience wrapper so you can do: `from mahalanobis_pd import ...`
# without dealing with `01_pd_analysis` as a Python module name.

from pathlib import Path
import importlib.util

_ROOT = Path(__file__).resolve().parent
_IMPL_PATH = _ROOT / "01_pd_analysis" / "mahalanobis_pd.py"

_spec = importlib.util.spec_from_file_location("_mahalanobis_pd_impl", _IMPL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load Mahalanobis implementation from {_IMPL_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MahalanobisModel = _mod.MahalanobisModel
fit_mahalanobis_model = _mod.fit_mahalanobis_model
mahalanobis_d2 = _mod.mahalanobis_d2
mahalanobis_d2_from_df_row = _mod.mahalanobis_d2_from_df_row
mahalanobis_d2_batch = _mod.mahalanobis_d2_batch

