from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_mae(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return None
    return float(np.mean(np.abs(y_true.values - y_pred.values)))
