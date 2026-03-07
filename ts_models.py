import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ModelResult:
    name: str
    y_pred: pd.Series
    params: Dict[str, Any]


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_len: int = 12) -> pd.Series:
    """
    Pronóstico seasonal naive: repite el valor del mismo mes del año anterior.
    Si no hay suficiente historia, cae a naive simple.
    """
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    if len(train) >= season_len:
        last_season = train.iloc[-season_len:].values
        reps = int(np.ceil(horizon / season_len))
        pred = np.tile(last_season, reps)[:horizon]
    else:
        pred = np.repeat(train.iloc[-1], horizon)

    return pd.Series(pred, index=idx, name="y_pred")


def ets_forecast(train: pd.Series, horizon: int, seasonal: str = "add", season_len: int = 12) -> pd.Series:
    """
    ETS / Holt-Winters con tendencia + estacionalidad.
    """
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    # Si hay poca data, reducir complejidad
    if len(train) < 2 * season_len:
        model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
    else:
        model = ExponentialSmoothing(
            train, trend="add", seasonal=seasonal, seasonal_periods=season_len, initialization_method="estimated"
        )

    fit = model.fit(optimized=True)
    pred = fit.forecast(horizon)
    pred.index = idx
    return pred.rename("y_pred")


def sarimax_grid_forecast(
    train: pd.Series,
    horizon: int,
    season_len: int = 12,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
) -> tuple[pd.Series, dict]:
    """
    Búsqueda pequeña (grid) para SARIMAX. No es autoarima completo, pero funciona bien para un demo reproducible.
    """
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    best_aic = np.inf
    best_fit = None
    best_order = None
    best_seasonal = None

    # Si hay poca data, limitar estacionalidad
    use_season = len(train) >= 3 * season_len

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                if use_season:
                    seasonal_space = [
                        (P, D, Q, season_len)
                        for P in range(max_P + 1)
                        for D in range(max_D + 1)
                        for Q in range(max_Q + 1)
                    ]
                else:
                    seasonal_space = [(0, 0, 0, 0)]

                for seas in seasonal_space:
                    try:
                        model = SARIMAX(
                            train,
                            order=(p, d, q),
                            seasonal_order=seas if seas[-1] != 0 else (0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fit = model.fit(disp=False)
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_fit = fit
                            best_order = (p, d, q)
                            best_seasonal = seas
                    except Exception:
                        continue

    # Fallback: naive si nada ajusta
    if best_fit is None:
        pred = seasonal_naive_forecast(train, horizon, season_len)
        return pred, {"order": None, "seasonal_order": None, "aic": None}

    pred = best_fit.forecast(horizon)
    pred.index = idx
    return pred.rename("y_pred"), {
        "order": best_order,
        "seasonal_order": best_seasonal if best_seasonal and best_seasonal[-1] != 0 else (0, 0, 0, 0),
        "aic": float(best_aic),
    }
