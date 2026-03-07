import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ts_models import seasonal_naive_forecast, ets_forecast, sarimax_grid_forecast


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    denom = y_true.replace(0, np.nan).abs()
    return float((np.abs((y_true - y_pred)) / denom).mean() * 100)


def backtest_models(
    ts: pd.Series,
    horizon: int = 6,
    n_splits: int = 6,
    season_len: int = 12,
) -> pd.DataFrame:
    """
    Backtesting rolling-origin.
    Devuelve dataframe con métricas promedio por modelo.
    """
    ts = ts.sort_index()
    min_train = max(24, 2 * season_len)  # mínimo razonable

    if len(ts) < min_train + horizon + n_splits:
        raise ValueError(
            f"Serie muy corta para backtesting. "
            f"Se requieren aprox >= {min_train + horizon + n_splits} meses y hay {len(ts)}."
        )

    rows = []

    # puntos de corte (últimos n_splits cortes)
    cutoffs = []
    end = len(ts) - horizon
    start = end - n_splits
    for i in range(start, end):
        cutoffs.append(i)

    for cut in cutoffs:
        train = ts.iloc[:cut]
        test = ts.iloc[cut:cut + horizon]

        # 1) Seasonal Naive
        pred_sn = seasonal_naive_forecast(train, horizon, season_len).reindex(test.index)
        rows.append(_score_fold("SeasonalNaive", test, pred_sn, extra={}))

        # 2) ETS
        try:
            pred_ets = ets_forecast(train, horizon, seasonal="add", season_len=season_len).reindex(test.index)
            rows.append(_score_fold("ETS(Holt-Winters)", test, pred_ets, extra={}))
        except Exception as e:
            rows.append(_score_fold("ETS(Holt-Winters)", test, None, extra={"error": str(e)}))

        # 3) SARIMAX grid
        try:
            pred_sar, info = sarimax_grid_forecast(train, horizon, season_len=season_len)
            pred_sar = pred_sar.reindex(test.index)
            rows.append(_score_fold("SARIMAX(grid)", test, pred_sar, extra=info))
        except Exception as e:
            rows.append(_score_fold("SARIMAX(grid)", test, None, extra={"error": str(e)}))

    df = pd.DataFrame(rows)

    # promedio por modelo (ignorando folds fallidos)
    ok = df[df["ok"] == True].copy()
    summary = (
        ok.groupby("model", as_index=False)
          .agg(
              RMSE=("rmse", "mean"),
              MAE=("mae", "mean"),
              MAPE=("mape", "mean"),
              folds=("model", "count"),
          )
          .sort_values(["RMSE", "MAE"], ascending=[True, True])
    )
    return summary


def _score_fold(model_name: str, y_true: pd.Series, y_pred: pd.Series | None, extra: dict) -> dict:
    if y_pred is None or y_pred.isna().all():
        return {
            "model": model_name,
            "ok": False,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "extra": extra,
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        mp = mape(y_true, y_pred)
    except Exception:
        mp = np.nan

    return {
        "model": model_name,
        "ok": True,
        "rmse": rmse,
        "mae": mae,
        "mape": mp,
        "extra": extra,
    }
