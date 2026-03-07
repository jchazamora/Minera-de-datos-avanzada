import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from statsmodels.tsa.seasonal import STL
from scipy.stats import kruskal


def plot_monthly_series(ts: pd.Series, title: str):
    df = ts.reset_index()
    df.columns = ["Date", "Departing_Pax"]
    fig = px.line(df, x="Date", y="Departing_Pax", title=title)
    return fig


def seasonal_index(ts: pd.Series) -> pd.DataFrame:
    """
    Índice estacional mensual: promedio de cada mes / promedio global.
    Devuelve DF con Month(1-12), SeasonalIndex.
    """
    df = ts.to_frame("y")
    df["month"] = df.index.month
    overall = df["y"].mean() if df["y"].mean() != 0 else 1.0
    idx = df.groupby("month")["y"].mean() / overall
    out = idx.reset_index().rename(columns={"month": "Month", "y": "SeasonalIndex"})
    out["MonthName"] = out["Month"].apply(lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    return out


def plot_seasonal_profile(season_df: pd.DataFrame, title: str):
    fig = px.line(season_df, x="MonthName", y="SeasonalIndex", markers=True, title=title)
    return fig


def stl_strength(ts: pd.Series, period: int = 12) -> float:
    """
    Fuerza estacional simple basada en STL:
    strength = 1 - Var(resid) / Var(seasonal + resid)
    """
    if len(ts) < 3 * period:
        return float("nan")

    stl = STL(ts, period=period, robust=True).fit()
    resid = stl.resid
    seas = stl.seasonal
    denom = np.var(seas + resid)
    if denom == 0:
        return float("nan")
    strength = 1 - (np.var(resid) / denom)
    return float(strength)


def seasonal_difference_test(seasonal_indices_by_region: dict[str, pd.Series]):
    """
    Prueba Kruskal-Wallis sobre los 12 índices estacionales (uno por mes) de cada región.
    Devuelve estadístico y p-value.
    """
    groups = []
    labels = []
    for region, s in seasonal_indices_by_region.items():
        vals = pd.Series(s).dropna().values
        if len(vals) >= 10:
            groups.append(vals)
            labels.append(region)

    if len(groups) < 2:
        return {"ok": False, "message": "No hay suficientes regiones con datos para la prueba."}

    stat, p = kruskal(*groups)
    return {"ok": True, "statistic": float(stat), "p_value": float(p), "regions_used": labels}
