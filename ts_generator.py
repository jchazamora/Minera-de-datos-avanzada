import pandas as pd


REQUIRED_COLS = [
    "Year",
    "Month",
    "Day",
    "Sum of Departure_Total Departing Pax Count",
    "Departure_Region"
]


def load_and_prepare_monthly(csv_path: str) -> pd.DataFrame:
    """
    Lee Data.csv con columnas:
      Year, Month, Day, Sum of Departure_Total Departing Pax Count, Departure_Region
    y devuelve un dataframe mensual (freq MS) con suma de pasajeros por región.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Construir fecha. Month viene como texto (e.g., December).
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-" + df["Day"].astype(str),
        errors="coerce"
    )
    df = df.dropna(subset=["Date"]).copy()

    # Normalizar región (opcional)
    df["Departure_Region"] = df["Departure_Region"].astype(str).str.strip()

    pax_col = "Sum of Departure_Total Departing Pax Count"

    # Agregar a mensual (inicio de mes MS)
    monthly = (
        df.groupby([pd.Grouper(key="Date", freq="MS"), "Departure_Region"], as_index=False)[pax_col]
          .sum()
          .rename(columns={pax_col: "Departing_Pax"})
          .sort_values(["Departure_Region", "Date"])
    )

    return monthly


def get_regions(monthly_df: pd.DataFrame) -> list[str]:
    return sorted(monthly_df["Departure_Region"].unique().tolist())


def get_region_series(monthly_df: pd.DataFrame, region: str) -> pd.Series:
    """
    Devuelve la serie mensual (index Date) de una región, asegurando continuidad mensual (rellena faltantes con 0).
    """
    sub = monthly_df[monthly_df["Departure_Region"] == region].copy()
    if sub.empty:
        raise ValueError(f"No hay datos para la región: {region}")

    sub = sub.set_index("Date")["Departing_Pax"].sort_index()

    # Rellenar meses faltantes con 0 (o podría ser interpolación, pero para conteos suele ser 0 si no hay registro)
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="MS")
    sub = sub.reindex(full_idx, fill_value=0)
    sub.index.name = "Date"
    return sub
