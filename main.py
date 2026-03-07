import streamlit as st
import pandas as pd

from streamlit_option_menu import option_menu

from ts_generator import load_and_prepare_monthly, get_regions, get_region_series
from ts_evaluator import backtest_models
from visualizer import (
    plot_monthly_series,
    seasonal_index,
    plot_seasonal_profile,
    stl_strength,
    seasonal_difference_test
)

st.set_page_config(
    page_title="Predicción Mensual de Pasajeros Salientes - SJO",
    layout="wide"
)

st.title("Predicción Mensual de Pasajeros Salientes por Región (SJO)")
st.caption("Modelos automatizados de series de tiempo + análisis de estacionalidad por región")


@st.cache_data
def load_monthly_cached(path: str) -> pd.DataFrame:
    return load_and_prepare_monthly(path)


def page_data(monthly: pd.DataFrame):
    st.header("1) Datos")
    st.write("Vista mensual (suma de pasajeros salientes por región):")
    st.dataframe(monthly, use_container_width=True)

    regions = get_regions(monthly)
    region = st.selectbox("Región para visualizar", regions)
    ts = get_region_series(monthly, region)

    st.plotly_chart(plot_monthly_series(ts, f"Serie mensual - {region}"), use_container_width=True)

    st.info(
        "Nosotros aqui preparamos los datos para las dos preguntas:\n"
        "- Serie mensual por región (input de modelos)\n"
        "- Patrón mensual (input de estacionalidad)"
    )


def page_models(monthly: pd.DataFrame):
    st.header("2) Modelos y desempeño predictivo (por región)")

    regions = get_regions(monthly)
    region = st.selectbox("Región a evaluar", regions)

    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.number_input("Horizonte (meses a predecir)", min_value=1, max_value=24, value=6, step=1)
    with col2:
        n_splits = st.number_input("Backtesting splits", min_value=3, max_value=24, value=6, step=1)
    with col3:
        season_len = st.number_input("Estacionalidad (meses)", min_value=6, max_value=24, value=12, step=1)

    ts = get_region_series(monthly, region)

    if st.button("Evaluar modelos (backtesting)"):
        with st.spinner("Evaluando modelos..."):
            try:
                summary = backtest_models(ts, horizon=int(horizon), n_splits=int(n_splits), season_len=int(season_len))
                st.subheader(f"Ranking de modelos - {region}")
                st.dataframe(summary, use_container_width=True)

                if not summary.empty:
                    best = summary.iloc[0]
                    st.success(
                        f"Mejor modelo para **{region}** según RMSE promedio: "
                        f"**{best['model']}** (RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}, MAPE={best['MAPE']:.2f}%)"
                    )

                st.markdown(
                    " **Nuestro planteamiento de la priemr pregunta // Corregir esto:**\n"
                    "El modelo con menor RMSE/MAE/MAPE en backtesting se considera el mejor para esa región."
                )
            except Exception as e:
                st.error(str(e))


def page_seasonality(monthly: pd.DataFrame):
    st.header("3) Estacionalidad: comparación entre regiones")

    regions = get_regions(monthly)

    selected = st.multiselect(
        "Seleccione regiones para comparar",
        options=regions,
        default=regions[:min(6, len(regions))]
    )

    if not selected:
        st.warning("Seleccione al menos una región.")
        return

    colA, colB = st.columns([2, 1])
    seasonal_by_region = {}
    strength_rows = []

    with colA:
        st.subheader("Perfiles estacionales (índice mensual)")
        for r in selected:
            ts = get_region_series(monthly, r)
            sidx = seasonal_index(ts)
            seasonal_by_region[r] = sidx.set_index("Month")["SeasonalIndex"]
            fig = plot_seasonal_profile(sidx, f"Índice estacional mensual - {r}")
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Fuerza estacional (STL)")
        for r in selected:
            ts = get_region_series(monthly, r)
            strength = stl_strength(ts, period=12)
            strength_rows.append({"Region": r, "SeasonalStrength_STL": strength})
        st.dataframe(pd.DataFrame(strength_rows).sort_values("SeasonalStrength_STL", ascending=False),
                     use_container_width=True)

        st.subheader("Prueba global de diferencias")
        test = seasonal_difference_test(seasonal_by_region)
        if test["ok"]:
            st.write("Kruskal-Wallis sobre índices estacionales (12 meses por región):")
            st.json(test)
            p = test["p_value"]
            if p < 0.05:
                st.success("p < 0.05: hay evidencia de diferencias en estacionalidad entre regiones (según esta prueba).")
            else:
                st.info("p >= 0.05: no hay evidencia suficiente de diferencias (según esta prueba).")
        else:
            st.warning(test["message"])

        st.markdown(
            "**Nuestro planteamiento de 2da pregunta // Corregir esto :**\n"
            "- Se comparan perfiles mensuales (índices estacionales).\n"
            "- Se reporta fuerza estacional (STL).\n"
            "- Se aplica una prueba global (Kruskal-Wallis) para evidenciar diferencias."
        )


def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menú",
            options=["Datos", "Modelos", "Estacionalidad"],
            icons=["table", "graph-up", "calendar3"],
            default_index=0
        )

        st.divider()
        csv_path = st.text_input("Ruta del CSV", value="Data.csv")

    try:
        monthly = load_monthly_cached(csv_path)
    except Exception as e:
        st.error(f"No se pudo cargar el CSV: {e}")
        st.stop()

    if selected == "Datos":
        page_data(monthly)
    elif selected == "Modelos":
        page_models(monthly)
    elif selected == "Estacionalidad":
        page_seasonality(monthly)


if __name__ == "__main__":
    main()
