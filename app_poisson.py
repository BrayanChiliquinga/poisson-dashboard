import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson, chisquare

# ---------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------
st.set_page_config(page_title="Simulación Poisson", layout="wide")

st.title("📊 Dashboard Interactivo - Distribución de Poisson")

# ---------------------------
# SIDEBAR - PARÁMETROS
# ---------------------------
st.sidebar.header("⚙️ Parámetros")

# Lambda
lambda_val = st.sidebar.number_input(
    "Ingrese λ (lambda):",
    min_value=0.0001,
    value=4.0,
    step=0.1
)

# Modo de datos
modo = st.sidebar.radio(
    "Modo de datos:",
    ("Generar aleatoriamente", "Ingresar datos manualmente")
)

# Tamaño muestra
n = st.sidebar.slider("Tamaño de muestra:", 1, 200, 50)

# Botón regenerar
if "seed" not in st.session_state:
    st.session_state.seed = np.random.randint(0, 10000)

if st.sidebar.button("🔄 Regenerar datos"):
    st.session_state.seed = np.random.randint(0, 10000)

np.random.seed(st.session_state.seed)

# ---------------------------
# INGRESO DE DATOS
# ---------------------------
data = None

if modo == "Generar aleatoriamente":
    data = np.random.poisson(lambda_val, n)

else:
    user_input = st.sidebar.text_area(
        "Ingrese datos separados por comas o espacios:"
    )

    try:
        if user_input.strip() == "":
            st.warning("Ingrese datos válidos.")
        else:
            data = np.array(
                [int(x) for x in user_input.replace(",", " ").split()]
            )
    except:
        st.error("❌ Error: Datos inválidos. Use solo números enteros.")

# Validación
if data is None or len(data) == 0:
    st.stop()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Parámetros",
    "📊 Visualizaciones",
    "📈 Estadísticas",
    "📉 Comparación Teórica"
])

# ---------------------------
# TAB 1 - PARÁMETROS
# ---------------------------
with tab1:
    st.subheader("Configuración actual")
    st.write(f"λ = {lambda_val}")
    st.write(f"Tamaño muestra = {len(data)}")
    st.write(f"Modo = {modo}")

# ---------------------------
# TAB 2 - VISUALIZACIONES
# ---------------------------
with tab2:
    st.subheader("📊 Histograma vs Distribución Teórica")

    # Histograma
    hist_vals, bins = np.histogram(data, bins=range(int(min(data)), int(max(data))+2))

    x_vals = np.arange(0, max(data)+5)
    pmf_vals = poisson.pmf(x_vals, lambda_val)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bins[:-1],
        y=hist_vals,
        name="Frecuencia Observada"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=pmf_vals * len(data),
        mode='lines+markers',
        name="PMF Teórica"
    ))

    fig.update_layout(
        title="Histograma con PMF de Poisson",
        xaxis_title="Valores",
        yaxis_title="Frecuencia"
    )

    st.plotly_chart(fig, use_container_width=True)

    # PMF pura
    st.subheader("📈 Función de Masa de Probabilidad")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=pmf_vals,
        mode='lines+markers',
        name="PMF"
    ))

    fig2.update_layout(
        title="Distribución Teórica de Poisson",
        xaxis_title="Valores",
        yaxis_title="Probabilidad"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# TAB 3 - ESTADÍSTICAS
# ---------------------------
with tab3:
    st.subheader("📈 Estadísticos Muestrales")

    media_m = np.mean(data)
    var_m = np.var(data, ddof=1)
    std_m = np.std(data, ddof=1)

    col1, col2, col3 = st.columns(3)

    col1.metric("Media muestral", f"{media_m:.4f}", f"Teórica: {lambda_val}")
    col2.metric("Varianza muestral", f"{var_m:.4f}", f"Teórica: {lambda_val}")
    col3.metric("Desv. estándar", f"{std_m:.4f}")

    # Errores relativos
    st.subheader("📊 Error relativo")

    error_media = abs(media_m - lambda_val) / lambda_val
    error_var = abs(var_m - lambda_val) / lambda_val

    st.write(f"Error relativo media: {error_media:.4%}")
    st.write(f"Error relativo varianza: {error_var:.4%}")

# ---------------------------
# TAB 4 - COMPARACIÓN TEÓRICA
# ---------------------------
with tab4:
    st.subheader("📉 Frecuencias Observadas vs Teóricas")

    valores, freq_obs = np.unique(data, return_counts=True)

    prob_teo = poisson.pmf(valores, lambda_val)
    freq_teo = prob_teo * len(data)

    df = pd.DataFrame({
        "Valor": valores,
        "Frecuencia Observada": freq_obs,
        "Frecuencia Teórica": np.round(freq_teo, 2)
    })

    st.dataframe(df)

    # Chi-cuadrado
    try:
        chi2, p_val = chisquare(freq_obs, freq_teo)

        st.subheader("📊 Prueba Chi-cuadrado")
        st.write(f"Chi² = {chi2:.4f}")
        st.write(f"p-valor = {p_val:.4f}")

        if p_val > 0.05:
            st.success("✅ Buen ajuste a la distribución de Poisson")
        else:
            st.error("❌ No hay buen ajuste")

    except:
        st.warning("⚠️ No se pudo calcular Chi-cuadrado (ajuste de datos requerido)")
