import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson, binom, norm, chisquare

# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------
st.set_page_config(page_title="Simulador de Distribuciones", layout="wide")

st.title("Simulador Interactivo de Distribuciones")
st.markdown("Explora distribuciones **Poisson, Binomial y Normal** con análisis estadístico y visual.")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.header("⚙️ Configuración")

distribucion = st.sidebar.selectbox(
    "Distribución:",
    ["Poisson", "Binomial", "Normal"]
)

modo = st.sidebar.radio(
    "Modo de datos:",
    ["Generar aleatoriamente", "Ingresar datos manualmente"]
)

n = st.sidebar.slider("Tamaño de muestra", 1, 200, 50)

# Parámetros dinámicos
if distribucion == "Poisson":
    lambda_val = st.sidebar.number_input("λ (lambda)", min_value=0.0001, value=4.0)

elif distribucion == "Binomial":
    n_bin = st.sidebar.number_input("n (ensayos)", min_value=1, value=10)
    p_bin = st.sidebar.slider("p (probabilidad)", 0.0, 1.0, 0.5)

elif distribucion == "Normal":
    mu = st.sidebar.number_input("μ (media)", value=0.0)
    sigma = st.sidebar.number_input("σ (desviación)", min_value=0.0001, value=1.0)

# Botón regenerar
if "seed" not in st.session_state:
    st.session_state.seed = np.random.randint(0, 10000)

if st.sidebar.button("🔄 Regenerar datos"):
    st.session_state.seed = np.random.randint(0, 10000)

np.random.seed(st.session_state.seed)

# ----------------------------------
# DATOS
# ----------------------------------
data = None

if modo == "Generar aleatoriamente":
    if distribucion == "Poisson":
        data = np.random.poisson(lambda_val, n)
    elif distribucion == "Binomial":
        data = np.random.binomial(n_bin, p_bin, n)
    elif distribucion == "Normal":
        data = np.random.normal(mu, sigma, n)

else:
    user_input = st.sidebar.text_area("Ingrese datos separados por coma o espacio")

    try:
        data = np.array([float(x) for x in user_input.replace(",", " ").split()])
    except:
        st.error("❌ Datos inválidos")
        st.stop()

if data is None or len(data) == 0:
    st.warning("Ingrese o genere datos")
    st.stop()

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualización",
    "📈 Estadísticas",
    "📉 Comparación",
    "📘 Información"
])

# ----------------------------------
# VISUALIZACIÓN
# ----------------------------------
with tab1:
    st.subheader("Histograma y distribución teórica")

    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
    x=data,
    histnorm="probability",
    name="Datos",
    opacity=0.6,
     marker=dict(
        line=dict(
            color='black',   # color del borde
            width=1.5        # grosor del borde
        )
    )
    ))

    x_vals = np.linspace(min(data), max(data), 100)

    if distribucion == "Poisson":
        x_vals = np.arange(min(data), max(data)+1)
        y_vals = poisson.pmf(x_vals, lambda_val)

    elif distribucion == "Binomial":
        x_vals = np.arange(min(data), max(data)+1)
        y_vals = binom.pmf(x_vals, n_bin, p_bin)

    elif distribucion == "Normal":
        y_vals = norm.pdf(x_vals, mu, sigma)

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name="Teórica"
    ))

    fig.update_layout(title="Comparación distribución")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# ESTADÍSTICAS
# ----------------------------------
with tab3:
    media_m = np.mean(data)
    var_m = np.var(data, ddof=1)
    std_m = np.std(data, ddof=1)

    if distribucion == "Poisson":
        media_t = lambda_val
        var_t = lambda_val
    elif distribucion == "Binomial":
        media_t = n_bin * p_bin
        var_t = n_bin * p_bin * (1 - p_bin)
    elif distribucion == "Normal":
        media_t = mu
        var_t = sigma**2

    col1, col2, col3 = st.columns(3)

    col1.metric("Media", f"{media_m:.4f}", f"Teórica: {media_t:.4f}")
    col2.metric("Varianza", f"{var_m:.4f}", f"Teórica: {var_t:.4f}")
    col3.metric("Desv. Estándar", f"{std_m:.4f}")

    st.subheader("Error relativo")
    st.write(f"Media: {abs(media_m-media_t)/abs(media_t):.2%}")
    st.write(f"Varianza: {abs(var_m-var_t)/abs(var_t):.2%}")

# ----------------------------------
# COMPARACIÓN
# ----------------------------------
with tab2:
    st.subheader("Frecuencias")

    valores, freq_obs = np.unique(data.astype(int), return_counts=True)

    if distribucion == "Poisson":
        prob = poisson.pmf(valores, lambda_val)
    elif distribucion == "Binomial":
        prob = binom.pmf(valores, n_bin, p_bin)
    else:
        prob = None

    if prob is not None:
        freq_teo = prob * len(data)

        df = pd.DataFrame({
            "Valor": valores,
            "Observado": freq_obs,
            "Teórico": np.round(freq_teo, 2)
        })

        st.dataframe(df)

# Filtrar frecuencias válidas
mask = freq_teo >= 5

freq_obs_valid = freq_obs[mask]
freq_teo_valid = freq_teo[mask]

if len(freq_obs_valid) > 1:
    try:
        chi2, p = chisquare(freq_obs_valid, freq_teo_valid)
        st.write(f"Chi²: {chi2:.4f}")
        st.write(f"p-valor: {p:.4f}")
        
        if p > 0.05:
            st.success("Buen ajuste")
        else:
            st.error("Mal ajuste")
            
        
    except:
        st.warning("No se pudo calcular Chi-cuadrado")
    
    
    else:
        st.warning("No hay suficientes datos válidos (freq esperada ≥ 5)")

# ----------------------------------
# INFORMACIÓN
# ----------------------------------
with tab4:
    st.subheader("Sobre las distribuciones")

    if distribucion == "Poisson":
        st.write("Modela eventos en un intervalo de tiempo.")
    elif distribucion == "Binomial":
        st.write("Número de éxitos en ensayos independientes.")
    elif distribucion == "Normal":
        st.write("Distribución continua simétrica.")
