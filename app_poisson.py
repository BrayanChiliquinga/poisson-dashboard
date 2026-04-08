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
    ["Poisson", "Binomial", "Normal",
     "Exponencial", "Weibull", "Gamma", "Lognormal", "Bernoulli"]
)
# NUEVAS DISTRIBUCIONES

modo = st.sidebar.radio(
    "Modo de datos:",
    ["Generar aleatoriamente", "Ingresar datos manualmente"]
)

n = st.sidebar.slider("Tamaño de muestra", 1, 200, 50)

# Parámetros dinámicos
if modo == "Generar aleatoriamente":

    r = np.random.uniform(0, 1, n)

    if distribucion == "Poisson":
        data = np.random.poisson(lambda_val, n)

    elif distribucion == "Binomial":
        data = np.random.binomial(n_bin, p_bin, n)

    elif distribucion == "Normal":
        data = np.random.normal(mu, sigma, n)
    
    elif distribucion == "Exponencial":
        lambda_exp = st.sidebar.number_input("λ", min_value=0.0001, value=1.0)
    
    elif distribucion == "Weibull":
        alpha = st.sidebar.number_input("α (forma)", value=1.5)
        beta = st.sidebar.number_input("β (escala)", value=1.0)
        gamma_w = st.sidebar.number_input("γ (localización)", value=0.0)
    
    elif distribucion == "Gamma":
        k = st.sidebar.number_input("k (forma)", min_value=1, value=2)
        lambda_g = st.sidebar.number_input("λ", min_value=0.0001, value=1.0)
    
    elif distribucion == "Lognormal":
        mu_ln = st.sidebar.number_input("μ", value=0.0)
        sigma_ln = st.sidebar.number_input("σ", min_value=0.0001, value=1.0)
    
    elif distribucion == "Bernoulli":
        p_ber = st.sidebar.slider("p", 0.0, 1.0, 0.5)

    # ----------------------------------
    # NUEVAS DISTRIBUCIONES
    # ----------------------------------

    elif distribucion == "Exponencial":
        data = -np.log(1 - r) / lambda_exp

    elif distribucion == "Weibull":
        data = gamma_w + beta * (-np.log(1 - r))**(1/alpha)

    elif distribucion == "Gamma":
        data = np.zeros(n)
        for i in range(n):
            prod = 1
            for _ in range(int(k)):
                prod *= np.random.uniform(0,1)
            data[i] = -np.log(prod) / lambda_g

    elif distribucion == "Lognormal":
        z = np.random.normal(0, 1, n)
        data = np.exp(mu_ln + sigma_ln * z)

    elif distribucion == "Bernoulli":
        data = np.where(r <= p_ber, 1, 0)

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visualización",
    "📈 Estadísticas",
    "📉 Comparación",
    "📘 Información",
    "📋 Datos"
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

    elif distribucion == "Exponencial":
        y_vals = lambda_exp * np.exp(-lambda_exp * x_vals)
    
    elif distribucion == "Weibull":
        y_vals = (alpha/beta) * ((x_vals-gamma_w)/beta)**(alpha-1) * np.exp(-((x_vals-gamma_w)/beta)**alpha)
    
    elif distribucion == "Gamma":
        from scipy.stats import gamma
        y_vals = gamma.pdf(x_vals, a=k, scale=1/lambda_g)
    
    elif distribucion == "Lognormal":
        from scipy.stats import lognorm
        y_vals = lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))
    
    elif distribucion == "Bernoulli":
        x_vals = [0,1]
        y_vals = [1-p_ber, p_ber]

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

    elif distribucion == "Exponencial":
        media_t = 1/lambda_exp
        var_t = 1/(lambda_exp**2)
    elif distribucion == "Weibull":
        media_t = beta
        var_t = beta**2
    elif distribucion == "Gamma":
        media_t = k/lambda_g
        var_t = k/(lambda_g**2)
    elif distribucion == "Lognormal":
        media_t = np.exp(mu_ln + sigma_ln**2/2)
        var_t = (np.exp(sigma_ln**2)-1)*np.exp(2*mu_ln + sigma_ln**2)
    elif distribucion == "Bernoulli":
        media_t = p_ber
        var_t = p_ber*(1-p_ber)

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
        st.warning("No hay suficientes datos válidos para determinar Chi-Cuadrado (freq esperada ≥ 5)")
    
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

# ----------------------------------
# DATOS TABLA
# ----------------------------------
with tab5:
    st.subheader("📋 Datos utilizados en la simulación")

    df_data = pd.DataFrame({
        "Índice": range(1, len(data)+1),
        "Valor": data
    })

    st.dataframe(df_data, use_container_width=True)

    # Estadística rápida
    st.markdown("### Resumen rápido")
    st.write(f"Cantidad de datos: {len(data)}")
    st.write(f"Valor mínimo: {np.min(data):.4f}")
    st.write(f"Valor máximo: {np.max(data):.4f}")

csv = df_data.to_csv(index=False).encode('utf-8')

st.download_button(
    "📥 Descargar datos",
    csv,
    "datos_simulacion.csv",
    "text/csv"
)
