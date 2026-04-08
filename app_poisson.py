import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson, binom, norm, chisquare, gamma, lognorm

# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------
st.set_page_config(page_title="Simulador de Distribuciones", layout="wide")

st.title("📊 Simulador Interactivo de Distribuciones")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.header("⚙️ Configuración")

distribucion = st.sidebar.selectbox(
    "Distribución:",
    ["Poisson", "Binomial", "Normal",
     "Exponencial", "Weibull", "Gamma", "Lognormal", "Bernoulli"]
)

modo = st.sidebar.radio(
    "Modo de datos:",
    ["Generar aleatoriamente", "Ingresar datos manualmente"]
)

n = st.sidebar.slider("Tamaño de muestra", 1, 200, 50)

# Parámetros
if distribucion == "Poisson":
    lambda_val = st.sidebar.number_input("λ", min_value=0.0001, value=4.0)

elif distribucion == "Binomial":
    n_bin = st.sidebar.number_input("n", min_value=1, value=10)
    p_bin = st.sidebar.slider("p", 0.0, 1.0, 0.5)

elif distribucion == "Normal":
    mu = st.sidebar.number_input("μ", value=0.0)
    sigma = st.sidebar.number_input("σ", min_value=0.0001, value=1.0)

elif distribucion == "Exponencial":
    lambda_exp = st.sidebar.number_input("λ", min_value=0.0001, value=1.0)

elif distribucion == "Weibull":
    alpha = st.sidebar.number_input("α", value=1.5)
    beta = st.sidebar.number_input("β", value=1.0)
    gamma_w = st.sidebar.number_input("γ", value=0.0)

elif distribucion == "Gamma":
    k = st.sidebar.number_input("k", min_value=1, value=2)
    lambda_g = st.sidebar.number_input("λ", min_value=0.0001, value=1.0)

elif distribucion == "Lognormal":
    mu_ln = st.sidebar.number_input("μ", value=0.0)
    sigma_ln = st.sidebar.number_input("σ", min_value=0.0001, value=1.0)

elif distribucion == "Bernoulli":
    p_ber = st.sidebar.slider("p", 0.0, 1.0, 0.5)

# Seed
if "seed" not in st.session_state:
    st.session_state.seed = np.random.randint(0, 10000)

if st.sidebar.button("🔄 Regenerar"):
    st.session_state.seed = np.random.randint(0, 10000)

np.random.seed(st.session_state.seed)

# ----------------------------------
# GENERACIÓN DE DATOS
# ----------------------------------
data = None

if modo == "Generar aleatoriamente":

    if distribucion == "Poisson":
        # MÉTODO DE LA IMAGEN (producto de uniformes)
        data = []
        for _ in range(n):
            T = 1
            N = 0
            while True:
                r = np.random.uniform(0,1)
                T *= r
                if T < np.exp(-lambda_val):
                    break
                N += 1
            data.append(N)
        data = np.array(data)

    elif distribucion == "Binomial":
        data = np.random.binomial(n_bin, p_bin, n)

    elif distribucion == "Normal":
        data = np.random.normal(mu, sigma, n)

    elif distribucion == "Exponencial":
        r = np.random.uniform(0,1,n)
        data = -np.log(1-r)/lambda_exp

    elif distribucion == "Weibull":
        r = np.random.uniform(0,1,n)
        data = gamma_w + beta*(-np.log(1-r))**(1/alpha)

    elif distribucion == "Gamma":
        data = []
        for _ in range(n):
            prod = np.prod(np.random.uniform(0,1,int(k)))
            data.append(-np.log(prod)/lambda_g)
        data = np.array(data)

    elif distribucion == "Lognormal":
        data = np.exp(np.random.normal(mu_ln, sigma_ln, n))

    elif distribucion == "Bernoulli":
        r = np.random.uniform(0,1,n)
        data = np.where(r <= p_ber, 1, 0)

else:
    user_input = st.sidebar.text_area("Ingrese datos")
    try:
        data = np.array([float(x) for x in user_input.replace(",", " ").split()])
    except:
        st.error("Datos inválidos")
        st.stop()

if data is None or len(data) == 0:
    st.stop()

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualización",
    "📈 Estadísticas",
    "📉 Comparación",
    "📋 Datos"
])

# ----------------------------------
# VISUALIZACIÓN
# ----------------------------------
with tab1:

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        histnorm="probability",
        marker=dict(line=dict(color='black', width=1))
    ))

    x_vals = np.linspace(min(data), max(data), 100)

    if distribucion == "Poisson":
        x_vals = np.arange(min(data), max(data)+1)
        y_vals = poisson.pmf(x_vals, lambda_val)

    elif distribucion == "Binomial":
        y_vals = binom.pmf(x_vals, n_bin, p_bin)

    elif distribucion == "Normal":
        y_vals = norm.pdf(x_vals, mu, sigma)

    elif distribucion == "Exponencial":
        y_vals = lambda_exp*np.exp(-lambda_exp*x_vals)

    elif distribucion == "Weibull":
        y_vals = (alpha/beta)*((x_vals-gamma_w)/beta)**(alpha-1)*np.exp(-((x_vals-gamma_w)/beta)**alpha)

    elif distribucion == "Gamma":
        y_vals = gamma.pdf(x_vals, a=k, scale=1/lambda_g)

    elif distribucion == "Lognormal":
        y_vals = lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))

    elif distribucion == "Bernoulli":
        x_vals = [0,1]
        y_vals = [1-p_ber, p_ber]

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines"))

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# ESTADÍSTICAS
# ----------------------------------
with tab2:

    media = np.mean(data)
    var = np.var(data)

    st.write("Media:", media)
    st.write("Varianza:", var)

# ----------------------------------
# COMPARACIÓN (CHI)
# ----------------------------------
with tab3:
    if distribucion in ["Poisson", "Binomial"]:
        valores, freq_obs = np.unique(data.astype(int), return_counts=True)
        
        if distribucion == "Poisson":
            prob = poisson.pmf(valores, lambda_val)
        else:
            prob = binom.pmf(valores, n_bin, p_bin)
            
            freq_teo = prob * len(data)
            mask = freq_teo >= 5
            
            if sum(mask) > 1:
                chi2, p = chisquare(freq_obs[mask], freq_teo[mask])
                st.write("Chi²:", chi2)
                st.write("p-valor:", p)

# ----------------------------------
# DATOS
# ----------------------------------
with tab4:
    st.dataframe(pd.DataFrame(data, columns=["Datos"]))
