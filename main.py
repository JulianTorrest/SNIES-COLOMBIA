import streamlit as st
import pandas as pd

st.title("SNIES - Análisis de Programas e Instituciones")

# ✅ URLs de archivos CSV en GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

@st.cache_data
def cargar_datos():
    programas = pd.read_csv(url_programas)
    instituciones = pd.read_csv(url_instituciones)
    return programas, instituciones

# Cargar datos
programas_df, instituciones_df = cargar_datos()

# Mostrar datos
st.subheader("📘 Datos de Programas")
st.dataframe(programas_df.head())

st.subheader("🏫 Datos de Instituciones")
st.dataframe(instituciones_df.head())

