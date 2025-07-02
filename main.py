import streamlit as st
import pandas as pd
import requests

st.title("SNIES - Análisis de Programas e Instituciones")

# URLs RAW desde GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/refs/heads/main/Instituciones.csv"

@st.cache_data
def cargar_datos():
    # Leer Programas.xlsx desde GitHub (todavía es Excel)
    r1 = requests.get(url_programas)
    programas = pd.read_excel(r1.content, engine="openpyxl")

    # Leer Instituciones.csv directamente
    instituciones = pd.read_csv(url_instituciones)

    return programas, instituciones

# Cargar datos
programas_df, instituciones_df = cargar_datos()

# Mostrar
st.subheader("📘 Datos de Programas")
st.dataframe(programas_df.head())

st.subheader("🏫 Datos de Instituciones")
st.dataframe(instituciones_df.head())

