import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("SNIES - Análisis de Programas e Instituciones")

# URLs RAW
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.xlsx"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones%20(2).xlsx"

@st.cache_data
def cargar_datos():
    # Descargar archivo de Programas
    r1 = requests.get(url_programas)
    programas = pd.read_excel(BytesIO(r1.content), engine="openpyxl")

    # Descargar archivo de Instituciones
    r2 = requests.get(url_instituciones)
    instituciones = pd.read_excel(BytesIO(r2.content), engine="openpyxl")

    return programas, instituciones

# Cargar datos
programas_df, instituciones_df = cargar_datos()

# Mostrar
st.subheader("📘 Datos de Programas")
st.dataframe(programas_df.head())

st.subheader("🏫 Datos de Instituciones")
st.dataframe(instituciones_df.head())
