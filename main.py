import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="SNIES Analítica Completa", layout="wide")

URL_PROGRAMAS = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
URL_INSTITUCIONES = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

@st.cache_data
def cargar_datos():
    df_programas = pd.read_csv(URL_PROGRAMAS)
    df_instituciones = pd.read_csv(URL_INSTITUCIONES)
    return df_programas, df_instituciones

def limpiar_datos(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("No informado")
        else:
            df[col] = df[col].fillna(df[col].median(numeric_only=True))
    return df

st.title("📊 SNIES - Exploración Inicial de Datos")

df_programas, df_instituciones = cargar_datos()

st.header("1. Datos de Programas Académicos")

st.subheader("Vista Previa (Primeras 5 filas)")
st.dataframe(df_programas.head())

st.subheader("Tipos de Datos por Columna")
st.write(df_programas.dtypes)

st.subheader("Valores Nulos Antes de la Limpieza")
st.write(df_programas.isnull().sum())

# Clean the DataFrame and show results
st.subheader("Aplicando Limpieza de Datos a Programas")
df_programas_cleaned = limpiar_datos(df_programas.copy()) # Use a copy to not modify the cached original
st.success("Datos de programas limpiados correctamente.")
st.subheader("Valores Nulos Después de la Limpieza (Programas)")
st.write(df_programas_cleaned.isnull().sum())
st.subheader("Vista Previa Después de la Limpieza (Programas)")
st.dataframe(df_programas_cleaned.head())


st.header("2. Datos de Instituciones Educativas")

st.subheader("Vista Previa (Primeras 5 filas)")
st.dataframe(df_instituciones.head())

st.subheader("Tipos de Datos por Columna")
st.write(df_instituciones.dtypes)

st.subheader("Valores Nulos Antes de la Limpieza")
st.write(df_instituciones.isnull().sum())

# Clean the DataFrame and show results
st.subheader("Aplicando Limpieza de Datos a Instituciones")
df_instituciones_cleaned = limpiar_datos(df_instituciones.copy()) # Use a copy
st.success("Datos de instituciones limpiados correctamente.")
st.subheader("Valores Nulos Después de la Limpieza (Instituciones)")
st.write(df_instituciones_cleaned.isnull().sum())
st.subheader("Vista Previa Después de la Limpieza (Instituciones)")
st.dataframe(df_instituciones_cleaned.head())

st.title("📊 SNIES - Analítica de Programas e Instituciones")
df_programas, df_instituciones = cargar_datos()
opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])
if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)

