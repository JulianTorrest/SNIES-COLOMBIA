import streamlit as st
import pandas as pd

st.title("SNIES - Análisis Exploratorio de Programas e Instituciones")

# URLs en GitHub (ambos CSV)
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

# ✅ Carga con caché para que no se bloquee la app
@st.cache_data
def cargar_datos():
    df_programas = pd.read_csv(url_programas)
    df_instituciones = pd.read_csv(url_instituciones)
    return df_programas, df_instituciones

programas_df, instituciones_df = cargar_datos()

# Selección de dataset
opcion = st.selectbox("Selecciona el dataset para analizar:", ["Programas", "Instituciones"])

df = programas_df if opcion == "Programas" else instituciones_df

st.subheader(f"📄 Vista previa de {opcion}")
st.dataframe(df.head())

st.subheader("📏 Dimensiones")
st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

st.subheader("🔍 Tipos de datos")
st.write(df.dtypes)

st.subheader("🧪 Valores nulos")
st.write(df.isnull().sum())

st.subheader("📈 Estadísticas básicas")
# Mostrar solo columnas categóricas si son pocas
categoricas = df.select_dtypes(include='object')
if not categoricas.empty:
    st.write("🎯 Columnas categóricas (valores únicos ≤ 10):")
    for col in categoricas.columns:
        if df[col].nunique() <= 10:
            st.write(f"- {col}: {df[col].unique().tolist()}")

# Estadísticas numéricas
st.write("📊 Estadísticas numéricas:")
st.write(df.describe())


