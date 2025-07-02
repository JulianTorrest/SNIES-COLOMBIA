import streamlit as st
import pandas as pd

# Títulos y descripción
st.title("SNIES - Colombia: Análisis de Programas e Instituciones")
st.markdown("""
Este dashboard muestra datos del SNIES usando archivos desde GitHub.
""")

# URLs RAW de los archivos Excel en GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.xlsx"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.xlsx"

# Cargar archivos
@st.cache_data
def cargar_datos():
    programas = pd.read_excel(url_programas)
    instituciones = pd.read_excel(url_instituciones)
    return programas, instituciones

programas_df, instituciones_df = cargar_datos()

# Mostrar datos
st.subheader("📘 Datos de Programas")
st.dataframe(programas_df.head(10))

st.subheader("🏫 Datos de Instituciones")
st.dataframe(instituciones_df.head(10))

# Opciones de filtro simple (puedes expandirlo)
st.subheader("🔍 Filtro por Departamento")
departamentos = programas_df['DEPARTAMENTO'].dropna().unique()
opcion_dep = st.selectbox("Selecciona un Departamento", sorted(departamentos))

# Filtrar datos por departamento
programas_filtrados = programas_df[programas_df['DEPARTAMENTO'] == opcion_dep]

st.write(f"### Programas en el departamento: {opcion_dep}")
st.dataframe(programas_filtrados)

# Mostrar resumen por nivel de formación
st.subheader("📊 Conteo por Nivel de Formación")
conteo_nivel = programas_df['NIVEL'].value_counts()
st.bar_chart(conteo_nivel)

# Mostrar resumen por tipo de institución
st.subheader("🏛️ Instituciones por Naturaleza Jurídica")
conteo_tipo = instituciones_df['NATURALEZA'].value_counts()
st.bar_chart(conteo_tipo)

