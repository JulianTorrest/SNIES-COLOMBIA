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

def kpi_avanzado_programas(df):
    st.markdown("### 📊 Análisis avanzado de KPIs para Programas")
    opciones = [
        "NÚMERO DE CRÉDITOS",
        "COSTO_MATRÍCULA_ESTUD_NUEVOS",
        "SECTOR",
        "CARÁCTER ACADÉMICO",
        "ESTADO DEL PROGRAMA",
        "NIVEL ACADÉMICO",
        "NIVEL DE FORMACIÓN",
        "MODALIDAD",
        "PERIODICIDAD",
        "DEPARTAMENTO OFERTA PROGRAMA"
    ]
    seleccion = st.selectbox("Selecciona variable de análisis:", opciones)
    if seleccion == "NÚMERO DE CRÉDITOS":
        top = df[["NOMBRE_PROGRAMA", "NÚMERO DE CRÉDITOS"]].sort_values(by="NÚMERO DE CRÉDITOS", ascending=False).head(10)
        low = df[["NOMBRE_PROGRAMA", "NÚMERO DE CRÉDITOS"]].sort_values(by="NÚMERO DE CRÉDITOS", ascending=True).head(10)
        st.write("Top 10 programas con más créditos:")
        st.dataframe(top)
        st.write("Top 10 programas con menos créditos:")
        st.dataframe(low)
        fig, ax = plt.subplots()
        sns.histplot(df["NÚMERO DE CRÉDITOS"].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    elif seleccion == "COSTO_MATRÍCULA_ESTUD_NUEVOS":
        st.metric("Promedio matrícula", f"${df['COSTO_MATRÍCULA_ESTUD_NUEVOS'].mean():,.0f}")
        fig, ax = plt.subplots()
        sns.histplot(df["COSTO_MATRÍCULA_ESTUD_NUEVOS"].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        conteo = df[seleccion].value_counts().reset_index()
        conteo.columns = [seleccion, "Cantidad"]
        fig = px.bar(conteo, x=seleccion, y="Cantidad")
        st.plotly_chart(fig, use_container_width=True)

def kpi_avanzado_instituciones(df):
    st.markdown("### 🏛️ Análisis avanzado de KPIs para Instituciones")
    opciones = [
        "NOMBRE INSTITUCIÓN",
        "SECTOR",
        "CARÁCTER ACADÉMICO",
        "NATURALEZA JURÍDICA",
        "DEPARTAMENTO DOMICILIO",
        "MUNICIPIO DOMICILIO",
        "ACREDITACIÓN",
        "VIGILADA POR"
    ]
    seleccion = st.selectbox("Selecciona variable de análisis (Instituciones):", opciones)
    conteo = df[seleccion].value_counts().reset_index()
    conteo.columns = [seleccion, "Cantidad"]
    fig = px.bar(conteo, x=seleccion, y="Cantidad")
    st.plotly_chart(fig, use_container_width=True)

def clustering(df):
    st.markdown("### 🤖 Clustering KMeans")
    df_num = df.select_dtypes(include='number').dropna()
    if df_num.shape[1] < 2:
        st.warning("❌ Se requieren al menos 2 columnas numéricas sin nulos")
        return
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)
    k = st.slider("Selecciona número de clusters:", 2, min(10, df_scaled.shape[0]), 3)
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    clusters = model.fit_predict(df_scaled)
    df["CLUSTER"] = clusters
    st.dataframe(df["CLUSTER"].value_counts())
    fig = px.scatter(x=df_scaled[:, 0], y=df_scaled[:, 1], color=clusters.astype(str), labels={'x': df_num.columns[0], 'y': df_num.columns[1]})
    st.plotly_chart(fig, use_container_width=True)

def visualizacion_avanzada(df):
    st.markdown("### 📈 Visualización Interactiva")
    cualitativas = df.select_dtypes(include='object').columns.tolist()
    cuantitativas = df.select_dtypes(include='number').columns.tolist()
    num_cat = st.selectbox("¿Cuántas variables cualitativas quieres usar?", [1, 2, 3])
    cat_vars = st.multiselect(f"Selecciona {num_cat} columna(s) cualitativa(s):", cualitativas, max_selections=num_cat)
    y_var = st.selectbox("Selecciona variable cuantitativa:", cuantitativas)
    operacion = st.selectbox("Operación a realizar sobre la cuantitativa:", ["count", "sum", "mean", "median"])
    if len(cat_vars) != num_cat:
        st.warning(f"Selecciona exactamente {num_cat} variable(s) cualitativa(s)")
        return
    if num_cat == 1:
        opciones = ["Barras", "Torta", "Boxplot", "Violin", "Histograma", "Treemap"]
    elif num_cat == 2:
        opciones = ["Barras agrupadas", "Heatmap", "Boxplot", "Violin", "Treemap"]
    else:
        opciones = ["Sunburst", "Treemap"]
    tipo = st.selectbox("Selecciona tipo de gráfico:", opciones)
    agg_df = df.groupby(cat_vars)[y_var].agg(operacion).reset_index(name='valor')
    if tipo == "Barras":
        fig = px.bar(agg_df, x=cat_vars[0], y='valor')
    elif tipo == "Torta":
        fig = px.pie(agg_df, names=cat_vars[0], values='valor')
    elif tipo == "Boxplot":
        fig = px.box(df, x=cat_vars[0], y=y_var)
    elif tipo == "Violin":
        fig = px.violin(df, x=cat_vars[0], y=y_var, box=True)
    elif tipo == "Histograma":
        fig = px.histogram(df, x=y_var, color=cat_vars[0])
    elif tipo == "Barras agrupadas" and num_cat == 2:
        fig = px.bar(agg_df, x=cat_vars[0], y='valor', color=cat_vars[1], barmode="group")
    elif tipo == "Heatmap" and num_cat == 2:
        pivot = agg_df.pivot(index=cat_vars[0], columns=cat_vars[1], values='valor')
        fig = px.imshow(pivot, text_auto=True, aspect="auto")
    elif tipo == "Treemap":
        fig = px.treemap(agg_df, path=cat_vars, values='valor')
    elif tipo == "Sunburst":
        fig = px.sunburst(agg_df, path=cat_vars, values='valor')
    else:
        st.error("Tipo de gráfico no implementado")
        return
    fig.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)

def eda_completo(nombre_df, df):
    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])
    with tabs[0]:
        st.subheader("Vista previa")
        st.dataframe(df.head())
        st.write("📋 Tipos de datos:")
        st.write(df.dtypes)
        st.write("🔍 Valores nulos por columna:")
        st.write(df.isnull().sum())
    with tabs[1]:
        st.subheader("Limpieza de datos")
        df = limpiar_datos(df)
        st.success("Datos limpiados correctamente")
        st.dataframe(df.head())
    with tabs[2]:
        visualizacion_avanzada(df)
    with tabs[3]:
        if nombre_df == "Programas":
            kpi_avanzado_programas(df)
        else:
            kpi_avanzado_instituciones(df)
    with tabs[4]:
        clustering(df)

st.title("📊 SNIES - Analítica de Programas e Instituciones")
df_programas, df_instituciones = cargar_datos()
opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])
if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)

