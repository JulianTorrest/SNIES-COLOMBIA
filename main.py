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

def graficos(df):
    st.markdown("### 📊 Visualización Interactiva")
    cols_qual = df.select_dtypes(include='object').columns.tolist()
    cols_quant = df.select_dtypes(include='number').columns.tolist()

    num_qual = st.selectbox("¿Cuántas columnas cualitativas deseas seleccionar?", [1, 2, 3])

    col_qualitativas = st.multiselect(f"Selecciona {num_qual} columna(s) cualitativa(s):", cols_qual, max_selections=num_qual)
    col_cuantitativa = st.selectbox("Selecciona la columna cuantitativa:", cols_quant)
    operacion = st.selectbox("Operación sobre columna cuantitativa:", ["conteo", "suma", "promedio", "mediana"])

    graficos_disponibles = {
        1: ["Barras", "Torta", "Histograma", "Boxplot", "Violin", "Wordcloud"],
        2: ["Barras agrupadas", "Mapa de calor", "Gráfico de dispersión"],
        3: ["Treemap", "Sunburst", "Bubble Chart"]
    }
    tipo_grafico = st.selectbox("Tipo de gráfico:", graficos_disponibles[num_qual])

    if len(col_qualitativas) != num_qual:
        st.warning("Selecciona el número correcto de columnas cualitativas")
        return

    df_viz = df.copy()

    agg_func = {
        "conteo": (col_cuantitativa, "count"),
        "suma": (col_cuantitativa, "sum"),
        "promedio": (col_cuantitativa, "mean"),
        "mediana": (col_cuantitativa, "median")
    }

    if num_qual == 1:
        group = df_viz.groupby(col_qualitativas)[col_cuantitativa].agg(agg_func[operacion][1]).reset_index()
        if tipo_grafico == "Barras":
            fig = px.bar(group, x=col_qualitativas[0], y=col_cuantitativa)
        elif tipo_grafico == "Torta":
            fig = px.pie(group, names=col_qualitativas[0], values=col_cuantitativa)
        elif tipo_grafico == "Histograma":
            fig = px.histogram(df_viz, x=col_cuantitativa, color=col_qualitativas[0])
        elif tipo_grafico == "Boxplot":
            fig = px.box(df_viz, x=col_qualitativas[0], y=col_cuantitativa)
        elif tipo_grafico == "Violin":
            fig = px.violin(df_viz, x=col_qualitativas[0], y=col_cuantitativa, box=True)
        elif tipo_grafico == "Wordcloud":
            from wordcloud import WordCloud
            wc = WordCloud(width=800, height=400).generate(" ".join(df[col_qualitativas[0]].astype(str)))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            return

    elif num_qual == 2:
        group = df_viz.groupby(col_qualitativas)[col_cuantitativa].agg(agg_func[operacion][1]).reset_index()
        if tipo_grafico == "Barras agrupadas":
            fig = px.bar(group, x=col_qualitativas[0], y=col_cuantitativa, color=col_qualitativas[1], barmode="group")
        elif tipo_grafico == "Mapa de calor":
            heatmap_data = group.pivot(index=col_qualitativas[0], columns=col_qualitativas[1], values=col_cuantitativa)
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Blues")
            st.pyplot(fig)
            return
        elif tipo_grafico == "Gráfico de dispersión":
            fig = px.scatter(group, x=col_qualitativas[0], y=col_cuantitativa, color=col_qualitativas[1])

    elif num_qual == 3:
        group = df_viz.groupby(col_qualitativas)[col_cuantitativa].agg(agg_func[operacion][1]).reset_index()
        if tipo_grafico == "Treemap":
            fig = px.treemap(group, path=col_qualitativas, values=col_cuantitativa)
        elif tipo_grafico == "Sunburst":
            fig = px.sunburst(group, path=col_qualitativas, values=col_cuantitativa)
        elif tipo_grafico == "Bubble Chart":
            fig = px.scatter(group, x=col_qualitativas[0], y=col_cuantitativa, size=col_cuantitativa, color=col_qualitativas[1], hover_name=col_qualitativas[2])

    fig.update_layout(yaxis_tickformat=',d')
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
    fig.update_layout(yaxis_tickformat=',d')
    st.plotly_chart(fig, use_container_width=True)

# Se mantienen las funciones de KPI previamente definidas
# Se mantiene la estructura principal de tabs

st.title("📊 SNIES - Analítica de Programas e Instituciones")
df_programas, df_instituciones = cargar_datos()

def modulo_eda(nombre, df):
    st.subheader(f"📂 Módulo: {nombre}")
    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])

    with tabs[0]:
        st.dataframe(df.head())
        st.write("Tipos de datos:", df.dtypes)
        st.write("Nulos:", df.isnull().sum())

    with tabs[1]:
        df = limpiar_datos(df)
        st.success("Datos limpiados")
        st.dataframe(df.head())

    with tabs[2]:
        graficos(df)

    with tabs[3]:
        st.metric("Filas", df.shape[0])
        st.metric("Columnas", df.shape[1])
        for col in df.select_dtypes(include='number').columns:
            st.metric(col, f"{df[col].mean():,.0f}")
        if nombre == "Programas":
            kpi_avanzado_programas(df)
        elif nombre == "Instituciones":
            kpi_avanzado_instituciones(df)

    with tabs[4]:
        clustering(df)

modulo = st.radio("Selecciona módulo:", ["Programas", "Instituciones"])
if modulo == "Programas":
    modulo_eda("Programas", df_programas)
else:
    modulo_eda("Instituciones", df_instituciones)

