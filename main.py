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

def obtener_tipos_graficos(n):
    opciones = {
        1: ["Barra", "Pie", "Boxplot", "Violinplot", "Stripplot", "Swarmplot", "Countplot", "Histograma", "Altair Bar", "Plotly Bar", "Plotly Pie", "Treemap", "Sunburst", "FacetGrid", "Catplot"],
        2: ["Barra Agrupada", "Heatmap", "Boxplot", "Violinplot", "Stripplot", "Swarmplot", "Plotly Sunburst", "Plotly Treemap", "FacetGrid", "Altair Bar", "Catplot"],
        3: ["Plotly Sunburst", "Plotly Treemap"]
    }
    return opciones.get(n, [])

def formato_eje_y(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

def graficos(df):
    st.markdown("### Visualización dinámica")

    num_cat = st.selectbox("¿Cuántas columnas cualitativas deseas usar?", [1, 2, 3])
    col_cat = st.multiselect("Selecciona columnas cualitativas:", df.select_dtypes(include='object').columns.tolist(), max_selections=num_cat)
    col_num = st.selectbox("Selecciona columna cuantitativa:", df.select_dtypes(include='number').columns.tolist())
    operacion = st.selectbox("Operación sobre cuantitativa:", ["Contar", "Sumar", "Promedio", "Mediana"])

    if len(col_cat) != num_cat:
        st.info("Selecciona exactamente las columnas cualitativas indicadas.")
        return

    tipos_graficos = obtener_tipos_graficos(num_cat)
    tipo = st.selectbox("Tipo de gráfico compatible:", tipos_graficos)

    if operacion == "Contar":
        df_g = df.groupby(col_cat).size().reset_index(name="Valor")
    elif operacion == "Sumar":
        df_g = df.groupby(col_cat)[col_num].sum().reset_index(name="Valor")
    elif operacion == "Promedio":
        df_g = df.groupby(col_cat)[col_num].mean().reset_index(name="Valor")
    else:
        df_g = df.groupby(col_cat)[col_num].median().reset_index(name="Valor")

    st.markdown(f"#### {tipo}")

    try:
        if tipo == "Barra":
            fig, ax = plt.subplots()
            sns.barplot(data=df_g, x=col_cat[0], y="Valor", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Pie":
            fig, ax = plt.subplots()
            df_g.groupby(col_cat[0])["Valor"].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        elif tipo == "Boxplot":
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col_cat[0], y=col_num, ax=ax)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Violinplot":
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x=col_cat[0], y=col_num, ax=ax)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Stripplot":
            fig, ax = plt.subplots()
            sns.stripplot(data=df, x=col_cat[0], y=col_num, ax=ax, jitter=True)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Swarmplot":
            fig, ax = plt.subplots()
            sns.swarmplot(data=df, x=col_cat[0], y=col_num, ax=ax)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Countplot":
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col_cat[0], ax=ax)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Histograma":
            fig, ax = plt.subplots()
            df[col_num].hist(ax=ax, bins=20)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "Altair Bar":
            chart = alt.Chart(df_g).mark_bar().encode(
                x=alt.X(col_cat[0], sort='-y'),
                y='Valor'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif tipo == "Plotly Bar":
            fig = px.bar(df_g, x=col_cat[0], y="Valor")
            st.plotly_chart(fig, use_container_width=True)

        elif tipo == "Plotly Pie":
            fig = px.pie(df_g, names=col_cat[0], values="Valor")
            st.plotly_chart(fig, use_container_width=True)

        elif tipo == "Heatmap" and len(col_cat) == 2:
            pivot = df_g.pivot(index=col_cat[0], columns=col_cat[1], values="Valor")
            fig, ax = plt.subplots()
            sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".0f", ax=ax)
            st.pyplot(fig)

        elif tipo == "Barra Agrupada" and len(col_cat) == 2:
            fig, ax = plt.subplots()
            sns.barplot(data=df_g, x=col_cat[0], y="Valor", hue=col_cat[1], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            formato_eje_y(ax)
            st.pyplot(fig)

        elif tipo == "FacetGrid" and len(col_cat) >= 1:
            g = sns.FacetGrid(df, col=col_cat[0])
            g.map(sns.histplot, col_num)
            st.pyplot(g.fig)

        elif tipo == "Catplot" and len(col_cat) == 2:
            g = sns.catplot(data=df, x=col_cat[0], y=col_num, hue=col_cat[1], kind="bar")
            st.pyplot(g.fig)

        elif tipo == "Plotly Sunburst" and len(col_cat) >= 2:
            fig = px.sunburst(df_g, path=col_cat, values='Valor')
            st.plotly_chart(fig, use_container_width=True)

        elif tipo == "Plotly Treemap" and len(col_cat) >= 2:
            fig = px.treemap(df_g, path=col_cat, values='Valor')
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Este tipo de gráfico no es compatible con la cantidad de columnas seleccionadas.")

    except Exception as e:
        st.error(f"Error generando gráfico: {e}")

def clustering(df):
    st.markdown("### 🤖 Clustering KMeans")
    df_num = df.select_dtypes(include='number').dropna()
    if df_num.shape[1] < 2 or df_num.shape[0] < 2:
        st.warning("❌ Se necesitan al menos 2 columnas y 2 filas numéricas sin nulos para clustering.")
        return

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)
    k = st.slider("Selecciona número de clusters:", 2, min(10, df_scaled.shape[0]), 3)
    model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    clusters = model.fit_predict(df_scaled)
    df_copy = df_num.copy()
    df_copy['CLUSTER'] = clusters
    st.dataframe(df_copy['CLUSTER'].value_counts())

    fig, ax = plt.subplots()
    sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=clusters, palette='tab10', ax=ax)
    ax.set_title("Visualización Clusters")
    st.pyplot(fig)

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

    with tabs[4]:
        clustering(df)

# App principal
st.title("📊 SNIES - Analítica de Programas e Instituciones")
df_programas, df_instituciones = cargar_datos()

modulo = st.radio("Selecciona módulo:", ["Programas", "Instituciones"])
if modulo == "Programas":
    modulo_eda("Programas", df_programas)
else:
    modulo_eda("Instituciones", df_instituciones)
