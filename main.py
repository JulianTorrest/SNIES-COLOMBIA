import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_title="SNIES Colombia", layout="wide")

# URLs de los archivos
URL_PROGRAMAS = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
URL_INSTITUCIONES = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

@st.cache_data
def cargar_datos():
    df_programas = pd.read_csv(URL_PROGRAMAS)
    df_instituciones = pd.read_csv(URL_INSTITUCIONES)
    return df_programas, df_instituciones

def limpiar_datos(df):
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("No informado")
        else:
            df[col] = df[col].fillna(df[col].median(numeric_only=True))
    return df

def mostrar_kpis(df):
    st.metric("🔢 Total Filas", df.shape[0])
    st.metric("📊 Total Columnas", df.shape[1])
    if "COSTO_MATRÍCULA_ESTUD_NUEVOS" in df.columns:
        st.metric("💰 Costo Promedio", round(df["COSTO_MATRÍCULA_ESTUD_NUEVOS"].mean(), 2))

def graficos(df):
    col_categorica = st.selectbox("Selecciona una columna categórica:", df.select_dtypes(include='object').columns)
    fig, ax = plt.subplots()
    df[col_categorica].value_counts().head(20).plot(kind='barh', ax=ax)
    st.pyplot(fig)

    col_numerica = st.selectbox("Selecciona una columna numérica:", df.select_dtypes(include='number').columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col_numerica], kde=True, ax=ax)
    st.pyplot(fig)

def clustering(df):
    st.markdown("### 🤖 Clustering KMeans")
    df_num = df.select_dtypes(include='number').dropna()
    if df_num.shape[1] < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas.")
        return

    k = st.slider("Selecciona número de clusters (K):", 2, 10, 3)
    model = KMeans(n_clusters=k, n_init="auto")
    clusters = model.fit_predict(df_num)
    df["CLUSTER"] = clusters

    st.write("🔻 Distribución por cluster:")
    st.dataframe(df["CLUSTER"].value_counts())

    # Graficar primeros 2 componentes numéricos
    if df_num.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_num.iloc[:, 0], y=df_num.iloc[:, 1], hue=clusters, palette="tab10", ax=ax)
        st.pyplot(fig)

def eda_completo(nombre_df, df):
    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])

    with tabs[0]:
        st.subheader("Vista previa")
        st.dataframe(df.head())
        st.write("🔍 Tipos de datos:")
        st.write(df.dtypes)
        st.write("🧪 Valores nulos:")
        st.write(df.isnull().sum())

    with tabs[1]:
        st.subheader("Limpieza de Datos")
        st.write("Antes:", df.shape)
        df = limpiar_datos(df)
        st.write("Después:", df.shape)
        st.dataframe(df.head())

    with tabs[2]:
        st.subheader("Gráficos y Distribuciones")
        graficos(df)

    with tabs[3]:
        st.subheader("Indicadores Clave")
        mostrar_kpis(df)

    with tabs[4]:
        st.subheader("Modelado Automatizado")
        clustering(df)

# Main
st.title("📊 SNIES - Analítica de Datos")
st.markdown("Análisis de datos para programas e instituciones de educación superior en Colombia")

opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])

df_programas, df_instituciones = cargar_datos()

if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)

