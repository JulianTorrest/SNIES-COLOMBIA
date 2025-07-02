import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SNIES Colombia", layout="wide")

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
    st.metric("🔢 Total de registros", df.shape[0])
    st.metric("📊 Total de columnas", df.shape[1])
    if "COSTO_MATRÍCULA_ESTUD_NUEVOS" in df.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Promedio", round(df["COSTO_MATRÍCULA_ESTUD_NUEVOS"].mean(), 2))
        col2.metric("⬇️ Mínimo", round(df["COSTO_MATRÍCULA_ESTUD_NUEVOS"].min(), 2))
        col3.metric("⬆️ Máximo", round(df["COSTO_MATRÍCULA_ESTUD_NUEVOS"].max(), 2))

def graficos(df):
    st.markdown("### 📊 Visualización de datos")

    columnas_cat = st.multiselect("Selecciona hasta 3 columnas cualitativas:", 
                                  options=df.select_dtypes(include="object").columns.tolist(),
                                  max_selections=3)
    
    columna_num = st.selectbox("Selecciona una columna cuantitativa:", 
                               options=df.select_dtypes(include="number").columns.tolist())

    operacion = st.selectbox("Selecciona operación sobre la cuantitativa:", ["Contar", "Sumar", "Promedio", "Mediana"])
    tipo_grafico = st.selectbox("Selecciona tipo de gráfico:", ["Barras", "Heatmap", "Barplot"])

    if len(columnas_cat) == 0 or not columna_num:
        st.info("Debes seleccionar al menos 1 columna cualitativa y una cuantitativa.")
        return

    # Agrupación
    if operacion == "Contar":
        df_graf = df.groupby(columnas_cat).size().reset_index(name="Valor")
    elif operacion == "Sumar":
        df_graf = df.groupby(columnas_cat)[columna_num].sum().reset_index(name="Valor")
    elif operacion == "Promedio":
        df_graf = df.groupby(columnas_cat)[columna_num].mean().reset_index(name="Valor")
    elif operacion == "Mediana":
        df_graf = df.groupby(columnas_cat)[columna_num].median().reset_index(name="Valor")

    fig = plt.figure()

    try:
        if tipo_grafico == "Barras":
            if len(columnas_cat) == 1:
                sns.barplot(data=df_graf, x=columnas_cat[0], y="Valor")
                plt.xticks(rotation=45)
                plt.title(f"{operacion} de {columna_num} por {columnas_cat[0]}")
                st.pyplot(fig)
            elif len(columnas_cat) == 2:
                pivot = df_graf.pivot(index=columnas_cat[0], columns=columnas_cat[1], values="Valor").fillna(0)
                pivot.plot(kind="bar", stacked=True)
                plt.title(f"{operacion} de {columna_num} por {columnas_cat[0]} y {columnas_cat[1]}")
                st.pyplot(fig)
            else:
                st.warning("Las Barras solo soportan hasta 2 columnas cualitativas.")
        
        elif tipo_grafico == "Heatmap":
            if len(columnas_cat) == 2:
                pivot = df_graf.pivot(index=columnas_cat[0], columns=columnas_cat[1], values="Valor")
                sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
                plt.title(f"Heatmap: {operacion} de {columna_num}")
                st.pyplot(fig)
            else:
                st.warning("El heatmap requiere exactamente 2 columnas cualitativas.")
        
        elif tipo_grafico == "Barplot":
            if len(columnas_cat) == 1:
                sns.barplot(data=df_graf, x=columnas_cat[0], y="Valor")
                plt.xticks(rotation=45)
                plt.title(f"{operacion} de {columna_num} por {columnas_cat[0]}")
                st.pyplot(fig)
            elif len(columnas_cat) == 2:
                sns.barplot(data=df_graf, x=columnas_cat[0], y="Valor", hue=columnas_cat[1])
                plt.xticks(rotation=45)
                plt.title(f"{operacion} de {columna_num} por {columnas_cat[0]} y {columnas_cat[1]}")
                st.pyplot(fig)
            elif len(columnas_cat) == 3:
                st.warning("Barplot solo soporta hasta 2 columnas cualitativas. Usa agrupaciones externas para más dimensiones.")

    except Exception as e:
        st.error(f"❌ Error al generar el gráfico: {e}")

def clustering(df):
    st.markdown("### 🤖 Clustering KMeans (sin nulos y normalizado)")
    df_num = df.select_dtypes(include='number').copy()
    df_num = df_num.dropna()

    if df_num.shape[1] < 2 or df_num.shape[0] < 2:
        st.warning("❌ Se necesitan al menos 2 columnas y 2 filas numéricas SIN nulos para clustering.")
        return

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)

    k = st.slider("Selecciona número de clusters (K):", 2, min(10, df_scaled.shape[0]), 3)

    try:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        clusters = model.fit_predict(df_scaled)
    except ValueError as e:
        st.error(f"❌ Error en KMeans: {e}")
        return

    df_copy = df_num.copy()
    df_copy["CLUSTER"] = clusters

    st.subheader("📊 Distribución de registros por cluster:")
    st.dataframe(df_copy["CLUSTER"].value_counts())

    if df_scaled.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df_scaled[:, 0],
            y=df_scaled[:, 1],
            hue=clusters,
            palette="tab10",
            ax=ax
        )
        ax.set_xlabel(df_num.columns[0])
        ax.set_ylabel(df_num.columns[1])
        ax.set_title("Visualización de Clusters")
        st.pyplot(fig)

def eda_completo(nombre_df, df):
    st.subheader(f"Análisis de {nombre_df}")
    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])

    with tabs[0]:
        st.subheader("Vista previa")
        st.dataframe(df.head())
        st.write("📋 Tipos de datos:")
        st.write(df.dtypes)
        st.write("🔍 Valores nulos por columna:")
        st.write(df.isnull().sum())
        st.write("📊 Resumen estadístico:")
        st.dataframe(df.describe())

    with tabs[1]:
        st.subheader("Limpieza de datos")
        st.write(f"🔢 Registros antes de limpieza: {df.shape}")
        df = limpiar_datos(df)
        st.write(f"✅ Registros después de limpieza: {df.shape}")
        st.dataframe(df.head())

    with tabs[2]:
        st.subheader("Visualizaciones")
        graficos(df)

    with tabs[3]:
        st.subheader("Indicadores Clave")
        mostrar_kpis(df)

    with tabs[4]:
        st.subheader("Modelado con KMeans")
        clustering(df)

# Cargar y limpiar datos
df_programas, df_instituciones = cargar_datos()
df_programas = limpiar_datos(df_programas)
df_instituciones = limpiar_datos(df_instituciones)

# Interfaz principal
st.title("📊 SNIES - Analítica de Datos")
st.markdown("Análisis exploratorio de Programas e Instituciones de Educación Superior en Colombia.")

opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])

if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)

