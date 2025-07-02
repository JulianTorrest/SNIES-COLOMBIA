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

# --- Advanced KPI Functions (Placeholder for now, can be expanded) ---
def kpi_avanzado_programas(df):
    st.markdown("### 📊 Análisis avanzado de KPIs para Programas (En desarrollo)")
    st.info("Aquí se mostrarían KPIs específicos para programas. Por ejemplo, distribución de créditos, costos de matrícula, etc.")
    # You would add your KPI logic here, similar to the original kpi_avanzado_programas
    # For instance, a simple metric:
    st.metric("Número Total de Programas", df.shape[0])
    if "NÚMERO DE CRÉDITOS" in df.columns: # Added a check to prevent KeyError if column doesn't exist
        st.metric("Promedio de Créditos por Programa", f"{df['NÚMERO DE CRÉDITOS'].mean():.2f}")


def kpi_avanzado_instituciones(df):
    st.markdown("### 🏛️ Análisis avanzado de KPIs para Instituciones (En desarrollo)")
    st.info("Aquí se mostrarían KPIs específicos para instituciones. Por ejemplo, distribución por sector, naturaleza jurídica, etc.")
    # You would add your KPI logic here, similar to the original kpi_avanzado_instituciones
    st.metric("Número Total de Instituciones", df.shape[0])
    if "SECTOR" in df.columns: # Added a check
        st.write("Conteo de Instituciones por Sector:")
        st.dataframe(df["SECTOR"].value_counts())

# --- Clustering Function (Placeholder) ---
def clustering(df):
    st.markdown("### 🤖 Clustering KMeans (En desarrollo)")
    st.info("Aquí se implementaría el algoritmo de clustering K-Means. Requiere al menos dos columnas numéricas.")
    st.warning("Asegúrate de que haya suficientes columnas numéricas sin nulos para un análisis de clustering significativo.")
    df_num = df.select_dtypes(include='number').dropna()
    if df_num.shape[1] < 2:
        st.error("Se necesitan al menos 2 columnas numéricas para realizar el clustering.")
        return
    # Placeholder for the actual clustering logic
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df_num)
    # k = st.slider("Selecciona número de clusters:", 2, min(10, df_scaled.shape[0]), 3)
    # model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    # clusters = model.fit_predict(df_scaled)
    # df["CLUSTER"] = clusters
    # st.dataframe(df["CLUSTER"].value_counts())
    # fig = px.scatter(x=df_scaled[:, 0], y=df_scaled[:, 1], color=clusters.astype(str), labels={'x': df_num.columns[0], 'y': df_num.columns[1]})
    # st.plotly_chart(fig, use_container_width=True)

# --- Advanced Visualization Function (Placeholder) ---
def visualizacion_avanzada(df):
    st.markdown("### 📈 Visualización Interactiva (En desarrollo)")
    st.info("Aquí se podrían crear gráficos interactivos basados en la selección del usuario (barras, tortas, dispersión, etc.).")
    st.warning("Selecciona al menos una variable cualitativa y una cuantitativa para crear visualizaciones.")
    # Placeholder for the actual visualization logic
    # cualitativas = df.select_dtypes(include='object').columns.tolist()
    # cuantitativas = df.select_dtypes(include='number').columns.tolist()
    # ... (rest of your visualizacion_avanzada code) ...


# --- Main EDA Function with Tabs ---
def eda_completo(nombre_df, df):
    # Use a copy of the DataFrame for cleaning within this function
    # This ensures the original cached DataFrame is not modified directly
    df_working = df.copy()

    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])

    with tabs[0]:
        st.subheader(f"Vista Previa de {nombre_df}")
        st.dataframe(df_working.head())
        st.write(f"📋 **Tipos de Datos en {nombre_df}:**")
        st.write(df_working.dtypes)
        st.write(f"🔍 **Valores Nulos por Columna en {nombre_df} (Antes de Limpieza):**")
        st.write(df_working.isnull().sum())

    with tabs[1]:
        st.subheader(f"Limpieza de Datos para {nombre_df}")
        df_cleaned = limpiar_datos(df_working)
        st.success(f"Datos de {nombre_df} limpiados correctamente.")
        st.subheader(f"Valores Nulos Después de la Limpieza en {nombre_df}:")
        st.write(df_cleaned.isnull().sum())
        st.subheader(f"Vista Previa de {nombre_df} Después de la Limpieza:")
        st.dataframe(df_cleaned.head())
        # Pass the cleaned DataFrame to subsequent tabs
        df_working = df_cleaned

    with tabs[2]:
        visualizacion_avanzada(df_working)

    with tabs[3]:
        if nombre_df == "Programas":
            kpi_avanzado_programas(df_working)
        else:
            kpi_avanzado_instituciones(df_working)

    with tabs[4]:
        clustering(df_working)

# --- Main Streamlit App Layout ---
st.title("📊 SNIES - Analítica de Programas e Instituciones")

df_programas, df_instituciones = cargar_datos()

opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])

if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)
