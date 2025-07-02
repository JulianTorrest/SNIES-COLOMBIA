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

    num_columnas = st.selectbox("Selecciona cuántas columnas deseas graficar:", [1, 2, 3])

    tipos_graficos_por_columnas = {
        1: ["Barras", "Histograma", "KDE", "Pie", "Countplot"],
        2: ["Dispersión", "Boxplot", "Lineplot", "Heatmap", "Violinplot", "Barplot"],
        3: ["Dispersión 3D", "Pairplot", "Heatmap correlación"]
    }

    tipo_grafico = st.selectbox("Selecciona tipo de gráfico:", tipos_graficos_por_columnas[num_columnas])
    columnas = st.multiselect(f"Selecciona {num_columnas} columna(s):", df.columns, max_selections=num_columnas)

    if len(columnas) != num_columnas:
        st.info(f"Por favor selecciona exactamente {num_columnas} columna(s) para continuar.")
        return

    fig = plt.figure()

    try:
        if tipo_grafico == "Barras":
            df[columnas[0]].value_counts().head(20).plot(kind="barh")
            plt.title(f"Barras de {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Histograma":
            sns.histplot(df[columnas[0]], kde=False)
            plt.title(f"Histograma de {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "KDE":
            sns.kdeplot(df[columnas[0]], fill=True)
            plt.title(f"KDE de {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Pie":
            df[columnas[0]].value_counts().head(10).plot.pie(autopct='%1.1f%%')
            plt.title(f"Gráfico de pastel: {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Countplot":
            sns.countplot(data=df, x=columnas[0])
            plt.title(f"Conteo de {columnas[0]}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif tipo_grafico == "Dispersión":
            sns.scatterplot(data=df, x=columnas[0], y=columnas[1])
            plt.title(f"Dispersión entre {columnas[0]} y {columnas[1]}")
            st.pyplot(fig)

        elif tipo_grafico == "Boxplot":
            sns.boxplot(data=df, x=columnas[0], y=columnas[1])
            plt.title(f"Boxplot: {columnas[1]} por {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Lineplot":
            sns.lineplot(data=df, x=columnas[0], y=columnas[1])
            plt.title(f"Línea: {columnas[1]} vs {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Heatmap":
            pivot = df.pivot_table(values=columnas[1], index=columnas[0], aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("Heatmap")
            st.pyplot(fig)

        elif tipo_grafico == "Violinplot":
            sns.violinplot(data=df, x=columnas[0], y=columnas[1])
            plt.title(f"Violinplot: {columnas[1]} por {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Barplot":
            sns.barplot(data=df, x=columnas[0], y=columnas[1])
            plt.title(f"Barplot: {columnas[1]} por {columnas[0]}")
            st.pyplot(fig)

        elif tipo_grafico == "Dispersión 3D":
            ax = fig.add_subplot(projection='3d')
            ax.scatter(df[columnas[0]], df[columnas[1]], df[columnas[2]])
            ax.set_xlabel(columnas[0])
            ax.set_ylabel(columnas[1])
            ax.set_zlabel(columnas[2])
            plt.title("Dispersión 3D")
            st.pyplot(fig)

        elif tipo_grafico == "Pairplot":
            sns.pairplot(df[columnas])
            st.pyplot()

        elif tipo_grafico == "Heatmap correlación":
            corr = df[columnas].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Mapa de calor de correlación")
            st.pyplot(fig)

        else:
            st.warning("❌ Tipo de gráfico no soportado.")
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

# Cargar y limpiar datos al inicio
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

