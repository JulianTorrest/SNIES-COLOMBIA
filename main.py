import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="SNIES Analítica Completa", layout="wide")

# --- URLs for Data ---
URL_PROGRAMAS = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
URL_INSTITUCIONES = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

# --- Data Loading Function ---
@st.cache_data
def cargar_datos():
    """Loads program and institution data from GitHub URLs."""
    df_programas = pd.read_csv(URL_PROGRAMAS)
    df_instituciones = pd.read_csv(URL_INSTITUCIONES)
    return df_programas, df_instituciones

# --- Data Cleaning Function ---
def limpiar_datos(df):
    """Fills missing values in DataFrame: 'object' columns with 'No informado', others with median."""
    df_copy = df.copy() # Work on a copy to avoid modifying original cached data
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].fillna("No informado")
        else:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median(numeric_only=True))
    return df_copy

# --- KPI Functions ---
def kpi_avanzado_programas(df):
    """Displays advanced KPIs for programs."""
    st.markdown("### 📊 Análisis Avanzado de KPIs para Programas")

    st.subheader("Métricas Clave de Programas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Número Total de Programas", df.shape[0])
    with col2:
        if "NÚMERO_CRÉDITOS" in df.columns:
            avg_creditos = df["NÚMERO_CRÉDITOS"].mean()
            st.metric("Promedio de Créditos por Programa", f"{avg_creditos:.2f}")
        else:
            st.warning("Columna 'NÚMERO_CRÉDITOS' no encontrada.")
    with col3:
        if "COSTO_MATRÍCULA_ESTUD_NUEVOS" in df.columns:
            avg_matricula = df['COSTO_MATRÍCULA_ESTUD_NUEVOS'].mean()
            st.metric("Costo Promedio de Matrícula", f"${avg_matricula:,.0f}")
        else:
            st.warning("Columna 'COSTO_MATRÍCULA_ESTUD_NUEVOS' no encontrada.")

    st.subheader("Distribuciones Clave")

    # Distribution by Nivel Académico
    if "NIVEL_ACADÉMICO" in df.columns:
        st.write("#### Distribución de Programas por Nivel Académico")
        nivel_counts = df["NIVEL_ACADÉMICO"].value_counts().reset_index()
        nivel_counts.columns = ["Nivel Académico", "Cantidad"]
        fig = px.bar(nivel_counts, x="Nivel Académico", y="Cantidad", title="Programas por Nivel Académico")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'NIVEL_ACADÉMICO' no encontrada para KPI.")

    # Distribution by Modalidad
    if "MODALIDAD" in df.columns:
        st.write("#### Distribución de Programas por Modalidad")
        modalidad_counts = df["MODALIDAD"].value_counts().reset_index()
        modalidad_counts.columns = ["Modalidad", "Cantidad"]
        fig = px.pie(modalidad_counts, names="Modalidad", values="Cantidad", title="Programas por Modalidad")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'MODALIDAD' no encontrada para KPI.")

    st.info("💡 **Más KPIs:** Aquí se podrían añadir análisis de 'PERIODICIDAD_ADMISIONES', 'SECTOR', o 'ÁREA_DE_CONOCIMIENTO'.")

def kpi_avanzado_instituciones(df):
    """Displays advanced KPIs for institutions."""
    st.markdown("### 🏛️ Análisis Avanzado de KPIs para Instituciones")

    st.subheader("Métricas Clave de Instituciones")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Número Total de Instituciones", df.shape[0])
    with col2:
        if "PROGRAMAS_VIGENTES" in df.columns:
            avg_programas = df["PROGRAMAS_VIGENTES"].mean()
            st.metric("Promedio de Programas por Institución", f"{avg_programas:.0f}")
        else:
            st.warning("Columna 'PROGRAMAS_VIGENTES' no encontrada.")
    with col3:
        if "ACREDITADA_ALTA_CALIDAD" in df.columns:
            acreditadas_count = df[df["ACREDITADA_ALTA_CALIDAD"] == 'SI'].shape[0]
            st.metric("Instituciones Acreditadas (Alta Calidad)", acreditadas_count)
        else:
            st.warning("Columna 'ACREDITADA_ALTA_CALIDAD' no encontrada.")

    st.subheader("Distribuciones Clave")

    # Distribution by Naturaleza Jurídica
    if "NATURALEZA_JURÍDICA" in df.columns:
        st.write("#### Distribución de Instituciones por Naturaleza Jurídica")
        juridica_counts = df["NATURALEZA_JURÍDICA"].value_counts().reset_index()
        juridica_counts.columns = ["Naturaleza Jurídica", "Cantidad"]
        fig = px.bar(juridica_counts, x="Naturaleza Jurídica", y="Cantidad", title="Instituciones por Naturaleza Jurídica")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'NATURALEZA_JURÍDICA' no encontrada para KPI.")

    # Distribution by Sector
    if "SECTOR" in df.columns:
        st.write("#### Distribución de Instituciones por Sector")
        sector_counts = df["SECTOR"].value_counts().reset_index()
        sector_counts.columns = ["Sector", "Cantidad"]
        fig = px.pie(sector_counts, names="Sector", values="Cantidad", title="Instituciones por Sector")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'SECTOR' no encontrada para KPI.")

    st.info("💡 **Más KPIs:** Se podría analizar la distribución por 'CARÁCTER_ACADÉMICO' o por 'DEPARTAMENTO_DOMICILIO'.")


# --- Machine Learning Functions (Placeholders) ---
def ml_programas(df):
    """Placeholder for Machine Learning models for programs."""
    st.markdown("### 🤖 Modelos de Machine Learning para Programas")
    st.info("""
        Aquí se aplicarían modelos de Machine Learning.
        
        **Ejemplos de Modelos a Implementar:**
        1.  **Clustering (K-Means):** Para agrupar programas similares por `NÚMERO_CRÉDITOS`, `COSTO_MATRÍCULA_ESTUD_NUEVOS`.
        2.  **Regresión (Random Forest Regressor):** Para predecir `COSTO_MATRÍCULA_ESTUD_NUEVOS` basado en características del programa.
        3.  **Clasificación (Decision Tree):** Para clasificar el `NIVEL_ACADÉMICO` de un programa.
        """)
    st.write("---")
    st.subheader("Ejemplo: Clustering de Programas (K-Means)")
    df_num = df[["NÚMERO_CRÉDITOS", "COSTO_MATRÍCULA_ESTUD_NUEVOS"]].dropna() if "NÚMERO_CRÉDITOS" in df.columns and "COSTO_MATRÍCULA_ESTUD_NUEVOS" in df.columns else pd.DataFrame()

    if df_num.empty or df_num.shape[1] < 2:
        st.warning("Se necesitan las columnas 'NÚMERO_CRÉDITOS' y 'COSTO_MATRÍCULA_ESTUD_NUEVOS' y datos suficientes para este ejemplo de clustering.")
        return

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)

    k = st.slider("Número de clusters (Programas):", 2, min(10, df_scaled.shape[0] // 2), 3) # Ensure k is not too large
    try:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        clusters = model.fit_predict(df_scaled)
        df_num["CLUSTER"] = clusters

        fig = px.scatter(df_num, x="NÚMERO_CRÉDITOS", y="COSTO_MATRÍCULA_ESTUD_NUEVOS",
                         color=df_num["CLUSTER"].astype(str),
                         title=f"Clusters de Programas (K={k})",
                         hover_data=df_num.columns)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Conteo de Programas por Cluster:")
        st.dataframe(df_num["CLUSTER"].value_counts())
    except Exception as e:
        st.error(f"Error al ejecutar el clustering: {e}")

def ml_instituciones(df):
    """Placeholder for Machine Learning models for institutions."""
    st.markdown("### 🤖 Modelos de Machine Learning para Instituciones")
    st.info("""
        Aquí se aplicarían modelos de Machine Learning.

        **Ejemplos de Modelos a Implementar:**
        1.  **Clustering (Agglomerative Clustering):** Para agrupar instituciones similares por `PROGRAMAS_VIGENTES`, `SECTOR`.
        2.  **Clasificación (Logistic Regression):** Para predecir si una institución está `ACREDITADA_ALTA_CALIDAD`.
        3.  **NLP (Topic Modeling):** Para extraer temas de la columna `MISIÓN`.
        """)
    st.write("---")
    st.subheader("Ejemplo: Clasificación de Instituciones (Acreditación)")
    if "ACREDITADA_ALTA_CALIDAD" not in df.columns or "PROGRAMAS_VIGENTES" not in df.columns or "SECTOR" not in df.columns:
        st.warning("Se necesitan las columnas 'ACREDITADA_ALTA_CALIDAD', 'PROGRAMAS_VIGENTES' y 'SECTOR' para este ejemplo de clasificación.")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    df_ml = df.copy()
    # Basic Feature Engineering and One-Hot Encoding
    df_ml['SECTOR_encoded'] = df_ml['SECTOR'].astype('category').cat.codes
    df_ml['NATURALEZA_JURÍDICA_encoded'] = df_ml['NATURALEZA_JURÍDICA'].astype('category').cat.codes

    features = ['PROGRAMAS_VIGENTES', 'SECTOR_encoded', 'NATURALEZA_JURÍDICA_encoded']
    target = 'ACREDITADA_ALTA_CALIDAD'

    # Filter out 'No informado' or NaN values in target if they exist
    df_ml = df_ml[df_ml[target].isin(['SI', 'NO'])]

    if df_ml.empty or not all(col in df_ml.columns for col in features):
        st.warning("Datos insuficientes o columnas requeridas no presentes para el ejemplo de clasificación.")
        return

    X = df_ml[features].dropna()
    y = df_ml.loc[X.index, target] # Align y with X's index after dropping NA

    if X.empty or len(X) < 2:
        st.warning("No hay suficientes datos para entrenar el modelo después de la preparación.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Precisión del Modelo de Clasificación (Random Forest): **{accuracy:.2f}**")
    st.write("Características de importancia:")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(feature_importances)


# --- Advanced Visualization Function ---
def visualizacion_avanzada(df):
    """Allows interactive visualization based on user selection."""
    st.markdown("### 📈 Visualización Interactiva")

    cualitativas = df.select_dtypes(include='object').columns.tolist()
    cuantitativas = df.select_dtypes(include='number').columns.tolist()

    if not cualitativas or not cuantitativas:
        st.warning("No hay suficientes columnas cualitativas o cuantitativas para esta visualización.")
        return

    # User selections for visualization
    selected_x = st.selectbox("Selecciona Variable para Eje X (Cualitativa):", cualitativas)
    selected_y = st.selectbox("Selecciona Variable para Eje Y (Cuantitativa):", cuantitativas)
    chart_type = st.selectbox("Selecciona Tipo de Gráfico:", ["Barras", "Boxplot", "Violín", "Histograma", "Dispersión"])

    if chart_type == "Barras":
        st.write(f"#### Conteo o Suma de '{selected_y}' por '{selected_x}'")
        aggregation_option = st.selectbox("Selecciona Operación de Agregación:", ["Conteo", "Suma", "Promedio"])
        if aggregation_option == "Conteo":
            agg_df = df[selected_x].value_counts().reset_index()
            agg_df.columns = [selected_x, 'Cantidad']
            fig = px.bar(agg_df, x=selected_x, y='Cantidad', title=f"Conteo de '{selected_x}'")
        else:
            agg_df = df.groupby(selected_x)[selected_y].agg(aggregation_option.lower()).reset_index()
            agg_df.columns = [selected_x, 'Valor Agregado']
            fig = px.bar(agg_df, x=selected_x, y='Valor Agregado', title=f"{aggregation_option} de '{selected_y}' por '{selected_x}'")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Boxplot":
        st.write(f"#### Boxplot de '{selected_y}' por '{selected_x}'")
        fig = px.box(df, x=selected_x, y=selected_y, title=f"Distribución de '{selected_y}' por '{selected_x}'")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violín":
        st.write(f"#### Violin Plot de '{selected_y}' por '{selected_x}'")
        fig = px.violin(df, x=selected_x, y=selected_y, box=True, title=f"Densidad de '{selected_y}' por '{selected_x}'")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histograma":
        st.write(f"#### Histograma de '{selected_y}' (Coloreado por '{selected_x}')")
        fig = px.histogram(df, x=selected_y, color=selected_x, title=f"Distribución de '{selected_y}' por '{selected_x}'")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Dispersión":
        st.write(f"#### Gráfico de Dispersión entre '{selected_y}' y '{selected_x}'")
        color_option = st.selectbox("Colorear por (Opcional):", ["Ninguno"] + cualitativas)
        if color_option == "Ninguno":
            fig = px.scatter(df, x=selected_x, y=selected_y, title=f"Relación entre '{selected_x}' y '{selected_y}'")
        else:
            fig = px.scatter(df, x=selected_x, y=selected_y, color=color_option, title=f"Relación entre '{selected_x}' y '{selected_y}' (Coloreado por {color_option})")
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Más Visualizaciones:** Se pueden añadir gráficos de torta, treemaps, sunburst, o heatmaps para explorar más a fondo los datos.")


# --- Main EDA Function with Tabs ---
def eda_completo(nombre_df, df_original):
    """
    Performs Exploratory Data Analysis (EDA) in a modular way using Streamlit tabs.
    Args:
        nombre_df (str): Name of the DataFrame (e.g., "Programas", "Instituciones").
        df_original (pd.DataFrame): The original DataFrame to analyze.
    """
    # Create a working copy of the DataFrame to apply cleaning without affecting the cached original
    df_working = df_original.copy()

    tabs = st.tabs(["📄 Datos", "🧼 Limpieza", "📈 Visualización", "📊 KPIs", "🤖 ML"])

    with tabs[0]:
        st.subheader(f"Vista Previa de {nombre_df}")
        st.dataframe(df_working.head())
        st.write(f"📋 **Tipos de Datos en {nombre_df}:**")
        st.dataframe(df_working.dtypes) # Use st.dataframe for better readability
        st.write(f"🔍 **Valores Nulos por Columna en {nombre_df} (Antes de Limpieza):**")
        st.dataframe(df_working.isnull().sum().reset_index(name='Nulos').rename(columns={'index': 'Columna'})) # Better display of nulls

    with tabs[1]:
        st.subheader(f"Limpieza de Datos para {nombre_df}")
        st.info("Aplicando la estrategia de limpieza: Rellenar nulos de tipo 'object' con 'No informado' y nulos numéricos con la mediana.")
        df_cleaned = limpiar_datos(df_working) # Clean the working copy
        st.success(f"Datos de {nombre_df} limpiados correctamente.")
        st.subheader(f"Valores Nulos Después de la Limpieza en {nombre_df}:")
        st.dataframe(df_cleaned.isnull().sum().reset_index(name='Nulos').rename(columns={'index': 'Columna'})) # Display nulls after cleaning
        st.subheader(f"Vista Previa de {nombre_df} Después de la Limpieza:")
        st.dataframe(df_cleaned.head())
        # Update the working DataFrame to the cleaned version for subsequent tabs
        df_working = df_cleaned

    with tabs[2]:
        # Visualizations should use the cleaned data
        visualizacion_avanzada(df_working)

    with tabs[3]:
        # KPIs should use the cleaned data
        if nombre_df == "Programas":
            kpi_avanzado_programas(df_working)
        else:
            kpi_avanzado_instituciones(df_working)

    with tabs[4]:
        # ML models should use the cleaned data
        if nombre_df == "Programas":
            ml_programas(df_working)
        else:
            ml_instituciones(df_working)

# --- Main Streamlit Application Entry Point ---
st.title("📊 SNIES - Analítica de Programas e Instituciones")

# Load data once and cache it
df_programas, df_instituciones = cargar_datos()

# Main navigation radio button
opcion = st.radio("Selecciona el módulo a explorar:", ["Programas", "Instituciones"])

# Route to the appropriate EDA module based on user selection
if opcion == "Programas":
    eda_completo("Programas", df_programas)
else:
    eda_completo("Instituciones", df_instituciones)
