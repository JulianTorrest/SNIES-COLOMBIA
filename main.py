import pandas as pd

# URLs de los CSV en GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

# Cargar archivos
df_programas = pd.read_csv(url_programas)
df_instituciones = pd.read_csv(url_instituciones)

# Función EDA básica
def eda_basico(df, nombre_df, max_categorias=20):
    print(f"\n📊 EDA para: {nombre_df}")
    print("-" * 60)
    print(f"🔢 Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("\n🔍 Tipos de datos:")
    print(df.dtypes)
    print("\n🧪 Valores nulos por columna:")
    print(df.isnull().sum())

    print("\n🧬 Valores únicos por columna:")
    for col in df.columns:
        uniques = df[col].dropna().unique()
        if len(uniques) <= max_categorias:
            print(f"  - {col}: {uniques.tolist()}")
        else:
            print(f"  - {col}: {len(uniques)} valores únicos")

    print("\n📈 Estadísticas numéricas:")
    print(df.describe(include='all').transpose())

# EDA para ambos archivos
eda_basico(df_programas, "Programas.csv")
eda_basico(df_instituciones, "Instituciones.csv")

