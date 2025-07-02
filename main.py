import pandas as pd

# URLs de los CSV en GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

# Cargar archivos
df_programas = pd.read_csv(url_programas)
df_instituciones = pd.read_csv(url_instituciones)

def analizar_campos(df, nombre_df, max_opciones=20):
    print(f"\n📁 Archivo: {nombre_df}")
    print(f"🧮 Total de columnas: {len(df.columns)}\n")

    for col in df.columns:
        print(f"🔹 Campo: {col}")
        valores_unicos = df[col].dropna().unique()
        total_valores = len(valores_unicos)
        print(f"   - Tipo: {df[col].dtype}")
        print(f"   - Valores únicos: {total_valores}")

        if total_valores <= max_opciones:
            print(f"   - Opciones de respuesta: {valores_unicos.tolist()}")
        else:
            print("   - Demasiadas opciones para mostrar.")
        print()

# Analizar ambos archivos
analizar_campos(df_programas, "Programas.csv")
analizar_campos(df_instituciones, "Instituciones.csv")

