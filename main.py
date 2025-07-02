import pandas as pd

# URLs de los CSV en GitHub
url_programas = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Programas.csv"
url_instituciones = "https://raw.githubusercontent.com/JulianTorrest/SNIES-COLOMBIA/main/Instituciones.csv"

# Cargar los archivos
df_programas = pd.read_csv(url_programas)
df_instituciones = pd.read_csv(url_instituciones)

# Listar campos de cada archivo
print("📁 Campos en Programas.csv:")
for col in df_programas.columns:
    print(f"- {col}")

print("\n📁 Campos en Instituciones.csv:")
for col in df_instituciones.columns:
    print(f"- {col}")

