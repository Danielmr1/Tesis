import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
YEAR = 2022  # <--- ¡CAMBIA ESTO POR EL AÑO QUE QUIERAS! (2020, 2021, etc.)

# Ruta dinámica del archivo
archivo = fr'D:\SIG\csv\{YEAR}\Dataset_{YEAR}_BALANCEADO_FINAL.csv'

# ==========================================
# 2. CARGAR Y ENTRENAR
# ==========================================
print(f"--- Procesando Año: {YEAR} ---")

if not os.path.exists(archivo):
    print(f"❌ ERROR: No se encuentra el archivo: {archivo}")
    exit()

df = pd.read_csv(archivo)

# Eliminamos las columnas que no son variables predictoras
# Nota: Pandas es inteligente, si 'year' es una columna, la borra.
X = df.drop(columns=['class', 'year', 'fecha_txt', 'lat', 'lon'])
y = df['class']

# División Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar Modelo
print("Entrenando Random Forest...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# ==========================================
# 3. CALCULAR MÉTRICAS
# ==========================================
y_prob = modelo.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*40)
print(f"⭐ AUC-ROC Score ({YEAR}): {auc:.4f}")
print("="*40)

# ==========================================
# 4. IMPORTANCIA DE VARIABLES
# ==========================================
importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\nRANKING DE VARIABLES {YEAR} (De más a menos importante):")
print("-" * 50)
print(importancia.to_string(index=False))
print("-" * 50)

# ==========================================
# 5. GRÁFICOS
# ==========================================
print("\nGenerando gráfico de barras... (Se abrirá en una ventana)")

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=importancia, hue='Variable', legend=False, palette='viridis')

# Título dinámico con el año
plt.title(f'¿Qué variables causan más incendios en Leoncio Prado? ({YEAR})')
plt.xlabel('Nivel de Importancia (0-1)')
plt.tight_layout()
plt.show()