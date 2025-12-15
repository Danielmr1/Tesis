import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier # <--- LA ESTRELLA DEL SHOW

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
YEAR = 2020  # <--- CAMBIA EL AÑO AQUÍ

# Ruta dinámica
archivo = fr'D:\SIG\csv\{YEAR}\Dataset_{YEAR}_BALANCEADO_FINAL.csv'

# ==========================================
# 2. CARGAR DATOS
# ==========================================
print(f"🔥 Procesando con XGBoost - Año: {YEAR}")

if not os.path.exists(archivo):
    print(f"❌ ERROR: No se encuentra: {archivo}")
    exit()

df = pd.read_csv(archivo)

# Definir X (Variables) e y (Objetivo)
# Eliminamos columnas que no son predictoras
X = df.drop(columns=['class', 'year', 'fecha_txt', 'lat', 'lon'])
y = df['class']

# División Train/Test (70% entrenar, 30% validar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 3. ENTRENAMIENTO XGBOOST
# ==========================================
print("🚀 Entrenando modelo XGBoost...")

# Configuración del modelo (Hiperparámetros Clave)
modelo_xgb = XGBClassifier(
    n_estimators=100,      # Número de árboles (igual que RF)
    learning_rate=0.1,     # Velocidad de aprendizaje (Más bajo = más preciso pero lento)
    max_depth=5,           # Profundidad máxima del árbol (Controla el sobreajuste)
    subsample=0.8,         # Usa el 80% de datos para cada árbol (evita overfitting)
    colsample_bytree=0.8,  # Usa el 80% de columnas para cada árbol
    eval_metric='logloss', # Métrica de error
    random_state=42,
    n_jobs=-1              # Usar todos los núcleos del CPU
)

modelo_xgb.fit(X_train, y_train)

# ==========================================
# 4. EVALUACIÓN
# ==========================================
# Probabilidad de clase 1 (Incendio)
y_prob = modelo_xgb.predict_proba(X_test)[:, 1]
y_pred = modelo_xgb.predict(X_test)

# Calcular AUC
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*40)
print(f"🏆 XGBoost AUC-ROC Score ({YEAR}): {auc:.4f}")
print("="*40)

# Reporte detallado
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. IMPORTANCIA DE VARIABLES
# ==========================================
importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo_xgb.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\nRANKING VARIABLES (XGBoost {YEAR}):")
print("-" * 50)
print(importancia.to_string(index=False))
print("-" * 50)

# ==========================================
# 6. GRÁFICO
# ==========================================
print("\nGenerando gráfico...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=importancia, hue='Variable', legend=False, palette='magma')
plt.title(f'Importancia de Variables (XGBoost) - {YEAR}')
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.show()