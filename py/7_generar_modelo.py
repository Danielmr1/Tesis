import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # <--- LIBRERÍA PARA GUARDAR EL MODELO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
YEAR = 2020  # Usa el año con el que quieras entrenar (o une varios años)

# Archivo de entrada (CSV Balanceado)
archivo = fr'D:\SIG\csv\{YEAR}\Dataset_{YEAR}_BALANCEADO_FINAL.csv'

# Archivo de SALIDA (El modelo guardado)
# Nota: Lo guardamos en una carpeta general "modelos"
ruta_modelo_salida = r'D:\SIG\modelos\Modelo_Incendios_XGBoost_Entrenado.pkl'

# ==========================================
# 2. CARGAR Y PREPARAR DATOS
# ==========================================
print(f"🔥 Cargando datos del año: {YEAR}")

if not os.path.exists(archivo):
    print(f"❌ ERROR: No se encuentra el archivo: {archivo}")
    exit()

df = pd.read_csv(archivo)

# --- PASO CRUCIAL: ESTANDARIZAR NOMBRES ---
# Quitamos el "_2020" para que el modelo sea universal y sirva para 2021, 2022...
mapa_nombres = {
    f'dist_vias_{YEAR}': 'dist_vias',
    f'dist_water_{YEAR}': 'dist_water',
    f'dist_built_{YEAR}': 'dist_built'
}
df = df.rename(columns=mapa_nombres)
print("✅ Nombres de columnas estandarizados (sin año).")

# Definir X e y
# Borramos las columnas que no sirven para predecir
X = df.drop(columns=['class', 'year', 'fecha_txt', 'lat', 'lon'])
y = df['class']

# División Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
print("🚀 Entrenando XGBoost...")

modelo = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train)

# ==========================================
# 4. EVALUACIÓN (Opcional, para verificar)
# ==========================================
y_prob = modelo.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"⭐ AUC-ROC Score: {auc:.4f}")

# ==========================================
# 5. GUARDAR EL MODELO (EL PASO QUE FALTABA)
# ==========================================
print("-" * 30)
print("💾 GUARDANDO MODELO EN DISCO...")

# Crear carpeta si no existe
carpeta_modelos = os.path.dirname(ruta_modelo_salida)
if not os.path.exists(carpeta_modelos):
    os.makedirs(carpeta_modelos)

# Guardar el objeto 'modelo' en un archivo .pkl
joblib.dump(modelo, ruta_modelo_salida)

print(f"✅ ¡LISTO! Modelo guardado exitosamente en:")
print(ruta_modelo_salida)
print("-" * 30)
print("Ahora puedes usar este archivo .pkl para generar mapas en cualquier script.")