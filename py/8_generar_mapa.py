import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import joblib
import os

# ==============================================================================
# 1. CONFIGURACIÓN (INPUTS)
# ==============================================================================
YEAR = 2020  # <--- CAMBIA EL AÑO AQUÍ (2020, 2021, 2022...)

# A. Ruta del Modelo Entrenado (.pkl)
# Este archivo es único (o puedes tener uno por año si entrenaste separado)
ruta_modelo = r'D:\SIG\modelos\Modelo_Incendios_XGBoost_Entrenado.pkl'

# B. Ruta del Stack de Variables (Descargado de GEE)
input_stack = fr'D:\SIG\raster\{YEAR}\Stack_Susceptibilidad_{YEAR}_Estandarizado.tif'

# C. Ruta de la Máscara de Exclusión (Agua/Construcciones)
# Se asume que está en la carpeta del año correspondiente
ruta_mascara = fr'D:\SIG\raster\{YEAR}\Mascara_Agua_Construcciones_{YEAR}.tif'

# D. Ruta de Salida (Mapa Final)
output_map = fr'D:\SIG\raster\{YEAR}\Mapa_Susceptibilidad_Final_{YEAR}.tif'

# E. Definición de Bandas (Orden EXACTO del script de GEE)
# El modelo espera los nombres estandarizados (sin _2020)
nombres_bandas = [
    'elev', 'slope', 'aspect',      # Topografía
    'dist_vias',                    # Distancias
    'dist_water', 
    'dist_built',
    'precip_60d',                   # Clima
    'temp_mean',
    'ndvi_mean'                     # Vegetación
]

# ==============================================================================
# 2. PROCESAMIENTO
# ==============================================================================
print(f"🌍 Generando Mapa de Susceptibilidad para el año {YEAR}...")

# --- PASO 1: CARGAR MODELO ---
if not os.path.exists(ruta_modelo):
    print(f"❌ ERROR: No se encuentra el modelo: {ruta_modelo}")
    exit()

print("1. Cargando cerebro digital (Modelo XGBoost)...")
modelo = joblib.load(ruta_modelo)

# --- PASO 2: LEER STACK DE VARIABLES ---
if not os.path.exists(input_stack):
    print(f"❌ ERROR: No se encuentra el Stack Tiff: {input_stack}")
    exit()

print(f"2. Leyendo imagen satelital: {os.path.basename(input_stack)}")
with rasterio.open(input_stack) as src:
    meta = src.meta.copy()
    height = src.height
    width = src.width
    crs_stack = src.crs
    transform_stack = src.transform
    
    # Leer datos (Bandas, Filas, Columnas)
    img_data = src.read()
    
    # Verificación de seguridad
    if img_data.shape[0] != len(nombres_bandas):
        print(f"❌ ERROR: El Tiff tiene {img_data.shape[0]} bandas, pero se definieron {len(nombres_bandas)} nombres.")
        exit()

# --- PASO 3: PREPARAR DATOS ---
print("3. Preparando matriz de datos...")
n_pixels = height * width

# Aplanar: (Bandas, Y, X) -> (Pixeles, Bandas)
X_flat = img_data.reshape(img_data.shape[0], n_pixels).T 
df_map = pd.DataFrame(X_flat, columns=nombres_bandas)

# Filtro de Nulos (Para no predecir en bordes o nubes)
valid_pixels_mask = ~df_map.isnull().any(axis=1)

# --- PASO 4: PREDICCIÓN (INFERENCIA) ---
print("4. Realizando predicción (Calculando probabilidades)...")

# Array base lleno de -9999 (NoData)
mapa_plano = np.full(n_pixels, -9999.0, dtype=np.float32)

if valid_pixels_mask.sum() > 0:
    # Predecir solo datos válidos
    datos_validos = df_map.loc[valid_pixels_mask]
    probs = modelo.predict_proba(datos_validos)[:, 1] # Columna 1 = Probabilidad de Incendio
    mapa_plano[valid_pixels_mask] = probs
else:
    print("⚠️ ADVERTENCIA: La imagen parece estar vacía o llena de nulos.")

# Reconstruir imagen 2D
mapa_final_2d = mapa_plano.reshape(height, width)

# --- PASO 5: APLICAR MÁSCARA DE EXCLUSIÓN (Agua/Construcciones) ---
print("5. Aplicando máscara de exclusión (Limpiando ríos y zonas urbanas)...")

if os.path.exists(ruta_mascara):
    with rasterio.open(ruta_mascara) as src_mask:
        # Crear array vacío del tamaño del MAPA FINAL (probablemente 20m)
        mascara_ajustada = np.zeros((height, width), dtype=np.uint8)
        
        # Reproyectar/Remuestrear la máscara (de 10m a 20m) al vuelo
        reproject(
            source=rasterio.band(src_mask, 1),
            destination=mascara_ajustada,
            src_transform=src_mask.transform,
            src_crs=src_mask.crs,
            dst_transform=transform_stack,
            dst_crs=crs_stack,
            resampling=Resampling.nearest # Nearest conserva los valores 0 y 1 puros
        )
        
        # Aplicar el borrado: Donde máscara es 1, poner NoData (-9999)
        # 1 en máscara = Agua/Construcción
        pixeles_a_borrar = (mascara_ajustada == 1)
        mapa_final_2d[pixeles_a_borrar] = -9999.0
        
        count_borrados = np.sum(pixeles_a_borrar)
        print(f"   -> Se enmascararon {count_borrados} píxeles (Ríos/Casas eliminados).")
else:
    print(f"⚠️ ALERTA: No se encontró la máscara en: {ruta_mascara}")
    print("   -> El mapa se guardará sin limpiar los ríos (puede haber falsos positivos).")

# --- PASO 6: GUARDAR RESULTADO ---
print(f"6. Guardando archivo final...")

# Actualizar metadata para 1 sola banda float
meta.update({
    'count': 1,
    'dtype': 'float32',
    'nodata': -9999
})

# Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(output_map), exist_ok=True)

with rasterio.open(output_map, 'w', **meta) as dst:
    dst.write(mapa_final_2d, 1)

print("\n" + "="*50)
print(f"✅ ¡ÉXITO! Mapa de Susceptibilidad {YEAR} generado.")
print(f"📂 Archivo: {output_map}")
print("="*50)