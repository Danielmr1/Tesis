import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import os # Necesario para verificar carpetas

# ================= CONFIGURACIÓN =================
YEAR = 2021  # <--- ¡CAMBIA ESTO POR AÑO! (2020, 2021, 2022, etc.)

# Rutas dinámicas usando f-strings (f"...")
# Nota: La 'r' es para rutas de Windows, la 'f' es para insertar variables. Se usan juntas: fr"..."

input_csv = fr'D:\SIG\csv\{YEAR}\FIRMS_{YEAR}.csv'
output_squares_shp = fr'D:\SIG\shapes\{YEAR}\Leoncio_Prado_cuadrados_FIRMS_{YEAR}.shp'

# El AOI suele ser estático (el mismo para todos los años), se deja igual
aoi_shp = r'D:\SIG\shapes\general\Leoncio_Prado.shp'
# =================================================

print(f"--- Procesando Año: {YEAR} ---")

# 0. VERIFICACIÓN DE CARPETAS (Seguridad)
# Si la carpeta de salida (ej: D:\SIG\shapes\2022) no existe, la crea.
carpeta_salida = os.path.dirname(output_squares_shp)
if not os.path.exists(carpeta_salida):
    try:
        os.makedirs(carpeta_salida)
        print(f"📁 Carpeta creada automáticamente: {carpeta_salida}")
    except OSError:
        print(f"⚠️ Error al intentar crear la carpeta: {carpeta_salida}")

# 1. CARGAR DATOS
print(f"1. Leyendo CSV: {input_csv}")
try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo CSV del año {YEAR}.")
    exit()

aoi = gpd.read_file(aoi_shp)

# 2. LIMPIEZA DE DATOS
df['confidence'] = df['confidence'].replace({'h': 80, 'n': 60, 'l': 30})
df['confidence'] = pd.to_numeric(df['confidence'])

# 3. CREAR GEODATAFRAME DE PUNTOS
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Asegurar misma proyección AOI
if aoi.crs != gdf.crs:
    aoi = aoi.to_crs(gdf.crs)

# 4. FILTRAR PUNTOS DENTRO DEL AOI
print("2. Filtrando puntos dentro del área de interés...")
aoi_union = aoi.geometry.union_all()
gdf_filtrado = gdf[gdf.within(aoi_union)].copy()

if gdf_filtrado.empty:
    print(f"❌ ALERTA: No hay puntos FIRMS en el año {YEAR} dentro del AOI.")
else:
    print(f"   -> {len(gdf_filtrado)} puntos encontrados.")

    # 5. REPROYECTAR Y GENERAR CUADRADOS
    print("3. Generando cuadrados (UTM 18S)...")
    gdf_utm = gdf_filtrado.to_crs('EPSG:32718')
    aoi_utm = aoi.to_crs('EPSG:32718')

    def create_dynamic_square(row):
        if row['instrument'] == 'MODIS':
            size = 1000
        elif row['instrument'] == 'VIIRS':
            size = 375
        else:
            size = 375
        
        half_size = size / 2
        point = row.geometry
        return box(point.x - half_size, point.y - half_size, point.x + half_size, point.y + half_size)

    gdf_utm['geometry'] = gdf_utm.apply(create_dynamic_square, axis=1)

    # 6. RECORTAR CON AOI
    print("4. Recortando bordes...")
    gdf_recortado = gpd.overlay(gdf_utm, aoi_utm, how='intersection')

    # 7. GUARDAR
    print(f"5. Guardando: {output_squares_shp}")
    gdf_recortado.to_file(output_squares_shp)
    
    print(f"✅ ¡Proceso del año {YEAR} finalizado!")