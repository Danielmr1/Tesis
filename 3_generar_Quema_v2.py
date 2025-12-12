import rasterio
from rasterio import features
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
import os
from datetime import datetime, timedelta

# ================= CONFIGURACIÓN =================
YEAR = 2022  # <--- ¡CAMBIA ESTO POR EL AÑO QUE QUIERAS! (2020, 2021, 2023...)

# Rutas dinámicas (fr'...' permite usar llaves {} dentro de rutas de Windows)
input_raster = fr'D:\SIG\raster\{YEAR}\leoncio_prado_{YEAR}_mosaic.tif' 


output_file = fr'D:\SIG\shapes\{YEAR}\Puntos_Quema_{YEAR}.shp'

# Parámetros
num_puntos_objetivo = 1100  
distancia_minima = 250      
umbral_ruido = 10           
epsg_origen = 32718         
banda_mascara = 1
banda_fecha = 2
# =================================================

def procesar_incendios():
    print(f"--- Procesando Año {YEAR} con Tipos de Datos Correctos ---")
    
    # 0. VERIFICAR EXISTENCIA DE CARPETA DE SALIDA
    carpeta_salida = os.path.dirname(output_file)
    if not os.path.exists(carpeta_salida):
        try:
            os.makedirs(carpeta_salida)
            print(f"📁 Carpeta creada: {carpeta_salida}")
        except OSError:
            print(f"⚠️ Error al crear carpeta: {carpeta_salida}")

    # 0.1 VERIFICAR EXISTENCIA DE INPUT
    if not os.path.exists(input_raster):
        print(f"❌ ERROR: No se encuentra el archivo de entrada: {input_raster}")
        return

    # 1. LEER MÁSCARA
    print(f"1. Leyendo raster...")
    with rasterio.open(input_raster) as src:
        band_mask = src.read(banda_mascara)
        transform = src.transform
        sieved_band = features.sieve(band_mask, size=umbral_ruido, connectivity=8)

    # 2. POLIGONIZAR
    print("2. Poligonizando áreas de quema...")
    results = (
        {'properties': {'DN': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            shapes(sieved_band, mask=(sieved_band > 0), transform=transform))
    )
    geoms = list(results)
    if not geoms: 
        print("⚠️ No se encontraron polígonos de quema.")
        return

    gdf = gpd.GeoDataFrame.from_features(geoms, crs=f"EPSG:{epsg_origen}")
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf['area'] = gdf.area

    # 3. GENERAR PUNTOS
    print("3. Generando puntos aleatorios...")
    total_area = gdf['area'].sum()
    gdf['num_puntos'] = (gdf['area'] / total_area * num_puntos_objetivo).round().astype(int)
    
    diff = num_puntos_objetivo - gdf['num_puntos'].sum()
    if diff != 0 and len(gdf) > 0:
        gdf.loc[gdf['area'].idxmax(), 'num_puntos'] += diff

    def get_points_with_min_dist(poly, n_needed, min_dist):
        points = []
        minx, miny, maxx, maxy = poly.bounds
        max_intentos = n_needed * 50 
        intentos = 0
        while len(points) < n_needed and intentos < max_intentos:
            intentos += 1
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if not poly.contains(pnt): continue
            
            muy_cerca = False
            for existing_p in points:
                if pnt.distance(existing_p) < min_dist:
                    muy_cerca = True; break
            if not muy_cerca: points.append(pnt)
        return points

    lista_puntos_geom = []
    subset = gdf[gdf['num_puntos'] > 0]
    for idx, row in subset.iterrows():
        pts = get_points_with_min_dist(row.geometry, row['num_puntos'], distancia_minima)
        lista_puntos_geom.extend(pts)

    if not lista_puntos_geom:
        print("⚠️ No se generaron puntos.")
        return

    # 4. MUESTREO
    print("4. Extrayendo fechas del raster...")
    coords = [(p.x, p.y) for p in lista_puntos_geom]
    valores_muestreados = []
    with rasterio.open(input_raster) as src:
        for val in src.sample(coords, indexes=banda_fecha):
            valores_muestreados.append(val[0])

    # 5. CONSTRUIR RESULTADO
    gdf_final = gpd.GeoDataFrame(
        {'dias_julianos': valores_muestreados}, 
        geometry=lista_puntos_geom, 
        crs=f"EPSG:{epsg_origen}"
    )

    # --- A) FECHA (Tipo DATE) ---
    base_date = datetime(1970, 1, 1)
    def calcular_fecha(dias):
        try:
            dias_int = int(dias)
            if dias_int < 1000: return None 
            return base_date + timedelta(days=dias_int)
        except: return None

    gdf_final['fecha'] = gdf_final['dias_julianos'].apply(calcular_fecha)
    gdf_final = gdf_final.dropna(subset=['fecha'])
    
    # Convertir explícitamente a formato datetime
    gdf_final['fecha'] = pd.to_datetime(gdf_final['fecha'])

    # --- B) CLASS (Tipo INTEGER) ---
    gdf_final['class'] = 1
    gdf_final['class'] = gdf_final['class'].astype(int)

    # --- C) LAT / LON (Tipo REAL / FLOAT) ---
    gdf_wgs84 = gdf_final.to_crs(epsg=4326)
    
    gdf_final['lon'] = gdf_wgs84.geometry.x.round(6).astype(float)
    gdf_final['lat'] = gdf_wgs84.geometry.y.round(6).astype(float)

    # Seleccionar y reordenar
    cols_finales = ['fecha', 'lat', 'lon', 'class', 'geometry'] 
    gdf_export = gdf_final[cols_finales]

    print(f"5. Guardando Shapefile en: {output_file}")
    
    # Guardamos con driver por defecto
    try:
        gdf_export.to_file(output_file, driver='ESRI Shapefile')
    except Exception as e:
        print(f"❌ Error al guardar shapefile: {e}")
        return

    if len(gdf_export) > 0:
        print("\n--- ¡ÉXITO! ---")
        print(f"Año: {YEAR}")
        print(f"Puntos generados: {len(gdf_export)}")
        print("Tipos de datos:")
        print(gdf_export.dtypes)
        print("-" * 30)
    else:
        print("Advertencia: El archivo final está vacío.")

if __name__ == "__main__":
    procesar_incendios()