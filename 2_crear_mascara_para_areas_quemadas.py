import os
import sys

# ==============================================================================
# 🛠️ BLOQUE PROJ (Para evitar errores de proyección)
# ==============================================================================
def configurar_proj():
    entorno = sys.exec_prefix
    # Rutas comunes donde se esconde proj.db
    rutas = [
        os.path.join(entorno, 'Library', 'share', 'proj'),
        os.path.join(entorno, 'share', 'proj'),
        os.path.join(entorno, 'Lib', 'site-packages', 'pyproj', 'proj_dir', 'share', 'proj'),
        os.path.join(entorno, 'Lib', 'site-packages', 'rasterio', 'proj_data'),
    ]
    for r in rutas:
        if os.path.exists(os.path.join(r, 'proj.db')):
            os.environ['PROJ_LIB'] = r
            print(f"✅ PROJ configurado en: {r}")
            return
    print("⚠️ No se encontró proj.db automáticamente (si no falla, ignora esto).")

configurar_proj()
# ==============================================================================

import geopandas as gpd
import rasterio
from rasterio import features, mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_origin
import numpy as np

# ================= CONFIGURACIÓN =================
# 1. INPUTS
ruta_area_estudio = r'D:\SIG\shapes\general\Leoncio_Prado.shp'
ruta_raster_agua  = r'D:\SIG\raster\2020\preproceso\Cobertura_agua_2020.tif'
ruta_shp_constr   = r'D:\SIG\shapes\2020\Construcciones_2020_FINAL_COMPLETO_b.shp'

# 2. OUTPUT
carpeta_salida = r'D:\SIG\raster\2020\preproceso'
nombre_salida = 'Mascara_Agua_Construcciones_2020.tif'  # Para cargar en GEE
ruta_salida = os.path.join(carpeta_salida, nombre_salida)

# 3. PARÁMETROS
RESOLUCION = 10        # Tamaño de píxel (metros)
EPSG_OBJETIVO = 32718  # UTM 18S (Para medir en metros)
# =================================================

def generar_mascara_unificada():
    print("--- INICIANDO GENERACIÓN DE MÁSCARA UNIFICADA ---")
    
    # Crear carpeta si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # 1. CARGAR Y PREPARAR EL ÁREA DE ESTUDIO (BASE DE LA GRILLA)
    print("1. Definiendo la cuadrícula base (Leoncio Prado)...")
    gdf_area = gpd.read_file(ruta_area_estudio, encoding='latin1')
    
    if gdf_area.crs.to_epsg() != EPSG_OBJETIVO:
        gdf_area = gdf_area.to_crs(epsg=EPSG_OBJETIVO)

    # Calcular dimensiones de la imagen basada en el Shapefile
    minx, miny, maxx, maxy = gdf_area.total_bounds
    width = int(np.ceil((maxx - minx) / RESOLUCION))
    height = int(np.ceil((maxy - miny) / RESOLUCION))
    
    # Transformación (Georreferenciación)
    transform_base = from_origin(minx, maxy, RESOLUCION, RESOLUCION)
    
    print(f"   -> Dimensiones: {width} x {height} píxeles")
    
    # 2. RASTERIZAR CONSTRUCCIONES (SHAPE -> RASTER)
    print("2. Rasterizando construcciones...")
    gdf_constr = gpd.read_file(ruta_shp_constr, encoding='latin1')
    
    if gdf_constr.crs.to_epsg() != EPSG_OBJETIVO:
        gdf_constr = gdf_constr.to_crs(epsg=EPSG_OBJETIVO)
    
    # Creamos array vacío de construcciones
    # all_touched=True asegura que si el polígono toca el pixel, se pinta
    array_constr = features.rasterize(
        shapes=((geom, 1) for geom in gdf_constr.geometry),
        out_shape=(height, width),
        transform=transform_base,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8
    )

    # 3. ALINEAR RASTER DE AGUA (RASTER -> RASTER)
    print("3. Alineando raster de agua a la nueva cuadrícula...")
    
    # Array vacío para el agua alineada
    array_agua = np.zeros((height, width), dtype=rasterio.uint8)
    
    with rasterio.open(ruta_raster_agua) as src:
        # Reproyectamos el agua para que calce EXACTAMENTE en la grilla base
        # Usamos 'nearest' para mantener el valor 1 puro (sin interpolar decimales)
        reproject(
            source=rasterio.band(src, 1),
            destination=array_agua,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_base,
            dst_crs=f'EPSG:{EPSG_OBJETIVO}',
            resampling=Resampling.nearest
        )

    # 4. COMBINAR (AGUA O CONSTRUCCIÓN)
    print("4. Fusionando capas (Lógica: Agua OR Construcción)...")
    # Si es agua (1) O construcción (1), el resultado es 1. Si no, 0.
    array_final = ((array_agua == 1) | (array_constr == 1)).astype(rasterio.uint8)

    # 5. RECORTAR CON EL POLÍGONO DEL ÁREA DE ESTUDIO
    # (Para que lo de afuera del contorno sea NoData/0)
    print("5. Recortando bordes externos...")
    
    # Usamos MemoryFile para hacer el recorte en memoria antes de guardar
    from rasterio.io import MemoryFile
    
    meta_temp = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': EPSG_OBJETIVO,
        'transform': transform_base,
        'nodata': 0,
        'compress': 'lzw'
    }

    with MemoryFile() as memfile:
        with memfile.open(**meta_temp) as dataset_temp:
            dataset_temp.write(array_final, 1)
            
            # Aplicamos la máscara del polígono Leoncio Prado
            out_image, out_transform = mask.mask(
                dataset_temp,
                gdf_area.geometry,
                crop=True, # Recorta al bounding box ajustado
                nodata=0
            )
            
            # Actualizamos metadata final
            out_meta = dataset_temp.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

    # 6. GUARDAR RESULTADO
    print(f"6. Guardando en: {ruta_salida}")
    with rasterio.open(ruta_salida, "w", **out_meta) as dest:
        dest.write(out_image)

    print("\n✅ ¡MÁSCARA CREADA CON ÉXITO!")
    print(f"   Archivo: {nombre_salida}")
    print("   Valores: 1 = Obstáculo (Agua o Casa), 0 = Terreno libre")

if __name__ == "__main__":
    try:
        generar_mascara_unificada()
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")