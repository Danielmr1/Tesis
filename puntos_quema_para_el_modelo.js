// ==========================================================
// SCRIPT UNIVERSAL: EXTRACCIÓN DE VARIABLES (CLASE 1 - PUNTOS QGIS)
// ==========================================================

// 1. CONFIGURACIÓN (¡CAMBIA ESTO POR AÑO!)
var YEAR = 2022; // Cambia a 2021, 2022, etc.

// 2. CARGAR TU ASSET DE PUNTOS DE ESE AÑO
var puntos = ee.FeatureCollection("projects/generar-contorno/assets/" + YEAR + "/Puntos_Quema_" + YEAR); 

// ----------------------------------------------------------
// 3. CARGA DE VARIABLES AMBIENTALES
// ----------------------------------------------------------

// A. TOPOGRAFÍA (Estática)
var topo = ee.Image("projects/generar-contorno/assets/Variables/Variables_Topograficas")
    .rename(['elev', 'slope', 'aspect']); 

// B. DISTANCIA VÍAS (DINÁMICO)
var dist_vias = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Vias_" + YEAR)
    .rename(['dist_vias_' + YEAR]); 

// C. DISTANCIAS RÍOS (DINÁMICO)
var dist_agua = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Rios_" + YEAR)
    .rename(['dist_water_' + YEAR]); 

// D. DISTANCIAS CONSTRUCCIONES (DINÁMICO)
var dist_urbano = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Construcciones_" + YEAR)
    .rename(['dist_built_' + YEAR]);

// ----------------------------------------------------------
// 4. FUNCIÓN DE EXTRACCIÓN
// ----------------------------------------------------------
var extraerDatos = function(feature) {
  // 1. GESTIÓN DE FECHAS
  var valFecha = feature.get('fecha'); 
  var fechaEvento = ee.Date(valFecha); 
  
  // 2. CÁLCULO DE CLIMA
  
  // Precipitación
  var precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterDate(fechaEvento.advance(-60, 'day'), fechaEvento)
    .sum()
    .rename('precip_60d');

  // Temperatura: CORREGIDO (ID Correcto: MODIS/061/MOD11A1)
  var temp = ee.ImageCollection("MODIS/061/MOD11A1") // <--- AQUÍ ESTABA EL ERROR (es 061, no 0061)
    .filterDate(fechaEvento.advance(-15, 'day'), fechaEvento)
    .select('LST_Day_1km')
    .map(function(img) {
      return img.multiply(0.02).subtract(273.15);
    })
    .mean()
    .rename('temp_mean');

  // NDVI
  var ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(feature.geometry())
    .filterDate(fechaEvento.advance(-30, 'day'), fechaEvento)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
    .map(function(img) {
      return img.normalizedDifference(['B8', 'B4']).rename('ndvi_mean');
    })
    .median()
    .rename('ndvi_mean');

  // 3. STACK
  var stack = topo
    .addBands(dist_vias)
    .addBands(dist_agua)
    .addBands(dist_urbano)
    .addBands(precip.unmask(0))
    .addBands(temp) 
    .addBands(ndvi);

  // 4. EXTRAER VALORES
  var valores = stack.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: feature.geometry(),
    scale: 20
  });
  
  return feature.set(valores)
    .set('year', YEAR)
    .set('fecha_txt', fechaEvento.format('YYYY-MM-dd'));
};

// ----------------------------------------------------------
// 5. EJECUCIÓN
// ----------------------------------------------------------

var datasetEnriquecido = puntos.map(extraerDatos);

// Columnas a exportar
var columnas = [
  'class', 'year', 'fecha_txt', 'lat', 'lon',      
  'elev', 'slope', 'aspect',                       
  'dist_vias_' + YEAR,       
  'dist_water_' + YEAR, 
  'dist_built_' + YEAR, 
  'precip_60d', 'temp_mean',
  'ndvi_mean' 
];

Export.table.toDrive({
  collection: datasetEnriquecido,
  description: 'Dataset_Entrenamiento_' + YEAR + '_CLASE1',
  folder: "" + YEAR,
  fileFormat: 'CSV',
  selectors: columnas 
});

print('Procesando Clase 1 (' + YEAR + ')');