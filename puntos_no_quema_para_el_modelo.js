// ==========================================================
// SCRIPT UNIVERSAL: GENERADOR CLASE 0 (NO QUEMA) - FINAL
// ==========================================================

// 1. CONFIGURACIÓN DEL AÑO
var YEAR = 2022; // <--- CAMBIAR AQUÍ: 2020, 2021, etc.

// 2. CONFIGURACIÓN Y ASSETS
var leoncioPrado = ee.FeatureCollection("projects/generar-contorno/assets/Leoncio_Prado_superficie");
var fechaInicio = YEAR + '-01-01';
var fechaFin = YEAR + '-12-31';

// ----------------------------------------------------------
// CARGAR ASSETS DINÁMICOS
// ----------------------------------------------------------
var dist_agua = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Rios_" + YEAR).rename(['dist_water_' + YEAR]);
var dist_urbano = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Construcciones_" + YEAR).rename(['dist_built_' + YEAR]);
var dist_vias = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Distancia_Vias_" + YEAR).rename(['dist_vias_' + YEAR]); 
var topo = ee.Image("projects/generar-contorno/assets/Variables/Variables_Topograficas").rename(['elev', 'slope', 'aspect']);

// ASSETS DE EXCLUSIÓN
var firmsPolys = ee.FeatureCollection("projects/generar-contorno/assets/" + YEAR + "/Leoncio_Prado_cuadrados_FIRMS_" + YEAR);
var mascaraObstaculos = ee.Image("projects/generar-contorno/assets/" + YEAR + "/Mascara_Agua_Construcciones_" + YEAR);

// ----------------------------------------------------------
// 3. CREAR MÁSCARA DE ZONA ELIGIBLE (CLASE 0)
// ----------------------------------------------------------

// A. Dynamic World
var dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
  .filterDate(fechaInicio, fechaFin)
  .filterBounds(leoncioPrado)
  .select('label')
  .mode(); 
var mask_Agua = dw.neq(0); 
var mask_Urbano = dw.neq(6);
var mask_SueloDesnudo = dw.neq(7); 

// B. Filtro FIRMS
var img_FIRMS = ee.Image(0).byte().paint(firmsPolys, 1);
var mask_NoFIRMS = img_FIRMS.eq(0);

// C. Filtro Obstáculos
var mask_NoObstaculos = mascaraObstaculos.unmask(0).neq(1);

// --- UNIÓN ---
var zonaEligible = mask_NoFIRMS
    .and(mask_NoObstaculos)
    .and(mask_Agua)
    .and(mask_Urbano)
    .and(mask_SueloDesnudo)
    .clip(leoncioPrado.geometry())
    .rename('eligible');

Map.centerObject(leoncioPrado, 10);
Map.addLayer(zonaEligible.selfMask(), {palette: ['00FF00']}, 'Zona Eligible (Clase 0)', true);

// ----------------------------------------------------------
// 4. GENERAR PUNTOS Y VALIDAR POR PÍXEL
// ----------------------------------------------------------

var puntosCandidatos = ee.FeatureCollection.randomPoints({
  region: leoncioPrado.geometry(),
  points: 2500, 
  seed: YEAR 
});

var validarUbicacion = function(feature) {
  var valorPixel = zonaEligible.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: feature.geometry(),
    scale: 10 
  }).get('eligible');
  
  return feature.set('es_valido', valorPixel);
};

var puntosFiltrados = puntosCandidatos
  .map(validarUbicacion)
  .filter(ee.Filter.eq('es_valido', 1)) 
  .limit(1000);

var puntosFinales = puntosFiltrados.randomColumn('random').map(function(feature){
  var dias = ee.Number(feature.get('random')).multiply(365).toInt();
  var fecha = ee.Date(fechaInicio).advance(dias, 'day');
  var coords = feature.geometry().coordinates();
  
  return feature.set({
      'fecha_obj': fecha,
      'fecha_txt': fecha.format('YYYY-MM-dd'),
      'lat': coords.get(1),
      'lon': coords.get(0),
      'class': 0,
      'year': YEAR
  });
});

print('Puntos válidos encontrados:', puntosFinales.size());

// ----------------------------------------------------------
// 5. EXTRACCIÓN DE VARIABLES
// ----------------------------------------------------------

var extraerDatos = function(feature) {
  var fechaEvento = ee.Date(feature.get('fecha_obj'));
  
  // Clima: Precipitación
  var precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterDate(fechaEvento.advance(-60, 'day'), fechaEvento)
    .sum().rename('precip_60d');

  // Clima: Temperatura (CORREGIDO - MODIS V6.1)
  // -------------------------------------------------------
  var temp = ee.ImageCollection("MODIS/061/MOD11A1") // <--- CAMBIO 1: Colección 061
    .filterDate(fechaEvento.advance(-15, 'day'), fechaEvento)
    .select('LST_Day_1km')
    .map(function(img) { // <--- CAMBIO 2: Matemática segura
      return img.multiply(0.02).subtract(273.15);
    })
    .mean()
    .rename('temp_mean');
  // -------------------------------------------------------

  // NDVI Dinámico
  var s2_ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(feature.geometry())
    .filterDate(fechaEvento.advance(-30, 'day'), fechaEvento)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
    .map(function(img) {
      return img.normalizedDifference(['B8', 'B4']).rename('ndvi_mean');
    })
    .median()
    .rename('ndvi_mean');

  // Stack Final
  var stack = topo
    .addBands(dist_vias)
    .addBands(dist_agua) 
    .addBands(dist_urbano)
    .addBands(precip.unmask(0))
    .addBands(temp) // Si viene vacía, no rompe el código
    .addBands(s2_ndvi);

  // Extracción
  var valores = stack.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: feature.geometry(),
    scale: 20
  });
  
  return feature.set(valores);
};

var datasetClase0 = puntosFinales.map(extraerDatos);

// ----------------------------------------------------------
// 6. EXPORTAR
// ----------------------------------------------------------

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
  collection: datasetClase0,
  description: 'Dataset_Entrenamiento_' + YEAR + '_CLASE0',
  folder: "" + YEAR,
  fileFormat: 'CSV',
  selectors: columnas 
});

print('Procesando Clase 0 (' + YEAR + ')');