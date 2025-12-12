// =================================================================================
// 1. CONFIGURACIÓN (¡CAMBIA ESTO POR AÑO!)
// =================================================================================
var YEAR = 2022; // <--- CAMBIA AQUÍ: 2020, 2021, 2022, etc.

print('⚙️ Configurando para el año:', YEAR);

// Construcción dinámica de rutas (Asegúrate que tus archivos en Assets sigan este patrón)
var ruta_firms = 'projects/generar-contorno/assets/' + YEAR + '/Leoncio_Prado_cuadrados_FIRMS_' + YEAR;
var ruta_mascara = 'projects/generar-contorno/assets/' + YEAR + '/Mascara_Agua_Construcciones_' + YEAR; 

// =================================================================================
// CARGA DE DATOS
// =================================================================================
var fireAreas = ee.FeatureCollection(ruta_firms);
var exclusionMask = ee.Image(ruta_mascara);
var filteredFireAreas = fireAreas.filter(ee.Filter.gte('confidence', 60));

print('📂 Leyendo FIRMS de:', ruta_firms);
print('📂 Leyendo Máscara de:', ruta_mascara);

Map.centerObject(filteredFireAreas, 10);
Map.addLayer(filteredFireAreas, {color: 'white', fillColor: '00000000'}, 'Cuadrados FIRMS ' + YEAR);

// =================================================================================
// FUNCIONES
// =================================================================================

var maskCloudsSCL = function(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(1).and(scl.neq(3)).and(scl.neq(6)).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask).divide(10000);
};

var calculateDeltaNBR = function(feature) {
  var acqDate = ee.Date(feature.get('acq_date'));

  // Ventanas de tiempo
  var pre_fire_start = acqDate.advance(-60, 'day');
  var pre_fire_end = acqDate.advance(-1, 'day');
  var post_fire_start = acqDate.advance(1, 'day');
  var post_fire_end = acqDate.advance(60, 'day');

  // Imágenes
  var pre_fire_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(feature.geometry())
    .filterDate(pre_fire_start, pre_fire_end)
    .map(maskCloudsSCL).median();

  var post_fire_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(feature.geometry())
    .filterDate(post_fire_start, post_fire_end)
    .map(maskCloudsSCL).median();

  // Índices
  var nbr_pre = pre_fire_image.normalizedDifference(['B8', 'B12']).rename('NBR_pre');
  var nbr_post = post_fire_image.normalizedDifference(['B8', 'B12']).rename('NBR_post');
  
  // dNBR con unmask para evitar huecos en la matemática
  var delta_nbr = nbr_pre.subtract(nbr_post).rename('dNBR').unmask(0);
  
  // --- LÓGICA DE DETECCIÓN ---
  var threshold = 0.27;
  var potential_burn = delta_nbr.gte(threshold);
  
  // Aplicar tu Máscara de Obstáculos (Agua/Casas)
  var is_obstacle = exclusionMask.eq(1).unmask(0); // 1 = Obstáculo
  var is_valid_surface = is_obstacle.not();        // 1 = Suelo Válido
  
  // Resultado Binario (1 = Quemado y Válido, 0 = No)
  var final_burn_mask = potential_burn.and(is_valid_surface).rename('burn_binary');

  // --- FECHA ---
  var baseDate = ee.Date('1970-01-01');
  var days = acqDate.difference(baseDate, 'day');
  
  // Creamos la banda de fecha.
  var date_band = ee.Image.constant(days).rename('burn_date').toInt16()
    .updateMask(final_burn_mask); 

  // Retornamos la imagen
  return final_burn_mask.addBands(date_band).toInt16().clip(feature.geometry());
};

// =================================================================================
// PROCESAMIENTO Y EXPORTACIÓN (Con qualityMosaic)
// =================================================================================

var fireAreasList = filteredFireAreas.toList(filteredFireAreas.size());
var totalAreasCount = fireAreasList.size().getInfo();
print('Total áreas encontradas:', totalAreasCount);

// 1. Crear colección de imágenes
var allBurnedMasks = fireAreasList.map(function(feature) {
  return calculateDeltaNBR(ee.Feature(feature));
});
var allBurnedCollection = ee.ImageCollection.fromImages(allBurnedMasks);

// 2. REDUCCIÓN POR CALIDAD
var allBurnedMosaic = allBurnedCollection.qualityMosaic('burn_binary');

// 3. Reproyección Final
var targetCRS = 'EPSG:32718'; 
var scale = 20; 

var allBurnedMosaic_utm = allBurnedMosaic.reproject({
  crs: targetCRS,
  scale: scale
});

// 4. Exportar
Export.image.toDrive({
  image: allBurnedMosaic_utm.select(['burn_binary', 'burn_date']),
  description: 'leoncio_prado_' + YEAR + '_mosaic', // Nombre dinámico
  scale: scale,
  region: filteredFireAreas.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

print('✅ Tarea lista: union_' + YEAR + '_mosaic');