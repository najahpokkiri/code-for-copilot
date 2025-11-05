// ============================================================================
// Building Height Validation - FINAL WORKING VERSION
// ============================================================================
// Tailored for: Point geometries with comprehensive storey data
// Asset: building_enrichment_tsi_proportions_ind
// ============================================================================

// =========================
// CONFIGURATION
// =========================

var CONFIG = {
  gridAsset: 'projects/ee-smbattagliajorc/assets/building_enrichment_tsi_proportions_ind',
  gridCellSize: 5000, // 5km in meters
  
  region: 'DELHI',
  
  openBuildings: {
    presenceThreshold: 0.3,
    fractionalCountThreshold: 0.0005,
    heightMin: 2,
    heightMax: 100
  },
  
  sampling: {
    enabled: true,
    size: 100  // Process 100 grids
  },
  
  storeyHeight: 3.5, // meters per storey
  minPixelsForValid: 5,
  
  export: {
    enabled: true,
    description: 'building_validation_results'
  }
};

// Regions
var REGIONS = {
  DELHI: ee.Geometry.Rectangle([77.0, 28.4, 77.5, 28.9]),
  CAIRO: ee.Geometry.Rectangle([31.1, 29.9, 31.5, 30.2]),
  MUMBAI: ee.Geometry.Rectangle([72.8, 18.9, 72.95, 19.3])
};

var roi = REGIONS[CONFIG.region];

// =========================
// LOAD DATA
// =========================

print('==============================================');
print('BUILDING HEIGHT VALIDATION - PROCESSING');
print('==============================================');

// Load grid points
var gridPoints = ee.FeatureCollection(CONFIG.gridAsset).filterBounds(roi);
print('Total grid points in ROI:', gridPoints.size());

// Sample for efficiency
if (CONFIG.sampling.enabled) {
  gridPoints = gridPoints.randomColumn('random', 42)
    .sort('random')
    .limit(CONFIG.sampling.size);
  print('Processing sample size:', CONFIG.sampling.size);
}

// =========================
// CONVERT POINTS TO 5KM CELLS
// =========================

print('Converting point centroids to 5km grid cells...');

/**
 * Convert point geometry to 5km square polygon
 */
function pointToGridCell(feature) {
  // Get original properties
  var props = feature.toDictionary();
  
  // Get point geometry
  var point = feature.geometry();
  
  // Buffer to create 5km cell (2500m radius creates 5km square approximately)
  var buffered = point.buffer(CONFIG.gridCellSize / 2, 30).bounds();
  
  // Return new feature with buffered geometry and all original properties
  return ee.Feature(buffered, props);
}

// Convert all points to grid cells
var gridCells = gridPoints.map(pointToGridCell);
print('Grid cells created');

// =========================
// LOAD OPEN BUILDINGS
// =========================

print('Loading Open Buildings Temporal dataset...');

var openBuildings = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
  .filterBounds(roi)
  .sort('system:time_start', false)
  .first();

var height = openBuildings.select('building_height');
var presence = openBuildings.select('building_presence');
var fractionalCount = openBuildings.select('building_fractional_count');

// Create building mask
var buildingMask = presence.gt(CONFIG.openBuildings.presenceThreshold)
  .and(fractionalCount.gt(CONFIG.openBuildings.fractionalCountThreshold))
  .and(height.gte(CONFIG.openBuildings.heightMin))
  .and(height.lte(CONFIG.openBuildings.heightMax));

var maskedHeight = height.updateMask(buildingMask);

print('Building data prepared');

// =========================
// CALCULATE MODEL HEIGHTS
// =========================

/**
 * Calculate weighted mean height from storey percentages
 */
function calculateModeledHeight(props, lobPrefix) {
  var storeyBands = ['1', '2', '3', '4_5', '6_8', '9_20', '20'];
  var storeyMidpoints = [1, 2, 3, 4.5, 7, 14.5, 40];
  
  var totalWeighted = 0;
  var totalPercent = 0;
  
  for (var i = 0; i < storeyBands.length; i++) {
    var propName = lobPrefix + '_Storey_' + storeyBands[i] + '_perc';
    var percent = props[propName] || 0;
    
    totalWeighted += (percent / 100) * storeyMidpoints[i];
    totalPercent += (percent / 100);
  }
  
  var meanStorey = totalPercent > 0 ? totalWeighted / totalPercent : 0;
  return meanStorey * CONFIG.storeyHeight;
}

// =========================
// EXTRACT STATISTICS
// =========================

print('Extracting building statistics for each grid cell...');

function extractObservedStats(feature) {
  var geom = feature.geometry();
  
  // Get observed building statistics
  var stats = maskedHeight.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.median(), '', true)
      .combine(ee.Reducer.percentile([10, 90]), '', true)
      .combine(ee.Reducer.stdDev(), '', true)
      .combine(ee.Reducer.count(), '', true)
      .combine(ee.Reducer.minMax(), '', true),
    geometry: geom,
    scale: 30,
    maxPixels: 1e8,
    bestEffort: true,
    tileScale: 2
  });
  
  // Extract with null protection
  var pixelCount = ee.Number(stats.get('building_height_count')).max(0);
  var meanHeight = ee.Number(stats.get('building_height_mean')).max(0);
  var medianHeight = ee.Number(stats.get('building_height_median')).max(0);
  var p10 = ee.Number(stats.get('building_height_p10')).max(0);
  var p90 = ee.Number(stats.get('building_height_p90')).max(0);
  var stdDev = ee.Number(stats.get('building_height_stdDev')).max(0);
  var minHeight = ee.Number(stats.get('building_height_min')).max(0);
  var maxHeight = ee.Number(stats.get('building_height_max')).max(0);
  
  // Store in feature properties (will be computed client-side for model heights)
  return feature.set({
    'obs_mean_height': meanHeight,
    'obs_median_height': medianHeight,
    'obs_p10_height': p10,
    'obs_p90_height': p90,
    'obs_std_height': stdDev,
    'obs_min_height': minHeight,
    'obs_max_height': maxHeight,
    'obs_pixel_count': pixelCount,
    'has_buildings': pixelCount.gte(CONFIG.minPixelsForValid)
  });
}

var gridWithStats = gridCells.map(extractObservedStats);

// Filter to valid grids
var validGrids = gridWithStats.filter(ee.Filter.eq('has_buildings', true));

print('Grids with building data:', validGrids.size());

// =========================
// ADD MODEL HEIGHTS
// =========================

print('Calculating modeled heights from storey distributions...');

// This needs to be done client-side or with computed values
// Add modeled heights by mapping over features
function addModelHeights(feature) {
  // Get all properties
  var props = feature.toDictionary();
  
  // Calculate modeled heights for each LOB
  // We'll create computed properties that calculate these on the fly
  var resStoreySum = ee.Number(0);
  var resWeightedSum = ee.Number(0);
  
  var storeyBands = ['1', '2', '3', '4_5', '6_8', '9_20', '20'];
  var storeyMidpoints = [1, 2, 3, 4.5, 7, 14.5, 40];
  
  // Calculate RES weighted height
  for (var i = 0; i < storeyBands.length; i++) {
    var propName = 'RES_Storey_' + storeyBands[i] + '_perc';
    var percent = ee.Number(props.get(propName)).divide(100);
    var midpoint = storeyMidpoints[i];
    
    resWeightedSum = resWeightedSum.add(percent.multiply(midpoint));
    resStoreySum = resStoreySum.add(percent);
  }
  
  var resMeanStorey = resWeightedSum.divide(resStoreySum.max(0.01));
  var resHeight = resMeanStorey.multiply(CONFIG.storeyHeight);
  
  // Do same for COM
  var comStoreySum = ee.Number(0);
  var comWeightedSum = ee.Number(0);
  
  for (var i = 0; i < storeyBands.length; i++) {
    var propName = 'COM_Storey_' + storeyBands[i] + '_perc';
    var percent = ee.Number(props.get(propName)).divide(100);
    var midpoint = storeyMidpoints[i];
    
    comWeightedSum = comWeightedSum.add(percent.multiply(midpoint));
    comStoreySum = comStoreySum.add(percent);
  }
  
  var comMeanStorey = comWeightedSum.divide(comStoreySum.max(0.01));
  var comHeight = comMeanStorey.multiply(CONFIG.storeyHeight);
  
  // IND
  var indStoreySum = ee.Number(0);
  var indWeightedSum = ee.Number(0);
  
  for (var i = 0; i < storeyBands.length; i++) {
    var propName = 'IND_Storey_' + storeyBands[i] + '_perc';
    var percent = ee.Number(props.get(propName)).divide(100);
    var midpoint = storeyMidpoints[i];
    
    indWeightedSum = indWeightedSum.add(percent.multiply(midpoint));
    indStoreySum = indStoreySum.add(percent);
  }
  
  var indMeanStorey = indWeightedSum.divide(indStoreySum.max(0.01));
  var indHeight = indMeanStorey.multiply(CONFIG.storeyHeight);
  
  // Use RES as primary for comparison (or could weight by building counts)
  var modelHeight = resHeight;
  
  // Calculate difference
  var obsHeight = ee.Number(feature.get('obs_mean_height'));
  var diff = modelHeight.subtract(obsHeight);
  var relError = diff.divide(obsHeight.max(1)).multiply(100);
  
  return feature.set({
    'model_res_height': resHeight,
    'model_com_height': comHeight,
    'model_ind_height': indHeight,
    'model_mean_height': modelHeight,
    'height_difference': diff,
    'relative_error_pct': relError
  });
}

validGrids = validGrids.map(addModelHeights);

// =========================
// VALIDATION METRICS
// =========================

print('==============================================');
print('VALIDATION METRICS');
print('==============================================');

var validCount = validGrids.size();
print('Valid grids:', validCount);

// Calculate correlation
var correlation = validGrids.reduceColumns({
  reducer: ee.Reducer.pearsonsCorrelation(),
  selectors: ['model_mean_height', 'obs_mean_height']
});

print('Pearson Correlation:', correlation.get('correlation'));

// Calculate bias
var meanBias = validGrids.aggregate_mean('height_difference');
print('Mean Bias (Model - Observed):', meanBias, 'm');

// Calculate MAE
var mae = validGrids.map(function(f) {
  var absError = ee.Number(f.get('height_difference')).abs();
  return f.set('abs_error', absError);
}).aggregate_mean('abs_error');

print('Mean Absolute Error:', mae, 'm');

// Calculate RMSE
var rmse = validGrids.map(function(f) {
  var diff = ee.Number(f.get('height_difference'));
  return f.set('sq_error', diff.pow(2));
}).aggregate_mean('sq_error');

rmse = ee.Number(rmse).sqrt();
print('RMSE:', rmse, 'm');

// Mean heights
print('Mean Observed Height:', validGrids.aggregate_mean('obs_mean_height'), 'm');
print('Mean Modeled Height:', validGrids.aggregate_mean('model_mean_height'), 'm');

// =========================
// VISUALIZATION
// =========================

print('==============================================');
print('MAP VISUALIZATION');
print('==============================================');

Map.clear();
Map.centerObject(roi, 11);

// ROI
Map.addLayer(roi, {color: 'yellow', fillColor: '00000000'}, 'Region of Interest', false);

// Observed building heights (raw pixels)
Map.addLayer(maskedHeight, {
  min: 3, max: 30,
  palette: ['0000ff', '00ffff', 'ffff00', 'ff0000']
}, 'Building Height (30m pixels)', true, 0.7);

// Valid grid boundaries
Map.addLayer(validGrids.style({color: 'green', fillColor: '00000000', width: 1}), 
  {}, 'Valid Grid Cells', false);

// Observed mean height per grid
var obsImage = validGrids.reduceToImage({
  properties: ['obs_mean_height'],
  reducer: ee.Reducer.first()
});

Map.addLayer(obsImage, {
  min: 5, max: 25,
  palette: ['blue', 'cyan', 'yellow', 'orange', 'red']
}, 'Observed Mean Height (per grid)', true);

// Modeled mean height per grid
var modelImage = validGrids.reduceToImage({
  properties: ['model_mean_height'],
  reducer: ee.Reducer.first()
});

Map.addLayer(modelImage, {
  min: 5, max: 25,
  palette: ['blue', 'cyan', 'yellow', 'orange', 'red']
}, 'Modeled Mean Height (per grid)', true);

// Difference map
var diffImage = validGrids.reduceToImage({
  properties: ['height_difference'],
  reducer: ee.Reducer.first()
});

Map.addLayer(diffImage, {
  min: -10, max: 10,
  palette: ['0000ff', '4444ff', 'ffffff', 'ff4444', 'ff0000']
}, 'Difference (Model - Observed)', true);

// =========================
// CHARTS
// =========================

print('==============================================');
print('CHARTS');
print('==============================================');

// Check if we have valid data before charting
validCount.evaluate(function(count) {
  if (count > 0) {
    print('Creating charts for', count, 'valid grids...');
    
    // Scatterplot
    var scatterChart = ui.Chart.feature.byFeature({
      features: validGrids.limit(500),
      xProperty: 'obs_mean_height',
      yProperties: ['model_mean_height']
    })
    .setSeriesNames(['Modeled'])
    .setChartType('ScatterChart')
    .setOptions({
      title: 'Modeled vs Observed Mean Building Height',
      hAxis: {
        title: 'Observed Height (m)',
        minValue: 0,
        maxValue: 30,
        gridlines: {count: 7}
      },
      vAxis: {
        title: 'Modeled Height (m)',
        minValue: 0,
        maxValue: 30,
        gridlines: {count: 7}
      },
      pointSize: 4,
      colors: ['#1f77b4'],
      trendlines: {
        0: {
          type: 'linear',
          showR2: true,
          visibleInLegend: true,
          color: 'red',
          lineWidth: 2
        }
      },
      legend: {position: 'right'}
    });
    
    print(scatterChart);
    
    // Histogram of differences
    var diffHist = ui.Chart.feature.histogram({
      features: validGrids,
      property: 'height_difference',
      minBucketWidth: 1,
      maxBuckets: 30
    })
    .setOptions({
      title: 'Distribution of Height Differences (Model - Observed)',
      hAxis: {title: 'Difference (m)'},
      vAxis: {title: 'Frequency'},
      colors: ['#d62728'],
      bar: {gap: 0}
    });
    
    print(diffHist);
    
    // Observed height distribution
    var obsHist = ui.Chart.feature.histogram({
      features: validGrids,
      property: 'obs_mean_height',
      minBucketWidth: 2,
      maxBuckets: 20
    })
    .setOptions({
      title: 'Distribution of Observed Mean Heights',
      hAxis: {title: 'Height (m)'},
      vAxis: {title: 'Frequency'},
      colors: ['#2ca02c']
    });
    
    print(obsHist);
    
  } else {
    print('❌ No valid grids found!');
    print('Suggestions:');
    print('  • Lower presenceThreshold (currently:', CONFIG.openBuildings.presenceThreshold + ')');
    print('  • Lower fractionalCountThreshold (currently:', CONFIG.openBuildings.fractionalCountThreshold + ')');
    print('  • Increase sample size');
    print('  • Try different region');
  }
});

// =========================
// EXPORT
// =========================

if (CONFIG.export.enabled) {
  print('==============================================');
  print('EXPORT');
  print('==============================================');
  
  Export.table.toDrive({
    collection: validGrids,
    description: CONFIG.export.description,
    folder: 'GEE_Exports',
    fileFormat: 'CSV'
  });
  
  print('✓ Export task created');
  print('→ Go to Tasks tab and click RUN');
}

// =========================
// INTERACTIVE INSPECTOR
// =========================

var inspector = ui.Panel({
  style: {
    width: '400px',
    position: 'bottom-right',
    padding: '8px'
  }
});

var title = ui.Label({
  value: '🔍 Grid Cell Inspector',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 10px 0'}
});

inspector.add(title);
inspector.add(ui.Label('Click any grid cell on the map'));

Map.add(inspector);

Map.onClick(function(coords) {
  inspector.clear();
  inspector.add(title);
  inspector.add(ui.Label('Loading...', {color: '#888'}));
  
  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var clicked = validGrids.filterBounds(point).first();
  
  clicked.evaluate(function(feature) {
    inspector.clear();
    inspector.add(title);
    
    if (!feature) {
      inspector.add(ui.Label('No valid grid found here', {color: '#ff0000'}));
      return;
    }
    
    var p = feature.properties;
    
    // Grid ID
    if (p.GRID_ID) {
      inspector.add(ui.Label('Grid: ' + p.GRID_ID, {
        fontSize: '12px',
        color: '#666',
        margin: '0 0 8px 0'
      }));
    }
    
    // Observed
    inspector.add(ui.Label('📊 OBSERVED', {
      fontWeight: 'bold',
      color: '#0066cc',
      margin: '10px 0 5px 0'
    }));
    inspector.add(ui.Label('  Mean: ' + p.obs_mean_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  Median: ' + p.obs_median_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  Range: ' + p.obs_min_height.toFixed(1) + 
      ' - ' + p.obs_max_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  Std Dev: ' + p.obs_std_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  Building Pixels: ' + p.obs_pixel_count));
    
    // Modeled
    inspector.add(ui.Label('🏗️ MODELED', {
      fontWeight: 'bold',
      color: '#cc6600',
      margin: '10px 0 5px 0'
    }));
    inspector.add(ui.Label('  RES: ' + p.model_res_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  COM: ' + p.model_com_height.toFixed(1) + ' m'));
    inspector.add(ui.Label('  IND: ' + p.model_ind_height.toFixed(1) + ' m'));
    
    // Comparison
    inspector.add(ui.Label('📐 COMPARISON', {
      fontWeight: 'bold',
      color: '#009900',
      margin: '10px 0 5px 0'
    }));
    
    var diff = p.height_difference;
    var diffColor = diff > 0 ? '#cc0000' : '#0000cc';
    var diffLabel = diff > 0 ? '↑ Overestimate' : '↓ Underestimate';
    
    inspector.add(ui.Label('  Difference: ' + diff.toFixed(1) + ' m', {
      color: diffColor,
      fontWeight: 'bold'
    }));
    inspector.add(ui.Label('  ' + diffLabel));
    inspector.add(ui.Label('  Relative Error: ' + p.relative_error_pct.toFixed(1) + '%'));
    
    // Storey distribution
    inspector.add(ui.Label('📏 RES STOREY DIST.', {
      fontWeight: 'bold',
      color: '#666',
      margin: '10px 0 5px 0',
      fontSize: '11px'
    }));
    
    var storeys = [
      ['1', p.RES_Storey_1_perc],
      ['2', p.RES_Storey_2_perc],
      ['3', p.RES_Storey_3_perc],
      ['4-5', p.RES_Storey_4_5_perc],
      ['6-8', p.RES_Storey_6_8_perc],
      ['9-20', p.RES_Storey_9_20_perc],
      ['20+', p.RES_Storey_20_perc]
    ];
    
    storeys.forEach(function(s) {
      if (s[1] && s[1] > 0) {
        inspector.add(ui.Label('  ' + s[0] + ' storeys: ' + s[1].toFixed(1) + '%', {
          fontSize: '11px'
        }));
      }
    });
  });
});

// =========================
// LEGEND
// =========================

var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 12px',
    backgroundColor: 'white'
  }
});

var legendTitle = ui.Label('Building Height (m)', {
  fontWeight: 'bold',
  fontSize: '13px',
  margin: '0 0 6px 0'
});

legend.add(legendTitle);

var colors = ['#0000ff', '#00ffff', '#ffff00', '#ff9900', '#ff0000'];
var labels = ['3-9', '9-15', '15-21', '21-27', '27+'];

colors.forEach(function(color, i) {
  var item = ui.Panel({
    widgets: [
      ui.Label({
        style: {
          backgroundColor: color,
          padding: '8px',
          margin: '0 8px 0 0'
        }
      }),
      ui.Label(labels[i] + ' m', {fontSize: '11px'})
    ],
    layout: ui.Panel.Layout.Flow('horizontal'),
    style: {margin: '2px 0'}
  });
  legend.add(item);
});

Map.add(legend);

// =========================
// COMPLETE
// =========================

print('==============================================');
print('✓ PROCESSING COMPLETE');
print('==============================================');
print('Review:');
print('  • Map layers above');
print('  • Validation metrics');
print('  • Charts and statistics');
print('  • Click grids for details');
print('  • Run export task if needed');
