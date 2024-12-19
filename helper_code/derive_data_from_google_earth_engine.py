// Function to mask clouds and cirrus in Sentinel-2 imagery
function maskS2clouds(images) {
  // Select the QA60 band that contains cloud and cirrus information
  var qa = images.select('QA60');
  
  // Bitmask for identifying clouds and cirrus
  var cloudBitMask = 1 << 10; // Cloud mask (bit 10)
  var cirrusBitMask = 1 << 11; // Cirrus mask (bit 11)
  
  // Create a mask where neither clouds nor cirrus are present
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0) // No clouds
              .and(qa.bitwiseAnd(cirrusBitMask).eq(0)); // No cirrus
  
  // Apply the mask, scale image values, and retain properties
  return images.updateMask(mask) // Apply cloud/cirrus mask
               .divide(10000) // Scale pixel values
               .copyProperties(images) // Copy image properties
               .set('system:time_start', images.get('system:time_start')); // Retain time property
}

// Filter Sentinel-2 image collection based on criteria
var images_filtered = sentinel_h
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)) // Filter images with less than 20% cloud cover
  .filterDate('2017-03-28', '2021-07-28') // Filter images within the specified date range
  .filter(ee.Filter.calendarRange(4, 6, 'month')) // Filter images for April to June
  .filterBounds(geometry) // Filter images within the specified geometry
  .map(maskS2clouds); // Apply cloud masking function

print(images_filtered); // Output the filtered image collection

// Function to calculate NDVI for an image
var ndvi = function(image) {
  var ndv = image.normalizedDifference(['B8', 'B4']); // Calculate NDVI using near-infrared (B8) and red (B4) bands
  return ndv.copyProperties(image, ['system:index', 'system:time_start']); // Retain original image properties
};

// Apply NDVI calculation to the filtered image collection
var ndvi = images_filtered.map(ndvi);

print(ndvi); // Output the NDVI image collection

// Clip the first image from the filtered collection to the specified geometry
var img_filtered_clip = images_filtered.first().clip(geometry);
var nd = ndvi.first().clip(geometry);

// Display the filtered and clipped false-color composite image
Map.addLayer(img_filtered_clip, {bands: ['B8', 'B4', 'B2'], min: 0, max: 4000}, 'FalseColor_clip');

// Display the NDVI image with a specified color palette
Map.addLayer(nd, {min: 0, max: 1, palette: ['Orange', 'Green']}, 'NDVI');

// Center the map view on the specified geometry
Map.centerObject(geometry);

// Create a time series chart of NDVI by region
var chart = ui.Chart.image.seriesByRegion({
  imageCollection: ndvi, // Use the NDVI image collection
  regions: geometry, // Regions for analysis
  reducer: ee.Reducer.mean(), // Mean reducer for aggregation
  scale: 10, // Spatial resolution of 10 meters
  seriesProperty: 'class' // Property for series grouping
});

print(chart); // Output the chart
