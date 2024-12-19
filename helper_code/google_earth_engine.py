 function maskS2clouds(images) {
  var qa = images.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  	
  return images.updateMask(mask).divide(10000).copyProperties(images).set('system:time_start', images.get('system:time_start'));
}
 
 
//Filter images
var images_filtered = sentinel_h.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))
  .filterDate('2017-03-28', '2021-07-28')
  .filter(ee.Filter.calendarRange(4, 6,'month'))
  .filterBounds(geometry)
  .map(maskS2clouds);
 
print(images_filtered);
 
 
//calculate ndvi
var ndvi = function(image){
 
	var ndv = image.normalizedDifference(['B8','B4']);
	return ndv.copyProperties(image, ['system:index', 'system:time_start']);
	
};
 
var ndvi = images_filtered.map(ndvi);
 
print(ndvi);
 
var img_filtered_clip = images_filtered.first().clip(geometry);
var nd = ndvi.first().clip(geometry);
 
 
//Display data
Map.addLayer(img_filtered_clip,{bands:['B8','B4','B2'],min:0,max:4000},'FalseColor_clip');
Map.addLayer(nd,{min:0,max:1,palette:['Orange','Green']},'NDVI');
Map.centerObject(geometry);
 
 
var chart = ui.Chart.image.seriesByRegion({
  imageCollection:ndvi,
  regions:geometry,
  reducer: ee.Reducer.mean(),
  scale:10,
  seriesProperty:'class'
 
});
 
print(chart);
