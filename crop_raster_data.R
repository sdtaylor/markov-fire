#Take the original modis landcover rasters for each year and crop them to
#the training and testng areas
library(raster)
library(rgdal)
library(sp)

raw_modis_dir='~/data/markov_fire/modis/'

file_list=list.files(raw_modis_dir, pattern='*tif', full.names = TRUE)

file_list=file_list[!grepl('xml', file_list)]
file_list=file_list[!grepl('ovr', file_list)]


training_area=readOGR('/home/shawn/projects/markov-fire/gis/', 'training_area')
testing_area=readOGR('/home/shawn/projects/markov-fire/gis/', 'testing_area')

for(this_year in 2001:2013){
  file_path=grep(paste0('A',this_year), file_list, value=TRUE)
  whole_raster=raster::raster(file_path)
  train=crop(whole_raster, training_area)
  test=crop(whole_raster, testing_area)
  
  filename=paste0(this_year,'.tif')
  writeRaster(train, paste0('./data/training/',filename))
  writeRaster(test, paste0('./data/testing/',filename))
  
  
}