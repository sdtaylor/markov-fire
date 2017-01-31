#Take the original modis landcover rasters for each year and crop them to
#the training and testng areas
library(raster)
library(rgdal)
library(sp)

raw_modis_dir='~/data/markov_fire/modis/'

file_list=list.files(raw_modis_dir, pattern='*tif', full.names = TRUE)

file_list=file_list[!grepl('xml', file_list)]
file_list=file_list[!grepl('ovr', file_list)]


study_area=readOGR('/home/shawn/projects/markov-fire/gis/', 'study_area')

#Crop training years
for(this_year in 2001:2004){
  file_path=grep(paste0('A',this_year), file_list, value=TRUE)
  whole_raster=raster::raster(file_path)
  cropped_raster=crop(whole_raster, study_area)

  filename=paste0(this_year,'.tif')
  writeRaster(cropped_raster, paste0('./data/training/',filename))
}

#Crop testing years
for(this_year in 2005:2013){
  file_path=grep(paste0('A',this_year), file_list, value=TRUE)
  whole_raster=raster::raster(file_path)
  cropped_raster=crop(whole_raster, study_area)
  
  filename=paste0(this_year,'.tif')
  writeRaster(cropped_raster, paste0('./data/testing/',filename))
}