library(raster)
library(tidyverse)


false_negative_costs = function(forecast, observation, L){
  fn = (as.vector(observation)==1 & as.vector(forecast)==0)
  total_loss_area = sum(fn)
  total_loss = total_loss_area * L
  return(total_loss)
}

aggregate_temporally = function(x, fact=2, fun=max, keep_original_layer_count=TRUE){
  total_layers = nlayers(x)
  final_layers = total_layers / fact
  if(final_layers%%1 != 0 ){stop('fact must divide into total layers')}
  
  out = raster::stack()
  #Aggregate each grouping
  for(starting_layer in seq(1,total_layers, fact)){
    all_layers = starting_layer:(starting_layer+fact-1)
    out = raster::addLayer(out, raster::calc(x[[all_layers]], fun=fun))
    if(keep_original_layer_count){
      for(y in 1:(fact-1)){
        out = raster::addLayer(out, raster::calc(x[[all_layers]], fun=fun))
      }
    }
  }

  return(out)  
}

resample_temporally = function(x, y){
  output_layers = nlayers(y)
}

spatial_scales = c(1,2,5)
temporal_scales = c(1,2,3,6)

#The per year per cell costs
treatment_cost = 10
possible_loss_costs = 10 / seq(0.11, 0.99, 0.01)

forecast_rasters = list.files('./results', pattern='*.tif', full.names = TRUE)
observation_rasters = list.files('./data/testing', pattern='*.tif', full.names = TRUE)
forecast_rasters = forecast_rasters[!grepl('xml', forecast_rasters)]
observation_rasters = observation_rasters[!grepl('xml', observation_rasters)]


forecasts = raster::stack(forecast_rasters)
observations = raster::stack(observation_rasters)

#Rows * columns * years
total_cells = prod(dim(forecasts))

#Convert to yes or no for class 10
is_class_10 = function(x){(x==10)*1}
forecasts = calc(forecasts, is_class_10)
observations = calc(observations, is_class_10)

r=data.frame()
for(loss_cost in possible_loss_costs){
  smallest_grain_perfect = (sum(as.vector(observations)) * treatment_cost) / total_cells
  smallest_grain_cost_never = (sum(as.vector(observations)) * loss_cost) / total_cells
  smallest_grain_cost_maximimum = min(treatment_cost, smallest_grain_cost_never)
  
  for(this_spatial_scale in spatial_scales){
    for(this_temporal_scale in temporal_scales){
      if(this_spatial_scale > 1){
        forecasts_upscaled = aggregate(forecasts, fact=this_spatial_scale, fun=max)
        #Resample so it retains the origin dimensions and cell numbers
        forecasts_upscaled = resample(forecasts_upscaled, forecasts, method='ngb')
      } else {
        forecasts_upscaled = forecasts
      }
  
      if(this_temporal_scale > 1){
        forecasts_upscaled = aggregate_temporally(forecasts_upscaled, fact = this_temporal_scale)
      } else {
        forecasts_upscaled = forecasts_upscaled
      }
        
      this_scale_treatment_cost = (sum(as.vector(forecasts_upscaled)) * treatment_cost)
      this_scale_fn_cost = false_negative_costs(forecasts_upscaled, observations, loss_cost)
      this_scale_expense = (this_scale_treatment_cost + this_scale_fn_cost) / total_cells
      
      r = r %>%
        bind_rows(data.frame('a' = treatment_cost / loss_cost,
                             'expense_max' = smallest_grain_cost_maximimum,
                             'expense_perfect' = smallest_grain_perfect, 
                             'expense_forecast' = this_scale_expense,
                             'spatial_scale' = this_spatial_scale,
                             'temporal_scale' = this_temporal_scale))
    }
  }
}

r$value = with(r, (expense_max - expense_forecast) / (expense_max - expense_perfect))
r$this_scale = with(r, paste0('space_',spatial_scale,'_time_',temporal_scale))

ggplot(r, aes(x=a, y=value, color=as.factor(this_scale), group=as.factor(this_scale))) + 
  geom_line() +
  ylim(0,1) 
  facet_grid(~temporal_scale)
