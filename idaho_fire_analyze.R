library(raster)
library(tidyverse)
source('utils.R')

spatial_scales = c(1,2,4)
temporal_scales = c(1,2,3,6)

#The per year per cell costs
treatment_cost = 10
possible_loss_costs = 10 / seq(0.01, 0.7, 0.01)


scaled_data_file = 'results/idaho_fire_scaled_results.csv'
cost_loss_value_file = 'results/idaho_fire_cost_loss_results.csv'
#############################################################

forecast_rasters = list.files('./results', pattern='*.tif', full.names = TRUE)
observation_rasters = list.files('./data/testing', pattern='*.tif', full.names = TRUE)
forecast_rasters = forecast_rasters[!grepl('xml', forecast_rasters)]
observation_rasters = observation_rasters[!grepl('xml', observation_rasters)]


forecasts = raster::stack(forecast_rasters)
observations = raster::stack(observation_rasters)

#Total 500x500m cells. Rows * columns * years
total_cells = prod(dim(forecasts))

#Convert to yes or no for class 10
is_class_10 = function(x){(x==10)*1}
#forecasts = raster::calc(forecasts, is_class_10)
observations = raster::calc(observations, is_class_10)


#############################################################################
#Get csv of predictions/observations at all scales
# 
# scaled_data = data.frame()
# for(this_spatial_scale in spatial_scales){
#   for(this_temporal_scale in temporal_scales){
#     if(this_spatial_scale > 1){
#       forecasts_upscaled = raster::aggregate(forecasts, fact=this_spatial_scale, fun=max)
#       #Resample so it retains the original dimensions and cell numbers
#       forecasts_upscaled = raster::resample(forecasts_upscaled, forecasts, method='ngb')
#     } else {
#       forecasts_upscaled = forecasts
#     }
#     
#     if(this_temporal_scale > 1){
#       forecasts_upscaled    = aggregate_temporally(forecasts_upscaled, fact = this_temporal_scale, keep_original_layer_count = FALSE)
#     } 
#     
#     this_scale_data = data.frame(observed = as.vector(observations),
#                                  predicted = as.vector(forecasts_upscaled))
#     
#     this_scale_data$spatial_scale = this_spatial_scale
#     this_scale_data$temporal_scale = this_temporal_scale
#     this_scale_data$cell_id = 1:nrow(this_scale_data)
#     
#     scaled_data = scaled_data %>%
#       bind_rows(this_scale_data)
#     
#   }
# }
# 
# write_csv(scaled_data, scaled_data_file)

###############################################################################
#Calculate cost/loss model curve
cost_loss_values=data.frame()
for(loss_cost in possible_loss_costs){
  # Get a binary prediction map based on this cost/loss ratio
  a = treatment_cost/loss_cost
  forecasts_binary = raster::calc(forecasts, function(x, threshold){(x>=0.9)*1})
  
  smallest_grain_perfect = (sum(as.vector(observations)) * treatment_cost) / total_cells
  smallest_grain_cost_never = (sum(as.vector(observations)) * loss_cost) / total_cells
  smallest_grain_cost_maximimum = min(treatment_cost, smallest_grain_cost_never)
  
  for(this_spatial_scale in spatial_scales){
    for(this_temporal_scale in temporal_scales){
      if(this_spatial_scale > 1){
        forecasts_upscaled = raster::aggregate(forecasts_binary, fact=this_spatial_scale, fun=max)
        #Resample so it retains the original dimensions and cell numbers
        forecasts_upscaled = raster::resample(forecasts_upscaled, forecasts_binary, method='ngb')
      } else {
        forecasts_upscaled = forecasts_binary
      }
  
      if(this_temporal_scale > 1){
        forecasts_upscaled = aggregate_temporally(forecasts_upscaled, fact = this_temporal_scale)
      } else {
        forecasts_upscaled = forecasts_upscaled
      }
        
      this_scale_treatment_cost = (sum(as.vector(forecasts_upscaled)) * treatment_cost)
      this_scale_fn_cost = false_negative_costs(forecasts_upscaled, observations, loss_cost)
      this_scale_expense = (this_scale_treatment_cost + this_scale_fn_cost) / total_cells
      
      cost_loss_values = cost_loss_values %>%
        bind_rows(data.frame('a' = treatment_cost / loss_cost,
                             'expense_max' = smallest_grain_cost_maximimum,
                             'expense_perfect' = smallest_grain_perfect, 
                             'expense_forecast' = this_scale_expense,
                             'spatial_scale' = this_spatial_scale,
                             'temporal_scale' = this_temporal_scale))
    }
  }
}

cost_loss_values$value = with(cost_loss_values, (expense_max - expense_forecast) / (expense_max - expense_perfect))
cost_loss_values$this_scale = with(cost_loss_values, paste0('space_',spatial_scale,'_time_',temporal_scale))

ggplot(cost_loss_values, aes(x=a, y=value, group=this_scale, color=as.factor(temporal_scale))) + 
  ylim(0,1) +
  geom_line(size=1.5, aes(linetype=as.factor(spatial_scale))) +
  scale_linetype_manual(values = c('solid','dashed','dotted')) +
  theme(plot.title = element_text(size = 30),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        legend.text = element_text(size = 15), 
        legend.title = element_text(size = 20),
        strip.text.x=element_text(size=22),
        strip.text.y=element_text(size=22),
        legend.position = "bottom", 
        legend.direction = "horizontal",
        legend.key.width = unit(35, units = 'mm')) +
  labs(title = "Idaho Fire Cost Loss Analysis",
       color = 'Temporal Grain',
       linetype = 'Spatial Grain') 

write_csv(cost_loss_values, cost_loss_value_file)
