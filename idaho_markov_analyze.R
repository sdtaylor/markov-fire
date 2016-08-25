library(dplyr)
library(ggplot2)
library(tidyr)

results=read.csv('results.csv')

results=results %>%
  mutate(year_lag=(timestep*temporal_scale) - (temporal_scale/2)) %>%
  mutate(spatial_scale=(250^2)*(spatial_scale^2)/1000^2) #convert 250 m^2 grid cells to square km

#################################################################
to_keep=data.frame(temporal_scale=c(1,2,3,4,5), 
                   year_lag=c(6.5, 7.0, 7.5, 6.0, 7.5),
                   keep='yes')

results_summarized=results %>%
  group_by(spatial_scale, temporal_scale, year_lag, model_type) %>%
  summarise(r2 = mean(r2)) %>%
  left_join(to_keep, by=c('temporal_scale','year_lag')) %>%
  filter(keep=='yes', model_type=='markov') %>%
  select(-keep)


ggplot(results_summarized, aes(x=spatial_scale, y=r2, color=as.factor(temporal_scale), group=as.factor(temporal_scale)))+
  geom_line(size=2)+
  geom_point(size=3) +
  scale_color_brewer(palette = 'Set2') + 
  theme(panel.grid.major = element_line(colour = "gray38"), 
        panel.background = element_rect(fill = "gray95"), 
        legend.position = "right", legend.direction = "vertical") +
  labs(title = "Prediction accuracy \n of Idaho landscape markov model", 
       x = "Spatial Grain Size (km sq)", y = "R^2", 
       colour = " Temporal \nGrain Size\n (years)") 
  
  
ggplot(results_summarized, aes(x=temporal_scale, y=r2, color=as.factor(spatial_scale), group=as.factor(spatial_scale)))+
  geom_line(size=2)+
  geom_point(size=3) +
  scale_color_brewer(palette = 'Set2') + 
  theme(panel.grid.major = element_line(colour = "gray38"), 
        panel.background = element_rect(fill = "gray95"), 
        legend.position = "right", legend.direction = "vertical") +
  labs(title = "Prediction accuracy \n of Idaho landscape markov model", 
       x = "Temporal Grain Size (years)", y = "R^2", 
       colour = " Spatial\n Grain Size\n (km sq)") 