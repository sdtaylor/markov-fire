library(dplyr)
library(ggplot2)
library(tidyr)

results=read.csv('results.csv')

results=results %>%
  mutate(year_lag=(timestep*temporal_scale) - (temporal_scale/2)) %>%
  mutate(spatial_scale=(250^2)*(spatial_scale^2)/1000^2) #convert 250 m^2 grid cells to square km


results=results %>%
  group_by(spatial_scale, temporal_scale, timestep, year_lag, model_type) %>%
  summarize(r2=mean(r2), mse=mean(mse), eucl_dist=mean(eucl_dist)) %>%
  ungroup()

ggplot(results, aes(x=year_lag, y=r2, colour=model_type, group=model_type)) + geom_point()+ geom_line()+
  geom_hline(yintercept=0.9) +
  xlab('Years into future') + ylab('R^2') +
  theme_bw() +
  facet_grid(temporal_scale~spatial_scale, labeller=label_both)

################################################################33
#Skill score by incorporating long term average. 
results=read.csv('results.csv') %>%
  mutate(year_lag=(timestep*temporal_scale) - (temporal_scale/2)) %>%
  mutate(spatial_scale=(250^2)*(spatial_scale^2)/1000^2) %>% #convert 250 m^2 grid cells to square km
  select(spatial_scale, temporal_scale, timestep, year_lag, model_type, replicate, r2) %>%
  spread(model_type, r2) %>%
  mutate(skill_score = (markov-naive) / (1-naive)) %>%
  mutate(skill_score = ifelse(skill_score<(-1), -10, skill_score)) %>%
  group_by(spatial_scale, temporal_scale, timestep, year_lag) %>%
  summarize(skill_score=mean(skill_score)) %>%
  ungroup() 

ggplot(results, aes(x=year_lag, y=skill_score)) + geom_point()+ geom_line()+
  xlab('Years into future') + ylab('Skill') +
  theme_bw() +
  facet_grid(temporal_scale~spatial_scale, labeller=label_both) +
  geom_hline(yintercept=0) 

#################################################################
to_keep=data.frame(temporal_scale=c(1,2,3,4,5), 
                   year_lag=c(6.5, 7.0, 7.5, 6.0, 7.5),
                   keep='yes')

results=results %>%
  left_join(to_keep, by=c('temporal_scale','year_lag')) %>%
  filter(keep=='yes') %>%
  select(-keep)


ggplot(results, aes(as.factor(spatial_scale), as.factor(temporal_scale), fill=r2, label=round(r2,2))) + 
  geom_raster() +
  scale_fill_gradient(low='grey100', high='grey40') + geom_text() +
  xlab('Spatial scale (# km^2)') + ylab('Temporal scale (years)') +
  theme_bw() +
  theme(legend.position='none')