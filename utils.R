
#Calculate the fn cost of two raster data sets
false_negative_costs = function(forecast, observation, L){
  fn = (as.vector(observation)==1 & as.vector(forecast)==0)
  total_loss_area = sum(fn)
  total_loss = total_loss_area * L
  return(total_loss)
}

#Aggregate a raster stack temporally. 
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


#######################################################################


#Various scoring metrics
precision = function(obs, pred){
  correctly_predicted = obs==pred
  tp = sum(correctly_predicted & pred==1)
  fp = sum((!correctly_predicted) & pred==1)
  return( tp / (tp + fp))
}

recall = function(obs, pred){
  correctly_predicted = obs==pred
   tp = sum(correctly_predicted & pred==1)
  fn = sum((!correctly_predicted) & pred==0)
   return( tp / (tp + fn))
}

f1_score = function(obs, pred){
  prec = precision(obs, pred)
  rec = recall(obs, pred)

  return( (2 * prec * rec) / sum(prec, rec))

}
