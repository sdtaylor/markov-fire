import gdalnumeric
import pandas as pd
import numpy as np
import gdal
from sklearn.linear_model import LogisticRegression


#Read in a list of rasters and stack them into a single array. (rows x column x numRasters)
def stackImages(fileList):
    fullYear=gdalnumeric.LoadFile(fileList[0]).astype(np.int)
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)

#Given some 2d image of catagorical data. compute the composition. essentially percent cover
#a = 2d array, catagories=list of possible catagories
def get_composition(a, catagories, timesteps=None):
    composition=[]
    for c in catagories:
        composition.append(np.sum(a==c))
    composition=np.array(composition)/np.product(a.shape)
    return composition

#Predictor values are the current state of a pixel x,y + the state of surrounding pixels n
#image: 2d x y array
#x,y: location of the focal pixel
#n  : number of surrounding pixels to consider
def extract_predictor_values(image, row, col, n):
    all_pixel_data={}
    all_pixel_data['t0']=image[row,col]

    if n==8:
        surrounding=image[row-1:row+2 , col-1:col+2].reshape((9))
        #Delete the focal pixel that is in this 3x3 array
        surrounding=np.delete(surrounding, 4)
    elif n==24:
        surrounding=image[row-2:row+3 , col-2:col+3].reshape((25))
        surrounding=np.delete(surrounding, 12)

    #Convert surrounding pixel values to percent of each class
    surrounding_size=len(surrounding)
    cover_catagories=list(range(1,15))
    for catagory in range(1, len(cover_catagories)+1):
        all_pixel_data['surrounding_cat'+str(catagory)]= np.sum(surrounding==catagory) / surrounding_size

    return(all_pixel_data)

#Extract and organize a timeseries of arrays for model fitting
def array_to_model_input_fitting(a):
    array_data=[]
    for row in range(1, a.shape[0]-1):
        for col in range(1, a.shape[1]-1):
            for time in range(0, a.shape[2]-1):
                array_data.append(extract_predictor_values(a[:,:,time], row, col, 8))
                array_data[-1]['t1'] = a[row,col,time+1]

    array_data=pd.DataFrame(array_data)
    return array_data

#Extract and organize data for model predictions
#Same as above function but doesn't get t+1, and
#accepts just a single timestep at a time
def array_to_model_input(a):
    array_data=[]
    for row in range(1, a.shape[0]-1):
        for col in range(1, a.shape[1]-1):
            array_data.append(extract_predictor_values(a, row, col, 8))

    array_data = pd.DataFrame(array_data)
    return array_data

#The model outpus a 1d array of pixel predictions. This converts it back to that
#and adds on the padded layer that is subtracted in array_to_model_input.
#The value of the padded layer is 1, the most common class (evergreen needleleaft)
def model_output_to_array(output, desired_shape, pad_value=1):
    reshaped = np.zeros(desired_shape)
    reshaped[:] = pad_value
    reshaped[1:-1,1:-1] = output.reshape((desired_shape[0]-2, desired_shape[1]-2))
    return reshaped

#Take 2d array and a list of spatial sizes.
#Return a dict of indexes to use when slicing out the different spatial scales
#leaves out the edges when the spatial_scale does not line up perfectly with the array
def create_spatial_scale_indexes(image_shape, sp_scales):
    indexes={}
    nrow, ncol = image_shape
    for this_scale in sp_scales:
        i={}
        max_squares_col=int(np.floor(ncol/this_scale))
        max_squares_row   =int(np.floor(nrow/this_scale))
        #The endpoints for all squares for this spatial scale
        i['col_breaks']=(np.arange(1, max_squares_col+1) * this_scale).tolist()
        i['row_breaks']=(np.arange(1, max_squares_row+1) * this_scale).tolist()
        indexes[this_scale]=i
    return indexes

#stop points in a 0:num_timesteps array for all temporal scales
def create_temporal_scale_indexes(num_timesteps, t_scales):
    indexes={}
    for this_scale in t_scales:
        i={}
        max_years = int(np.floor(num_timesteps / this_scale))
        i['time_breaks']=(np.arange(1, max_years+1) * this_scale).tolist()
        indexes[this_scale]=i
    return indexes


#Takes catagory x year composition and compresses it using the temporal scale. 
#ie. 20 years of data with scale=2 becomes 10 data points
def apply_temporal_scale(composition, temporal_scale):
    if temporal_scale==1:
        return(composition)

    new_num_cols=int(np.floor(composition.shape[1] / temporal_scale))
    new_matrix=np.empty((composition.shape[0], new_num_cols))

    for i in range(new_num_cols):
        if i==0:
            cols_to_include=list(range(0, temporal_scale))
        else:
            cols_to_include=list(range(i*temporal_scale, (i+1)*temporal_scale))
        new_matrix[:,i]=np.mean(composition[:,cols_to_include], 1)

    return(new_matrix)

#Various evaluation metrics

#mean squared error
def mse(observed, predicted):
    return(np.mean((observed-predicted)**2))

#Euclidean distance
def eucl_dist(observed, predicted):
    return(np.sqrt(  np.sum( (observed-predicted)**2) ) )

#R^2 of the 1:1 line
#log transform because Marks & Muller-Landau 2007 10.1126/science.1140190 
def r2(obs, pred):
    obs=np.log(obs+0.001)
    pred=np.log(pred+0.001)
    return(1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2))

def evaluate(obs, pred):
    assert pred.shape == obs.shape, 'Observed and predicted shapes must match'

    all_timestep_results=[]
    #Each column is a timestep
    for this_col in range(pred.shape[1]):
       results_this_timestep={}
       results_this_timestep['mse']=mse(obs[:,this_col], pred[:,this_col])
       results_this_timestep['eucl_dist']=eucl_dist(obs[:,this_col], pred[:,this_col])
       results_this_timestep['r2']=r2(obs[:,this_col], pred[:,this_col])
       results_this_timestep['timestep']=this_col+1
       all_timestep_results.append(results_this_timestep)

    return(all_timestep_results)

###################################################################
###################################################################
test_data_dir='./data/testing/'
train_data_dir='./data/training/'


#Spatial scales are original 500m grid cells to a side. 
spatial_scales=[2,5,10,20,40,80]
temporal_scales=[1,2,3,4,5]

#MCD12Q1 type 1 is 14 cover classes
catagories=range(1,15)

##########################################################
#Build the training model

training_file_list = [train_data_dir+str(year)+'.tif' for year in range(2001,2005)]

training_timeseries = stackImages(training_file_list)
training_data = array_to_model_input_fitting(training_timeseries)

model = LogisticRegression()
model.fit(training_data.drop('t1',1), training_data['t1'])

############################################################
#Make predictions of the testinig data using only the initial year as input
#Also accumulate observations of the rest of the data.

testing_file_list=[test_data_dir+str(year)+'.tif' for year in range(2005,2014)]
all_training_years=stackImages(testing_file_list)
#1st year will be used as the initial values
initial_year=all_training_years[:,:,0]
#2nd year onwards will be validation
observations=all_training_years[:,:,1:]
#A matching array to hold  predictions
predictions = np.zeros_like(observations)

#extents in the testing timeseries
num_years=observations.shape[2]
area_shape=initial_year.shape

this_year_prediction=initial_year.copy()
for year in range(num_years-1):
    #this_year_prediction = model.predict(array_to_model_input(this_year_prediction)).reshape(area_shape)
    this_year_prediction = model.predict(array_to_model_input(this_year_prediction))
    this_year_prediction = model_output_to_array(this_year_prediction, desired_shape=area_shape)
    predictions[:,:,year] = this_year_prediction

################################################################
#Upscale the observations and predictions and score the metrics

all_results=[]
#This iterates over the spatial scales with info about them defined in the function.
for this_spatial_scale, sp_info in create_spatial_scale_indexes(area_shape, spatial_scales).items():
    for this_temporal_scale, time_info in create_temporal_scale_indexes(num_years, temporal_scales).items():
        spatial_replicate=0
        #Num replicates = number of squares of size spatial_scale x spatial_scale that can fit into the raster
        for row_end in sp_info['row_breaks']:
            for col_end in sp_info['col_breaks']:
                temporal_replicate=0
                for time_end in time_info['time_breaks']:
                    row_start = row_end-this_spatial_scale
                    col_start = col_end-this_spatial_scale
                    time_start = time_end-this_temporal_scale
                    #Send the end point to the last index in the array if at the
                    #end of the timeseries
                    #time_end = time_start if time_end == num_years else time_end

                    #The observed and predicted values of this spatiotemporal cell
                    replicate_observations=observations[row_start:row_end, col_start:col_end, time_start:time_end]
                    replicate_predictions =predictions[row_start:row_end, col_start:col_end, time_start:time_end]

                    #Convert to percent cover of catagories and score
                    replicate_observations=get_composition(replicate_observations, catagories)
                    replicate_predictions=get_composition(replicate_predictions, catagories)

                    i={}
                    i['temporal_scale'] = this_temporal_scale
                    i['spatial_scale'] = this_spatial_scale
                    i['temporal_replicate'] = temporal_replicate
                    i['spatial_replicate'] = spatial_replicate
                    i['sum_square'] = np.sum((replicate_observations - replicate_predictions)**2)

                    all_results.append(i)
                    temporal_replicate+=1
                spatial_replicate+=1

all_results=pd.DataFrame(all_results)
all_results.to_csv('./results/landscape_fire_model.csv', index=False)
