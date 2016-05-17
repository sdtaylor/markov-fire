import gdalnumeric
import pandas as pd
import numpy as np
import gdal


testing_folder='./data/testing/'

training_data=pd.read_csv('./data/training.csv')

spatial_scales=[2,5,10,20,40,80]
temporal_scales=[1,2,3,4,5]
#spatial_scales=[40,50]
#temporal_scales=[3,4,5]
#print(training_data[training_data['t']==0].groupby('t1').count().reset_index())
#print(np.unique(training_data['t1'][training_data['t']==0].values, return_counts=True))
#exit()

#Build the markov transition matrix. Columns are are states at t_0, rows are transition
#probabilites to each other state. This feeds directly into matrix multiplication using np.dot()
t_matrix=training_data[training_data['t']==0].groupby('t1').count().reset_index()
sum_this_set=t_matrix['t'].sum()
t_matrix['t']=t_matrix['t']/sum_this_set
t_matrix.columns=['t1','0']
#Build transition matrix from the data
for this_class in range(1, max(np.unique(training_data['t'].values)+1)):
    temp=training_data[training_data['t']==this_class].groupby('t1').count().reset_index()
    temp['t']=temp['t']/temp['t'].sum()
    temp.columns=['t1',str(this_class)]

    t_matrix=t_matrix.merge(temp, how='outer',on='t1')

#Only 1 or 2 occurences of these types.
t_matrix.drop(['2','4','13','15'], 1, inplace=True)

t_matrix.sort_values('t1', 0, inplace=True)
t_matrix.fillna(0, inplace=True)
t_matrix.drop('t1', 1, inplace=True)

catagories=t_matrix.columns.values.astype(np.int)
num_catagories=len(catagories)

#A numpy array to do the actual calculations
t_matrix=t_matrix.values

#Read in a list of rasters and stack them into a single array. (rows x column x numRasters)
def stackImages(fileList):
    fullYear=gdalnumeric.LoadFile(fileList[0]).astype(np.int)
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)


#print(t_matrix, t_matrix.shape)
#Given some 2d array of catagorical data. compute the composition. essentially percent cover
#a = 2d array, num_c=number of possible catagories
def get_composition(a, catagories):
    composition=[]
    for c in catagories:
        composition.append(np.sum(a==c))
    composition=np.array(composition)/np.product(a.shape)
    return(composition)

#Run a markov model given a set of initial conditions and timesteps
#return a timeseries of community composition
def run_model(model, initials, timesteps):
    results=np.empty([num_catagories, timesteps])
    current_state=initials.copy()
    for t in range(timesteps):
        current_state=np.dot(model, current_state)
        results[:,t]=current_state
    return(results)

#Take 2d array and a list of spatial sizes.
#Return a dict of indexes to use when slicing out the different spatial scales
#leaves out the edges when the spatial_scale does not line up perfectly with the array
def create_spatial_scale_indexes(template_raster, sp_scales):
    to_return={}
    nrow=template_raster.shape[0]
    ncol=template_raster.shape[1]
    for this_scale in sp_scales:
        i={}
        max_squares_col=int(np.floor(ncol/this_scale))
        max_squares_row   =int(np.floor(nrow/this_scale))
        #The endpoints for all squares for this spatial scale
        i['col_breaks']=(np.arange(1, max_squares_col+1) * this_scale).tolist()
        i['row_breaks']=(np.arange(1, max_squares_row+1) * this_scale).tolist()
        i['num_replicates']=max_squares_col*max_squares_row #How many 'boxes' of this size are in the raster?
        to_return[this_scale]=i
    return(to_return)

#Takes catagory x year composition and compresses it using the temporal scale. 
#ie. 20 years of data with scale=2 becomes 10 data points
def apply_temporal_scale(composition, temporal_scale):
    if temporal_scale==1:
        return(composition)

    new_num_cols=int(np.floor(composition.shape[1] / temporal_scale))
    new_matrix=np.empty((composition.shape[0], new_num_cols))

    for i in range(new_num_cols):
        cols_to_include=list(range(i*temporal_scale, (i*temporal_scale)+1))
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
def r2(obs, pred):
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
file_list=[test_data_dir+str(year)+'.tif' for year in range(2005,2014)]
all_training_years=stackImages(file_list)

#Total years in the timeseries
num_years=9
#1st year will be used as the initial values
year_0=all_training_years[:,:,0]
#2nd year onwards will be validation
validation=all_training_years[:,:,1:]

all_results=[]
#Lets do it
for this_temporal_scale in temporal_scales:
    #This iterates over the spatial scales with info about them defined in the function.
    for this_spatial_scale, sp_info in create_spatial_scale_indexes(year_0, spatial_scales).items():
        replicate=0
        #Num replicates = number of squares of size spatial_scale x spatial_scale that can fit into the raster
        for row_end in sp_info['row_breaks']:
            for col_end in sp_info['col_breaks']:
                row_start=row_end-this_spatial_scale
                col_start=col_end-this_spatial_scale
                row_start, row_end, col_start, col_end=0,10,0,10

                #Get initial values and run model
                initial=year_0[row_start:row_end, col_start:col_end]
                initial=get_composition(initial, catagories)

                predictions=run_model(t_matrix, initial, num_years-1)

                exit()
                #Save observed values over all years
                obs_all_years=validation[row_start:row_end, col_start:col_end, :]
                obs_all_years=get_composition(obs_all_years, catagories, num_years-1)

                #Temporal averaging of both observed and predicted. 
                predictions=apply_temporal_scale(predictions, this_temporal_scale)
                obs_all_years=apply_temporal_scale(obs_all_years, this_temporal_scale)

                metrics=evaluate(obs_all_years, predictions)

                #print debug info and quit if nan's start popping up in results 
                if np.isnan(metrics[0]['r2']):
                    print(year_0.shape)
                    print(year_0[row_start:row_end, col_start:col_end])
                    print(predictions)
                    print(row_start, row_end, col_start, col_end)
                    exit()
                #Add in scale info for these results
                for  i in metrics:
                    i['temporal_scale']=this_temporal_scale
                    i['spatial_scale']=this_spatial_scale
                    i['replicate']=replicate

                replicate+=1
                all_results.extend(metrics)

all_results=pd.DataFrame(all_results)
all_results.to_csv('results.csv', index=False)
