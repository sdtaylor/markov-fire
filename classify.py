import gdalnumeric
import pandas as pd
import numpy as np
import gdal


testing_folder='./data/testing/'

training_data=pd.read_csv('./data/training.csv')

spatial_scales=[1,2,5,10,20,30,40,50]
temporal_scales=[1,2,3,4,5]
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

catagories=t_matrix.columns.values
num_catagories=len(catagories)

#A numpy array to do the actual calculations
t_matrix=t_matrix.values

#Read in a list of rasters and stack them into a single array. (rows x column x numRasters)
def stackImages(fileList):
    fullYear=gdalnumeric.LoadFile(fileList[0])
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)


#print(t_matrix, t_matrix.shape)
#Given some 2d array of catagorical data. compute the composition. essentially percent cover
#a = 2d array, num_c=number of possible catagories
def get_composition(a, num_c):
    hist, edges=np.histogram(a, bins=num_c)
    hist=hist/np.product(a.shape)
    return(hist)

#Run a markov model given a set of initial conditions and timesteps
#return a timeseries of community composition
def run_model(model, initials, timesteps):
    results=np.empty([num_catagories, timesteps])
    current_state=initials.copy()
    results[:,0]=current_state
    for t in range(1,timesteps):
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
        i['step_size']=this_scale
        i['row_start']=0
        i['row_end']=int(np.floor(nrow/this_scale)*this_scale)
        i['col_start']=0
        i['col_end']=int(np.floor(ncol/this_scale)*this_scale)
        i['num_replicates']=int(np.floor(ncol/this_scale) * np.floor(nrow/this_scale)) #How many 'boxes' of this size are in the raster?
        to_return[this_scale]=i
    return(to_return)

#Takes catagory x year composition and compresses it using the temporal scale. 
#ie. 20 years of data with scale=2 becomes 10 data points
def apply_temporal_scale(composition, temporal_scale):
    new_num_cols=int(np.floor(composition.shape[1] / temporal_scale))
    new_matrix=np.empty((composition.shape[0], new_num_cols))

    for i in range(new_num_cols):
        cols_to_include=list(range(i*temporal_scale, (i*temporal_scale)+1))
        new_matrix[:,i]=np.mean(composition[:,cols_to_include], 1)

    return(new_matrix)
###################################################################
###################################################################
test_data_dir='./data/testing/'
file_list=[test_data_dir+str(year)+'.tif' for year in range(2001,2014)]
all_training_years=stackImages(file_list)

#Total years in the timeseries
num_years=13
#1st year will be used as the initial values
year_0=all_training_years[:,:,0].copy()
#2nd year onwards will be validation
validation=all_training_years[:,:,1:]

#Lets do it
for this_temporal_scale in temporal_scales:
    #This iterates over the spatial scales with info about them defined in the function.
    for this_spatial_scale, sp_info in create_spatial_scale_indexes(year_0, spatial_scales).items():
        row_start, row_end, col_start, col_end = sp_info['row_start'], sp_info['row_end'], sp_info['col_start'], sp_info['col_end']
        step=sp_info['step_size']
        #Num replicates = number of squares of size spatial_scale x spatial_scale that can fit into the raster
        for this_replicate in range(sp_info['num_replicates']):
            initial=year_0[row_start:row_end, col_start:col_end].copy()
            initial=get_composition(initial, this_temporal_scale)
            predictions=run_model(t_matrix, get_composition(initial, num_catagories), num_years)
print(apply_temporal_scale(predictions, 4))
