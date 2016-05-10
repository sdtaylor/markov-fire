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
for this_class in range(1, max(np.unique(training_data['t'].values))):
    temp=training_data[training_data['t']==this_class].groupby('t1').count().reset_index()
    temp['t']=temp['t']/temp['t'].sum()
    temp.columns=['t1',str(this_class)]

    t_matrix=t_matrix.merge(temp, how='outer',on='t1')

t_matrix.fillna(0, inplace=True)
num_catagories=t_matrix.shape[1]

#print(t_matrix, t_matrix.shape)
#Given some 2d array of catagorical data. compute the composition. essentially percent cover
#a = 2d array, c=1d array of possible catagories
def get_composition(a, c):
    hist, edges=np.histogram(a, bins=len(c))
    hist=hist/np.product(x.shape)
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
    to_return=[]
    nrow=template_raster.shape[0]
    ncol=template_raster.shape[1]
    for this_scale in sp_scales:
        i={}
        i['spatial_scale']=this_scale
        i['row_start']=0
        i['row_end']=int(np.floor(nrow/this_scale)*this_scale)
        i['col_start']=0
        i['col_end']=int(np.floor(ncol/this_scale)*this_scale)
        i['num_replicates']=int(np.floor(ncol/this_scale) * np.floor(nrow/this_scale)) #How many 'boxes' of this size are in the raster?
        to_return.append(i)
    return(to_return)

