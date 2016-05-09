import gdalnumeric
import pandas as pd
import numpy as np
import gdal


testing_folder='./data/testing/'

training_data=pd.read_csv('./data/training.csv')

#print(training_data[training_data['t']==0].groupby('t1').count().reset_index())
#print(np.unique(training_data['t1'][training_data['t']==0].values, return_counts=True))
#exit()

matrix=training_data[training_data['t']==0].groupby('t1').count().reset_index()
sum_this_set=matrix['t'].sum()
matrix['t']=matrix['t']/sum_this_set
matrix.columns=['t1','0']
#Build transition matrix from the data
for this_class in range(1, max(np.unique(training_data['t'].values))):
    temp=training_data[training_data['t']==this_class].groupby('t1').count().reset_index()
    temp['t']=temp['t']/temp['t'].sum()
    temp.columns=['t1',str(this_class)]

    matrix=matrix.merge(temp, how='outer',on='t1')

matrix.fillna(0, inplace=True)
print(matrix)
print(matrix.drop('t1', 1).values)
