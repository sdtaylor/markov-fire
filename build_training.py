import numpy as np
import pandas as pd
import gdalnumeric


#Read in all rasters and stack them into a single array. (rows x column x numRasters)
def stackImages(fileList):
    fullYear=gdalnumeric.LoadFile(fileList[0])
    for thisImage in fileList[1:]:
        image=gdalnumeric.LoadFile(thisImage)
        fullYear=np.dstack((fullYear, image))
    return(fullYear)

#extract an arbitrary pixel (x,y) value at time t, t+1, and it's surrounding n pixel values at time t
def extractValue(stack,x,y,t, n):
    focalPixel_t=stack[x,y,t]
    focalPixel_t1=stack[x,y,t+1]

    if n==0:
        return(focalPixel_t, focalPixel_t1)
    if n==8:
        surrounding=stack[x-1:x+2 , y-1:y+2, t].reshape((9))
        #Delete the focal pixel that is in this 3x3 array
        surrounding=np.delete(surrounding, 4)
    elif n==24:
        surrounding=imageStack[x-2:x+3 , y-2:y+3, t].reshape((25))
        surrounding=np.delete(surrounding, 12)

    return(focalPixel_t, focalPixel_t1, surrounding)

###################################################################
###################################################################
dataDir='./data/training/'

#Build an row x column x year array of all the training data. 
file_list=[dataDir+str(year)+'.tif' for year in range(2001,2014)]
imageStack=stackImages(file_list)

#Extract all values into a format suitable for sklearn. 
count=0
data=[]
for row in range(1, imageStack.shape[0]-1):
    for col in range(1, imageStack.shape[1]-1):
        for time in range(0, imageStack.shape[2]-1):
            dataThisObs={}
            t, t1=extractValue(imageStack, row, col, time, 0)

            dataThisObs['t']=t
            dataThisObs['t1']=t1

            #Save this for later...
            #Process the surrounding pixel data as fraction in each of the tree death number catagories
            #surroundingSize=len(surrounding)
            #for catagory in range(1, len(treeDeathBins)+1):
            #    dataThisObs['Surrounding-Cat'+str(catagory)]= np.sum(surrounding==catagory) / surroundingSize

            data.append(dataThisObs)

data=pd.DataFrame(data).astype(np.int)
data.to_csv('./data/training.csv', index=False)
