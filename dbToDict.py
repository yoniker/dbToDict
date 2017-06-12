import os
import numpy as np
from scipy.misc import imread
import pickle
"""
###################
#12-Jun-2017


This module will convert any Database to a python dictionary and pickle it.
"Input"/Assumption:
We are in a folder which has subfolders named after the actual classes.
Inside those subfolders we should have the actual images for those classes.
Another assumption: the extension of all image files in those sub-directories is jpg.


Output(files):

dataDict.pkl:
a pickled dictionary, where the keys are:
'data'- ndarray containing all the data. d['data'][i] is the ith example/image
'labels'-ndarray containing a one hot encoded representation of the labels for the corresponding examples. labels[i] is the correct label for the ith example.

encoding2classes.txt:
the conversion between the names of the classes and their encoding.


###################
"""



IMG_EXTENSION='.JPG'
ENCODING_TO_CLASSES_FILE_NAME='encoding2classes.txt'
DATA_KEY='data'
LABELS_KEY='labels'
PICKLED_DATA_FILE_NAME='dataDict.pkl'
def add_data_dir_to_label(dataDict,directory,oneHotEncoding):
	file_names=os.listdir(directory)
	label_as_row=oneHotEncoding.reshape(1,*oneHotEncoding.shape)
	for file_name in file_names:
		if len(file_name)>len(IMG_EXTENSION) and str.upper(file_name[-len(IMG_EXTENSION):])==IMG_EXTENSION:
			fname_with_path=os.path.join(directory,file_name)
			img=imread(fname_with_path)
			img=img.reshape(1,*img.shape) #So that it will be just another numpy row in our ndarray.
			if not DATA_KEY in dataDict: 
				dataDict[DATA_KEY]=img
			else:
				dataDict[DATA_KEY]=np.append(dataDict[DATA_KEY],img,axis=0) #TODO this is a potential bottleneck since it creates a new ndarray each time. If it's way too slow,look for a faster way
			if not LABELS_KEY in dataDict:
				dataDict[LABELS_KEY]=label_as_row
			else:
				dataDict[LABELS_KEY]=np.append(dataDict[LABELS_KEY],label_as_row,axis=0)
	return



directories=[]
for fileName in os.listdir():
	if os.path.isdir(fileName):
		directories.append(fileName)

#Now I will output the file with the classes one hot encoding and names
numberOfClasses=len(directories)
dataDict={}
with open(ENCODING_TO_CLASSES_FILE_NAME, 'w') as f:
	classIndex=0
	for className in directories:
		oneHotEncoding=np.zeros(numberOfClasses)
		oneHotEncoding[classIndex]=1
		f.write(className+':'+str(oneHotEncoding)+'\n')
		dataDict[className]=oneHotEncoding
		dataDict[str(oneHotEncoding)]=className
		add_data_dir_to_label(dataDict,className,oneHotEncoding)
		classIndex+=1
with open(PICKLED_DATA_FILE_NAME,'wb') as f:
	pickle.dump(dataDict,f)

#That's it. After pickling, to unpickle all you need to do is:

# with open(PICKLED_DATA_FILE_NAME,'rb') as f:
# 	dataDict=pickle.load(f)