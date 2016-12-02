#
# Script to convert binvox files to numpy arrays
#

#
# Import this helper package
# Thanks! : https://github.com/dimatura/binvox-rw-py
#
import binvox_rw.binvox_rw as binvox_rw

# Numpy
import numpy as np
import pickle

#
# Converts a boolean image into a binary image
#
# @param b Matrix of booleans
# @return A binary image
#
def bool_2_bin(b):
    return b * 1

# Get relevant models
models = ['bathtub', 'chair', 'table', 'toilet', 'monitor']
path = 'ModelNet10/{}/{}/'
fullpath = 'ModelNet10/{}/{}/{}'

# Get all of the filenames in each folder and then run binvox
from os import listdir
from os.path import isfile, join
import subprocess

# Numpy train and test data
data = {}
data['train'] = {}
data['test'] = {}

image_size = None

for t in ['train', 'test']:
    for model_name in models:
        mypath = path.format(model_name, t)
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.binvox')]
        for file in files:
            filepath = fullpath.format(model_name, t, file)
            model_class = file.split('_')[0]
            print(filepath)
            with open(filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
                m_data = bool_2_bin(model.data)
                if image_size == None:
                    image_size = m_data.shape
                new_shape = m_data.shape[0] * m_data.shape[1] * m_data.shape[2]
                data_vec = np.reshape(m_data, (1, new_shape))

                if (model_class not in data[t]): data[t][model_class] = data_vec
                else: data[t][model_class] = np.concatenate((data[t][model_class], data_vec), axis=0)

outfile = 'numpyzip/class_split_data.p'
with open(outfile, 'wb') as f:
    pickle.dump(data, f)

sizepath = 'numpyzip/image_size.p'
with open(sizepath, 'wb') as f:
    pickle.dump(image_size, f)