# Script to convert to binvox

# Get relevant models
models = ['bathtub', 'chair', 'table', 'toilet', 'monitor']
path = 'ModelNet10/{}/{}/'
fullpath = 'ModelNet10/{}/{}/{}'

# Get all of the filenames in each folder and then run binvox
from os import listdir
from os.path import isfile, join
import subprocess

for model in models:
    for t in ['train', 'test']:
        mypath = path.format(model, t)
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for file in files:
            filepath = fullpath.format(model, t, file)
            # ./binvox ModelNet10/bathtub/train/bathtub_0101.off -d 50 -cb
            subprocess.call(['./binvox', filepath, '-d', '50', '-cb'])