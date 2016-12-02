#!/bin/sh

# get model data
wget http://3dshapenets.cs.princeton.edu/ModelNet10.zip
unzip ModelNet10

# Convert to binvox (must be using XQuartz if on mac)
chmod u+x binvox
python3 convertToBinvox.py

# Convert the binvox to a numpy .npz dataset as a dict
# Where:
# data['train'] = training set
# data['text'] = test set
python3 convertBinvoxToNumpy.py