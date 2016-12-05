# Exterior packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Internal packages
import sys
sys.path.insert(0, 'Data/modelnet')
import modelnet

# Import modelnet data
data = modelnet.modelnet()
train = data.get_train()
test = data.get_test()
validation = data.get_validation()

def main():
    print("Train Size: {}".format(train.shape))
    print("Test Size: {}".format(test.shape))
    print("Validation Size: {}".format(validation.shape))

if __name__ == '__main__':
    main()