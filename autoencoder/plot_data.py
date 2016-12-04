import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

def plot(final_array):
    size = round( np.size(final_array) ** (1./3) )
    data_3d = np.reshape(final_array, [size]*3)
    print(data_3d.shape)
    xx1, yy1, zz1 = np.where((data_3d > 0.2)  & (data_3d < 0.99))
    f = plt.figure()
    a = f.add_subplot(111, projection='3d')
    a.scatter(xx1, yy1, zz1, s=10)
    plt.show()

    from IPython import embed
    embed()

if __name__ == '__main__':
    filename = sys.argv[1]
    data_array = np.load(filename)
    plot(data_array[0])