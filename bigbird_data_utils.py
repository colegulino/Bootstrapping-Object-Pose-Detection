# 
# Data utility for working with the Berkeley BigBird dataset
# 

from PIL import Image
import numpy as np
import random
import copy
import pickle
from scipy.ndimage.filters import gaussian_filter

# 
# Function to import an image from the Berkeley BigBird Dataset
# 
# @param filepath Filepath of the image
# @return A Dictionary of the images
# 
def load_bigbird(filepath):
    image_dict = dict()

    for azim in range(1, 6):
        for angle in range(0, 360, 3):
            im = Image.open(filepath+'NP{}_{}_mask.pbm'.format(azim, angle)).convert('L')

            image_dict[(azim, angle)] = np.array(im)

    return image_dict

# 
# Function to build the triplets
# 
# @param image_dict Dictionary of the images
# @return List of triplets
# 
def build_triplets(image_dict):
    triplets = []
    d = copy.deepcopy(image_dict)

    count = 0
    while len(d) != 0:
        # Get a random element of the dictionary
        index = random.choice(list(d.keys()))
        im = d[index]
        del d[index]

        close_image_key = min(d.keys(), key=lambda x : np.linalg.norm(im-d[x]))
        close_image = d[close_image_key]
        del d[close_image_key]

        far_image_key = max(d.keys(), key=lambda x : np.linalg.norm(im-d[x]))
        far_image = d[far_image_key]
        del d[far_image_key]

        count = count + 1
        print("Adding: {} | Number added: {}".format((index, close_image_key, far_image_key), count))
        triplets.append((im, close_image, far_image))

    return triplets

# 
# Function to build the doubles
# This function takes some images and then makes a pair with the same image
# but adds some gaussian noise to enforce that images with the same pose
# have similar descriptors despite changes in illumination
# 
# @param image_dict Dictionary of the images
# @param sigma Sigma for the gaussian filter
# @return List of doubles
# 
def build_doubles(image_dict, sigma=7):
    return [(val, gaussian_filter(val, sigma=sigma)) for key, val in image_dict.items()]

if __name__ == '__main__':

    file_path = '/home/cole/Downloads/detergent/masks/'

    # triplets = build_triplets(load_bigbird(file_path))
    doubles = build_doubles(load_bigbird(file_path))

    with open('doubles_detergent.p', 'wb') as f:
        dill.dump(outpath,f)
        dill.load(f)

    # with open('triplets_detergent.p', 'wb') as f:
    #     pickle.dump(triplets, f)
