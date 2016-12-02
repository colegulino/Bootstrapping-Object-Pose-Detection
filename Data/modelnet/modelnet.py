#
# Utilities for loading modelnet dataset
#
import numpy as np
import pickle

class modelnet:
    #
    # Constructor
    #
    # @param datapath Where the numpy data for the training and test data is
    # Default path is creating after running getModels.sh
    #
    def __init__(self, datapath='numpyzip/class_split_data.p'):
        # Open the dataset
        self.datapath = datapath
        self.data = {}

        # Get the image size
        with open('numpyzip/image_size.p', 'rb') as f:
            self.image_size = pickle.load(f)

        # Get the data sets
        self.final_datapath = 'numpyzip/data.npz'
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        try:
            with open(self.final_datapath, 'rb') as f:
                archive = np.load(f)
                self.train_data = archive['arr_0']
                self.validation_data = archive['arr_1']
                self.test_data = archive['arr_2']
        except IOError:
            print("No file {} was found. Reloading the data".format(self.final_datapath))
            self.data = self.load_dataset()
            self.train_data, self.validation_data, self.test_data = self.construct_data()
            # Save
            with open(self.final_datapath, 'wb') as f:
                np.savez(f, self.train_data, self.validation_data, self.test_data)

        # Create objects
        self.train_data = modelnetData(self.train_data, 'training', self.image_size)
        self.validation_data = modelnetData(self.validation_data, 'validation', self.image_size)
        self.test_data = modelnetData(self.test_data, 'test', self.image_size)

    #
    # Load the dataset
    #
    def load_dataset(self):
        try:
            with open(self.datapath, 'rb') as f:
                return pickle.load(f)
        except IOError:
            print("File path expected as {}. Run getModels.sh to get the data".format(self.datapath))

    #
    # Construct the datasets
    #
    def construct_data(self):
        train_data = np.array([])
        test_data = np.array([])
        validation_data = np.array([])

        # Iterate through the various classes of the test set and construct
        # the validation and the training set
        for model_class, data in self.data['train'].items():
            # Get 1/5 of the training data as validation data
            no_valid = int(max(1, data.shape[0] / 5))

            valid_d = data[-no_valid:]
            train_d = data[:-no_valid]

            # Set training and validation
            if (train_data.size == 0): train_data = train_d
            else: train_data = np.concatenate((train_data, train_d), axis=0)
            if (validation_data.size == 0): validation_data = valid_d
            else: validation_data = np.concatenate((validation_data, valid_d), axis=0)

        # Same thing for test set
        for model_class, data in self.data['test'].items():
            if (test_data.size == 0): test_data = data
            else: test_data = np.concatenate((test_data, data), axis=0)

        return train_data, validation_data, test_data

    #
    # Return the training set
    #
    def get_train(self):
        return self.train_data

    #
    # Return the validation set
    #
    def get_validation(self):
        return self.validation_data

    #
    # Return the test set
    #
    def get_test(self):
        return self.test_data

class modelnetData:
    #
    # Constructor
    #
    # @param data Data (numpy array)
    # @param name Typically 'train' or 'test'
    # @param image_size Size of the voxel images
    #
    def __init__(self, data, name, image_size):
        self.data = data
        self.shape = data.shape
        self.name = name

        self.image_size = image_size
        self.num_examples = data.shape[0]

        # Pointer for getting batch size
        self.point = 0
        self.random_indices = None

    #
    # Get shape of the data
    #
    def shape(self):
        return self.data.shape

    #
    # Get name of the dataset
    #
    def name(self):
        return self.name

    #
    # Get image size
    #
    def image_size(self):
        return self.image_size

    #
    # Get data
    #
    def data(self):
        return self.data

    #
    # Get labels (same as data)
    #
    def labels(self):
        return self.data

    #
    # Number of examples
    #
    def num_examples(self):
        return self.num_examples

    #
    # Get images
    #
    # @param no_images Number of images to get
    # @return A dictionary of the images (key = row)
    #
    def get_image(self, no_images):
        images = {}
        for i in range(min(no_images, self.data.shape[0])):
            images[i] = np.reshape(self.data[i], self.image_size)

        return images

    #
    # Get next batch of the data
    #
    # @param batch_size Batch size of the data
    # @return Batch of data
    #
    def next_batch(self, batch_size):
        if self.point == 0: # reshuffle
            self.random_indices = np.random.permutation(self.num_examples)

        begin = self.point
        end = min(self.point + batch_size, self.num_examples)

        batch_ind = self.random_indices[begin:end]
        self.point = (self.point + batch_size)
        if self.point >= self.num_examples: self.point = 0

        return self.data[batch_ind, :]