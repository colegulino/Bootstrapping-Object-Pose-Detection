# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class MNIST():
	def __init__(self):
		self.data = input_data.read_data_sets("/tmp/data/", one_hot=True)

	def getTrainingData(self):
		return self.data.train

	def getTestData(self):
		return self.data.test