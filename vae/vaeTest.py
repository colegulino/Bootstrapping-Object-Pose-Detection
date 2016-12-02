# Import tensorflow
import tensorflow as tf


class vaeTest():
	# 
	# Construction of variational autoencoder model
	# 
	# @param model_dimensions List of model dimensions for the layers going from bottom to top
	# @param learning_rate Learning rate of the optimizer
	# 
	def __init__(self, model_dimensions, learning_rate):
		self.model_dimensions = model_dimensions
		self.learning_rate = learning_rate
