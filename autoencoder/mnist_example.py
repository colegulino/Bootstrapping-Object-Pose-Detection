# Exterior packages
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

# Internal packages
import mnist_example_model
import mnist_data_utils

def main():
	# Get data
	data = mnist_data_utils.MNIST()
	train_data = data.getTrainingData()
	test_data = data.getTestData()

	# Set up parameters
	learning_rate = 0.01
	training_epochs = 20
	batch_size = 256
	display_step = 1
	examples_to_show = 10

	# Set up autoencoder
	ae = mnist_example_model.autoencoderTest([784, 256, 128], learning_rate)

	# Initialize all variables
	init = tf.initialize_all_variables()

	with tf.Session() as sess:
		sess.run(init)

		total_batch = int(train_data.num_examples/batch_size)

		for epoch in range(training_epochs):
			# Loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = train_data.next_batch(batch_size)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([ae.optimizer, ae.cost], feed_dict={ae.X: batch_xs})
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

		print("Optimization Finished!")

		# Applying encode and decode over test set
		encode_decode = sess.run(
			ae.y_pred, feed_dict={ae.X: test_data.images[:examples_to_show]})
		# Compare original images with their reconstructions
		f, a = plt.subplots(2, 10, figsize=(10, 2))
		for i in range(examples_to_show):
			a[0][i].imshow(np.reshape(test_data.images[i], (28, 28)))
			a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
		f.show()
		plt.draw()
		plt.waitforbuttonpress()

if __name__ == '__main__':
	main()