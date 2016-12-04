# Exterior packages
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Internal packages
import vox_example_model

import sys
sys.path.insert(0, '../Data/modelnet')
import modelnet

def main():
    # Get data
    data = modelnet.modelnet()
    train_data = data.get_train()
    test_data = data.get_test()

    # Set up parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 64
    display_step = 1
    examples_to_show = 10

    print('Train Shape', train_data.shape)
    print('Test Shape', test_data.shape)
    # Set up autoencoder
    ae = vox_example_model.autoencoderTest([train_data.shape[1], 1024, 512], learning_rate)

    # Initialize all variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(train_data.num_examples/batch_size)

        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_data.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([ae.optimizer, ae.cost], feed_dict={ae.X: batch_xs})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Applying encode and decode over test set
        encode_decode = sess.run(
            ae.y_pred, feed_dict={ae.X: test_data.data[:examples_to_show]})
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2), subplot_kw={'projection':'3d'})
        for i in range(examples_to_show):
            data_3d = np.reshape(test_data.data[i], data.image_size)
            print(data_3d.shape)
            xx1, yy1, zz1 = np.where(data_3d > 0.5)
            a[0][i].scatter(xx1, yy1, zz1)
            data_3d = np.reshape(encode_decode[i], data.image_size)
            xx2, yy2, zz2 = np.where(data_3d > 0.5)
            a[1][i].scatter(xx2, yy2, zz2)
        f.show()
        plt.show()
        np.save("encode_decode2.npy", encode_decode)
        # plt.waitforbuttonpress()

if __name__ == '__main__':
    main()