# Exterior packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Internal packages
import sys
sys.path.insert(0, 'Data/modelnet')
import modelnet

import vae.variationalAutoEncoder as vae

# Import modelnet data
data = modelnet.modelnet()
train = data.get_train()
test = data.get_test()
validation = data.get_validation()
print("Loaded the Data!")

# Set up parameters
learning_rate = 0.0001
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
debug = False
load_params = True

# Setup the architecture
architecture = \
{
    'no_inputs' : data.input_dim,
    'no_hidden_units' : 2,
    'hidden_dims' : \
    {
        'h1' : 500,
        'h2' : 500,
        'h3' : 500
    },
    'no_latent_dims' : 50
}

def main():
    print("Train Size: {}".format(train.shape))
    print("Test Size: {}".format(test.shape))
    print("Validation Size: {}".format(validation.shape))

    print("Initialized the Variables")
    print("Begin training")

    with tf.Session() as sess:
        # Set up autoencoder
        myVAE = vae.variationalAutoEncoder(architecture, learning_rate, tf.nn.elu, sess, load_params=load_params)
        print("Setup the Model")

        # Initialize all variables
        init = tf.initialize_all_variables()

        sess.run(init)

        total_batch = int(train.num_examples/batch_size)

        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train.next_batch(batch_size)
                _, c = sess.run([myVAE.optimizer, myVAE.cost], feed_dict={myVAE.X: batch_xs})

                if debug:
                    #=======DEBUG=============================================================================
                    for i in batch_xs:
                        print(i)
                    print(batch_xs)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    x_tilde, l1, l2 = sess.run([myVAE.x_tild, myVAE.l1, myVAE.l2], feed_dict={myVAE.X:batch_xs})
                    z_mean, z_sig = sess.run([myVAE.z_mean, myVAE.z_sigma], feed_dict={myVAE.X:batch_xs})
                    print("Z_mean: {}".format(z_mean))
                    print("Z_sig: {}".format(z_sig))
                    print("X_tilde: {}".format(x_tilde))
                    print("L1: {}".format(l1))
                    print("L2: {}".format(l2))
                    recon_error, KL = sess.run([myVAE.reconError, myVAE.KLdiv], feed_dict={myVAE.X:batch_xs})
                    print("Reconstruction Error: {}".format(recon_error))
                    print("KL Divergence: {}".format(KL))
                    print("Batch Cost: {}".format(c))
                    #=======DEBUG=============================================================================

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), " | cost = ", "{:.9f}".format(c))

        print("Optimization Finished!")

        if(not load_params):
            myVAE.save_params()

        # # Shuffle the test input and plot the reconstruction
        # random_indices = np.random.permutation(test.num_examples)
        # test.data = test.data[random_indices, :]
        # x_sample = test.data[:200]
        # # x_sample = mnist.test.next_batch(100)[0]
        # x_reconstruct = myVAE.reconstruct(x_sample, sess)
        # print("Image Shape: {}".format(x_reconstruct.shape))

        # plt.figure()
        # for i in range(5):
        #     a1 = plt.subplot(5, 2, 2*i + 1, projection='3d')
        #     # plt.imshow(x_sample[i].reshape(24,24,24), vmin=0, vmax=1)
        #     data_3d = np.reshape(test.data[i], (24,24,24))
        #     # data_3d = np.reshape(modelnet.test_data.data[i], modelnet.image_size)
        #     xx, yy, zz = np.where(data_3d > 0.1)
        #     a1.scatter(xx,yy,zz)
        #     plt.title("Test input")
        #     a2 = plt.subplot(5, 2, 2*i + 2, projection='3d')
        #     # plt.imshow(x_reconstruct[i].reshape(24,24,24), vmin=0, vmax=1)
        #     data_3d = x_reconstruct[i].reshape(24,24,24)
        #     xx2, yy2, zz2 = np.where(data_3d > 0.1)
        #     a2.scatter(xx2,yy2,zz2)
        #     plt.title("Reconstruction")
        # plt.show()


        # # Applying encode and decode over test set
        # encode_decode = sess.run(
        #     ae.y_pred, feed_dict={ae.X: test_data.images[:examples_to_show]})
        # # Compare original images with their reconstructions
        # f, a = plt.subplots(2, 10, figsize=(10, 2))
        # for i in range(examples_to_show):
        #     a[0][i].imshow(np.reshape(test_data.images[i], (28, 28)))
        #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # f.show()
        # plt.draw()
        # plt.waitforbuttonpress()

if __name__ == '__main__':
    main()