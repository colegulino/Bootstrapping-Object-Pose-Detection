# Exterior packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

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
learning_rate = 0.001
training_epochs = 2000
batch_size = 256
display_step = 1
examples_to_show = 10
debug = False
load_params = True

# Setup the architecture
architecture = \
{
    'no_inputs' : data.input_dim,
    'no_hidden_units' : 3,
    'hidden_dims' : \
    {
        'h1' : 500,
        'h2' : 500,
        'h3' : 100
    },
    'no_latent_dims' : 50
}

def main():
    print("Train Size: {}".format(train.shape))
    print("Test Size: {}".format(test.shape))
    print("Validation Size: {}".format(validation.shape))

    print("Initialized the Variables")
    print("Begin training")

    costs = {}
    with tf.Session() as sess:
        # Set up autoencoder
        myVAE = vae.variationalAutoEncoder(architecture, learning_rate, tf.nn.elu, sess, load_params=load_params)
        print("Setup the Model")

        # Initialize all variables
        init = tf.initialize_all_variables()

        sess.run(init)

        total_batch = int(train.num_examples/batch_size)
        latent_vals = {}

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
                    x_tilde = sess.run([myVAE.x_tild, myVAE.l1, myVAE.l2], feed_dict={myVAE.X:batch_xs})
                    z_mean, z_sig = sess.run([myVAE.z_mean, myVAE.z_sigma], feed_dict={myVAE.X:batch_xs})
                    print("Z_mean: {}".format(z_mean))
                    print("Z_sig: {}".format(z_sig))
                    print("X_tilde: {}".format(x_tilde))
                    recon_error, KL = sess.run([myVAE.reconError, myVAE.KLdiv], feed_dict={myVAE.X:batch_xs})
                    print("Reconstruction Error: {}".format(recon_error))
                    print("KL Divergence: {}".format(KL))
                    print("Batch Cost: {}".format(c))
                    #=======DEBUG=============================================================================

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), " | cost = ", "{:.9f}".format(c))
                costs[epoch] = c
                # myVAE.plot_2d_latent_space(test, sess)
                # latent_vals[epoch] = myVAE.getLatentParams(test.get_data(), sess)

        print("Optimization Finished!")

        # Save parameters
        if(True):
            myVAE.save_params()

        # Save costs
        with open('costs.p', 'wb') as f:
            pickle.dump(costs, f)


        # # Save latent values
        # with open('latent_vals.p', 'wb') as f:
        #     pickle.dump(latent_vals, f)

        # # Shuffle the test input and plot the reconstruction
        # random_indices = np.random.permutation(test.num_examples)
        # test.data = test.data[random_indices, :]
        # x_sample = test.data[:200]
        # # x_sample = mnist.test.next_batch(100)[0]
        # x_reconstruct = vae.sample_binary(myVAE.reconstruct(x_sample, sess))
        # print(x_reconstruct)
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

        # plt.figure()
        # for i in range(5):
        #     a1 = plt.subplot(5, 2, 2*i + 1, projection='3d')
        #     # plt.imshow(x_sample[i].reshape(24,24,24), vmin=0, vmax=1)
        #     data_3d = np.reshape(test.data[i], (24,24,24))
        #     # data_3d = np.reshape(modelnet.test_data.data[i], modelnet.image_size)
        #     xx, yy, zz = np.where(data_3d > 0.1)
        #     a1.scatter(xx,yy,zz)
        #     # plt.title("Test input")
        #     # a2 = plt.subplot(5, 2, 2*i + 2, projection='3d')
        #     # # plt.imshow(x_reconstruct[i].reshape(24,24,24), vmin=0, vmax=1)
        #     # data_3d = x_reconstruct[i].reshape(24,24,24)
        #     # xx2, yy2, zz2 = np.where(data_3d > 0.1)
        #     # a2.scatter(xx2,yy2,zz2)
        #     plt.title("Generated Data")
        # plt.show()


def print_latent():
    with open('latent_vals.p', 'rb') as f:
        latent_vals = pickle.load(f)

    class_colors = vae.generate_random_colors(len(data.class_dict))

    labels = test.get_labels()

    for key in latent_vals.keys():
        fig = plt.figure(key)
        fig.suptitle('Latent Space After {} Epochs'.format(key + 1))
        for latent, c in zip(latent_vals[key], labels):
            plt.scatter(latent[0], latent[1], c=class_colors[c, :])

    plt.show()

def generate():

    with tf.Session() as sess:
        # Set up autoencoder
        myVAE = vae.variationalAutoEncoder(architecture, learning_rate, tf.nn.elu, sess, load_params=load_params)
        print("Setup the Model")

        # Initialize all variables
        init = tf.initialize_all_variables()

        sess.run(init)

        samples = myVAE.generate(5, sess)
        plt.figure()
        for i in range(5):
            fig = plt.figure(i)
            a1 = fig.add_subplot(1,1,1, projection='3d')
            data_3d = np.reshape(samples[i, :], (24,24,24))
            xx, yy, zz = np.where(data_3d > 0.1)
            a1.scatter(xx,yy,zz)
            plt.title("Generated Data")
        plt.show()

def voxel_fill():
    labels = test.get_labels()

    toilet_number = data.class_dict['bed']
    print('Toilet Number: {}'.format(toilet_number))

    idx = np.where(labels == toilet_number)
    index = np.random.randint(idx[0].shape[0])
    data_3d = np.reshape(test.data[idx[0][index], :], (24,24,24))
    old_data = np.array(data_3d, copy=True)
    print(test.image_size)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    a1 = fig1.add_subplot(1,1,1, projection='3d')
    a2 = fig2.add_subplot(1,1,1, projection='3d')
    a3 = fig3.add_subplot(1,1,1, projection='3d')
    a4 = fig4.add_subplot(1,1,1, projection='3d')

    a1.set_xlim([0, 24])
    a1.set_ylim([0, 24])
    a1.set_zlim([0, 24])
    a2.set_xlim([0, 24])
    a2.set_ylim([0, 24])
    a2.set_zlim([0, 24])
    a3.set_xlim([0, 24])
    a3.set_ylim([0, 24])
    a3.set_zlim([0, 24])
    a4.set_xlim([0, 24])
    a4.set_ylim([0, 24])
    a4.set_zlim([0, 24])

    xx, yy, zz = np.where(data_3d > 0.1)
    a1.scatter(xx,yy,zz)

    s = data_3d[:12,:,:].shape

    data_3d[:12,:,:] = np.zeros(s)

    xx, yy, zz = np.where(data_3d > 0.1)
    a2.scatter(xx,yy,zz)

    data_3d[:12,:,:] = np.round(np.random.uniform(size=s))

    xx, yy, zz = np.where(data_3d > 0.1)
    a3.scatter(xx,yy,zz)

    data_1d = np.reshape(data_3d, (1,data_3d.size))

    with tf.Session() as sess:
        # Set up autoencoder
        myVAE = vae.variationalAutoEncoder(architecture, learning_rate, tf.nn.elu, sess, load_params=load_params)
        print("Setup the Model")

        # Initialize all variables
        init = tf.initialize_all_variables()

        sess.run(init)
        for i in range(200):
            data_1d = vae.sample_binary(myVAE.reconstruct(data_1d, sess))
            data_3d = np.reshape(data_1d, (24,24,24))
            data_3d[12:,:,:] = old_data[12:,:,:]
            data_1d = np.reshape(data_3d, (1,data_3d.size))

    data_3d = np.reshape(data_1d, (24,24,24))
    data_3d[12:,:,:] = old_data[12:,:,:]
    xx, yy, zz = np.where(data_3d > 0.1)
    a4.scatter(xx,yy,zz)

    plt.show()

if __name__ == '__main__':
    # main()

    voxel_fill()

    # print_latent()

    # generate()