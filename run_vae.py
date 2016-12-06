# Exterior packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Setup the architecture
architecture = \
{
    'no_inputs' : data.input_dim,
    'no_hidden_units' : 2,
    'hidden_dims' : \
    {
        'h1' : 500,
        'h2' : 500
    },
    'no_latent_dims' : 50
}

print("Architecture: {}".format(architecture))
def main():
    print("Train Size: {}".format(train.shape))
    print("Test Size: {}".format(test.shape))
    print("Validation Size: {}".format(validation.shape))

    print(train.get_labels())

    # Set up autoencoder
    myVAE = vae.variationalAutoEncoder(architecture, learning_rate, tf.nn.elu)
    print("Setup the Model")

    # Initialize all variables
    init = tf.initialize_all_variables()
    print("Initialized the Variables")
    print("Begin training")

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(train.num_examples/batch_size)

        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train.next_batch(batch_size)
                #=======DEBUG=============================================================================
                # for i in batch_xs:
                #     print(i)
                # print(batch_xs)
                # # Run optimization op (backprop) and cost op (to get loss value)
                # x_tilde, l1, l2 = sess.run([myVAE.x_tild, myVAE.l1, myVAE.l2], feed_dict={myVAE.X:batch_xs})
                # z_mean, z_sig = sess.run([myVAE.z_mean, myVAE.z_sigma], feed_dict={myVAE.X:batch_xs})
                # print("Z_mean: {}".format(z_mean))
                # print("Z_sig: {}".format(z_sig))
                # print("X_tilde: {}".format(x_tilde))
                # print("L1: {}".format(l1))
                # print("L2: {}".format(l2))
                # recon_error, KL = sess.run([myVAE.reconError, myVAE.KLdiv], feed_dict={myVAE.X:batch_xs})
                # print("Reconstruction Error: {}".format(recon_error))
                # print("KL Divergence: {}".format(KL))
                #=======DEBUG=============================================================================

                _, c = sess.run([myVAE.optimizer, myVAE.cost], feed_dict={myVAE.X: batch_xs})
                # print("Batch Cost: {}".format(c))
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: {} | Cost={}".format(epoch+1, c))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

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