# Import tensorflow
import tensorflow as tf
# Import numpy
import numpy as np

class variationalAutoEncoder():
    #
    # Construction of autoencoder model
    #
    # @param model_dimensions List of model dimensions for the layers going from bottom to top
    # @param learning_rate Learning rate of the optimizer
    # @param nonlin_fxn Nonlinear function for the layers
    # @param dropout Dropout probability
    #
    def __init__(self, model_dimensions, learning_rate, nonlin_fxn, dropout=0.5):
        self.model_dimensions = model_dimensions
        self.learning_rate = learning_rate
        self.nonlin_fxn = nonlin_fxn
        self.dropout = dropout

        # Generate the model parameters before the hidden layer
        self.weights, self.biases = self.setUpParams(self.model_dimensions[:-1])
        # Generate the model parameters for the hidden layer
        self.setUpLatentParams()

        # Setup the operations for encoder and decoder for feedforward
        self.X = tf.placeholder("float", [None, model_dimensions[0]])
        self.z_mean, self.z_sigma = self.encoder(self.X) # Get the parameters
        self.z = self.sampleLatent(self.z_mean, self.z_sigma) # Sample the Latent space
        self.x_tild = self.decode(self.z) # Decode

        self.decoder_op = self.decoder(self.encoder_op)

        # Set up the operators for the output and predictions
        self.y_pred = self.decoder_op
        self.y_true = self.X


    #
    # Function that sets up tensorflow variables that serve as the parameters of the model
    # Documentation on Variables: https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
    #
    # @param model_dimensions List of model dimensions for the layers going from bottom to top
    # @return Tuple (weights, biases) Tuple of dictionaries containing weights and biases
    #
    def setUpParams(self, model_dimensions):
        n = len(model_dimensions) - 1

        # Initialize the weight variables
        weights = {}
        for i in range(n):
            weights["encoder_W{}".format(i + 1)] = \
                tf.Variable(tf.random_normal([model_dimensions[i], model_dimensions[i+1]]))
            weights["decoder_W{}".format(i + 1)] = \
                tf.Variable(tf.random_normal([model_dimensions[n - i], model_dimensions[n - i - 1]]))

        # Initialize the bias variables
        biases = {}
        for i in range(1, len(model_dimensions)):
            biases["encoder_b{}".format(i)] = \
                tf.Variable(tf.random_normal([model_dimensions[i]]))
            biases["decoder_b{}".format(i)] = \
                tf.Variable(tf.random_normal([model_dimensions[n - i]]))

        return weights, biases

    #
    # Sets up the parameters for the hidden layers
    #
    def setUpLatentParams(self):
        self.weights["encoder_W_mu"] = tf.Variable(tf.random_normal([model_dimensions[-2], model_dimensions[-1]]))
        self.weights["encoder_W_sig"] = tf.Variable(tf.random_normal([model_dimensions[-2], model_dimensions[-1]]))
        self.weights["decoder_W_z"] = tf.Variable(tf.random_normal([model_dimensions[-2], model_dimensions[-1]]))
        self.weights["encoder_b_mu"] = tf.Variable(tf.random_normal([model_dimensions[-1]]))
        self.weights["encoder_b_sig"] = tf.Variable(tf.random_normal([model_dimensions[-1]]))
        self.weights["decoder_b_z"] = tf.Variable(tf.random_normal([model_dimensions[-1]]))

    #
    # Encoder of the network
    #
    # @param x input
    # @param nonlin_fxn Nonlinear function
    # @return Tuple (z_mean, z_sigma) - Tuple of the latent space model parameters
    #
    def encoder(self, x, nonlin_fxn):
        # Get the output before the latent space
        for i in range(1, len(model_dimensions)):
            x = nonlin_fxn(tf.add(tf.matmul(x, self.weights['encoder_W{}'.format(i)]),
                            self.biases['encoder_b{}'.format(i)]))

        z_mean = nonlin_fxn(tf.add(tf.matmul(x, self.weights['encoder_W_mu']),
                                                self.bias['encoder_b_mu']))
        z_sigma = nonlin_fxn(tf.add(tf.matmul(x, self.weights['encoder_W_sig']),
                                                 self.bias['encoder_b_sig']))

        return z_mean, z_sigma

    #
    # Do the reparameterization trick to sample the gaussian
    #
    # @param z_mean Latent mean
    # @param z_sigma Latent sigma
    # @return Latent space sample
    #
    def sampleLatent(self, z_mean, z_sigma):
        epsilon = tf.random_normal(tf.shape(z_sigma))
        return z_mean + epsilon * tf.exp(z_sigma)

    #
    # Decode hte latent variable to sample the input space
    #
    # @param z Input
    # @param nonlin_fxn Nonlinear activation function
    # @return Decoded output
    #
    def decoder(self, z, nonlin_fxn):
        z = nonlin_fxn(tf.add(tf.matmul(x, self.weights['decoder_W_z']),
                                            self.bias['decoder_b_z']))
        for i in range(1, len(self.weights) - 2):
            z = tf.nn.sigmoid(tf.add(tf.matmul(z, self.weights['decoder_W{}'.format(i)]),
                              self.biases['decoder_b{}'.format(i)]))

        return z

if __name__ == '__main__':
    vae = variationalAutoEncoder([784, 100, 50, 2], 0.5, 0)

    for key, value in vae.weights.items():
        print("Layer: {} | Shape: {}".format(key, value))

    for key, value in vae.biases.items():
        print("Layer: {} | Shape: {}".format(key, value))