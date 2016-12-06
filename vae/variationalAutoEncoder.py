# Import tensorflow
import tensorflow as tf
# Import numpy
import numpy as np

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class variationalAutoEncoder():
    #
    # Construction of autoencoder model
    #
    # @param architecture Architecture dictionary of the model; example:
        # architecture = \
        # {
        #     'no_inputs' : data.input_dim,
        #     'hidden_dims' : \
        #     {
        #         'h1' : 500,
        #         'h2' : 500
        #     },
        #     'no_latent_dims' : 50
        # }
    # @param learning_rate Learning rate of the optimizer
    # @param nonlin_fxn Nonlinear function for the layers
    # @param dropout Dropout probability
    #
    def __init__(self, architecture, learning_rate, nonlin_fxn, dropout=0.5):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.nonlin_fxn = nonlin_fxn
        self.dropout = dropout

        # Generate the model parameters before the hidden layer
        self.weights, self.biases = self.setUpParams(architecture)

        print("Weights: {}".format(self.weights))
        print("Biases: {}".format(self.biases))
        # Generate the model parameters for the hidden layer
        # self.setUpLatentParams(model_dimensions)

        # for key, value in self.weights.items():
        #     print("Layer: {} | Shape: {}".format(key, value))

        # for key, value in self.biases.items():
        #     print("Layer: {} | Shape: {}".format(key, value))

        # Setup the operations for encoder and decoder for feedforward
        self.X = tf.placeholder("float", [None, self.architecture['no_inputs']])
        self.z_mean, self.z_sigma = self.encoder(self.X, nonlin_fxn) # Get the parameters
        self.z = self.sampleLatent(self.z_mean, self.z_sigma) # Sample the Latent space
        self.x_tilde = self.decoder(self.z, nonlin_fxn) # Decode

        # Set up the operators for the output and predictions
        self.y_pred = self.x_tilde
        self.y_true = self.X

        # Setup the cost
        self.reconError = self.crossEntropy(self.y_true, self.y_pred)
        self.KLdiv = self.KL(self.z_mean, self.z_sigma)
        self.cost = tf.reduce_mean(self.reconError + self.KLdiv)

        # Setup the optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    # architecture = \
    # {
    #     'no_inputs' : data.input_dim,
    #     'no_hidden_units' : 2,
    #     'hidden_dims' : \
    #     {
    #         'h1' : 500,
    #         'h2' : 500
    #     },
    #     'no_latent_dims' : 50
    # }
    #
    # Function that sets up tensorflow variables that serve as the parameters of the model
    # Documentation on Variables: https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
    #
    # @param model_dimensions architecture Architecture dictionary of the model: example in run_vae.py
    # @return Tuple (weights, biases) Tuple of dictionaries containing weights and biases
    #
    def setUpParams(self, architecture):
        latent_weights = \
        {
            'W_mu' :  tf.Variable(xavier_init(architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])],
                                              architecture['no_latent_dims'])),
            'W_sig' :  tf.Variable(xavier_init(architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])],
                                               architecture['no_latent_dims'])),
            'W_z' :  tf.Variable(xavier_init(architecture['no_latent_dims'],
                                             architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])]))
        }

        latent_biases = \
        {
            'b_mu' :  tf.Variable(tf.zeros(architecture['no_latent_dims'])),
            'b_sig' :  tf.Variable(tf.zeros(architecture['no_latent_dims'])),
            'b_z' :  tf.Variable(tf.zeros(architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])]))
        }

        weight_list = [architecture['no_inputs']]
        for i in range(architecture['no_hidden_units']):
            weight_list.append(architecture['hidden_dims']['h{}'.format(i+1)])

        # Initialize the encoder weights and biases
        encoder_weights = {}
        encoder_biases = {}
        weight_count = 1
        for i in range(len(weight_list) - 1):
            encoder_weights['W{}'.format(weight_count)] = tf.Variable(xavier_init(weight_list[i], weight_list[i+1]))
            weight_count = weight_count + 1
        bias_count = 1
        for i in range(len(weight_list) - 1):
            encoder_biases['b{}'.format(bias_count)] = tf.Variable(tf.zeros(weight_list[i+1]))
            bias_count = bias_count + 1

        # Initialize the decoder weights and biases
        decoder_weights = {}
        decoder_biases = {}
        weight_count = 1
        weight_list = weight_list[::-1] # reverse
        for i in range(len(weight_list) - 1):
            decoder_weights['W{}'.format(weight_count)] = tf.Variable(xavier_init(weight_list[i], weight_list[i+1]))
            weight_count = weight_count + 1
        bias_count = 1
        for i in range(len(weight_list) - 1):
            decoder_biases['b{}'.format(bias_count)] = tf.Variable(tf.zeros(weight_list[i+1]))
            bias_count = bias_count + 1

        # Put it all together
        weights = {}
        biases = {}
        weights['encoder'] = encoder_weights
        weights['decoder'] = decoder_weights
        weights['latent'] = latent_weights
        biases['encoder'] = encoder_biases
        biases['decoder'] = decoder_biases
        biases['latent'] = latent_biases
        # Initialize the weight variables
        # weights = {}
        # for i in range(n):
        #     weights["encoder_W{}".format(i + 1)] = \
        #         tf.Variable(tf.random_normal([model_dimensions[i], model_dimensions[i+1]]), dtype=tf.float32)
        #     weights["decoder_W{}".format(i + 1)] = \
        #         tf.Variable(tf.random_normal([model_dimensions[n - i], model_dimensions[n - i - 1]]), dtype=tf.float32)

        # weights ={
        #     'encoder_W1' : tf.Variable(xavier_init(model_dimensions[0], model_dimensions[1])),
        #     'encoder_W2' : tf.Variable(xavier_init(model_dimensions[1], model_dimensions[2])),
        #     'encoder_W_mu' : tf.Variable(xavier_init(model_dimensions[2], model_dimensions[3])),
        #     'encoder_W_sig' : tf.Variable(xavier_init(model_dimensions[2], model_dimensions[3])),
        #     'decoder_W_z' : tf.Variable(xavier_init(model_dimensions[3], model_dimensions[2])),
        #     'decoder_W1' : tf.Variable(xavier_init(model_dimensions[2], model_dimensions[1])),
        #     'decoder_W2' : tf.Variable(xavier_init(model_dimensions[1], model_dimensions[0]))}

        # biases ={
        #     'encoder_b1' : tf.Variable(tf.zeros([model_dimensions[1]]), dtype=tf.float32),
        #     'encoder_b2' : tf.Variable(tf.zeros([model_dimensions[2]]), dtype=tf.float32),
        #     'encoder_b_mu' : tf.Variable(tf.zeros([model_dimensions[3]]), dtype=tf.float32),
        #     'encoder_b_sig' : tf.Variable(tf.zeros([model_dimensions[3]]), dtype=tf.float32),
        #     'decoder_b_z' : tf.Variable(tf.zeros([model_dimensions[2]]), dtype=tf.float32),
        #     'decoder_b1' : tf.Variable(tf.zeros([model_dimensions[1]]), dtype=tf.float32),
        #     'decoder_b2' : tf.Variable(tf.zeros([model_dimensions[0]]), dtype=tf.float32)}

        # # # Initialize the bias variables
        # biases = {}
        # for i in range(1, len(model_dimensions)):
        #     biases["encoder_b{}".format(i)] = \
        #         tf.Variable(tf.random_normal([model_dimensions[i]]), dtype=tf.float32)
        #     biases["decoder_b{}".format(i)] = \
        #         tf.Variable(tf.random_normal([model_dimensions[n - i]]), dtype=tf.float32)

        return weights, biases

    #
    # Sets up the parameters for the hidden layers
    #
    def setUpLatentParams(self, model_dimensions):
        self.weights["encoder_W_mu"] = tf.Variable(tf.random_normal([model_dimensions[-2], model_dimensions[-1]]), dtype=tf.float32)
        self.weights["encoder_W_sig"] = tf.Variable(tf.random_normal([model_dimensions[-2], model_dimensions[-1]]), dtype=tf.float32)
        self.weights["decoder_W_z"] = tf.Variable(tf.random_normal([model_dimensions[-1], model_dimensions[-2]]), dtype=tf.float32)
        self.biases["encoder_b_mu"] = tf.Variable(tf.random_normal([model_dimensions[-1]]), dtype=tf.float32)
        self.biases["encoder_b_sig"] = tf.Variable(tf.random_normal([model_dimensions[-1]]), dtype=tf.float32)
        self.biases["decoder_b_z"] = tf.Variable(tf.random_normal([model_dimensions[-2]]), dtype=tf.float32)

    #
    # Encoder of the network
    #
    # @param x input
    # @param nonlin_fxn Nonlinear function
    # @return Tuple (z_mean, z_sigma) - Tuple of the latent space model parameters
    #
    def encoder(self, x, nonlin_fxn):
        # Get the output before the latent space
        # l1 = nonlin_fxn(tf.add(tf.matmul(x, self.weights['encoder_W1']),
        #                     self.biases['encoder_b1']))
        # l2 = nonlin_fxn(tf.add(tf.matmul(l1, self.weights['encoder_W2']),
        #                     self.biases['encoder_b2']))

        # z_mean = tf.add(tf.matmul(l2, self.weights['encoder_W_mu']),
        #                 self.biases['encoder_b_mu'])

        # z_sigma = tf.add(tf.matmul(l2, self.weights['encoder_W_sig']),
        #                 self.biases['encoder_b_sig'])

        # return z_mean, z_sigma


        # Get the parameters
        weights = self.weights['encoder']
        biases = self.biases['encoder']
        latent_weights = self.weights['latent']
        latent_biases = self.biases['latent']
        no_hu = self.architecture['no_hidden_units']

        for i in range(no_hu):
            x = nonlin_fxn(tf.add(tf.matmul(x, weights['W{}'.format(i+1)]),
                                               biases['b{}'.format(i+1)]))

        z_mean = tf.add(tf.matmul(x, latent_weights['W_mu']),
                                     latent_biases['b_mu'])

        z_sigma = tf.add(tf.matmul(x, latent_weights['W_sig']),
                                      latent_biases['b_sig'])

        return z_mean, z_sigma

    #
    # Do the reparameterization trick to sample the gaussian
    #
    # @param z_mean Latent mean
    # @param z_sigma Latent sigma
    # @return Latent space sample
    #
    def sampleLatent(self, z_mean, z_sigma):
        epsilon = tf.random_normal(tf.shape(z_sigma), 0, 1)
        return z_mean + epsilon * tf.exp(z_sigma)

    #
    # Decode hte latent variable to sample the input space
    #
    # @param z Input
    # @param nonlin_fxn Nonlinear activation function
    # @return Decoded output
    #
    def decoder(self, z, nonlin_fxn):
        # l1 = nonlin_fxn(tf.add(tf.matmul(z, self.weights['decoder_W_z']),
        #                                     self.biases['decoder_b_z']))

        # l2 = nonlin_fxn(tf.add(tf.matmul(l1, self.weights['decoder_W1']),
        #                     self.biases['decoder_b1']))
        # return tf.nn.sigmoid(tf.add(tf.matmul(l2, self.weights['decoder_W2']),
        #                     self.biases['decoder_b2'])), l1, l2

        # Get the parameters
        weights = self.weights['decoder']
        biases = self.biases['decoder']
        latent_weights = self.weights['latent']
        latent_biases = self.biases['latent']
        no_hu = self.architecture['no_hidden_units']

        z = nonlin_fxn(tf.add(tf.matmul(z, latent_weights['W_z']),
                                           latent_biases['b_z']))

        for i in range(no_hu - 1):
            z = nonlin_fxn(tf.add(tf.matmul(z, weights['W{}'.format(i+1)]),
                                               biases['b{}'.format(i+1)]))

        return tf.nn.sigmoid(tf.add(tf.matmul(z, weights['W{}'.format(no_hu)]),
                                                  biases['b{}'.format(no_hu)]))

    #
    # Binary cross-entropy error
    #
    # @param x_tilde Reconstructed input
    # @param x Actual input
    # @param offset Offset for clipping
    # @return Cross-entropy error between x_tilde and x
    #
    def crossEntropy(self, x_tilde, x, offset=1e-7):
        x_tilde = tf.clip_by_value(x_tilde, offset, 1 - offset)
        return -tf.reduce_sum(x * tf.log(x_tilde) +
                              (1 - x) * tf.log(1 - x_tilde), 1)

    #
    # KL divergence
    #
    # @param z_mean Latent space mean
    # @param z_sigma Latent space sigma
    # @return KL divergence KL(q||q)
    #
    def KL(self, z_mean, z_sigma):
        return - 0.5 * tf.reduce_sum(1 + 2 * z_sigma - z_mean**2 -
                                     tf.exp(2 * z_sigma), 1)

    #
    # Get the total cost
    #
    # @param x_tilde Reconstructed input
    # @param x Actual input
    # @param z_mean Latent space mean
    # @param z_sigma Latent space sigma
    # @return Total cost
    #
    def getCost(self, x_tilde, x, z_mean, z_sigma):
        return self.crossEntropy(x_tilde, x) + self.KL(z_mean, z_sigma)

def s(x):
    return x

if __name__ == '__main__':
    vae = variationalAutoEncoder([784, 100, 50, 2], 0.5, s)

