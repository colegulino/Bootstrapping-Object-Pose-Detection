# Import tensorflow
import tensorflow as tf
# Import numpy
import numpy as np
# Import pickle
import pickle

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def sample_binary(probs):
    return (probs > np.random.rand(probs.shape[0], probs.shape[1])) * 1

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
    # @param sess Tensorflow session
    # @param load_params Boolean whether to load or restore the parameters
    #
    def __init__(self, architecture, learning_rate, nonlin_fxn, sess, load_params=False):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.nonlin_fxn = nonlin_fxn

        if(load_params):
            self.weights, self.biases = self.load_params()
            print("Loaded the parameters")
        else:
            self.weights, self.biases = self.setUpParams(architecture)

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
                                              architecture['no_latent_dims']), name='W_mu'),
            'W_sig' :  tf.Variable(xavier_init(architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])],
                                               architecture['no_latent_dims']), name='W_sig'),
            'W_z' :  tf.Variable(xavier_init(architecture['no_latent_dims'],
                                             architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])]), name='W_z')
        }

        latent_biases = \
        {
            'b_mu' :  tf.Variable(tf.zeros(architecture['no_latent_dims']), name='b_mu'),
            'b_sig' :  tf.Variable(tf.zeros(architecture['no_latent_dims']), name='b_sig'),
            'b_z' :  tf.Variable(tf.zeros(architecture['hidden_dims']['h{}'.format(architecture['no_hidden_units'])]), name='b_z')
        }

        weight_list = [architecture['no_inputs']]
        for i in range(architecture['no_hidden_units']):
            weight_list.append(architecture['hidden_dims']['h{}'.format(i+1)])

        # Initialize the encoder weights and biases
        encoder_weights = {}
        encoder_biases = {}
        weight_count = 1
        for i in range(len(weight_list) - 1):
            encoder_weights['W{}'.format(weight_count)] = tf.Variable(xavier_init(weight_list[i], weight_list[i+1]),
                            name='encoder_W{}'.format(weight_count))
            weight_count = weight_count + 1
        bias_count = 1
        for i in range(len(weight_list) - 1):
            encoder_biases['b{}'.format(bias_count)] = tf.Variable(tf.zeros(weight_list[i+1]),
                            name='encoder_b{}'.format(bias_count))
            bias_count = bias_count + 1

        # Initialize the decoder weights and biases
        decoder_weights = {}
        decoder_biases = {}
        weight_count = 1
        weight_list = weight_list[::-1] # reverse
        for i in range(len(weight_list) - 1):
            decoder_weights['W{}'.format(weight_count)] = tf.Variable(xavier_init(weight_list[i], weight_list[i+1]),
                            name='decoder_W{}'.format(weight_count))
            weight_count = weight_count + 1
        bias_count = 1
        for i in range(len(weight_list) - 1):
            decoder_biases['b{}'.format(bias_count)] = tf.Variable(tf.zeros(weight_list[i+1]),
                            name='decoder_b{}'.format(bias_count))
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

        return weights, biases

    #
    # Encoder of the network
    #
    # @param x input
    # @param nonlin_fxn Nonlinear function
    # @return Tuple (z_mean, z_sigma) - Tuple of the latent space model parameters
    #
    def encoder(self, x, nonlin_fxn):
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

    #
    # Get latent params
    #
    # @param x - Input
    # @param sess - Tensorflow session
    # @return z_mean, z_sigma
    #
    def getLatentParams(self, x, sess):
        return sess.run([self.z_mean, self.z_sigma], feed_dict={self.X : x})

    #
    # Get reconstruction
    # @param x - Input
    # @param sess - Tensorflow sesion
    # @return Reconstruciton
    #
    def reconstruct(self, x, sess):
        return sess.run([self.x_tilde], feed_dict={self.X : x})[0]

    #
    # Sample the latent space
    #
    # @param no_samples Number of latent spaces to sample
    # @param sess Tensorflow session
    # @return Samples from the space
    #
    def generate(self, no_samples, sess):
        z = np.random.normal(size=(no_samples, self.architecture['no_latent_dims']))
        return sess.run([self.x_tilde], feed_dict={self.z : z})[0]

    #
    # Helper function to initialize the weight and bias dicitonary
    #
    def init_weight_and_biases_dict(self):
        weights = {}
        biases = {}
        for t in ['decoder', 'encoder', 'latent']:
            weights[t] = {}
            biases[t] = {}

        return weights, biases

    #
    # Save the parameters
    #
    # @param sess Tensorflow session
    # @param filename Name of the filepath to save the weights
    #
    def load_params(self):
        # new_saver = tf.train.import_meta_graph('models/vae/my-model.meta')
        # new_saver.restore(sess, tf.train.latest_checkpoint('./models/vae/'))

        # weights, biases = sess.run([self.weights, self.biases])
        path = 'models/vae/params/{}_{}_{}.npz'

        # Load the architecture
        with open('models/vae/params/architecture.p', 'rb') as f:
            arch = pickle.load(f)

        weights, biases = self.init_weight_and_biases_dict()

        print(arch)
        for param_type in arch:
            print(param_type)
            with open(path.format(param_type[0], param_type[1], param_type[2]), 'rb') as f:
                archive = np.load(f)
                if(param_type[0] == 'W'):
                    weights[param_type[1]][param_type[2]] = tf.Variable(archive['arr_0'], name='{}_{}'.format(param_type[1], param_type[2]))
                else:
                    biases[param_type[1]][param_type[2]] = tf.Variable(archive['arr_0'], name='{}_{}'.format(param_type[1], param_type[2]))

        return weights, biases

        # # Save the weights
        # for t in self.weights.keys():
        #     for x in self.weights[t].keys():
        #         with open(path.format('W', t, x), 'rb') as f:
        #             archive = np.load(f)
        #             self.weights[t][x] = archive['arr_0']

        # # Save the biases
        # types_of_biases = self.biases.keys()
        # for t in self.biases.keys():
        #     for x in self.biases[t].keys():
        #         with open(path.format('b', t, x), 'rb') as f:
        #             archive = np.load(f)
        #             self.biases[t][x] = archive['arr_0']

    #
    # Load Parameters
    #
    # @param sess Tensorflow session
    #
    def save_params(self):
        # saver = tf.train.Saver()
        # saver.save(sess, 'models/vae/my-model')

        # weights, biases = sess.run([self.weights, self.biases])
        path = 'models/vae/params/{}_{}_{}.npz'

        architecture = []

        # Save the weights
        for t in self.weights.keys():
            for x in self.weights[t].keys():
                with open(path.format('W', t, x), 'wb') as f:
                    architecture.append(('W',t,x))
                    np.savez(f, self.weights[t][x].eval())

        # Save the biases
        types_of_biases = self.biases.keys()
        for t in self.biases.keys():
            for x in self.biases[t].keys():
                with open(path.format('b', t, x), 'wb') as f:
                    architecture.append(('b',t,x))
                    np.savez(f, self.biases[t][x].eval())

        # Save the types of the network
        with open('models/vae/params/architecture.p', 'wb') as f:
            pickle.dump(architecture, f)

        print("Done saving the model!")

def s(x):
    return x

if __name__ == '__main__':
    vae = variationalAutoEncoder([784, 100, 50, 2], 0.5, s)