# Import tensorflow
import tensorflow as tf

class autoencoderTest():
    # 
    # Construction of autoencoder model
    # 
    # @param model_dimensions List of model dimensions for the layers going from bottom to top
    # @param learning_rate Learning rate of the optimizer
    # 
    def __init__(self, model_dimensions, learning_rate):
        self.model_dimensions = model_dimensions
        self.learning_rate = learning_rate

        # Initialize the weights and biases
        self.weights, self.biases = self.setUpParams(model_dimensions)

        # Get the operators for the feedforward model
        self.X = tf.placeholder("float", [None, model_dimensions[0]])
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)

        # Set up the operators for the output and predictions
        self.y_pred = self.decoder_op
        self.y_true = self.X

        # Set up the cost
        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        # pred_tf = tf.convert_to_tensor(self.y_pred)
        # true_tf = tf.convert_to_tensor(self.y_true)
        # self.cost = -1 * tf.reduce_sum(tf.add(tf.mul(tf.log(1e-10 + pred_tf), true_tf), tf.mul(tf.log(1e-10+1-pred_tf), (1-true_tf))), 1)

        # Set up the optimizer
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

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
    # Run the forward pass of the encoder
    # 
    # @param x Input to the encoder
    # @return Output of the encoder feedforward model
    # 
    def encoder(self, x):
        for i in range(1, len(self.weights) - 1):
            x = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_W{}'.format(i)]),
                              self.biases['encoder_b{}'.format(i)]))

        return x

    # 
    # Run the forward pass of the decoder
    # 
    # @param x Input to the decoder
    # @return Output of the decoder feedforward model
    # 
    def decoder(self, x):
        for i in range(1, len(self.weights) - 1):
            x = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_W{}'.format(i)]),
                              self.biases['decoder_b{}'.format(i)]))

        return x