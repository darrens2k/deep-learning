"""
A neural network built using only numpy.
Note that the code achieves 90% accuracy
within 10 epochs approximately 6 out every
10 times the code is run.
"""

# pep8 formatted using autopep8

# Assignment 1 - MMAI 5500 - Deep Learning
# Darren Singh
# 216236275

import numpy as np

# Read in assignment data.
# Update path to data file when testing.
# For some reason windows wants the full path even though the file is in the working directory.
fname = R"C:\Users\darre\Documents\GitHub\deep-learning\Assignments\assign1_data.csv"
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Get number of classes in y_train.
print("Number of unique classes in y:", len(np.unique(y_train)))
n_class = 3

# Dense layer class.


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize weights & biases.
        Weights should be initialized with values drawn from a normal
        distribution scaled by 0.01.
        Biases are initialized to 0.0.
        """

        # Weights control the shape of the decision boundary if it was 2d.
        # Each edge must have a weight.
        # Weight matrix has dimensions of inputs x outputs for the layer.
        # here the outputs are the number of neurons.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # Bias is like a vertical shift of the decision boundary if it was 2d.
        # Each output should have a bias so this will be a 1d array of 0s.
        # Can achieve this by using np.zeros.
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        """
        A forward pass through the layer to give z.
        Compute it using np.dot(...) and then add the biases.
        """

        self.inputs = inputs

        # z is the multiplication of the input array with the weight matrix plus the biases.
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dz):
        """
        Backward pass
        """
        # Gradients of weights.
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases.
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs.
        self.dinputs = np.dot(dz, self.weights.T)

# Activation functions.


class ReLu:
    """
    ReLu activation
    """

    def forward(self, z):
        """
        Forward pass
        """

        self.z = z

        # Return z if z >= 0, otherwise return 0.
        # Accomplish this with np.where.
        self.activity = np.where(z >= 0, z, 0)

    def backward(self, dactivity):
        """
        Backward pass
        """

        self.dz = dactivity.copy()

        self.dz[self.z <= 0] = 0.0


class Softmax:
    def forward(self, z):
        """
        forward pass
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dprobs):
        """
        backward
        """
        # Empty array
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)

# Loss.


class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Clip to prevent division by 0.
        # Clip both sides to not bias up.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # Negative log likelihoods.
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)

    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes.
        batch_sz, n_class = probs.shape
        # Get the gradient.
        self.dprobs = -oh_y_true / probs
        # normalize the gradient.
        self.dprobs = self.dprobs / batch_sz

# Optimizer.


class SGD:
    """
    Stochastic gradient descent optimizer
    """

    def __init__(self, learning_rate=1.0):
        # Initialize the optimizer with a learning rate.
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights = layer.weights - (self.learning_rate * layer.dweights)
        layer.biases = layer.biases - (self.learning_rate * layer.dbiases)

# Helper functions.


def predictions(probs):
    """
    Convert probabilities to predictions
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds


def accuracy(y_preds, y_true):
    """
    accuracy metric calculation
    """
    return np.mean(y_preds == y_true)

# Before training must create the network and activations.


# First layer takes 3 inputs, has 4 neurons.
dense1 = DenseLayer(3, 4)
# First activation should be relu, we were told output will have softmax, assumed hidden layers will use relu.
activation1 = ReLu()

# Second layer takes 4 inputs, has 8 neurons.
dense2 = DenseLayer(4, 8)
# Second activation will be relu since it is a hidden layer.
activation2 = ReLu()

# Output layer takes 8 inputs, has 3 neurons.
output_layer = DenseLayer(8, 3)
# Final activation will be softmax.
output_activation = Softmax()

# Loss.
crossentropy = CrossEntropyLoss()

# Optimizer.
optimizer = SGD()

# Batch size, cant be bigger than the size of training data.
# Batch size of 16 was found to be best, 32 offers similar potential but is not as consistent.
batch_sz = 16

# Number of batches, calculated based on the batch size wrt the size of training data.
# // to force integer division.
n_batch = X_train.shape[0] // batch_sz

# Number of training epochs.
epochs = 10


# Full forward pass.
def forward_pass(X, y_true, oh_y_true):
    """
    full forward pass through the entire network
    """
    # Forward pass through first layer.
    # First layer takes raw data as input.
    dense1.forward(X)
    # Forward pass through the first activation.
    # Takes the z output from the layer after activation.
    activation1.forward(dense1.z)

    # Forward pass through second layer.
    # Input is activation output from the previous layer.
    dense2.forward(activation1.activity)
    # Foward pass through second activation.
    # Takes z as input from the layer.
    activation2.forward(dense2.z)

    # Forward pass through third layer.
    # Input is activation from previous layer.
    output_layer.forward(activation2.activity)
    # Forward pass through third activation.
    # Takes z as input from current layer and outputs probabilities.
    probs = output_activation.forward(output_layer.z)

    return probs


# One hot encoding.
oh_y_true = np.eye(n_class)[y_train]

# Full backward pass through network.


def backward_pass(probs, y_true, oh_y_true):
    """
    full backward pass through entire network
    """

    # Use the backwards pass of loss to compute dprobs.
    # This is the gradient of the probabilities.
    crossentropy.backward(probs, oh_y_true)

    # Next we need derivative of the activation output from the final layer.
    output_activation.backward(crossentropy.dprobs)
    # This lets us obtain dz for the output layer, can then pass this through a backwards pass of the output layer.

    # Backward pass of output layer.
    output_layer.backward(output_activation.dz)

    # Pass derivative of output layer input to layer 2 activation.
    activation2.backward(output_layer.dinputs)
    # Pass layer 2 dz from activation 2.
    dense2.backward(activation2.dz)

    # Pass derivative of layer 2 input to layer 1 activation.
    activation1.backward(dense2.dinputs)
    # Pass layer 1 dz from activation 1.
    dense1.backward(activation1.dz)


# Train network.

for epoch in range(epochs):

    print('epoch:', epoch)

    # Counter to store the end index of the current batch.
    batch_index = 0

    for batch_i in range(n_batch):

        print("batch:", batch_i)

        # Get current batches.
        X_train_batch = X_train[batch_index:batch_index+batch_sz]
        y_train_batch = y_train[batch_index:batch_index+batch_sz]

        # Update batch index.
        batch_index += batch_sz

        # One hot encoding.
        oh_y_true_batch = np.eye(n_class)[y_train_batch]

        # Forward pass.
        probs = forward_pass(X_train_batch, y_train_batch, oh_y_true_batch)

        # Loss.
        loss = crossentropy.forward(probs, oh_y_true_batch)
        print("loss:", loss)

        # Get predictions.
        preds = predictions(probs)

        # Accuracy.
        acc = accuracy(preds, y_train_batch)
        print("accuracy:", acc)

        # Backward pass.
        backward_pass(probs, y_train_batch, oh_y_true_batch)

        # Update weights.
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(output_layer)


# Test the network.

# One hot encoding of test set.
oh_y_true_test = np.eye(n_class)[y_test]

# Forward pass on test set.
probs_test = forward_pass(X_test, y_test, oh_y_true_test)

# Loss on test set.
loss_test = crossentropy.forward(probs_test, oh_y_true_test)
print("loss on test set:", loss_test)

# Get test predictions.
preds_test = predictions(probs_test)

# Accuracy on test set.
acc_test = accuracy(preds_test, y_test)
print("accuracy on test set:", acc_test)
