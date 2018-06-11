import numpy as np
import typing

from sklearn.neural_network import MLPClassifier
class MLPNetwork:
    def __init__(self, num_features: int, num_classes: int, num_examples: int,
                 num_hidden_neurons: typing.Dict[int, int], reg_lambda: int = 0.01, epsilon: int = 0.01):
        """
        Multi layer perceptron neural network model
        :param num_features: number of features in classified data (size of input layer)
        :param num_classes:  number of classes in data (size of output layer - softmax)
        :param num_examples: size of training data
        :param num_hidden_neurons: number of hidden neurons in each layers: { layer_number : neurons_number }
                                   Hidden layers count starts from 1 to n. First input layer is not counted as hidden
                                   layer.
        :param reg_lambda: regularization parameter
        :param epsilon: epsilon
        """
        self.num_examples = num_examples
        self.num_features = num_features
        self.num_classes = num_classes

        # TODO Add multi layers
        self.num_hidden_neurons = num_hidden_neurons

        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.model = self.init_model()

    def init_model(self):
        # TODO multi layers
        hidden_1_neurons = self.num_hidden_neurons[1]
        np.random.seed(0)
        W1 = np.random.randn(self.num_features, hidden_1_neurons) / np.sqrt(self.num_features)
        b1 = np.zeros((1, hidden_1_neurons))
        W2 = np.random.randn(hidden_1_neurons, self.num_classes) / np.sqrt(hidden_1_neurons)
        b2 = np.zeros((1, self.num_classes))

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def calculate_loss(self, X, y):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    # Helper function to predict an output (0 or 1)
    def predict(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, batches=20000, print_loss=False):
        """

        :param batches:
        :param print_loss:
        :return:
        """

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

        # Gradient descent. For each batch...
        for i in range(0, batches):
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(self.num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration {}: {}".format(i, self.calculate_loss(X, y)))

        return self.model

if __name__ == '__main__':
    pass