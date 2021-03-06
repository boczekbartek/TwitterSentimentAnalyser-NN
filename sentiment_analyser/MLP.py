import numpy as np
from tqdm import tqdm
import pickle


class MLPNetwork:
    def __init__(self, n_classes, n_features, n_hidden_units=30,
                 l1=0.0, l2=0.0, epochs=500, learning_rate=0.01,
                 n_batches=1, random_seed=None):

        if random_seed:
            np.random.seed(random_seed)
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.w1, self.w2 = self._init_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_batches = n_batches

    def _init_weights(self):
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden_units * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden_units, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_classes * (self.n_hidden_units + 1))
        w2 = w2.reshape(self.n_classes, self.n_hidden_units + 1)
        return w1, w2

    def _bias(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    def _forward(self, X):
        """ Forward gradient propagation """
        net_input = self._bias(X, how='column')
        net_hidden = self.w1.dot(net_input.T)
        act_hidden = self.sigmoid(net_hidden)
        act_hidden = self._bias(act_hidden, how='row')
        net_out = self.w2.dot(act_hidden)
        act_out = self.sigmoid(net_out)
        return net_input, net_hidden, act_hidden, net_out, act_out

    def _backward(self, net_input, net_hidden, act_hidden, act_out, y):
        """ Backward gradient propagation """
        sigma3 = act_out - y
        net_hidden = self._bias(net_hidden, how='row')
        sigma2 = self.w2.T.dot(sigma3) * self.sigmoid_prime(net_hidden)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(net_input)
        grad2 = sigma3.dot(act_hidden.T)
        return grad1, grad2

    def _error(self, y, output):
        """ Error between network output and reference """
        error = self.cross_entropy(output, y)

        return 0.5 * np.mean(error)

    def _training_step(self, X, y):
        """ One """
        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(X)
        y = y.T

        grad1, grad2 = self._backward(net_input, net_hidden, act_hidden, act_out, y)

        # regularize
        grad1[:, 1:] += (self.w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (self.w2[:, 1:] * (self.l1 + self.l2))

        error = self._error(y, act_out)

        return error, grad1, grad2

    def predict(self, X):
        """ Predict neural network output """
        Xt = X.copy()
        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(Xt)
        return self.softmax(act_out.T)

    def fit(self, X, y):
        """ Fit training data to given labels, train model. """
        self.error_ = []
        X_data, y_data = X.copy(), y.copy()

        y_data_enc = y
        for i in tqdm(range(self.epochs)):

            X_mb = np.array_split(X_data, self.n_batches)
            y_mb = np.array_split(y_data_enc, self.n_batches)

            epoch_errors = []

            for Xi, yi in zip(X_mb, y_mb):
                # update weights
                error, grad1, grad2 = self._training_step(Xi, yi)
                epoch_errors.append(error)
                self.w1 -= (self.learning_rate * grad1)
                self.w2 -= (self.learning_rate * grad2)

            # remember mean or errors
            self.error_.append(np.mean(epoch_errors))

    def evaluate(self, X, y):
        matching = 0
        for pred, real in zip(self.predict(X), y):
            if np.argmax(pred) == np.argmax(real):
                matching += 1
        return float(matching) / len(y)

    @staticmethod
    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(targets * np.log(predictions + 1e-9))) / N
        return ce

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return MLPNetwork.sigmoid(x) * (1 - MLPNetwork.sigmoid(x))

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def save(self, path):
        """ Save model"""
        with open(path, "wb") as f:
            pickle.dump(self, f)