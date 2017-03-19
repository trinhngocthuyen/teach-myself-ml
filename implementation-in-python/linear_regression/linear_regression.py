import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class LinearRegressionGradientDescent():
    def __init__(self, learning_rate=0.01, n_iterations=1000, should_normalize=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.should_normalize = should_normalize

    def mean_square_error(self, y_predicted, y_actual):
        tmp = y_predicted - y_actual
        return tmp.T.dot(tmp) / (2 * y_actual.shape[0])

    def compute_cost(self, X, y, theta):
        return self.mean_square_error(X.dot(theta), y)

    def fit(self, X, y):
        if self.should_normalize:
            self.meanX = X.mean(axis=0)
            self.stdX = X.std(axis=0)
            X = (X - self.meanX) / self.stdX

        # Prepend column ones
        X = np.insert(X, 0, 1, axis=1)

        n_examples, n_features = X.shape
        self.theta = np.zeros((n_features, 1))
        self.cost_history = np.zeros((self.n_iterations, 1))

        for idx in range(0, self.n_iterations):
            self.theta -= X.T.dot(X.dot(self.theta) - y) * (self.learning_rate / n_examples)
            # Save the cost so that we could make the plot later
            self.cost_history[idx, 0] = self.compute_cost(X, y, self.theta).item()

    def predict(self, X):
        if self.should_normalize:
            X = (X - self.meanX) / self.stdX

        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.theta)


# MARK: - Main
if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=1, bias=100, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # By somehow, `y_train`.shape = (123,). The 2nd param of its shape seems to be unknown,
    # breaking some matrix computations
    # --> Reshape it to the column vector (123, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Training
    print("Training with gradient descent...")
    lr = LinearRegressionGradientDescent()
    lr.fit(X_train, y_train)
    print("Training finished with theta = ", lr.theta)

    # Testing & print the cost
    y_test_predicted = lr.predict(X_test)
    print("Testing finished with cost = ", lr.mean_square_error(y_test, y_test_predicted))

    # Plot the results
    plt.subplot(1, 2, 1)
    plt.title("Correlation between x and y")
    plt.plot(X_train, y_train, 'bx')

    plt.subplot(1, 2, 2)
    plt.title("Cost function by iterations")
    plt.plot(lr.cost_history)
    plt.show()

    plt.title("Testing: predicted vs. actual")
    plt.plot(X_test, y_test, 'bx', X_test, y_test_predicted, 'rx')
    plt.show()
