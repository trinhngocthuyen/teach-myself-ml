import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import scipy.optimize as op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticClassification():

    def __init__(self):
        self.theta = np.zeros(1)
        self.cost_history = np.zeros(1)

    @staticmethod
    def compute_cost(X, y, theta, regularization_param=0):
        # Workaround for fit_using_minimize_function()
        # When using op.minimize(), theta.shape is somehow not correct
        theta = theta.reshape(-1, 1)

        n_examples = X.shape[0]
        hx = sigmoid(X.dot(theta))
        theta_first_is_zero = theta.copy()
        theta_first_is_zero[0][0] = 0

        # 1/m * [-y' * log(hx) - (1-y)' * log(1-hx)]
        return 1.0 / n_examples * (
            -y.T.dot(np.log(hx)) - (1 - y).T.dot(np.log(1 - hx))
            + regularization_param * theta_first_is_zero.T.dot(theta_first_is_zero)
        )

    def fit_using_minimize_function(self, X, y, regularization_param=0):
        # Prepend column ones
        X = np.insert(X, 0, 1, axis=1)

        def compute_cost_givenXy(theta):
            return self.compute_cost(X, y, theta, regularization_param)

        n_examples, n_features = X.shape
        initial_theta = np.zeros((n_features, 1))     # Initial guess

        optimization_result = op.minimize(compute_cost_givenXy, initial_theta, options={
            'maxiter': 400,     # Hardcode for now
            'disp': False
        })

        if not optimization_result.success:
            raise Exception('Fail to find the optimal of cost function')

        self.theta = optimization_result.x.T.reshape(-1, 1)    # The result returned from optimize() is a row vector

    def fit_using_gradient_descent(self, X, y, learning_rate, n_iterations, regularization_param=0):
        # Prepend column ones
        X = np.insert(X, 0, 1, axis=1)

        n_examples, n_features = X.shape

        self.theta = np.zeros((n_features, 1))
        self.cost_history = np.zeros((n_iterations, 1))

        theta_first_is_zero = self.theta.copy()
        theta_first_is_zero[0][0] = 0

        for idx in range(0, n_iterations):
            hx = sigmoid(X.dot(self.theta))
            self.theta -= (learning_rate / n_examples) * (X.T.dot(hx - y) + regularization_param * theta_first_is_zero)
            self.cost_history[idx, 0] = self.compute_cost(X, y, self.theta).item()

    def classify(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.round(sigmoid(X.dot(self.theta)))

    def get_decision_boundary(self, X):
        if self.theta.shape[0] <= 3:
            boundary_x = np.array([[X.min()], [X.max()]])
            boundary_y = (-1/self.theta[2]) * (self.theta[1] * boundary_x + self.theta[0])
            return boundary_x, boundary_y
        else:
            # TODO: Use contour plot
            print('Not yet implement decision boundary in which length(theta > 3')


if __name__ == '__main__':
    X, y = make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    plt.title('Training data')
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)
    plt.show()

    clf = LogisticClassification()

    # Training
    try:
        # clf.fit_using_minimize_function(X_train, y_train, regularization_param=1)
        clf.fit_using_gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=1000, regularization_param=0)
        print('Training finished. Found theta = ', clf.theta)
    except Exception as error:
        print('Fail to fit the training data. error: ', error)

    # Testing
    y_test_predicted = clf.classify(X_test)

    accuracy = np.mean(np.double(y_test_predicted == y_test))
    print('Accuracy = ', accuracy)

    # Plotting results
    plt.title('Testing results')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test, label='Actual')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='+', c=y_test_predicted, label='Predicted')

    decision_boundary_x, decision_boundary_y = clf.get_decision_boundary(X_test)
    plt.plot(decision_boundary_x, decision_boundary_y)
    plt.legend()
    plt.show()

