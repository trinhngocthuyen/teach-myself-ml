import numpy as np
from .rnn import RNN

def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    learning_rate = config.get('learning_rate')
    decay_rate = config.get('decay_rate')
    eps = config.get('epsilon')
    cache = config.get('cache')

    cache = decay_rate * cache + (1 - decay_rate) * (dx ** 2)
    next_x = x - learning_rate * dx / (np.sqrt(cache) + eps)
    config['cache'] = cache
    return next_x, config

def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)


    learning_rate = config.get('learning_rate')
    beta1 = config.get('beta1')
    beta2 = config.get('beta2')
    eps = config.get('epsilon')
    m = config.get('m')
    v = config.get('v')
    t = config.get('t') + 1

    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    mb = m / (1 - beta1 ** t)
    vb = v / (1 - beta2 ** t)
    next_x = x - learning_rate * mb / (np.sqrt(vb) + eps)

    config['m'] = m
    config['v'] = v

    return next_x, config

class RNNTrainer:
    def __init__(self, model: RNN, update_rule, optim_config):
        self.model = model
        self.optim_config = optim_config

        if update_rule == 'adam':
            self.update_func = adam
        elif update_rule == 'rmsprop':
            self.update_func = rmsprop
        else:
            raise Exception('update_rule not valid. Available: Adam, RMSProp')

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.loss_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs_by_params = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs_by_params[p] = d

    def update_params(self, grads):
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs_by_params[p]
            next_w, next_config = self.update_func(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs_by_params[p] = next_config

    def train(self, data, n_iterations=100, monitoring=None):
        X, y = data['X_train'], data['y_train']
        chunk_size = self.model.seq_length
        n_chunks = X.shape[0] // chunk_size

        count = 0

        for i in range(n_iterations):
            h_prev = None
            for k in range(n_chunks):

                X_chunk = X[k * chunk_size:(k + 1) * chunk_size]
                y_chunk = y[k * chunk_size:(k + 1) * chunk_size]

                loss, grads, h_prev = self.model.one_chunk_loss(X_chunk, y_chunk, h_prev)
                self.loss_history.append(loss)

                self.update_params(grads)

                if monitoring is not None:
                    running_stats = {'count': count, 'iter': i, 'loss': loss}
                    monitoring(running_stats)

                count += 1


        return {
            'loss_history': self.loss_history,
        }