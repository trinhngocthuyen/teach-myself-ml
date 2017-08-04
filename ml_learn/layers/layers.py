import numpy as np

def rnn_fc_hid_forward(x, h_prev, Wxh, Whh, bh):
    a = np.dot(h_prev, Whh) + bh + np.dot(x, Wxh)
    h = np.tanh(a)
    cache = Wxh, Whh, bh, x, h_prev, h
    return h, cache

def rnn_fc_hid_backward(dh, cache):
    Wxh, Whh, bh, x, h_prev, h = cache

    da = dh * (1 - h ** 2)  # gradient of tanh
    dWhh = np.dot(h_prev.T, da)
    dWxh = np.dot(x.T, da)
    dbh = np.sum(da, axis=0)
    dh = np.dot(da, Whh.T)
    return dWxh, dWhh, dbh, dh # dx is useless in this case

def rnn_fc_out_forward(h, Why, by):
    out = np.dot(h, Why) + by
    cache = Why, by, h
    return out, cache

def rnn_fc_out_backward(dout, cache):
    Why, by, h = cache
    dh = np.dot(dout, Why.T)
    dWhy = np.dot(h.T, dout)
    dby = np.sum(dout, axis=0)
    return dh, dWhy, dby

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx