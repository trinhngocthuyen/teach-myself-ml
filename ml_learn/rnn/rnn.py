from ..utils.data_encode import one_hot
from ..layers import *

class RNN:
    def __init__(self, hid_size, seq_length, vocab_size):
        self.hid_size = hid_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        weight_scale = 1e-2
        self.params = {
            'Wxh':  np.random.normal(0, weight_scale, (vocab_size, hid_size)),
            'Whh':  np.random.normal(0, weight_scale, (hid_size, hid_size)),
            'bh':   np.zeros(hid_size),
            'Why':  np.random.normal(0, weight_scale, (hid_size, vocab_size)),
            'by':   np.zeros(vocab_size),
        }

    def one_chunk_forward(self, X, h_prev=None):
        """
        Forward pass
            - X: one-hot input. X.shape[0] <= seq_length
            - h_prev: hidden state calculated from the previous chunk. If none, it will be set to zeros
        Returns
            - scores: unnormalized propability
            - h: the last hidden state of this chunk
            - caches: caches for backward pass, by timesteps
        """
        Wxh, Whh, bh = self.params['Wxh'], self.params['Whh'], self.params['bh']
        Why, by = self.params['Why'], self.params['by']

        # -----------------------------
        scores, caches = [], []
        h = h_prev if h_prev is not None else np.zeros((1, self.hid_size))
        for t in range(X.shape[0]):
            h, cache_h = rnn_fc_hid_forward(X[t].reshape((1, -1)), h, Wxh, Whh, bh)
            out, cache_out = rnn_fc_out_forward(h, Why, by)
            caches.append((cache_h, cache_out))
            scores.append(out)
        return scores, h, caches

    def one_chunk_backward(self, douts, caches):
        """
        Backward pass
            - douts: upper gradients, by timesteps
            - caches: caches from forward pass, by timesteps
        Returns
            - dWhh, dbh, dWxh, dWhy, dby
        """
        zeros_like_param = lambda name: np.zeros_like(self.params[name])

        dWxh, dWhh, dbh = zeros_like_param('Wxh'), zeros_like_param('Whh'), zeros_like_param('bh')
        dWhy, dby = zeros_like_param('Why'), zeros_like_param('by')

        dh_next = np.zeros((1, self.hid_size))
        for t in range(len(douts))[::-1]:
            (cache_h, cache_out) = caches[t]
            dh, dWhy_t, dby_t = rnn_fc_out_backward(douts[t], cache_out)
            dh += dh_next
            dWxh_t, dWhh_t, dbh_t, dh_next = rnn_fc_hid_backward(dh, cache_h)
            for dtotal, dpartial in zip([dWxh, dWhh, dbh, dWhy, dby],
                                        [dWxh_t, dWhh_t, dbh_t, dWhy_t, dby_t]):
                dtotal += dpartial
        return dWhh, dbh, dWxh, dWhy, dby

    def one_chunk_loss(self, X, y, h_prev=None):
        """
        Loss function. (X, y) is a chunk of data (by seq_length).
        Reason for the requirement is that this function performs a huge amount of cache for backward pass.
        Therefore, the number of timesteps for a loss computation should be restricted.
            - X: one-hot input. X.shape[0] <= seq_length
            - y: raw labels
            - h_prev: hidden state calculated from the previous chunk. If none, it will be set to zeros
        Returns
            - mean_loss, grads
            - h_prev: the last hidden state of this chunk, used for the next chunk
        """
        # Architecture:
        #   a(t) = Whh . h(t-1) + bh + Wxh * x
        #   h(t) = tanh(a(t))
        #   o(t) = Why . h(t) + by
        #   y(t) = softmax(o(t))

        total_loss, douts = 0, {}

        # Forward
        scores, h_prev, caches = self.one_chunk_forward(X, h_prev=h_prev)
        actual_seq_length = len(scores)
        for t in range(actual_seq_length):
            loss_t, douts[t] = softmax_loss(scores[t], y[t])
            total_loss += loss_t

        mean_loss = total_loss / actual_seq_length

        # Backward
        dWhh, dbh, dWxh, dWhy, dby = self.one_chunk_backward(douts, caches)

        # Clipping to mitigate exploding grads
        for dparam in [dWhh, dbh, dWxh, dWhy, dby]:
            dparam /= actual_seq_length
            np.clip(dparam, -5, 5, out=dparam)

        # Update params
        grads = {
            'Wxh':  dWxh,
            'Whh':  dWhh,
            'bh':   dbh,
            'Why':  dWhy,
            'by':   dby,
        }

        return mean_loss, grads, h_prev # Also return the last hidden state (used for the next run if necessary)

    def predict(self, X):
        """
        Returns the prediction (raw labels) for X
            - X: one-hot input. X could be of variable length.
        """
        chunk_size = self.seq_length
        n_chunks = X.shape[0] // chunk_size

        h_prev = None
        scores = []
        for c in range(n_chunks):
            X_c = X[c * chunk_size:(c + 1) * chunk_size]
            scores_c, _, h_prev = self.one_chunk_forward(X_c, h_prev=h_prev)
            scores += scores_c
        return np.argmax(scores, axis=1)

    def sample(self, seed_idx, size):
        """
        Sampling outputs by timesteps.
            - seed_idx: index at the start of timesteps
            - size: length of timesteps
        Returns
            - labels: raw labels (list-like)
        """
        h_prev = None
        idx = seed_idx
        y_pred = [idx]

        for t in range(size):
            X = one_hot([idx], self.vocab_size) # X only has 1 elements, no need to split it into chunks
            scores, h_prev, _ = self.one_chunk_forward(X, h_prev=h_prev)
            idx = np.argmax(scores)
            y_pred.append(idx)
        return y_pred
