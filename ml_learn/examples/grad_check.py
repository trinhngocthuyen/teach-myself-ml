import numpy as np
from random import uniform
from ml_learn.rnn.rnn import RNN
from ml_learn.utils.data_encode import one_hot

# Source (with modification): https://gist.github.com/karpathy/d4dee566867f8291f086

def grad_check(model: RNN, inputs, targets, hprev):
    get_params = lambda dict, pnames: [dict[p] for p in pnames]

    num_checks, delta = 10, 1e-5
    _, grads, _ = model.one_chunk_loss(inputs, targets, hprev)
    [dWxh, dWhh, dWhy, dbh, dby] = get_params(grads, ['Wxh', 'Whh', 'Why', 'bh', 'by'])

    [Wxh, Whh, Why, bh, by] = get_params(model.params, ['Wxh', 'Whh', 'Why', 'bh', 'by'])

    for param, dparam, name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby],
        ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)

        print('----- [%s] -----' % name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _ = model.one_chunk_loss(inputs, targets, hprev)

            param.flat[ri] = old_val - delta
            cg1, _, _ = model.one_chunk_loss(inputs, targets, hprev)

            param.flat[ri] = old_val  # reset old value for this parameter

            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f --> relative error: %.3e ' % (grad_numerical, grad_analytic, rel_error))

# ---------------------
model = RNN(hid_size=100, seq_length=25, vocab_size=62)
inputs = one_hot([i for i in range(25)], 62)
targets = [i+1 for i in range(25)]
h_prev = np.zeros((1, 100))

grad_check(model, inputs, targets, h_prev)
