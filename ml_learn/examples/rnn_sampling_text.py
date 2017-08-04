import numpy as np
import matplotlib.pyplot as plt
from ml_learn.rnn.rnn import RNN
from ml_learn.rnn.trainer import RNNTrainer
from ml_learn.utils.data_encode import one_hot

class DataWrapper:
    def __init__(self, fname):
        self.fname = fname

        raw_data = open(fname, 'r').read().strip('\n')
        chars = list(set(raw_data))

        print('[DataWrapper] raw_data size: %d. unique: %d' % (len(raw_data), len(chars)))

        char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
        data_idxs = [char_to_idx[ch] for ch in raw_data]

        # Store data
        self.raw_data = raw_data
        self.vocab = chars
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.data_idxs = data_idxs

    def data(self, data_size, one_hot_on_X=True):
        # Note: since y = next(x), X should exclude the last element in data_idxs
        # (b/c what's following the last one?)
        idxs = np.array(self.data_idxs[:data_size + 1])
        X = idxs[:-1]
        y = idxs[1:]
        if one_hot_on_X:
            X = one_hot(X, len(self.vocab))
        return {
            'X_train': X,
            'y_train': y
        }

    def translate(self, idxs):
        chars = [self.idx_to_char[idx] for idx in idxs]
        return ''.join(chars)

if __name__ == "__main__":
    data_wrapper = DataWrapper('data/shakespear.txt')
    seq_length = 25
    data_size = 10000
    # data_size = len(data_wrapper.data_idxs) // seq_length * seq_length
    # data_size = len(data_wrapper.data_idxs)
    data = data_wrapper.data(data_size=data_size)

    optim_config = {
        'learning_rate': 1e-3,
        'decay_rate': 0.99,
    }

    model = RNN(hid_size=100, seq_length=seq_length, vocab_size=len(data_wrapper.vocab))
    solver = RNNTrainer(model, update_rule='rmsprop', optim_config=optim_config)

    def log(file, message):
        print(message)
        file.write(message)

    def monitoring(running_stats):
        count, iter, loss = running_stats['count'], running_stats['iter'], running_stats['loss']
        if count % 200 == 0:
            print('Loss at iter %d, count %d: %.3f' % (iter, count, loss))

            seed_idx = data_wrapper.char_to_idx['t']
            samples_idxs = model.sample(seed_idx=seed_idx, size=500)
            samples = data_wrapper.translate(samples_idxs)

            f = open('dummy_output/samples_out_%d.txt' % (count % 1000), 'w', encoding='utf-8')
            log(f, '----- iter %d, count %d -----\n' % (iter, count))
            log(f, samples)
            f.close()


    stats = solver.train(data=data, n_iterations=100, monitoring=monitoring)

    plt.plot(stats['loss_history'], color='green')
    plt.show()
