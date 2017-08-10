# Utility Functions for Importing Dataset
import numpy as np
import os
import urllib.request

class Textdataset:

    def __init__(self, batch_size, seq_length, shuffled = False):
        self.seq_length = seq_length
        self.batch_size = batch_size
        # Downloading Dataset
        self.file_url = "https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt"
        self.file_name = 'tinyshakespeare.txt'
        if not os.path.exists(self.file_name):
            urllib.request.urlretrieve(self.file_url, self.file_name)
        # Reading Raw Data
        with open(self.file_name, 'r') as f:
            raw_data = f.read()
            print('Data length:', len(raw_data))
        data_len = len(raw_data)
        self.batch_len = data_len//batch_size
        # Listing Char Vocabulary
        self.vocab = sorted(set(raw_data))
        print('Vocabulary:', self.vocab)
        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        self.batch_itr = 0
        data = [self.vocab_to_idx[c] for c in raw_data]
        del raw_data
        # Devide Data into batches
        self.batch_data = np.zeros([batch_size, self.batch_len], dtype=np.int32)
        for i in range(batch_size):
            self.batch_data[i] = data[(self.batch_len * i):(self.batch_len * (i + 1))]
        del data
        self.total_batches = (self.batch_len - 1) // self.seq_length
        if self.total_batches <= 1:
            raise ValueError("total_batches <= 1, decrease batch_size or seq_length")
        # Shuffle the order of batches
        self.batch_idx = np.arange(self.total_batches - 1)
        if shuffled is True:
            np.random.shuffle(np.arange(self.total_batches-1))

    def next_batch(self):
        x_one_hot = np.zeros([self.batch_size,self.seq_length, self.vocab_size])
        y_one_hot = np.zeros([self.batch_size,self.seq_length, self.vocab_size])
        if(self.batch_itr>=self.total_batches-1):
            self.batch_itr = 0
        x = self.batch_data[:,
            self.batch_idx[self.batch_itr]*self.seq_length:
            (self.batch_idx[self.batch_itr]+1) * self.seq_length]
        y =self.batch_data[:,
            self.batch_idx[self.batch_itr]*self.seq_length+1:
            (self.batch_idx[self.batch_itr]+1) * self.seq_length+1]
        self.batch_itr = self.batch_itr + 1
        for i in range(self.batch_size):
            x_one_hot[i, np.arange(self.seq_length), x[i]] = 1
            y_one_hot[i, np.arange(self.seq_length), y[i]] = 1

        return x, y
