# Training Process
import tensorflow as tf
import numpy as np
import time
import argparse
from RNN_TF import RNN_LM
from utils import Textdataset


def main():
    parser = argparse.ArgumentParser()
    # Number of Layers
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of stacked RNN layers')
    # Cell Type
    parser.add_argument('--cell_type', type=str, default='gru',
                        help='rnn, lstm, gru')
    # State Size
    parser.add_argument('--state_size', type=int, default=100,
                        help='Number of hidden neurons of RNN cells')
    #Embedding Size
    parser.add_argument('--embedding_size', type=int, default=10,
                        help='learning rate')
    # 1-Drop out
    parser.add_argument('--keep_prob', type=int, default=0.9,
                        help='keeping probability(1-dropout)')
    # Length of Unrolled RNN (Sequence Length)
    parser.add_argument('--seq_length', type=int, default=200,
                        help='maximum sequences considered for backprop')
    # Number of Training Epoch
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    # Learning Rate
    parser.add_argument('--lr', type=int, default=0.01,
                        help='learning rate')
    # Training Device
    parser.add_argument('--device', type=str, default='gpu:0',
                        help='for cpu: \'cpu:0\', for gpu: \'gpu:0\'')
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of batches for training')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='./save/model_1',
                        help='Name of saved model.')

    args = parser.parse_args()
    train(args)

def train(args):
    dataset = Textdataset(args.batch_size, args.seq_length)
    RNN_model = RNN_LM(args, dataset.vocab_size)
    optimizer, loss = RNN_model.train()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        start_process = time.time()
        for epoch in range(args.num_epochs):
            start_epoch = time.time()
            avg_loss = 0
            total_batches = dataset.total_batches
            # Loop over batches
            for i in range(total_batches):
                batch_x, batch_y = dataset.next_batch()
                _, batch_loss = \
                    sess.run([optimizer, loss],
                             feed_dict={RNN_model.x: batch_x,RNN_model.label_data: batch_y})

                avg_loss += batch_loss/total_batches
            end_epoch = time.time()
            print("Epoch:", epoch+1, "Train Loss:",avg_loss,
                  "in:", int(end_epoch - start_epoch), "sec")
            RNN_model.saver.save(sess, args.checkpoint)
        end_process = time.time()
        print("Train completed in:",
        int(end_process - start_process), "sec")


if __name__ == '__main__':
    main()