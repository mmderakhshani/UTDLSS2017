#Sampling Process
import tensorflow as tf
import numpy as np
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
    parser.add_argument('--seq_length', type=int, default=1,
                        help='maximum sequences considered for backprop')
    # Number of Training Epoch
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Learning Rate
    parser.add_argument('--lr', type=int, default=0.01,
                        help='learning rate')
    # Training Device
    parser.add_argument('--device', type=str, default='gpu:0',
                        help='for cpu: \'cpu:0\', for gpu: \'gpu:0\'')
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Size of batches for training')
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='./save/model',
                        help='Name of saved model.')

    args = parser.parse_args()
    generate(args, num_char = 750, first_letter ='A', pick_top_chars = 5)

def generate(args, num_char = 1000, first_letter = 'A', pick_top_chars = None):
    dataset = Textdataset(args.batch_size, args.seq_length)
    RNN_model = RNN_LM(args, dataset.vocab_size, generation=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        RNN_model.saver.restore(sess, args.checkpoint)

        state = None
        current_char = dataset.vocab_to_idx[first_letter]
        chars = [current_char]
        # Generating
        for epoch in range(num_char):
            if state is not None:
                feed_dict = {RNN_model.x: [[current_char]], RNN_model.init_state: state}
            else:
                feed_dict = {RNN_model.x: [[current_char]]}
            preds, state = sess.run([RNN_model.preds, RNN_model.final_state], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(dataset.vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(dataset.vocab_size, 1, p=np.squeeze(preds))[0]
            #current_char = np.argmax(preds)
            chars.append(current_char)
        chars = map(lambda x: dataset.idx_to_vocab[x], chars)
        print("".join(chars))

if __name__ == '__main__':
    main()