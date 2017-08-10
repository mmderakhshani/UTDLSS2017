# Definition of Model
import tensorflow as tf
# RNN for Language Modeling
class RNN_LM():
    def __init__(self, args, vocab_size, generation = False):

        self.device = args.device
        self.cell_type = args.cell_type
        self.state_size = args.state_size
        # In generation mode(Sampling) the network should be run, 1 sample by 1.
        if generation is True:
            self.batch_size = 1
            self.seq_length = 1
        else:
            self.batch_size = args.batch_size
            self.seq_length = args.seq_length
            self.lr = args.lr
            self.label_data = tf.placeholder(tf.int64, [self.batch_size, self.seq_length])
            self.label_data_reshaped = tf.reshape(self.label_data, [-1])

        self.x = tf.placeholder(tf.int64, [self.batch_size, self.seq_length])

        with tf.device(args.device):
            tf.set_random_seed(1)
            self.embedding = tf.get_variable("embedding", [vocab_size, args.embedding_size])
            self.input_data = tf.nn.embedding_lookup(self.embedding, self.x)
            self.rnn_cells = [self.rnn_cell() for _ in range(args.num_layers)]

            if generation is False:
                self.rnn_cells = [tf.contrib.rnn.DropoutWrapper(
                    cell = cell,
                    output_keep_prob = args.keep_prob) for cell in self.rnn_cells]

            self.stacked_rnn = tf.contrib.rnn.MultiRNNCell(self.rnn_cells)

            self.init_state = self.stacked_rnn.zero_state(self.batch_size, tf.float32)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=self.stacked_rnn,
                inputs=self.input_data, initial_state=self.init_state)

            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, args.state_size])

            self.logits = tf.layers.dense(
                inputs=self.rnn_outputs,
                units=vocab_size)
            self.preds = tf.nn.softmax(self.logits)
        self.saver = tf.train.Saver()

    def rnn_cell(self):
        if self.cell_type is 'rnn':
            cell = tf.contrib.rnn.BasicRNNCell(
                num_units=self.state_size,
                activation=tf.nn.relu)
        elif self.cell_type is 'gru':
            cell = tf.contrib.rnn.GRUCell(
                num_units=self.state_size,
                activation=tf.nn.relu)
        elif self.cell_type is 'lstm':
            cell = tf.contrib.rnn.LSTMCell(
                num_units=self.state_size,
                activation=tf.nn.relu,
                state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(
                num_units=self.state_size,
                activation=tf.nn.relu)
        return cell
    def train(self):
        with tf.device(self.device):
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
                                                                  labels=self.label_data_reshaped))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(loss)
        return optimizer, loss
