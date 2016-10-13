import numpy as np
import tensorflow as tf


ENCODER = "encoder{0}"
DECODER = "decoder{0}"
WEIGHT = "weight{0}"


class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size,
                 size, num_layers, max_gradient_norm, encoder_size,
                 batch_size, learning_rate, learning_rate_decay_factor,
                 predict_only=False):
        """
        Create the sequence to sequence model
        :param source_vocab_size: the vocabulary size for the input.
        :param target_vocab_size: the vocabulary size for the output.
        :param size: the size of each layer and word embedding.
        :param num_layers: the number of hidden layers for each cell.
        :param max_gradient_norm: the maximal gradient norm.
        :param batch_size: the batch size for each iteration.
        :param learning_rate: the learning rate.
        :param learning_rate_decay_factor: the learning rate decay factor.
        :param predict_only: predict only mode.
        """
        self.global_step = tf.Variable(0, trainable=False)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.encoder_size = encoder_size

        single_cell = tf.nn.rnn_cell.GRUCell(size)
        cell = single_cell if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # Feeds for inputs
        self.encoder_inputs = [
            tf.placeholder(tf.int32, shape=[None], name=ENCODER.format(i)) for i in xrange(self.encoder_size)]
        self.decoder_inputs = [
            tf.placeholder(tf.int32, shape=[None], name=DECODER.format(i)) for i in xrange(self.encoder_size + 1)]
        self.target_weights = [
            tf.placeholder(tf.float32, shape=[None], name=WEIGHT.format(i)) for i in xrange(self.encoder_size + 1)]
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Construct embedding attention seq2seq model.
        if predict_only:
            self.output, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs[:self.encoder_size], cell,
                num_encoder_symbols=source_vocab_size, num_decoder_symbols=target_vocab_size,
                embedding_size=size, feed_previous=True)
            self.loss = tf.nn.seq2seq.sequence_loss(
                self.output, targets[:self.encoder_size], self.target_weights[: self.encoder_size])
        else:
            self.output, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs[:self.encoder_size], cell,
                num_encoder_symbols=source_vocab_size, num_decoder_symbols=target_vocab_size,
                embedding_size=size, feed_previous=False)
            self.loss = tf.nn.seq2seq.sequence_loss(
                self.output, targets[:self.encoder_size], self.target_weights[: self.encoder_size])

        # Calculate the gradients and apply it
        params = tf.trainable_variables()
        if not predict_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, predict_only):
        """
        Run a step of the model feeding the given inputs
        :param session: the tensorflow session.
        :param encoder_inputs: the encoder inputs.
        :param decoder_inputs: the decoder inputs.
        :param target_weights: the target weights
        :param predict_only: predict only mode.
        :return:
        """
        input_feed = {}
        for l in xrange(self.encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(self.encoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[self.encoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if predict_only:
            output_feed = [self.loss]
            for l in xrange(self.encoder_size):
                output_feed.append(self.output[l])
        else:
            output_feed = [self.update, self.gradient_norm, self.loss]

        outputs = session.run(output_feed, input_feed)
        if predict_only:
            return None, outputs[0], outputs[1:]
        else:
            return outputs[1], outputs[2], None # Gradient norm, loss, no outputs
