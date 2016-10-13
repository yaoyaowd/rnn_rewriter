import math
import os

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model


LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE = 64
SIZE = 512
NUM_LAYERS = 2
EN_VOCAB_SIZE = 41
STEPS_PER_CHECKPOINT = 100
MAX_INPUT_LENGTH = 50
MODEL_LENGTH = 52
DATA_PATH = "/Users/dong/Downloads/navboost/"


def prepare_spell_correction(input_file, vocabulary):
    data_set = []
    line_count = 0
    with open(input_file) as f:
        for line in f.readlines():
            items = line.strip().lower().split('\t')
            typo_ids = data_utils.string_to_token_ids(items[0], vocabulary)
            correction_ids = data_utils.string_to_token_ids(items[1], vocabulary)
            correction_ids.append(data_utils.EOS_ID)
            if (data_utils.UNK_ID) in typo_ids \
                    or (len(typo_ids) > MAX_INPUT_LENGTH) \
                    or (len(correction_ids) > MODEL_LENGTH):
                continue
            data_set.append([typo_ids, correction_ids])
            line_count += 1
            if line_count % 10000 == 0:
                print "load lines", line_count
    print "Data:", input_file, "contains", line_count, "examples"
    return data_set


def create_model(session, predict_only):
    model = seq2seq_model.Seq2SeqModel(EN_VOCAB_SIZE, EN_VOCAB_SIZE,
                                       SIZE, NUM_LAYERS, MAX_GRADIENT_NORM, MODEL_LENGTH,
                                       BATCH_SIZE, LEARNING_RATE, LEARNING_RATE_DECAY_FACTOR,
                                       predict_only=predict_only)
    ckpt = tf.train.get_checkpoint_state(DATA_PATH)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "created model with fresh parameters"
        session.run(tf.initialize_all_variables())
    return model


def train(train_set, test_set, vocabulary):
    with tf.Session() as session:
        model = create_model(session, False)

        loss = 0.0
        current_step = 0
        previous_losses = []
        while True:
            current_step += 1
            encoder_inputs, decoder_inputs, target_weights = data_utils.get_batch(train_set, BATCH_SIZE, MODEL_LENGTH)

            _, step_loss, _ = model.step(
                session, encoder_inputs, decoder_inputs, target_weights, False)
            loss += step_loss / STEPS_PER_CHECKPOINT

            if current_step % STEPS_PER_CHECKPOINT == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print "global step %d learning rate %.4f perpexity %.2f" % (
                    current_step, model.learning_rate.eval(), perplexity)
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    session.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                loss = 0.0

                checkpoint_path = os.path.join(DATA_PATH, "spell_correction.ckpt")
                model.saver.save(session, checkpoint_path, global_step=model.global_step)

                test(session, model, vocabulary, test_set)


def visualize(rev_vocabulary, inputs, should_reverse=False):
    if isinstance(inputs, list):
        output_matrix = np.empty((len(inputs), BATCH_SIZE))
        for i in xrange(len(inputs)):
            for j in xrange(len(inputs[i])):
                output_matrix[i][j] = inputs[i][j]
        inputs = output_matrix

    inputs = inputs.transpose()
    ret = []
    for input in inputs:
        str = ""
        for c in input:
            if c == data_utils.EOS_ID:
                break
            str = str + rev_vocabulary.get(c)
        if should_reverse:
            str = str[::-1]
        ret.append(str)
    return ret


def test(sess, model, vocabulary, test_set):
    rev_vocabulary = {v: k for k, v in vocabulary.items()}
    rev_vocabulary[data_utils.PAD_ID] = ''
    rev_vocabulary[data_utils.GO_ID] = ''
    rev_vocabulary[data_utils.UNK_ID] = ''

    encoder_inputs, decoder_inputs, target_weights = data_utils.get_batch(test_set, BATCH_SIZE, MODEL_LENGTH)
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, True)

    output_matrix = np.empty((len(output_logits), BATCH_SIZE))
    for lenIdx in xrange(len(output_logits)):
        output_matrix[lenIdx] = np.array([int(np.argmax(logit)) for logit in output_logits[lenIdx]])

    typos = visualize(rev_vocabulary, encoder_inputs, should_reverse=True)
    rewrites = visualize(rev_vocabulary, decoder_inputs)
    guesses = visualize(rev_vocabulary, output_matrix)
    total = 0
    correct = 0
    for i in xrange(len(typos)):
        total += 1
        correct += 1 if rewrites[i] == guesses[i] else 0
        print typos[i], ' - ', rewrites[i], ' - ', guesses[i]
    print 'total: ', total, ' corrected: ', correct, ' acc: ', correct / (total + 0.0)


def main(_):
    vocabulary = data_utils.initialize_a_to_z()
    train_data = prepare_spell_correction(DATA_PATH + "train.txt", vocabulary)
    test_data = prepare_spell_correction(DATA_PATH + "test.txt", vocabulary)
    train(train_data, test_data, vocabulary)


if __name__ == "__main__":
    tf.app.run()
