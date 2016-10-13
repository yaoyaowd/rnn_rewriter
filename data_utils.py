import random
import numpy as np


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def initialize_a_to_z():
    word_to_id = dict()
    word_to_id[_PAD] = PAD_ID
    word_to_id[_GO] = GO_ID
    word_to_id[_EOS] = EOS_ID
    word_to_id[_UNK] = UNK_ID
    id = UNK_ID
    for x in "abcdefghijklmnopqustuvwxyz 0123456789":
        id += 1
        word_to_id[x] = id
    return word_to_id


def string_to_token_ids(str, vocabulary):
    return [vocabulary.get(c, UNK_ID) for c in str]


def get_batch(data, batch_size, model_length):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    :param data: a list of examples.
    :param batch_size: the sample batch size.
    :param model_length: the model size.
    :return:
    """
    encoder_inputs, decoder_inputs = [], []
    # Encoder inputs, decoder inputs are of format [[input1, input2]], [[output1], [output2]]
    for _ in xrange(batch_size):
        encoder_input, decoder_input = random.choice(data)
        encoder_pad = [PAD_ID] * (model_length - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        decoder_pad_size = model_length - len(decoder_input) - 1
        decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)

    # Transform input, output to batches
    # input -> [[input1 token1, input2 token1, ...], [input1 token2, input2 token2, ...], ...]
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    for length_idx in xrange(model_length):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

    for length_idx in xrange(model_length):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            if length_idx < model_length - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == model_length - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
