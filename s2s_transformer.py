import numpy as np
import random
import keras

import sys
import os

from keras.utils import set_random_seed
from keras.utils import to_categorical
from keras import backend as K

from metrics_evaluation import metrics as m
from data_file import Data
from data_preparation import *

new = 0

batch_size = 1  # 256
epochs = 20
repeat = 0

print("starting transform2seq")

model_file_name = "models/transform_smol_delete"
history_dict = model_file_name + '_HistoryDict'
print(model_file_name)

h = 2          # Number of self-attention heads
d_k = 32       # Dimensionality of the linearly projected queries and keys
d_v = 32       # Dimensionality of the linearly projected values
d_ff = 512    # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 2          # Number of layers in the encoder stack
params = h, d_k, d_v, d_ff, d_model, n

a = random.randrange(0, 2**32 - 1)
a = 12612638
set_random_seed(a)
print("seed = ", a)


from model_file_2 import model_func
from model_file_2 import *  # for loading


def load_model_mine(model_name):
    custom_objects = {
        'EncoderLayer': EncoderLayer,
        'Encoder': Encoder,
        'DecoderLayer': DecoderLayer,
        'Decoder': Decoder,
        'TransformerModel': TransformerModel,
        'MultiHeadAttention': MultiHeadAttention,
        'PositionEmbeddingFixedWeights': PositionEmbeddingFixedWeights,
        'AddNormalization': AddNormalization,
        'FeedForward': FeedForward
    }
    return keras.models.load_model(model_name, custom_objects=custom_objects)

def save_model_info(model_name, ):
    pass

# ---------------------------- DATA PROCESSING -------------------------------------------------
source, target, val_source, val_target = prepare_data()

# --------------------------------- MODEL ---------------------------------------------------------------------------
old_dict = get_history_dict(history_dict, new)
print("model starting...")
if new:
    print("CREATING A NEW MODEL")
    model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen, params)
else:
    print("LOADING A MODEL")
    model = load_model_mine(model_file_name)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
print()

# exit()
# --------------------------------- TRAINING ------------------------------------------------------------------------
print("training")
for i in range(repeat):
    history = model.fit(
        (source.padded, target.padded), target.padded_shift_one,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one))

    model.save(model_file_name)

    new_dict = join_dicts(old_dict, history.history)
    old_dict = new_dict
    with open(history_dict, 'wb') as file_pi:
        pickle.dump(new_dict, file_pi)

    K.clear_session()
print()




# ---------------------------------- TESTING ------------------------------------------------------------------------
def model_test_old(self, sample, valid_shift, valid, model_name):  # input = padded array of tokens
    model = load_model_mine(model_name)
    sample_len = len(sample)
    value = model.predict((sample, valid))  # has to be in the shape of the input for it to predict

    dict_chars = self.dict_chars
    rev_dict = self.create_reverse_dict(dict_chars)
    assert sample_len == len(valid_shift)

    value_one = np.zeros_like(value)
    valid_one = np.zeros_like(value)  # has to be value
    for i in range(sample_len):
        for j in range(len(value[i])):
            # input one-hot-ization
            token1 = int(valid_shift[i][j])
            valid_one[i][j][token1] = 1
            # output tokenization
            token2 = self.array_to_token(value[i][j])
            value_one[i][j][token2] = 1
            # print(rev_dict[token1], "/",  rev_dict[token2], end=' ')
            print(rev_dict[token2], end=' ')  # the translation part
        print()

    value_tokens = self.one_hot_to_token(value_one)

    # SOME STATISTICS
    num_sent = len(value)
    sent_len = len(value[0])
    embed = len(value[0][0])
    val_all = 0
    for i in range(num_sent):
        # print("prediction:", self.one_hot_to_token([value[i]]))
        # print("true value:", self.one_hot_to_token([valid_one[i]]))
        val = 0
        for j in range(sent_len):
            for k in range(embed):
                val += abs(value_one[i][j][k] - valid_one[i][j][k])
        # print("difference:", val, "accuracy:", 1-(val/sent_len))
        val_all += val
    print("accuracy all:", round(1-(val_all/(sent_len*num_sent)), 2))  # formating na dve desetina mista
    print("f1 prec rec :", m.f1_precision_recall(target, value_tokens, valid_shift))
    # TODO self or target??
def model_test_new(encoder, decoder, x_test_pad, y_test_pad, rev_dict):
    decoder_output_all = []

    x_sent_len = x_test_pad[0].size
    y_sent_len = y_test_pad[0].size
    x_test_pad = x_test_pad.reshape(samples, 1, x_sent_len)  # reshape so encoder takes just one sentence
    y_test_pad = y_test_pad.reshape(samples, 1, y_sent_len)  # and is not angry about dimensions
    # print("y_test_pad_shape trans", y_test_pad.shape)
    print("printing stopped")
    # ------ stop printing --------
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    for x in range(len(y_test_pad)):  # for veta in test data len(y_test_pad)
        # ENCODER
        encoder_output = encoder.predict(x_test_pad[x])  # get encoding for first sentence
        # print("encoder dims:", len(encoder_output), len(encoder_output[0]))

        # DECODER
        decoder_output = []
        letter = np.array([[1]])  # the <bos> token, should be shape (1,1)
        decoder_output_throughts = encoder_state_transform(encoder_output)

        for i in range(len(y_test_pad[x][0])):  # x-ta veta ma shape (1, neco), proto [0]
            decoder_output_word = decoder.predict([letter] + decoder_output_throughts)

            decoder_output_throughts = decoder_output_word[1:]
            decoder_output_word = decoder_output_word[0]  # select just the content
            decoder_output_word = decoder_output_word[0][0]  # first sentence first word

            token = test_y.array_to_token(decoder_output_word)

            letter = np.array([[token]])
            decoder_output.append(int(token))

        decoder_output_all.append(decoder_output)

    # -------- start printing ----------
    sys.stdout = old_stdout

    # SOME STUFF AS IN CLASS
    valid = y_test_pad  # = y_test_pad_shape trans (samples_len, 1, 90)
    predicted = decoder_output_all
    predicted = np.array(predicted)

    # print("decoder output sent, num:", len(decoder_output_all))
    # print("valid.shape", valid.shape)
    # print("predicted.shape", predicted.shape)

    # PRINT OUTPUT
    output_string = ''
    for i in range(samples):
        for j in range(y_sent_len):
            letter = rev_dict[predicted[i][j]]
            output_string += letter
            output_string += sep
            # print(letter, end=' ')  # the translation part
        # print()
        output_string += "\n"
    print(output_string)
    # it is not the best - implement cosine distance instead?                 TODO different then accuracy
    #                                                                         todo it be quite slow
    character_level_acc = m.calc_accuracy(predicted, valid, samples, y_sent_len)
    print("character accuracy:", character_level_acc)
    print("f1 prec rec :", m.f1_precision_recall(target, predicted, valid))   # needs to be the target file
    return output_string

print("testing...")
print("testing data preparation")

test_x = Data(sep, mezera, end_line)
with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_x.file = f.read()
    f.close()
test_y = Data(sep, mezera, end_line)
with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_y.file = f.read()
    f.close()

samples = 3
test_x.dict_chars = source.dict_chars
x_test = test_x.split_n_count(False)[:samples]
x_test_pad = test_x.padding(x_test, source.maxlen)

test_y.dict_chars = target.dict_chars
y_test = test_y.split_n_count(False)[:samples]
y_test_pad = test_y.padding(y_test, target.maxlen)
y_test_pad_shift = test_y.padding_shift(y_test, target.maxlen)

assert len(x_test) == len(y_test)

#  OLD TESTING
print("Testing")

# output = np.zeros((1, target.maxlen))
# output[:, 0] = 1
output = [[1] for _ in y_test_pad]
print(output)
print(len(output))
print(np.array(output).shape)
j = 0

# print(x_test_pad[0])
# print(y_test_pad[0])
# print(y_test_pad_shift[0])

while j < len(y_test_pad[:samples]):
    print(j)
    i = 1
    # print(x_test_pad[j])
    while i < target.maxlen:
        # old_stdout = sys.stdout
        # sys.stdout = open(os.devnull, "w")
        # print(np.array([output[j]]))
        # print(np.array([x_test_pad[j]]))
        prediction = model.call((np.array([x_test_pad[j]]), np.array([output[j]])), training=False)
        # sys.stdout = old_stdout

        # Get the probabilities for the next token
        next_token_probs = prediction[0, -1, :]  # Prediction is shape (1, i, 63)

        # Sample the next token based on the predicted probabilities
        # next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
        next_token = np.argmax(next_token_probs)
        # print(next_token)

        if next_token == 0:
            print(output[j])
            print("END")
            break
        # Update the output sequence with the sampled token
        # output[j, i] = next_token
        output[j].append(next_token)
        # print(output)
        # print(i, np.argmax(prediction[j][i]))
        i += 1
    j += 1


valid = list(target.padded.astype(np.int32))

print("prediction:", output)
# print("target:", valid)

print()

mistake_count = 0
for j in range(len(list(output))):
    for i in range(len(list(output[j]))):
        if output[j] == 2:
            break
        if valid[j][i] == 2 or i > len(valid[j]) - 1:
            break
        if valid[j][i] != output[j][i]:
            mistake_count += 1
print(mistake_count)

