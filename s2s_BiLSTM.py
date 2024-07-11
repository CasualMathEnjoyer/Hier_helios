import numpy as np
import random
import keras
import pickle

import sys
import os

from keras.utils import set_random_seed
from keras.utils import to_categorical
from keras import backend as K

# TODO - check for lines of all zeros in tokens
# TODO - cropping sentences might be a problem!

print("starting transform2seq")

from metrics_evaluation import metrics as m
from data_file import Data
from testing_s2s import test_translation, add_to_json

a = random.randrange(0, 2**32 - 1)
# a = 1261263827
set_random_seed(a)
print("seed = ", a)

# from model_file_LSTM import model_func, load_and_split_model
from model_file_BiLSTM import model_func, load_and_split_model, encoder_state_transform

# model_file_name = "transform2seq_1"
# training_file_name = "../data/src-sep-train.txt"
# target_file_name = "../data/tgt-train.txt"
# # validation_file_name = "../data/src-sep-val.txt"
# ti_file_name = "../data/src-sep-test.txt"  # test input file
# tt_file_name = "../data/tgt-test.txt"  # test target
# sep = ' '
# mezera = '_'
# end_line = '\n'

# model_name = "transform2seq_LSTM_em32_dim64"
# model_file_name = f"models/{model_name}"
# model_file_name = "/home/katka/Documents/models_LSTM/transform2seq_LSTM_em32_dim64"
# history_dict = model_file_name + '_HistoryDict'
# print(model_file_name)


# model_file_name = "transform2seq_LSTM_delete"
# train_in_file_name = "../data/smallvoc_fr_.txt"
# train_out_file_name = "../data/smallvoc_en_.txt"
# val_in_file_name = "../data/smallvoc_fr_.txt"
# val_out_file_name = "../data/smallvoc_en_.txt"
# test_in_file_name = "../data/smallervoc_fr_.txt"
# test_out_file_name = "../data/smallervoc_en_.txt"

# train_in_file_name = "../data/fr_train.txt"
# train_out_file_name = "../data/en_train.txt"
# val_in_file_name = "../data/fr_val.txt"
# val_out_file_name = "../data/en_val.txt"
# test_in_file_name = "../data/fr_test.txt"
# test_out_file_name = "../data/en_test.txt"


train_in_file_name = "../data/src-sep-train.txt"
train_out_file_name = "../data/tgt-train.txt"
val_in_file_name = "../data/src-sep-val.txt"
val_out_file_name = "../data/tgt-val.txt"
test_in_file_name = "../data/src-sep-test.txt"
test_out_file_name = "../data/tgt-test.txt"
# train_in_file_name = "../data/src-sep-train-short.txt"
# train_out_file_name = "../data/tgt-train-short.txt"
# val_in_file_name = "../data/src-sep-train-short.txt"
# val_out_file_name = "../data/tgt-train-short.txt"
# test_in_file_name = "../data/src-sep-train-short.txt"
# test_out_file_name = "../data/tgt-train-short.txt"

sep = ' '
mezera = '_'
end_line = '\n'

new = 0

batch_size = 256
epochs = 0
repeat = 0  # full epoch_num=epochs*repeat

sample_limit = 2
version = "2_sample"
keras_version = "2.10.0"
result_json_path = f"LSTM_results_{version}.json"

def load_model_mine(model_name):
    # from model_file import PositionalEmbedding, TransformerEncoder, TransformerDecoder
    # return keras.models.load_model(model_name, custom_objects={'PositionalEmbedding': PositionalEmbedding,
    #                                                            'TransformerEncoder': TransformerEncoder,
    #                                                            'TransformerDecoder': TransformerDecoder
    # })
    return keras.models.load_model(model_name)

print()
print("data preparation...")
source = Data(sep, mezera, end_line)
target = Data(sep, mezera, end_line)
with open(train_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    source.file = f.read()
    f.close()
with open(train_out_file_name, "r", encoding="utf-8") as ff:
    target.file = ff.read()
    ff.close()

print("first file:")
x_train = source.split_n_count(True)
x_train_pad = source.padding(x_train, source.maxlen)
print("second file:")
y_train = target.split_n_count(True)
y_train_pad = target.padding(y_train, target.maxlen)
# y_train_pad_one = to_categorical(y_train_pad)
y_train_pad_shift = target.padding_shift(y_train, target.maxlen)
y_train_pad_shift_one = to_categorical(y_train_pad_shift)
assert len(x_train_pad) == len(y_train_pad_shift)
assert len(x_train_pad) == len(y_train_pad_shift_one)

# print(np.array(x_train_pad).shape)            # (1841, 98)
# print(np.array(y_train_pad).shape)            # (1841, 109)
# print(np.array(y_train_pad_shift).shape)      # (1841, 109)
# print(np.array(y_train_pad_shift_one).shape)  # (1841, 109, 55)

# VALIDATION:
print("validation files:")
val_source = Data(sep, mezera, end_line)
val_target = Data(sep, mezera, end_line)
with open(val_in_file_name, "r", encoding="utf-8") as f:
    val_source.file = f.read()
    f.close()
with open(val_out_file_name, "r", encoding="utf-8") as ff:
    val_target.file = ff.read()
    ff.close()

val_source.dict_chars = source.dict_chars
x_val = val_source.split_n_count(False)
x_val_pad = val_source.padding(x_val, source.maxlen)

val_target.dict_chars = target.dict_chars
y_val = val_target.split_n_count(False)
y_val_pad = val_target.padding(y_val, target.maxlen)
y_val_pad_shift = val_target.padding_shift(y_val, target.maxlen)
y_val_pad_shift_one = to_categorical(y_val_pad_shift, num_classes=len(target.dict_chars))

# print("source.maxlen:", source.maxlen)
# print("target.maxlen:", target.maxlen)
# print("source_val.maxlen:", val_source.maxlen)
# print("target_val.maxlen:", val_target.maxlen)
# print("source.dict:", len(source.dict_chars))
# print("target.dict:", len(target.dict_chars))
# print("source_val.dict:", len(val_source.dict_chars))
# print("target_val.dict:", len(val_target.dict_chars))

assert len(x_val) == len(x_val_pad)
assert len(x_val) == len(y_val)
assert len(x_val_pad) == len(y_val_pad_shift)
assert len(x_val_pad) == len(y_val_pad_shift_one)

print(np.array(x_val_pad).shape)            # (1841, 98)
print(np.array(y_val_pad).shape)            # (1841, 109)
print(np.array(y_val_pad_shift).shape)      # (1841, 109)
print(np.array(y_val_pad_shift_one).shape)  # (1841, 109, 55)

print("testing data preparation")

test_x = Data(sep, mezera, end_line)
with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_x.file = f.read()
    f.close()
test_y = Data(sep, mezera, end_line)
with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_y.file = f.read()
    f.close()

test_x.dict_chars = source.dict_chars
if sample_limit == -1: x_test = test_x.split_n_count(False)
else: x_test = test_x.split_n_count(False)[:sample_limit]
x_test_pad = test_x.padding(x_test, source.maxlen)

test_y.dict_chars = target.dict_chars
if sample_limit == -1: y_test = test_y.split_n_count(False)
else: y_test = test_y.split_n_count(False)[:sample_limit]
y_test_pad = test_y.padding(y_test, target.maxlen)
y_test_pad_shift = test_y.padding_shift(y_test, target.maxlen)

num_testing_sentences = len(x_test)

assert len(x_test) == len(y_test)
assert len(y_test) == len(y_test_pad_shift)

# ValueError: Shapes (None, 109, 55) and (None, 109, 63) are incompatible

# print(y_train_pad_one)
# print()
# print(x_train_pad.shape)
# print(y_train_pad.shape)
# print(y_train_pad_shift.shape)
# print(y_train_pad_one.shape)
print()
print("MODEL EVALUATION")
model_folder = "/home/katka/Documents/models_LSTM"
print(os.listdir(model_folder))

import json

for model_name in os.listdir(model_folder):
    if not os.path.isdir(os.path.join(model_folder, model_name)):
        continue

    # Check if the file exists
    if os.path.exists(result_json_path):
        # Load the existing data
        with open(result_json_path, 'r') as file:
            data = json.load(file)
            if model_name in data:
                print("SKIPPING MODEL:", model_name)
                continue

    # model_name = "transform2seq_LSTM_em32_dim64"
    # model_file_name = f"models/{model_name}"
    model_file_name = f"{model_folder}/{model_name}"
    history_dict = model_file_name + '_HistoryDict'
    print(model_file_name)
    try:
        embedding_dim = int(model_file_name.split('_')[-2])
    except ValueError:
        embedding_dim = int(model_name.split("_")[-2][2:])
    latent_dim = int(model_name.split("_")[-1][3:])


    # --------------------------------- MODEL ---------------------------------------------------------------------------
    print("model starting...")
    if new:
        print("CREATING A NEW MODEL")
        model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen)
    else:
        print("LOADING A MODEL")
        model = load_model_mine(model_file_name)

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    print()


    def get_history_dict(dict_name, new):
        dict_exist = os.path.isfile(dict_name)
        if dict_exist:
            if new:
                q = input(f"Dict with the name {dict_name} exist but we create a new one, ok?")
                if q == "ok":
                    return {}
                else:
                    raise Exception("Dont do this")
            else:
                with open(dict_name, "rb") as file_pi:
                    old_dict = pickle.load(file_pi)
                    return old_dict
        return {}


    old_dict = get_history_dict(history_dict, new)
    def join_dicts(dict1, dict2):
        dict = {}
        if dict1 == {}:
            dict = dict2

        if dict1.keys() == dict2.keys():
            pass
        else:
            print(dict1.keys(), " != ", dict2.keys())

        for i in range(len(dict1.keys())):
            history_list = list(dict1.keys())
            # print(history_list[i])
            ar = []
            for item in dict1[history_list[i]]:
                ar.append(item)
            for item in dict2[history_list[i]]:
                ar.append(item)
            dict[history_list[i]] = ar
        # print(dict)
        return dict
    # --------------------------------- TRAINING ------------------------------------------------------------------------
    for i in range(repeat):
        history = model.fit(
            (x_train_pad, y_train_pad), y_train_pad_shift_one,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=((x_val_pad, y_val_pad), y_val_pad_shift_one))
        model.save(model_file_name)
        # model.save_weights(model_file_name + ".h5")
        K.clear_session()
        # save model history
        new_dict = join_dicts(old_dict, history.history)
        old_dict = new_dict
        with open(history_dict, 'wb') as file_pi:
            pickle.dump(new_dict, file_pi)
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
    def model_test_new(encoder, decoder, x_test_pad, y_test_pad, y_test_pad_shift, rev_dict, num_sentences):
        decoder_output_all = []

        x_sent_len = x_test_pad[0].size
        y_sent_len = y_test_pad[0].size
        x_test_pad = x_test_pad.reshape(num_sentences, 1, x_sent_len)  # reshape so encoder takes just one sentence
        y_test_pad = y_test_pad.reshape(num_sentences, 1, y_sent_len)  # and is not angry about dimensions
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
                decoder_output_word = decoder.predict([letter] + decoder_output_throughts)  # TODO

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
        # = y_test_pad_shape trans (samples_len, 1, 90)
        valid = y_test_pad_shift
        print(valid.shape)
        print(valid[0])
        print(decoder_output_all[0])
        predicted = decoder_output_all
        predicted = np.array(predicted)

        # print("decoder output sent, num:", len(decoder_output_all))
        # print("valid.shape", valid.shape)
        # print("predicted.shape", predicted.shape)

        # PRINT OUTPUT
        output_string = ''
        assert num_sentences != 0
        assert y_sent_len != 0
        for i in range(num_sentences):
            for j in range(y_sent_len):
                letter = rev_dict[predicted[i][j]]
                if letter == "<bos>":
                    pass
                if letter == "<eos>":
                    break
                output_string += letter
                output_string += sep
                # print(letter, end=' ')  # the translation part
            # print()
            output_string += "\n"
        print(output_string)
        # it is not the best - implement cosine distance instead?                 TODO different then accuracy
        #                                                                         todo it be quite slow

        # commenting because it needs all
        character_level_acc = m.calc_accuracy(predicted, valid, num_sentences, y_sent_len)
        print("character accuracy:", character_level_acc)
        print("f1 prec rec :", m.f1_precision_recall(target, predicted, valid))   # needs to be the target file

        for index in range(len(decoder_output_all)):
            line = decoder_output_all[index]
            for i in range(len(line)):
                if line[i] == 0:
                    decoder_output_all[index] = [1] + decoder_output_all[index][:i] # adding the <bos> token
                    break

        return output_string, decoder_output_all

    # print("testing...")


    #  OLD TESTING
    # print("old testing")
    # model_test_old(test_y, x_test_pad, y_test_pad_shift, y_test_pad, model_file_name)

    #  BETTER TESTING
    print("new testing")
    # GET ENCODER AND DECODER
    # inputs should be the same as in training data
    encoder, decoder = load_and_split_model(model_file_name, source.vocab_size, target.vocab_size,
                                            source.maxlen, target.maxlen, latent_dim, embedding_dim)
    rev_dict = test_y.create_reverse_dict(test_y.dict_chars)

    output_text, list_output = model_test_new(encoder, decoder, x_test_pad, y_test_pad, y_test_pad_shift, rev_dict, num_testing_sentences)

    #  WORD LEVEL ACCURACY
    split_output_text = output_text.split(end_line)
    split_valid_text = test_y.file.split(end_line)

    new_pred, new_valid = [], []

    # make into lists
    for i in range(len(split_output_text)-1):
        new_pred.append(split_output_text[i].split(mezera))
        new_valid.append(split_valid_text[i].split(mezera))

    # show sentences
    for i in range(len(new_pred)):
        prediction = split_output_text[i]
        valid = split_valid_text[i]
        # print(len(prediction), "- ", len(valid),  # shows values shifted by one because the predicted has one more space
        #       "=", len(prediction) - len(valid))
        print(prediction)
        print(valid)
        print()

    word_accuracy = m.on_words_accuracy(new_pred, new_valid)
    print("word_accuracy:", word_accuracy)

    valid = list(y_test_pad.astype(np.int32))

    dict = test_translation(list_output, valid, rev_dict, sep, mezera)


    def get_epochs_train_accuracy(history_dict):
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            epochs = len(history['accuracy'])
            results = {
                "train_accuracy": history['accuracy'][-1],
                "val_accuracy": history['val_accuracy'][-1],
                "train_loss": history['loss'][-1],
                "val_loss": history['val_loss'][-1]
            }
        return epochs, results

    all_epochs, training_data = get_epochs_train_accuracy(history_dict)

    add_to_json(result_json_path, model_name, dict, sample_limit,
                    all_epochs, training_data, keras_version)
    print()