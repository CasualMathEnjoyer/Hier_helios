import nltk.translate.bleu_score
import numpy as np
import random

import sys
import os
import pickle
from tqdm import tqdm
import time
import joblib
import nltk

from metrics_evaluation import metrics as m
from data_file import Data
from data_preparation import *

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.utils import set_random_seed
from keras import backend as K

new = 0
new_class_dict = 0
caching = 0

batch_size = 2  # 256
epochs = 100
repeat = 1

print("starting transform2seq")

model_file_name = "models/transform_smol_delete"
history_dict = model_file_name + '_HistoryDict'
testing_cache_filename = model_file_name + '_TestingCache'
print(model_file_name)

class_data = "processed_data_dict.plk"

h = 4          # Number of self-attention heads
d_k = 64       # Dimensionality of the linearly projected queries and keys
d_v = 63       # Dimensionality of the linearly projected values                                     # values not used
d_ff = 512      # Dimensionality of the inner fully connected layer
d_model = 128  # Dimensionality of the model sub-layers' outputs
n = 2          # Number of layers in the encoder stack
params = h, d_k, d_v, d_ff, d_model, n

a = random.randrange(0, 2**32 - 1)
a = 12612638
set_random_seed(a)
print("seed = ", a)


# from model_file_2 import model_func
# from model_file_2 import *  # for loading
from model_file_mine import *

def load_model_mine(model_name):
    custom_objects = {
        "MyMaskingLayer" : MyMaskingLayer,
        "CustomSinePositionEncoding" : CustomSinePositionEncoding
    }
    return keras.models.load_model(model_name + ".keras", custom_objects=custom_objects)

def save_model(model, model_file_name):
    try:
        model.save(model_file_name)
        print("Model saved successfully the old way")
    except Exception as e:
        model.save(model_file_name + ".keras")
        print("Model saved using KERAS 3")


# ---------------------------- DATA PROCESSING -------------------------------------------------
if new_class_dict:
    start = time.time()
    print("data preparation...")
    source, target, val_source, val_target = prepare_data()
    to_save_list = [source, target, val_source, val_target]
    end = time.time()
    print("preparation of data took:", end - start)
    def save_object(obj, filename):
        print("Saving data")
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    save_start = time.time()
    save_object(to_save_list, class_data)
    save_end = time.time()
    print("saving of data took: ", save_end - save_start)
else:
    start = time.time()
    print("loading class data")
    with open(class_data, 'rb') as class_data_dict:
        source, target, val_source, val_target = pickle.load(class_data_dict)
        print("Class data loaded.")
        end = time.time()
    print("loadig took:", end - start)

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
# model.summary()
print()

# exit()
# --------------------------------- TRAINING ------------------------------------------------------------------------
# existuje generator na trenovaci data
print("training")
if repeat*epochs == 0:
    print("Skipping training")
for i in range(repeat):
    history = model.fit(
        (source.padded, target.padded), target.padded_shift_one,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one))

    save_model(model, model_file_name)

    new_dict = join_dicts(old_dict, history.history)
    old_dict = new_dict
    with open(history_dict, 'wb') as file_pi:
        pickle.dump(new_dict, file_pi)

    K.clear_session()
print()
# try:
#     model.save(model_file_name + ".keras", save_format="tf")
# except Exception as e:
#     model.export(model_file_name)  # saving for Keras3


# ---------------------------------- TESTING ------------------------------------------------------------------------
from Levenshtein import distance
distance("lewenstein", "levenshtein")
import matplotlib.pyplot as plt

def plot_attention_weights(attention_list, input_sentence, output_sentence, n, h):
    fig = plt.figure(figsize=(16, 8))
    for i, attention in enumerate(attention_list):
        attention = attention[-1][0].numpy()
        for j, attention_head in enumerate(attention):
            ax = fig.add_subplot(n, h, i*h + j + 1)

            # Plot the attention weights
            ax.matshow(attention_head, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(input_sentence)))
            ax.set_yticks(range(len(output_sentence)))

            ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(output_sentence, fontdict=fontdict)

            ax.set_xlabel(f'Head {j + 1}')

    plt.tight_layout()
    plt.show()
def visualise_attention(model, encoder_input_data, decoder_input_data, n, h):
    n_attention_scores = []
    for i in range(n):
        model = keras.Model(inputs=model.input,
                            outputs=[model.output, model.get_layer(f'cross_att{i}').output])
        _, attention_scores = model.call((encoder_input_data, decoder_input_data), training=False)
        n_attention_scores.append(attention_scores)

    # placeholder code before i fill in the text
    input_sentence = [str(i) for i in range(encoder_input_data.shape[1])]
    output_sentence = [str(i) for i in range(decoder_input_data.shape[1])]

    plot_attention_weights(n_attention_scores, input_sentence, output_sentence, n, h)

def translate(model, encoder_input, output_maxlen):
    output_line = [1]
    # i = 1
    i = 0
    while i < output_maxlen:
        prediction = model.call((encoder_input, np.array([output_line])), training=False)  # enc shape: (1, maxlen), out shape: (1, j)
        # next_token_probs = prediction[0, -1, :]  # Prediction is shape (1, i, 63)
        next_token_probs = prediction[0, i, :]  # prediction has the whole sentence every time
        # next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
        next_token = np.argmax(next_token_probs)
        if next_token == 0:
            break
        # Update the output sequence with the sampled token
        output_line.append(next_token)
        i += 1
    # try:
    #     visualise_attention(model, encoder_input, np.array([output_line]), n, h)
    # except Exception as e:
    #     print(e)
    return output_line

print("Testing data preparation")
test_source = Data(sep, mezera, end_line)
test_target = Data(sep, mezera, end_line)
with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_source.file = f.read()
    f.close()
with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_target.file = f.read()
    f.close()

samples = 5
test_source.dict_chars = source.dict_chars
x_test = test_source.split_n_count(False)[:samples]
test_source.padded = test_source.padding(x_test, source.maxlen)

test_target.dict_chars = target.dict_chars
y_test = test_target.split_n_count(False)[:samples]
test_target.padded = test_target.padding(y_test, target.maxlen)
test_target.padded_shift = test_target.padding_shift(y_test, target.maxlen)

valid = list(test_target.padded.astype(np.int32))

assert len(x_test) == len(y_test)
del x_test, y_test

output = []

print("Testing...")
if caching:
    print("Caching is ON")
    tested_dict = load_cached_dict(testing_cache_filename)
else:
    print("Caching is OFF")
# Testing Loop
for j in tqdm(range(len(test_source.padded))):
    i = 1
    encoder_input = np.array([test_source.padded[j]])
    if caching:
        encoder_cache_code = tuple(encoder_input[0])  # cos I can't use np array or list as a hash, [0] removes [around]
        if encoder_cache_code in tested_dict:
            output_line = tested_dict[encoder_cache_code]
        else:
            output_line = translate(model, encoder_input, target.maxlen)
            tested_dict[encoder_cache_code] = output_line
    else:
        output_line = translate(model, encoder_input, target.maxlen)
        # print(output_line)
    output.append(output_line)
# End Testing Loop
if caching:
    cache_dict(tested_dict, testing_cache_filename)

# PRETY TESTING PRINTING
rev_dict = test_target.create_reverse_dict(test_target.dict_chars)

mistake_count, all_chars, all_levenstein = 0, 0, 0
line_lengh = len(valid[0])
num_lines = len(valid)
output_list_words, valid_list_words = [], []  # i could take the valid text from y_test but whatever
output_list_chars, valid_list_chars = [], []
for j in range(len(list(output))):
    print("test line number:", j)
    predicted_line = np.array(output[j])
    valid_line = np.array(valid[j])
    if 0 in valid_line:  # aby to neusekavalo vetu
        zero_index = np.argmax(valid_line == 0)
        valid_line = valid_line[:zero_index]
    min_size = min([predicted_line.shape[0], valid_line.shape[0]])
    max_size = max([predicted_line.shape[0], valid_line.shape[0]])

    mistake_in_line = 0
    if min_size != max_size:
        print("Lines are not the same length")
        mistake_in_line += (max_size - min_size)

    for i in range(min_size):
        if valid[j][i] != output[j][i]:
            mistake_in_line += 1

    output_text_line, valid_text_line = "", ""
    output_list_line, valid_list_line = [], []
    for char in predicted_line:
        output_text_line += (rev_dict[char] + sep)
        output_list_line.append(rev_dict[char])
    for char in valid_line:
        valid_text_line += (rev_dict[char] + sep)
        valid_list_line.append(rev_dict[char])
    output_list_words.append(output_text_line)
    valid_list_words.append(valid_text_line)
    output_list_chars.append(output_list_line)
    valid_list_chars.append([valid_list_line])  # to be accepted by BLEU scocre
    levenstein = distance(output_text_line, valid_text_line)
    print("prediction: ", output_text_line)
    print("valid     : ", valid_text_line)
    print("mistakes  : ", mistake_in_line)
    print("levenstein: ", levenstein)
    print()
    mistake_count += mistake_in_line
    all_levenstein += levenstein
    all_chars += max_size

pred_words_split_mezera, valid_words_split_mezera, valid_words_split_mezeraB = [], [], []
for i in range(len(output_list_words)):
    pred_words_split_mezera.append(output_list_words[i].split(mezera))
    valid_words_split_mezera.append(valid_list_words[i].split(mezera))
    valid_words_split_mezeraB.append([valid_list_words[i].split(mezera)])

word_accuracy = m.on_words_accuracy(pred_words_split_mezera, valid_words_split_mezera)
print("word_accuracy:", round(word_accuracy*100, 5), "%")
print("character accuracy:", round((1 - (mistake_count / all_chars))*100, 5), "%")
print("average Levenstein: ", all_levenstein / num_lines)
print("BLEU SCORE words:", nltk.translate.bleu_score.corpus_bleu(valid_words_split_mezeraB, pred_words_split_mezera))
print("BLEU SCORE chars:", nltk.translate.bleu_score.corpus_bleu(valid_list_chars, output_list_chars))