import numpy as np
import random
import keras
from keras.utils import set_random_seed
from keras.utils import to_categorical
from keras import backend as K

# TODO - check for lines of all zeros in tokens
# TODO - cropping sentences might be a problem!


print("starting transform2seq")

a = random.randrange(0, 2**32 - 1)
# a = 1261263827
set_random_seed(a)
print("seed = ", a)

#todo : When you use padding, you should always do masking on the output
# and other places where it is relevant (i.e., when computing the attention
# distribution in attention), which ensures no gradient gets propagated
# from the "non-existing" padding positions.

# TODO : plan
# check if data is processed correctly
# 1 implement masking for lstm
# 2 make easily switchable lstm here
# see if it work
# fix precission

# celkem skoro 68 tisic slov
# 47.5 tisic slov jenom jednou

# from model_file import model_func
from model_file_LSTM import model_func

# model_file_name = "transform2seq_1"
# training_file_name = "../data/src-sep-train.txt"
# target_file_name = "../data/tgt-train.txt"
# # validation_file_name = "../data/src-sep-val.txt"
# ti_file_name = "../data/src-sep-test.txt"  # test input file
# tt_file_name = "../data/tgt-test.txt"  # test target
# sep = ' '
# mezera = '_'
# end_line = '\n'

model_file_name = "transform2seq_fr-eng_3LSTM"
training_file_name = "../data/smallvoc_fr_.txt"
target_file_name = "../data/smallvoc_en_.txt"
# validation_file_name = "../data/src-sep-val.txt"
ti_file_name = "../data/smallervoc_fr_.txt"  # test input file
tt_file_name = "../data/smallervoc_en_.txt"  # test target
sep = ' '
mezera = '_'
end_line = '\n'

new = 0

batch_size = 128
epochs = 2
repeat = 2  # full epoch_num=epochs*repeat

class Data():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads

    maxlen = 0
    file = ''
    dict_chars = {}
    reverse_dict = {}
    vocab_size = 0

    def __init__(self, sep, mezera, end_line):
        super().__init__()
        self.sep = sep
        self.space = mezera
        self.end_line = end_line

    def array_to_token(self, input_array): # takes array returns the max index
        if input_array.size == 0:
            # Handle empty array case
            return np.array([])
        max_index = np.argmax(input_array)
        # result_array = np.zeros_like(input_array)  # so it is the same shape
        # result_array[max_index] = 1
        return max_index
    def one_hot_to_token(self, vec): # takes one hot array returns list of tokens
        tokens = []
        for line in vec:
            ll = []
            for char in line:
                ll.append(np.argmax(char))
            tokens.append(ll)
        return tokens
    def create_reverse_dict(self, dictionary):
        reverse_dict = {}
        for key, value in dictionary.items():
            reverse_dict.setdefault(value, key)  # assuming values and keys unique
        self.reverse_dict = reverse_dict
        return reverse_dict
    def model_test(self, sample, valid_shift, valid, model_name, sample_len):  # input = padded array of tokens
        model = load_model_mine(model_name)
        rev_dict = self.create_reverse_dict(self.dict_chars)
        value = model.predict((sample, valid))  # has to be in the shape of the input for it to predict
        # TODO -putting the correct ones there - not a good idea
        # TODO - do we put just the validated stuff in it or do we want to unpack the encoder?
        # print("value.shape=", value.shape)
        # print("valid.shape=", valid_shift.shape)
        # valid jsou tokeny -> one hot


        assert sample_len == len(valid_shift)
        dim = len(valid_shift[0])
        # print ("dim:", dim)
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
        print("f1 prec rec :", f1_precision_recall(value_one, valid_one))
        # print("F1 score:", F1_score(value, valid_one.astype('float32')).numpy())
        # print("F1 score value_one:", F1_score(value_one, valid_one.astype('float32')).numpy())
    def split_n_count(self, create_dic):  # creates a list of lists of TOKENS and a dictionary
        maxlen, complete = 0, 0
        output = []
        len_list = []
        dict_chars = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "_": 3, "OVV": 4}
        for line in self.file.split(self.end_line):
            line = ["<bos>"] + line.split(self.sep) + ["<eos>"]
            ll = len(line)
            len_list.append(len(line))
            if ll > maxlen:
                maxlen = ll
            complete += ll
            l = []
            for c in line:  # leave mezery !!
                if c != '':
                    if create_dic:
                        if c not in dict_chars:
                            dict_chars[c] = len(dict_chars)
                        l.append(dict_chars[c])
                    else:
                        if c in self.dict_chars:
                            l.append(self.dict_chars[c])
                        else:
                            l.append(self.dict_chars["OVV"])
            output.append(l)
        # for line in output:
        #     print(line)
        print("average:     ", round(complete / len(self.file.split('\n')), 2))
        print("maxlen:      ", maxlen)
        likelyhood = 39 / 40
        weird_median = sorted(len_list)[int(len(len_list) * likelyhood)]
        print('with:', likelyhood,":", weird_median)  # mene nez 2.5% ma sequence delsi, nez 100 znaku
        # maxlen: 1128
        # average: 31.42447596485441
        self.maxlen = weird_median
        if create_dic:
            self.dict_chars = dict_chars
            self.vocab_size = len(dict_chars)
            print("dict chars:", self.dict_chars)
            print("vocab size:", self.vocab_size)
        return output
    def padding(self, input_list, lengh):
        input_list_padded = np.zeros((len(input_list), lengh))  # maybe zeros?
        for i, line in enumerate(input_list):
            if len(line) > lengh: # shorten
                input_list_padded[i] = np.array(line[:lengh])
            elif len(line) < lengh:  # padd, # 0 is the code for padding
                input_list_padded[i] = np.array(line + [0 for i in range(lengh - len(line))])
            else:
                pass
        # print(input_list_padded)
        return input_list_padded
    def padding_shift(self, input_list):
        input_list_padded = np.zeros((len(input_list), self.maxlen))  # maybe zeros?
        for i, line in enumerate(input_list):
            if len(line) > self.maxlen: # shorten
                input_list_padded[i] = np.array(line[1 : self.maxlen + 1])
            elif len(line) < self.maxlen:  # padd, # 0 is the code for padding
                input_list_padded[i] = np.array(line[1:] + [0 for i in range(self.maxlen - len(line) + 1)])
            else:
                pass
        # print(input_list_padded)
        return input_list_padded

# precision = to minimise false alarms
# precision = TP/(TP + FP)
# recall = to minimise missed spaces
# recall = TP/(TP+FN)

def calculate_precision_recall_f1(y_true, y_pred, label):
    true_positive = np.sum((y_true == label) & (y_pred == label))
    false_positive = np.sum((y_true != label) & (y_pred == label))
    false_negative = np.sum((y_true == label) & (y_pred != label))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
def f1_precision_recall(y_true, y_pred):
    dict = target.dict_chars
    y_true = np.array(target.one_hot_to_token(y_true))
    y_pred = np.array(target.one_hot_to_token(y_pred))
    # unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    unique_labels = np.array(list(dict.values()))
    # print("labels:", unique_labels)
    # print("d labs:", np.array(list(dict.values())))
    total_precision = 0
    total_recall = 0

    for label in unique_labels:
        precision, recall, _ = calculate_precision_recall_f1(y_true, y_pred, label)
        total_precision += precision
        total_recall += recall
        target.create_reverse_dict(target.dict_chars)
        print("char:", target.reverse_dict[label], "- f1:", round(2*precision*recall/(precision+recall), 5) if (precision+recall) > 0 else "zero")
        # TODO  zero

    macro_precision = total_precision / len(unique_labels) if len(unique_labels) > 0 else 0
    macro_recall = total_recall / len(unique_labels) if len(unique_labels) > 0 else 0

    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    # return f1, precision and recall formated na dve desetinna mista
    # return float(f'{macro_f1:.2f}'), float(f'{macro_precision:.2f}'), float(f'{macro_recall:.2f}')
    return round(macro_f1, 2), round(macro_precision, 2), round(macro_recall, 2)

    # # Example usage:
    # # Assume y_true and y_pred are your true and predicted labels, respectively.
    #
    # # Generating example labels for demonstration
    # np.random.seed(42)
    # y_true = np.random.randint(0, 3, size=100)  # 3 classes
    # y_pred = np.random.randint(0, 3, size=100)
    #
    # # Calculate multiclass F1 score
    # f1 = multiclass_f1_score(y_true, y_pred)
    #
    # print(f'Multiclass F1 Score: {f1}')`

def F1_score(y_true, y_pred): #taken from old keras source code  # TODO transform
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    # print("precision:", precision.numpy(), "recall:", recall.numpy())
    return f1_val

    # return f1_score(y_true, y_pred, average=None)
def load_model_mine(model_name):
    from model_file import PositionalEmbedding, TransformerEncoder, TransformerDecoder
    return keras.models.load_model(model_name, custom_objects={"F1_score": F1_score,
                                                               'PositionalEmbedding': PositionalEmbedding,
                                                               'TransformerEncoder': TransformerEncoder,
                                                               'TransformerDecoder': TransformerDecoder
    })

print()
print("data preparation...")
source = Data(sep, mezera, end_line)
target = Data(sep, mezera, end_line)
with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    source.file = f.read()
    f.close()
with open(target_file_name, "r", encoding="utf-8") as ff:
    target.file = ff.read()
    ff.close()

print("first file:")
x_train = source.split_n_count(True)
x_train_pad = source.padding(x_train, source.maxlen)
del x_train
print()
print("second file:")
y_train = target.split_n_count(True)
y_train_pad = target.padding(y_train, target.maxlen)
# y_train_pad_one = to_categorical(y_train_pad)
y_train_pad_shift = target.padding_shift(y_train)
del y_train
y_train_pad_shift_one = to_categorical(y_train_pad_shift)
del y_train_pad_shift
print()

# print(y_train_pad_one)
# print()
# print(x_train_pad.shape)
# print(y_train_pad.shape)
# print(y_train_pad_shift.shape)
# print(y_train_pad_one.shape)

# --------------------------------- MODEL ---------------------------------------------------------------------------
print("model starting...")
if new:
    model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen)
else:
    model = load_model_mine(model_file_name)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy", F1_score])
model.summary()
print()
# --------------------------------- TRAINING ------------------------------------------------------------------------
# TODO shuffle ?
for i in range(repeat):
    history = model.fit(
        (x_train_pad, y_train_pad), y_train_pad_shift_one, batch_size=batch_size, epochs=epochs)
        # validation_data=(x_valid_tokenized, y_valid))
    model.save(model_file_name)
    K.clear_session()
print()
# ---------------------------------- TESTING ------------------------------------------------------------------------
print("testing...")
test_x = Data(sep, mezera, end_line)
with open(ti_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_x.file = f.read()
    f.close()
test_y = Data(sep, mezera, end_line)
with open(tt_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_y.file = f.read()
    f.close()

samples = 10
test_x.dict_chars = source.dict_chars  # mohla bych prepsat file v source a jen znova rozbehnout funkci
print("source dict len: ", len(source.dict_chars))
x_test = test_x.split_n_count(False)[:10]  # ale tohle je lepsi
x_test_pad = test_x.padding(x_test, source.maxlen)

test_y.dict_chars = target.dict_chars
print("target dict len: ", len(target.dict_chars))
y_test = test_y.split_n_count(False)[:10]
y_test_pad = test_y.padding(y_test, target.maxlen)
y_test_pad_shift = test_y.padding_shift(y_test)

assert len(x_test) == len(y_test)

lengh = len(x_test)
test_y.model_test(x_test_pad, y_test_pad_shift, y_test_pad, model_file_name, lengh)