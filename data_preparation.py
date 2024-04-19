import numpy as np
import os
import pickle
from keras.utils import to_categorical


from data_file import Data

# TODO - check for lines of all zeros in tokens

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


print()


def prepare_data():
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
    source.padded = source.padding(x_train, source.maxlen)
    print("second file:")
    y_train = target.split_n_count(True)
    target.padded = target.padding(y_train, target.maxlen)
    # y_train_pad_one = to_categorical(y_train_pad)
    target.padded_shift = target.padding_shift(y_train, target.maxlen)
    target.padded_shift_one = to_categorical(target.padded_shift)

    assert len(source.padded) == len(target.padded_shift)
    assert len(source.padded) == len(target.padded_shift_one)

    print(np.array(source.padded).shape)            # (1841, 98)
    print(np.array(target.padded).shape)            # (1841, 109)
    print(np.array(target.padded_shift).shape)      # (1841, 109)
    print(np.array(target.padded_shift_one).shape)  # (1841, 109, 55)

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
    val_source.padded = val_source.padding(x_val, source.maxlen)

    val_target.dict_chars = target.dict_chars
    y_val = val_target.split_n_count(False)
    val_target.padded = val_target.padding(y_val, target.maxlen)
    val_target.padded_shift = val_target.padding_shift(y_val, target.maxlen)
    val_target.padded_shift_one = to_categorical(val_target.padded_shift, num_classes=len(target.dict_chars))

    # print("source.maxlen:", source.maxlen)
    # print("target.maxlen:", target.maxlen)
    # print("source_val.maxlen:", val_source.maxlen)
    # print("target_val.maxlen:", val_target.maxlen)
    # print("source.dict:", len(source.dict_chars))
    # print("target.dict:", len(target.dict_chars))
    # print("source_val.dict:", len(val_source.dict_chars))
    # print("target_val.dict:", len(val_target.dict_chars))

    assert len(x_val) == len(val_source.padded)
    assert len(x_val) == len(y_val)
    assert len(val_source.padded) == len(val_target.padded_shift)
    assert len(val_source.padded) == len(val_target.padded_shift_one)

    print(np.array(val_source.padded).shape)            # (1841, 98)
    print(np.array(val_target.padded).shape)            # (1841, 109)
    print(np.array(val_target.padded_shift).shape)      # (1841, 109)
    print(np.array(val_target.padded_shift_one).shape)  # (1841, 109, 55)

    # ValueError: Shapes (None, 109, 55) and (None, 109, 63) are incompatible

    # print(y_train_pad_one)
    # print()
    # print(x_train_pad.shape)
    # print(y_train_pad.shape)
    # print(y_train_pad_shift.shape)
    # print(y_train_pad_one.shape)

    del source.file
    del target.file
    del val_source.file
    del val_target.file

    print("Data prepared")
    return source, target, val_source, val_target


def get_history_dict(dict_name, new):
    dict_exist = os.path.isfile(dict_name)
    if dict_exist:
        if new:
            # q = input(f"Dict with the name {dict_name} exist but we create a new one, ok?")
            q = "ok"
            print("Rewritting the dict")
            if q == "ok":
                return {}
            else:
                raise Exception("Dont do this")
        else:
            with open(dict_name, "rb") as file_pi:
                old_dict = pickle.load(file_pi)
                return old_dict
    return {}

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

if __name__ == "__main__":
    source, target, val_source, val_target = prepare_data()