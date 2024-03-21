import numpy as np



class Data():
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

        # print("average:     ", round(complete / len(self.file.split('\n')), 2))
        # print("maxlen:      ", maxlen)

        likelyhood = 39 / 40
        weird_median = sorted(len_list)[int(len(len_list) * likelyhood)]
        # print('with:', likelyhood,":", weird_median)  # mene nez 2.5% ma sequence delsi, nez 100 znaku
        # maxlen: 1128
        # average: 31.42447596485441
        self.maxlen = weird_median
        if create_dic:
            self.dict_chars = dict_chars
            self.vocab_size = len(dict_chars)
            # print("dict chars:", self.dict_chars)
            # print("vocab size:", self.vocab_size)
        return output
    def padding(self, input_list, lengh):
        input_list_padded = np.zeros((len(input_list), lengh))
        for i, line in enumerate(input_list):
            if len(line) > lengh: # shorten
                input_list_padded[i] = np.array(line[:lengh])
            elif len(line) <= lengh:  # padd, # 0 is the code for padding
                input_list_padded[i] = np.array(line + [0 for i in range(lengh - len(line))])
            else:
                assert False
        # print(input_list_padded)
        return input_list_padded
    def padding_shift(self, input_list, lengh):
        input_list_padded = np.zeros((len(input_list), lengh))  # maybe zeros?
        for i, line in enumerate(input_list):
            if len(line) > lengh: # shorten
                input_list_padded[i] = np.array(line[1 : lengh + 1])
            elif len(line) <= lengh:  # padd, # 0 is the code for padding
                input_list_padded[i] = np.array(line[1:] + [0 for i in range(lengh - len(line) + 1)])
            else:
                assert False
        # print(input_list_padded)
        return input_list_padded
