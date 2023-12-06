from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional, Flatten, RepeatVector, Permute, Multiply, Lambda
from keras.models import Model, Sequential

import keras.backend as K

from keras.layers import Masking, Embedding

# 0.3734

def model_func(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    embed_dim = 32
    latent_dim = 32

    # not bidirectional yet
    encoder_inputs = Input(shape=(None, ), dtype="int64")
    masked_encoder = Masking(mask_value=0)(encoder_inputs)
    embed_masked_encoder = Embedding(in_vocab_size, embed_dim, input_length=in_seq_len)(masked_encoder)

    encoder = LSTM(latent_dim, return_state=True, return_sequences=False, activation='sigmoid')
    encoder_outputs, state_h, state_c = encoder(embed_masked_encoder)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, ), dtype="int64")  # sent_len tam mozna byt nemusi?
    masked_decoder = Masking(mask_value=0)(decoder_inputs)
    embed_masked_decoder = Embedding(out_vocab_size, embed_dim, input_length=out_seq_len)(masked_decoder)
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, activation='sigmoid')
    decoder_outputs, _, _ = decoder(embed_masked_decoder, initial_state=encoder_states)

    # attention = Attention()([decoder_outputs, encoder_outputs])
    #context_vector = Concatenate(axis=-1)([decoder_outputs, attention])

    decoder_dense = Dense(out_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    return model