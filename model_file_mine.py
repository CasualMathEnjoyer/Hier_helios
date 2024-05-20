import keras
import numpy as np
import tensorflow as tf
import keras_nlp

class CustomSinePositionEncoding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSinePositionEncoding, self).__init__(**kwargs)
        self.sine_position_encoding = keras_nlp.layers.SinePositionEncoding()

    def call(self, inputs, mask=None):
        positional = self.sine_position_encoding(inputs)
        return inputs + positional

    def compute_mask(self, inputs, mask=None):
        return mask

class MyMaskingLayer(keras.layers.Layer):
    def call(self, x):
        mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
        mask = mask[:, tf.newaxis, :]
        # mask = x[:, tf.newaxis, :]  # this works when input is a mask
        return mask

def model_func(encoder_vocab_len, decoder_vocab_len, encoder_maxlen, decoder_maxlen, params):
    num_heads, key_dim, value_dim, d_ff, d_model, n = params

    encoder_input = keras.Input(shape=(encoder_maxlen,))  # fixed len input to apply positional encoding
    decoder_input = keras.Input(shape=(decoder_maxlen,))

    # Encoder part
    encoder_mask = MyMaskingLayer()(encoder_input)
    embedded = keras.layers.Embedding(input_dim=encoder_vocab_len, output_dim=d_model, mask_zero=False)(encoder_input)
    # not usig default masking in embedding because it doesn't propagate anyway

    embedded_position = CustomSinePositionEncoding()(embedded)
    embedded_position = keras.layers.Dropout(0.1)(embedded_position)

    encoded = embedded_position
    for i in range(n):
        attended_encoded = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(encoded, encoded, encoded,
                           attention_mask=encoder_mask  # [batch, sequences, model_dim(embedding)]
                           )
        attended_encoded_d = keras.layers.Dropout(0.1)(attended_encoded)

        add = encoded + attended_encoded_d
        normalized = keras.layers.LayerNormalization()(add)

        fed_f = keras.layers.Dense(d_ff)(normalized)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add2 = normalized + fed_ff_d
        normalized2 = keras.layers.LayerNormalization()(add2)

        encoded = normalized2  # and the loop is repeated

    encoder_output = encoded  # output from encoder

    # Decoder part
    decoder_mask = MyMaskingLayer()(decoder_input)
    de_embed = keras.layers.Embedding(input_dim=decoder_vocab_len, output_dim=d_model, mask_zero=False)(decoder_input)

    de_embed_pos = CustomSinePositionEncoding()(de_embed)
    de_embed_pos = keras.layers.Dropout(0.1)(de_embed_pos)

    decoded = de_embed_pos
    cross_attention_vecs = []
    for i in range(n):
        self_attention = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(decoded, decoded, decoded
                           , attention_mask=decoder_mask
                           , use_causal_mask=True
                           )
        self_attention_d = keras.layers.Dropout(0.1)(self_attention)

        add = decoded + self_attention_d
        normalized1 = keras.layers.LayerNormalization()(add)

        cross_attention, cross_attention_scores = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True,
            name=f'cross_att{i}')(normalized1, encoder_output, encoder_output
                                   , attention_mask=encoder_mask
                                   , return_attention_scores=True  # to calculate attention matrix
                                   )
        # cross_attention_vecs.append(cross_attention_scores)

        cross_attention_d = keras.layers.Dropout(0.1)(cross_attention)

        add2 = normalized1 + cross_attention_d
        normalized2 = keras.layers.LayerNormalization()(add2)

        fed_f = keras.layers.Dense(d_ff)(normalized2)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add3 = normalized2 + fed_ff_d
        normalized3 = keras.layers.LayerNormalization()(add3)

        decoded = normalized3

    decoder_dense_output = keras.layers.Dense(decoder_vocab_len, activation='softmax', name='decoder_output')(decoded)

    return keras.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense_output])

if __name__ == '__main__':
    params = (4, 64, 64, 256, 512, 2)
    model = model_func(10, 10, 50, 50, params)
    # model.summary()

    # Generate random input data with appropriate shapes
    encoder_input_data = np.random.randint(0, 10, (1, 50))  # (batch_size, sequence_length)
    decoder_input_data = np.random.randint(0, 10, (1, 50))  # (batch_size, sequence_length)

    # visualise_attention(model, encoder_input_data, decoder_input_data, 2, 4)
