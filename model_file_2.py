
# used tutorial from:
# https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/

from utils import *
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from keras import Model, Input

class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x

class DecoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, input_layer, None, None, True))

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head attention layer
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)

        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by another multi-head attention layer
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)

        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)

        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout3(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)

# Implementing the Decoder
class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff_inner = d_ff_inner
        self.n = n
        self.rate = rate

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size, activation="softmax")

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, inputs, training):
        encoder_input, decoder_input = inputs

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output

    def get_config(self):
        config = {
            'enc_vocab_size': self.enc_vocab_size,
            'dec_vocab_size': self.dec_vocab_size,
            'enc_seq_length': self.enc_seq_length,
            'dec_seq_length': self.dec_seq_length,
            'h': self.h,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_model': self.d_model,
            'd_ff_inner': self.d_ff_inner,
            'n': self.n,
            'rate': self.rate
        }
        base_config = super(TransformerModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def model_func(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len, params):
    h, d_k, d_v, d_ff, d_model, n = params

    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

    training_model = TransformerModel(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len,
                                      h, d_k, d_v,
                                      d_model, d_ff, n, dropout_rate)
    encoder_input_example = tf.ones((1, in_seq_len))  # Assuming encoder input shape is (batch_size, in_seq_len)
    decoder_input_example = tf.ones((1, out_seq_len))  # Assuming decoder input shape is (batch_size, out_seq_len)

    # Call the model with the sample inputs
    _ = training_model((encoder_input_example, decoder_input_example), training=False)
    return training_model

if __name__ == "__main__":
    # enc_vocab_size = 20  # Vocabulary size for the encoder
    # input_seq_length = 5  # Maximum length of the input sequence
    # h = 8  # Number of self-attention heads
    # d_k = 64  # Dimensionality of the linearly projected queries and keys
    # d_v = 64  # Dimensionality of the linearly projected values
    # d_ff = 2048  # Dimensionality of the inner fully connected layer
    # d_model = 512  # Dimensionality of the model sub-layers' outputs
    # n = 6  # Number of layers in the encoder stack
    #
    # batch_size = 64  # Batch size from the training process
    # dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    #
    # input_seq = random.random((batch_size, input_seq_length))
    #
    # encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
    # print(encoder(input_seq, None, True))
    #
    # dec_vocab_size = 20  # Vocabulary size for the decoder
    # input_seq_length = 5  # Maximum length of the input sequence
    # h = 8  # Number of self-attention heads
    # d_k = 64  # Dimensionality of the linearly projected queries and keys
    # d_v = 64  # Dimensionality of the linearly projected values
    # d_ff = 2048  # Dimensionality of the inner fully connected layer
    # d_model = 512  # Dimensionality of the model sub-layers' outputs
    # n = 6  # Number of layers in the decoder stack
    #
    # batch_size = 64  # Batch size from the training process
    # dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    #
    # input_seq = random.random((batch_size, input_seq_length))
    # enc_output = random.random((batch_size, input_seq_length, d_model))
    #
    # decoder = Decoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
    # print(decoder(input_seq, enc_output, None, True))

    enc_vocab_size = 20  # Vocabulary size for the encoder
    dec_vocab_size = 20  # Vocabulary size for the decoder

    enc_seq_length = 5  # Maximum length of the input sequence
    dec_seq_length = 5  # Maximum length of the target sequence

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 6  # Number of layers in the encoder stack

    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

    # Create model
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v,
                                      d_model, d_ff, n, dropout_rate)
    encoder_input_example = tf.ones((1, enc_seq_length))  # Assuming encoder input shape is (batch_size, enc_seq_length)
    decoder_input_example = tf.ones((1, dec_seq_length))  # Assuming decoder input shape is (batch_size, dec_seq_length)
    decoder_input_example_2 = tf.ones((1, dec_seq_length, dec_vocab_size))  # output is one hot vector

    # Call the model with the sample inputs
    _ = training_model((encoder_input_example, decoder_input_example), training=False)

    training_model.summary()

    history = training_model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

    batch_size = 2
    epochs = 2

    training_model.fit((encoder_input_example, decoder_input_example), decoder_input_example_2,
                       batch_size=batch_size, epochs=epochs,
                       )

    # encoder = EncoderLayer(enc_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
    # encoder.build_graph().summary()
    #
    # decoder = DecoderLayer(dec_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
    # decoder.build_graph().summary()
