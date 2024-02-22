import tensorflow as tf
from tensorflow import keras
from keras.layers import Masking, Input, TimeDistributed, Dense, Layer, Embedding, Dropout
from keras.layers import MultiHeadAttention, LayerNormalization

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def model_func(vocab_size, maxlen, embed_dim, num_heads, ff_dim):
    inputs = Input(shape=(maxlen,))
    masked = Masking(mask_value=0, name="masking_layer")(inputs)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(masked)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    network_timestep = TimeDistributed(Dense(1, activation='sigmoid'))
    outputs = network_timestep(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
