import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Masking, Embedding, LSTM, Dense
from keras.models import Model
def get_model():
    embed_dim = 32
    latent_dim = 32
    in_vocab_size = 100
    in_seq_len = 10
    out_vocab_size = 50
    out_seq_len = 20

    # not bidirectional yet
    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_input")
    masked_encoder = Masking(mask_value=0, name="encoder_mask")(encoder_inputs)
    embed_masked_encoder = Embedding(in_vocab_size, embed_dim, input_length=in_seq_len, name="encoder_embed")(
        masked_encoder)

    encoder = LSTM(latent_dim, return_state=True, return_sequences=False, activation='sigmoid', name="encoder_LSTM")
    encoder_outputs, state_h, state_c = encoder(embed_masked_encoder)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_input")  # sent_len tam mozna byt nemusi?
    masked_decoder = Masking(mask_value=0, name="decoder_mask")(decoder_inputs)
    embed_masked_decoder = Embedding(out_vocab_size, embed_dim, input_length=out_seq_len, name="decoder_embed")(
        masked_decoder)
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, activation='sigmoid', name="decoder_LSTM")
    decoder_outputs, _, _ = decoder(embed_masked_decoder, initial_state=encoder_states)

    # attention = Attention()([decoder_outputs, encoder_outputs])
    # context_vector = Concatenate(axis=-1)([decoder_outputs, attention])

    decoder_dense = Dense(out_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model

def get_encoder_model():
    embed_dim = 32
    latent_dim = 32
    in_vocab_size = 100
    in_seq_len = 10

    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_input")
    masked_encoder = Masking(mask_value=0, name="encoder_mask")(encoder_inputs)
    embed_masked_encoder = Embedding(in_vocab_size, embed_dim, input_length=in_seq_len, name="encoder_embed")(
        masked_encoder)

    encoder = LSTM(latent_dim, return_state=True, return_sequences=False, activation='sigmoid', name="encoder_LSTM")
    encoder_outputs, state_h, state_c = encoder(embed_masked_encoder)
    encoder_states = [state_h, state_c]

    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    return encoder_model

def get_decoder_model():
    embed_dim = 32
    latent_dim = 32
    out_vocab_size = 50
    out_seq_len = 20

    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_input")
    masked_decoder = Masking(mask_value=0, name="decoder_mask")(decoder_inputs)
    embed_masked_decoder = Embedding(out_vocab_size, embed_dim, input_length=out_seq_len, name="decoder_embed")(
        masked_decoder)

    decoder_states_input_h = Input(shape=(latent_dim,), name="decoder_input_h")
    decoder_states_input_c = Input(shape=(latent_dim,), name="decoder_input_c")
    decoder_states_inputs = [decoder_states_input_h, decoder_states_input_c]

    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, activation='sigmoid', name="decoder_LSTM")
    decoder_outputs, state_h, state_c = decoder(embed_masked_decoder, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_dense = Dense(out_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

    return decoder_model

model = get_model()

# Train the model.
# Train the model.
test_input_encoder = np.random.random((128, 10))  # Assuming in_seq_len is 10
test_input_decoder = np.random.random((128, 20))  # Assuming out_seq_len is 20
test_target = np.random.random((128, 20, 50))  # Assuming out_vocab_size is 50

model.fit({'encoder_input': test_input_encoder, 'decoder_input': test_input_decoder}, test_target)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("my_model.keras")
model.save_weights("trial.h5")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model.keras")

model2 = get_model()
model2.load_weights("trial.h5")

# Let's check:
print("check")
predictions1 = model.predict({'encoder_input': test_input_encoder, 'decoder_input': test_input_decoder})
predictions2 = reconstructed_model.predict({'encoder_input': test_input_encoder, 'decoder_input': test_input_decoder})
predictions3 = model2.predict({'encoder_input': test_input_encoder, 'decoder_input': test_input_decoder})


# Check if the predictions are close within a certain tolerance
tolerance = 1e-5
for p1, p2, p3 in zip(predictions1, predictions2, predictions3):
    np.testing.assert_allclose(p1, p2, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(p1, p3, rtol=tolerance, atol=tolerance)

print("All checks passed successfully.")