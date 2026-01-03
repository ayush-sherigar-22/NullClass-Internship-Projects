# Load the libraries
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Concatenate


# Load trained model and tokenizers

model = load_model("eng_hin_seq2seq.keras", compile=False)

with open("tokenizers.pkl", "rb") as f:
    eng_tokenizer, hin_tokenizer, MAX_ENG, MAX_HIN = pickle.load(f)

latent_dim = 256


# Build Encoder Model 

encoder_inputs = model.get_layer("encoder_inputs").input
encoder_embedding = model.get_layer("encoder_embedding")(encoder_inputs)
encoder_lstm = model.get_layer("encoder_lstm")

encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_embedding)

encoder_model = Model(
    encoder_inputs, 
    [encoder_outputs, state_h_enc, state_c_enc]
)


# Build Decoder Model 

decoder_inputs = Input(shape=(1,), name="decoder_input_token")

# States input for inference
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")


# Encoder outputs input 
encoder_outputs_input = Input(shape=(MAX_ENG, latent_dim), name="encoder_output_inf")

decoder_embedding_layer = model.get_layer("decoder_embedding")
decoder_lstm = model.get_layer("decoder_lstm")
attention_layer = model.get_layer("attention_layer")
dense_output = model.get_layer("output_dense")


decoder_emb = decoder_embedding_layer(decoder_inputs)

decoder_outputs, dec_h, dec_c = decoder_lstm(
    decoder_emb,
    initial_state=[decoder_state_input_h, decoder_state_input_c]
)

# Attention layer
attention_out = attention_layer([decoder_outputs, encoder_outputs_input])

# Combine attention + decoder output
concat_out = Concatenate(axis=-1)([decoder_outputs, attention_out])

# Final prediction
decoder_output_tokens = dense_output(concat_out)

# Build full decoder model
decoder_model = Model(
    [decoder_inputs, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
    [decoder_output_tokens, dec_h, dec_c]
)




reverse_hin_word_index = {v: k for k, v in hin_tokenizer.word_index.items()}
start_idx = hin_tokenizer.word_index["<start>"]
end_idx = hin_tokenizer.word_index["<end>"]



# TRANSLATE FUNCTION 

def translate(sentence):

    # Convert input → padded tokens
    seq = eng_tokenizer.texts_to_sequences([sentence.lower()])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_ENG, padding="post")

    # Encode input
    enc_outputs, h, c = encoder_model.predict(seq)

    # Start token
    target_seq = np.array([[start_idx]])

    result = []

    for _ in range(MAX_HIN):
        # predict 1 step
        output_tokens, h, c = decoder_model.predict(
            [target_seq, h, c, enc_outputs]
        )

        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_hin_word_index.get(sampled_idx, "")

        if sampled_word == "<end>" or sampled_word == "":
            break

        result.append(sampled_word)

        target_seq = np.array([[sampled_idx]])

    return " ".join(result)
