from gpu_config import configure_gpu
configure_gpu()

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Attention, Concatenate
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad

LATENT_DIM = 256

# Load full seq2seq model
model = load_model("eng_hin_seq2seq.keras")

# Load tokenizers & config
with open("tokenizers.pkl", "rb") as f:
    eng_tokenizer, hin_tokenizer, MAX_ENG, MAX_HIN = pickle.load(f)


# Inference Model
enc_inf_input = Input(shape=(MAX_ENG,))
enc_inf_emb = model.get_layer("encoder_embedding")(enc_inf_input)

encoder_outputs, h_enc, c_enc = model.get_layer("encoder_lstm")(enc_inf_emb)


encoder_model = Model(enc_inf_input, [encoder_outputs, h_enc, c_enc])



dec_input = Input(shape=(1,))
dec_h = Input(shape=(LATENT_DIM,))
dec_c = Input(shape=(LATENT_DIM,))
enc_out_inf = Input(shape=(MAX_ENG, LATENT_DIM))

dec_emb = model.get_layer("decoder_embedding")(dec_input)

dec_out, h_new, c_new = model.get_layer("decoder_lstm")(
    dec_emb, initial_state=[dec_h, dec_c]
)

attn_out = model.get_layer("attention_layer")([dec_out, enc_out_inf])

concat = Concatenate(axis=-1)([dec_out, attn_out])

dec_logits = model.get_layer("output_dense")(concat)

decoder_model = Model(
    [dec_input, dec_h, dec_c, enc_out_inf],
    [dec_logits, h_new, c_new]
)

# Translate Function


reverse_hin = {v: k for k, v in hin_tokenizer.word_index.items()}
start_idx = hin_tokenizer.word_index["<start>"]
end_idx = hin_tokenizer.word_index["<end>"]

def translate(sentence, max_len=MAX_HIN):
    # 1️Encode input sentence
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad(seq, maxlen=MAX_ENG, padding="post")
    
    encoder_outputs, h, c = encoder_model.predict(seq, verbose=0)

    #  Start decoding with <start>
    target_seq = np.array([[start_idx]])
    output_words = []

    for _ in range(max_len):
        preds, h, c = decoder_model.predict(
            [target_seq, h, c, encoder_outputs], verbose=0
        )

        # Pick highest probability token
        token_id = int(np.argmax(preds[0, -1, :]))

        # Stop at <end>
        if token_id == end_idx:
            break

        # Append word
        word = reverse_hin.get(token_id, "<unk>")
        output_words.append(word)

        # Update target_seq for next step
        target_seq = np.array([[token_id]])

    return " ".join(output_words)
