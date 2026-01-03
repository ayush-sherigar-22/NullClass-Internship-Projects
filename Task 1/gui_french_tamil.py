import tkinter as tk
from tkinter import messagebox
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Load tokenizers 

with open("french_tokenizer.json", "r", encoding="utf-8") as f:
    french_tk = tokenizer_from_json(f.read())

with open("tamil_tokenizer.json", "r", encoding="utf-8") as f:
    tamil_tk = tokenizer_from_json(f.read())


# Parameters

max_french_len = 5
max_tamil_len = 15


# Load the models

encoder_model = load_model("encoder_model.h5")
decoder_model = load_model("decoder_model.h5")


# Decode Function

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tamil_tk.word_index['\t']  # START TOKEN

    decoded_sentence = ''

    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tamil_tk.index_word.get(sampled_index, '')

        if sampled_char == '\n' or len(decoded_sentence) > max_tamil_len:
            break

        decoded_sentence += sampled_char

        target_seq[0, 0] = sampled_index
        states_value = [h, c]

    return decoded_sentence.strip()


# Translate Function

def translate():
    word = french_entry.get().strip().lower()

    if len(word) != 5:
        messagebox.showerror("Error", "Enter exactly 5-letter French word")
        return

    seq = french_tk.texts_to_sequences([word])
    seq = pad_sequences(seq, maxlen=max_french_len, padding="post")

    tamil_word = decode_sequence(seq)
    tamil_output.set(tamil_word)


# GUI

root = tk.Tk()
root.title("French → Tamil Translator")
root.geometry("420x260")
root.resizable(False, False)

tk.Label(root, text="French Word (5 letters)", font=("Arial", 12)).pack(pady=10)
french_entry = tk.Entry(root, font=("Arial", 14), justify="center")
french_entry.pack()

tk.Button(
    root,
    text="Translate",
    font=("Arial", 12),
    bg="#4CAF50",
    fg="white",
    command=translate
).pack(pady=15)

tk.Label(root, text="Tamil Translation", font=("Arial", 12)).pack()
tamil_output = tk.StringVar()
tk.Entry(
    root,
    textvariable=tamil_output,
    font=("Arial", 14),
    justify="center",
    state="readonly"
).pack(pady=5)

root.mainloop()
