# Load the libraries
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox



import json
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np, json, pickle
import tensorflow as tf



# Load encoder decoder models
encoder_model = load_model("encoder_model.keras")
decoder_model = load_model("decoder_model.keras")

# English to Hindi
model = load_model("eng_hindi_translation_model.keras")

# Load tokenizers
with open("english2_tokenizer.json", "r", encoding="utf-8") as f:
    eng_data = f.read()
    eng_tokenizer = tokenizer_from_json(eng_data)

with open("hindi_tokenizer.json", "r", encoding="utf-8") as f:
    hin_data = f.read()
    hin_tokenizer = tokenizer_from_json(hin_data)

# Load sequence lengths
with open("seq_lengths.pkl", "rb") as f:
    max_encoder_seq_length, max_decoder_seq_length = pickle.load(f)

# Reverse lookup
reverse_target_word_index = hin_tokenizer.index_word
target_word_index = hin_tokenizer.word_index



max_len_src = 20     
max_len_tgt = 20     

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

def translate_to_hindi(english_text):
    # 10-letter rule
    if len(english_text.replace(" ", "")) < 10:
        return " Text too short (needs ≥10 letters)."

    seq = eng_tokenizer.texts_to_sequences([english_text.lower()])
    seq = np.array(pad_sequences(seq, maxlen=max_encoder_seq_length, padding='post'))
    return decode_sequence(seq)

 
# English to French model

# Load French model
model_fr = load_model("english_to_french_model")

# Load tokenizer
with open("english1_tokenizer.json") as f:
    eng1_data = json.load(f)
    english_tokenizer = tokenizer_from_json(eng1_data) 

with open('french_tokernizer.json')as f : 
    french_data = json.load(f)
    french_tokenizer = tokenizer_from_json(french_data)


# Load  max length 
with open('sequence_length.json')as f:
    max_length = json.load(f)

def pad(x, length =None):
    return pad_sequences(x, maxlen=length, padding = 'post')


def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()

    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    # Tokenize and pad
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)

    english_sentence = english_sentence.reshape((-1, max_length))

    french_sentence = model_fr.predict(english_sentence)[0]

    french_sentence = [np.argmax(word) for word in french_sentence]

    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]

    print("French translastion :" , french_sentence)

    return french_sentence


# Translation function


def handle_translate():
    english_text = text_input.get("1.0", "end-1c").strip()
    selected_language = language_var.get()

    translation_output.delete("1.0", "end")

# Check input validity
    if len(english_text.replace(" ", "")) < 10:
        translation = " Text too short! Must contain at least 10 letters."
        translation_output.insert("end",   f"\n\n {translation}")


    else:
        if selected_language == "Hindi":
        
            english_text = text_input.get("1.0", "end-1c")
            translation = translate_to_hindi(english_text)
            
                    
        elif selected_language=="French":
            translation = translate_to_french(english_text)
    
        else:
            translation = "Unknown language."

        translation_output.insert("end", f"\n\n {selected_language} translation: {translation}")




# Setting up the root window
root = tk.Tk()
root.title("Language Translation")
root.geometry("550x600")


# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text = "Enter the text to be translated", font=(font_style, font_size, 'bold'))

# Text inout for English Setence
text_input = tk.Text(input_frame, height = 5, width = 50, font=(font_style, font_size))
text_input.pack()

# Language Selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select  the language to translate to", font=(font_style, font_size, "bold"))
language_select = ttk.Combobox(root, textvariable = language_var, values = ["French", "Hindi"], font= (font_style, font_size),state='readonly')
language_select.pack()

# Submit Button
submit_button = ttk.Button(root, text = "Translate", command = handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

# Heading for output 
output_heading = tk.Label(output_frame, text = "Translation:", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translation
translation_output = tk.Text(output_frame, height =10, width=50, font = (font_style, font_size))
translation_output.pack()

# Running the application
root.mainloop()