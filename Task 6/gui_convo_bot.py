import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import pickle
import numpy as np
import threading
import tempfile
import os
from gtts import gTTS
import pygame
import pyttsx3


# GPU MEMORY EXTENSION

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# CUSTOM ATTENTION 

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, enc_out, dec_out):
        score = self.V(tf.nn.tanh(
            tf.expand_dims(self.W1(enc_out), 1) +
            tf.expand_dims(self.W2(dec_out), 2)
        ))
        attention_weights = tf.nn.softmax(score, axis=2)
        context = attention_weights * tf.expand_dims(enc_out, 1)
        context = tf.reduce_sum(context, axis=2)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config



# LOAD MODELS

spa_to_eng_model = load_model( "spa_to_eng_translation.h5")


eng_to_spa_model = load_model(
    "eng_to_spa_translation.h5",
    custom_objects={"BahdanauAttention": BahdanauAttention},
    compile=False
)


# LOAD TOKENIZERS

with open("spa_tokenizer_1.pkl", "rb") as f:
    spa_tok_1 = pickle.load(f)

with open("eng_tokenizer_1.pkl", "rb") as f:
    eng_tok_1 = pickle.load(f)

with open("spa_tokenizer_2.pkl", "rb") as f:
    spa_tok_2 = pickle.load(f)

with open("eng_tokenizer_2.pkl", "rb") as f:
    eng_tok_2 = pickle.load(f)

with open("config_eng_spa.pkl", "rb") as f:
    config = pickle.load(f)

MAX_ENG_2 = config["MAX_ENG"]
MAX_SPA_2 = config["MAX_SPA"]

MAX_SPA_1= 20
MAX_ENG_1 = 22

rev_eng = {v: k for k, v in eng_tok_1.word_index.items()}
rev_spa = {v: k for k, v in spa_tok_2.word_index.items()}


# TRANSLATION FUNCTIONS

def translate_spa_to_eng(text):
    seq = spa_tok_1.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=MAX_SPA_1, padding="post")

    decoder = [eng_tok_1.word_index["<start>"]]
    result = []

    for _ in range(MAX_ENG_1):
        dec_seq = pad_sequences([decoder], maxlen=MAX_ENG_1, padding="post")
        pred = spa_to_eng_model.predict([seq, dec_seq], verbose=0)
        idx = np.argmax(pred[0, len(decoder)-1])
        if idx == 0 or rev_eng.get(idx) == "<end>":
            break
        result.append(rev_eng[idx])
        decoder.append(idx)

    return " ".join(result)

def translate_eng_to_spa(sentence):
    # Encode input
    seq = eng_tok_2.texts_to_sequences([sentence.lower()])
    seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_ENG_2, padding="post"
    )

    # Start token
    start_id = spa_tok_2.word_index["<start>"]
    end_id   = spa_tok_2.word_index["<end>"]

    decoder_input = np.zeros((1, MAX_SPA_2 - 1))
    decoder_input[0, 0] = start_id

    result = []

    for t in range(1, MAX_SPA_2 - 1):
        pred = eng_to_spa_model.predict([seq, decoder_input], verbose=0)
        pred_id = np.argmax(pred[0, t-1])

        if pred_id == end_id:
            break

        result.append(pred_id)
        decoder_input[0, t] = pred_id

    # Convert ids to words
    rev_spa = {v: k for k, v in spa_tok_2.word_index.items()}
    words = [rev_spa.get(i, "") for i in result]

    return " ".join(words)


# TEXT TO SPEECH 

def speak(text, lang="en"):
    if not text.strip():
        return

    # ENGLISH 
    if lang == "en":
        try:
            engine = pyttsx3.init("sapi5")
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)

            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[0].id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print("English TTS Error:", e)

    #SPANISH 
    elif lang == "es":
        def _spanish():
            try:
                tts = gTTS(text=text, lang="es")

                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_path = temp.name
                temp.close()

                tts.save(temp_path)

                pygame.mixer.init()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pass

                pygame.mixer.music.stop()
                pygame.mixer.quit()
                os.remove(temp_path)

            except Exception as e:
                print("Spanish TTS Error:", e)

        threading.Thread(target=_spanish, daemon=True).start()


# SPEECH TO TEXT

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as src:
        audio = r.listen(src)

    try:
        if direction.get() == "SPA2ENG":
            txt = r.recognize_google(audio, language="es-ES")
        else:
            txt = r.recognize_google(audio, language="en-US")
        input_box.delete("1.0", tk.END)
        input_box.insert(tk.END, txt)
        output_box.insert(tk.END, f"\n 👂{txt}\n")
    except Exception as e:
        output_box.insert(tk.END, f"\n {e}\n")


# TRANSLATE BUTTON

def translate():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        return

    if direction.get() == "SPA2ENG":
        translated = translate_spa_to_eng(text)
        output_box.insert(tk.END, f"\n 🗣️Translation (EN): {translated}\n")
        speak(translated, lang="en")   

    else:
        translated = translate_eng_to_spa(text)
        output_box.insert(tk.END, f"\n 🗣️Translation (ES): {translated}\n")
        speak(translated, lang="es")   



# GUI

root = tk.Tk()
root.title("Dual Voice Translator (ENG ⇄ SPA)")
root.geometry("750x650")

tk.Label(root, text="Realtime Conversation with Voice Translation",
         font=("Arial", 20)).pack(pady=10)

direction = tk.StringVar(value="SPA2ENG")

frame = tk.Frame(root)
frame.pack()

tk.Radiobutton(frame, text="Spanish → English",
               variable=direction, value="SPA2ENG").pack(side="left", padx=20)

tk.Radiobutton(frame, text="English → Spanish",
               variable=direction, value="ENG2SPA").pack(side="left", padx=20)

input_box = scrolledtext.ScrolledText(root, height=6)
input_box.pack(fill="x", padx=20, pady=10)

btns = tk.Frame(root)
btns.pack()

tk.Button(btns, text="Speech", command=speech_to_text, width=20).grid(row=0, column=0, padx=10)
tk.Button(btns, text="Translate", command=translate, width=20).grid(row=0, column=1, padx=10)

output_box = scrolledtext.ScrolledText(root, height=18)
output_box.pack(fill="x", padx=20)

root.mainloop()
