# Load the libraries
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from datetime import datetime, time
from translate_utils import translate


# SPEECH RECOGNITION FUNCTION
def listen_audio():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("👂 Listening...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="en-IN")
        print("🗣️ You said:", text)
        return text

    except sr.UnknownValueError:
        print("Could not understand. Please repeat.")
        return None


def within_time():
    now = datetime.now().time()
    return time(21,30) <= now <= time(22,0)


# TRANSLATION FUNCTION
def start_translation():

    text = listen_audio()

    if not text:
        messagebox.showwarning("Warning", "Please repeat your sentence.")
        return

    # Show recognized English text
    input_box.delete(1.0, tk.END)
    input_box.insert(tk.END, text)

    # Translate
    hindi_translation = translate(text)

    # Show translated Hindi text
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, hindi_translation)


# GUI SETUP 

root = tk.Tk()
root.title("Voice Translator (English → Hindi)")

#  Recognized Text
tk.Label(root, text="Recognized Text (English):", font=("Arial", 12)).pack()
input_box = tk.Text(root, height=3, width=40, font=("Arial", 14))
input_box.pack(pady=5)

#  Translated Output
tk.Label(root, text="Output (Hindi):", font=("Arial", 12)).pack()
output_box = tk.Text(root, height=5, width=40, font=("Arial", 14))
output_box.pack(pady=5)

tk.Button(
    root,
    text="🎤 Start Listening",
    font=("Arial", 12),
    command=start_translation
).pack(pady=10)

root.mainloop()
