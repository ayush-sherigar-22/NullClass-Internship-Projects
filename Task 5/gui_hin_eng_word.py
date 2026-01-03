# Load the libraries
import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity



#  Load model 
with open("translator_model.pkl", "rb") as f:
    model, vectorizer, english_words, hindi_words = pickle.load(f)



#  Translation logic 
def translate_word():
    word = entry.get().strip().lower()
    output_box.delete(0, tk.END)


    # Validate input
    if not word.isalpha():
        messagebox.showerror("Invalid Input", " Please enter a valid English word.")
        return


    #  Time + vowel restriction
    vowels = ['a', 'e', 'i', 'o', 'u']
    current_hour = datetime.now().hour

    if word[0] in vowels and current_hour != 21:
        messagebox.showerror(
            "Error",
            " This word starts with a vowel.\n You can only translate such words between 9 PM and 10 PM."
        )
        return


    #  Vectorize and predict
    X_test = vectorizer.transform([word])
    probs = model.predict_proba(X_test)
    top_prob = np.max(probs)
    pred = model.classes_[np.argmax(probs)]


    #  Confidence fallback
    if top_prob < 0.3:
        X_all = vectorizer.transform(english_words)
        sim = cosine_similarity(X_test, X_all)[0]
        best_idx = np.argmax(sim)
        pred = hindi_words.iloc[best_idx]
        output_box.insert(0, f" {pred}")
    else:
        output_box.insert(0, f"{pred}")



#  GUI setup 
root = tk.Tk()
root.title(" English → Hindi Translator (ML Model)")
root.geometry("500x300")
root.config(bg="#f7f9fb")

title_label = tk.Label(
    root,
    text="English → Hindi Translator",
    font=("Arial", 16, "bold"),
    bg="#f7f9fb",
    fg="#333"
)
title_label.pack(pady=10)

entry_label = tk.Label(
    root,
    text="Enter English Word:",
    font=("Arial", 12),
    bg="#f7f9fb"
)
entry_label.pack()

entry = tk.Entry(root, width=35, font=("Arial", 12))
entry.pack(pady=5)

translate_btn = tk.Button(
    root,
    text="Translate",
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    command=translate_word,
    relief="raised",
    padx=10,
    pady=5
)
translate_btn.pack(pady=10)

output_label = tk.Label(
    root,
    text="Hindi Translation:",
    font=("Arial", 12),
    bg="#f7f9fb"
)
output_label.pack()

output_box = tk.Entry(root, width=35, font=("Arial", 12))
output_box.pack(pady=5)

footer = tk.Label(
    root,
    text="Note: Words starting with vowels can only be translated between 9 PM and 10 PM.",
    font=("Arial", 9),
    fg="gray",
    bg="#f7f9fb",
    wraplength=450,
    justify="center"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
