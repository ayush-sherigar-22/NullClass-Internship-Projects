# Extend the GPU memory
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


import tkinter as tk
from tkinter import filedialog, Text, Label, Button, Scrollbar, messagebox
from ocr_utils import ocr_from_image, ocr_from_video
from translate_utils import translate


# Functio to accept the images

def upload_image():
    
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if path:
        process_ocr(path, mode="image")

# Function to accept video

def upload_video():
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    if path:
        process_ocr(path, mode="video")


# Function to extract the words using the OCR

def process_ocr(path, mode="image"):
    original_box.delete(1.0, tk.END)
    translated_box.delete(1.0, tk.END)

    # Get OCR text
    if mode == "image":
        text = ocr_from_image(path)
    else:
        text = ocr_from_video(path)

    # Insert only ONCE (not in a loop!)
    original_box.insert(tk.END, text + "\n")
    translated_box.insert(tk.END, translate(text) + "\n")

   



# GUI Window

root = tk.Tk()
root.title("OCR + Translator")
root.geometry("900x600")
root.configure(bg="#eef2f7")

Label(root, text="Extract & Translate (Image / Video)",
      font=("Arial", 18, "bold"), bg="#2e8b57", fg="white").pack(fill=tk.X)


btn_frame = tk.Frame(root, bg="#eef2f7")
btn_frame.pack(pady=10)

Button(btn_frame, text="Upload Image", command=upload_image,
       bg="#4682B4", fg="white", font=("Arial", 12), padx=10).grid(row=0, column=0, padx=10)

Button(btn_frame, text="Upload Video", command=upload_video,
       bg="#6A5ACD", fg="white", font=("Arial", 12), padx=10).grid(row=0, column=1, padx=10)


frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

Label(frame, text="Extracted Text", font=("Arial", 12, "bold")).grid(row=0, column=0)
Label(frame, text="Translated Text", font=("Arial", 12, "bold")).grid(row=0, column=2)

original_box = Text(frame, height=20, width=45, wrap=tk.WORD)
translated_box = Text(frame, height=20, width=45, wrap=tk.WORD)

original_box.grid(row=1, column=0, padx=10)
translated_box.grid(row=1, column=2, padx=10)

scroll1 = Scrollbar(frame, command=original_box.yview)
scroll1.grid(row=1, column=1, sticky='ns')
original_box.config(yscrollcommand=scroll1.set)

scroll2 = Scrollbar(frame, command=translated_box.yview)
scroll2.grid(row=1, column=3, sticky='ns')
translated_box.config(yscrollcommand=scroll2.set)

root.mainloop()
