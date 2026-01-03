from gpu_config import configure_gpu
configure_gpu()

import easyocr
import cv2
import numpy as np


# INITIALIZE EASYOCR

reader = easyocr.Reader(['en'], gpu=True)



# BASIC SAFE RESIZE

def preprocess_image(img, max_dim=1024):
    if img is None:
        raise ValueError("❌ preprocess_image(): Input image is None")

    h, w = img.shape[:2]
    scale = max_dim / max(h, w)

    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img


def sort_easyocr_results(results):
    # results = [ [bbox, text, confidence], ... ]
    def get_tl(bbox):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return min(ys), min(xs)

    return sorted(results, key=lambda x: get_tl(x[0]))



# MAIN OCR FOR IMAGE

def ocr_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)

    # OCR
    results = reader.readtext(img, detail=1)

    # Sort
    results = sort_easyocr_results(results)

    # Extract words only
    words = [res[1] for res in results]

    # Join into sentence
    final_sentence = " ".join(words)

    return final_sentence.strip()



# OCR FOR VIDEO

def ocr_from_video(video_path, frame_step=15):
    cap = cv2.VideoCapture(video_path)
    extracted_sentences = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_step != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = preprocess_image(img_rgb)

        # OCR
        results = reader.readtext(img_rgb, detail=1)

        # Sort
        results = sort_easyocr_results(results)

        # Create sentence
        sentence = " ".join([res[1] for res in results])
        extracted_sentences.append(sentence.strip())

    cap.release()

    # Remove duplicates 
    final_output = " ".join(dict.fromkeys(extracted_sentences))

    return final_output.strip()
