import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import torch
import whisper
import time
import cv2
import numpy as np

from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
from PIL import Image
from utility.captureaudio import capture_audio


st.title("🎭 Live Sentiment Analysis")


mode = st.radio("Capture:", "Image")


resultAudio = [0, 0]
resultImage = [0, 0]

def analyse_audio():
    st.write(" Recording audio...")
    audio_file = capture_audio()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model("medium.en", download_root="~/.cache/whisper").to(device)

    result = model.transcribe(audio_file)

    st.write(" Transcribed Text:", result["text"])
    

    sentiment_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
    res = sentiment_pipeline(result["text"])

    for i in res[0]:
        if i['label'] in ['anger', 'fear', 'sadness']:
            resultAudio[0] += i['score']
        elif i['label'] in ['joy', 'love', 'surprise']:
            resultAudio[1] += i['score']

    st.write("Sentiment Analysis Result:", res)


def capture_image():
    st.write("Initializing camera...")

    cam = cv2.VideoCapture(0)  
    time.sleep(3)  

    ret, frame = cam.read()  
    cam.release()  

    if not ret:
        st.error(" Failed to capture image. Please check your camera.")
        return None

    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    
    image_path = "captured_image.jpg"
    image.save(image_path)

    return image_path


def analyse_image():
    st.write("📷 Capturing image...")
    image_file = capture_image()

    if image_file:
        st.image(image_file, caption="Captured Image", use_container_width=True)

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTForImageClassification.from_pretrained("Ketanwip/happy_sad_model")

        image = Image.open(image_file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_lbl = torch.topk(probs, 1)

        prediction = "Happy" if top_lbl.item() == 0 else "Sad"

        st.write(f" Detected Emotion: **{prediction}** (Confidence: {top_prob.item():.2%})")


if st.button("Run Analysis"):
    if mode == "Speech":
        analyse_audio()
    if mode == "Image":
        analyse_image()
    else:
        analyse_audio()
        analyse_image()
