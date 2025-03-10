# 🎭 Live Sentiment Analysis

## 📌 Project Overview
This project is a **Live Sentiment Analysis** tool that captures **audio and image** data to analyze sentiment. The application utilizes **natural language processing (NLP)** for speech sentiment analysis and **computer vision** for facial emotion detection. The tool is built using **Streamlit** for an interactive UI and employs various deep learning models for analysis.

## 🚀 Features
- **Real-time Speech Sentiment Analysis**
- **Facial Emotion Recognition**
- **Interactive Web App using Streamlit**
- **Uses Pretrained Deep Learning Models**

## 🧠 Models Used
### **1️⃣ Whisper (by OpenAI)**
**Purpose:** Speech-to-Text Transcription
- Used to convert captured **speech** into **text**.
- The **medium.en** model is used, which is trained on diverse datasets.
- Implements a transformer-based **sequence-to-sequence** architecture.

### **2️⃣ DistilBERT (for Sentiment Analysis)**
**Purpose:** Text Sentiment Analysis
- Pretrained model: `bhadresh-savani/distilbert-base-uncased-emotion`
- Classifies text into emotions like **joy, sadness, anger, fear, love, and surprise**.
- A **lighter** version of BERT (Bidirectional Encoder Representations from Transformers), optimized for speed.

### **3️⃣ ViT (Vision Transformer) for Emotion Recognition**
**Purpose:** Facial Emotion Classification
- Uses `google/vit-base-patch16-224-in21k` for image processing.
- Fine-tuned on `Ketanwip/happy_sad_model` to classify facial emotions as **happy or sad**.
- Works by dividing an image into small patches, then processing them using **self-attention mechanisms**.

---

## 🏗️ Key Concepts
### **1️⃣ Transformer Models**
- Used in both **Whisper** and **DistilBERT**.
- Utilize **self-attention** mechanisms to process sequential data efficiently.

### **2️⃣ Convolution vs. Transformers (ViT)**
- Traditional **CNNs (Convolutional Neural Networks)** are used for image recognition.
- **ViTs (Vision Transformers)** process images like NLP tasks, breaking them into patches instead of using convolutions.

### **3️⃣ Sentiment Analysis in NLP**
- Uses **BERT-based transformers** to extract emotions from text.
- Determines sentiment based on contextual meaning rather than just word frequency.

---

## 👨‍💻 Author
- **Developer:** Your Name
- **Contact:** your.email@example.com

**Happy Coding! 🚀**

