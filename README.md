# 🎤 Emotion-Based Multimodal Conversational Feedback

An AI-powered **Communication Coach** that analyzes non-verbal cues (Facial Expressions + Voice Tone) to provide actionable feedback on your presentation style.

This project utilizes **Multimodal Deep Learning**, fusing Facial and Audio Model Processing to detect 7 distinct emotional states with high precision.

**[👉 Try the Live App Here](https://personal-emotion-coach.streamlit.app/)**

---

## 🚀 Features

* **Multimodal Analysis**: Processes both video frames (Face) and audio waveforms (Voice) simultaneously.
* **Real-time Feedback**: Acts as an "Elite Communication Coach," giving specific advice on posture, tone, and energy based on the detected emotion.
* **Robust Architecture**: Uses **Late Fusion** with a custom **Gated Unit** to weigh audio vs. visual signals dynamically.
* **Ensemble Stacking**: Implements a meta-learner (XGBoost + Logistic Regression) to optimize predictions.
* **Data Augmentation**: Features custom geometric augmentation for faces and **SpecAugment** for audio to ensure model robustness.

---

## 🧠 Technical Architecture

This system is built on a **Two-Stream Network** architecture:

### 1. Unimodal Feature Extraction

* **Visual Stream**: Uses **EfficientNetB0** (Transfer Learning) to extract spatial features from facial images.
* *Preprocessing*: Haar Cascades for face detection + Geometric Augmentation (Shift/Rotate/Zoom).


* **Audio Stream**: Uses **EfficientNetB0** on **Mel-Spectrograms** to extract frequency-time features.
* *Preprocessing*: Librosa for spectrogram generation + **SpecAugment** (Time/Frequency Masking).



### 2. Multimodal Fusion

Instead of simple concatenation, this project uses a **Gated Fusion Network**:

* A learnable **Sigmoid Gate** determines the reliability of each modality.
* If the face is occluded or the audio is noisy, the network automatically shifts its attention to the clearer signal.

### 3. Ensemble Stacking

To maximize test accuracy, a **Dual-Stack Meta-Learner** aggregates predictions:

* **Level 0**: Face Model and Audio Model
* **Level 1**: Fusion Model.
* **Level 2**: **XGBoost** (captures non-linear correlations) + **Logistic Regression** (prevents overfitting).

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Nicolechy04/DSP.git
cd personal-emotion-coach

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*(Note: Requires `tensorflow`, `opencv-python`, `librosa`, `moviepy`, `streamlit`, `xgboost`)*
3. **Run the App:**
```bash
streamlit run app.py

```



---

## 📂 Project Structure

```bash
├── models_zoo_1/          # Pre-trained models
│   ├── efficientnet_improved.keras       # Face Model
│   ├── Audio_EfficientNet_Refined.keras  # Audio Model
│   └── modelfusion_2.keras               # Fusion Model
├── user.py                 # Streamlit Application
└── requirements.txt       # Dependencies

```

---

## 📊 Model Performance

* **Training Strategy**: Used **GroupShuffleSplit** to ensure no actor overlap between train/test sets (preventing data leakage).
* **Optimization**: Employed **Test Time Augmentation (TTA)** during evaluation to ensure theoretical maximum accuracy.
* **Loss Function**: Categorical Crossentropy with **Label Smoothing** to prevent overconfidence.

---

### ✨ Acknowledgements

* Dataset: [Insert Dataset Name, e.g., RAVDESS/CREMA-D]
* Libraries: TensorFlow Keras, Streamlit, Librosa, OpenCV
