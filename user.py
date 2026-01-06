import streamlit as st
import os
import cv2
import numpy as np
import librosa
import tempfile
import pandas as pd
import gc
import time
import joblib  
import google.generativeai as genai
import plotly.graph_objects as go

from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from xgboost import XGBClassifier 

st.set_page_config(page_title="Personal Emotion Coach", page_icon="🧘", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FDFBFF; }
    .report-container { 
        background-color: #FFFFFF; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #E0E0E0; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); 
    }
    .stMetric { 
        background-color: #F3F0FF; 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #A29BFE; 
    }
    .stButton>button { 
        background-color: #6C5CE7; 
        color: white; 
        border-radius: 20px; 
        width: 100%; 
        height: 50px; 
        font-weight: bold; 
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #5849C4; border: none; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = "models_zoo_1"
STRIDE = 2.0  

# Session State Initialization
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

with st.sidebar:
    st.header("System Status")
    
    # Check for Gemini API Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        st.success("AI Coach Online")
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("Key Loaded")
        else:
            st.warning("Enter key to enable Chat.")

@st.cache_resource
def load_hybrid_system():
    """
    Loads the 3 Heavy Keras Models (Workers) and 3 Light Joblib Files (Managers).
    """
    system = {}
    progress_bar = st.progress(0, text="Initializing Neural Networks...")
    
    try:
        # A. Load Deep Learning Workers
        system['face']   = load_model(os.path.join(BASE_DIR, "Face_EfficientNet.keras"), compile=False)
        progress_bar.progress(30, text="Loading Audio Model...")
        
        system['audio']  = load_model(os.path.join(BASE_DIR, "Audio_EfficientNet_Refined.keras"), compile=False)
        progress_bar.progress(60, text="Loading Fusion Model...")
        
        system['fusion'] = load_model(os.path.join(BASE_DIR, "Model_Fusion.keras"), compile=False, safe_mode=False)
        
        # B. Load Meta-Learners (The Managers)
        progress_bar.progress(80, text="Loading Ensemble Logic...")
        system['meta_xgb'] = joblib.load(os.path.join(BASE_DIR, "meta_xgboost.pkl"))
        system['meta_lr']  = joblib.load(os.path.join(BASE_DIR, "meta_logreg.pkl"))
        system['le']       = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
        
        progress_bar.empty()
        return system

    except Exception as e:
        progress_bar.empty()
        st.error(f"Critical Error Loading Models: {e}")
        st.error(f"Please ensure all 6 model files are in the '{BASE_DIR}' folder.")
        return None

def get_dual_stacking_prediction(system, aud_in, vis_in):
    """
    Implements the exact TTA + Dual Stacking logic from training.
    """
    # 1. Expand dimensions to create batch (1, 224, 224, 3)
    if vis_in.ndim == 3: vis_in = np.expand_dims(vis_in, 0)
    if aud_in.ndim == 3: aud_in = np.expand_dims(aud_in, 0)

    # 2. Face Prediction with TTA (Test Time Augmentation)
    #    Predict on original image
    p_face_normal = system['face'].predict(vis_in, verbose=0)
    
    #    Predict on horizontally flipped image
    vis_in_flipped = np.array([cv2.flip(img, 1) for img in vis_in])
    p_face_flipped = system['face'].predict(vis_in_flipped, verbose=0)
    
    #    Average them
    p_face = (p_face_normal + p_face_flipped) / 2.0 

    # 3. Audio Prediction
    p_audio = system['audio'].predict(aud_in, verbose=0)

    # 4. Fusion Prediction
    #    Note: Keys must match the input layer names of your Fusion model
    fusion_inputs = {"face_input_new": vis_in, "audio_input_new": aud_in}
    p_fusion = system['fusion'].predict(fusion_inputs, verbose=0)

    # 5. Feature Engineering (Must match training exactly)
    #    Base Features: Stack [Fusion, Face, Audio]
    base_features = np.hstack([p_fusion, p_face, p_audio])
    
    #    Agreement Feature: 1.0 if Face and Audio agree, 0.0 otherwise
    agreement = (np.argmax(p_face, axis=1) == np.argmax(p_audio, axis=1)).astype(float).reshape(-1, 1)
    
    #    Max Probability Feature: How confident is the most confident model?
    max_p = np.max(base_features, axis=1, keepdims=True)
    
    #    Final Meta-Vector
    X_meta = np.hstack([base_features, agreement, max_p])

    # 6. Dual Stacking Prediction
    #    Get probabilities from both "Managers"
    prob_xgb = system['meta_xgb'].predict_proba(X_meta)
    prob_lr  = system['meta_lr'].predict_proba(X_meta)

    # 7. Final Soft Voting
    final_probs = (prob_xgb + prob_lr) / 2.0
    
    return final_probs[0] # Return 1D array of probabilities

def extract_inputs(y, sr, video_path, t):
    # Audio Spectrogram
    start, end = int(t * sr), int((t + 3.0) * sr)
    y_seg = y[start:end]
    # Pad if shorter than 3 seconds
    y_seg = np.pad(y_seg, (0, max(0, 48000 - len(y_seg))), 'constant')[:48000]
    
    mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to 0-1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    # Resize to 224x224 and stack to 3 channels
    spec = cv2.resize(np.stack([mel_norm]*3, axis=-1), (224, 224))
    aud_in = eff_preprocess(spec * 255.0) # EfficientNet expects 0-255 range

    # Visual Frame
    cap = cv2.VideoCapture(video_path)
    # Grab frame at t + 1.0s (middle of audio window)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + 1.0) * cap.get(cv2.CAP_PROP_FPS)))
    ret, frame = cap.read()
    cap.release()
    
    vis_in = None
    if ret:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
            vis_in = eff_preprocess(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
    return aud_in, vis_in

def get_coaching_feedback(analysis, chat_history, emotion_labels):
    is_initial = len(chat_history) == 0
    
    if is_initial:
        system_instruction = (
            "You are an Elite Communication Coach using Multimodal AI data. "
            "Give a structured, professional summary of the user's performance. "
            "Use ## Headers for: Executive Summary, Micro-Expression Analysis, and Actionable Tips."
        )
    else:
        system_instruction = "You are an Elite Communication Coach. Be concise, warm, and helpful."

    try:
        model_ai = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=system_instruction)
        
        dominant = analysis['overall']
        radar_str = ", ".join([f"{emo}: {prob:.2f}" for emo, prob in zip(emotion_labels, analysis['all_probs'])])
        
        context = f"ANALYSIS DATA: Dominant Emotion={dominant}. Full Probability Profile: {radar_str}."

        gemini_history = []
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = model_ai.start_chat(history=gemini_history)
        
        if is_initial:
            prompt = f"SYSTEM: The user has uploaded a video. {context}. Provide the initial report."
        else:
            prompt = chat_history[-1]['content']
        
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"**Coach Offline (API Error):** {str(e)}"

st.title("🧘 Personal Emotion Coach")

# Load the full system
system = load_hybrid_system()

# Main Container
with st.container():
    uploaded_file = st.file_uploader("Upload a video clip (mp4/mov)", type=['mp4', 'mov'])
    if uploaded_file:
        st.video(uploaded_file)

if uploaded_file and system and api_key:
    # Get class names dynamically from the loaded encoder
    EMOTIONS = system['le'].classes_
    
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    if st.button("✨ Analyze My Emotion"):
        with st.spinner("Processing video (this may take a moment)..."):
            try:
                # Extract Audio
                clip = VideoFileClip(tfile.name)
                audio_path = tfile.name.replace(".mp4", ".wav")
                clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
                y, sr = librosa.load(audio_path, sr=16000)
                
                results = []
                # Iterate through video
                for t in np.arange(0, clip.duration - 1.0, STRIDE):
                    aud_in, vis_in = extract_inputs(y, sr, tfile.name, t)
                    
                    if vis_in is not None:
                        probs = get_dual_stacking_prediction(system, aud_in, vis_in)
                        
                        results.append({
                            "time": t, 
                            "probs": probs, 
                            "emotion": EMOTIONS[np.argmax(probs)]
                        })
                
                # Aggregate Results
                if results:
                    avg_p = np.mean([x['probs'] for x in results], axis=0)
                    st.session_state.analysis_result = {
                        "overall": EMOTIONS[np.argmax(avg_p)],
                        "timeline": results,
                        "all_probs": avg_p
                    }
                    st.session_state.chat_history = [] # Reset chat
                else:
                    st.error("Could not detect any faces in the video.")
                
                # Cleanup
                clip.close()
                if os.path.exists(audio_path): os.remove(audio_path)
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")


if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    st.divider()

    if 'active_view' not in st.session_state:
        st.session_state.active_view = "📊 Data Insights"

    st.divider()
    col_nav, _ = st.columns([1, 2])
    with col_nav:
        selected_view = st.radio(
            "Select View Mode:",
            ["📊 Data Insights", "📋 Full Report"],
            horizontal=True,
            label_visibility="collapsed",
            key="active_view"
        )

    if selected_view == "📊 Data Insights":
        col_m, col_c = st.columns(2)
        with col_m:
            st.metric("Dominant Expression", res['overall'])
        with col_c:
            # Radar Chart
            fig = go.Figure(data=go.Scatterpolar(
                r=res['all_probs'], theta=EMOTIONS, fill='toself'
            ))
            st.plotly_chart(fig, use_container_width=True)

    elif selected_view == "📋 Full Report":
        # Generate Report 
        if not st.session_state.chat_history:
            with st.spinner("Consulting AI Coach"):
                report = get_coaching_feedback(res, [], EMOTIONS)
                st.session_state.chat_history.append({"role": "assistant", "content": report})
        
        chat_container = st.container()
        with chat_container:
            # Main Report (Styled Box)
            if st.session_state.chat_history:
                st.markdown(f"""
                <div class="report-container">
                    {st.session_state.chat_history[0]['content']}
                </div>
                """, unsafe_allow_html=True)

            # Subsequent Conversation
            for msg in st.session_state.chat_history[1:]:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
            
            # Add some padding at the bottom so the last message isn't hidden by the input bar
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # 3. CHAT INPUT (Placed Globally at the Bottom) 
    if st.session_state.active_view == "📋 Full Report":
        if prompt := st.chat_input("Ask your coach a question"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            

            st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == "user":
        with st.spinner("AI coach is thinking"):
            reply = get_coaching_feedback(res, st.session_state.chat_history, EMOTIONS)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()