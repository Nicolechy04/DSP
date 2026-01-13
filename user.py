import streamlit as st
import os
import cv2
import numpy as np
import librosa
import tempfile
import pandas as pd
import joblib   
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px  
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

st.set_page_config(page_title="Personal Emotion Coach", page_icon="🧘", layout="wide")

# CSS
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

if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

with st.sidebar:
    st.title("Navigation")
    
    # Initialize the current page in session state if it doesn't exist
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "📂 Project Overview"
    
    # Create full-width buttons for navigation
    # When a button is clicked, we update the session state
    if st.button("📂 Project Overview", use_container_width=True):
        st.session_state.page_selection = "📂 Project Overview"
        
    if st.button("📈 Result Evaluation", use_container_width=True):
        st.session_state.page_selection = "📈 Result Evaluation"
        
    if st.button("🚀 Demo", use_container_width=True):
        st.session_state.page_selection = "🚀 Demo"

    # Assign the session state to your variable so the rest of the code works
    page_selection = st.session_state.page_selection
    
    
    # Check for Gemini API Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        st.success("API Key is Set Up")
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
    Loads all the models.
    """
    system = {}
    
    try:
        system['face']   = load_model(os.path.join(BASE_DIR, "efficientnet_improved.keras"), compile=False)
        system['audio']  = load_model(os.path.join(BASE_DIR, "Audio_EfficientNet_Refined.keras"), compile=False)
        system['fusion'] = load_model(os.path.join(BASE_DIR, "modelfusion_2.keras"), compile=False, safe_mode=False)
        system['meta_xgb'] = joblib.load(os.path.join(BASE_DIR, "meta_xgboost.pkl"))
        system['meta_lr']  = joblib.load(os.path.join(BASE_DIR, "meta_logreg.pkl"))
        system['le']       = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
        return system
    except Exception as e:
        return None

def get_dual_stacking_prediction(system, aud_in, vis_in):
    if vis_in.ndim == 3: vis_in = np.expand_dims(vis_in, 0)
    if aud_in.ndim == 3: aud_in = np.expand_dims(aud_in, 0)

    p_face = system['face'].predict(vis_in, verbose=0)
    
    p_audio = system['audio'].predict(aud_in, verbose=0)

    fusion_inputs = {"face_input_new": vis_in, "audio_input_new": aud_in}
    p_fusion = system['fusion'].predict(fusion_inputs, verbose=0)

    base_features = np.hstack([p_fusion, p_face, p_audio])
    
    agreement = (np.argmax(p_face, axis=1) == np.argmax(p_audio, axis=1)).astype(float).reshape(-1, 1)
    
    max_p = np.max(base_features, axis=1, keepdims=True)
    
    X_meta = np.hstack([base_features, agreement, max_p])

    prob_xgb = system['meta_xgb'].predict_proba(X_meta)
    prob_lr  = system['meta_lr'].predict_proba(X_meta)

    final_probs = (prob_xgb + prob_lr) / 2.0
    
    return final_probs[0]

def extract_inputs(y, sr, video_path, t):
    start, end = int(t * sr), int((t + 3.0) * sr)
    y_seg = y[start:end]
    y_seg = np.pad(y_seg, (0, max(0, 48000 - len(y_seg))), 'constant')[:48000]
    
    mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    spec = cv2.resize(np.stack([mel_norm]*3, axis=-1), (224, 224))
    aud_in = eff_preprocess(spec * 255.0) 

    cap = cv2.VideoCapture(video_path)
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
    
    # --- 1. DEFINE THE STRICT REPORT FORMAT (For the first analysis only) ---
    report_instruction = (
            "You are an Elite Communication Coach. Analyze the user's non-verbal cues based on the data provided. "
            "Strictly follow the structure below:\n\n"

            "Write 2-3 insightful sentences explaining the dominant emotion. Discuss how this specific vibe affects audience perception.\n\n"
          
            "## Micro-Expression Analysis\n"
            "List ONLY the top 3 emotions with the highest probabilities. Use this exact bullet format:\n"
            "* **[Emotion Name] ([Percentage]%)**: [One sentence explaining the probable facial expressions and vocal tones shown].\n\n"

            "## Actionable Tips\n"
            "Provide 5 specific coaching tips. Format strictly as '**Topic**: [Action]. [Benefit].'\n"
            "IMPORTANT - Tailor your advice based on the dominant emotion:\n"
            "- If **Happy**: Focus on **CHANNELING** (use the enthusiasm to inspire the audience, but ensure you don't look manic).\n"
            "- If **Anger/Fear/Disgust**: Focus on **REGULATION** (calm the intensity to appear composed and professional).\n"
            "- If **Sad**: Focus on **ELEVATION** (you likely look low-energy or resigned; focus on lifting your posture, voice, and energy to show resilience).\n"
            "- If **Surprise**: Focus on **RECOVERY** (surprise should be fleeting; focus on pivoting quickly from shock to curiosity/analysis).\n"
            "- If **Neutral**: Focus on **AMPLIFICATION** (add vocal variety and facial animation so you do not appear bored or robotic).\n\n"

            "1. [Tip to adjust posture/face]. [How this improves presence].\n"
            "2. [Tip on speed/volume]. [How this ensures clarity].\n"
            "3. [A phrase to say]. [How this gives context to your expression].\n"
            "4. [Internal thought]. [How this aligns intent with impact].\n"
            "5. [Body language tip]. [How this connects you with the listener]."
    )

    chat_persona = (
        "You are an Elite Communication Coach. The user has already received their initial video analysis report. "
        "Now, you are in a conversational mode. Answer their follow-up questions naturally, concisely, and warmly. "
        "Do NOT repeat the full report format (Executive Summary, etc.) unless specifically asked to re-analyze. "
        "Focus on giving specific, bite-sized advice based on the context of their questions."
    )

    try:
        current_instruction = report_instruction if is_initial else chat_persona

        model_ai = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=current_instruction)
        
        dominant = analysis['overall']
        radar_str = ", ".join([f"{emo}: {prob:.2f}" for emo, prob in zip(emotion_labels, analysis['all_probs'])])
        context_str = f"ANALYSIS DATA: Dominant Emotion={dominant}. Full Probability Profile: {radar_str}."

        gemini_history = []
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = model_ai.start_chat(history=gemini_history)
        
        if is_initial:

            prompt = f"SYSTEM: The user has uploaded a video. {context_str}. Provide the initial report."
        else:
            prompt = chat_history[-1]['content']
        
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"**Coach Offline (API Error):** {str(e)}"

st.title("🧘 Personal Emotion Coach")

#PROJECT OVERVIEW
if page_selection == "📂 Project Overview":
    st.subheader("DSP Title: Emotion-Based Multimodal Conversational Feedback")
    st.caption("Student: Chua Hui Ying Nicole | Supervisor: Dr. Hoo Wai Lam")
    st.divider()

    st.subheader("1. 🌍 Context & Introduction")
    st.write("""
    Over **970 million people** globally suffer from mental health disorders. A major, overlooked driver is **ineffective communication**.
    
    Misunderstanding one's own non-verbal cues (e.g., aggressive faces, dismissive tones) creates a cycle of **social friction and self-doubt**.
    """)
    
    st.success("🎯 **Goal:** Develop multimodal emotion detection with actionable tips to improve non-verbal cues.")

    st.divider()

    st.subheader("2. 🚩 Problem Statement")
    st.write("We are addressing three critical limitations in current communication tools:")

    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.info("**Lack of Self-Awareness**")
        st.caption("People often don't realize how their tone or face looks, leading to **accidental misunderstandings**.")

    with col_p2:
        st.warning("**Unimodal Reliance**")
        st.caption("Systems using *only* face or *only* voice fail easily. Real-world robustness requires **multimodal fusion**.")

    with col_p3:
        st.error("**No Actionable Feedback**")
        st.caption("Detecting emotion isn't enough. Users need **actionable advice** (via LLMs) to actually improve.")

    st.divider()

    st.subheader("3. 🎯 Objectives")
    st.markdown("""
    * To develop multimodal emotion recognition using late fusion strategy. 
    * To evaluate the classification performance of all developed models using metrics.
    * To implement a post-analysis feedback system by utilizing Large Language Model (LLM) to generate feedback that enable users to understand their non-verbal cues.
    """)
    
    st.divider()

    st.subheader("4. 💾 Dataset & Methodology")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**📂 Datasets**")
        st.markdown("""
        * **RAVDESS**
        * **CREMA-D**
        """)
        
    with col_d2:
        st.markdown("**⚙️ Preprocessing**")
        st.markdown("""
        * **🎤 Audio:** Extracted using MoviePy.
        * **📷 Facial:** Detection using Haar Cascade.
        """)

    st.info("**7 Emotions Class:** Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise")
    
    st.divider()

    st.subheader("5. 🏗️ Architecture Used")
    st.write("The system utilizes a **Late Fusion** approach:")

    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        st.markdown("### Level 0️⃣: Backbones") 
        st.markdown("**Feature Extraction**")
        st.markdown("""
        * **📷 Facial:** EfficientNet
        * **🎤 Audio:** EfficientNet
        """)

    with col_a2:
        st.markdown("### Level 1️⃣: Fusion") 
        st.markdown("**Late Fusion Mechanism**")
        st.markdown("""
        * **Gated Fusion:** Dynamically weights modalities.
        * **Logic:** `(Face * z) + (Audio * (1-z))`
        """)

    with col_a3:
        st.markdown("### Level 2️⃣: Stacking") 
        st.markdown("**Meta-Classifiers**")
        st.markdown("""
        * **Ensemble:** XGBoost + Logistic Regression
        * **Final Output:** Weighted Consensus
        """)

#RESULT EVALUATION
elif page_selection == "📈 Result Evaluation":
    st.subheader("📊 Model Performance Evaluation")
    
    st.markdown("#### 1. Classification Performance")
    st.write("Comparison of unimodal (facial and audio model) and multimodal.")

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.metric(label="😃 Facial Accuracy", value="53%", delta=" ") 
        " "
        
    with m_col2:
        st.metric(label="🎤 Audio Accuracy", value="55%", delta=" ")
        " "
        
    with m_col3:
        st.metric(label="🧩 Fusion Accuracy", value="72%", delta="+17-19% Improvement")

    st.write("") 

    data_metrics = {
        "Model": ["Facial", "Facial", "Facial", 
                  "Audio", "Audio", "Audio", 
                  "Fusion", "Fusion", "Fusion"],
        "Metric": ["Precision", "Recall", "F1 Score", 
                   "Precision", "Recall", "F1 Score", 
                   "Precision", "Recall", "F1 Score"],
        "Score": [0.57, 0.53, 0.52,  # Facial 
                  0.57, 0.55, 0.55,  # Audio 
                  0.73, 0.72, 0.72]  # Fusion 
    }
    df_metrics = pd.DataFrame(data_metrics)

    fig_metrics = px.bar(
        df_metrics, 
        x="Metric", 
        y="Score", 
        color="Model", 
        barmode="group",
        text="Score",
        color_discrete_sequence=["#B2BEC3", "#74B9FF", "#6C5CE7"],
        title="Detailed Metrics Comparison"
    )
    fig_metrics.update_layout(yaxis_range=[0, 1])
    
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.divider()

    st.markdown("#### 2. Confusion Matrix")
    st.write("Visualizing the true positive rates across different emotion classes.")
    
    img_col1, img_col2, img_col3 = st.columns(3)
    
    def show_cm_image(path, caption, col):
        with col:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True) 
            else:
                st.warning(f"Image not found: {path}")
                st.caption(f"Please save '{path}' in your project folder.")
                
    show_cm_image("assets/facial.png", "Facial Model", img_col1)
    show_cm_image("assets/audio.png", "Audio Model", img_col2)
    show_cm_image("assets/output.png", "Fusion Model", img_col3)

    st.divider()

    st.markdown("#### 3. LLM-As-A-Judge Evaluation")
    st.write("We compared different LLMs (Llama-3, Gemini-2.5, Zephyr) using GPT-5.2 and Gemini Pro as judges.")

    llm_data_gemini = {
        "Model": ["Llama-3-8B", "Gemini-2.5-Flash", "Zephyr-7B"],
        "Faithfulness": ["4/5", "3.5/5", "2/5"],
        "Helpfulness": ["3/5", "5/5", "2/5"],
        "Formatting": ["4/5", "5/5", "2/5"]
    }
    
    llm_data_gpt = {
        "Model": ["Llama-3-8B", "Gemini-2.5-Flash", "Zephyr-7B"],
        "Faithfulness": ["3.5/5", "4.5/5", "2.5/5"],
        "Helpfulness": ["3/5", "4.5/5", "3.5/5"],
        "Formatting": ["4/5", "5/5", "3/5"]
    }

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.markdown("**Evaluator: Gemini Pro**")
        st.dataframe(pd.DataFrame(llm_data_gemini), hide_index=True)
        st.caption("Gemini Pro favored Gemini-Flash for Helpfulness.")

    with ev_col2:
        st.markdown("**Evaluator: GPT-5.2**")
        st.dataframe(pd.DataFrame(llm_data_gpt), hide_index=True)
        st.caption("GPT-5.2 consistently rated Gemini-Flash highest.")

    st.success("Conclusion: **Gemini-2.5-Flash** was selected for the final system due to superior Helpfulness (5/5) and Formatting scores.")

#DEMO
elif page_selection == "🚀 Demo":
    system = load_hybrid_system()
    
    if system is None:
        st.error("System failed to load. Please check 'models_zoo_1' folder.")
    else:
        EMOTIONS = system['le'].classes_

        with st.container():
            uploaded_file = st.file_uploader("Upload a video clip (mp4/mov)", type=['mp4', 'mov'])
            if uploaded_file:
                st.video(uploaded_file)

        if uploaded_file and api_key: 
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            if st.button("✨ Analyze My Emotion"):
                with st.spinner("Processing video (this may take a moment)..."):
                    try:
                        clip = VideoFileClip(tfile.name)
                        audio_path = tfile.name.replace(".mp4", ".wav")
                        clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
                        y, sr = librosa.load(audio_path, sr=16000)
                        
                        results = []
                        for t in np.arange(0, clip.duration - 1.0, STRIDE):
                            aud_in, vis_in = extract_inputs(y, sr, tfile.name, t)
                            
                            if vis_in is not None:
                                probs = get_dual_stacking_prediction(system, aud_in, vis_in)
                                
                                results.append({
                                    "time": t, 
                                    "probs": probs, 
                                    "emotion": EMOTIONS[np.argmax(probs)]
                                })
                        
                        if results:
                            avg_p = np.mean([x['probs'] for x in results], axis=0)
                            st.session_state.analysis_result = {
                                "overall": EMOTIONS[np.argmax(avg_p)],
                                "timeline": results,
                                "all_probs": avg_p
                            }
                            st.session_state.chat_history = [] 
                        else:
                            st.error("Could not detect any faces in the video.")
                        
                        clip.close()
                        if os.path.exists(audio_path): os.remove(audio_path)
                        
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")

        if st.session_state.analysis_result:
                    res = st.session_state.analysis_result
                    st.write("") 

                    if "demo_view" not in st.session_state:
                        st.session_state.demo_view = "📊 Data Insights"

                    st.radio(
                        "Select View:",
                        ["📊 Data Insights", "🤖 AI Coach Chat"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key="demo_view"
                    )
                    
                    st.write("") 

                    if st.session_state.demo_view == "📊 Data Insights":
                        st.divider() 
                        col_m, col_c = st.columns(2)
                        
                        emoji_map = {
                            "neutral": "😐", "happy": "😄", "sad": "😔",
                            "angry": "😠", "fear": "😨", "disgust": "🤢", "surprise": "😲"
                        }

                        raw_emotion = res['overall'].lower()
                        clean_emotion = raw_emotion.title()
                        emoji = emoji_map.get(raw_emotion, "🤔")
                        display_text = f"{emoji} {clean_emotion}"

                        with col_m:
                            st.metric("Dominant Expression", display_text)
                            st.info("💡 Tip: Switch to the **AI Coach Chat** tab to get your full report!")
                            
                        with col_c:
                            fig = go.Figure(data=go.Scatterpolar(
                                r=res['all_probs'], theta=EMOTIONS, fill='toself'
                            ))
                            st.plotly_chart(fig, use_container_width=True)

                    elif st.session_state.demo_view == "🤖 AI Coach Chat":
                        
                        st.markdown("#### 📋 Executive Coaching Report")
                        

                        if not st.session_state.chat_history:
                            with st.spinner("Consulting AI Coach..."):
                                report = get_coaching_feedback(res, [], EMOTIONS)
                                st.session_state.chat_history.append({"role": "assistant", "content": report})
                        
                        if st.session_state.chat_history:

                            st.markdown(f"""
                            <div class="report-container">
                                {st.session_state.chat_history[0]['content']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("") 

                            for msg in st.session_state.chat_history[1:]:
                                with st.chat_message(msg['role']):
                                    st.write(msg['content'])

                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                        if prompt := st.chat_input("Ask your coach a question..."):
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            st.rerun()

                    if st.session_state.demo_view == "🤖 AI Coach Chat" and st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == "user":
                        with st.spinner("AI coach is thinking..."):
                            reply = get_coaching_feedback(res, st.session_state.chat_history, EMOTIONS)
                            st.session_state.chat_history.append({"role": "assistant", "content": reply})
                            st.rerun()