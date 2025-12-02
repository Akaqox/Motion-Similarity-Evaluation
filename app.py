import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
from datetime import datetime

# --- IMPORT YOUR MODULES ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference import Inference
from data.data_utils import keypoints_pipeline, clean_invisible_points_single_frame
from data.augment import process_one_file

# --- 1. SETUP & CONFIG ---
st.set_page_config(layout="wide", page_title="Motion Evaluator")

st.markdown("""
    <style>
        .block-container {
            padding-top: 25px; 
        }
        /* Make all buttons bigger */
        div.stButton > button p {
            font-size: 32px !important; /* Force font size */
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_config():
    return Config()

CONF = load_config()

@st.cache_resource
def load_model(_conf):
    inf = Inference(_conf, CONF.get("eval","m_model"))
    return inf

@st.cache_data
def load_reference_data(_conf, ref_list):
    loaded_refs = {}
    for item in ref_list:
        name = item['name']
        vid_path = item.get('video_path')
        if vid_path and os.path.exists(vid_path):
            try:
                data = process_one_file(_conf, vid_path)
                loaded_refs[name] = data
            except Exception as e:
                st.warning(f"⚠️ Could not process {name}: {e}")
    return loaded_refs

def get_webcam():
    for idx in [0, 1]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: return cap
                cap.release()
    return None

# --- STATE MANAGEMENT ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'recorded_data' not in st.session_state:
    st.session_state.recorded_data = None
if 'temp_landmarks' not in st.session_state:
    st.session_state.temp_landmarks = []

# --- CALLBACKS ---
def stop_recording_callback():
    st.session_state.recording = False

def start_recording_callback():
    st.session_state.recording = True
    st.session_state.show_results = False
    st.session_state.recorded_data = None
    st.session_state.temp_landmarks = []

def back_to_camera_callback():
    st.session_state.recording = False
    st.session_state.show_results = False
    st.session_state.recorded_data = None
    st.session_state.temp_landmarks = []

# --- 2. LAYOUT & LOGIC ---
st.markdown("### Search & Compare: Real-time Motion Evaluation")

# [CHANGE] Adjusted ratio here: 1.1 (Main) vs 1.0 (Sidebar)
col_main, col_refs = st.columns([1.1, 1.0])

# --- SIDEBAR (REFERENCES) ---
with col_refs:
    st.subheader("Reference Library")
    app_settings = CONF.get("app_settings") if CONF.get("app_settings") else {}
    ref_list = app_settings.get("references") if app_settings.get("references") else []
    ref_data_map = load_reference_data(CONF, ref_list)
    
    grid_cols = st.columns(2)
    for i, item in enumerate(ref_list):
        with grid_cols[i % 2]:
            st.caption(f"**{item['name']}**")
            vid_path = item.get('video_path', '')
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.video(f.read(), format="video/mp4", muted=True, autoplay=st.session_state.recording)

# --- MAIN AREA ---
with col_main:
    
    # Placeholders
    status_text = st.empty()
    cam_placeholder = st.empty()

    with st.spinner("Loading AI Model..."):
        inf = load_model(CONF)

    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    
    FPS = 30.0
    MIN_DURATION_SEC = 3.0
    MIN_FRAME_COUNT = int(MIN_DURATION_SEC * 15.0) 

    # ==========================================
    # LOGIC: CHECK FOR UNPROCESSED DATA
    # ==========================================
    if not st.session_state.recording and len(st.session_state.temp_landmarks) > 0:
        if len(st.session_state.temp_landmarks) > MIN_FRAME_COUNT:
            st.session_state.recorded_data = np.array(st.session_state.temp_landmarks)
            st.session_state.show_results = True
        else:
            st.warning(f"Recording too short (< {MIN_DURATION_SEC}s). Please try again.")
            st.session_state.show_results = False
            st.session_state.recorded_data = None
        
        st.session_state.temp_landmarks = []

    # ==========================================
    # PHASE 1: RECORDING
    # ==========================================
    if st.session_state.recording:
        time.sleep(0.2)
        cap = get_webcam()
        if not cap:
            st.error("🚨 Camera error.")
            st.session_state.recording = False
            st.stop()

        save_dir = "data/recorded_sessions"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(save_dir, f"session_{timestamp}.mp4")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
        
        max_duration = 14.0
        start_time = time.time()
        
        st.button("⏹️ Stop Recording", type="primary", on_click=stop_recording_callback)
        progress_bar = st.progress(0.0)

        try:
            while True:
                if not st.session_state.recording: break

                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    st.session_state.recording = False
                    st.rerun() 
                    break
                
                progress_bar.progress(min(elapsed / max_duration, 1.0))
                if elapsed < MIN_DURATION_SEC:
                    status_text.warning(f"Keep moving... {int(MIN_DURATION_SEC - elapsed)}s more needed.")
                else:
                    status_text.success(f"Recording... {int(max_duration - elapsed)}s left")

                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                if out_writer.isOpened(): out_writer.write(frame)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_pose.process(image_rgb)
                
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                
                cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                
                if results.pose_world_landmarks:
                    lm = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_world_landmarks.landmark])
                    lm = clean_invisible_points_single_frame(lm)
                    nose_pos = lm[0, :3]
                    lm[:, :3] -= nose_pos
                    st.session_state.temp_landmarks.append(lm)
        
        finally:
            cap.release()
            out_writer.release()

    # ==========================================
    # PHASE 2: RESULTS (Camera OFF)
    # ==========================================
    elif st.session_state.show_results:
        st.button("🔄 Back to Camera", on_click=back_to_camera_callback)
        
        st.success("Analysis Complete!")
        
        if st.session_state.recorded_data is not None:
            with st.spinner("Calculating Scores..."):
                input_seq = st.session_state.recorded_data 
                processed_data = keypoints_pipeline(
                    CONF.get("data"), input_seq, angle_rad=0, n_features=CONF.get("ae").get("N_FEATURES")
                )
                
                all_scores = []
                for name, ref_seq in ref_data_map.items():
                    if len(ref_seq.shape) == 2: ref_seq = ref_seq[np.newaxis, ...]
                    try:
                        score, _, _ = inf.dtw_similarity(processed_data, ref_seq, vis=False)
                        all_scores.append((name, score))
                    except Exception as e:
                        pass

                if all_scores:
                    all_scores.sort(key=lambda x: x[1], reverse=True)
                    best_match, best_score = all_scores[0]
                    threshold = app_settings.get("threshold", 0.75)
                    
                    st.divider()
                    if best_score > threshold:
                        st.success(f"### Best Match: {best_match} ({best_score:.2f})")
                    else:
                        st.error(f"### Invalid Movement (Best: {best_match} @ {best_score:.2f})")
                    
                    st.write("**Detailed Comparison:**")
                    res_cols = st.columns(len(all_scores))
                    for i, (name, score) in enumerate(all_scores):
                        with res_cols[i]:
                            st.metric(name, f"{score:.2f}", delta=f"{score-threshold:.2f}")
                            st.progress(min(score, 1.0))
                            
    # ==========================================
    # PHASE 3: PREVIEW (Idle)
    # ==========================================
    else:
        st.button("🔴 Start Recording (Max 14s)", type="primary", on_click=start_recording_callback)
        
        time.sleep(0.5) 
        cap = get_webcam()
        if cap:
            try:
                while True:
                    if st.session_state.recording: break
                    
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mp_pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    
                    cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
            finally:
                cap.release()
            
            if st.session_state.recording:
                st.rerun()