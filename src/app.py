import streamlit as st
import os
import shutil
import pandas as pd
import copy
import json

# Import our pipeline functions
from pipeline.step01_extraction import run_extraction
from pipeline.step02_stt import run_stt
from pipeline.step03_diarization import run_diarization
from pipeline.step03b_proofread import run_proofread
from pipeline.step04_transcreation import run_transcreation
from pipeline.step05_voice_generation import run_voice_generation, extract_reference_audio, run_seed_vc_conversion
from pipeline.step06_audio_sync import run_audio_sync
from pipeline.step07_final_mix import run_final_mix

import importlib
import sys
importlib.reload(sys.modules['pipeline.step02_stt'])
importlib.reload(sys.modules['pipeline.step05_voice_generation'])
from pipeline.step05_voice_generation import run_voice_generation, extract_reference_audio, run_seed_vc_conversion
from pipeline.step02_stt import run_stt

if 'pipeline.step03_diarization' in sys.modules:
    importlib.reload(sys.modules['pipeline.step03_diarization'])
if 'pipeline.step06_audio_sync' in sys.modules:
    importlib.reload(sys.modules['pipeline.step06_audio_sync'])

# -----------------
# State Management
# -----------------
from pipeline.utils import save_project_state, load_project_state

if "state_loaded" not in st.session_state:
    saved_state = load_project_state()
    if saved_state:
        for k, v in saved_state.items():
            st.session_state[k] = v
    st.session_state.state_loaded = True

def trigger_save():
    save_keys = ["step", "video_path", "vocals_path", "bgm_path", "stt_segments", "diarized_segments", "working_segments", "scene_context", "translated_segments", "generated_segments", "base_audio_generated", "seed_vc_complete", "final_video_path"]
    state_to_save = {k: st.session_state[k] for k in save_keys if k in st.session_state}
    save_project_state(state_to_save)

if "step" not in st.session_state:
    st.session_state.step = 0
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "vocals_path" not in st.session_state:
    st.session_state.vocals_path = None
if "base_audio_generated" not in st.session_state:
    st.session_state.base_audio_generated = False
if "bgm_path" not in st.session_state:
    st.session_state.bgm_path = None
if "stt_segments" not in st.session_state:
    st.session_state.stt_segments = None
if "diarized_segments" not in st.session_state:
    st.session_state.diarized_segments = None
if "working_segments" not in st.session_state:
    st.session_state.working_segments = None
if "scene_context" not in st.session_state:
    st.session_state.scene_context = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None

importlib.reload(sys.modules['pipeline.step04_transcreation'])
from pipeline.step04_transcreation import run_transcreation, run_context_analysis
if "proofread_suggestions" not in st.session_state:
    st.session_state.proofread_suggestions = None
if "translated_segments" not in st.session_state:
    st.session_state.translated_segments = None
if "generated_segments" not in st.session_state:
    st.session_state.generated_segments = None
if "seed_vc_complete" not in st.session_state:
    st.session_state.seed_vc_complete = False
if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None

OUTPUT_DIR = "data/processed"
os.makedirs("data/raw", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------
# UI Layout
# -----------------
st.set_page_config(page_title="NaturalDub-AI Interactive Director", layout="wide")
st.title("🎬 NaturalDub-AI: Interactive Director")

st.markdown(f"**Current Pipeline Phase: {st.session_state.step} / 4**")
st.progress(st.session_state.step / 4.0)
st.divider()

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = None
if "sarvam_api_key" not in st.session_state:
    st.session_state.sarvam_api_key = None

st.sidebar.header("API Keys")
st.session_state.openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=st.session_state.openrouter_api_key or "")
st.session_state.sarvam_api_key = st.sidebar.text_input("Sarvam API Key", type="password", value=st.session_state.sarvam_api_key or "")
st.session_state.groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key or "")
st.session_state.gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key or "")
st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key or "")
st.session_state.anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password", value=st.session_state.anthropic_api_key or "")

st.sidebar.divider()
if st.sidebar.button("🗑️ Start New Project (Clear Cache)", type="primary"):
    if os.path.exists("data/processed/project_state.json"):
        os.remove("data/processed/project_state.json")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Phase Navigation")
if st.session_state.step > 0:
    if st.sidebar.button("⏪ Go to Phase 1 (Audio Extraction)"):
        st.session_state.step = 1
        trigger_save()
        st.rerun()
if st.session_state.step > 1:
    if st.sidebar.button("⏪ Go to Phase 2 (Transcription)"):
        st.session_state.step = 2
        trigger_save()
        st.rerun()
if st.session_state.step > 2:
    if st.sidebar.button("⏪ Go to Phase 3 (Transcreation)"):
        st.session_state.step = 3
        trigger_save()
        st.rerun()
if st.session_state.step > 3:
    if st.sidebar.button("⏪ Go to Phase 4 (Voice Gen)"):
        st.session_state.step = 4
        trigger_save()
        st.rerun()

# --- STEP 0: Upload ---
if st.session_state.step == 0:
    st.header("Phase 0: Source Video & Script")
    uploaded_video = st.file_uploader("Upload your English MP4 video", type=["mp4"])
    
    dubbing_mode = st.radio("Select Dubbing Mode", ["Mode 1: Automatic Drag & Drop (YouTube/Podcasts)", "Mode 2: Studio Alignment (Upload Script)"], index=0)
    
    uploaded_script = None
    if "Mode 2" in dubbing_mode:
        uploaded_script = st.file_uploader("Upload Original English Script", type=["txt", "srt"])
    
    if uploaded_video is not None:
        video_path = os.path.join("data/raw", uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.session_state.video_path = video_path
        st.video(video_path)
        
        # Save script if uploaded
        if uploaded_script is not None:
            script_path = os.path.join("data/raw", uploaded_script.name)
            with open(script_path, "wb") as f:
                f.write(uploaded_script.getbuffer())
            st.session_state.script_path = script_path
            st.session_state.dubbing_mode = "studio"
        else:
            st.session_state.script_path = None
            st.session_state.dubbing_mode = "auto"
            
        if st.button("Proceed to Phase 1: Audio Extraction ➡️"):
            st.session_state.step = 1
            trigger_save()
            st.rerun()

# --- STEP 1: Extraction ---
elif st.session_state.step == 1:
    st.header("Phase 1: Audio Extraction (MDX-Net)")
    
    if st.session_state.vocals_path is None:
        with st.spinner("Extracting vocals and background music... This may take a moment."):
            paths = run_extraction(st.session_state.video_path, OUTPUT_DIR)
            st.session_state.vocals_path = paths["vocals"]
            st.session_state.bgm_path = paths["bgm"]
            trigger_save()
            st.rerun()
            
    col_succ, col_btn = st.columns([4, 1])
    with col_succ:
        st.success("Extraction Complete! Please review the separated tracks.")
    with col_btn:
        if st.button("🔄 Re-Run Extraction"):
            st.session_state.vocals_path = None
            st.session_state.bgm_path = None
            st.session_state.stt_segments = None
            st.session_state.diarized_segments = None
            st.session_state.working_segments = None
            st.session_state.translated_segments = None
            st.session_state.base_audio_generated = False
            st.session_state.seed_vc_complete = False
            st.session_state.final_video_path = None
            st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Isolated Vocals")
        st.audio(st.session_state.vocals_path)
    with col2:
        st.subheader("Isolated Background Music")
        st.audio(st.session_state.bgm_path)
        
    if st.button("Approve Separation & Run STT", type="primary"):
        st.session_state.step = 2
        trigger_save()
        st.rerun()

# --- STEP 2: STT & Diarization ---
elif st.session_state.step == 2:
    st.header("Phase 2: Transcription & Diarization")
    
    stt_engine_ui = st.selectbox("Speech-to-Text Engine", options=["Local (Apple Silicon mlx-whisper)", "Cloud (Groq whisper-large-v3)"], index=0, help="Local is private but uses Mac CPU/GPU. Cloud is near-instant, uses 0 RAM, and is 100% accurate.")
    stt_engine_key = "local" if "Local" in stt_engine_ui else "groq"
    
    groq_api_key = st.session_state.groq_api_key
    
    if st.session_state.working_segments is None:
        if st.session_state.get("dubbing_mode") == "studio":
            st.info("🎬 **Studio Alignment Mode Active!**\nYour uploaded script will be used to perfectly align the dialogue and map character names.")
            st.session_state.speaker_mapping_mode = st.radio(
                "Speaker Mapping Method:",
                ["Hybrid Audio-Script (Uses Pyannote audio cues for generic scripts)", "LLM Only (Best if your script has explicit character names)"],
                index=0
            )
            
        use_stt_cache = st.checkbox("Use Cached Transcription (Fast)", value=False, help="Enable this to skip Whisper/Groq and load the previously saved transcription from disk.")
        
        if st.button("🎙️ Run STT & Diarization", type="primary"):
            if stt_engine_key == "groq" and not groq_api_key and not use_stt_cache:
                st.error("Please enter a Groq API Key to use Cloud STT.")
                st.stop()
                
            with st.spinner(f"Running STT ({stt_engine_key.upper()}) & Pyannote..."):
                try:
                    if use_stt_cache and os.path.exists(f"{OUTPUT_DIR}/diarized_cache.json"):
                        print("Loading STT from cache...")
                        try:
                            with open(f"{OUTPUT_DIR}/diarized_cache.json", "r") as f:
                                cached_data = json.load(f)
                                st.session_state.stt_segments = cached_data.get("stt")
                                st.session_state.diarized_segments = cached_data.get("diarized")
                                st.session_state.working_segments = cached_data.get("working")
                        except json.JSONDecodeError:
                            st.error("The transcription cache file is empty or corrupted. Please uncheck 'Use Cached Transcription' and run again to generate a fresh transcription.")
                            st.stop()
                    else:
                        stt_segments = run_stt(st.session_state.vocals_path, engine=stt_engine_key, api_key=groq_api_key)
                        diarized = run_diarization(st.session_state.vocals_path, stt_segments)
                        
                        working = diarized
                        # Studio Alignment Mode
                        if st.session_state.get("dubbing_mode") == "studio" and st.session_state.get("script_path"):
                            if not groq_api_key:
                                st.error("Groq API Key is required for Studio Alignment Mode.")
                                st.stop()
                            from pipeline.step02b_alignment import run_groq_align
                            with open(st.session_state.script_path, "r", encoding="utf-8") as f:
                                script_text = f.read()
                            st.info("Studio Mode: Aligning script to audio timestamps...")
                            mapping_mode = "hybrid" if "Hybrid" in st.session_state.speaker_mapping_mode else "llm"
                            diarized = run_groq_align(script_text, diarized, groq_api_key, mapping_mode=mapping_mode)
                            
                        st.session_state.diarized_segments = diarized
                        
                        # Save to disk cache for future runs
                        with open(f"{OUTPUT_DIR}/diarized_cache.json", "w") as f:
                            json.dump({
                                "stt": stt_segments,
                                "diarized": diarized,
                                "working": working
                            }, f, indent=4)
                            
                    st.session_state.working_segments = copy.deepcopy(st.session_state.diarized_segments)
                    st.session_state.proofread_suggestions = None
                    trigger_save()
                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription/Alignment failed: {e}")
        st.stop()
    else:
        col_succ, col_btn = st.columns([4, 1])
        with col_succ:
            st.success("Transcription Complete! Review the detected speakers and timing.")
        with col_btn:
            if st.button("🔄 Re-Run Transcription"):
                st.session_state.working_segments = None
                st.session_state.translated_segments = None
                st.session_state.base_audio_generated = False
                st.session_state.seed_vc_complete = False
                st.session_state.final_video_path = None
                st.rerun()
        
        # Export Script functionality
        script_export = ""
        for seg in st.session_state.working_segments:
            script_export += f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}\n"
            
        st.download_button(
            label="⬇️ Download Aligned Script (.txt)",
            data=script_export,
            file_name="aligned_script.txt",
            mime="text/plain"
        )
    
    col_ai, col_spacer = st.columns([1, 4])
    with col_ai:
        if st.button("🤖 Scan for Errors (LLM)", type="primary"):
            with st.spinner("LLM Proofreader scanning script..."):
                suggestions = run_proofread(st.session_state.working_segments)
                st.session_state.proofread_suggestions = suggestions
    
    if st.session_state.proofread_suggestions:
        st.info(f"AI Proofreader found {len(st.session_state.proofread_suggestions)} suggestions. See highlights below.")
        
    st.divider()
    
    to_delete = None
    to_split = None
    
    for i, seg in enumerate(st.session_state.working_segments):
        if st.session_state.proofread_suggestions:
            for sugg in st.session_state.proofread_suggestions:
                if sugg.get("segment_index") == i:
                    st.warning(f"🤖 **AI Proofreader Flag:** {sugg.get('suggestion')}")
                    break
                    
        col_meta, col_spk, col_txt, col_btn1, col_btn2 = st.columns([2, 2, 4, 1, 1])
        with col_meta:
            new_start = st.number_input("Start", value=float(seg['start']), format="%.2f", step=0.1, key=f"start_{i}")
            new_end = st.number_input("End", value=float(seg['end']), format="%.2f", step=0.1, key=f"end_{i}")
        with col_spk:
            new_spk = st.text_input("Speaker (Add [F] for Female)", value=seg.get('speaker', 'UNKNOWN'), key=f"spk_{i}")
        with col_txt:
            new_txt = st.text_input("Transcript", value=seg.get('text', ''), key=f"txt_{i}")
        with col_btn1:
            if st.button("✂️ Split", key=f"split_{i}"):
                to_split = i
        with col_btn2:
            if st.button("🗑️ Del", key=f"del_{i}"):
                to_delete = i
                
        seg['start'] = new_start
        seg['end'] = new_end
        seg['speaker'] = new_spk
        seg['text'] = new_txt
        
    if to_delete is not None:
        st.session_state.working_segments.pop(to_delete)
        # Clear Streamlit's widget cache so deleted elements don't get stuck
        for key in list(st.session_state.keys()):
            if key.startswith("start_") or key.startswith("end_") or key.startswith("spk_") or key.startswith("txt_"):
                del st.session_state[key]
        st.rerun()
        
    if to_split is not None:
        new_seg = copy.deepcopy(st.session_state.working_segments[to_split])
        mid = (new_seg['start'] + new_seg['end']) / 2
        st.session_state.working_segments[to_split]['end'] = mid
        new_seg['start'] = mid
        st.session_state.working_segments.insert(to_split + 1, new_seg)
        for key in list(st.session_state.keys()):
            if key.startswith("start_") or key.startswith("end_") or key.startswith("spk_") or key.startswith("txt_"):
                del st.session_state[key]
        st.rerun()
        
    st.divider()
    
    if st.button("Approve Transcript & Run Transcreation", type="primary"):
        st.session_state.diarized_segments = copy.deepcopy(st.session_state.working_segments)
        st.session_state.step = 3
        trigger_save()
        st.rerun()

# --- STEP 3: Transcreation (Editable) ---
elif st.session_state.step == 3:
    st.header("Phase 3: The Dubbing Director (Transcreation)")
    
    col_model, col_words = st.columns(2)
    with col_model:
        selected_model = st.selectbox("LLM Engine", options=["Google (Gemini 3.5 Flash)", "Google (Gemini Flash Latest)", "Google (Gemini Pro Latest)", "Cloud (OpenRouter Gemma 3 27B)", "Cloud (Groq LLaMA-3.3-70B)", "Anthropic (Claude 3.5 Sonnet)", "OpenAI (GPT-4o)", "phi4-mini", "llama3.2", "gemma2", "mistral"], index=0, help="Choose the LLM to use for translation. We recommend Google Gemini for Multi-Stage Transcreation.")
    with col_words:
        protected_words = st.text_input("Protected Words", value="", help="Comma separated list of words the AI should NEVER translate (e.g., FBI, Suit, Hack).")
        
    st.subheader("Scene Context & Lore (Auto-Analyzer)")
    if st.session_state.scene_context is None:
        if st.button("🔍 Auto-Analyze Scene Tone", type="primary"):
            if "Groq" in selected_model and not st.session_state.groq_api_key:
                st.error("Groq API Key is missing. Please check the sidebar.")
                st.stop()
            if "OpenAI" in selected_model and not st.session_state.openai_api_key:
                st.error("OpenAI API Key is missing. Please check the sidebar.")
                st.stop()
            if "Anthropic" in selected_model and not st.session_state.anthropic_api_key:
                st.error("Anthropic API Key is missing. Please check the sidebar.")
                st.stop()
            if "Google" in selected_model and not st.session_state.gemini_api_key:
                st.error("Gemini API Key is missing. Please check the sidebar.")
                st.stop()
            if "OpenRouter" in selected_model and not st.session_state.openrouter_api_key:
                st.error("OpenRouter API Key is missing. Please check the sidebar.")
                st.stop()
            with st.spinner("LLM is reading the script and deducing the cinematic context..."):
                context = run_context_analysis(st.session_state.diarized_segments, model=selected_model, groq_key=st.session_state.groq_api_key, openai_key=st.session_state.openai_api_key, anthropic_key=st.session_state.anthropic_api_key, gemini_key=st.session_state.gemini_api_key, openrouter_key=st.session_state.openrouter_api_key)
                st.session_state.scene_context = context
                trigger_save()
                st.rerun()
        st.stop() # Wait for analysis
    else:
        # Allow user to edit the context
        updated_context = st.text_area("AI Deduced Context (Editable)", value=st.session_state.scene_context, height=100)
        st.session_state.scene_context = updated_context
        
        if st.button("🔄 Re-Analyze Scene Tone"):
            st.session_state.scene_context = None
            st.rerun()
        
        if st.session_state.translated_segments is None:
            use_translation_cache = st.checkbox("Use Cached Translation (Fast)", value=False, help="Enable this to skip the LLM API and load the previously saved translation from disk.")
            
            if st.button("🚀 Run AI Transcreation", type="primary"):
                if "Groq" in selected_model and not st.session_state.groq_api_key and not use_translation_cache:
                    st.error("Groq API Key is missing.")
                    st.stop()
                if "OpenAI" in selected_model and not st.session_state.openai_api_key and not use_translation_cache:
                    st.error("OpenAI API Key is missing.")
                    st.stop()
                if "Anthropic" in selected_model and not st.session_state.anthropic_api_key and not use_translation_cache:
                    st.error("Anthropic API Key is missing.")
                    st.stop()
                if "Google" in selected_model and not st.session_state.gemini_api_key and not use_translation_cache:
                    st.error("Gemini API Key is missing.")
                    st.stop()
                if "OpenRouter" in selected_model and not st.session_state.openrouter_api_key and not use_translation_cache:
                    st.error("OpenRouter API Key is missing.")
                    st.stop()
                with st.spinner(f"Running {selected_model} for Isochronous Transcreation (with Python Length Validation loop)..."):
                    if use_translation_cache and os.path.exists(f"{OUTPUT_DIR}/translation_cache.json"):
                        print("Loading translation from cache...")
                        try:
                            with open(f"{OUTPUT_DIR}/translation_cache.json", "r") as f:
                                translated = json.load(f)
                        except json.JSONDecodeError:
                            st.error("The translation cache file is empty or corrupted (likely due to a connection drop while saving). Please uncheck 'Use Cached Translation' and run again to generate a fresh translation.")
                            st.stop()
                    else:
                        translated = run_transcreation(st.session_state.diarized_segments, model=selected_model, protected_words=protected_words, scene_context=updated_context, groq_key=st.session_state.groq_api_key, openai_key=st.session_state.openai_api_key, anthropic_key=st.session_state.anthropic_api_key, gemini_key=st.session_state.gemini_api_key, openrouter_key=st.session_state.openrouter_api_key)
                        # Save to disk cache for future runs
                        with open(f"{OUTPUT_DIR}/translation_cache.json", "w") as f:
                            json.dump(translated, f, indent=4)
                            
                    st.session_state.translated_segments = translated
                    trigger_save()
                    st.rerun()
            st.stop()
        else:
            col_succ, col_btn = st.columns([4, 1])
            with col_succ:
                st.success("Transcreation Complete! You can manually edit the Hindi script below to fix any translation errors or syllable mismatches.")
            with col_btn:
                if st.button("🔄 Re-Run Transcreation"):
                    st.session_state.translated_segments = None
                    st.session_state.base_audio_generated = False
                    st.session_state.seed_vc_complete = False
                    st.session_state.final_video_path = None
                    st.rerun()
            
            # Export Hindi Script functionality
            hindi_script_export = ""
            for seg in st.session_state.translated_segments:
                hindi_script_export += f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg.get('hindi_translation', '')}\n"
                
            st.download_button(
                label="⬇️ Download Translated Hindi Script (.txt)",
                data=hindi_script_export,
                file_name="translated_hindi_script.txt",
                mime="text/plain"
            )
        
        # Render editable text fields for each segment
        updated_segments = []
        
        for i, seg in enumerate(st.session_state.translated_segments):
            st.markdown(f"**Speaker: {seg['speaker']} | Time: {seg['start']:.2f}s - {seg['end']:.2f}s**")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("English (Original)", value=seg.get("original_text", seg.get("text", "")), height=100, disabled=True, key=f"eng_{i}")
                st.caption(f"Target Syllables: **{seg.get('english_syllables', '?')}**")
            with col2:
                # The editable Hindi text field
                new_hindi = st.text_area("Hindi (Editable)", value=seg.get("hindi_translation", ""), height=100, key=f"hin_{i}")
                # The user is responsible for ensuring syllables match if they edit it manually
                st.caption(f"Current Syllables (by LLM): **{seg.get('hindi_syllables', '?')}**")
                
            # Update the segment with the potentially edited text
            seg_copy = seg.copy()
            seg_copy["hindi_translation"] = new_hindi
            updated_segments.append(seg_copy)
            st.divider()
            
        if st.button("Approve Script & Run Voice Generation", type="primary"):
            # Save the edited segments back to state
            st.session_state.translated_segments = updated_segments
            
            # Reset Phase 4/5 state to force regeneration of the new script
            st.session_state.base_audio_generated = False
            st.session_state.seed_vc_complete = False
            st.session_state.final_video_path = None
            
            st.session_state.step = 4
            trigger_save()
            st.rerun()

# --- STEP 4: Voice Gen & Mix ---
elif st.session_state.step == 4:
    st.header("Phase 4: Voice Generation, Sync & Final Mix")
    
    tts_engine = st.selectbox("TTS Engine", options=["Kokoro (Fast, Multi-Speaker)", "Kokoro + Voice Conversion (Seed-VC)", "Sarvam (Bulbul V3) + Voice Conversion (Seed-VC)", "AI4Bharat IndicF5 (True Zero-Shot Cloning)"], index=2, help="Select the Voice Generation backend. Sarvam API provides the most natural prosody for Hindi.")
    engine_key = "indicf5" if "IndicF5" in tts_engine else ("sarvam+seedvc" if "Sarvam" in tts_engine else ("kokoro" if "Kokoro (Fast" in tts_engine else "kokoro+seedvc"))
    
    use_audio_cache = st.checkbox("Use Cached Audio (Fast)", value=False, help="Enable this to skip regenerating lines that already have audio. By default, this is OFF to ensure a clean run of the entire script.")
    
    if not st.session_state.base_audio_generated:
        st.info("Phase 4A: Generate the clean base Hindi vocals first.")
        if st.button("🎙️ Generate Base Audio", type="primary"):
            if engine_key == "sarvam+seedvc" and not st.session_state.sarvam_api_key:
                st.error("Sarvam API Key is missing. Please check the sidebar.")
                st.stop()
            try:
                if engine_key in ["kokoro+seedvc", "indicf5", "sarvam+seedvc"]:
                    with st.spinner("Running Auto-Cloner (Extracting References)..."):
                        extract_reference_audio(st.session_state.diarized_segments, st.session_state.vocals_path, output_dir="data/references")
                        
                with st.spinner(f"Running Base Audio Generation ({tts_engine})..."):
                    generated = run_voice_generation(st.session_state.translated_segments, f"{OUTPUT_DIR}/raw_dub", engine=engine_key, use_cache=use_audio_cache, sarvam_key=st.session_state.sarvam_api_key)
                    st.session_state.generated_segments = generated
                        
                st.session_state.base_audio_generated = True
                trigger_save()
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
                
    elif st.session_state.final_video_path is None:
        st.success("✅ Base audio generated!")
        st.markdown("Listen to a sample of the generated base audio:")
        # Just play the first successfully generated audio segment as a preview
        for seg in st.session_state.generated_segments:
            if seg.get("audio_raw_path") and os.path.exists(seg["audio_raw_path"]):
                st.audio(seg["audio_raw_path"])
                break
                
        if engine_key == "kokoro+seedvc" and not st.session_state.seed_vc_complete:
            if st.button("🧬 Approve & Run Seed-VC Clone", type="primary"):
                try:
                    with st.spinner("Applying Voice Conversion (Seed-VC)..."):
                        run_seed_vc_conversion(st.session_state.generated_segments, reference_dir="data/references", use_cache=use_audio_cache)
                    st.session_state.seed_vc_complete = True
                    trigger_save()
                    st.rerun()
                except NotImplementedError as e:
                    st.error(f"Engine Error: {e}")
                except Exception as e:
                    st.error(f"Seed-VC failed: {e}")
        else:
            if engine_key == "kokoro+seedvc" and st.session_state.seed_vc_complete:
                st.success("✅ Seed-VC Voice Conversion complete!")
                st.markdown("### Review Cloned Voices")
                st.markdown("Listen to the cloned voice preview for each segment before running the final mix:")
                
                for i, seg in enumerate(st.session_state.generated_segments):
                    if seg.get("audio_vc_path") and os.path.exists(seg["audio_vc_path"]):
                        st.markdown(f"**Segment {i+1}**: {seg.get('hindi_translation', '')}")
                        st.audio(seg["audio_vc_path"])
                        st.divider()
                        
            vocal_eq_profile = st.selectbox("Vocal EQ Profile (Acoustic Matching)", options=["Default (Clean)", "Deep & Raspy (Warrior/Villain)", "Bright & Clear (Dialogue)"], index=0, help="Applies specific EQ filters to match the character's physical weight.")
            
            if st.button("🎧 Preview EQ Profile"):
                with st.spinner(f"Applying {vocal_eq_profile} EQ..."):
                    from pipeline.step07_final_mix import apply_acoustic_matching
                    sample_audio = None
                    for seg in st.session_state.generated_segments:
                        sample_audio = seg.get("audio_vc_path") if engine_key == "kokoro+seedvc" else seg.get("audio_raw_path")
                        if sample_audio and os.path.exists(sample_audio):
                            break
                    if sample_audio:
                        temp_preview = "data/output/preview_eq.wav"
                        os.makedirs("data/output", exist_ok=True)
                        apply_acoustic_matching(sample_audio, temp_preview, vocal_eq_profile)
                        st.success(f"Previewing '{vocal_eq_profile}' on Segment 1:")
                        st.audio(temp_preview)
                    else:
                        st.warning("No audio segment available to preview.")
                        
            run_lipsync = st.checkbox("Run Visual Lip-Sync (Wav2Lip - Slow & Experimental)", value=False, help="Modifies the video frames so the character's mouth matches the Hindi audio. Takes ~15 mins on Mac.")
            
            if st.button("🎬 Approve & Run Final Mix", type="primary"):
                try:
                    with st.spinner("Aligning Audio (PyRubberband)..."):
                        full_vocals = run_audio_sync(st.session_state.generated_segments, f"{OUTPUT_DIR}/synced_dub")
                        
                    current_video = st.session_state.video_path
                    
                    if run_lipsync:
                        from pipeline.step08_lipsync import run_visual_lipsync
                        with st.spinner("Running Visual Lip-Sync (Wav2Lip)... This will take a long time."):
                            synced_video_out = f"data/output/lipsync_{os.path.basename(st.session_state.video_path)}"
                            current_video = run_visual_lipsync(current_video, full_vocals, synced_video_out)
                        
                    with st.spinner("Rendering Final Video (FFmpeg)..."):
                        video_name = os.path.basename(st.session_state.video_path)
                        final_out = f"data/output/dubbed_{video_name}"
                        run_final_mix(current_video, st.session_state.bgm_path, full_vocals, final_out, st.session_state.generated_segments, vocal_eq_profile)
                        st.session_state.final_video_path = final_out
                    
                    trigger_save()
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
            
    else:
        if not os.path.exists(st.session_state.final_video_path):
            st.warning("⚠️ The final video file is missing from the disk (it may have been deleted or failed to save). Resetting so you can re-generate it.")
            st.session_state.final_video_path = None
            trigger_save()
            st.rerun()
            
        st.balloons()
        st.success("🎉 Pipeline Complete! Your final localized video is ready.")
        
        st.video(st.session_state.final_video_path)
        
        with open(st.session_state.final_video_path, "rb") as f:
            st.download_button("⬇️ Download Dubbed Video", f, file_name="dubbed_video.mp4", type="primary")
            
        if st.button("Start Over"):
            # Clear state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
