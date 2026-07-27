import os
import gc
from kokoro_onnx import Kokoro
import soundfile as sf
from pydub import AudioSegment

def extract_reference_audio(diarized_segments: list, vocals_wav_path: str, output_dir: str = "data/references") -> dict:
    """
    The Auto-Cloner: Extracts a clean reference audio clip for each speaker to use in Voice Cloning.
    Finds the longest contiguous segment for each speaker (up to 5 seconds) and snips it from the vocals.
    Returns a dictionary mapping speaker_id to the path of their reference audio.
    """
    print(f"[Auto-Cloner] Extracting reference audio for Voice Cloning...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group all segments by speaker
    speaker_segments = {}
    for seg in diarized_segments:
        speaker = seg["speaker"]
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(seg)
            
    reference_map = {}
    try:
        audio = AudioSegment.from_wav(vocals_wav_path)
        for speaker, segments in speaker_segments.items():
            combined_snippet = AudioSegment.empty()
            
            # Sort segments by duration descending to grab the longest clean chunks first
            segments.sort(key=lambda x: x["end"] - x["start"], reverse=True)
            
            for seg in segments:
                start_ms = int(seg["start"] * 1000)
                end_ms = int(seg["end"] * 1000)
                snippet = audio[start_ms:end_ms]
                combined_snippet += snippet
                
                # Stop if we have accumulated at least 5 seconds of audio
                if len(combined_snippet) >= 5000:
                    break
                    
            # Hard cap at 5 seconds
            if len(combined_snippet) > 5000:
                combined_snippet = combined_snippet[:5000]
                
            out_path = os.path.join(output_dir, f"{speaker}_ref.wav")
            combined_snippet.export(out_path, format="wav")
            reference_map[speaker] = out_path
            print(f"  -> Extracted {out_path} ({len(combined_snippet)/1000:.2f}s)")
    except Exception as e:
        print(f"Warning: Failed to extract reference audio: {e}")
        
    return reference_map

def run_seed_vc_conversion(generated_segments, reference_dir="data/references", use_cache=False):
    """
    Runs the Seed-VC inference by passing a JSON config to the isolated virtual environment.
    """
    import json
    import subprocess
    
    # 1. Build the list of tasks
    tasks = []
    for seg in generated_segments:
        speaker = seg['speaker']
        raw_audio_path = seg.get('audio_raw_path')
        if not raw_audio_path or not os.path.exists(raw_audio_path):
            continue
            
        ref_audio_path = f"{reference_dir}/{speaker}_ref.wav"
        if not os.path.exists(ref_audio_path):
            print(f"[Seed-VC] Warning: No reference audio found for {speaker}. Skipping conversion.")
            seg['audio_vc_path'] = raw_audio_path
            continue
            
        vc_out_path = raw_audio_path.replace("raw_", "vc_")
        seg['audio_vc_path'] = vc_out_path
        
        if use_cache and os.path.exists(vc_out_path):
            print(f"[Seed-VC] Skipping {vc_out_path}, already generated.")
            continue
            
        tasks.append({
            "source": raw_audio_path,
            "target": ref_audio_path,
            "output": vc_out_path
        })
        
    if not tasks:
        print("[Seed-VC] No tasks to process.")
        return generated_segments
        
    # 2. Write the JSON config
    json_path = f"{reference_dir}/seed_vc_tasks.json"
    with open(json_path, "w") as f:
        json.dump(tasks, f, indent=4)
        
    # 3. Call the batch_inference script via the isolated venv
    seed_vc_dir = "src/seed_vc"
    
    # We execute subprocess from the root directory so the paths stay correct!
    venv_python = f"{seed_vc_dir}/venv/bin/python"
    inference_script = f"{seed_vc_dir}/batch_inference.py"
    
    print(f"\n[Seed-VC] Starting Batch Inference for {len(tasks)} segments...")
    try:
        subprocess.run([
            venv_python,
            inference_script,
            "--json_path", json_path,
            "--diffusion-steps", "25"
        ], check=True)
        print("[Seed-VC] Batch Inference Complete!")
    except subprocess.CalledProcessError as e:
        print(f"[Seed-VC Error] Batch inference failed: {e}")
        raise RuntimeError(f"Seed-VC inference failed. Check terminal logs.")
        
    return generated_segments

def run_voice_generation(translated_segments: list, output_dir: str, engine: str = "kokoro", use_cache: bool = False, sarvam_key: str = None) -> list:
    """
    Step 6 & 7: AI Voice Generation for translated segments.
    Assigns logical speakers based on the diarization labels and generates TTS audio.
    If use_cache is True, it will skip generation for lines where the audio already exists.
    """
    print(f"[Step 6 & 7] Running AI Voice Generation ({engine})...")
    os.makedirs(output_dir, exist_ok=True)
    
    if engine == "kokoro" or engine == "kokoro+seedvc":
        print(f"Loading Kokoro-ONNX Engine...")
        kokoro = Kokoro("data/models/kokoro-v1.0.onnx", "data/models/voices-v1.0.bin")
        
        # We need a stable mapping of speakers to Kokoro voices
        available_male_voices = ["hm_omega", "hm_psi"]
        available_female_voices = ["hf_alpha", "hf_beta"]
        
        speaker_map = {}
        
        for idx, segment in enumerate(translated_segments):
            speaker_id = segment["speaker"]
            if speaker_id not in speaker_map:
                speaker_upper = speaker_id.upper()
                if "[F]" in speaker_upper or "(F)" in speaker_upper or "FEMALE" in speaker_upper or "WOMAN" in speaker_upper:
                    # Pick a female voice
                    assigned_females = [v for v in speaker_map.values() if v in available_female_voices]
                    voice_choice = available_female_voices[len(assigned_females) % len(available_female_voices)]
                else:
                    # Pick a male voice
                    assigned_males = [v for v in speaker_map.values() if v in available_male_voices]
                    voice_choice = available_male_voices[len(assigned_males) % len(available_male_voices)]
                    
                speaker_map[speaker_id] = voice_choice
                
            voice = speaker_map[speaker_id]
            hindi_text = segment["hindi_translation"]
            
            import hashlib
            text_hash = hashlib.md5(hindi_text.encode('utf-8')).hexdigest()[:8]
            out_path = os.path.join(output_dir, f"segment_{idx}_{text_hash}_raw.wav")
            
            if use_cache and os.path.exists(out_path):
                print(f"  -> Skipping Segment {idx} (Speaker: {speaker_id}) - Cached audio found!")
                segment["audio_raw_path"] = out_path
                continue
                
            try:
                audio_samples, sample_rate = kokoro.create(
                    hindi_text, voice=voice, speed=1.0, lang="hi"
                )
                if len(audio_samples) == 0:
                    raise ValueError("Kokoro TTS returned an empty audio array.")
                    
                sf.write(out_path, audio_samples, sample_rate)
                segment["audio_raw_path"] = out_path
                segment["audio_sample_rate"] = sample_rate
            except Exception as e:
                # Specific check for Kokoro's 500 character length limit
                if len(hindi_text) > 400:
                    print(f"Warning: Kokoro TTS failed for segment {idx}. (Length: {len(hindi_text)} chars). Text is likely too long for the engine! Error: {e}")
                else:
                    print(f"Warning: Kokoro TTS failed for segment {idx}: {e}")
                segment["audio_raw_path"] = None
        print(f"[Step 6 & 7] Voice generation complete. Clearing Kokoro memory...")
        del kokoro
        gc.collect()
        
    elif engine == "sarvam+seedvc":
        print(f"Using Sarvam API (Bulbul V3)...")
        import requests
        import base64
        import hashlib
        
        # Mapping available Sarvam voices based on gender (V3 compatible)
        available_male_voices = ["aditya", "rahul", "amit", "dev", "varun", "kabir"]
        available_female_voices = ["ritu", "priya", "neha", "pooja", "simran", "kavya"]
        speaker_map = {}
        
        for idx, segment in enumerate(translated_segments):
            speaker_id = segment["speaker"]
            if speaker_id not in speaker_map:
                speaker_upper = speaker_id.upper()
                if "[F]" in speaker_upper or "(F)" in speaker_upper or "FEMALE" in speaker_upper or "WOMAN" in speaker_upper:
                    assigned_females = [v for v in speaker_map.values() if v in available_female_voices]
                    voice_choice = available_female_voices[len(assigned_females) % len(available_female_voices)]
                else:
                    assigned_males = [v for v in speaker_map.values() if v in available_male_voices]
                    voice_choice = available_male_voices[len(assigned_males) % len(available_male_voices)]
                speaker_map[speaker_id] = voice_choice
                
            voice = speaker_map[speaker_id]
            hindi_text = segment["hindi_translation"]
            
            text_hash = hashlib.md5(hindi_text.encode('utf-8')).hexdigest()[:8]
            out_path = os.path.join(output_dir, f"segment_{idx}_{text_hash}_raw.wav")
            
            if use_cache and os.path.exists(out_path):
                print(f"  -> Skipping Segment {idx} (Speaker: {speaker_id}) - Cached audio found!")
                segment["audio_raw_path"] = out_path
                continue
                
            try:
                print(f"  -> Generating Segment {idx} with Sarvam ({voice})...")
                url = "https://api.sarvam.ai/text-to-speech"
                headers = {
                    "api-subscription-key": sarvam_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": [hindi_text],
                    "target_language_code": "hi-IN",
                    "speaker": voice,

                    "speech_sample_rate": 16000,
                    "enable_preprocessing": True,
                    "model": "bulbul:v3"
                }
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    audio_base64 = data.get("audios", [])[0]
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(audio_base64))
                    segment["audio_raw_path"] = out_path
                    segment["audio_sample_rate"] = 16000
                else:
                    print(f"Warning: Sarvam API failed for segment {idx}. HTTP {response.status_code}: {response.text}")
                    segment["audio_raw_path"] = None
            except Exception as e:
                print(f"Warning: Sarvam TTS failed for segment {idx}: {e}")
                segment["audio_raw_path"] = None

    elif engine == "indicf5":
        print(f"Loading AI4Bharat IndicF5 Engine...")
        import subprocess
        
        # We will iterate and use the CLI to ensure memory safety on Macs
        for idx, segment in enumerate(translated_segments):
            speaker_id = segment["speaker"]
            hindi_text = segment["hindi_translation"]
            
            # Auto-Extract reference audio if missing (Phase 4A)
            ref_path = f"data/references/{speaker_id}_ref.wav"
            if not os.path.exists(ref_path):
                print(f"No reference audio found for {speaker_id}. Skipping IndicF5 for segment {idx}.")
                segment["audio_raw_path"] = None
                continue
                
            import hashlib
            text_hash = hashlib.md5(hindi_text.encode('utf-8')).hexdigest()[:8]
            out_path = os.path.join(output_dir, f"segment_{idx}_{text_hash}_indicf5.wav")
            
            # **CRITICAL CACHE CHECK**: Skip if we already generated it!
            if use_cache and os.path.exists(out_path):
                print(f"  -> Skipping Segment {idx} (Speaker: {speaker_id}) - Cached audio found!")
                segment["audio_raw_path"] = out_path
                continue
                
            # The CLI command for f5-tts
            cmd = [
                "python", "-m", "f5_tts.infer.infer_cli",
                "--model", "F5TTS_Base",
                "--ckpt_file", "hf://ai4bharat/IndicF5/model.safetensors",
                "--vocab_file", "hf://ai4bharat/IndicF5/checkpoints/vocab.txt",
                "--ref_audio", ref_path,
                "--ref_text", "", # Auto-transcribe reference if empty
                "--gen_text", hindi_text,
                "--output_dir", output_dir,
                "--output_file", os.path.basename(out_path)
            ]
            
            print(f"  -> Generating Segment {idx} (Speaker: {speaker_id}) via IndicF5...")
            try:
                subprocess.run(cmd, check=True)
                segment["audio_raw_path"] = out_path
            except subprocess.CalledProcessError as e:
                print(f"Warning: IndicF5 failed for segment {idx}: {e}")
                segment["audio_raw_path"] = None
                
    elif engine == "xtts":
        raise NotImplementedError("XTTSv2 Engine is structurally supported but not yet installed. Please stick to Kokoro for now to prevent PyTorch Apple Silicon conflicts.")
        
    else:
        raise ValueError(f"Unknown TTS Engine: {engine}")
        
    return translated_segments
