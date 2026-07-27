import os
import json
import subprocess
import tempfile

def run_stt(audio_path: str, engine: str = "local", api_key: str = None) -> list:
    """
    Step 2: Speech-to-Text.
    Supports local MLX-Whisper (Apple Silicon) or Cloud Groq API (whisper-large-v3).
    """
    if engine == "groq":
        print(f"[Step 2] Running Cloud STT (Groq Whisper-Large-v3) on {audio_path}...")
        from groq import Groq
        
        if not api_key:
            raise ValueError("Groq API Key is required for Cloud STT.")
            
        client = Groq(api_key=api_key)
        
        # Groq has a 25MB file size limit. To bypass this, we compress the audio to a lightweight 16kHz Mono MP3.
        # Whisper internally downsamples to 16kHz anyway, so this preserves 100% accuracy while reducing file size by ~95%.
        print(f"[Step 2] Compressing audio for Groq API (bypassing 25MB limit)...")
        fd, temp_mp3 = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path, 
                "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k",
                temp_mp3
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"[Step 2] Compression complete. Uploading to Groq LPU Servers...")
            import requests
            
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                # Pass as individual items for the array in form-data
                "timestamp_granularities[]": ["word", "segment"]
            }
            
            with open(temp_mp3, "rb") as file:
                files = {
                    "file": ("audio.mp3", file, "audio/mpeg")
                }
                print(f"[Step 2] Sending direct HTTP POST to Groq API...")
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Groq API Error ({response.status_code}): {response.text}")
                
            transcription = response.json()
            
        finally:
            # Clean up the temporary MP3 file
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
        segments = []
        segs = getattr(transcription, "segments", [])
        if not segs and isinstance(transcription, dict):
            segs = transcription.get("segments", [])
            
        # Also grab the top-level words array from Groq's response if it exists
        all_groq_words = getattr(transcription, "words", [])
        if not all_groq_words and isinstance(transcription, dict):
            all_groq_words = transcription.get("words", [])
            
        for segment in segs:
            start = getattr(segment, "start", None) if not isinstance(segment, dict) else segment.get("start")
            end = getattr(segment, "end", None) if not isinstance(segment, dict) else segment.get("end")
            text = getattr(segment, "text", "") if not isinstance(segment, dict) else segment.get("text", "")
            
            # Extract word-level timestamps (Groq returns them in the segment, or at the top level)
            raw_words = getattr(segment, "words", []) if not isinstance(segment, dict) else segment.get("words", [])
            if not raw_words and all_groq_words:
                # If they are at the top level, filter them for this segment
                raw_words = [w for w in all_groq_words if (isinstance(w, dict) and w.get("start", 0) >= start and w.get("end", 0) <= end) or (not isinstance(w, dict) and getattr(w, "start", 0) >= start and getattr(w, "end", 0) <= end)]
                
            words = []
            for w in raw_words:
                w_start = getattr(w, "start", start) if not isinstance(w, dict) else w.get("start", start)
                w_end = getattr(w, "end", end) if not isinstance(w, dict) else w.get("end", end)
                w_text = getattr(w, "word", "") if not isinstance(w, dict) else w.get("word", "")
                words.append({
                    "word": w_text,
                    "start": w_start,
                    "end": w_end
                })
            
            segments.append({
                "start": start,
                "end": end,
                "text": text.strip(),
                "words": words
            })
            
        print(f"[Step 2] Cloud Transcription complete.")
        return segments

    print(f"[Step 2] Running Local MLX-Whisper STT on {audio_path}...")
    
    # Create a temporary JSON file to hold the output
    fd, out_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    
    # Python script to run in isolated process
    script_code = f"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import mlx_whisper

result = mlx_whisper.transcribe(
    r"{audio_path}",
    path_or_hf_repo="mlx-community/whisper-small-mlx",
    word_timestamps=True
)

segments = []
for segment in result.get("segments", []):
    segments.append({{
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"].strip(),
        "words": segment.get("words", [])
    }})

with open(r"{out_json}", "w") as f:
    json.dump(segments, f)
"""
    
    # Write the script to a temp file
    fd, script_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    with open(script_path, "w") as f:
        f.write(script_code)
        
    import sys
    try:
        # Run the isolated process
        output = subprocess.check_output([sys.executable, script_path], stderr=subprocess.STDOUT, text=True)
        print(output)
        
        # Read results
        with open(out_json, "r") as f:
            segments = json.load(f)
            
        print(f"[Step 2] Local Transcription complete. Isolated memory cleared natively.")
        return segments
    except subprocess.CalledProcessError as e:
        print("MLX-Whisper subprocess crashed! Output:")
        print(e.output)
        raise RuntimeError(f"MLX-Whisper crashed during transcription. Output: {e.output}")
    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)
        if os.path.exists(out_json):
            os.remove(out_json)
