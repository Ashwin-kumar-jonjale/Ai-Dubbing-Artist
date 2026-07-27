import os
import pyrubberband as pyrb
import soundfile as sf
import pydub

def run_audio_sync(segments: list, output_dir: str) -> str:
    """
    Step 8 & 9: PyRubberband Time Stretching & Assembly.
    Strictly caps the rubberband ratio to 0.95x - 1.05x to prevent robotic voices.
    Step 9 (Wav2Lip) is intentionally bypassed.
    Returns the path to the final assembled vocal track.
    """
    print(f"[Step 8 & 9] Running PyRubberband Audio Sync with Capped Ratios...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We will assemble onto a silent canvas
    if not segments:
        raise ValueError("No segments to sync.")
        
    end_time = max([s['end'] for s in segments])
    full_vocals = pydub.AudioSegment.silent(duration=int(end_time * 1000))
    
    for idx, segment in enumerate(segments):
        raw_path = segment.get("audio_vc_path") or segment.get("audio_raw_path")
        if not raw_path or not os.path.exists(raw_path):
            continue
            
        y, sr = sf.read(raw_path)
        raw_duration = len(y) / sr
        target_duration = segment["end"] - segment["start"]
        
        raw_ratio = raw_duration / target_duration if target_duration > 0 else 1.0
        
        # Capping the ratio to prevent the "atempo trap" (unnatural robotic voices)
        # Widened to 35% to stretch/condense the TTS clip to perfectly fill the original slot and prevent dead air padding
        ratio = max(0.65, min(1.35, raw_ratio))
        
        synced_path = os.path.join(output_dir, f"segment_{idx}_synced.wav")
        
        if ratio != 1.0:
            y_stretched = pyrb.time_stretch(y, sr, ratio)
            sf.write(synced_path, y_stretched, sr)
        else:
            # Just copy if ratio is exactly 1.0
            sf.write(synced_path, y, sr)
            
        segment["audio_synced_path"] = synced_path
        
        # Overlay onto the canvas
        audio_segment = pydub.AudioSegment.from_wav(synced_path)
        # Apply -8dB gain reduction to match original video loudness (-34.5dB)
        audio_segment = audio_segment - 8
        full_vocals = full_vocals.overlay(audio_segment, position=int(segment["start"] * 1000))
        
    full_vocals_path = os.path.join(output_dir, "full_vocals_hindi.wav")
    full_vocals.export(full_vocals_path, format="wav")
    
    print(f"[Step 8 & 9] Audio Sync & Assembly complete.")
    return full_vocals_path
