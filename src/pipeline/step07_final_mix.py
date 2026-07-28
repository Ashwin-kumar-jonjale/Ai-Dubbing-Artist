import os
import subprocess

def apply_acoustic_matching(audio_path: str, output_path: str, vocal_eq_profile: str = "Default (Clean)"):
    """
    Applies a cinematic room reverb to the vocals to make them sit naturally
    in a physical space, avoiding the 'vacuum' TTS sound.
    Applies dynamic EQ based on the selected vocal_eq_profile.
    """
    try:
        from pedalboard import Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter
        from pedalboard.io import AudioFile
        print(f"[Acoustic Matching] Applying Cinematic Reverb & '{vocal_eq_profile}' EQ to {audio_path}...")
        
        with AudioFile(audio_path, 'r') as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate
            
        effects = []
        
        # Apply specific EQ curves
        if vocal_eq_profile == "Deep & Raspy (Warrior/Villain)":
            # Boost the low-end chest resonance (bass) and cut some harsh highs
            effects.append(LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=4.5))
            effects.append(HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=-2.0))
        elif vocal_eq_profile == "Bright & Clear (Dialogue)":
            # Cut the boominess for a clean dialogue track
            effects.append(HighpassFilter(cutoff_frequency_hz=100.0))
        else:
            # Default
            effects.append(HighpassFilter(cutoff_frequency_hz=80.0))
            
        # Add the cinematic Reverb
        effects.append(Reverb(room_size=0.6, damping=0.4, wet_level=0.25, dry_level=0.9, width=1.0))
        
        board = Pedalboard(effects)
        effected = board(audio, samplerate)
        
        try:
            with AudioFile(output_path, 'w', samplerate, effected.shape[0]) as f:
                f.write(effected)
        except TypeError as e:
            raise RuntimeError(f"Pedalboard AudioFile failed with type error on output_path: {type(output_path)} - {output_path}. Error: {e}")
            
        return output_path
    except ImportError:
        print("[Acoustic Matching] Pedalboard not installed. Skipping reverb.")
        return audio_path

def run_final_mix(video_path: str, bgm_path: str, full_vocals_path: str, output_path: str, segments: list, vocal_eq_profile: str = "Default (Clean)") -> str:
    """
    Step 10: FFmpeg Video Rendering & Final Mix.
    Applies a 50% audio ducking to the BGM track when characters are speaking.
    """
    debug_info = (
        f"video_path type={type(video_path)}, value={video_path}\n"
        f"bgm_path type={type(bgm_path)}, value={bgm_path}\n"
        f"full_vocals_path type={type(full_vocals_path)}, value={full_vocals_path}\n"
        f"output_path type={type(output_path)}, value={output_path}\n"
        f"segments type={type(segments)}\n"
        f"vocal_eq_profile type={type(vocal_eq_profile)}, value={vocal_eq_profile}"
    )
    print(f"[Step 10] Running FFmpeg Final Mix...\n{debug_info}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except TypeError as e:
        raise RuntimeError(f"os.makedirs failed! Types:\n{debug_info}\nOriginal Error: {e}")
    
    # 1. Apply Acoustic Matching (Reverb & EQ) to vocals
    processed_vocals = full_vocals_path.replace(".wav", "_reverb.wav")
    processed_vocals = apply_acoustic_matching(full_vocals_path, processed_vocals, vocal_eq_profile)
    
    # Construct volume ducking string for BGM
    duck_filter = "[1:a][2:a]"
    if segments:
        parts = []
        for s in segments:
            parts.append(f"between(t,{s['start']},{s['end']})")
        if parts:
            enable_str = "+".join(parts)
            # Gentle audio ducking: drop BGM volume to 50% while characters are speaking
            duck_filter = f"[1:a]volume=0.5:enable='{enable_str}'[bgm_ducked]; [bgm_ducked][2:a]"
            
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,            # Input 0: Original video
        "-i", bgm_path,              # Input 1: Isolated BGM/SFX (from Step 1)
        "-i", processed_vocals,      # Input 2: Processed Hindi Dubbed Audio
        "-filter_complex", f"{duck_filter}amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]",
        "-map", "0:v:0",             # Keep original video stream
        "-map", "[aout]",            # Use new mixed audio stream
        "-c:v", "copy",              # Copy video without re-encoding
        "-c:a", "aac",               # Encode mixed audio to AAC
        output_path
    ]
    for i, item in enumerate(ffmpeg_cmd):
        if not isinstance(item, (str, bytes, os.PathLike)):
            print(f"ERROR: item at index {i} is a {type(item)}: {item}")

    
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"[Step 10] Final Mix complete! Output saved to: {output_path}")
        
        # Cleanup temp vocal file
        if processed_vocals != full_vocals_path and os.path.exists(processed_vocals):
            os.remove(processed_vocals)
            
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg composition failed. Error details:\n{e.stderr}")
        raise RuntimeError(f"FFmpeg error: {e.stderr}")
