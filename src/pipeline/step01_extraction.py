import os
import subprocess
import shutil
from pathlib import Path

def run_extraction(video_path: str, output_dir: str) -> dict:
    """
    Step 1: Isolate vocals from BGM using MDX-Net (audio-separator).
    Returns a dict with paths to the isolated vocals and bgm.
    """
    print(f"[Step 1] Extracting vocals from {video_path}...")
    video_path_obj = Path(video_path)
    out_dir_obj = Path(output_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)
    
    vocals_path = str(out_dir_obj / f"{video_path_obj.stem}_vocals.wav")
    no_vocals_path = str(out_dir_obj / f"{video_path_obj.stem}_bgm.wav")
    
    if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
        print(f"[Step 1] Cached audio found. Skipping MDX-Net.")
        return {"vocals": vocals_path, "bgm": no_vocals_path}
    
    # We extract the vocals and BGM using BS-Roformer (Dual-Stem Model)
    # We specify model_file_dir to prevent Mac from wiping /tmp on reboot
    os.makedirs("data/models", exist_ok=True)
    cmd = [
        "audio-separator",
        video_path,
        "--model_file_dir", "data/models",
        "--output_dir", str(out_dir_obj),
        "--output_format", "WAV"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    
    # Locate BS-Roformer output files
    all_wavs = list(out_dir_obj.glob(f"{video_path_obj.stem}*.wav"))
    gen_vocals = next((w for w in all_wavs if "(Vocals)" in w.name), None)
    gen_bgm = next((w for w in all_wavs if "(Instrumental)" in w.name), None)
    
    if gen_vocals and gen_bgm:
        # Normalize the vocals before finalizing
        normalized_vocals = str(out_dir_obj / f"{video_path_obj.stem}_normalized_vocals.wav")
        print(f"[Step 1] Applying FFmpeg Loudness Normalization to vocals...")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(gen_vocals),
            "-af", "loudnorm=I=-16:TP=-1.5",
            normalized_vocals
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up the raw BS-Roformer output and keep only normalized
        os.remove(str(gen_vocals))
        shutil.move(normalized_vocals, vocals_path)
        
        shutil.move(str(gen_bgm), no_vocals_path)
        print(f"[Step 1] Vocal extraction & normalization complete.")
        return {"vocals": vocals_path, "bgm": no_vocals_path}
    else:
        raise RuntimeError("Failed to locate BS-Roformer separated files.")
