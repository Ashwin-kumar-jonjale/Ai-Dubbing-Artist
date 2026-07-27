import os
import subprocess
import tempfile
import urllib.request
import urllib.error

# Wav2Lip setup paths
WAV2LIP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wav2lip"))
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
FACE_DETECTOR_PATH = os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd", "s3fd.pth")

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)}...")
        try:
            # Simple fallback for gdown if it's a gdrive link, or direct download
            if "drive.google.com" in url:
                subprocess.run(["pip", "install", "-q", "gdown"], check=True)
                subprocess.run(["gdown", url, "-O", dest], check=True)
            else:
                urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"Failed to download {dest}: {e}")
            raise

def setup_wav2lip():
    """Clones Wav2Lip and downloads necessary models."""
    if not os.path.exists(WAV2LIP_DIR):
        print("[Step 11] Setting up Wav2Lip...")
        # Clone a Wav2Lip repository
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", WAV2LIP_DIR], check=True)
        
    # Download weights from a reliable HuggingFace mirror instead of Google Drive
    if not os.path.exists(CHECKPOINT_PATH):
        print("[Step 11] Downloading Wav2Lip GAN weights (this may take a minute)...")
        # Direct link to the raw model weights on HuggingFace
        download_file("https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth", CHECKPOINT_PATH)
        
    if not os.path.exists(FACE_DETECTOR_PATH):
        print("[Step 11] Downloading S3FD Face Detection weights...")
        download_file("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", FACE_DETECTOR_PATH)
        
    # Verify the weights actually downloaded correctly
    if not os.path.exists(CHECKPOINT_PATH):
        print("[Step 11] WARNING: Wav2Lip weights could not be downloaded automatically!")
        return False
        
    # Patch Wav2Lip for Apple Silicon (MPS)
    # We replace .cuda() with .to(device) to support MPS
    patch_file = os.path.join(WAV2LIP_DIR, "inference.py")
    with open(patch_file, "r") as f:
        content = f.read()
    if "mps" not in content and "cuda" in content:
        print("[Step 11] Patching Wav2Lip for Apple Silicon (MPS)...")
        content = content.replace("device = 'cuda' if torch.cuda.is_available() else 'cpu'", "device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')")
        content = content.replace(".cuda()", ".to(device)")
        with open(patch_file, "w") as f:
            f.write(content)
            
    return True

def run_visual_lipsync(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Step 11: Visual Lip Sync via Wav2Lip.
    """
    print(f"[Step 11] Running Visual Lip Sync (Wav2Lip) on {video_path}...")
    
    if not setup_wav2lip():
        raise RuntimeError("Wav2Lip setup incomplete. Please download the weights manually as instructed.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run Wav2Lip inference
    # Note: Wav2Lip pads the face heavily. We can use --pads to adjust.
    inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
    
    cmd = [
        "python3", inference_script,
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--pads", "0", "20", "0", "0" # slight bottom padding helps
    ]
    
    print(f"[Step 11] Executing Wav2Lip. This will take a long time on Mac...")
    try:
        subprocess.run(cmd, check=True)
        print(f"[Step 11] Lip Sync complete! Saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Wav2Lip failed. Check if dependencies (opencv, face_alignment) are installed.")
        raise
