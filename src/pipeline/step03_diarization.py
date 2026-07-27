import os
import json
import subprocess
import tempfile
import sys

def run_diarization(audio_path: str, stt_segments: list) -> list:
    """
    Step 3: Pyannote Speaker Diarization.
    Runs in an isolated subprocess to prevent PyTorch/Apple Silicon segmentation faults.
    """
    print(f"[Step 3] Running Pyannote Diarization on {audio_path}...")
    
    fd, in_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(in_json, "w") as f:
        json.dump(stt_segments, f)
        
    fd, out_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    
    script_code = f"""
import os
import json
import torch
import gc
from pyannote.audio import Pipeline

# Load input
with open(r"{in_json}", "r") as f:
    stt_segments = json.load(f)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.environ.get("HF_TOKEN")
)

if torch.backends.mps.is_available():
    pipeline.to(torch.device("mps"))
    
output = pipeline(r"{audio_path}")
diarization = getattr(output, "speaker_diarization", output)

# Flatten words
all_words = []
for seg in stt_segments:
    if "words" in seg and len(seg["words"]) > 0:
        all_words.extend(seg["words"])
    else:
        # Fallback if Whisper failed to extract word timestamps for this segment
        all_words.append({{"word": seg["text"], "start": seg["start"], "end": seg["end"]}})

# Assign speakers based on max overlap, with fallback to nearest block
for word in all_words:
    w_start = word["start"]
    w_end = word["end"]
    
    speaker_overlaps = {{}}
    nearest_spk = "UNKNOWN"
    min_dist = float('inf')
    
    for turn, _, spk in diarization.itertracks(yield_label=True):
        # Calculate overlap
        overlap = max(0, min(w_end, turn.end) - max(w_start, turn.start))
        if overlap > 0:
            speaker_overlaps[spk] = speaker_overlaps.get(spk, 0) + overlap
            
        # Calculate distance (if overlap == 0)
        # distance from word interval [w_start, w_end] to turn interval [turn.start, turn.end]
        if w_end < turn.start:
            dist = turn.start - w_end
        elif turn.end < w_start:
            dist = w_start - turn.end
        else:
            dist = 0
            
        if dist < min_dist:
            min_dist = dist
            nearest_spk = spk
            nearest_turn_start = turn.start
            
    if speaker_overlaps:
        word["speaker"] = max(speaker_overlaps, key=speaker_overlaps.get)
        
        # Find the specific turn for the chosen speaker that overlaps the most
        best_turn_start = 0
        best_overlap = -1
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if spk == word["speaker"]:
                overlap = max(0, min(w_end, turn.end) - max(w_start, turn.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_turn_start = turn.start
        word["turn_start"] = best_turn_start
    else:
        # If no overlap at all (word spoken in a gap), assign to the nearest speaker block
        word["speaker"] = nearest_spk
        word["turn_start"] = nearest_turn_start if 'nearest_turn_start' in locals() else 0

# Regroup consecutive words by speaker into new clean segments
new_segments = []
current_segment = None

for word in all_words:
    if current_segment is None:
        # Trust Whisper's relative spacing by default
        actual_start = word["start"]
        
        # Targeted Override: Fix Whisper's 0.0s hallucination bug on initial silence.
        # Only anchor to Pyannote if Whisper thinks it starts instantly, but Pyannote detected silence.
        if actual_start < 0.5 and word.get("turn_start", 0) > 1.0:
            actual_start = word.get("turn_start", 0)
        
        current_segment = {{
            "start": actual_start,
            "end": word["end"],
            "text": word["word"].lstrip(),
            "speaker": word["speaker"],
            "words": [word]
        }}
    elif word["speaker"] == current_segment["speaker"]:
        # Cinematic Splitting: Granular Anchoring
        gap = word["start"] - current_segment["end"]
        duration = word["end"] - current_segment["start"]
        ends_with_punctuation = current_segment["text"].strip().endswith((".", "?", "!"))
        
        # Force split if:
        # 1. Silence gap > 0.4s
        # 2. Or segment is > 4.0s long
        # 3. Or previous word ended a sentence (punctuation) and there is a tiny gap > 0.1s
        if gap > 0.4 or duration > 4.0 or (ends_with_punctuation and gap > 0.1):
            new_segments.append(current_segment)
            actual_start = word["start"]
            current_segment = {{
                "start": actual_start,
                "end": word["end"],
                "text": word["word"].lstrip(),
                "speaker": word["speaker"],
                "words": [word]
            }}
        else:
            # Extend the segment end time and append text
            current_segment["end"] = word["end"]
            # Groq Whisper does not include leading spaces in the 'word' field, so we must add it
            if current_segment["text"]:
                current_segment["text"] += " " + word["word"].lstrip()
            else:
                current_segment["text"] += word["word"]
            current_segment["words"].append(word)
    else:
        # Speaker changed: save the old segment and start a new one
        new_segments.append(current_segment)
        actual_start = word["start"]
        current_segment = {{
            "start": actual_start,
            "end": word["end"],
            "text": word["word"].lstrip(),
            "speaker": word["speaker"],
            "words": [word]
        }}

if current_segment:
    new_segments.append(current_segment)
    
stt_segments = new_segments

with open(r"{out_json}", "w") as f:
    json.dump(stt_segments, f)
"""
    fd, script_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    with open(script_path, "w") as f:
        f.write(script_code)
        
    try:
        output = subprocess.check_output([sys.executable, script_path], stderr=subprocess.STDOUT, text=True)
        print(output)
        
        with open(out_json, "r") as f:
            diarized_segments = json.load(f)
            
        print(f"[Step 3] Diarization complete. Isolated memory cleared natively.")
        return diarized_segments
    except subprocess.CalledProcessError as e:
        print("Pyannote subprocess crashed! Output:")
        print(e.output)
        raise RuntimeError(f"Pyannote crashed during diarization. Output: {e.output}")
    finally:
        for p in [in_json, out_json, script_path]:
            if os.path.exists(p):
                os.remove(p)
