import argparse
import sys
from pipeline.step01_extraction import run_extraction
from pipeline.step02_stt import run_stt
from pipeline.step03_diarization import run_diarization
from pipeline.step04_transcreation import run_transcreation
from pipeline.step05_voice_generation import run_voice_generation
from pipeline.step06_audio_sync import run_audio_sync
from pipeline.step07_final_mix import run_final_mix

def main():
    parser = argparse.ArgumentParser(description="10-Step Lightweight M1 Architecture Dubbing Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()
    
    video_path = args.video
    output_dir = "data/processed"
    final_output_path = f"data/output/dubbed_{video_path.split('/')[-1]}"
    
    print("\n--- 🎬 NATURALDUB-AI STARTING ---\n")
    
    # Phase 1: Isolation & Understanding
    paths = run_extraction(video_path, output_dir)
    vocals_path = paths["vocals"]
    bgm_path = paths["bgm"]
    
    stt_segments = run_stt(vocals_path)
    
    diarized_segments = run_diarization(vocals_path, stt_segments)
    
    # Phase 2: Dubbing Director
    translated_segments = run_transcreation(diarized_segments, model="phi4-mini")
    
    # Phase 3: Generation & Sync
    generated_segments = run_voice_generation(translated_segments, f"{output_dir}/raw_dub")
    
    full_vocals_path = run_audio_sync(generated_segments, f"{output_dir}/synced_dub")
    
    run_final_mix(video_path, bgm_path, full_vocals_path, final_output_path, generated_segments)
    
    print(f"\n--- 🎬 PIPELINE COMPLETE! Final output saved to: {final_output_path} ---")

if __name__ == "__main__":
    main()
