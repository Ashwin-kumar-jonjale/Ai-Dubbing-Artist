import json
import gc
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger("Proofreader")

def run_proofread(segments: list, model: str = "phi4-mini") -> list:
    """
    Step 3b: AI Diarization Proofreader.
    Uses LLM to scan the diarization output for missing speaker turns or merged interruptions.
    Returns a list of suggestion objects.
    """
    print(f"[Step 3b] Running AI Proofreader using {model}...")
    
    llm = OllamaLLM(model=model, temperature=0.1)
    
    valid_segments = [{"index": i, "speaker": s["speaker"], "text": s["text"]} for i, s in enumerate(segments) if str(s.get('text', '')).strip() != ""]
    segments_json = json.dumps(valid_segments, indent=2)
    
    prompt = f"""
[SYSTEM]
You are a brilliant dialogue proofreader. Below is an English movie transcript where the speaker tags (e.g. SPEAKER_00) were assigned by an imperfect audio AI. 
Sometimes the audio AI misses quick interruptions and incorrectly merges two different speakers into a single block of text.

[TASK]
Read the dialogue carefully. Look for conversational context clues where a single text block clearly contains speech from two different people (for example, someone asking a question and someone else answering, or someone yelling an interruption).
If you find a segment that clearly merged two speakers, flag it.

[INPUT TRANSCRIPT]
{segments_json}

[CONSTRAINTS]
Respond ONLY with a valid JSON array of your suggestions. Do not include markdown formatting or conversational text outside the JSON.
Each object in the array must look exactly like this:
{{
  "segment_index": 5,
  "suggestion": "The phrase '150?' clearly belongs to a different speaker interrupting."
}}
If there are absolutely no obvious errors, return an empty array: []
"""

    import json_repair
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"  Scanning script (Attempt {attempt + 1}/{max_retries})...")
            response = llm.invoke(prompt)
            
            decoded = json_repair.loads(response)
            
            if isinstance(decoded, list):
                suggestions = decoded
                while len(suggestions) > 0 and isinstance(suggestions[0], list):
                    suggestions = suggestions[0]
                    
                if len(suggestions) > 0 and not isinstance(suggestions[0], dict):
                    raise ValueError(f"LLM returned a list of {type(suggestions[0])}, expected dictionaries.")
                    
                print("[Step 3b] Proofread complete.")
                del llm
                del response
                gc.collect()
                
                return suggestions
            else:
                raise ValueError("LLM did not return a valid JSON array.")
                
        except Exception as e:
            logger.error(f"Proofread attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning empty suggestions.")
                return []
            
    return []
