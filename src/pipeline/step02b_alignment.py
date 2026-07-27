import json
import re
import difflib
from langchain_groq import ChatGroq

def extract_json(text: str) -> str:
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: return match.group(1)
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def normalize_word(word: str) -> str:
    """Strip punctuation and lowercase for fuzzy matching."""
    return re.sub(r'[^\w\s]', '', word).lower().strip()

def clean_script_line(line: str) -> str:
    """Removes timestamps, speaker tags (>>), and bracketed [music] tags from a script line."""
    # Remove absolute timestamps (e.g., 00:00:01.560 or 00:00:01)
    line = re.sub(r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*', '', line)
    # Remove >>
    line = re.sub(r'>>\s*', '', line)
    # Remove bracketed tags like [music]
    line = re.sub(r'\[.*?\]', '', line)
    # Remove extra spaces left behind
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip()

def split_into_sentences(text: str) -> list:
    """Splits a block of text into sentences based on punctuation (.?!)"""
    # Split on punctuation followed by a space, keeping the punctuation attached to the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def run_groq_align(script_text: str, stt_segments: list, api_key: str, mapping_mode: str = "hybrid") -> list:
    """
    Deterministic Python Forced Aligner + Hybrid Turn Mapping.
    1. Parses script into Speaker Turns (using >> or explicit names).
    2. Splits each Turn into granular Sentences to preserve audio gaps.
    3. Maps sentences to audio via difflib to get perfect timestamps.
    4. Maps Speakers using either Pyannote overlap (Hybrid) or LLM Context (LLM).
    """
    print(f"[Groq-Align] Starting Deterministic Python Alignment with {mapping_mode.upper()} mapping...")
    
    # 1. Flatten ASR words
    asr_words = []
    for seg in stt_segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                # Preserve Pyannote speaker tag for Hybrid mapping!
                w_copy = w.copy()
                w_copy["speaker"] = seg["speaker"]
                asr_words.append(w_copy)
                
    if not asr_words:
        print("[Groq-Align] Warning: No word-level timestamps found in ASR! Falling back to raw segments.")
        return stt_segments
        
    asr_normalized = [normalize_word(w["word"]) for w in asr_words]
    
    # 2. Parse Clean Script into Speaker Turns
    clean_lines_raw = script_text.split('\n')
    
    speaker_turns = []
    current_turn = ""
    # We optionally extract explicit names if the script has them like "Mike: Hello"
    turn_explicit_names = [] 
    current_explicit_name = None
    
    for line in clean_lines_raw:
        if not line.strip() or line.strip().startswith('#'): 
            continue
            
        # Strip timestamps temporarily just for detection
        line_no_timestamp = re.sub(r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*', '', line).strip()
            
        explicit_match = re.match(r'^([a-zA-Z0-9\s\[\]()_-]+):\s*|^\[(.*?)\]\s*', line_no_timestamp)
        if ">>" in line_no_timestamp or explicit_match or (not speaker_turns and not current_turn):
            # Save previous turn
            if current_turn:
                speaker_turns.append(current_turn.strip())
                turn_explicit_names.append(current_explicit_name)
            current_turn = clean_script_line(line)
            if explicit_match:
                current_explicit_name = (explicit_match.group(1) or explicit_match.group(2)).strip()
            else:
                current_explicit_name = None
        else:
            cleaned = clean_script_line(line)
            if cleaned:
                if current_turn:
                    current_turn += " " + cleaned
                else:
                    current_turn = cleaned
                    
    if current_turn:
        speaker_turns.append(current_turn.strip())
        turn_explicit_names.append(current_explicit_name)
        
    # If the entire script was parsed as ONE turn (because it lacked >> markers), we fallback to line-by-line turns
    if len(speaker_turns) <= 1 and len(clean_lines_raw) > 3:
        print("[Groq-Align] No explicit turn markers found. Treating each valid line as a separate speaker turn.")
        speaker_turns = []
        turn_explicit_names = []
        for line in clean_lines_raw:
            cleaned = clean_script_line(line)
            if cleaned:
                speaker_turns.append(cleaned)
                turn_explicit_names.append(None)
    
    # 3. Split Turns into Sentences
    clean_words_flat = []
    sentence_mappings = [] # maps sentence index to (turn_idx, text, start_word_idx, end_word_idx)
    
    for turn_idx, turn_text in enumerate(speaker_turns):
        sentences = split_into_sentences(turn_text)
        # If the turn doesn't have standard punctuation, fallback to the whole turn
        if not sentences:
            sentences = [turn_text]
            
        for sentence_text in sentences:
            words = sentence_text.split()
            start_idx = len(clean_words_flat)
            for w in words:
                norm = normalize_word(w)
                if norm:
                    clean_words_flat.append(norm)
            end_idx = len(clean_words_flat) - 1
            sentence_mappings.append({
                "turn_id": turn_idx,
                "text": sentence_text,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
        
    # 4. Dynamic Sequence Alignment
    print(f"[Groq-Align] Running difflib sequence matching on {len(sentence_mappings)} sentences...")
    matcher = difflib.SequenceMatcher(None, asr_normalized, clean_words_flat)
    
    clean_to_asr_map = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for asr_i, clean_j in zip(range(i1, i2), range(j1, j2)):
                clean_to_asr_map[clean_j] = asr_i
                
    # 5. Reconstruct Sentences with Timestamps
    aligned_segments = []
    missing_sentences = [] 
    
    for sent_idx, smap in enumerate(sentence_mappings):
        matched_asr_starts = []
        matched_asr_ends = []
        matched_speakers = []
        
        for j in range(smap["start_idx"], smap["end_idx"] + 1):
            if j in clean_to_asr_map:
                asr_i = clean_to_asr_map[j]
                matched_asr_starts.append(asr_words[asr_i]["start"])
                matched_asr_ends.append(asr_words[asr_i]["end"])
                matched_speakers.append(asr_words[asr_i]["speaker"])
                
        if matched_asr_starts and matched_asr_ends:
            sent_start = min(matched_asr_starts)
            sent_end = max(matched_asr_ends)
            
            # Determine the dominant Pyannote speaker for this sentence
            dominant_speaker = "UNKNOWN"
            if matched_speakers:
                from collections import Counter
                dominant_speaker = Counter(matched_speakers).most_common(1)[0][0]
            
            # Prevent overlapping timestamps
            if aligned_segments and sent_start < aligned_segments[-1]["end"]:
                sent_start = aligned_segments[-1]["end"] + 0.01
                if sent_start >= sent_end:
                    sent_end = sent_start + 0.5 
            
            aligned_segments.append({
                "id": len(aligned_segments), 
                "turn_id": smap["turn_id"],
                "text": smap["text"],
                "start": sent_start,
                "end": sent_end,
                "speaker": dominant_speaker,
                "original_sent_idx": sent_idx
            })
        else:
            missing_sentences.append((sent_idx, smap))
            
    # Interpolate Missing Sentences
    if missing_sentences:
        print(f"[Groq-Align] Interpolating {len(missing_sentences)} missing sentences...")
        for m_idx, smap in missing_sentences:
            prev_seg = None
            next_seg = None
            
            for seg in reversed(aligned_segments):
                if seg["original_sent_idx"] < m_idx:
                    prev_seg = seg
                    break
                    
            for seg in aligned_segments:
                if seg["original_sent_idx"] > m_idx:
                    next_seg = seg
                    break
                    
            interp_start = prev_seg["end"] + 0.01 if prev_seg else 0.0
            interp_end = next_seg["start"] - 0.01 if next_seg else interp_start + 1.0
            
            if interp_end <= interp_start:
                interp_end = interp_start + 0.5
                
            new_seg = {
                "id": -1,
                "turn_id": smap["turn_id"],
                "text": smap["text"],
                "start": interp_start,
                "end": interp_end,
                "speaker": "UNKNOWN",
                "original_sent_idx": m_idx
            }
            aligned_segments.append(new_seg)
            
        aligned_segments.sort(key=lambda x: x["original_sent_idx"])
        for i, seg in enumerate(aligned_segments):
            seg["id"] = i
            if i > 0 and seg["start"] < aligned_segments[i-1]["end"]:
                seg["start"] = aligned_segments[i-1]["end"] + 0.01
                if seg["end"] <= seg["start"]:
                    seg["end"] = seg["start"] + 0.5
            
    print(f"[Groq-Align] Alignment complete. Reconstructed {len(aligned_segments)} sentences.")
    
    # 6. Apply Speaker Mapping
    if mapping_mode == "llm":
        print(f"[Groq-Align] Sending Turn blocks to LLM for Character mapping...")
        llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)
        
        # We only send the full Turn text to the LLM to get context-aware mappings
        turn_payload = [{"turn_id": str(i), "text": turn_text} for i, turn_text in enumerate(speaker_turns)]
        
        prompt = f"""
You are an expert Dialogue Editor. I will give you a chronologically ordered array of conversational "Turns" extracted from a script.
Your ONLY job is to figure out which character is speaking in each Turn ID based on the dialogue flow and context.

Output ONLY a JSON dictionary mapping the "turn_id" to the Character Name. Do NOT invent character names if they are not in the dialogue; use generic names like "Interviewer" or "Actor" if their real name is unknown.
Example: {{"0": "Mike", "1": "Sylvester", "2": "Casting Director"}}

--- Conversational Turns ---
{json.dumps(turn_payload, indent=2)}

Output ONLY the valid JSON dictionary. Do not include markdown formatting or explanations.
"""
        try:
            res = llm.invoke(prompt)
            json_str = extract_json(res.content)
            speaker_map = json.loads(json_str)
            print(f"[Groq-Align] Resolved Names: {speaker_map}")
            
            # Apply the mapping to every sentence based on its parent turn_id
            for seg in aligned_segments:
                turn_id_str = str(seg["turn_id"])
                if turn_id_str in speaker_map:
                    seg["speaker"] = speaker_map[turn_id_str]
                else:
                    seg["speaker"] = "Unknown"
        except Exception as e:
            print(f"[Groq-Align] Warning: Name Resolution Failed. Keeping Pyannote/UNKNOWN tags. Error: {e}")
            
    elif mapping_mode == "hybrid":
        print(f"[Groq-Align] Using Hybrid Audio-Script Pyannote tags for mapping...")
        # Override with explicit script tags if they exist!
        for seg in aligned_segments:
            turn_idx = seg["turn_id"]
            if turn_idx < len(turn_explicit_names) and turn_explicit_names[turn_idx]:
                seg["speaker"] = turn_explicit_names[turn_idx]
            
    # Clean up intermediate fields
    for seg in aligned_segments:
        if "original_sent_idx" in seg:
            del seg["original_sent_idx"]
        if "turn_id" in seg:
            del seg["turn_id"]
            
    return aligned_segments
