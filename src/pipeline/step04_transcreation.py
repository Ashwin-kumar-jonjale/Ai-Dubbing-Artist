import json
import gc
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
import logging

logger = logging.getLogger("Transcreation")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm_instance(model_name: str, groq_key: str = None, openai_key: str = None, anthropic_key: str = None, gemini_key: str = None, openrouter_key: str = None, temperature: float = 0.2):
    # Sanitize keys in case the user accidentally pasted "Bearer " prefix or hidden whitespace
    if openrouter_key:
        openrouter_key = openrouter_key.strip()
        if openrouter_key.lower().startswith("bearer "):
            openrouter_key = openrouter_key[7:].strip()
    if groq_key:
        groq_key = groq_key.strip()
        if groq_key.lower().startswith("bearer "):
            groq_key = groq_key[7:].strip()
    if openai_key:
        openai_key = openai_key.strip()
        if openai_key.lower().startswith("bearer "):
            openai_key = openai_key[7:].strip()
        
    if "Groq" in model_name:
        return ChatGroq(api_key=groq_key, model_name="llama-3.3-70b-versatile", temperature=temperature)
    elif "Anthropic" in model_name:
        return ChatAnthropic(api_key=anthropic_key, model_name="claude-3-5-sonnet-20240620", temperature=temperature)
    elif "OpenAI" in model_name:
        return ChatOpenAI(api_key=openai_key, model="gpt-4o", temperature=temperature)
    elif "Google (Gemini Flash Latest)" in model_name:
        return ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-flash-latest", temperature=temperature)
    elif "Google (Gemini Pro Latest)" in model_name:
        return ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-pro-latest", temperature=temperature)
    elif "Google (Gemini 3.5 Flash)" in model_name:
        return ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-3.5-flash", temperature=temperature)
    elif "Google" in model_name:
        return ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-flash-latest", temperature=temperature)
    elif "Groq Gemma" in model_name:
        return ChatGroq(api_key=groq_key, model_name="gemma2-9b-it", temperature=temperature)
    elif "OpenRouter" in model_name:
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=openrouter_key, 
            model="google/gemma-3-27b-it", 
            temperature=temperature,
            max_tokens=4000,
            default_headers={"Authorization": f"Bearer {openrouter_key}", "HTTP-Referer": "http://localhost:8501"}
        )
    else:
        return OllamaLLM(model=model_name, temperature=temperature)

def run_context_analysis(segments: list, model: str = "phi4-mini", groq_key: str = None, openai_key: str = None, anthropic_key: str = None, gemini_key: str = None, openrouter_key: str = None) -> str:
    """
    Step 3 (Pre-processing): Analyzes the English script to deduce the genre, tone, and lore.
    """
    print(f"[Step 3 Pre] Analyzing Scene Context using {model}...")
    llm = get_llm_instance(model, groq_key, openai_key, anthropic_key, gemini_key, openrouter_key, temperature=0.1)
    
    valid_segments = [s for s in segments if s.get('text', '').strip() != ""]
    script_text = "\n".join([f"{s.get('speaker', 'UNKNOWN')}: {s.get('text', '')}" for s in valid_segments])
    
    prompt = f"""
[SYSTEM]
You are an expert Hollywood-to-Bollywood localization director.
Read the following English script and deduce its cinematic context.

Analyze:
1. Genre & Vibe (e.g. Marvel superhero trailer, solemn drama)
2. Emotional Tone (e.g. Reverent, fast-paced, serious, poetic)
3. Lore/Glossary (Identify fantasy terms or slang that should be translated a specific way).

Output your analysis as a concise 2-3 sentence paragraph. 
Do NOT output any markdown, headers, or conversational filler. Just the paragraph that describes the tone and rules.

[SCRIPT]
{script_text}
"""
    try:
        response = llm.invoke(prompt)
        print(f"[Step 3 Pre] Context Analysis Complete.")
        if hasattr(response, "content"):
            return response.content.strip()
        return response.strip()
    except Exception as e:
        logger.error(f"Context analysis failed: {e}")
        return "Standard cinematic scene. Translate naturally."

def run_transcreation(segments: list, model: str = "phi4-mini", protected_words: str = "", scene_context: str = "", groq_key: str = None, openai_key: str = None, anthropic_key: str = None, gemini_key: str = None, openrouter_key: str = None) -> list:
    """
    Step 3: Translates diarized English segments to Hindi with strict isochronous timing control.
    """
    print(f"\n[Step 4 & 5] Starting Isochronous Transcreation using {model}...")
    llm = get_llm_instance(model, groq_key, openai_key, anthropic_key, gemini_key, openrouter_key, temperature=0.2)
    
    # Filter out empty segments
    valid_segments = [s for s in segments if s.get('text', '').strip() != ""]
    
    # STAGE A: Target Calculation
    # We calculate the max duration for each segment and append it to the JSON context
    for seg in valid_segments:
        duration = seg['end'] - seg['start']
        seg['target_duration_seconds'] = round(duration, 2)
        # Average Hindi speaking rate is ~12-14 characters per second. We set a max character limit.
        # CRITICAL FIX: Bound the max characters by the source text length to mathematically prevent hallucinations during long pauses.
        time_based_limit = int(duration * 15)
        text_based_bound = max(30, len(seg.get('text', '')) * 3)
        seg['max_hindi_characters_allowed'] = min(time_based_limit, text_based_bound)
        
    glossary_instruction = ""
    if protected_words.strip():
        glossary_instruction = f"\nCRITICAL GLOSSARY: Do NOT translate the following words. Keep them in English: {protected_words}\n"
        
    context_instruction = f"\nSCENE CONTEXT, TONE & LORE:\n{scene_context}\nCRITICAL: You MUST adopt this tone perfectly. If it is a serious/lore-heavy scene, use respectful, poetic, cinematic Hindi.\n" if scene_context.strip() else ""
    
    # STAGE B: RAG / Few-Shot Golden Examples
    few_shot_examples = """
[GOLDEN EXAMPLES (RAG) - BOLLYWOOD HINGLISH REGISTER]
English: "Acting's what I live and breathe for."
Formal (BAD): "अभिनय ही मेरा जीवन है।"
Bollywood (GOOD): "एक्टिंग ही मेरी जिंदगी है।"

English: "I can't even moonwalk."
Literal (BAD): "मैं मूव भी नहीं कर सकता।"
Bollywood (GOOD): "मैं मूनवॉक भी नहीं कर सकता..."

English: "Thumbs up."
Literal (BAD): "अंगूठा ऊपर।"
Bollywood (GOOD): "थम्ब्स अप।"

English (Fragment over long pause): "while."
Hallucination (BAD): "थोड़ी देर के लिए... ये प्रोजेक्ट हेल मैरी है, सूरज मर रहा है..."
Strict Fidelity (GOOD): "कुछ देर के लिए..."

English: "Give me the strength of the ancestors."
Literal (BAD): "मुझे पूर्वजों की ताकत दो।"
Poetic (GOOD): "मुझे सभी परमपिताओं की शक्ति दीजिए।"
"""
    
    import json_repair
    import gc
    import time
    
    batch_size = 30
    all_translated_segments = []
    
    for batch_idx in range(0, len(valid_segments), batch_size):
        translated_segments = None # Prevent scope leakage from previous batches
        response_text = None
        batch = valid_segments[batch_idx:batch_idx + batch_size]
        batch_json = json.dumps(batch, indent=2)
        
        print(f"  -> Processing Batch {batch_idx//batch_size + 1}/{(len(valid_segments)-1)//batch_size + 1} (Segments {batch_idx} to {batch_idx + len(batch) - 1})")
        
        base_prompt = f"""
[SYSTEM]
You are a professional Bollywood Dubbing Director and Elite Translator. Adapt this English dialogue into Hindi using "Length-controlled Isochronous Transcreation".
{context_instruction}{glossary_instruction}
{few_shot_examples}

CRITICAL RULES FOR STUDIO-QUALITY DUBBING:
1. HINGLISH & COLLOQUIAL BOLLYWOOD TONE: NEVER use overly formal or textbook Hindi (e.g. do not use "अभिनय", "अभिनेता"). You MUST use colloquial English loanwords written in Devanagari for modern or industry terms, pop-culture references, or physical gestures (e.g. एक्टिंग, एक्टर, थम्ब्स अप, मूनवॉक, बॉक्सिंग). 
2. CONSISTENT TRANSLITERATION: Ensure proper nouns (names, places, projects) are transliterated perfectly and consistently throughout the batch (e.g., ALWAYS use "हेल मैरी", NEVER "हैइल मेरी"). Maintain an internal glossary across the batch.
3. LOCALIZATION: Convert cultural markers naturally. (e.g., convert "millions" to "lakhs" or "crores" when referring to money or large quantities).
4. PRONOUN & RELATIONSHIP CONTINUITY (CRITICAL): Establish a consistent pronoun register for each relationship (e.g., always "तू" for close friends/cousins, always "आप" for formal). You MUST maintain this perfectly across all segments for the same speaker pairs. Do not arbitrarily flip-flop between "तू" and "तुम".
5. EMOTIONAL PROSODY (PACING): Acting is about the silence between words. You MUST actively insert ellipses (`...`) into the Hindi text wherever a human actor would take a breath or pause for dramatic effect.
6. ISOMETRIC TIMING MATCHING: Each segment has a `max_hindi_characters_allowed` based on the video length. The total character count of `hindi_translation` MUST be less than or equal to this limit. If it is too long, the voice will sound incredibly fast and robotic. Condense the phrasing!
7. COMPLETE PRESERVATION: Do NOT drop active verbs or key semantic concepts.
8. STRICT FIDELITY (NO HALLUCINATION): You MUST ONLY translate the exact words present in `original_text`. Do NOT invent, fabricate, or summarize the plot to fill time, EVEN IF the video duration is very long (e.g., 10 seconds for a 1-word fragment). If the input is a 1-word fragment, your output MUST be a 1-word fragment. 

[INPUT DETAILS]
Dialogue Segments to Translate:
{batch_json}

[OUTPUT FORMAT]
Respond ONLY with a valid JSON array matching the exact order and number of segments. Each object must have these keys:
  - "speaker": The original speaker label.
  - "start": The start time.
  - "end": The end time.
  - "original_text": The original English text.
  - "hindi_translation": The translated Hindi text (including `...` for pauses).
"""
        max_retries = 4
        current_prompt = base_prompt
        batch_translated = None
        
        for attempt in range(max_retries):
            try:
                print(f"    Attempt {attempt + 1}/{max_retries}...")
                response = llm.invoke(current_prompt)
                
                if hasattr(response, "content"):
                    response_text = response.content
                else:
                    response_text = response
                    
                decoded = json_repair.loads(response_text)
                
                if isinstance(decoded, list):
                    translated_segments = decoded
                    while len(translated_segments) > 0 and isinstance(translated_segments[0], list):
                        translated_segments = translated_segments[0]
                        
                    if len(translated_segments) > 0 and not all(isinstance(s, dict) for s in translated_segments):
                        raise ValueError("LLM returned an array with non-dictionary elements. Expected dictionaries.")
                    
                    if len(translated_segments) < len(batch):
                        raise ValueError(f"LLM returned only {len(translated_segments)} segments, but {len(batch)} were expected. The output was cut off (token limit hit). You MUST process ALL segments.")
                        
                    # Defensively truncate to exact batch length in case of hallucinations
                    translated_segments = translated_segments[:len(batch)]
                    
                    # Defensively merge original metadata back in
                    for i, seg in enumerate(translated_segments):
                        orig = batch[i]
                        seg['start'] = orig.get('start', 0.0)
                        seg['end'] = orig.get('end', 1.0)
                        seg['speaker'] = orig.get('speaker', 'UNKNOWN')
                        if 'original_text' not in seg:
                            seg['original_text'] = orig.get('text', '')
                    
                    # STAGE C: Length Validation (Python Feedback Loop)
                    errors = []
                    for i, seg in enumerate(translated_segments):
                        orig_len = len(seg.get('original_text', ''))
                        hin_len = len(seg.get('hindi_translation', ''))
                        duration = seg.get('end', 1.0) - seg.get('start', 0.0)
                        # We calculate strict character limits
                        hard_max = int(duration * 20) + 15 # 15 chars grace for very short clips
                        if hin_len > hard_max:
                            errors.append(f"Segment {i} ('{seg.get('original_text')}') translation is {hin_len} characters long, but the target duration is only {round(duration, 2)} seconds. A human can only speak ~{hard_max} characters in this time. You MUST condense this translation.")

                    if errors:
                        error_msg = "\n".join(errors)
                        raise ValueError(f"TIMING CONSTRAINT VIOLATIONS:\n{error_msg}\nRewrite the Hindi translations for these specific segments to be much shorter.")

                    batch_translated = translated_segments
                    break
                else:
                    raise ValueError("LLM did not return a valid JSON array.")
                    
            except Exception as e:
                logger.error(f"Transcreation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    if "TIMING CONSTRAINT VIOLATIONS" in str(e):
                        logger.warning("Max retries reached. Bypassing strict timing constraints to prevent pipeline failure.")
                        if 'translated_segments' in locals() and translated_segments is not None:
                            batch_translated = translated_segments
                            break
                    raise RuntimeError(f"Transcreation failed to generate valid JSON after {max_retries} attempts. Last error: {e}")
                
                # If we hit a Google API Free Tier rate limit (5 RPM), pause before retrying
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"      [API Rate Limit Hit] Pausing for 40 seconds to let free-tier quota reset...")
                    time.sleep(40)
                
                print(f"      Retrying and prompting LLM to fix the error...")
                current_prompt = base_prompt + f"\n\n[FEEDBACK FROM PREVIOUS ATTEMPT - YOU MUST FIX THIS BEFORE PROCEEDING]\n{str(e)}\n"
                
        if batch_translated:
            all_translated_segments.extend(batch_translated)

    unique_starts = set([str(s.get('start')) for s in all_translated_segments])
    if len(unique_starts) != len(valid_segments):
        raise ValueError(f"CRITICAL TRANSCREATION ERROR: Pipeline expected {len(valid_segments)} unique segments, but got {len(unique_starts)}. A batch chunk was duplicated or missed due to LLM errors. Please Re-Run Transcreation.")

    print("[Step 4 & 5] Transcreation complete. Clearing LLM context...")
    del llm
    gc.collect()
    
    return all_translated_segments
