import speech_recognition as sr
import io
import re

def transcribe_audio(audio_bytes):
    """
    Transcribes raw audio bytes into text using Google's Web Speech API.
    """
    recognizer = sr.Recognizer()
    
    try:
        # Convert bytes to a file-like object so SpeechRecognition can read it
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            # Use Google's free Web Speech API
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except sr.UnknownValueError:
        return "ERROR: Could not understand audio."
    except sr.RequestError as e:
        return f"ERROR: Could not request results; {e}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def extract_farm_parameters(text):
    """
    Uses basic rule-based NLP to extract farm parameters from transcribed text.
    Returns a dictionary of updated session state variables.
    """
    updates = {}
    
    # 1. Crop Type Extraction
    crops = ["rice", "wheat", "cotton", "sugarcane", "soybean", "tomato", "potato"]
    for crop in crops:
        if re.search(r'\b' + crop + r'\b', text):
            updates['crop_type'] = crop.capitalize()
            break
            
    # 2. Region Extraction
    regions = ["punjab", "maharashtra", "uttar pradesh", "west bengal"]
    for region in regions:
        if re.search(r'\b' + region + r'\b', text):
            updates['region'] = region.title()
            break
            
    # 3. Climate Scenario Extraction
    if "drought" in text:
        updates['climate_scenario'] = "Severe Drought"
    elif "heat" in text or "heatwave" in text:
        updates['climate_scenario'] = "Unseasonal Heatwave"
    elif "normal" in text or "monsoon" in text:
        updates['climate_scenario'] = "Normal Monsoon"
        
    # 4. Temperature parsing (e.g. "temperature is 35")
    temp_match = re.search(r'temperature\s*(?:is|to)?\s*(\d+)', text)
    if temp_match:
        try:
            updates['current_temp'] = float(temp_match.group(1))
        except ValueError:
            pass
            
    return updates
