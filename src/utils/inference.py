import streamlit as st
from deep_translator import GoogleTranslator

@st.cache_data(show_spinner=False)
def translate_text(text, language_code):
    """
    Translates the given text into the target language using deep-translator.
    Returns the original text if language_code is 'en' or if translation fails.
    """
    if language_code == 'en':
        return text
        
    try:
        translator = GoogleTranslator(source='auto', target=language_code)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return text

def generate_agronomy_summary(yield_forecast, disease_status, disease_confidence, weather_data, language_code='en'):
    """
    Generates an automated text summary for the farmer explaining the forecast and giving recommendations.
    Translates the output if language_code is not 'en'.
    """
    summary = f"### Agronomy Interpretation\n\n"
    
    # Yield section
    summary += f"**Yield Forecast:** The current models project a yield of **{yield_forecast:.2f} tons/ha**. "
    if yield_forecast > 3.0:
        summary += "This indicates a strong harvest season, likely driven by optimal rainfall and healthy NDVI levels. "
    elif yield_forecast < 2.0:
        summary += "This forecast is lower than average. Consider reviewing irrigation or nutrient application. "
    else:
        summary += "This is an average expected yield for this region. "
        
    summary += "\n\n"
    
    # Disease section
    if "healthy" in disease_status.lower():
        summary += f"**Crop Health:** The uploaded leaf image appears **Healthy** (Confidence: {disease_confidence*100:.1f}%). Continue standard monitoring protocols."
    else:
        clean_disease_name = disease_status.replace("___", " - ").replace("_", " ")
        summary += f"**⚠️ Disease Alert:** The uploaded image shows signs of **{clean_disease_name}** (Confidence: {disease_confidence*100:.1f}%). "
        summary += "Immediate action is recommended. Consider targeted fungicide application or consulting a local agronomist."
        
    return translate_text(summary, language_code)
