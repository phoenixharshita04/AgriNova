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
    def _t(text):
        return translate_text(text, language_code)

    title = _t("### Agronomy Interpretation")
    
    # Yield section
    forecast_prefix = _t("**Yield Forecast:** The current models project a yield of")
    tons_suffix = _t("tons/ha")
    
    if yield_forecast > 3.0:
        yield_insight = _t("This indicates a strong harvest season, likely driven by optimal rainfall and healthy NDVI levels.")
    elif yield_forecast < 2.0:
        yield_insight = _t("This forecast is lower than average. Consider reviewing irrigation or nutrient application.")
    else:
        yield_insight = _t("This is an average expected yield for this region.")
        
    summary = f"{title}\n\n{forecast_prefix} **{yield_forecast:.2f}** {tons_suffix}. {yield_insight}\n\n"
    
    # Disease section
    if "healthy" in disease_status.lower():
        health_prefix = _t("**Crop Health:** The uploaded leaf image appears **Healthy**")
        conf_text = _t("(Confidence:")
        action_text = _t("Continue standard monitoring protocols.")
        summary += f"{health_prefix} {conf_text} {disease_confidence*100:.1f}%). {action_text}"
    else:
        clean_disease_name = disease_status.replace("___", " - ").replace("_", " ")
        alert_prefix = _t("**⚠️ Disease Alert:** The uploaded image shows signs of")
        conf_text = _t("(Confidence:")
        action_text1 = _t("Immediate action is recommended. Consider targeted fungicide application or consulting a local agronomist.")
        summary += f"{alert_prefix} **{clean_disease_name}** {conf_text} {disease_confidence*100:.1f}%). {action_text1}"
        
    return summary
