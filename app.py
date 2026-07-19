import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import ML modules
from src.ml.cv_pipeline import get_cv_model, process_image, predict_disease
from src.ml.yield_model import get_dummy_yield_model
from src.data.feature_engineering import generate_synthetic_features, generate_mandi_prices
from src.utils.inference import generate_agronomy_summary, translate_text
from src.ml.price_model import get_price_model, generate_price_recommendation
from src.ml.optimization_engine import ResourceOptimizer
from src.ml.chatbot_engine import AgronomistChatbot
from src.utils.database import init_db, insert_log, get_all_logs
from audio_recorder_streamlit import audio_recorder
from src.ml.voice_engine import transcribe_audio, extract_farm_parameters
from src.utils.alerts import send_alert

# Initialize Database
init_db()

# Page Config
st.set_page_config(page_title="AgriNova - Crop Yield & Disease Prediction", layout="wide")

# Language Setup
st.sidebar.header("Language / भाषा")
lang_choice = st.sidebar.selectbox("Select Language", ['English', 'Hindi', 'Marathi', 'Punjabi', 'Telugu', 'Tamil', 'Bengali', 'Gujarati', 'Kannada', 'Malayalam'])
lang_map = {
    'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr', 'Punjabi': 'pa', 
    'Telugu': 'te', 'Tamil': 'ta', 'Bengali': 'bn', 'Gujarati': 'gu', 
    'Kannada': 'kn', 'Malayalam': 'ml'
}
selected_lang_code = lang_map[lang_choice]

# Translation helper
def _t(text):
    return translate_text(text, selected_lang_code)

# Authentication Logic
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False

if not st.session_state['authentication_status']:
    st.title("🔒 " + _t("AgriNova Login"))
    
    username = st.text_input(_t("Username"))
    password = st.text_input(_t("Password"), type="password")
    submit_button = st.button(_t("Login"))
    
    if submit_button:
        if username.strip() == "admin" and password.strip() == "admin123":
            st.session_state['authentication_status'] = True
            st.rerun()
        else:
            st.error(_t("Incorrect Username or Password"))
    st.stop()  # Stop executing the rest of the app if not authenticated

st.title(_t("🌱 AgriNova Dashboard"))
st.subheader(_t("Crop Yield Forecasting & Early Disease Detection System"))

# Initialize models (cached)
@st.cache_resource
def load_cv_model():
    return get_cv_model()

@st.cache_resource
def load_yield_model():
    return get_dummy_yield_model()

@st.cache_resource
def load_optimization_engine():
    return ResourceOptimizer()

@st.cache_resource
def load_chatbot():
    return AgronomistChatbot()

@st.cache_resource
def load_price_model(crop):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=3)
    df_prices = generate_mandi_prices(crop, start_date, end_date)
    return get_price_model(df_prices), df_prices

with st.spinner(_t("Initializing AI Models (this may take a minute if downloading weights)...")):
    cv_model = load_cv_model()
    yield_model = load_yield_model()
    optimizer = load_optimization_engine()
    chatbot = load_chatbot()

# Initialize session state keys for widgets
default_states = {
    'region': "Punjab", 'crop_type': "Rice", 'current_temp': 32.0, 
    'current_rain': 800.0, 'current_soil': 45.0, 'current_ndvi_max': 0.75, 
    'current_ndvi_avg': 0.55, 'climate_scenario': "Normal Monsoon"
}
for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.header(_t("🎙️ Voice Command"))
audio_bytes = audio_recorder(text=_t("Click to speak..."), key="audio_recorder")
if audio_bytes:
    with st.sidebar.spinner(_t("Transcribing...")):
        text = transcribe_audio(audio_bytes)
        if "ERROR" in text:
            st.sidebar.error(text)
        else:
            st.sidebar.success(f'"{text}"')
            updates = extract_farm_parameters(text)
            if updates:
                for k, v in updates.items():
                    st.session_state[k] = v
                st.rerun()

st.sidebar.header(_t("Farm Parameters"))
region = st.sidebar.selectbox(_t("Select State"), ["Punjab", "Maharashtra", "Uttar Pradesh", "West Bengal"], format_func=lambda x: _t(x), key="region")
crop_type = st.sidebar.selectbox(_t("Select Crop"), ["Rice", "Wheat", "Cotton", "Sugarcane", "Soybean", "Tomato", "Potato"], format_func=lambda x: _t(x), key="crop_type")

st.sidebar.header(_t("Current Environmental Readings"))
current_temp = st.sidebar.slider(_t("Avg Temperature (°C)"), 10.0, 50.0, st.session_state.current_temp, key="current_temp")
current_rain = st.sidebar.slider(_t("Total Rainfall (mm)"), 0.0, 2000.0, st.session_state.current_rain, key="current_rain")
current_soil = st.sidebar.slider(_t("Avg Soil Moisture (%)"), 0.0, 100.0, st.session_state.current_soil, key="current_soil")
current_ndvi_max = st.sidebar.slider(_t("Max NDVI"), 0.0, 1.0, st.session_state.current_ndvi_max, key="current_ndvi_max")
current_ndvi_avg = st.sidebar.slider(_t("Avg NDVI"), 0.0, 1.0, st.session_state.current_ndvi_avg, key="current_ndvi_avg")

st.sidebar.header(_t("What-If Climate Scenario"))
climate_scenario = st.sidebar.selectbox(_t("Select Scenario"), ["Normal Monsoon", "Severe Drought", "Unseasonal Heatwave"], format_func=lambda x: _t(x), key="climate_scenario")

st.sidebar.header(_t("Leaf Image Upload"))
uploaded_file = st.sidebar.file_uploader(_t("Upload leaf image for disease check"), type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.header(_t("📱 SMS Alerts (Opt-in)"))
alert_opt_in = st.sidebar.checkbox(_t("Enable proactive SMS alerts"), value=False)
phone_number = st.sidebar.text_input(_t("Phone Number (e.g. +91...)")) if alert_opt_in else None

st.sidebar.markdown("---")
st.sidebar.header(_t("📜 Historical Farm Logs"))
# Display DB Logs
logs_df = get_all_logs()
if not logs_df.empty:
    display_df = logs_df[['timestamp', 'crop', 'predicted_yield']].copy()
    display_df.columns = [_t("Date"), _t("Crop"), _t("Yield")]
    st.sidebar.dataframe(display_df, width='stretch', hide_index=True)
else:
    st.sidebar.info(_t("No logs saved yet."))

# Layout Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### " + _t("📈 Environmental Time-Series Data"))
    
    # Generate mock historical data for visualization
    with st.spinner(_t("Loading remote sensing & weather data...")):
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(days=120) # One season
        df_ts = generate_synthetic_features(start_date, end_date, region)
    
    tab1, tab2, tab3 = st.tabs([_t("Weather & Soil"), _t("NDVI (Remote Sensing)"), _t("Historical Analytics")])
    
    with tab1:
        fig_weather = go.Figure()
        fig_weather.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Temp_7d_avg'], name=_t('Temp (7d avg)'), line=dict(color='orange')))
        fig_weather.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Soil_Moisture_pct'], name=_t('Soil Moisture %'), line=dict(color='blue')))
        fig_weather.update_layout(title=_t("Seasonal Weather Trends"), xaxis_title=_t("Date"), yaxis_title=_t("Value"))
        st.plotly_chart(fig_weather, width='stretch')
        
    with tab2:
        fig_ndvi = px.line(df_ts, x='Date', y=['NDVI', 'NDVI_7d_avg'], title=_t("Vegetation Index (NDVI) over Season"))
        st.plotly_chart(fig_ndvi, width='stretch')
        
    with tab3:
        st.markdown("#### " + _t("5-Year Historical Trends"))
        with st.spinner(_t("Generating 5-year historical dataset...")):
            hist_start = end_date - pd.DateOffset(years=5)
            df_hist = generate_synthetic_features(hist_start, end_date, region)
            
            # Multi-line chart tracking NDVI vs Rainfall
            fig_hist = go.Figure()
            # We aggregate to monthly to make the 5-year chart readable
            df_hist_monthly = df_hist.resample('ME', on='Date').mean(numeric_only=True).reset_index()
            
            fig_hist.add_trace(go.Scatter(x=df_hist_monthly['Date'], y=df_hist_monthly['NDVI'], name=_t('Avg NDVI'), yaxis='y1', line=dict(color='green')))
            fig_hist.add_trace(go.Scatter(x=df_hist_monthly['Date'], y=df_hist_monthly['Rainfall_mm'], name=_t('Avg Daily Rain (mm)'), yaxis='y2', line=dict(color='blue')))
            
            fig_hist.update_layout(
                title=_t("5-Year Historical: NDVI vs Rainfall"),
                xaxis=dict(title=_t("Date")),
                yaxis=dict(title=_t("NDVI"), title_font=dict(color="green"), tickfont=dict(color="green")),
                yaxis2=dict(title=_t("Rainfall (mm)"), title_font=dict(color="blue"), tickfont=dict(color="blue"), overlaying="y", side="right")
            )
            st.plotly_chart(fig_hist, width='stretch')
            
            # Download Button
            csv_data = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 " + _t("Download 5-Year Historical Data (CSV)"),
                data=csv_data,
                file_name=f"{region}_5yr_historical_data.csv",
                mime='text/csv'
            )
        
    st.markdown("### " + _t("🗺️ Geospatial View"))
    # Dummy geospatial data for the farms
    farm_locations = pd.DataFrame({
        'State': ['Punjab', 'Maharashtra', 'Uttar Pradesh', 'West Bengal'],
        'lat': [31.1471, 19.7515, 26.8467, 22.9868],
        'lon': [75.3412, 75.7139, 80.9462, 87.8550],
        'Health_Score': [0.85, 0.70, 0.90, 0.75]
    })
    
    fig_map = px.scatter_mapbox(farm_locations, lat="lat", lon="lon", hover_name="State", 
                                color="Health_Score", size_max=15, zoom=4, height=350,
                                color_continuous_scale=px.colors.diverging.RdYlGn)
    fig_map.update_layout(mapbox_style="carto-positron")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, width='stretch')
    
    st.markdown("### " + _t("📊 Mandi Price Forecaster (30-Day)"))
    with st.spinner(_t("Generating price forecast...")):
        price_model, df_hist_prices = load_price_model(crop_type)
        forecast_prices = price_model.forecast(days=30)
        
        # Prepare data for plotting
        last_30_actual = df_hist_prices.tail(30)
        
        future_dates = pd.date_range(start=pd.Timestamp.today() + pd.DateOffset(days=1), periods=30, freq='D')
        df_forecast = pd.DataFrame({
            'Date': future_dates,
            'Price_INR_per_Qtl': forecast_prices,
            'Type': _t('Forecast')
        })
        
        df_actual_plot = last_30_actual.copy()
        df_actual_plot['Type'] = _t('Actual')
        
        plot_df = pd.concat([df_actual_plot[['Date', 'Price_INR_per_Qtl', 'Type']], df_forecast])
        
        fig_price = px.line(plot_df, x='Date', y='Price_INR_per_Qtl', color='Type',
                            title=f"{crop_type} " + _t("Price Trend & 30-Day Forecast"),
                            color_discrete_map={_t('Actual'): 'blue', _t('Forecast'): 'orange'})
        st.plotly_chart(fig_price, width='stretch')
        
        # Recommendation
        decision, rationale = generate_price_recommendation(forecast_prices)
        translated_rationale = _t(rationale)
        translated_decision = _t(decision)
        if decision == "Sell Now":
            st.error(f"**" + _t("AI Recommendation") + f": {translated_decision}** - {translated_rationale}")
        else:
            st.success(f"**" + _t("AI Recommendation") + f": {translated_decision}** - {translated_rationale}")

with col2:
    st.markdown("### " + _t("🔮 Yield Forecast"))
    
    current_features = {
        'Avg_Temp': current_temp,
        'Total_Rainfall': current_rain,
        'Avg_Soil_Moisture': current_soil,
        'Max_NDVI': current_ndvi_max,
        'Avg_NDVI': current_ndvi_avg
    }
    
    predicted_yield = yield_model.predict(current_features)
    stressed_yield = yield_model.predict_what_if(current_features, climate_scenario)
    
    st.metric(label=_t("Baseline Predicted Yield (tons/ha)"), value=f"{predicted_yield:.2f}")
    
    if climate_scenario != "Normal Monsoon":
        delta_val = stressed_yield - predicted_yield
        st.metric(label=_t("Scenario") + f": {_t(climate_scenario)}", value=f"{stressed_yield:.2f}", delta=f"{delta_val:.2f} " + _t("tons/ha"), delta_color="inverse")
        
        if alert_opt_in and phone_number:
            drop_percentage = (abs(delta_val) / predicted_yield) * 100
            if delta_val < 0 and drop_percentage > 10:
                alert_msg = f"URGENT: Projected yield drop of {drop_percentage:.1f}% detected under {climate_scenario} conditions."
                success, msg = send_alert(phone_number, alert_msg, selected_lang_code)
                if success:
                    st.toast(_t("SMS Alert Sent") + f": {msg}")
                    
        # Plot comparison chart
        comp_df = pd.DataFrame({
            'Scenario': [_t('Baseline'), _t(climate_scenario)],
            'Yield (tons/ha)': [predicted_yield, stressed_yield]
        })
        fig_comp = px.bar(comp_df, x='Scenario', y='Yield (tons/ha)', color='Scenario', 
                          title=_t("Climate Stress Impact on Yield"),
                          color_discrete_map={_t('Baseline'): 'green', _t(climate_scenario): 'red'})
        st.plotly_chart(fig_comp, width='stretch')
        
    st.markdown("### " + _t("🔬 Disease Detection"))
    disease_status = "No image uploaded"
    disease_conf = 0.0
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=_t("Uploaded Leaf Image"), use_column_width=True)
        
        with st.spinner(_t("Analyzing image...")):
            img_tensor = process_image(uploaded_file)
            disease_status, disease_conf = predict_disease(img_tensor, model=None) 
            
        if "healthy" in disease_status.lower():
            st.success(_t("Status") + f": {_t(disease_status.replace('___', ' - '))}")
        else:
            st.error(_t("Status") + f": {_t(disease_status.replace('___', ' - '))}")
            if disease_conf > 0.85 and alert_opt_in and phone_number:
                alert_msg = f"CRITICAL: {disease_status.replace('___', ' - ')} detected with {disease_conf*100:.1f}% confidence. Immediate treatment recommended."
                success, msg = send_alert(phone_number, alert_msg, selected_lang_code)
                if success:
                    st.toast(_t("SMS Alert Sent") + f": {msg}")
        st.info(_t("Confidence") + f": {disease_conf*100:.1f}%")
    else:
        st.info(_t("Upload a leaf image to run the computer vision model."))
        
    st.markdown("### " + _t("💰 Cost Savings & Resource Optimization"))
    with st.spinner(_t("Calculating NPK Requirements...")):
        optimization_results = optimizer.optimize_npk(crop_type, predicted_yield, current_soil)
        
        st.info(f"**" + _t("Optimal Fertilizer Requirements (kg/ha)") + "**\n\n" +
                f"**N (Nitrogen):** {optimization_results['N_kg_ha']} kg\n\n" +
                f"**P (Phosphorus):** {optimization_results['P_kg_ha']} kg\n\n" +
                f"**K (Potassium):** {optimization_results['K_kg_ha']} kg")
        
        st.metric(label=_t("Estimated Fertilizer Cost"), value=f"₹{optimization_results['Total_Cost_INR']}")
        st.caption(_t("AI Insight") + f": {_t(optimization_results['Savings_Insight'])}")

    st.markdown("---")
    # Agronomy Summary
    # generate_agronomy_summary handles its own internal translation string logic
    summary = generate_agronomy_summary(predicted_yield, disease_status, disease_conf, current_features, language_code=selected_lang_code)
    st.markdown(summary)
    
    # Save Button for SQLite Database
    if st.button(_t("💾 Save Current Scan to Log")):
        insert_log(region, crop_type, predicted_yield, disease_status)
        st.success(_t("Log saved successfully to database!"))
        # Rerun to update the sidebar dataframe
        st.rerun()

# --- AI Agronomist Chatbot (Sidebar Bottom) ---
st.sidebar.markdown("---")
st.sidebar.header(_t("💬 AI Agronomist Chat"))

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": _t("Hello! I am your AI Agronomist. How can I help you today?")}]

for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.sidebar.chat_input(_t("Ask about yield, disease, prices...")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)
        
    chat_context = {
        "crop_type": _t(crop_type),
        "region": _t(region),
        "predicted_yield": predicted_yield,
        "disease_status": _t(disease_status),
        "fertilizer_cost": optimization_results['Total_Cost_INR'],
        "climate_scenario": _t(climate_scenario)
    }
    
    # Get response from local Python NLP engine
    raw_response = chatbot.get_response(prompt, chat_context)
    
    # Translate response to selected language
    translated_response = _t(raw_response)
    
    st.session_state.messages.append({"role": "assistant", "content": translated_response})
    with st.sidebar.chat_message("assistant"):
        st.markdown(translated_response)

# --- Logout (Sidebar Bottom) ---
st.sidebar.markdown("---")
if st.sidebar.button("🚪 " + _t("Logout")):
    st.session_state['authentication_status'] = False
    st.rerun()
