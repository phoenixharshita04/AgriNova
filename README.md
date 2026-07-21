# 🌾 AgriNova: Intelligent Farming Dashboard

AgriNova is a comprehensive, AI-driven agricultural technology platform designed to help farmers make data-driven decisions. Built with Python and Streamlit, it combines Computer Vision for disease diagnostics, machine learning for yield prediction, and automated communication tools to ensure crop health and security.

## ✨ Key Features

*   **🔍 Computer Vision Disease Scanner:** Upload a photo of a crop leaf (e.g., Tomato), and the backend CNN model instantly predicts the disease (like Early Blight) and provides confidence scores.
*   **💊 Automated Treatment Plans:** Instantly generates actionable treatment recommendations based on the diagnosed plant disease.
*   **📈 Crop Yield Predictor:** Forecasts crop yields based on environmental data inputs like temperature and rainfall.
*   **🚨 Twilio SMS Alerts:** Automatically triggers and sends real-time critical SMS alerts to a registered mobile phone if severe disease or massive crop failure is detected.
*   **🔐 Secure Access:** Built-in secure user authentication and login system.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit (Python)
*   **Machine Learning/AI:** TensorFlow/Keras or PyTorch (CNN models)
*   **Alert System:** Twilio API
*   **Data Visualization:** Plotly / Grad-CAM Heatmaps

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/AgriNova.git](https://github.com/your-username/AgriNova.git)
cd AgriNova
