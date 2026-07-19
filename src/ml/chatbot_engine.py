import re

class AgronomistChatbot:
    def __init__(self):
        # Define basic intents using keywords
        self.intents = {
            "yield": ["yield", "harvest", "production", "output", "tons"],
            "disease": ["disease", "sick", "health", "yellow", "spots", "fungus", "virus", "bacteria"],
            "fertilizer": ["fertilizer", "npk", "nitrogen", "urea", "dap", "potassium", "phosphorus", "cost"],
            "price": ["price", "sell", "hold", "market", "mandi", "rate", "forecast"],
            "weather": ["weather", "rain", "temperature", "heat", "drought", "climate", "monsoon", "soil", "moisture"],
            "greeting": ["hello", "hi", "hey", "namaste", "help"]
        }
        
    def _get_intent(self, user_message):
        user_message = user_message.lower()
        # Find all matching intents based on keywords
        matched_intents = []
        for intent, keywords in self.intents.items():
            if any(re.search(r'\b' + kw + r'\b', user_message) for kw in keywords):
                matched_intents.append(intent)
        return matched_intents

    def get_response(self, user_message, context):
        """
        Generates a context-aware response based on the user's message and dashboard state.
        Context dictionary expected keys:
        - crop_type
        - region
        - predicted_yield
        - disease_status
        - fertilizer_cost
        - climate_scenario
        """
        intents = self._get_intent(user_message)
        
        # Default response if no intent is matched
        if not intents:
            return f"I'm your AI Agronomist! I can help you analyze your {context.get('crop_type', 'crop')} farm in {context.get('region', 'your area')}. You can ask me about yield predictions, disease status, fertilizer optimization, market prices, or climate impact."
            
        responses = []
        
        if "greeting" in intents:
            responses.append(f"Hello! I'm actively monitoring your {context.get('crop_type', 'crop')} farm in {context.get('region', 'your area')}. How can I assist you today?")
            
        if "yield" in intents:
            yield_val = context.get('predicted_yield', 0)
            if yield_val > 3.0:
                responses.append(f"Based on current models, your {context.get('crop_type', 'crop')} is on track for a strong yield of {yield_val:.2f} tons/ha. Keep up the good work!")
            else:
                responses.append(f"Your forecasted yield is currently {yield_val:.2f} tons/ha. This is slightly below optimal. Consider reviewing your soil moisture and fertilizer application.")
                
        if "disease" in intents:
            status = context.get('disease_status', 'No image uploaded').lower()
            if "no image" in status:
                responses.append("I cannot assess crop health right now. Please upload a leaf image using the sidebar so I can run the computer vision diagnostics.")
            elif "healthy" in status:
                responses.append("Great news! The computer vision model analyzed your leaf image and it appears healthy. Continue standard monitoring protocols.")
            else:
                clean_name = status.replace("___", " - ").replace("_", " ")
                responses.append(f"Alert: The model detected signs of {clean_name}. I recommend applying the appropriate fungicide or consulting a local expert immediately to prevent spread.")
                
        if "fertilizer" in intents:
            cost = context.get('fertilizer_cost', 0)
            responses.append(f"To optimize your {context.get('crop_type', 'crop')} growth, check the 'Cost Savings' section below the disease scanner. Based on your soil and yield forecast, the estimated optimal fertilizer cost is ₹{cost}.")
            
        if "price" in intents:
            responses.append(f"Market dynamics fluctuate. Check the 'Mandi Price Forecaster' chart on the dashboard for a live 30-day AI price prediction on {context.get('crop_type', 'crop')} to decide whether to sell now or hold inventory.")
            
        if "weather" in intents:
            scenario = context.get('climate_scenario', 'Normal Monsoon')
            if scenario != "Normal Monsoon":
                responses.append(f"You are currently viewing the '{scenario}' stress simulation. Notice how this negatively impacts your predicted yield in the charts.")
            else:
                responses.append("The current weather inputs are set to a Normal Monsoon baseline. You can adjust the temperature, rainfall, and soil moisture sliders to see how weather changes affect your crop.")

        # Combine all relevant responses
        if not responses:
            return "I have analyzed your request against the current farm parameters but couldn't generate a specific insight. Could you rephrase your question?"
            
        return " ".join(responses)
