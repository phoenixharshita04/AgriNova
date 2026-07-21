class ResourceOptimizer:
    def __init__(self):
        # Baseline NPK requirements (Nitrogen, Phosphorus, Potassium) in kg/ha
        # Approximate baseline for average Indian yield scenarios
        self.crop_baselines = {
            "Rice": {"N": 120, "P": 60, "K": 40},
            "Wheat": {"N": 120, "P": 60, "K": 40},
            "Cotton": {"N": 150, "P": 75, "K": 75},
            "Sugarcane": {"N": 250, "P": 100, "K": 100},
            "Soybean": {"N": 30, "P": 60, "K": 40}, # Legume, needs less N
            "Tomato": {"N": 150, "P": 80, "K": 100},
            "Potato": {"N": 180, "P": 80, "K": 120}
        }
        
        # Approximate average Indian retail prices (INR per kg of nutrient)
        # E.g., Urea is ~₹6/kg but is only 46% N, so ~₹13/kg of N. Let's use simple estimates.
        self.prices_inr_per_kg = {
            "N": 13, # Urea equivalent
            "P": 50, # DAP equivalent
            "K": 35  # MOP equivalent
        }

    def optimize_npk(self, crop_type, predicted_yield, soil_moisture):
        """
        Calculates the precise minimum NPK required and the estimated cost.
        Returns a dictionary with N, P, K requirements in kg/ha, and Total_Cost_INR.
        """
        baseline = self.crop_baselines.get(crop_type, {"N": 100, "P": 50, "K": 50})
        
        # Yield Multiplier: Higher predicted yield means the crop will extract more nutrients.
        # We assume baseline is for ~2.5 tons/ha. We scale proportionally.
        yield_multiplier = max(0.5, predicted_yield / 2.5)
        
        # Soil Moisture Efficiency: 
        # If soil is too dry (<20%) or too wet (>80%), nutrient uptake efficiency drops,
        # requiring a buffer (+15%). Optimal moisture (40-60%) allows a reduction (-10%).
        if soil_moisture < 20 or soil_moisture > 80:
            moisture_multiplier = 1.15
        elif 40 <= soil_moisture <= 60:
            moisture_multiplier = 0.90
        else:
            moisture_multiplier = 1.0
            
        final_multiplier = yield_multiplier * moisture_multiplier
        
        # Calculate requirements
        req_n = round(baseline["N"] * final_multiplier, 1)
        req_p = round(baseline["P"] * final_multiplier, 1)
        req_k = round(baseline["K"] * final_multiplier, 1)
        
        # Calculate Cost
        cost_n = req_n * self.prices_inr_per_kg["N"]
        cost_p = req_p * self.prices_inr_per_kg["P"]
        cost_k = req_k * self.prices_inr_per_kg["K"]
        total_cost = round(cost_n + cost_p + cost_k, 2)
        
        return {
            "N_kg_ha": req_n,
            "P_kg_ha": req_p,
            "K_kg_ha": req_k,
            "Total_Cost_INR": total_cost,
            "Adjustment_Pct": abs(round((1 - final_multiplier)*100)),
            "Savings_Insight": f"Adjusted by {abs(round((1 - final_multiplier)*100))}% based on yield forecast and soil moisture."
        }
