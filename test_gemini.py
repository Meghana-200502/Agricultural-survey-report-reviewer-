
from modules.gemini_service import GeminiService

gemini = GeminiService()

result = {
    "status": "REVIEW_REQUIRED",
    "ml_prediction": "Area error",
    "findings": [
        {
            "type": "Area discrepancy",
            "reported_value": 33310.5,
            "official_value": 22207.0,
            "difference_percent": 50.0
        },
        {
            "type": "Yield calculation inconsistency",
            "reported_yield": 1.26,
            "calculated_yield": 0.8415,
            "difference_percent": 49.74
        }
    ]
}

explanation = gemini.explain_findings(result)

print("\n--- GEMINI AGRICULTURAL REVIEW ---")
print(explanation)
