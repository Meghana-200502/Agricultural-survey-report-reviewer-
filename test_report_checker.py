
import pandas as pd

from modules.report_checker import AgriculturalReportChecker


# Load one confirmed record for a controlled test.
data = pd.read_csv("data/agricultural_survey_dataset_staging.csv")

confirmed = data[
    data["label_status"] == "USER_CONFIRMED"
].iloc[0]

# Create a test claim with an intentionally altered area.
claim = {
    "crop": confirmed["crop"],
    "season": confirmed["season"],
    "district": confirmed["district"],
    "year": confirmed["year"],
    "area": float(confirmed["area_ha"]) * 1.5,
    "production": float(confirmed["production_tonnes"]),
    "yield": float(confirmed["yield_ton_per_ha"]),
}

checker = AgriculturalReportChecker()
result = checker.check_claim(claim)

print("\n--- AGRICULTURAL REPORT REVIEW ---")
print("Test claim:", claim)
print("\nReview result:")

for key, value in result.items():
    print(f"{key}: {value}")
