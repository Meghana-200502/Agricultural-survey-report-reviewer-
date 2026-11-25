
from pathlib import Path

import pandas as pd

from train_model import train, LABEL_NAMES


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "agricultural_survey_dataset_staging.csv"


class AgriculturalReportChecker:

    def __init__(self):

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        self.data = pd.read_csv(DATA_PATH)

        self.data = self.data[
            self.data["label_status"] == "USER_CONFIRMED"
        ].copy()

        # Train a fresh model instead of loading an old saved model.
        print("\nTraining a fresh model for report checking...")

        self.model = train()

        print("\nReport Checker is ready.")

    def check_claim(self, claim):

        required = [
            "crop", "season", "district", "year",
            "area", "production", "yield"
        ]

        for field in required:
            if field not in claim:
                raise ValueError(f"Missing claim field: {field}")

        try:
            area = float(claim["area"])
            production = float(claim["production"])
            reported_yield = float(claim["yield"])

        except (TypeError, ValueError):
            raise ValueError(
                "Area, production and yield must be numbers."
            )

        if area <= 0 or production <= 0 or reported_yield <= 0:
            raise ValueError(
                "Area, production and yield must be positive."
            )

        # Find the matching confirmed reference record.
        match = self.data[
            (self.data["crop"].astype(str).str.strip() ==
             str(claim["crop"]).strip())
            &
            (self.data["season"].astype(str).str.strip() ==
             str(claim["season"]).strip())
            &
            (self.data["district"].astype(str).str.strip() ==
             str(claim["district"]).strip())
            &
            (self.data["year"].astype(str).str.strip() ==
             str(claim["year"]).strip())
        ]

        if match.empty:
            return {
                "status": "NOT_VERIFIABLE",
                "message": (
                    "No confirmed matching official record was found."
                ),
                "findings": []
            }

        reference = match.iloc[0]

        ref_area = float(reference["area_ha"])
        ref_production = float(reference["production_tonnes"])
        ref_yield = float(reference["yield_ton_per_ha"])

        if ref_area <= 0 or ref_production <= 0 or ref_yield <= 0:
            return {
                "status": "NOT_VERIFIABLE",
                "message": "Official reference contains invalid measurements.",
                "findings": []
            }

        calculated_yield = production / area

        def deviation(value, reference_value):

            return (
                (value - reference_value) / abs(reference_value)
            ) * 100

        # Prepare the same features used during training.
        features = {
            "crop": str(claim["crop"]),
            "season": str(claim["season"]),
            "district": str(claim["district"]),
            "year_start": float(str(claim["year"])[:4]),

            "reference_area": ref_area,
            "reference_production": ref_production,
            "reference_yield": ref_yield,

            "reported_area": area,
            "reported_production": production,
            "reported_yield": reported_yield,

            "area_deviation_pct": deviation(area, ref_area),

            "production_deviation_pct": deviation(
                production, ref_production
            ),

            "yield_deviation_pct": deviation(
                reported_yield, ref_yield
            ),

            "calculated_yield": calculated_yield,

            "yield_calculation_gap_pct": deviation(
                reported_yield, calculated_yield
            )
        }

        input_df = pd.DataFrame([features])

        # Predict using the freshly trained Neural Network.
        predicted_class = int(
            self.model.predict(input_df)[0]
        )

        findings = []

        # Rule-based comparison with reference values.
        tolerance_pct = 1.0

        comparisons = [
            ("Area", area, ref_area),
            ("Production", production, ref_production),
            ("Yield", reported_yield, ref_yield)
        ]

        for name, reported, official in comparisons:

            difference = abs(
                deviation(reported, official)
            )

            if difference > tolerance_pct:

                findings.append({
                    "type": f"{name} discrepancy",
                    "reported_value": reported,
                    "official_value": official,
                    "difference_percent": round(difference, 2)
                })

        # Check yield calculation separately.
        yield_gap = abs(
            deviation(reported_yield, calculated_yield)
        )

        if yield_gap > tolerance_pct:

            findings.append({
                "type": "Yield calculation inconsistency",
                "reported_yield": reported_yield,
                "calculated_yield": round(calculated_yield, 4),
                "difference_percent": round(yield_gap, 2)
            })

        return {
            "status": (
                "REVIEW_REQUIRED"
                if findings
                else "NO_RULE_DISCREPANCY"
            ),

            "ml_prediction": LABEL_NAMES[predicted_class],

            "findings": findings,

            "reference": {
                "area_ha": ref_area,
                "production_tonnes": ref_production,
                "yield_ton_per_ha": ref_yield
            },

            "calculated_yield": round(calculated_yield, 4)
        }
