
import pandas as pd
from pathlib import Path


class AgriculturalDataValidator:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.df = pd.read_csv(self.dataset_path)

        # Required dataset columns
        self.required_columns = [
            "crop",
            "season",
            "state",
            "district",
            "year",
            "area_ha",
            "production_tonnes",
            "yield_ton_per_ha",
            "source_file",
            "source_upload_group",
            "label_status",
            "data_note"
        ]

        self.check_required_columns()

        # Convert measurement columns into numeric values
        numeric_columns = [
            "area_ha",
            "production_tonnes",
            "yield_ton_per_ha"
        ]

        for column in numeric_columns:
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce"
            )

    # --------------------------------------------------
    # 1. CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    def check_required_columns(self):

        missing_columns = [
            column
            for column in self.required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    # --------------------------------------------------
    # 2. CHECK MISSING VALUES
    # --------------------------------------------------

    def check_missing_values(self):

        return self.df.isnull().sum()

    # --------------------------------------------------
    # 3. CHECK YIELD CALCULATION
    # Yield = Production / Area
    # --------------------------------------------------

    def check_yield(self, tolerance=0.01):

        data = self.df.copy()

        # Avoid division by zero
        valid_area = data["area_ha"] > 0

        data["calculated_yield"] = pd.NA

        data.loc[valid_area, "calculated_yield"] = (
            data.loc[valid_area, "production_tonnes"] /
            data.loc[valid_area, "area_ha"]
        )

        data["yield_difference"] = (
            data["yield_ton_per_ha"] -
            data["calculated_yield"]
        ).abs()

        inconsistent = data[
            data["yield_difference"] > tolerance
        ]

        return inconsistent

    # --------------------------------------------------
    # 4. CHECK INVALID MEASUREMENTS
    # --------------------------------------------------

    def check_invalid_values(self):

        invalid = self.df[
            (self.df["area_ha"] <= 0) |
            (self.df["production_tonnes"] <= 0) |
            (self.df["yield_ton_per_ha"] <= 0)
        ]

        return invalid

    # --------------------------------------------------
    # 5. GET CONFIRMED RECORDS
    # --------------------------------------------------

    def get_confirmed_records(self):

        confirmed = self.df[
            self.df["label_status"] == "USER_CONFIRMED"
        ].copy()

        return confirmed

    # --------------------------------------------------
    # 6. GET UNCONFIRMED RECORDS
    # --------------------------------------------------

    def get_unconfirmed_records(self):

        unconfirmed = self.df[
            self.df["label_status"] == "UNCONFIRMED"
        ].copy()

        return unconfirmed

    # --------------------------------------------------
    # 7. SHOW CROP-SEASON SUMMARY
    # --------------------------------------------------

    def get_crop_season_summary(self):

        confirmed = self.get_confirmed_records()

        if confirmed.empty:
            return pd.DataFrame(
                columns=["crop", "season", "record_count"]
            )

        summary = (
            confirmed
            .groupby(["crop", "season"])
            .size()
            .reset_index(name="record_count")
        )

        return summary

    # --------------------------------------------------
    # 8. RUN COMPLETE DATASET VALIDATION
    # --------------------------------------------------

    def validate_dataset(self):

        print("\n======================================")
        print(" AGRICULTURAL DATASET VALIDATION REPORT")
        print("======================================")

        total_records = len(self.df)

        confirmed = self.get_confirmed_records()
        unconfirmed = self.get_unconfirmed_records()

        missing_values = self.check_missing_values()
        yield_errors = self.check_yield()
        invalid_values = self.check_invalid_values()

        print("\n1. DATASET INFORMATION")

        print("Total records:", total_records)
        print("Total columns:", len(self.df.columns))

        print("\n2. MISSING VALUES")

        print(missing_values)

        print(
            "\nTotal missing values:",
            int(missing_values.sum())
        )

        print("\n3. YIELD VALIDATION")

        print(
            "Yield inconsistencies:",
            len(yield_errors)
        )

        print("\n4. INVALID MEASUREMENTS")

        print(
            "Invalid measurements:",
            len(invalid_values)
        )

        print("\n5. RECORD LABEL STATUS")

        print(
            "Confirmed records:",
            len(confirmed)
        )

        print(
            "Unconfirmed records:",
            len(unconfirmed)
        )

        print("\n6. CONFIRMED CROP-SEASON COMBINATIONS")

        crop_summary = self.get_crop_season_summary()

        if crop_summary.empty:
            print("No confirmed crop-season records found.")
        else:
            print(crop_summary.to_string(index=False))

        print("\n======================================")
        print(" VALIDATION COMPLETED")
        print("======================================")

        return {
            "total_records": total_records,
            "confirmed_records": len(confirmed),
            "unconfirmed_records": len(unconfirmed),
            "missing_values": int(missing_values.sum()),
            "yield_inconsistencies": len(yield_errors),
            "invalid_measurements": len(invalid_values)
        }


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    dataset_path = (
        "data/agricultural_survey_dataset_staging.csv"
    )

    validator = AgriculturalDataValidator(dataset_path)

    results = validator.validate_dataset()

    print("\nFINAL SUMMARY")
    print(results)
