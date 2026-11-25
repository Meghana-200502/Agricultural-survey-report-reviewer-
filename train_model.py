
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "agricultural_survey_dataset_staging.csv"

RANDOM_STATE = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CATEGORICAL_FEATURES = ["crop", "season", "district"]

NUMERIC_FEATURES = [
    "year_start",
    "reference_area",
    "reference_production",
    "reference_yield",
    "reported_area",
    "reported_production",
    "reported_yield",
    "area_deviation_pct",
    "production_deviation_pct",
    "yield_deviation_pct",
    "calculated_yield",
    "yield_calculation_gap_pct",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LABEL_NAMES = {
    0: "Valid",
    1: "Area error",
    2: "Production error",
    3: "Yield error",
    4: "Statistical anomaly",
}


# --------------------------------------------------
# 2. Neural Network architecture
# --------------------------------------------------

class AgriculturalNeuralNetwork(nn.Module):

    def __init__(self, input_size, number_of_classes=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, number_of_classes)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 3. Load confirmed agricultural records
# --------------------------------------------------

def load_confirmed_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = [
        "crop", "season", "district", "year",
        "area_ha", "production_tonnes",
        "yield_ton_per_ha", "label_status"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["label_status"] == "USER_CONFIRMED"].copy()

    for col in [
        "area_ha",
        "production_tonnes",
        "yield_ton_per_ha"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "crop", "season", "district", "year",
            "area_ha", "production_tonnes",
            "yield_ton_per_ha"
        ]
    )

    df = df[
        (df["area_ha"] > 0)
        & (df["production_tonnes"] > 0)
        & (df["yield_ton_per_ha"] > 0)
    ].copy()

    df["year_start"] = pd.to_numeric(
        df["year"].astype(str).str.extract(r"(\d{4})")[0],
        errors="coerce"
    )

    df = df.dropna(subset=["year_start"])

    if len(df) < 30:
        raise ValueError("Not enough confirmed records.")

    print("Confirmed records:", len(df))
    print("\nCrop-season combinations:")
    print(df.groupby(["crop", "season"]).size())

    return df


# --------------------------------------------------
# 4. Split data by year
# --------------------------------------------------

def split_by_year(df):

    years = sorted(df["year_start"].unique())

    if len(years) < 2:
        raise ValueError("At least two different years are required.")

    split_index = max(1, int(len(years) * 0.8))
    split_index = min(split_index, len(years) - 1)

    train_years = set(years[:split_index])
    test_years = set(years[split_index:])

    train_base = df[df["year_start"].isin(train_years)].copy()
    test_base = df[df["year_start"].isin(test_years)].copy()

    print("\nTraining years:", sorted(train_years))
    print("Testing years:", sorted(test_years))
    print("Training records:", len(train_base))
    print("Testing records:", len(test_base))

    return train_base, test_base


# --------------------------------------------------
# 5. Create report features
# --------------------------------------------------

def make_feature_row(record, area, production, reported_yield):

    ref_area = float(record["area_ha"])
    ref_production = float(record["production_tonnes"])
    ref_yield = float(record["yield_ton_per_ha"])

    calculated_yield = production / area if area > 0 else 0.0

    def deviation(value, reference):
        if reference == 0:
            return 0.0
        return ((value - reference) / abs(reference)) * 100

    return {
        "crop": str(record["crop"]),
        "season": str(record["season"]),
        "district": str(record["district"]),
        "year_start": float(record["year_start"]),

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
        ),
    }


# --------------------------------------------------
# 6. Generate synthetic training examples
# --------------------------------------------------

def generate_examples(base_df, seed=42):

    rng = np.random.default_rng(seed)
    rows = []

    for _, record in base_df.iterrows():

        area = float(record["area_ha"])
        production = float(record["production_tonnes"])
        reported_yield = float(record["yield_ton_per_ha"])

        # Class 0: Valid
        rows.append((
            make_feature_row(record, area, production, reported_yield),
            0
        ))

        # Class 1: Area error
        factor = float(rng.choice([0.5, 0.7, 1.3, 1.5]))
        rows.append((
            make_feature_row(
                record, area * factor, production, reported_yield
            ),
            1
        ))

        # Class 2: Production error
        factor = float(rng.choice([0.5, 0.7, 1.3, 1.5]))
        rows.append((
            make_feature_row(
                record, area, production * factor, reported_yield
            ),
            2
        ))

        # Class 3: Yield error
        factor = float(rng.choice([0.5, 0.7, 1.3, 1.5]))
        rows.append((
            make_feature_row(
                record, area, production, reported_yield * factor
            ),
            3
        ))

        # Class 4: Statistical anomaly
        factor = float(rng.choice([0.3, 0.4, 1.8, 2.0]))
        rows.append((
            make_feature_row(
                record,
                area * factor,
                production * factor,
                reported_yield
            ),
            4
        ))

    X = pd.DataFrame([item[0] for item in rows])
    y = np.array([item[1] for item in rows], dtype=np.int64)

    return X[FEATURES], y


# --------------------------------------------------
# 7. Model wrapper for predictions
# --------------------------------------------------

class TrainedAgriculturalModel:

    def __init__(self, preprocessor, network):
        self.preprocessor = preprocessor
        self.network = network

    def predict(self, X):

        transformed = self.preprocessor.transform(X)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        inputs = torch.tensor(
            np.asarray(transformed),
            dtype=torch.float32
        )

        self.network.eval()

        with torch.no_grad():
            outputs = self.network(inputs)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.numpy()


# --------------------------------------------------
# 8. Train the Neural Network for exactly 5 epochs
# --------------------------------------------------

def train():

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = load_confirmed_data()

    train_base, test_base = split_by_year(df)

    X_train, y_train = generate_examples(
        train_base, seed=RANDOM_STATE
    )

    X_test, y_test = generate_examples(
        test_base, seed=RANDOM_STATE + 1
    )

    # Prepare categorical and numerical columns separately.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ]
    )

    print("\nPreparing training data...")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_train_tensor = torch.tensor(
        np.asarray(X_train_transformed),
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(
        np.asarray(X_test_transformed),
        dtype=torch.float32
    )

    # Create neural network
    input_size = X_train_tensor.shape[1]

    network = AgriculturalNeuralNetwork(
        input_size=input_size,
        number_of_classes=5
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE
    )

    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Training loop
    print("\nStarting Neural Network training...")
    print("Total epochs:", EPOCHS)

    for epoch in range(EPOCHS):

        network.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = network(batch_X)
            loss = loss_function(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        average_loss = total_loss / len(dataset)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: {average_loss:.4f}"
        )

    # Evaluate on later-year synthetic examples
    network.eval()

    with torch.no_grad():
        outputs = network(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=[LABEL_NAMES[i] for i in range(5)],
        zero_division=0
    ))

    print("Confusion Matrix:")

    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4]
    ))

    print("\nFresh model trained successfully.")
    print("No previously saved model was loaded or used.")

    return TrainedAgriculturalModel(preprocessor, network)


if __name__ == "__main__":
    train()
