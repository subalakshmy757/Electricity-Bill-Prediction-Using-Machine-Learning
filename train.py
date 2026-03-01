"""
ALEC - Training Pipeline
Corrected: CSV validation, chronological split, scaler persistence,
           R² score, original-scale metrics, early stopping, error handling.
"""

import os
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import ALEC_TGCN

logger = logging.getLogger("alec.train")

EXPECTED_COLUMNS = ["fan", "fridge", "ac", "tv", "monitor", "motor"]
SEQ_LENGTH = 24
HIDDEN_DIM = 32
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
FROB_PENALTY = 0.01


def train_model(file_path: str):
    """
    Train the ALEC TGCN model on the uploaded CSV dataset.

    Returns:
        tuple: (mae, rmse, r2, sample_predictions, adjacency_matrix)

    Raises:
        ValueError: If the CSV is invalid, empty, or too small.
    """

    # ------------------------------------------------------------------
    # 1. Load and validate CSV
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Uploaded CSV file is empty.")

    # Check that all columns are present
    df.columns = df.columns.str.strip().str.lower()
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

    # Keep only expected columns in order
    df = df[EXPECTED_COLUMNS]

    # Check for non-numeric data
    if not df.apply(pd.to_numeric, errors="coerce").notnull().all().all():
        raise ValueError(
            "Dataset contains non-numeric values. All columns must be numeric."
        )

    df = df.astype(float)

    # Check minimum rows
    min_rows = SEQ_LENGTH + 2  # at least 1 train + 1 test sequence
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset too small. Need at least {min_rows} rows, got {len(df)}."
        )

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Normalize data & save scaler
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)

    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(scaler, "saved_model/scaler.pkl")
    logger.info("Scaler saved to saved_model/scaler.pkl")

    # ------------------------------------------------------------------
    # 3. Create sequences
    # ------------------------------------------------------------------
    sequences = []
    targets = []

    for i in range(len(data_scaled) - SEQ_LENGTH):
        sequences.append(data_scaled[i : i + SEQ_LENGTH])
        targets.append(data_scaled[i + SEQ_LENGTH])

    X = np.array(sequences)
    y = np.array(targets)

    logger.info(f"Created {len(X)} sequences (seq_length={SEQ_LENGTH})")

    # ------------------------------------------------------------------
    # 4. Chronological train/test split (no shuffle for time-series)
    # ------------------------------------------------------------------
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        split_idx = 1  # Ensure at least 1 training sample
    if split_idx == len(X):
        split_idx = len(X) - 1  # Ensure at least 1 test sample

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")

    # ------------------------------------------------------------------
    # 5. Initialize model
    # ------------------------------------------------------------------
    num_appliances = len(EXPECTED_COLUMNS)
    model = ALEC_TGCN(num_appliances=num_appliances, hidden_dim=HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 6. Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)
        loss += FROB_PENALTY * torch.norm(model.A, p="fro")

        loss.backward()
        optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_output = model(X_test)
            val_loss = criterion(val_output, y_test).item()

        logger.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS} — "
            f"Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {PATIENCE} epochs)"
                )
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 7. Evaluation on ORIGINAL scale
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        predictions_normalized = model(X_test).numpy()
    y_test_normalized = y_test.numpy()

    # Inverse-transform to original scale for meaningful metrics
    predictions_original = scaler.inverse_transform(predictions_normalized)
    y_test_original = scaler.inverse_transform(y_test_normalized)

    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = float(np.sqrt(mean_squared_error(y_test_original, predictions_original)))
    r2 = float(r2_score(y_test_original, predictions_original))

    logger.info(f"Evaluation — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # ------------------------------------------------------------------
    # 8. Save model and adjacency matrix
    # ------------------------------------------------------------------
    A_matrix = model.A.detach().numpy()
    torch.save(model.state_dict(), "saved_model/alec_model.pth")
    logger.info("Model saved to saved_model/alec_model.pth")

    return mae, rmse, r2, predictions_original[:1], A_matrix