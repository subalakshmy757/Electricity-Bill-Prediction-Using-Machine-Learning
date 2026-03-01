#!/usr/bin/env python3
"""
ALEC Application Test Script
Run with: .venv/bin/python test_app.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_model():
    """Test ALEC_TGCN model forward pass."""
    from model import ALEC_TGCN
    import torch

    model = ALEC_TGCN(num_appliances=6, hidden_dim=32)
    x = torch.randn(4, 24, 6)
    out = model(x)
    assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"
    print("[PASS] Model forward pass")


def test_training():
    """Test training pipeline on sample dataset."""
    from train import train_model

    mae, rmse, r2, pred, A = train_model("data/ALEC_sample_dataset.csv")
    assert mae >= 0, "MAE should be non-negative"
    assert rmse >= 0, "RMSE should be non-negative"
    assert A.shape == (6, 6), f"Adjacency shape should be (6,6), got {A.shape}"
    print(f"[PASS] Training: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")


def test_imports():
    """Test all app imports."""
    import app  # noqa: F401

    print("[PASS] App imports")


if __name__ == "__main__":
    print("=" * 50)
    print("ALEC Application Tests")
    print("=" * 50)
    try:
        test_model()
        test_training()
        test_imports()
        print("=" * 50)
        print("All tests passed!")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
