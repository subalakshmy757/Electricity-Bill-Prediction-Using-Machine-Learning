"""
ALEC - Appliance Level Energy Consumption Prediction System
FastAPI Backend — Corrected & Production-Hardened
"""

import os
import uuid
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns

from utils import calculate_price

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from model import ALEC_TGCN
from train import train_model

# --------------------------------------------
# Logging Configuration
# --------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("alec")

# --------------------------------------------
# App Initialization
# --------------------------------------------
app = FastAPI(title="ALEC - Energy Consumption Prediction")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "saved_model/alec_model.pth"
SCALER_PATH = "saved_model/scaler.pkl"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# --------------------------------------------
# Load model safely at startup (if exists)
# --------------------------------------------
model = ALEC_TGCN(6, 32)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.warning(f"Failed to load saved model: {e}")
else:
    logger.info("No trained model found. Please upload dataset first.")


# --------------------------------------------
# HOME PAGE
# --------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# --------------------------------------------
# PREDICTION PAGE
# --------------------------------------------
@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})


# --------------------------------------------
# HANDLE PREDICTION
# --------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    fan: float = Form(...),
    fridge: float = Form(...),
    ac: float = Form(...),
    tv: float = Form(...),
    monitor: float = Form(...),
    motor: float = Form(...),
):
    try:
        # ---------- Validate inputs ----------
        inputs = {"Fan": fan, "Fridge": fridge, "AC": ac,
                  "TV": tv, "Monitor": monitor, "Motor": motor}
        for name, val in inputs.items():
            if val < 0 or val > 24:
                return templates.TemplateResponse(
                    "result.html",
                    {"request": request,
                     "prediction": f"Invalid {name} value ({val}). Must be 0–24 hours."}
                )

        # ---------- Check model exists ----------
        if not os.path.exists(MODEL_PATH):
            return templates.TemplateResponse(
                "result.html",
                {"request": request,
                 "prediction": "Model not trained yet. Please upload dataset first."}
            )

        # ---------- Load scaler for normalization ----------
        import joblib
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            input_data = np.array([[fan, fridge, ac, tv, monitor, motor]])
            input_normalized = scaler.transform(input_data)
        else:
            logger.warning("Scaler not found — using raw input (results may be inaccurate).")
            input_normalized = np.array([[fan, fridge, ac, tv, monitor, motor]])

        # ---------- Build sequence (tile to 24 steps) ----------
        input_seq = np.tile(input_normalized, (1, 24, 1))
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)

        # ---------- Run inference ----------
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()

        with torch.no_grad():
            prediction_normalized = model(input_tensor).numpy()

        # ---------- Inverse-transform to original scale ----------
        if os.path.exists(SCALER_PATH):
            prediction_original = scaler.inverse_transform(prediction_normalized)
        else:
            prediction_original = prediction_normalized

        total = float(np.sum(prediction_original))
        price = calculate_price(total*100)

        logger.info(f"Prediction successful: total={total:.4f}, price=₹{price:.2f}")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "prediction": round(total, 4), "price": round(price, 2)}
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return templates.TemplateResponse(
            "result.html",
            {"request": request,
             "prediction": f"Prediction failed: {str(e)}"}
        )


# --------------------------------------------
# CHARTS PAGE
# --------------------------------------------
@app.get("/charts", response_class=HTMLResponse)
def charts_page(request: Request):
    return templates.TemplateResponse("charts.html", {"request": request})


# --------------------------------------------
# PERFORMANCE PAGE (passes stored metrics)
# --------------------------------------------
@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    mae = getattr(app.state, "mae", None)
    rmse = getattr(app.state, "rmse", None)
    r2 = getattr(app.state, "r2", None)

    context = {"request": request}
    if mae is not None:
        context["mae"] = round(mae, 4)
    if rmse is not None:
        context["rmse"] = round(rmse, 4)
    if r2 is not None:
        context["r2"] = round(r2, 4)

    return templates.TemplateResponse("performance.html", context)


# --------------------------------------------
# UPLOAD PAGE
# --------------------------------------------
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# --------------------------------------------
# HANDLE DATASET UPLOAD + TRAINING
# --------------------------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    try:
        # ---------- Validate file type ----------
        if not file.filename or not file.filename.lower().endswith(".csv"):
            return templates.TemplateResponse(
                "upload.html",
                {"request": request,
                 "message": "Invalid file type. Please upload a .csv file."}
            )

        # ---------- Validate file size ----------
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            return templates.TemplateResponse(
                "upload.html",
                {"request": request,
                 "message": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB."}
            )

        # ---------- Save file with safe name ----------
        os.makedirs("data", exist_ok=True)
        os.makedirs("saved_model", exist_ok=True)

        safe_filename = f"{uuid.uuid4().hex}.csv"
        file_path = os.path.join("data", safe_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        logger.info(f"File saved: {file_path} ({len(contents)} bytes)")

        # ---------- Train model ----------
        mae, rmse, r2, sample_pred, A_matrix = train_model(file_path)

        # ---------- Save results to app state ----------
        app.state.mae = mae
        app.state.rmse = rmse
        app.state.r2 = r2
        app.state.sample_pred = sample_pred
        app.state.A_matrix = A_matrix

        # ---------- Reload model into memory ----------
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        logger.info("Model reloaded after training.")

        # ---------- Return success ----------
        return templates.TemplateResponse(
            "upload.html",
            {"request": request,
             "message": f"Training complete! MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"}
        )

    except ValueError as ve:
        logger.error(f"Validation error during upload: {ve}")
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "message": f"Dataset error: {str(ve)}"}
        )
    except Exception as e:
        logger.error(f"Upload/training error: {e}", exc_info=True)
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "message": f"Training failed: {str(e)}"}
        )


# --------------------------------------------
# CHART GENERATION HELPER
# --------------------------------------------
def generate_visuals():
    """Generate bar chart and adjacency matrix heatmap for the PDF report."""
    os.makedirs("static/images", exist_ok=True)

    # Bar Chart
    appliances = ["Fan", "Fridge", "AC", "TV", "Monitor", "Motor"]
    values = [10, 15, 40, 12, 8, 20]

    plt.figure()
    plt.bar(appliances, values, color=["#00f2fe", "#4facfe", "#a855f7",
                                        "#10b981", "#f59e0b", "#ef4444"])
    plt.title("Average Appliance Usage")
    plt.ylabel("Usage (kWh)")
    plt.tight_layout()
    plt.savefig("static/images/bar_chart.png", dpi=150)
    plt.close()

    # Adjacency Matrix Heatmap
    if hasattr(app.state, "A_matrix") and app.state.A_matrix is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            app.state.A_matrix,
            annot=True, fmt=".2f", cmap="viridis",
            xticklabels=appliances, yticklabels=appliances
        )
        plt.title("Learnable Adjacency Matrix")
        plt.tight_layout()
        plt.savefig("static/images/heatmap.png", dpi=150)
        plt.close()
    else:
        logger.warning("No adjacency matrix available — heatmap skipped.")


# --------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------
@app.get("/generate-report")
def generate_report():
    """Generate and download the ALEC performance PDF report."""

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    # Validate that training has been done
    if not hasattr(app.state, "mae") or app.state.mae is None:
        raise HTTPException(
            status_code=400,
            detail="No training results available. Please upload a dataset and train the model first."
        )

    try:
        os.makedirs("reports", exist_ok=True)
        generate_visuals()

        report_path = "reports/ALEC_Advanced_Report.pdf"
        doc = SimpleDocTemplate(report_path)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        elements.append(Paragraph("ALEC Advanced Prediction Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Metrics Table
        data = [
            ["Metric", "Value"],
            ["MAE", str(round(app.state.mae, 4))],
            ["RMSE", str(round(app.state.rmse, 4))],
        ]
        if hasattr(app.state, "r2") and app.state.r2 is not None:
            data.append(["R² Score", str(round(app.state.r2, 4))])

        table = Table(data)
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        # Prediction Sample
        elements.append(Paragraph("Sample Prediction Output:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(str(app.state.sample_pred), styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Bar Chart Image
        bar_path = "static/images/bar_chart.png"
        if os.path.exists(bar_path):
            elements.append(Paragraph("Appliance Usage Chart:", styles["Heading2"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Image(bar_path, width=4 * inch, height=3 * inch))
            elements.append(Spacer(1, 0.3 * inch))

        # Heatmap Image
        heatmap_path = "static/images/heatmap.png"
        if os.path.exists(heatmap_path):
            elements.append(Paragraph("Adjacency Matrix Heatmap:", styles["Heading2"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Image(heatmap_path, width=4 * inch, height=3 * inch))

        doc.build(elements)
        logger.info("PDF report generated successfully.")

        return FileResponse(
            path=report_path,
            filename="ALEC_Advanced_Report.pdf",
            media_type="application/pdf",
        )

    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# --------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "scaler_loaded": os.path.exists(SCALER_PATH),
    }