from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
import shutil
import os
from model import ALEC_TGCN
from train import train_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "saved_model/alec_model.pth"

# --------------------------------------------
# Load model safely (if exists)
# --------------------------------------------
model = ALEC_TGCN(6, 32)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully.")
else:
    print("No trained model found. Please upload dataset first.")


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
def predict(request: Request,
            fan: float = Form(...),
            fridge: float = Form(...),
            ac: float = Form(...),
            tv: float = Form(...),
            monitor: float = Form(...),
            motor: float = Form(...)):

    if not os.path.exists(MODEL_PATH):
        return templates.TemplateResponse(
            "result.html",
            {"request": request,
             "prediction": "Model not trained yet. Please upload dataset."}
        )

    input_data = np.array([[fan, fridge, ac, tv, monitor, motor]])
    input_seq = np.tile(input_data, (1, 24, 1))
    input_tensor = torch.tensor(input_seq, dtype=torch.float32)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    prediction = model(input_tensor).detach().numpy()
    total = float(np.sum(prediction))

    return templates.TemplateResponse(
        "result.html",
        {"request": request,
         "prediction": round(total, 4)}
    )


# --------------------------------------------
# CHARTS PAGE
# --------------------------------------------
@app.get("/charts", response_class=HTMLResponse)
def charts_page(request: Request):
    return templates.TemplateResponse("charts.html", {"request": request})


# --------------------------------------------
# PERFORMANCE PAGE
# --------------------------------------------
@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    return templates.TemplateResponse("performance.html", {"request": request})


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

    os.makedirs("data", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)

    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Train model and get metrics
    mae, rmse, sample_pred, A_matrix = train_model(file_path)

    # Save results globally
    app.state.mae = mae
    app.state.rmse = rmse
    app.state.sample_pred = sample_pred
    app.state.A_matrix = A_matrix

def generate_visuals():

    os.makedirs("static/images", exist_ok=True)

    # Bar Chart
    appliances = ['Fan','Fridge','AC','TV','Monitor','Motor']
    values = [10,15,40,12,8,20]

    plt.figure()
    plt.bar(appliances, values)
    plt.title("Average Appliance Usage")
    plt.savefig("static/images/bar_chart.png")
    plt.close()

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(app.state.A_matrix, annot=False, cmap="viridis")
    plt.title("Learnable Adjacency Matrix")
    plt.savefig("static/images/heatmap.png")
    plt.close()

#Handle-pdf
def generate_visuals():

    os.makedirs("static/images", exist_ok=True)

    # Bar Chart
    appliances = ['Fan','Fridge','AC','TV','Monitor','Motor']
    values = [10,15,40,12,8,20]

    plt.figure()
    plt.bar(appliances, values)
    plt.title("Average Appliance Usage")
    plt.savefig("static/images/bar_chart.png")
    plt.close()

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(app.state.A_matrix, annot=False, cmap="viridis")
    plt.title("Learnable Adjacency Matrix")
    plt.savefig("static/images/heatmap.png")
    plt.close()

#pdf generator

@app.get("/generate-report")
def generate_report():

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    os.makedirs("reports", exist_ok=True)

    generate_visuals()

    file_path = "reports/ALEC_Advanced_Report.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("ALEC Advanced Prediction Report", styles['Title']))
    elements.append(Spacer(1, 0.3 * inch))

    # Metrics Table
    data = [
        ["Metric", "Value"],
        ["MAE", str(round(app.state.mae,4))],
        ["RMSE", str(round(app.state.rmse,4))]
    ]

    table = Table(data)
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # Prediction Sample
    elements.append(Paragraph("Sample Prediction Output:", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(str(app.state.sample_pred), styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    # Bar Chart
    elements.append(Paragraph("Appliance Usage Chart:", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image("static/images/bar_chart.png", width=4*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))

    # Heatmap
    elements.append(Paragraph("Adjacency Matrix Heatmap:", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image("static/images/heatmap.png", width=4*inch, height=3*inch))

    doc.build(elements)

    return FileResponse(
        path=file_path,
        filename="ALEC_Advanced_Report.pdf",
        media_type='application/pdf'
    )