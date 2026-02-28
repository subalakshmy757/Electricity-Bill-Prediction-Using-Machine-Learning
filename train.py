import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from model import ALEC_TGCN

def train_model(file_path):

    df = pd.read_csv(file_path)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)

    seq_length = 24
    sequences = []
    targets = []

    for i in range(len(data_scaled) - seq_length):
        sequences.append(data_scaled[i:i+seq_length])
        targets.append(data_scaled[i+seq_length])

    X = np.array(sequences)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = ALEC_TGCN(num_appliances=6, hidden_dim=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(30):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss += 0.01 * torch.norm(model.A, p='fro')
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    predictions = model(X_test).detach().numpy()
    y_test_np = y_test.numpy()

    mae = mean_absolute_error(y_test_np, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_np, predictions))

    # Save adjacency matrix
    A_matrix = model.A.detach().numpy()

    torch.save(model.state_dict(), "saved_model/alec_model.pth")

    return mae, rmse, predictions[:1], A_matrix