
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import mlflow
import pandas as pd
import numpy as np
import matplotlib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt


## 1 MLflow initialization
mlflow.set_experiment("SafetyGuard Rail For Indic Languages")

### 1.1 Enable system metrics logging
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


## 2 Data Loading and embedd text data
df = pd.read_csv('Data/Preprocessed/combined_prec.csv')
texts = np.load('Embeddings/text_embeddings.npy')
labels = df['labels'].values
lang_labels = df['language'].values

df['stratify_col'] = df['language'].astype(str) + "_" + df['labels'].astype(str)

## 3 Data Splitting
X_train, X_val, y_train, y_val = train_test_split(
    texts, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=df['stratify_col']
)

train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

## 4 Parameters
parser = argparse.ArgumentParser(description='Param parsing for Hate Speech Classification')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--learning_rate','--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')

args = parser.parse_args()
epochs = args.num_epochs
batch_size = args.batch_size
lr = args.learning_rate
p = args.dropout

### 5 Data Loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

params = {
    "optimizer": "Adam",
    "model_type": "FNN",
    "hidden_units": [1024, 512, 256, 128],
    "dropout": args.dropout
}

hidden_units = params["hidden_units"]

## 6 Model Building
 
class HateSpeech(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden_units[2], hidden_units[3]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden_units[3], num_classes)
        )

    def forward(self, x):
        x = x.float()
        x = self.classifier(x)
        return x
    

## 7 Model Training

model = HateSpeech(num_classes=2).to(device)
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print("Starting training...\n")

with mlflow.start_run(run_name=f"epochs_{epochs}_lr_{lr}_batch_{batch_size}_dropout_{p}"):
    mlflow.log_params(params)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        all_preds = []
        all_labels = []

        for text, batch_labels in train_loader:
                        
            text = text.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(text)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        mlflow.log_metrics({
            "train_loss": round(avg_train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "train_precision": round(train_precision, 4),
            "train_recall": round(train_recall, 4),
            "train_f1_score": round(train_f1, 4)
        }, step=epoch + 1)

        
        if (epoch + 1) % 5 == 0:
            mlflow.pytorch.log_model(model, artifact_path=f"checkpoint_{epoch + 1}")

        ### Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for text, batch_labels in val_loader:
                text = text.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(text)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

                loss = loss_fn(outputs, batch_labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='weighted')
            val_recall = recall_score(val_labels, val_preds, average='weighted')
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            mlflow.log_metrics({
                "val_loss": round(avg_val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_precision": round(val_precision, 4),
                "val_recall": round(val_recall, 4),
                "val_f1_score": round(val_f1, 4)
            }, step=epoch + 1)

           

    model_name = f"final_model_epochs_{epochs}_lr_{lr}_batch_{batch_size}_dropout_{p}"
    model_info = mlflow.pytorch.log_model(model, artifact_path="final_model")
    torch.save(model, f"Models/{model_name}.pt")

    print("Training Done!")
    print("="*40)
    