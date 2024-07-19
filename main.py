import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report

from config import CONFIG
from dataset import NewsDataset
from model import BiLSTMModel
from train_eval import train_fn, eval_fn
from predict import predict_fn
from visualize import plot_metrics, plot_confusion_matrix


# Load dataset
config = CONFIG()
dataset = NewsDataset(config)

# Split dataset
train_full, test_ds = random_split(dataset, lengths=[0.9, 0.1])
train_ds, val_ds = random_split(train_full, lengths=[0.9, 0.1])

# Data loaders
train_dataloader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# Model, optimizer
model = BiLSTMModel(total_words=dataset.total_words, num_layers=2, device=config.DEVICE)
model.to(config.DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)

# Training and evaluation
best_val_loss = float("inf")
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    train_loss, train_acc = train_fn(model, train_dataloader, optimizer, config.DEVICE)
    val_loss, val_acc = eval_fn(model, val_dataloader, config.DEVICE)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_fake_ig_detector.pth")
        print("___SAVED BEST WEIGHTS___")
        best_val_loss = val_loss
    print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
    print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

# Plot metrics
plot_metrics(train_losses, val_losses, train_accs, val_accs)

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_fake_ig_detector.pth"))
test_loss, test_acc, predictions, ground_truths = predict_fn(
    model, test_dataloader, config.DEVICE
)
print(classification_report(ground_truths, predictions))
plot_confusion_matrix(ground_truths, predictions)
