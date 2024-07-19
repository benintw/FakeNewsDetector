import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def predict_fn(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    predictions_list = []
    targets_list = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch_seq, batch_label in tqdm(dataloader):
            batch_seq, batch_label = batch_seq.to(device), batch_label.to(device)
            logits = model(batch_seq)
            preds = torch.argmax(logits, dim=-1)
            loss = loss_fn(logits, batch_label)
            total_acc += (preds == batch_label).sum().cpu().numpy() / len(batch_label)
            total_loss += loss.item()
            predictions_list.append(preds.cpu().numpy())
            targets_list.append(batch_label.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    predictions_list = np.concatenate(predictions_list)
    targets_list = np.concatenate(targets_list)
    return avg_loss, avg_acc, predictions_list, targets_list
