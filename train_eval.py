import torch
import torch.nn as nn
from tqdm import tqdm


def train_fn(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch_seq, batch_label in tqdm(dataloader):
        batch_seq, batch_label = batch_seq.to(device), batch_label.to(device)
        logits = model(batch_seq)
        preds = torch.argmax(logits, dim=-1)
        loss = loss_fn(logits, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_acc += (preds == batch_label).sum().cpu().numpy() / len(batch_label)
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


def eval_fn(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch_seq, batch_label in tqdm(dataloader):
            batch_seq, batch_label = batch_seq.to(device), batch_label.to(device)
            logits = model(batch_seq)
            preds = torch.argmax(logits, dim=-1)
            loss = loss_fn(logits, batch_label)
            total_acc += (preds == batch_label).sum().cpu().numpy() / len(batch_label)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc
