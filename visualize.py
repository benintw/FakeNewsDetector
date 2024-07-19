import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    ax.plot(train_losses, label="Training")
    ax.plot(val_losses, label="Validation")
    ax.set(
        xlabel="Epochs",
        ylabel="Loss",
        title="Training vs. Validation Loss",
        xticks=np.arange(0, len(train_losses)),
    )
    ax.grid(True)
    ax.legend(title="Loss")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    ax.plot(train_accs, label="Training")
    ax.plot(val_accs, label="Validation")
    ax.set(
        xlabel="Epochs",
        ylabel="Loss",
        title="Training vs. Validation Loss",
        xticks=np.arange(0, len(train_accs)),
    )
    ax.grid(True)
    ax.legend(title="Loss")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(ground_truths, predictions):
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(ground_truths, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
