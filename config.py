from dataclasses import dataclass
import torch


@dataclass
class CONFIG:
    TRUE_CSV: str = "True.csv"
    FAKE_CSV: str = "Fake.csv"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 8
    EPOCHS: int = 2
    LR: float = 3e-3
