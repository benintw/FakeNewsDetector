import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        total_words,
        num_layers,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=2,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=total_words, embedding_dim=embedding_dim
        )
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.fc1 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(
            self.device
        )
        x = self.embedding(x)
        x, (hidden_state, cell_state) = self.bilstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
