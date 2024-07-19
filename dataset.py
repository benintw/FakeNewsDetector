import torch
from torch.utils.data import Dataset
import tiktoken
import pandas as pd
from data_processing import process_dataframe


class NewsDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.df = process_dataframe(config.TRUE_CSV, config.FAKE_CSV)
        self.maxlen = self.get_maxlen()
        self.total_words = self.total_words_in_corpus()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.pad_token_id = 50256
        self.encoded_padded = self.encode_and_pad_texts()

    def get_maxlen(self):
        return max(self.df["clean_joined"].apply(lambda x: len(x.split())))

    def total_words_in_corpus(self):
        all_words = [word for text in self.df["clean"] for word in text]
        return len(set(all_words))

    def encode_and_pad_texts(self):
        encoded_texts = [
            self.tokenizer.encode(text) for text in self.df["clean_joined"]
        ]
        return [
            text + [self.pad_token_id] * (self.maxlen - len(text))
            for text in encoded_texts
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoded = self.encoded_padded[idx]
        label = self.df["isfake"].iloc[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )
