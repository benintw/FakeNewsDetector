import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


def process_dataframe(true_csv, fake_csv):
    df_true = pd.read_csv(true_csv)
    df_fake = pd.read_csv(fake_csv)

    df_true = df_true.assign(isfake=0)
    df_fake = df_fake.assign(isfake=1)

    df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)
    df = df.drop(columns=["date"], inplace=False)
    df = df.assign(original=df["title"] + " " + df["text"])
    df = df.assign(clean=df["original"].apply(stop_words_preprocess))
    df = df.assign(clean_joined=df["clean"].apply(lambda x: " ".join(x)))
    return df


def stop_words_preprocess(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in stop_words and len(token) > 2]
