import pandas as pd
from pathlib import Path
from datasets import load_dataset
import datasets
from datasets import Dataset


def get_custom_dataset(csv_path=Path.home()/'retweet-prediction-challenge/retweet_prediction_challenge/train.csv', test_size=0.1):
    print(csv_path.absolute())
    df = pd.read_csv(csv_path)
    df = pd.DataFrame(df)[['text', 'retweets_count']]
    df['retweets_count'] = df['retweets_count'].astype(float)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("retweets_count", "labels")
    return dataset.train_test_split(test_size=test_size, seed=42)
