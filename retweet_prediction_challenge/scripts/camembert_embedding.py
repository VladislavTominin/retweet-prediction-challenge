import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import CamembertModel, CamembertTokenizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd

save_path = Path('/shared/personal/vladtom/camembert_embeddings')


def text_transform(X, key='text', device='cuda'):
    tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large")
    camembert = CamembertModel.from_pretrained("camembert/camembert-large")
    camembert.to(device)
    camembert.eval()

    for i, sample in tqdm(enumerate(X[key])):
        tokenized_sentence = tokenizer.tokenize(sample)
        encoded_sentence = tokenizer.encode(tokenized_sentence)
        encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0).to(device)
        torch.save(camembert(encoded_sentence).last_hidden_state.squeeze().detach().cpu(), save_path / f'tensor{i}.pt')
        torch.cuda.empty_cache()


train_data = pd.read_csv("../train.csv")
text_transform(train_data)
