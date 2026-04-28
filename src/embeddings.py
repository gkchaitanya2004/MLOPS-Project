from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import torch
import os
from huggingface_hub import login

login(token="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")


## 1 Load model
model = AutoModel.from_pretrained("ai4bharat/indic-bert", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


## 2 Get Embeddings
def get_embeddings(texts, batch_size=32):
    embeddings = []
    num_batches = len(texts) // batch_size 

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

        if i > 0 and (i // batch_size) % 10 == 0:
            print(f"Processed {(i // batch_size)} batches out of {num_batches}")
            

    return np.vstack(embeddings)

def get_embeddings_single(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding




## 3 Save embeddings
if __name__ == "__main__":
    df = pd.read_csv('Data/Preprocessed/combined_prec.csv')
    texts = df['text'].tolist()
    embeddings = get_embeddings(texts)
    os.makedirs('Embeddings', exist_ok=True)
    np.save('Embeddings/text_embeddings.npy', embeddings)






