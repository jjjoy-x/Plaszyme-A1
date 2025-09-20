import re
import torch
import esm
import numpy as np
import pandas as pd

def clean_sequence(seq: str) -> str:
    """Only keep 20 standard amino acids and convert to uppercase"""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(seq).upper())

def get_embeddings(sequences, batch_size=4):
    """ESM1b embedding (mean pooling per sequence)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = [(f"protein{i+j}", s) for j, s in enumerate(sequences[i:i+batch_size])]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]
        for j, (_, seq) in enumerate(batch):
            emb = reps[j, 1 : len(seq) + 1].mean(0).detach().cpu().numpy()
            embeddings.append(emb)
    return np.array(embeddings)

def expand_labels(df, label_col="degradable_plastics"):
    """Expand rows with multiple labels separated by ';' into multiple rows"""
    rows = []
    for _, row in df.iterrows():
        labels = str(row[label_col]).split(";")
        for lab in labels:
            lab = lab.strip()
            if lab:
                new_row = row.copy()
                new_row[label_col] = lab
                rows.append(new_row)
    return pd.DataFrame(rows)
