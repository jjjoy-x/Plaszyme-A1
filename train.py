import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

from utils import clean_sequence, get_embeddings, expand_labels
from model import get_histgb_model

LABEL_COL = "degradable_plastics"
ID_COL = "protein_id"
RANDOM_SEED = 42

def train(train_csv, work_dir, model_dir, batch_size=4):
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    df["sequence"] = df["sequence"].apply(clean_sequence)
    df = df[df["sequence"].str.len() > 0].reset_index(drop=True)

    df = expand_labels(df, LABEL_COL)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    X = get_embeddings(df["sequence"].tolist(), batch_size=batch_size)
    pd.DataFrame(X).to_csv(os.path.join(work_dir, "embeddings.csv"), index=False)

    label_counts = pd.Series(y).value_counts()
    rare_classes = label_counts[label_counts == 1].index
    if len(rare_classes) > 0:
        new_X, new_y = [], []
        for cls in rare_classes:
            idx = np.where(y == cls)[0][0]
            new_X.append(X[idx])
            new_y.append(cls)
        if new_X:
            X = np.vstack([X, np.array(new_X)])
            y = np.concatenate([y, np.array(new_y)])

    max_count = pd.Series(y).value_counts().max()
    min_class_size = pd.Series(y).value_counts().min()
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    smote = SMOTE(
        sampling_strategy={cls: max_count for cls in np.unique(y)},
        random_state=RANDOM_SEED,
        k_neighbors=k_neighbors
    )
    X_res, y_res = smote.fit_resample(X, y)
    pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name="label")], axis=1).to_csv(
        os.path.join(work_dir, "train_resampled.csv"), index=False
    )

    clf = get_histgb_model(random_seed=RANDOM_SEED)
    clf.fit(X_res, y_res)

    joblib.dump(clf, os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    print(f"Model and label encoder saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train(args.train_csv, args.work_dir, args.model_dir, batch_size=args.batch_size)
