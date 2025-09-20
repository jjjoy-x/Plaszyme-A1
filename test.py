import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import clean_sequence, get_embeddings

LABEL_COL = "degradable_plastics"
ID_COL = "protein_id"
BUCKET_COL = "bucket_by_nn_all"   

def evaluate_multilabel(y_true, y_score, classes, top_k=(1,3,5), threshold=None):
    """Multi-label evaluation with top-k and thresholding methods."""
    results = {}

    y_true_bin = np.zeros_like(y_score, dtype=int)
    for i, labs in enumerate(y_true):
        for lab in str(labs).split(";"):
            lab = lab.strip()
            if lab in classes:
                y_true_bin[i, classes.index(lab)] = 1

    # Top-k hit/recall
    for k in top_k:
        hits, recalls = [], []
        for i in range(len(y_true_bin)):
            top_k_idx = np.argsort(y_score[i])[::-1][:k]
            true_idx = np.where(y_true_bin[i] == 1)[0]
            if len(true_idx) == 0:
                continue
            hit = int(len(set(top_k_idx) & set(true_idx)) > 0)
            hits.append(hit)
            recall = len(set(top_k_idx) & set(true_idx)) / len(true_idx)
            recalls.append(recall)
        results[f"hit@{k}"] = np.mean(hits) if hits else 0.0
        results[f"recall@{k}"] = np.mean(recalls) if recalls else 0.0

    # Metrics at given threshold
    if threshold is not None:
        y_pred_bin = (y_score >= threshold).astype(int)
        results["micro_precision"] = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        results["micro_recall"] = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        results["micro_f1"] = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        results["macro_precision"] = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        results["macro_recall"] = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        results["macro_f1"] = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    return results


def test(test_csv, model_dir, out_dir, batch_size=4, threshold=0.3):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(test_csv)
    df["sequence"] = df["sequence"].apply(clean_sequence)
    df = df[df["sequence"].str.len() > 0].reset_index(drop=True)

    clf = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    X_test = get_embeddings(df["sequence"].tolist(), batch_size=batch_size)

    proba = clf.predict_proba(X_test)
    classes = le.inverse_transform(clf.classes_)

    preds_df = pd.DataFrame(proba, columns=classes)
    preds_df.insert(0, ID_COL, df[ID_COL].astype(str).values)
    preds_file = os.path.join(out_dir, "preds_histgbdt.csv")
    preds_df.to_csv(preds_file, index=False, float_format="%.6f")
    print(f"[Saved] {preds_file}")

    results = []
    for bucket, group in df.groupby(BUCKET_COL):
        idx = group.index
        metrics = evaluate_multilabel(
            df.loc[idx, LABEL_COL].tolist(),
            proba[idx],
            list(classes),
            top_k=(1,3,5),
            threshold=threshold
        )
        metrics["bucket"] = bucket
        results.append(metrics)

    metrics_all = evaluate_multilabel(
        df[LABEL_COL].tolist(),
        proba,
        list(classes),
        top_k=(1,3,5),
        threshold=threshold
    )
    metrics_all["bucket"] = "all"
    results.append(metrics_all)

    # save summary
    summary_df = pd.DataFrame(results)
    summary_file = os.path.join(out_dir, "summary_by_bucket.csv")
    summary_df.to_csv(summary_file, index=False, float_format="%.4f")
    print(f"[Saved] {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    test(args.test_csv, args.model_dir, args.out_dir, batch_size=args.batch_size, threshold=args.threshold)

