import os
import subprocess
import sys
import importlib.util

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import torch
from collections import Counter
try:
    from dont_patronize_me import DontPatronizeMe
except ModuleNotFoundError:
    module_path = os.path.join(project_root, "dont_patronize_me.py")
    spec = importlib.util.spec_from_file_location("dont_patronize_me", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DontPatronizeMe = module.DontPatronizeMe
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import argparse

logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

cuda_available = torch.cuda.is_available()
print("Cuda available? ", cuda_available)


def labels2file(p, outf_path):
    with open(outf_path, "w") as outf:
        for pi in p:
            outf.write(",".join([str(k) for k in pi]) + "\n")


def build_task1_df(data, parids_df):
    parids_df = parids_df.copy()
    parids_df["par_id"] = parids_df["par_id"].astype(str)
    if "label" in parids_df.columns:
        parids_df = parids_df.drop(columns=["label"])
    data = data.copy()
    data["par_id"] = data["par_id"].astype(str)
    merged = parids_df.merge(
        data[["par_id", "keyword", "text", "label"]],
        on="par_id",
        how="left",
    )
    merged = merged.rename(columns={"keyword": "community"})
    return merged[["par_id", "community", "text", "label"]]


def f1_score_binary(y_true, y_pred, positive_label=1):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp == positive_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp == positive_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp != positive_label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1, precision, recall


def load_test_df(test_path):
    df = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["par_id", "art_id", "keyword", "country", "text"],
    )
    return df


def train(dev_result_path=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dev_result_path:
        dev_path = dev_result_path
    else:
        dev_path = os.path.join(project_root, "Predicted_output", "dev.txt")
    dev_dir = os.path.dirname(dev_path) or "."
    os.makedirs(dev_dir, exist_ok=True)

    dpm = DontPatronizeMe(project_root, project_root)
    dpm.load_task1()

    trids = pd.read_csv(os.path.join(project_root, "train_semeval_parids-labels.csv"))
    teids = pd.read_csv(os.path.join(project_root, "dev_semeval_parids-labels.csv"))

    data = dpm.train_task1_df
    trdf1 = build_task1_df(data, trids)
    tedf1 = build_task1_df(data, teids)
    tedf1.to_csv(
        os.path.join(project_root, "dev_dontpatronizeme_pcl.tsv"),
        sep="\t",
        index=False,
    )
    training_set1 = trdf1.sample(frac=1, random_state=seed).reset_index(drop=True)
    pos_df = training_set1[training_set1.label == 1]
    neg_df = training_set1[training_set1.label == 0]
    if len(pos_df) > 0:
        pos_oversampled = pos_df.sample(n=len(neg_df), replace=True, random_state=seed)
        training_set1 = pd.concat([neg_df, pos_oversampled]).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

    task1_model_args = ClassificationArgs(
        num_train_epochs=5,
        no_save=False,
        no_cache=True,
        overwrite_output_dir=True,
        evaluate_during_training=True,
        evaluate_during_training_steps=500,
        use_early_stopping=True,
        early_stopping_metric="f1",
        early_stopping_metric_minimize=False,
        early_stopping_patience=4,
        save_best_model=True,
        best_model_dir=os.path.join(project_root, "outputs", "best_model_task1"),
        output_dir=os.path.join(project_root, "outputs", "task1"),
        learning_rate=2e-5,
        train_batch_size=8,
        eval_batch_size=8,
        manual_seed=seed,
    )
    task1_model = ClassificationModel(
        "roberta",
        "roberta-large",
        args=task1_model_args,
        num_labels=2,
        use_cuda=cuda_available,
    )
    train_df = training_set1.rename(columns={"label": "labels"})[["text", "labels"]]
    eval_df = tedf1.rename(columns={"label": "labels"})[["text", "labels"]]
    task1_model.train_model(
        train_df,
        eval_df=eval_df,
        f1=lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=1),
        accuracy=lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    )
    preds_task1, _ = task1_model.predict(tedf1.text.tolist())

    Counter(preds_task1)

    f1, precision, recall = f1_score_binary(tedf1.label.tolist(), preds_task1, positive_label=1)
    print(f"F1 (PCL=1): {f1:.4f}")
    print(f"Precision (PCL=1): {precision:.4f}")
    print(f"Recall (PCL=1): {recall:.4f}")

    labels2file([[k] for k in preds_task1], dev_path)
    subprocess.run(["zip", "submission_task1_only.zip", dev_path], check=False, cwd=project_root)


def predict_test(test_result_path=None):
    if test_result_path:
        test_path_out = test_result_path
    else:
        test_path_out = os.path.join(project_root, "Predicted_output", "test.txt")
    os.makedirs(os.path.dirname(test_path_out), exist_ok=True)

    test_path = os.path.join(project_root, "task4_test.tsv")
    test_df = load_test_df(test_path)
    model_dir = os.path.join(project_root, "outputs", "best_model_task1")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(project_root, "outputs", "task1")
    task1_model = ClassificationModel(
        "roberta",
        model_dir,
        num_labels=2,
        use_cuda=cuda_available,
    )
    preds_task1, _ = task1_model.predict(test_df.text.tolist())
    labels2file([[k] for k in preds_task1], test_path_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--dev-result", dest="dev_result", default=None)
    parser.add_argument("--test-result", dest="test_result", default=None)
    args = parser.parse_args()
    if args.dev_result and args.test_result:
        train(args.dev_result)
        predict_test(args.test_result)
    elif args.mode == "test" or args.test_result:
        predict_test(args.test_result)
    else:
        train(args.dev_result)


if __name__ == "__main__":
    main()
