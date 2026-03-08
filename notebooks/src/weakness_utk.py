
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models

AGE_ORDER = ["0-12","13-24","25-39","40-59","60+"]

def load_df(csv_path):
    return pd.read_csv(csv_path)

def assert_schema(df):
    required = [
        "filepath",
        "age_bucket",
        "gender_norm",
        "race_unified",
        "dataset_source",
        "split_source",
    ]
    for c in required:
        assert c in df.columns, f"Missing column: {c}"

def make_test_split(df):
    return df[df["split_source"].astype(str).str.lower() == "test"].reset_index(drop=True)

def add_labels(df):
    age_to_label = {a:i for i,a in enumerate(AGE_ORDER)}
    df = df.copy()
    df["label"] = df["age_bucket"].map(age_to_label)
    return df

def build_model(num_classes=5):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def eval_preds(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(acc), float(f1)
