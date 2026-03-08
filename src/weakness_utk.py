
from pathlib import Path
import pandas as pd

AGE_ORDER = ["0-12", "13-24", "25-39", "40-59", "60+"]

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
    return df[df["split_source"].astype(str).str.lower() == "test"].reset_index(drop=True).copy()

def add_labels(df, age_order=AGE_ORDER):
    age_to_label = {a: i for i, a in enumerate(age_order)}
    df = df.copy()
    df["label"] = df["age_bucket"].map(age_to_label)
    return df

def check_label_range(df, age_order=AGE_ORDER):
    assert "label" in df.columns, "label column missing"
    assert df["label"].notna().all(), "Some labels are NaN"
    assert df["label"].between(0, len(age_order)-1).all(), "Labels out of range"

def file_exists_rate(df):
    exists = df["filepath"].apply(lambda p: Path(str(p)).exists())
    return float(exists.mean())

def assert_min_file_exists_rate(df, min_rate=0.95):
    rate = file_exists_rate(df)
    assert rate >= min_rate, f"File existence rate {rate:.3f} < required {min_rate:.3f}"
