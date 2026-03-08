
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pandas as pd
from src import weakness_utk

def test_label_creation():
    df = pd.DataFrame({
        "filepath": ["a.jpg"],
        "age_bucket": ["25-39"],
        "gender_norm": ["Male"],
        "race_unified": ["White"],
        "dataset_source": ["UTK"],
        "split_source": ["test"]
    })

    df = weakness_utk.add_labels(df)

    assert "label" in df.columns
    assert df["label"].iloc[0] == 2

def test_schema():
    df = pd.DataFrame({
        "filepath": ["a.jpg"],
        "age_bucket": ["25-39"],
        "gender_norm": ["Male"],
        "race_unified": ["White"],
        "dataset_source": ["UTK"],
        "split_source": ["test"]
    })

    weakness_utk.assert_schema(df)
