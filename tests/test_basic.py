
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pandas as pd
from src import weakness_utk


def test_schema_ok():
    df = pd.DataFrame({
        "filepath": ["a.jpg"],
        "age_bucket": ["25-39"],
        "gender_norm": ["Male"],
        "race_unified": ["White"],
        "dataset_source": ["UTK"],
        "split_source": ["test"],
    })
    weakness_utk.assert_schema(df)


def test_add_labels():
    df = pd.DataFrame({
        "filepath": ["a.jpg"],
        "age_bucket": ["25-39"],
        "gender_norm": ["Male"],
        "race_unified": ["White"],
        "dataset_source": ["UTK"],
        "split_source": ["test"],
    })
    out = weakness_utk.add_labels(df)
    assert "label" in out.columns
    assert int(out.loc[0, "label"]) == 2


def test_make_test_split():
    df = pd.DataFrame({
        "filepath": ["a.jpg", "b.jpg"],
        "age_bucket": ["25-39", "40-59"],
        "gender_norm": ["Male", "Female"],
        "race_unified": ["White", "Black"],
        "dataset_source": ["UTK", "UTK"],
        "split_source": ["train", "test"],
    })
    out = weakness_utk.make_test_split(df)
    assert len(out) == 1
    assert out.iloc[0]["split_source"].lower() == "test"


def test_check_label_range():
    df = pd.DataFrame({
        "filepath": ["a.jpg"],
        "age_bucket": ["0-12"],
        "gender_norm": ["Male"],
        "race_unified": ["White"],
        "dataset_source": ["UTK"],
        "split_source": ["test"],
    })
    out = weakness_utk.add_labels(df)
    weakness_utk.check_label_range(out)


def test_file_exists_rate():
    here = Path(__file__).resolve()
    df = pd.DataFrame({
        "filepath": [str(here), str(here)],
        "age_bucket": ["0-12", "13-24"],
        "gender_norm": ["Male", "Female"],
        "race_unified": ["White", "Black"],
        "dataset_source": ["UTK", "UTK"],
        "split_source": ["test", "test"],
    })
    rate = weakness_utk.file_exists_rate(df)
    assert rate == 1.0
