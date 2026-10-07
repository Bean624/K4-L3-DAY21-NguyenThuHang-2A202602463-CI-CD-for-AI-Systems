import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ giả lập với 10 đặc trưng để test mà không cần tải file thật.
    """
    rng = np.random.default_rng(0)
    n = 200

    # TODO 1: Tạo mảng X ngẫu nhiên kích thước (200, 10)
    X = rng.random((n, len(FEATURE_NAMES)))

    # TODO 2: Tạo mảng nhãn y gồm 200 số 0 hoặc 1 (nguyên ngẫu nhiên trong [0, 2))
    y = rng.integers(0, 2, size=n)

    # TODO 3: Ghép thành DataFrame và gán cột "target"
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # TODO 4: Lưu 160 dòng đầu làm tập học, 40 dòng cuối làm tập thi (holdout)
    train_path = str(tmp_path / "train.csv")
    eval_path  = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path,  index=False)

    # TODO 5: Trả về đường dẫn 2 file tạm
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() có trả về một số thực điểm F1 từ 0.0 đến 1.0 không."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # TODO 6 & 7: Gọi train() với tham số nhỏ và kiểm tra kết quả
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    """Kiểm tra file outputs/report.json có được tạo ra đúng định dạng không."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 8: Kiểm tra file tồn tại và có đủ 2 chỉ số f1_score, accuracy
    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.joblib có được lưu lại không."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 9: Kiểm tra file model tồn tại
    assert os.path.exists("models/model.joblib")
