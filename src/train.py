import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    # TODO 1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # TODO 3: Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và huấn luyện mô hình GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Dự đoán trên tập holdout và tính chỉ số
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # TODO 6: Ghi nhận chỉ số vào MLflow và lưu artifact
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả ra màn hình terminal
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 8: Lưu metrics ra file outputs/report.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f)

        # TODO 9: Lưu mô hình ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Trả về điểm f1
    return f1



if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
