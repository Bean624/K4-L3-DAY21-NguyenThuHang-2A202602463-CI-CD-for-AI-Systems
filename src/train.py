import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POS_RATIO = 0.248


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    # Bonus 1: Ket noi DagsHub remote tracking neu bien moi truong ton tai
    if os.environ.get("MLFLOW_TRACKING_URI") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Canh bao lech lac du lieu (Data Drift Alert)
    pos_ratio = float(y_train.mean())
    if abs(pos_ratio - REFERENCE_POS_RATIO) > 0.05:
        print(
            f"[CANH BAO DATA DRIFT] Ty le lop duong huan luyen: {pos_ratio:.4f}, "
            f"lech qua 5 diem phan tram so voi ty le tham chieu ({REFERENCE_POS_RATIO:.4f})!"
        )
    else:
        print(f"[DATA QUALITY] Ty le lop duong: {pos_ratio:.4f} (khop voi ty le tham chieu)")

    with mlflow.start_run():
        # Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # Khởi tạo và huấn luyện mô hình GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Dự đoán tren tap holdout voi nguong mac dinh 0.5
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Dieu chinh nguong quyet dinh (Threshold Tuning)
        probs = model.predict_proba(X_eval)[:, 1]
        thresholds = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.05)]
        best_threshold = 0.5
        best_f1 = f1
        for t in thresholds:
            t_preds = (probs >= t).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds, zero_division=0))
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_threshold = t

        print(f"F1 (nguong 0.5): {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Bonus 2 - Nguong toi uu: {best_threshold:.2f} (F1 toi uu: {best_f1:.4f})")

        # Bonus 3: Bao cao Precision / Recall & Confusion Matrix
        cm = confusion_matrix(y_eval, preds)
        clf_rep = classification_report(y_eval, preds, digits=4)
        detail_content = (
            f"=== MA TRAN NHAM LAN (CONFUSION MATRIX) ===\n{cm}\n\n"
            f"=== BANG DANH GIA CHI TIET (CLASSIFICATION REPORT) ===\n{clf_rep}\n"
            f"=== THRESHOLD TUNING (BONUS 2) ===\n"
            f"Nguong mac dinh: 0.50 -> F1: {f1:.4f}\n"
            f"Nguong toi uu:   {best_threshold:.2f} -> F1: {best_f1:.4f}\n\n"
            f"=== DATA DRIFT CHECK (BONUS 5) ===\n"
            f"Ty le lop duong huan luyen: {pos_ratio:.4f} (Tham chieu: {REFERENCE_POS_RATIO:.4f})\n"
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_content)

        # Ghi nhận chỉ số vào MLflow và lưu artifact
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("pos_ratio", pos_ratio)

        mlflow.log_artifact("outputs/detail.txt")
        mlflow.sklearn.log_model(model, "model")

        # In kết quả ra màn hình terminal
        print(f"Ket qua huan luyen: F1={f1:.4f}, Accuracy={acc:.4f}, Best F1={best_f1:.4f} (tai nguong {best_threshold:.2f})")

        # Lưu metrics ra file outputs/report.json
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "pos_ratio": pos_ratio,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # Lưu mô hình ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
