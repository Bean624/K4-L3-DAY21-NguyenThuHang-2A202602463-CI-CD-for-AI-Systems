# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Thu Hằng |
| MSSV | 2A202602463 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Bean624/K4-L3-DAY21-NguyenThuHang-2A202602463-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Em chọn lần 3 vì đạt `f1_score` cao nhất (0.7149 >= 0.65). Dù lần 1 accuracy cao hơn (0.8780), việc accuracy cao nhất không trùng f1_score cao nhất chứng tỏ lần 1 thiên vị lớp đa số mà bỏ sót lớp thu nhập cao. Cấu hình 200 cây và max_depth=5 giúp mô hình học quan hệ phi tuyến, không bị dưới khớp và tối ưu nhận diện lớp thiểu số.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Dữ liệu Adult mất cân bằng lớn (chỉ 24.8% thuộc lớp >50K). Một mô hình luôn đoán thu nhập thấp vẫn đạt Accuracy 75.2% nhưng vô dụng trong thực tế. F1-score là trung bình điều hòa của Precision và Recall, phản ánh chính xác khả năng phát hiện lớp thiểu số. Không dùng `average="weighted"` hay `average="macro"` vì lớp đa số (75.2%) sẽ kéo điểm lên cao, làm mất tính trung thực khi đánh giá.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi import FallbackAsyncAdaptedQueuePool trong MLflow | SQLAlchemy 2.1 không tương thích MLflow 2.13.0 | Ghim SQLAlchemy < 2.1 trong requirements.txt |
| Job Release trên Actions lỗi SSH handshake | Dán nhầm private key vào biến SERVER_USER | Sửa SERVER_USER thành ubuntu, thêm dòng trống cuối key |
| Service FastAPI trên EC2 lỗi unpickle mô hình | Bản scikit-learn trên EC2 (1.7.2) lệch bản huấn luyện (1.4.2) | Cài scikit-learn==1.4.2 đồng bộ trên EC2 |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Bổ sung 22.361 mẫu ở Bước 3 giúp F1 tăng từ 0.7149 lên 0.7354 và Accuracy tăng từ 0.8740 lên 0.8820. Cùng phân phối ngẫu nhiên, dữ liệu mở rộng giúp mô hình khái quát hóa tốt hơn ở các biên. Pipeline CI/CD tự động phát hiện thay đổi từ file `.dvc` để tái huấn luyện và cập nhật máy chủ mà không cần can thiệp thủ công.

---

## 5. Thách Thức Nâng Cao (Bonus 1 - 5)

- [x] **Bonus 1 (DagsHub MLflow):** Tích hợp MLflow tracking từ xa lên DagsHub server (`https://dagshub.com/Bean624/K4-L3-DAY21-NguyenThuHang-2A202602463-CI-CD-for-AI-Systems.mlflow`). Thí nghiệm được đồng bộ trực tuyến với đầy đủ metrics và artifacts (minh chứng: `06-dagshub-mlflow.png`).
- [x] **Bonus 2 (Threshold Tuning):** Quét ngưỡng quyết định [0.1, 0.9] (bước 0.05). Tại ngưỡng tối ưu **0.30**, F1-score đạt **0.7537** (tăng mạnh so với 0.7354 ở ngưỡng mặc định 0.50).
- [x] **Bonus 3 (Precision/Recall & Confusion Matrix):** Tự động sinh `outputs/detail.txt` và lưu artifact trên Actions. *Phân tích sai lầm:* Với bài toán tài chính, bỏ sót khách hàng thu nhập cao (Recall thấp) làm mất doanh thu lớn hơn nhiều so với gửi nhầm thư chào mời (Precision thấp), vì vậy tối ưu Recall/F1 mang ý nghĩa sống còn.
- [x] **Bonus 4 (Model Rollback Safety Guard):** Tự động tải `report.json` từ S3 để so sánh F1. Nếu model mới có F1 thấp hơn model hiện tại trên production, pipeline tự động hủy triển khai để bảo vệ hệ thống.
- [x] **Bonus 5 (Data Drift Alert):** Tự động kiểm tra phân phối dữ liệu huấn luyện: tỷ lệ lớp dương đạt 24.78% (khớp chuẩn 24.80%), in cảnh báo nếu lệch quá 5 điểm phần trăm và lưu vào `report.json`.

