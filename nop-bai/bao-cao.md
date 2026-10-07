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

**Lý do:** Em chọn bộ siêu tham số lần 3 vì đạt f1_score cao nhất (0.7149 >= 0.65), vượt ngưỡng chất lượng quy định. Dù lần 1 có accuracy cao nhất (0.8780 so với 0.8740), sự chênh lệch này cho thấy lần có accuracy cao nhất không trùng với lần có f1_score cao nhất. Mô hình lần 1 đoán đúng nhiều mẫu lớp đa số nhưng bỏ sót lớp thu nhập cao. Giữa n_estimators và learning_rate có sự đánh đổi trực tiếp; cấu hình 200 cây với max_depth=5 giúp mô hình học sâu các quan hệ phức tạp, tránh dưới khớp như lần 2 mà vẫn tối ưu khả năng bắt lớp thiểu số.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Dữ liệu Adult có phân bố lớp mất cân bằng lớn: chỉ 24.8% mẫu thuộc lớp thu nhập cao (>50K USD/năm). Một mô hình vô dụng luôn đoán thu nhập thấp vẫn đạt accuracy 0.752, tạo ra con số gây hiểu nhầm vì không bắt được ca thu nhập cao nào. Chỉ số F1 của lớp dương là trung bình điều hòa giữa Precision và Recall, đo lường chính xác khả năng phát hiện đúng và không bỏ sót người thu nhập cao mà accuracy bỏ qua. Không dùng average="weighted" hay average="macro" khi tính F1 vì việc tính trung bình sẽ bị lớp đa số (75.2%) kéo điểm lên cao, làm mất đi ý nghĩa đánh giá trên lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi import FallbackAsyncAdaptedQueuePool trong MLflow. | Thư viện SQLAlchemy 2.1 không tương thích với MLflow 2.13.0. | Ghim phiên bản SQLAlchemy < 2.1 trong requirements.txt và cài bản 2.0.54. |
| Job Release trên GitHub Actions lỗi SSH handshake. | Dán nhầm private key vào biến SERVER_USER thay vì username ubuntu. | Sửa secret SERVER_USER thành ubuntu và thêm dòng trống cuối private key. |
| Service FastAPI trên máy chủ EC2 lỗi unpickle mô hình. | Máy chủ cài scikit-learn 1.7.2 lệch với bản 1.4.2 lúc huấn luyện. | Đồng bộ cài đặt scikit-learn==1.4.2 trên máy chủ và restart service. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung 22.361 mẫu ở Bước 3, f1_score tăng từ 0.7149 lên 0.7354 và accuracy tăng từ 0.8740 lên 0.8820. Hai tập dữ liệu được chia ngẫu nhiên từ cùng nguồn nên cùng phân phối; sự tăng nhẹ này cho thấy mô hình nắm bắt thêm các trường hợp biên. Quan trọng nhất, pipeline CI/CD đã phản ứng hoàn hảo: commit dữ liệu mới tự động kích hoạt huấn luyện và triển khai ra môi trường phục vụ mà không cần can thiệp thủ công.
