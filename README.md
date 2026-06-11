# 🎬 Movie Review Sentiment Analysis with Streamlit

Dự án xây dựng mô hình học máy phân tích cảm xúc bình luận phim (Tích cực / Tiêu cực) từ dữ liệu văn bản tiếng Anh, kết hợp triển khai ứng dụng web trực quan bằng thư viện **Streamlit**.

---

## 📌 Tổng Quan Dự Án
Dự án được chia làm 2 giai đoạn chính tương ứng với 2 file cốt lõi:
1. **`Sentiment_Analysis (1).ipynb`**: File Notebook thực hiện phân tích dữ liệu khám phá (EDA), tiền xử lý văn bản, trích xuất đặc trưng bằng phương pháp **TF-IDF**, huấn luyện mô hình **LinearSVC (Support Vector Machine)** và đánh giá hiệu năng (Confusion Matrix, ROC Curve, Precision, Recall, F1-Score).
2. **`app.py`**: Mã nguồn ứng dụng web được xây dựng bằng **Streamlit**. Giao diện cho phép người dùng nhập trực tiếp một đoạn bình luận phim, hệ thống sẽ tự động xử lý và trả về kết quả dự đoán kèm theo biểu đồ phần trăm độ tự tin (Confidence Score).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
📁 movie-sentiment-app/
│
├── 📝 Sentiment_Analysis (1).ipynb   # Quy trình EDA, Huấn luyện & Đánh giá mô hình
├── 🐍 app.py                         # Mã nguồn giao diện Streamlit App
├── 📦 model.pkl                      # File mô hình đã huấn luyện (LinearSVC)
├── 📦 preprocessor.pkl               # File bộ chuyển đổi văn bản đã fit (TF-IDF Vectorizer)
├── 📄 requirements.txt               # Danh sách các thư viện cần cài đặt
└── 📄 .gitignore                     # Cấu hình bỏ qua các file rác của IDE (ví dụ: .vs/)
