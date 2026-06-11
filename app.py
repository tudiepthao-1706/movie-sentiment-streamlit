import streamlit as st
import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# CẤU HÌNH
st.set_page_config(page_title="Sentiment Analysis", layout="centered", page_icon="🎬")

# MODEL
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("preprocessor.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_resources()

# Nếu không tìm thấy file model thì báo lỗi và dừng
if model is None or vectorizer is None:
    st.error("⚠️ Lỗi: Không tìm thấy file 'model.pkl'.")
    st.stop()

# HÀM XỬ LÝ TEXT
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# HÀM DỰ ĐOÁN
def predict_sentiment(review: str):
    review_clean = clean_text(review)
    X = vectorizer.transform([review_clean])
    
    # Dự đoán nhãn (0 hoặc 1)
    pred_label = model.predict(X)[0]
    
    # Dự đoán xác suất (Neg, Pos)
    # Trả về mảng [xác suất Neg, xác suất Pos]
    probs = model.predict_proba(X)[0]
    
    return pred_label, probs[0], probs[1]

# GIAO DIỆN CHÍNH
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Dự đoán cảm xúc bình luận phim (Tích cực / Tiêu cực)")

review_input = st.text_area("Nhập nội dung review (Tiếng Anh):", height=150, placeholder="Type your review here...")

if st.button("🔍 Phân tích ngay", type="primary"):
    if review_input.strip() == "":
        st.warning("Vui lòng nhập nội dung trước khi bấm nút!")
    else:
        with st.spinner('Đang phân tích...'):
            pred_label, neg_score, pos_score = predict_sentiment(review_input)

        # Hiển thị kết quả Text
        if pos_score > neg_score:
            st.success(f"### Kết quả: Positive 😊")
        else:
            st.error(f"### Kết quả: Negative 😞")

        ## Vẽ biểu đồ
        # Chuẩn bị dữ liệu
        labels = ['Negative', 'Positive']
        sizes = [neg_score, pos_score]
        colors = ['#ff4b4b', '#28a745'] # Đỏ và Xanh lá

        fig, ax = plt.subplots()

        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels = labels, 
            colors = colors, 
            autopct = '%1.1f%%', # Hiển thị số phần trăm
            startangle = 90,
            wedgeprops = dict(width=0.7, edgecolor='w')
        )
        ax.axis('equal')
        plt.title("Độ tin cậy của mô hình\n")

        # Hiển thị
        st.pyplot(fig)