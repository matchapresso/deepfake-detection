import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# SETUP PATH & LOAD MODEL
# Gunakan path lokal (asumsi file model dan CSV ada di folder yang sama dengan app.py)
MODEL_PATH = 'xception_finetuned_model.h5'
LOG_PATH = 'xception_training_log.csv'

CLASS_NAMES = ['Fake', 'Real']
IMG_SIZE = 128

@st.cache_resource
def load_deeplearning_model():
    """Fungsi untuk load model sekali saja (cached) agar aplikasi kencang"""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        return None

#  STREAMLIT INTERFACE
st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    layout="wide"
)

st.title(" Deepfake Face Detection")
st.markdown("Aplikasi ini menggunakan model Deep Learning (**Xception**) untuk menganalisis apakah sebuah foto wajah adalah asli (Real) atau hasil manipulasi (Deepfake).")

# Load model utama
model = load_deeplearning_model()

if model is None:
    st.error(f"File model tidak ditemukan! Pastikan file `{MODEL_PATH}` ada di folder yang sama dengan `app.py`.")
    st.stop()

# SIDEBAR: PENGATURAN TAMPILAN
st.sidebar.header("Pengaturan")
show_metrics = st.sidebar.checkbox("Tampilkan Kurva Training (Loss & Acc)")

# HALAMAN UTAMA: PREDIKSI GAMBAR
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Foto Wajah")
    uploaded_file = st.file_uploader("Pilih gambar wajah (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang di-upload", use_container_width=True)

with col2:
    st.subheader("Hasil Analisis Model")
    if uploaded_file is not None:
        with st.spinner("Sedang menganalisis wajah..."):
            # Preprocessing Gambar
            img_resized = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized)
            
            # Jika gambar memiliki channel alpha (RGBA), convert ke RGB
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            img_array = img_array / 255.0  # Normalisasi
            img_tensor = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
            
            # Prediksi
            prediction_prob = model.predict(img_tensor).flatten()[0]
            
            # Tentukan Hasil Kelas (Thresh 0.5)
            # Karena 1 = Fake, maka probabilitas di atas threshold artinya FAKE
            if prediction_prob > 0.35:
                result_label = "FAKE (Deepfake/Manipulasi)"
                confidence = prediction_prob * 100
                st.error(f"### Hasil: **{result_label}**")
            else:
                result_label = "REAL (Asli)"
                confidence = (1 - prediction_prob) * 100
                st.success(f"### Hasil: **{result_label}**")
                
            # Tampilkan Progress Bar Confidence Score
            st.write("**Confidence Score:**")
            st.progress(int(confidence))
            st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")
            
    else:
        st.info("Silakan upload foto wajah di panel sebelah kiri untuk memulai analisis otomatis.")

# BAGIAN BAWAH: VISUALISASI JIKA DICENTANG
if show_metrics:
    st.markdown("---")
    st.subheader("Grafik Training Model")
    
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Accuracy
        ax1.plot(df_log['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in df_log.columns:
            ax1.plot(df_log['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        ax1.set_title('Kurva Akurasi')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--')
        
        # Plot Loss
        ax2.plot(df_log['loss'], label='Train Loss', color='red', linewidth=2)
        if 'val_loss' in df_log.columns:
            ax2.plot(df_log['val_loss'], label='Validation Loss', color='green', linewidth=2)
        ax2.set_title('Kurva Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--')
        
        st.pyplot(fig)
    else:
        st.warning(f"File log CSV `{LOG_PATH}` tidak ditemukan di folder ini.")
