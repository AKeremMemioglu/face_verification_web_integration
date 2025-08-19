# 🧑‍💻 Face Recognition with Multiple Models (VGG-Face, Facenet, ArcFace, KeremNet)

Bu proje, **LFW (Labeled Faces in the Wild)** veri seti üzerinde farklı yüz tanıma modelleri (VGG-Face, Facenet, ArcFace, KeremNet) ile embedding çıkarma, bunları SQLite veritabanına kaydetme ve **Flask tabanlı bir API** ile **Next.js tabanlı frontend** üzerinden doğrulama yapmayı amaçlar.

---

## 📂 Proje Yapısı

### 🔹 Backend (Flask)
- **`api_server.py`** → Flask tabanlı REST API servisidir.  
- **`embedding.py`** → Yüz embedding çıkarma işlevlerini içerir.  
- **`prepare_embeddings.py`** → Dataset’teki resimlerden embedding üretir ve DB’ye kaydeder.  
- **`face_embeddings.db`** → Embedding verilerinin tutulduğu SQLite veritabanı.  

### 🔹 Frontend (Next.js / React)
- **`pages/page.tsx`** → Ana sayfa (kullanıcı arayüzü).  
- **`lib/api.ts`** → Backend API ile haberleşme (fetch fonksiyonları).  
- **`styles/globals.css`** → Global CSS tanımları.  
- **`styles/page.module.css`** → Sayfa özel CSS tanımları.  
- **`app/layout.tsx`** → Next.js layout yapısı.  

---

## ⚙️ Backend Kurulum (Flask)

### 1. Ortam Kurulumu
```bash
# Python sanal ortam oluştur
python3 -m venv myvenv
source myvenv/bin/activate   # Linux/macOS
myvenv\Scripts\activate      # Windows

# Gerekli paketleri yükle
pip install -r requirements.txt
