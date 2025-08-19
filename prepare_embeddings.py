import os
import time
import numpy as np

from embedding import EmbeddingRecognizer, LFWLoader
from api_server import init_db, save_embedding_to_db, load_embeddings_from_db

# =============================
# Ayarlar
# =============================
backend = "dlib"  # DeepFace backend (VGG/Facenet/ArcFace için)
dataset = LFWLoader("lfw-deepfunneled")
models = ["VGG-Face", "Facenet", "ArcFace", "KeremNet"]

print("[*] Dataset yükleniyor...")
init_db()

for model_name in models:
    print(f"\n=== {model_name} modeli için embedding hazırlanıyor ===")
    recognizer = EmbeddingRecognizer(backend=backend, dataset=dataset, model_name=model_name)
    recognizer.load_all_images()

    db_embeddings = load_embeddings_from_db(model_name)
    already = len(db_embeddings)
    total = len(recognizer.images)

    if already >= total and total > 0:
        print(f"[✔] {model_name} zaten hazır ({already}/{total}). Atlanıyor.")
        continue

    # Eksikleri tamamla
    done = 0
    for i, (key, img) in enumerate(recognizer.images.items(), start=1):
        if key in db_embeddings:
            continue
        try:
            start = time.time()
            emb, _ = recognizer.extract_embedding(img)
            # Tüm embedding'leri float32 normalize sakla
            emb = np.asarray(emb, dtype=np.float32)
            # İsteğe bağlı: L2 normalize — doğrulama benzerlikleri daha stabil olur
            nrm = np.linalg.norm(emb) + 1e-12
            emb = emb / nrm

            elapsed = time.time() - start
            save_embedding_to_db(key, model_name, emb, elapsed)
            done += 1
            if i % 200 == 0:
                print(f"[+] {model_name} → {i}/{total} işlendi (bu turda yeni {done})")
        except Exception as e:
            print(f"[!] {model_name} → {key} hata: {e}")

    print(f"[✔] {model_name} için embedding hazırlığı tamamlandı. (Yeni eklenen: {done})")

print("\n[✓] Tüm modeller için embedding hazırlığı bitti.")
