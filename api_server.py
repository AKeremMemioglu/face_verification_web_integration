from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sqlite3
import numpy as np
from embedding import EmbeddingRecognizer, LFWLoader
import time
from urllib.parse import quote

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

DB_PATH = "face_embeddings.db"
DATASET_DIR = os.path.abspath("lfw-deepfunneled")
IMAGES_LIST_FILE = "all_images_list.txt"

# ✅ Veritabanı Fonksiyonları
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            name TEXT,
            model TEXT,
            embedding BLOB,
            embedding_time REAL,
            PRIMARY KEY (name, model)
        )
    """)
    conn.commit()
    conn.close()

def load_embeddings_from_db(model):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding, embedding_time FROM embeddings WHERE model=?", (model,))
    rows = c.fetchall()
    conn.close()
    return {name: {"embedding": np.frombuffer(emb_blob, dtype=np.float64), "time": emb_time} for name, emb_blob, emb_time in rows}

def save_embedding_to_db(name, model, embedding, embedding_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO embeddings (name, model, embedding, embedding_time)
        VALUES (?, ?, ?, ?)
    """, (name, model, embedding.tobytes(), embedding_time))
    conn.commit()
    conn.close()

# ✅ Statik Resim Servisi
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATASET_DIR, filename)

# ✅ all_images_list.txt üzerinden URL üret
def image_url(key):
    if os.path.exists(IMAGES_LIST_FILE):
        with open(IMAGES_LIST_FILE, "r") as f:
            for line in f:
                name_img, path = line.strip().split(",")
                if name_img == key:
                    rel_path = os.path.relpath(path, DATASET_DIR)
                    return f"http://localhost:5000/images/{rel_path.replace(os.sep, '/')}"
    folder = key.split("_")[0]
    return f"http://localhost:5000/images/{folder}/{key}.jpg"

# ✅ Model ve Dataset Yükleme
init_db()
backend = "dlib"
dataset = LFWLoader("lfw-deepfunneled")
models = ["VGG-Face", "Facenet", "ArcFace","KeremNet"]
recognizers = {}

print("[*] Dataset yükleniyor...")

for model_name in models:
    print(f"\n=== {model_name} modeli için işlem başlıyor ===")
    recognizer = EmbeddingRecognizer(backend=backend, dataset=dataset, model_name=model_name)
    recognizer.load_all_images()
    recognizers[model_name] = recognizer

    db_embeddings = load_embeddings_from_db(model_name)
    if len(db_embeddings) == len(recognizer.images):
        print(f"[*] {model_name} için {len(db_embeddings)}/{len(recognizer.images)} embedding mevcut.")
        continue

    print(f"[*] {model_name} için embedding hesaplanıyor...")
    count = 0
    for key, img in recognizer.images.items():
        if key in db_embeddings:
            continue
        start = time.time()
        emb, _ = recognizer.extract_embedding(img)
        elapsed = time.time() - start
        save_embedding_to_db(key, model_name, emb, elapsed)
        count += 1
        if count % 200 == 0:
            print(f"[+] {model_name} → {count}/{len(recognizer.images)} tamamlandı")
    print(f"[✔] {model_name} için embeddingler tamamlandı.")

# ✅ Auto-Suggest Endpoint
@app.route("/api/suggest", methods=["GET"])
def suggest_names():
    query = request.args.get("q", "").lower()
    suggestions = set()
    if os.path.exists("all_images_list.txt"):
        with open("all_images_list.txt", "r") as f:
            for line in f:
                name_img, _ = line.strip().split(",")
                folder_name = name_img.split("_")[0:2]
                full_name = " ".join(folder_name)
                if full_name.lower().startswith(query):
                    suggestions.add(full_name)
    return jsonify(sorted(list(suggestions))[:10])

# ✅ Face Verify API (mevcut)
@app.route("/api/verify", methods=["POST"])
def verify_face():
    data = request.get_json()
    model = data.get("model")
    name = data.get("name")
    if not model or not name:
        return jsonify({"error": "Model and name are required"}), 400
    db_embeddings = load_embeddings_from_db(model)
    if not db_embeddings:
        return jsonify({"error": f"{model} için embedding bulunamadı"}), 404
    normalized_name = name.replace(" ", "_")
    candidates = [k for k in db_embeddings.keys() if k.lower().startswith(normalized_name.lower() + "_")]
    if not candidates:
        return jsonify({"match": False, "similarity": 0, "message": f"{name} bulunamadı"}), 404
    target_key = candidates[0]
    target_emb = db_embeddings[target_key]["embedding"]
    results = []
    for k, data_emb in db_embeddings.items():
        dist = recognizers[model].cosine_distance(target_emb, data_emb["embedding"])
        similarity = (1 - dist) * 100
        results.append((k, similarity, data_emb["time"]))
    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]
    top3_data = [{"name": k, "similarity": float(round(sim, 2)), "embedding_time": round(t, 3), "url": image_url(k)} for k, sim, t in top3]
    return jsonify({
        "match": True,
        "model": model,
        "target": {"name": target_key, "url": image_url(target_key)},
        "top3": top3_data,
        "message": f"{name} için en yakın 3 eşleşme bulundu ({model} modeli ile)"
    })

# ✅ Yeni: Kişiye ait tüm resimler
@app.route("/api/person_images", methods=["GET"])
def get_person_images():
    name = request.args.get("name", "").replace(" ", "_")
    if not name:
        return jsonify({"error": "İsim gerekli"}), 400
    db_embeddings = load_embeddings_from_db(models[0])
    person_images = [k for k in db_embeddings.keys() if k.startswith(name + "_")]
    if not person_images:
        return jsonify({"error": f"{name} için resim bulunamadı"}), 404
    return jsonify({"images": [{"name": k, "url": image_url(k)} for k in sorted(person_images)]})

# ✅ Yeni: Seçilen resme göre doğrulama
@app.route("/api/verify_image", methods=["POST"])
def verify_image():
    data = request.get_json()
    model = data.get("model")
    image_name = data.get("image_name")
    if not model or not image_name:
        return jsonify({"error": "Model ve image_name gerekli"}), 400
    db_embeddings = load_embeddings_from_db(model)
    if image_name not in db_embeddings:
        return jsonify({"error": f"{image_name} embedding bulunamadı"}), 404
    target_emb = db_embeddings[image_name]["embedding"]
    results = []
    for k, data_emb in db_embeddings.items():
        dist = recognizers[model].cosine_distance(target_emb, data_emb["embedding"])
        similarity = (1 - dist) * 100
        results.append((k, similarity, data_emb["time"]))
    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]
    top3_data = [{"name": k, "similarity": float(round(sim, 2)), "embedding_time": round(t, 3), "url": image_url(k)} for k, sim, t in top3]
    return jsonify({
        "match": True,
        "model": model,
        "target": {"name": image_name, "url": image_url(image_name)},
        "top3": top3_data,
        "message": f"{image_name} için en yakın 3 eşleşme bulundu ({model} modeli ile)"
    })

if __name__ == "__main__":
    print("[*] API başlatılıyor... Embedding hesaplama yapılmayacak.")
    app.run(host="0.0.0.0", port=5000, debug=False)
