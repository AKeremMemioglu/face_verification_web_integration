import os
import time
import numpy as np
import cv2
from deepface import DeepFace
from dataset import LFWLoader
from utils import plot_results
import tkinter as tk
from tkinter import ttk
import dlib
import importlib  # <-- eklendi (keras / tf.keras esnek yükleme için)

# --- KeremNet esnek yükleyici ---
def _load_keremnet_model(path: str):
    """
    Önce Keras 3 ile yüklemeyi dene; olmazsa tf.keras'a düş.
    compile=False: eğitim objeleri gerektirmeden modeli açar.
    """
    try:
        keras = importlib.import_module("keras")  # Keras 3
        return keras.models.load_model(path, compile=False)
    except Exception as e1:
        try:
            from tensorflow import keras as tfk  # tf.keras
            return tfk.models.load_model(path, compile=False)
        except Exception as e2:
            raise RuntimeError(
                f"KeremNet yüklenemedi.\n"
                f"Keras hatası: {e1}\n"
                f"tf.keras hatası: {e2}"
            )

class EmbeddingRecognizer:
    def __init__(self, backend, dataset, model_name):
        self.backend = backend
        self.dataset = dataset
        self.model_name = model_name
        self.embeddings = {}
        self.images = {}  # { "name_imgnum": image_array }

        # KeremNet modeli
        if self.model_name == "KeremNet":
            # NOT: Modelin içinde zaten x/127.5 - 1 ölçekleme var.
            self.custom_model = _load_keremnet_model("keremnet/keremnet.keras")

        # Dlib modelleri
        if self.model_name == "Dlib":
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    def load_all_images(self):
        with open("all_images_list.txt", "w") as f:
            for root, dirs, files in os.walk(self.dataset.images_dir):
                for file in files:
                    if file.endswith(".jpg"):  # mevcut davranışı korudum
                        img_path = os.path.join(root, file)
                        name_img = os.path.splitext(file)[0]
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.images[name_img] = img
                            f.write(f"{name_img},{img_path}\n")

    def extract_embedding(self, img):
        start_time = time.time()

        if self.model_name == "Dlib":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(img_rgb)
            if len(faces) == 0:
                raise Exception("No face detected")
            shape = self.shape_predictor(img_rgb, faces[0])
            face_descriptor = self.face_rec_model.compute_face_descriptor(img_rgb, shape)
            embedding = np.array(face_descriptor, dtype=np.float32)

        elif self.model_name == "KeremNet":
            # ÖNEMLİ: preprocess_input YOK — model grafiğinde zaten var.
            resized = cv2.resize(img, (224, 224)).astype(np.float32)  # 0..255 beklenir
            preprocessed = np.expand_dims(resized, axis=0)
            embedding = self.custom_model.predict(preprocessed, verbose=0)[0].astype(np.float32)

        else:
            rep = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=self.backend,
                enforce_detection=False
            )
            embedding = np.array(rep[0]["embedding"], dtype=np.float32)

        elapsed = time.time() - start_time
        return embedding, elapsed

    def cosine_distance(self, emb1, emb2):
        a = np.array(emb1, dtype=np.float32)
        b = np.array(emb2, dtype=np.float32)
        return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

    def process_pair(self, img_key1, img_key2, threshold=0.6):
        img1 = self.images.get(img_key1)
        img2 = self.images.get(img_key2)

        if img1 is None or img2 is None:
            print(f"🚫 Could not find images: {img_key1}, {img_key2}")
            return None

        if img_key1 not in self.embeddings:
            emb1, t1 = self.extract_embedding(img1)
            self.embeddings[img_key1] = emb1
        else:
            emb1 = self.embeddings[img_key1]
            t1 = 0.0

        if img_key2 not in self.embeddings:
            emb2, t2 = self.extract_embedding(img2)
            self.embeddings[img_key2] = emb2
        else:
            emb2 = self.embeddings[img_key2]
            t2 = 0.0

        start_time = time.time()
        distance = self.cosine_distance(emb1, emb2)
        elapsed = time.time() - start_time

        is_match = distance < threshold

        info_text = f"Embedding1: {t1:.2f}s | Embedding2: {t2:.2f}s\n" \
                    f"Comparison: {elapsed:.4f}s | Distance: {distance:.4f}\n" \
                    f"{'MATCH' if is_match else 'NO MATCH'} (threshold: {threshold})"

        plot_results(img1, [(img2, is_match, distance)], f"{img_key1} vs {img_key2}", extra_text=info_text)

        return t1 + t2, is_match


def get_model_from_gui():
    def on_select():
        nonlocal selected_model
        selected_model = model_var.get()
        root.destroy()

    selected_model = None
    root = tk.Tk()
    root.title("Select Recognition Model")

    tk.Label(root, text="Select Recognition Model:").pack(pady=10)

    # KeremNet'i listeye ekledim
    models = ["VGG-Face", "Facenet", "ArcFace", "Dlib", "KeremNet"]
    model_var = tk.StringVar(value=models[0])

    model_menu = ttk.Combobox(root, textvariable=model_var, values=models, state="readonly")
    model_menu.pack(pady=10)

    btn = tk.Button(root, text="Start", command=on_select)
    btn.pack(pady=20)

    root.mainloop()
    return selected_model


def select_images_gui(images_dict):
    selected = {}

    def on_select():
        selected['person'] = person_var.get()
        selected['img1'] = img1_var.get()
        selected['img2'] = img2_var.get()
        root.destroy()

    root = tk.Tk()
    root.title("Select Images")

    unique_names = sorted({name.split('_')[0] for name in images_dict.keys()})

    person_var = tk.StringVar(value=unique_names[0])
    img1_var = tk.StringVar()
    img2_var = tk.StringVar()

    def update_images(*args):
        imgs = [k for k in images_dict.keys() if k.startswith(person_var.get())]
        img1_menu['values'] = imgs
        img2_menu['values'] = imgs
        if imgs:
            img1_var.set(imgs[0])
            img2_var.set(imgs[1] if len(imgs) > 1 else imgs[0])

    tk.Label(root, text="Select Person:").pack()
    person_menu = ttk.Combobox(root, textvariable=person_var, values=unique_names, state="readonly")
    person_menu.pack()
    person_var.trace_add("write", update_images)

    tk.Label(root, text="Select Image 1:").pack()
    img1_menu = ttk.Combobox(root, textvariable=img1_var, state="readonly")
    img1_menu.pack()

    tk.Label(root, text="Select Image 2:").pack()
    img2_menu = ttk.Combobox(root, textvariable=img2_var, state="readonly")
    img2_menu.pack()

    btn = tk.Button(root, text="Start", command=on_select)
    btn.pack(pady=10)

    update_images()
    root.mainloop()

    return selected['img1'], selected['img2']


if __name__ == "__main__":
    model_name = get_model_from_gui()
    backend = "opencv"  # sabit backend
    print(f"[*] Selected model: {model_name}")

    dataset = LFWLoader("lfw-deepfunneled")
    recognizer = EmbeddingRecognizer(backend=backend, dataset=dataset, model_name=model_name)

    print("[*] Loading all dataset images into memory...")
    recognizer.load_all_images()
    print(f"[*] Loaded {len(recognizer.images)} images and saved list to all_images_list.txt")

    # 📄 Embedding dosyası var mı kontrol et
    embeddings_file = "all_embeddings.npy"

    if os.path.exists(embeddings_file):
        print(f"[*] Loading embeddings from {embeddings_file}...")
        recognizer.embeddings = np.load(embeddings_file, allow_pickle=True).item()
        print(f"[*] Loaded embeddings for {len(recognizer.embeddings)} images.")
    else:
        print("[*] Calculating embeddings for all images...")
        for key, img in recognizer.images.items():
            if key not in recognizer.embeddings:
                embedding, _ = recognizer.extract_embedding(img)
                recognizer.embeddings[key] = embedding
        # 📄 Dosyaya kaydet
        np.save(embeddings_file, recognizer.embeddings)
        print(f"[*] Embeddings calculated and saved to {embeddings_file}")

    total_time = 0.0
    valid_comparisons = 0
    correct_predictions = 0
    threshold = 0.6

    try:
        while True:
            img_key1, img_key2 = select_images_gui(recognizer.images)

            print(f"\n=== Comparing: {img_key1} <-> {img_key2} ===")
            result = recognizer.process_pair(img_key1, img_key2, threshold)
            if result is None:
                continue

            time_spent, is_match = result
            total_time += time_spent
            valid_comparisons += 1

            # ✅ ground truth kontrolü
            name1 = img_key1.rsplit('_', 1)[0]
            name2 = img_key2.rsplit('_', 1)[0]
            ground_truth = (name1 == name2)

            if is_match == ground_truth:
                correct_predictions += 1
                print(f"✅ Correct prediction: expected {ground_truth}, got {is_match}")
            else:
                print(f"❌ Incorrect prediction: expected {ground_truth}, got {is_match}")

    except KeyboardInterrupt:
        print("\n🛑 İşlem kullanıcı tarafından durduruldu.")

    finally:
        if valid_comparisons > 0:
            avg_time = total_time / valid_comparisons
            accuracy = (correct_predictions / valid_comparisons) * 100
        else:
            avg_time = 0
            accuracy = 0

        print("\n=== 📊 Final Results ===")
        print(f"Toplam karşılaştırma: {valid_comparisons}")
        print(f"Ortalama embedding süresi (saniye): {avg_time:.2f}")
        print(f"Doğruluk (Accuracy): {accuracy:.2f}%")
